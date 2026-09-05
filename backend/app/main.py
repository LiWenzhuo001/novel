"""FastAPI 应用入口——启动、中间件、健康检查。

职责：
1. 应用生命周期管理（lifespan）：建目录、连库建表、预热 embedding
2. CORS 中间件配置
3. 路由注册（/api/chat、/api/kb）
4. 健康检查接口（/health）
"""

import asyncio
import os
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from contextlib import asynccontextmanager

from app.config import settings
from app.api import auth, chat, knowledge, memory
from app.db import init_db, async_engine
from app.core.logging_config import get_logger, new_request_id
from app.core.metrics import metrics
from app.core.context import set_current_user, reset_current_user
from app.core.security import decode_access_token
from sqlalchemy import text as sql_text

log = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(settings.raw_dir, exist_ok=True)
    # 等待 Postgres 就绪并建表（含 pgvector 扩展与 HNSW 索引）；
    # 若仍连不上则告警但不阻断启动
    app.state.db_ready = False
    try:
        await init_db()
        app.state.db_ready = True
    except Exception as e:  # noqa: BLE001
        print(f"[WARNING] PostgreSQL 初始化失败，数据库相关功能将不可用：{e}")

    # 恢复服务重启前遗留的待索引/租约过期任务。
    if app.state.db_ready:
        try:
            from app.services.kb_service import recover_stale_indexing, run_indexing

            for file_id, filename, ext, user_id, use_llm in await recover_stale_indexing():
                asyncio.create_task(run_indexing(file_id, filename, ext, user_id, use_llm))
        except Exception as e:  # noqa: BLE001
            print(f"[WARNING] 恢复索引任务失败：{e}")

    # 预热 embedding 模型：本地模型首次加载需读盘（必要时经镜像下载），
    # 提前在启动阶段完成，避免首个聊天请求被阻塞到网关超时。
    if settings.embedding_provider == "local":
        try:
            from app.core.embed import get_embeddings

            await asyncio.to_thread(get_embeddings)
            print("[INFO] embedding 模型预热完成")
        except Exception as e:  # noqa: BLE001
            print(f"[WARNING] embedding 模型预热失败：{e}")

    # 本地 reranker 需要提前加载，远程 provider 只初始化 HTTP client，不发送计费请求。
    # 失败仅告警，不阻断启动；检索阶段会回退为不重排并继续回答。
    if settings.enable_reranker:
        try:
            from app.core.rerank import get_reranker

            await asyncio.to_thread(get_reranker)
            if settings.reranker_provider == "siliconflow":
                print(f"[INFO] SiliconFlow reranker 客户端就绪（{settings.reranker_model}）")
            else:
                print(f"[INFO] reranker 模型预热完成（{settings.reranker_model}）")
        except Exception as e:  # noqa: BLE001
            print(f"[WARNING] reranker 初始化失败（检索将回退为不重排）：{e}")
    # B2: TTL sweeper background task (LangGraph start_ttl_sweeper pattern); off by default.
    if settings.memory_ttl_sweeper_enabled:

        async def _memory_ttl_sweeper_loop():
            from app.services.memory_service import sweep_expired_memories

            while True:
                try:
                    deleted = await sweep_expired_memories()
                    if deleted:
                        log.info("memory.ttl_swept", deleted=deleted)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    log.warning("memory.ttl_sweep_failed", error=str(exc)[:200])
                await asyncio.sleep(settings.memory_ttl_sweeper_interval_hours * 3600)

        asyncio.create_task(_memory_ttl_sweeper_loop())
    yield


app = FastAPI(title="小说 RAG 问答 Agent", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """为每个请求分配 request_id，记录方法/路径/状态码/耗时（毫秒），并埋点错误计数。

    注意：不读取响应体，避免破坏 SSE 流式（/api/chat）传输。
    """

    async def dispatch(self, request: Request, call_next):
        rid = new_request_id()
        request.state.request_id = rid
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            metrics.error()
            log.error("request.failed", request_id=rid, path=request.url.path)
            raise
        duration_ms = (time.perf_counter() - start) * 1000
        metrics.record_latency("request", duration_ms)
        if response.status_code >= 500:
            metrics.error()
        log.info(
            "request.handled",
            request_id=rid,
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            latency_ms=round(duration_ms, 1),
        )
        response.headers["X-Request-ID"] = rid
        return response


class AuthMiddleware(BaseHTTPMiddleware):
    """可选的 Bearer Token 鉴权 + 多租户 user_id 注入。

    - USER_AUTH_ENABLED=true：要求 JWT Bearer token，/api/auth/register 和 /api/auth/login 免鉴权。
    - API_TOKENS 仍作为脚本/本地调试兼容路径。
    - 都未开启时：关闭鉴权，user_id 统一取默认用户，兼容单用户本地开发。
    """

    _PUBLIC_PATHS = {"/health", "/metrics", "/api/auth/register", "/api/auth/login"}

    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS" or request.url.path in self._PUBLIC_PATHS:
            token = set_current_user(settings.default_user)
            try:
                return await call_next(request)
            finally:
                reset_current_user(token)

        # 保持本地开发开箱可用：USER_AUTH_ENABLED=false 且未配置 API_TOKENS 时仍走 default 租户。
        # 正式多用户运行需设置 USER_AUTH_ENABLED=true，并配置稳定的 JWT_SECRET。
        if not settings.auth_enabled:
            token = set_current_user(settings.default_user)
            try:
                return await call_next(request)
            finally:
                reset_current_user(token)

        auth = request.headers.get("Authorization", "")
        user_id = None
        if auth.startswith("Bearer "):
            bearer = auth[7:].strip()
            if settings.user_auth_enabled:
                try:
                    payload = decode_access_token(bearer)
                    candidate = str(payload["sub"])
                    async with async_engine.connect() as conn:
                        row = (await conn.execute(
                            sql_text("SELECT id FROM users WHERE id = :id AND is_active = true")
                            .bindparams(id=candidate)
                        )).first()
                    if row:
                        user_id = candidate
                except Exception:
                    user_id = None
            if user_id is None:
                for uid, tok in settings.api_tokens.items():
                    if bearer == tok:
                        user_id = uid
                        break
        if not user_id:
            return JSONResponse(
                status_code=401,
                content={"detail": "未授权：缺少或无效的登录凭证"},
            )

        token = set_current_user(user_id)
        try:
            response = await call_next(request)
            response.headers["X-User-ID"] = user_id
            return response
        finally:
            reset_current_user(token)


app.add_middleware(AuthMiddleware)
app.add_middleware(RequestLoggingMiddleware)

app.include_router(auth.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(knowledge.router, prefix="/api")
app.include_router(memory.router, prefix="/api")


@app.get("/health")
async def health():
    db_ok = getattr(app.state, "db_ready", False)
    if db_ok:
        try:
            async with async_engine.connect() as conn:
                await conn.execute(sql_text("SELECT 1"))
        except Exception:
            db_ok = False
    return {
        "status": "ok" if db_ok else "degraded",
        "database": "connected" if db_ok else "unavailable",
        "message": "小说 RAG 问答服务正常" if db_ok else "数据库不可用",
    }


@app.get("/metrics")
async def metrics_endpoint():
    """暴露进程内基础观测指标（计数 + 延迟分位），供简单监控/自测使用。"""
    return JSONResponse(metrics.snapshot())

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # 开发想要热重载可改成 True
    )
