"""聊天会话管理和 Agent SSE 流式问答接口。

所有请求（普通问答与角色扮演）进入同一个 LangGraph：角色扮演通过
interaction_mode=roleplay 走统一图（load_character_context 工具 + 专用提示词），
不再有独立旁路。回答结束后只入队 memory_jobs，由 worker 维护长期记忆。
"""
import asyncio
import json
import time
import uuid
from typing import List

from fastapi import APIRouter, HTTPException, Request
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from sqlalchemy import delete, select, update
from sse_starlette.sse import EventSourceResponse

from app.config import settings
from app.core.context import get_current_user
from app.core.visibility import visible_user_filter
from app.core.logging_config import get_logger
from app.core.metrics import metrics
from app.core.query_rewriter import rewrite_query
from app.db import AsyncSessionLocal
from app.db.models import ChatMessage, ChatSession, KnowledgeFile
from app.services import memory_service
from app.models.schemas import ChatRequest, WorldSelectRequest
from app.agent.runtime import stream_agent_question
from app.services import world_service

log = get_logger("chat")
router = APIRouter()


def _to_lc_messages(history: List[dict]) -> List[BaseMessage]:
    """把持久化的简化消息转换为 LangChain 消息对象。"""
    messages: List[BaseMessage] = []
    for item in history:
        if item.get("role") == "user":
            messages.append(HumanMessage(content=item.get("content", "")))
        elif item.get("role") == "assistant":
            messages.append(AIMessage(content=item.get("content", "")))
    return messages


def _sse_event(event: str, payload) -> dict:
    """把事件名和负载编码为 SSE 响应对象。"""
    data = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return {"event": event, "data": data}


def _error_event(message: str, code: str = "agent_error", **extra) -> dict:
    """构造统一格式的 Agent 错误事件（稳定错误码 + 用户可读消息，不含异常原文）。"""
    return _sse_event("error", {"code": code, "message": message, **extra})


async def _persist_message(
    session_id: str,
    role: str,
    content: str,
    sources: list[dict] | None = None,
    status: str = "completed",
) -> int | None:
    """保存聊天消息、来源和运行终态；持久化失败只记录日志，不阻断已生成答案。"""
    try:
        async with AsyncSessionLocal() as session:
            row = ChatMessage(
                session_id=session_id,
                role=role,
                content=content,
                sources=json.dumps(sources or [], ensure_ascii=False),
                status=status,
            )
            session.add(row)
            await session.commit()
            return row.id
    except Exception as exc:  # noqa: BLE001
        log.warning("chat.persist_failed", session_id=session_id, role=role, error=str(exc))
    return None


@router.post("/chat/sessions")
async def create_session(domain: str = "novel", file_id: str | None = None):
    """创建当前用户的聊天会话，并校验绑定小说已完成索引。"""
    if domain != "novel":
        raise HTTPException(status_code=400, detail="unsupported chat domain")
    user_id = get_current_user()
    async with AsyncSessionLocal() as session:
        if file_id:
            file_result = await session.execute(
                select(KnowledgeFile).where(
                    KnowledgeFile.id == file_id,
                    # 系统默认小说（system 租户）对所有用户可见
                    visible_user_filter(KnowledgeFile.user_id, user_id),
                    KnowledgeFile.domain == domain,
                    KnowledgeFile.status == "indexed",
                )
            )
            if file_result.scalars().first() is None:
                raise HTTPException(status_code=409, detail="目标小说尚未完成索引或不存在")
        session_id = uuid.uuid4().hex
        session.add(ChatSession(
            id=session_id,
            user_id=user_id,
            domain=domain,
            file_id=file_id,
        ))
        await session.commit()
    return {"code": 0, "data": {"id": session_id, "title": "new chat", "file_id": file_id}}


@router.get("/chat/sessions")
async def list_sessions(file_id: str | None = None, limit: int = 50):
    """返回当前用户的会话列表，可按绑定小说过滤。"""
    async with AsyncSessionLocal() as session:
        stmt = select(ChatSession).where(ChatSession.user_id == get_current_user())
        if file_id:
            stmt = stmt.where(ChatSession.file_id == file_id)
        rows = list((await session.execute(
            stmt.order_by(ChatSession.updated_at.desc()).limit(max(1, min(limit, 200)))
        )).scalars().all())
    # 角色扮演历史排查用：确认返回的会话是否带人物标记。
    for row in rows:
        log.info(
            "chat.session_listed",
            session_id=row.id,
            personas=row.personas,
            chapter_until=row.chapter_until,
        )
    return {"code": 0, "data": [{
        "id": row.id,
        "title": row.title,
        "role": row.role,
        "domain": row.domain,
        "file_id": row.file_id,
        "personas": json.loads(row.personas or "[]") if getattr(row, "personas", None) else [],
        "chapter_until": row.chapter_until,
        "updated_at": row.updated_at.strftime("%Y-%m-%d %H:%M") if row.updated_at else "",
    } for row in rows]}


@router.get("/chat/sessions/{session_id}/messages")
async def get_messages(session_id: str):
    """读取当前用户会话的消息和 JSON 来源。"""
    async with AsyncSessionLocal() as session:
        session_result = await session.execute(
            select(ChatSession).where(
                ChatSession.id == session_id,
                ChatSession.user_id == get_current_user(),
            )
        )
        if session_result.scalars().first() is None:
            raise HTTPException(status_code=404, detail="session not found")
        result = await session.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.id)
        )
        rows = result.scalars().all()
    return {"code": 0, "data": [{
        "id": message.id,
        "role": message.role,
        "content": message.content,
        "sources": json.loads(message.sources or "[]"),
        "status": getattr(message, "status", "completed") or "completed",
    } for message in rows]}


@router.patch("/chat/sessions/{session_id}")
async def rename_session(session_id: str, payload: dict):
    """重命名当前用户的会话。"""
    title = str(payload.get("title") or "").strip()[:50]
    if not title:
        raise HTTPException(status_code=400, detail="标题不能为空")
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            update(ChatSession)
            .where(ChatSession.id == session_id, ChatSession.user_id == get_current_user())
            .values(title=title)
        )
        renamed = result.rowcount
        await session.commit()
    if not renamed:
        raise HTTPException(status_code=404, detail="session not found")
    return {"code": 0, "data": {"id": session_id, "title": title}}


@router.get("/chat/world/characters")
async def world_characters(file_id: str, chapter_until: int | None = None):
    """返回小说世界的推荐人物名册（有缓存用缓存，否则 LLM 提取）。"""
    try:
        data = await world_service.list_characters(file_id, chapter_until)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except RuntimeError as exc:
        # 典型：索引由旧 embedding 模型生成，与当前检索配置不兼容。
        raise HTTPException(status_code=409, detail=str(exc))
    return {"code": 0, "data": data}


@router.post("/chat/world/characters/select")
async def world_select(payload: WorldSelectRequest):
    """为选中的人物生成或返回角色卡（1~3 个）与开场情景。"""
    try:
        result = await world_service.get_character_cards(
            payload.file_id, payload.names, payload.chapter_until,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"code": 0, "data": result}


@router.delete("/chat/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除当前用户的会话；消息、会话记忆与会话摘要由外键级联清理。"""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            delete(ChatSession).where(
                ChatSession.id == session_id,
                ChatSession.user_id == get_current_user(),
            )
        )
        deleted = result.rowcount
        await session.commit()
    if not deleted:
        raise HTTPException(status_code=404, detail="session not found")
    return {"code": 0, "data": {"id": session_id, "deleted": True}}


async def _chat_stream_response(req: ChatRequest, request: Request, persist: bool = True):
    """处理一次聊天请求，先做会话校验和 Query Rewrite，再转发 Agent 事件。"""
    if not req.file_id:
        raise HTTPException(status_code=400, detail="请先从小说列表中选择当前咨询对象")

    session_id = req.session_id or uuid.uuid4().hex
    user_id = get_current_user()
    # 角色扮演历史排查用：记录请求里的关键路由字段。
    log.info(
        "chat.request_fields",
        session_id=session_id,
        strategy=req.strategy,
        personas=req.personas,
        chapter_until=req.chapter_until,
    )

    if persist:
        async with AsyncSessionLocal() as session:
            target_file = await session.execute(
                select(KnowledgeFile).where(
                    KnowledgeFile.id == req.file_id,
                    visible_user_filter(KnowledgeFile.user_id, user_id),
                    KnowledgeFile.domain == req.domain,
                    KnowledgeFile.status == "indexed",
                )
            )
            if target_file.scalars().first() is None:
                raise HTTPException(status_code=409, detail="目标小说尚未完成索引或不存在")
            result = await session.execute(
                select(ChatSession).where(
                    ChatSession.id == session_id,
                    ChatSession.user_id == user_id,
                )
            )
            chat_session = result.scalars().first()
            if chat_session is None:
                # Reject reuse of another tenant's existing session id.
                existing = await session.execute(select(ChatSession.id).where(ChatSession.id == session_id))
                if existing.scalar_one_or_none() is not None:
                    raise HTTPException(status_code=404, detail="session not found")
                chat_session = ChatSession(
                    id=session_id,
                    role=req.role,
                    user_id=user_id,
                    domain=req.domain,
                    file_id=req.file_id,
                    # 首条消息自动命名，否则会话列表全是"新对话"无法区分。
                    title=(req.message.strip() or "新对话")[:30],
                    personas=json.dumps(req.personas or [], ensure_ascii=False),
                    chapter_until=req.chapter_until,
                )
                session.add(chat_session)
                await session.commit()
            elif chat_session.domain != req.domain:
                raise HTTPException(status_code=409, detail="session domain mismatch")
            elif chat_session.file_id and chat_session.file_id != req.file_id:
                raise HTTPException(status_code=409, detail="当前会话绑定了另一部小说，请切换会话")
            elif chat_session.file_id is None:
                chat_session.file_id = req.file_id
                await session.commit()
            if chat_session.title == "新对话":
                chat_session.title = (req.message.strip() or "新对话")[:30]
                await session.commit()
            # 懒创建路径：create_session 先建了空会话，首条消息到达时在这里补上
            # 人物与剧情边界（只补一次，避免后续轮次覆盖用户在 /world 的选择）。
            if req.personas and json.loads(chat_session.personas or "[]") == []:
                chat_session.personas = json.dumps(req.personas, ensure_ascii=False)
                chat_session.chapter_until = req.chapter_until
                await session.commit()
            # A2: bounded read - rewrite only needs the last query_rewrite_history_messages rows
            # (mirrors Zep bounded-read; raw rows stay forever).
            # partial/failed 的残缺回答不参与 Query 改写历史（completed 之外一律排除）。
            history_result = await session.execute(
                select(ChatMessage)
                .where(
                    ChatMessage.session_id == session_id,
                    ChatMessage.status == "completed",
                )
                .order_by(ChatMessage.id.desc())
                .limit(settings.query_rewrite_history_messages)
            )
            history_rows = list(reversed(list(history_result.scalars().all())))
            history_messages = _to_lc_messages([
                {"role": row.role, "content": row.content} for row in history_rows
            ])
        user_message_id = await _persist_message(session_id, "user", req.message)
    else:
        history_messages = _to_lc_messages(req.history or [])
        user_message_id = None

    # 事件生成器负责把 Agent 内部事件转换为前端约定的 SSE 事件，并在断开时清理任务。
    async def event_gen():
        yield {"event": "session", "data": session_id}
        full_reply: list[str] = []
        reply_sources: list[dict] = []
        started = time.perf_counter()
        model_memory_ops = 0
        completion_status = "completed"
        metrics.incr("chat_requests")
        try:
            async with asyncio.timeout(settings.agent_request_timeout):
                if await request.is_disconnected():
                    metrics.incr("sse_cancellations")
                    log.info("chat.novel_client_disconnected", session_id=session_id)
                    return
                memory_context: dict = {}
                if req.interaction_mode == "roleplay":
                    # 角色扮演不注入长期记忆：普通问答的偏好（如"只给总结"）会污染人设，
                    # 记忆维护任务也不创建（人设对话不产生偏好/事实沉淀）。
                    req.memory_mode = "off"
                if req.memory_mode == "auto" and settings.memory_enabled:
                    memory_context = await memory_service.safe_build_context(
                        session_id=session_id,
                        file_id=req.file_id,
                        query=req.message,
                    )
                    yield _sse_event("memory_context", {
                        "summary": memory_context.get("summary", ""),
                        "summary_id": memory_context.get("summary_id"),
                        "memories": memory_context.get("memories", []),
                        "output_policy": memory_context.get("output_policy", {}),
                        "count": len(memory_context.get("memories", [])),
                    })
                if req.interaction_mode == "roleplay":
                    # 角色扮演：跳过 Query 改写（角色卡即知识边界），统一图内按需补充检索。
                    standalone_query = req.message
                    query_preparation: dict = {}
                else:
                    rewrite = await rewrite_query(
                        req.message,
                        history_messages,
                        memory_context=memory_context,
                    )
                    standalone_query = rewrite.standalone_query
                    query_preparation = rewrite.as_dict()
                if await request.is_disconnected():
                    metrics.incr("sse_cancellations")
                    log.info("chat.novel_client_disconnected", session_id=session_id)
                    return

                async for stream_event in stream_agent_question(
                    standalone_query,
                    req.strategy,
                    req.file_id,
                    req.max_steps,
                    original_query=req.message,
                    retrieval_query=query_preparation.get("retrieval_query") or req.message,
                    query_preparation=query_preparation,
                    memory_context=memory_context,
                    session_id=session_id,
                    memory_agent_active=(
                        req.memory_mode == "auto"
                        and settings.memory_enabled
                        and settings.memory_agent_enabled
                    ),
                    interaction_mode=req.interaction_mode,
                    personas=req.personas or [],
                    chapter_until=req.chapter_until,
                    roleplay_history=history_messages,
                    deprecations=list(req.deprecations),
                ):
                    if await request.is_disconnected():
                        metrics.incr("sse_cancellations")
                        log.info("chat.novel_client_disconnected", session_id=session_id)
                        return

                    event_type = stream_event["type"]
                    payload = stream_event.get("data")
                    if event_type == "sources":
                        reply_sources = payload or []
                        yield _sse_event("sources", reply_sources)
                    elif event_type in {"run_started", "route", "plan", "step_start", "observation", "reflection", "validation", "agent_decision", "thinking_start", "thinking_token", "thinking_end"}:
                        yield _sse_event(event_type, payload)
                    elif event_type == "tool_start":
                        yield _sse_event("tool_start", payload)
                    elif event_type == "tool_token":
                        yield _sse_event("tool_token", payload)
                    elif event_type == "tool_end":
                        yield _sse_event("tool_end", payload)
                    elif event_type == "token":
                        token = str(payload or "")
                        full_reply.append(token)
                        yield _sse_event("token", token)
                    elif event_type == "token_replace":
                        # 输出护栏净化稿/修复稿覆盖流式拼接的内容，持久化以替换稿为准。
                        replaced = str(payload or "")
                        if replaced:
                            full_reply.clear()
                            full_reply.append(replaced)
                        yield _sse_event("token_replace", replaced)
                    elif event_type == "error":
                        completion_status = "failed"
                        yield _sse_event("error", payload)
                    elif event_type == "meta":
                        meta = payload or {}
                        model_memory_ops = len(meta.get("memory_ops") or [])
                        log.info(
                            "chat.agent_finished",
                            session_id=session_id,
                            requested_strategy=meta.get("requested_strategy", req.strategy),
                            effective_strategy=meta.get("effective_strategy"),
                            interaction_mode=meta.get("interaction_mode"),
                            steps=meta.get("steps"),
                            grounding_status=meta.get("grounding_status"),
                            stop_reason=meta.get("stop_reason"),
                        )
                        # meta 必须转发给前端：onMeta 依赖它更新 output_policy、
                        # effective_strategy 与验证/终态信息。
                        yield _sse_event("meta", meta)

                log.info(
                    "chat.novel_answered",
                    session_id=session_id,
                    interaction_mode=req.interaction_mode,
                    strategy=req.strategy,
                    sources=len(reply_sources),
                    completion_status=completion_status,
                )
        except TimeoutError:
            metrics.error()
            completion_status = "partial" if "".join(full_reply).strip() else "timeout"
            log.warning("chat.request_timeout", session_id=session_id, domain=req.domain, status=completion_status)
            yield _error_event("本次回答超过总超时，已停止生成。", "request_timeout")
        except asyncio.CancelledError:
            metrics.incr("sse_cancellations")
            raise
        except Exception as exc:  # noqa: BLE001
            metrics.error()
            completion_status = "partial" if "".join(full_reply).strip() else "failed"
            log.error("chat.stream_failed", session_id=session_id, status=completion_status, error=str(exc))
            # 错误响应只含稳定错误码与用户消息；异常原文只进服务端日志。
            yield _error_event("聊天流处理失败，请稍后重试。", "chat_stream_failed")
        finally:
            metrics.record_latency("chat", (time.perf_counter() - started) * 1000)

        reply = "".join(full_reply)
        if persist:
            # 只有正常完成的回答保存为正常历史；超时/异常的残缺输出标记 partial
            # 保留但不参与后续 Query 改写。
            persist_status = completion_status if completion_status in {"completed", "partial"} else "failed"
            assistant_message_id = await _persist_message(
                session_id, "assistant", reply, reply_sources, status=persist_status,
            )
            if req.memory_mode == "auto" and settings.memory_enabled and reply.strip():
                # 回答路径零直接写库：整轮记忆维护入队，worker 原子领取执行。
                await memory_service.enqueue_memory_job(
                    kind="maintain",
                    payload={
                        "user_text": req.message,
                        "assistant_text": reply,
                        "assistant_message_id": assistant_message_id,
                        "skip_extract": model_memory_ops > 0,
                    },
                    session_id=session_id,
                    file_id=req.file_id,
                    assistant_message_id=assistant_message_id,
                    priority="normal",
                )
                # The actual maintenance is intentionally detached; this event lets the UI
                # refresh/label the memory panel without delaying the answer stream.
                yield _sse_event("memory_updated", {"status": "scheduled"})
        yield {"event": "done", "data": ""}

    return EventSourceResponse(
        event_gen(),
        ping=10,
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/chat")
async def chat_endpoint(req: ChatRequest, request: Request):
    """聊天接口入口，返回最终答案和专家过程的 SSE 流。"""
    return await _chat_stream_response(req, request, persist=True)
