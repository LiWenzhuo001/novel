"""聊天会话管理和 Agent SSE 流式问答接口。"""
import asyncio
import json
import time
import uuid
from typing import List

from fastapi import APIRouter, HTTPException, Request
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from sqlalchemy import select
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
from app.models.schemas import ChatRequest
from app.agent.runtime import stream_agent_question

log = get_logger("chat")
router = APIRouter()

# fire-and-forget 的记忆维护任务必须持引用，否则可能被 GC 提前回收（asyncio 官方警告）。
_memory_tasks: set[asyncio.Task] = set()


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


def _tool_event(event: str, call_id: str, tool_name: str, **extra) -> dict:
    """构造统一格式的工具过程事件。"""
    payload = {"id": call_id, "tool": tool_name, **extra}
    return _sse_event(event, payload)


def _error_event(message: str, code: str = "agent_error", **extra) -> dict:
    """构造统一格式的 Agent 错误事件。"""
    return _sse_event("error", {"code": code, "message": message, **extra})


async def _persist_message(
    session_id: str,
    role: str,
    content: str,
    sources: list[dict] | None = None,
) -> int | None:
    """保存聊天消息和来源；持久化失败只记录日志，不阻断已生成答案。"""
    try:
        async with AsyncSessionLocal() as session:
            row = ChatMessage(
                session_id=session_id,
                role=role,
                content=content,
                sources=json.dumps(sources or [], ensure_ascii=False),
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
async def list_sessions():
    """返回当前用户的会话列表。"""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(ChatSession)
            .where(ChatSession.user_id == get_current_user())
            .order_by(ChatSession.updated_at.desc())
        )
        rows = result.scalars().all()
    return {"code": 0, "data": [{
        "id": row.id,
        "title": row.title,
        "role": row.role,
        "domain": row.domain,
        "file_id": row.file_id,
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
    } for message in rows]}


async def _chat_stream_response(req: ChatRequest, request: Request, persist: bool = True):
    """处理一次聊天请求，先做会话校验和 Query Rewrite，再转发 Agent 事件。"""
    if not req.file_id:
        raise HTTPException(status_code=400, detail="请先从小说列表中选择当前咨询对象")

    session_id = req.session_id or uuid.uuid4().hex
    user_id = get_current_user()

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
            # A2: bounded read - rewrite only needs the last query_rewrite_history_messages rows
            # (mirrors Zep bounded-read; raw rows stay forever).
            history_result = await session.execute(
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
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
        metrics.incr("chat_requests")
        try:
            async with asyncio.timeout(settings.agent_request_timeout):
                if await request.is_disconnected():
                    metrics.incr("sse_cancellations")
                    log.info("chat.novel_client_disconnected", session_id=session_id)
                    return
                memory_context: dict = {}
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
                rewrite = await rewrite_query(
                    req.message,
                    history_messages,
                    memory_context=memory_context,
                )
                if await request.is_disconnected():
                    metrics.incr("sse_cancellations")
                    log.info("chat.novel_client_disconnected", session_id=session_id)
                    return

                fallback_reason = ""
                async for stream_event in stream_agent_question(
                    rewrite.standalone_query,
                    req.strategy,
                    req.file_id,
                    req.max_steps,
                    original_query=req.message,
                    retrieval_query=rewrite.retrieval_query,
                    query_preparation=rewrite.as_dict(),
                    memory_context=memory_context,
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
                    elif event_type in {"route", "plan", "step_start", "observation", "reflection", "expert_tasks", "validation"}:
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
                        # 输出护栏净化稿覆盖流式拼接的内容，持久化以净化稿为准。
                        replaced = str(payload or "")
                        if replaced:
                            full_reply.clear()
                            full_reply.append(replaced)
                        yield _sse_event("token_replace", replaced)
                    elif event_type == "error":
                        yield _sse_event("error", payload)
                    elif event_type == "meta":
                        meta = payload or {}
                        fallback_reason = meta.get("fallback_reason", "")
                        log.info(
                            "chat.agent_finished",
                            session_id=session_id,
                            strategy=meta.get("strategy", req.strategy or "legacy"),
                            steps=meta.get("steps"),
                            fallback_reason=fallback_reason,
                        )
                        # meta 必须转发给前端：onMeta 依赖它更新 output_policy 与
                        # 策略/回退信息（此前只记日志不下发，前端永远收不到）。
                        yield _sse_event("meta", meta)

                log.info(
                    "chat.novel_answered",
                    session_id=session_id,
                    strategy=req.strategy,
                    sources=len(reply_sources),
                    fallback_reason=fallback_reason,
                )
        except TimeoutError:
            metrics.error()
            log.warning("chat.request_timeout", session_id=session_id, domain=req.domain)
            yield _error_event("本次回答超过总超时，已停止生成。", "request_timeout")
        except asyncio.CancelledError:
            metrics.incr("sse_cancellations")
            raise
        except Exception as exc:  # noqa: BLE001
            metrics.error()
            log.error("chat.stream_failed", session_id=session_id, error=str(exc))
            yield _error_event(f"聊天流处理失败：{exc}")
        finally:
            metrics.record_latency("chat", (time.perf_counter() - started) * 1000)

        reply = "".join(full_reply)
        if persist:
            assistant_message_id = await _persist_message(session_id, "assistant", reply, reply_sources)
            if req.memory_mode == "auto" and settings.memory_enabled and reply.strip():
                async def update_memory_background() -> None:
                    await memory_service.maintain_conversation_memory_safe(
                        session_id=session_id,
                        file_id=req.file_id,
                        user_text=req.message,
                        assistant_text=reply,
                        assistant_message_id=assistant_message_id,
                    )

                task = asyncio.create_task(update_memory_background())
                _memory_tasks.add(task)
                task.add_done_callback(_memory_tasks.discard)
                # The actual extraction is intentionally detached; this event lets the UI
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
