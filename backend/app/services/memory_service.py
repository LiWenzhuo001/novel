"""Automatic conversation summaries and scoped long-term memory helpers."""
from __future__ import annotations

import asyncio
import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import delete, or_, select, update

from app.config import settings
from app.agent.types import DEFAULT_OUTPUT_POLICY
from app.core.context import get_current_user
from app.core.llm import get_llm
from app.core.logging_config import get_logger
from app.db import AsyncSessionLocal
from app.db.models import AgentMemory, ChatMessage, ConversationSummary

log = get_logger("memory_service")

# 提示词允许模型回复"无新增"；这四个字不能落库，否则会顶掉真实历史摘要。
_NO_NEW_SUMMARY_RE = re.compile(r"^(?:无新增|没有新增)\s*[。.！!?？]*$")


@dataclass(frozen=True)
class MemoryContext:
    """Serializable context injected into Query Rewrite and Supervisor."""

    summary: str = ""
    summary_id: str | None = None
    memories: tuple[dict[str, Any], ...] = ()
    output_policy: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_OUTPUT_POLICY))

    def as_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "summary_id": self.summary_id,
            "memories": list(self.memories),
            "output_policy": dict(self.output_policy or DEFAULT_OUTPUT_POLICY),
        }


def _metadata(row: AgentMemory) -> dict[str, Any]:
    try:
        value = json.loads(row.meta_json or "{}")
        return value if isinstance(value, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def memory_to_dict(row: AgentMemory) -> dict[str, Any]:
    return {
        "id": row.id,
        "memory_type": row.memory_type,
        "preference_key": getattr(row, "preference_key", None),
        "memory_version": getattr(row, "memory_version", 1),
        "content": row.content,
        "importance": float(row.importance or 0.0),
        "session_id": row.session_id,
        "file_id": row.file_id,
        "source_message_id": row.source_message_id,
        "metadata": _metadata(row),
        "expires_at": row.expires_at.isoformat() if row.expires_at else None,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


async def get_latest_summary(session_id: str) -> ConversationSummary | None:
    """读取当前用户和会话的最新摘要。"""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(ConversationSummary)
            .where(
                ConversationSummary.session_id == session_id,
                ConversationSummary.user_id == get_current_user(),
            )
            .order_by(ConversationSummary.updated_at.desc(), ConversationSummary.created_at.desc())
            .limit(1)
        )
        return result.scalars().first()


async def save_summary(
    session_id: str,
    summary: str,
    covered_message_id: int = 0,
    token_estimate: int = 0,
) -> ConversationSummary:
    """保存新的会话摘要及其覆盖的消息范围。"""
    async with AsyncSessionLocal() as session:
        row = ConversationSummary(
            id=uuid.uuid4().hex,
            session_id=session_id,
            user_id=get_current_user(),
            summary=summary.strip(),
            covered_message_id=covered_message_id,
            token_estimate=max(0, token_estimate),
        )
        session.add(row)
        await session.commit()
        return row


async def _embed_text(content: str) -> list[float] | None:
    """尽力生成记忆向量；embedding 服务不可用时由调用方回退文本记忆。"""
    try:
        from app.core.embed import get_embeddings

        embeddings = await asyncio.to_thread(get_embeddings)
        return await asyncio.to_thread(embeddings.embed_query, content)
    except Exception:
        return None


async def save_memory(
    content: str,
    memory_type: str = "session_fact",
    *,
    session_id: str | None = None,
    file_id: str | None = None,
    importance: float = 0.5,
    source_message_id: int | None = None,
    expires_at: datetime | None = None,
    ttl_minutes: int | None = None,
    metadata: dict[str, Any] | None = None,
    preference_key: str | None = None,
) -> AgentMemory:
    """保存或更新一条用户、小说或会话范围的长期记忆。"""
    normalized = re.sub(r"\s+", " ", content.strip())
    if not normalized:
        raise ValueError("memory content cannot be empty")
    user_id = get_current_user()
    if ttl_minutes is not None and expires_at is None:
        expires_at = datetime.utcnow() + timedelta(minutes=ttl_minutes)
    embedding = await _embed_text(normalized)
    async with AsyncSessionLocal() as session:
        duplicate = await session.execute(
            select(AgentMemory).where(
                AgentMemory.user_id == user_id,
                AgentMemory.memory_type == memory_type,
                AgentMemory.session_id == session_id,
                AgentMemory.file_id == file_id,
                AgentMemory.preference_key == preference_key if preference_key else AgentMemory.content == normalized,
            ).limit(1)
        )
        row = duplicate.scalars().first()
        if row is None:
            row = AgentMemory(
                id=uuid.uuid4().hex,
                user_id=user_id,
                session_id=session_id,
                file_id=file_id,
                memory_type=memory_type,
                preference_key=preference_key,
                memory_version=1,
                content=normalized,
                embedding=embedding,
                importance=min(1.0, max(0.0, importance)),
                source_message_id=source_message_id,
                expires_at=expires_at,
                ttl_minutes=ttl_minutes,
                meta_json=json.dumps(metadata or {}, ensure_ascii=False),
            )
            session.add(row)
        else:
            row.importance = max(float(row.importance or 0.0), min(1.0, max(0.0, importance)))
            row.source_message_id = source_message_id or row.source_message_id
            if ttl_minutes is not None:
                row.ttl_minutes = ttl_minutes
            row.expires_at = expires_at or row.expires_at
            if preference_key:
                row.preference_key = preference_key
                row.memory_version = (row.memory_version or 1) + 1
                row.content = normalized
                row.meta_json = json.dumps(metadata or {}, ensure_ascii=False)
            if embedding is not None:
                row.embedding = embedding
        await session.commit()
        return row


async def list_memories(
    *,
    session_id: str | None = None,
    file_id: str | None = None,
    limit: int = 10,
) -> list[AgentMemory]:
    """按精确作用域列出当前用户可见记忆；兼容旧调用方。"""
    async with AsyncSessionLocal() as session:
        stmt = select(AgentMemory).where(AgentMemory.user_id == get_current_user())
        if session_id:
            stmt = stmt.where(AgentMemory.session_id == session_id)
        if file_id:
            stmt = stmt.where(AgentMemory.file_id == file_id)
        stmt = stmt.where(
            (AgentMemory.expires_at.is_(None)) | (AgentMemory.expires_at > datetime.utcnow())
        ).order_by(AgentMemory.importance.desc(), AgentMemory.updated_at.desc()).limit(max(1, min(limit, 100)))
        result = await session.execute(stmt)
        return list(result.scalars().all())


async def retrieve_memories(
    *,
    query: str,
    session_id: str,
    file_id: str | None,
    limit: int | None = None,
) -> list[AgentMemory]:
    """按当前会话、当前小说、用户偏好三层作用域召回记忆。

    用户偏好不是普通语义文档：必须始终纳入上下文，避免“不要展示原文”
    因为向量相关性不足而在下一轮失效；其余记忆再按相关性排序截断。
    """
    user_id = get_current_user()
    limit = max(1, min(limit or settings.memory_max_context_items,  50))
    pref_budget = max(0, min(settings.memory_preference_top_k, limit))
    fact_budget = max(1, min(settings.memory_fact_pool_size, limit - pref_budget))
    candidate_limit = max(limit, min(settings.memory_candidate_limit, 200))
    query_vector = await _embed_text(query) if query.strip() else None
    scope = or_(
        AgentMemory.session_id == session_id,
        AgentMemory.file_id == file_id if file_id else AgentMemory.file_id.is_(None),
        (AgentMemory.session_id.is_(None) & AgentMemory.file_id.is_(None)),
    )
    async with AsyncSessionLocal() as session:
        base = select(AgentMemory).where(
            AgentMemory.user_id == user_id,
            scope,
            (AgentMemory.expires_at.is_(None)) | (AgentMemory.expires_at > datetime.utcnow()),
        )
        preference_result = await session.execute(
            base.where(AgentMemory.memory_type == "user_preference")
            .order_by(AgentMemory.updated_at.desc())
            .limit(pref_budget)
        )
        preferences = list(preference_result.scalars().all())
        other_stmt = base.where(AgentMemory.memory_type != "user_preference")
        if query_vector:
            other_stmt = other_stmt.order_by(
                AgentMemory.embedding.cosine_distance(query_vector).asc().nullslast(),
                AgentMemory.importance.desc(),
                AgentMemory.updated_at.desc(),
            )
        else:
            other_stmt = other_stmt.order_by(AgentMemory.importance.desc(), AgentMemory.updated_at.desc())
        other_result = await session.execute(other_stmt.limit(candidate_limit))
        others = list(other_result.scalars().all())
    returned = preferences[:pref_budget] + others[:fact_budget]
    # B2: refresh-on-access for TTL rows (LangGraph refresh_ttl pattern); keep hot memories alive.
    # Recall rows are bounded (<= memory_max_context_items), so per-row refresh cost is negligible.
    if any(row.ttl_minutes for row in returned):
        utcnow = datetime.utcnow()
        for row in returned:
            if not row.ttl_minutes or not row.expires_at:
                continue
            expires = utcnow + timedelta(minutes=row.ttl_minutes)
            async with AsyncSessionLocal() as sess:
                await sess.execute(update(AgentMemory).where(AgentMemory.id == row.id).values(expires_at=expires))
                await sess.commit()
    return returned


async def build_memory_context(*, session_id: str, file_id: str | None, query: str) -> dict[str, Any]:
    """读取摘要与三层相关记忆；用户偏好以自由文本形式随 memories 注入。"""
    summary = await get_latest_summary(session_id)
    memories_rows = await retrieve_memories(query=query, session_id=session_id, file_id=file_id)
    memories = tuple(memory_to_dict(row) for row in memories_rows)
    return MemoryContext(
        summary=summary.summary if summary else "",
        summary_id=summary.id if summary else None,
        memories=memories,
    ).as_dict()


async def delete_memory(memory_id: str) -> bool:
    """删除当前用户拥有的指定记忆。"""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            delete(AgentMemory).where(
                AgentMemory.id == memory_id,
                AgentMemory.user_id == get_current_user(),
            )
        )
        await session.commit()
        return bool(result.rowcount)


async def sweep_expired_memories(batch: int = 500) -> int:
    """Delete expired TTL rows (LangGraph start_ttl_sweeper pattern)."""
    async with AsyncSessionLocal() as session:
        ids = list((await session.execute(
            select(AgentMemory.id).where(
                AgentMemory.expires_at.is_not(None),
                AgentMemory.expires_at < datetime.utcnow(),
            ).limit(batch)
        )).scalars().all())
        if not ids:
            return 0
        await session.execute(delete(AgentMemory).where(AgentMemory.id.in_(ids)))
        await session.commit()
        return len(ids)


async def _llm_text(prompt: str, max_tokens: int) -> str:
    model = get_llm(temperature=0, max_tokens=max_tokens, timeout=settings.memory_task_timeout, max_retries=0)
    response = await asyncio.wait_for(model.ainvoke([{"role": "user", "content": prompt}]), settings.memory_task_timeout)
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "".join(block.get("text", "") for block in content if isinstance(block, dict))
    return str(content or "").strip()


def _json_payload(text: str) -> Any:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I | re.S).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", cleaned)
        return json.loads(match.group(1)) if match else None


async def _generate_summary(previous: str, messages: list[ChatMessage]) -> str:
    transcript = "\n".join(f"{row.role}: {row.content}" for row in messages)
    prompt = (
        "请把下面的小说问答会话压缩为简洁、事实化的中文摘要。保留用户正在关注的作品、人物、"
        "问题、已确认结论、未解决疑问和用户表达的偏好。不要加入原文没有的事实，不要使用 Markdown。\n\n"
        "在已有摘要上合并新增消息的事实，不要重写整段；若新增消息没有值得保留的新事实，直接回复'无新增'即可，"
        f"已有摘要：{previous or '无'}\n\n新增消息：\n{transcript}"
    )
    return (await _llm_text(prompt, settings.memory_extract_max_tokens)).strip()


async def _update_memory(
    memory_id: str,
    *,
    content: str,
    importance: float | None = None,
) -> AgentMemory | None:
    """按 id 更新当前用户的一条记忆并重嵌向量；id 不属于该用户时返回 None。"""
    normalized = re.sub(r"\s+", " ", content.strip())
    if not normalized:
        return None
    embedding = await _embed_text(normalized)
    async with AsyncSessionLocal() as session:
        row = (await session.execute(
            select(AgentMemory).where(
                AgentMemory.id == memory_id,
                AgentMemory.user_id == get_current_user(),
            )
        )).scalars().first()
        if row is None:
            return None
        row.content = normalized
        if importance is not None:
            row.importance = min(1.0, max(0.0, importance))
        if embedding is not None:
            row.embedding = embedding
        await session.commit()
        return row


async def _extract_memories(
    user_text: str,
    assistant_text: str,
    file_id: str | None,
    existing: list[AgentMemory] | None = None,
) -> list[dict[str, Any]]:
    """对齐 mem0/LangGraph 维护模式：带已有记忆上下文，输出 add/update/delete 操作。"""
    existing = existing or []
    existing_lines = "\n".join(
        f"- id={row.id} ({row.memory_type}) {row.content}"
        for row in existing
    ) or "（无）"
    prompt = (
        "结合本轮对话维护长期记忆，只输出 JSON 数组。允许的 memory_type 只有 user_preference、novel_fact、session_fact。"
        "user_preference 必须来自用户明确表达的偏好；novel_fact 必须是对当前小说有帮助的稳定事实；"
        "session_fact 仅保存当前会话仍会使用的明确事实。不要保存问候、临时推断、助手自创内容或普通回答。\n"
        "每条记忆带 op 操作：\n"
        '- "add"：本轮对话带来的、已有记忆未覆盖的新信息，字段 content、memory_type、importance；\n'
        '- "update"：本轮对话修正、细化或取代了某条已有记忆时，必须改写该条而不是新增，字段 id（取自已有记忆列表）、content（改写后的完整内容）、importance；\n'
        '- "delete"：本轮对话明确废弃某条已有记忆（如撤销偏好、纠正错误事实），字段 id。\n'
        "与已有记忆矛盾的描述禁止并存，必须用 update 或 delete 处理旧条；与已有记忆重复或无变化时不要输出。"
        "update/delete 的 id 只能取自已有记忆列表。\n"
        f"最多输出 {settings.memory_extract_max_items} 条操作。"
        f"add 的 importance 是 0 到 1 的浮点数（如 0.9），表示对后续对话的长期价值；"
        f"低于 {settings.memory_min_importance} 的条目会被丢弃，请给出明确的数字。\n\n"
        f"当前小说 file_id：{file_id or '未知'}\n用户：{user_text}\n助手：{assistant_text}\n\n"
        f"已有记忆：\n{existing_lines}"
    )
    payload = _json_payload(await _llm_text(prompt, settings.memory_extract_max_tokens))
    if not isinstance(payload, list):
        # 提取失败不能完全静默，否则整轮记忆蒸发时无从排查。
        log.warning("memory_extract.parse_failed", user_text=user_text[:60])
        return []
    allowed = {"user_preference", "novel_fact", "session_fact"}
    known_ids = {row.id for row in existing}
    ops: list[dict[str, Any]] = []
    for item in payload[: settings.memory_extract_max_items]:
        if not isinstance(item, dict):
            continue
        op = str(item.get("op") or "add").strip().lower()
        if op == "delete":
            memory_id = str(item.get("id") or "").strip()
            if memory_id in known_ids:
                ops.append({"op": "delete", "id": memory_id})
            else:
                log.warning("memory_extract.delete_unknown_id", id=memory_id[:40])
            continue
        content = str(item.get("content") or "").strip()
        try:
            importance = float(item.get("importance", 0.5))
        except (TypeError, ValueError):
            importance = 0.5
        if op == "update":
            # 改写保持原行的 memory_type，因此这里不校验类型字段。
            memory_id = str(item.get("id") or "").strip()
            if memory_id not in known_ids:
                log.warning("memory_extract.update_unknown_id", id=memory_id[:40])
                continue
            if not content:
                continue
            ops.append({"op": "update", "id": memory_id, "content": content, "importance": importance})
            continue
        memory_type = str(item.get("memory_type") or "").strip()
        if not content or memory_type not in allowed:
            continue
        if importance < settings.memory_min_importance:
            log.warning(
                "memory_extract.item_dropped",
                memory_type=memory_type,
                importance=importance,
                content=content[:40],
            )
            continue
        ops.append({"op": "add", "content": content, "memory_type": memory_type, "importance": importance})
    return ops


async def maintain_conversation_memory(
    *,
    session_id: str,
    file_id: str | None,
    user_text: str,
    assistant_text: str,
    assistant_message_id: int | None = None,
) -> dict[str, Any]:
    """后台更新摘要和长期记忆；失败由调用方捕获，不影响回答。"""
    summary_updated = False
    memories_added: list[dict[str, Any]] = []
    if not settings.memory_enabled:
        return {"summary_updated": False, "memories_added": []}

    if assistant_text.strip():
        # 先召回已有记忆给提取 LLM，它才能把修正/废弃落到旧条上，而不是追加矛盾条目。
        existing = await retrieve_memories(query=user_text, session_id=session_id, file_id=file_id)
        for item in await _extract_memories(user_text, assistant_text, file_id, existing):
            if item["op"] == "delete":
                await delete_memory(item["id"])
                continue
            if item["op"] == "update":
                row = await _update_memory(
                    item["id"],
                    content=item["content"],
                    importance=item.get("importance"),
                )
            else:
                row = await save_memory(
                    item["content"], item["memory_type"],
                    session_id=session_id if item["memory_type"] == "session_fact" else None,
                    file_id=file_id if item["memory_type"] == "novel_fact" else None,
                    importance=item["importance"],
                    source_message_id=assistant_message_id,
                    ttl_minutes=(
                        settings.memory_session_fact_ttl_days * 24 * 60
                        if item["memory_type"] == "session_fact" and settings.memory_session_fact_ttl_days > 0
                        else None
                    ),
                )
            if row is not None:
                memories_added.append(memory_to_dict(row))

    summary = await get_latest_summary(session_id)
    covered_id = summary.covered_message_id if summary else 0
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(ChatMessage).where(
                ChatMessage.session_id == session_id,
                ChatMessage.id > covered_id,
            ).order_by(ChatMessage.id)
        )
        pending = list(result.scalars().all())
    pending_chars = sum(len(row.content or "") for row in pending)
    est_tokens = pending_chars // 4
    if pending and (
        len(pending) >= settings.memory_summary_trigger_messages
        or pending_chars >= settings.memory_summary_trigger_chars
        or est_tokens >= settings.memory_summary_trigger_tokens
    ):
        new_summary = await _generate_summary(summary.summary if summary else "", pending[-30:])
        if new_summary and not _NO_NEW_SUMMARY_RE.match(new_summary.strip()):
            await save_summary(
                session_id,
                new_summary,
                covered_message_id=pending[-1].id,
                token_estimate=max(1, len(new_summary) // 4),
            )
            summary_updated = True
    return {"summary_updated": summary_updated, "memories_added": memories_added}


# ===== 聊天编排用的安全包装（吞错降级语义集中在此，chat.py 只负责发事件）=====


async def safe_build_context(*, session_id: str, file_id: str | None, query: str) -> dict:
    """构建记忆上下文；任何失败都降级为空上下文并记录告警（不阻断问答）。"""
    try:
        return await build_memory_context(session_id=session_id, file_id=file_id, query=query)
    except Exception as exc:  # noqa: BLE001
        log.warning("chat.memory_context_failed", session_id=session_id, error=str(exc)[:200])
        return {}


async def maintain_conversation_memory_safe(
    *,
    session_id: str,
    file_id: str | None,
    user_text: str,
    assistant_text: str,
    assistant_message_id: int | None,
) -> None:
    """带超时与吞错的会话记忆维护；作为后台任务体运行。"""
    try:
        result = await asyncio.wait_for(
            maintain_conversation_memory(
                session_id=session_id,
                file_id=file_id,
                user_text=user_text,
                assistant_text=assistant_text,
                assistant_message_id=assistant_message_id,
            ),
            # 内层每次 LLM 调用已有独立超时；外层需覆盖"提取+摘要"两次串行调用再加余量，
            # 不能复用单次超时，否则稍慢一轮就整段记忆维护被取消。
            timeout=settings.memory_task_timeout * 2 + 5,
        )
        log.info(
            "chat.memory_updated",
            session_id=session_id,
            summary_updated=result.get("summary_updated", False),
            memories_added=len(result.get("memories_added", [])),
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        log.warning("chat.memory_update_failed", session_id=session_id, error=str(exc)[:200])
