"""Automatic conversation summaries and scoped long-term memory helpers."""
from __future__ import annotations

import asyncio
import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy import delete, or_, select

from app.config import settings
from app.agent.types import DEFAULT_OUTPUT_POLICY, merge_output_policy
from app.core.context import get_current_user
from app.core.llm import get_llm
from app.core.logging_config import get_logger
from app.db import AsyncSessionLocal
from app.db.models import AgentMemory, ChatMessage, ConversationSummary

log = get_logger("memory_service")


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
    metadata: dict[str, Any] | None = None,
    preference_key: str | None = None,
) -> AgentMemory:
    """保存或更新一条用户、小说或会话范围的长期记忆。"""
    normalized = re.sub(r"\s+", " ", content.strip())
    if not normalized:
        raise ValueError("memory content cannot be empty")
    user_id = get_current_user()
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
                meta_json=json.dumps(metadata or {}, ensure_ascii=False),
            )
            session.add(row)
        else:
            row.importance = max(float(row.importance or 0.0), min(1.0, max(0.0, importance)))
            row.source_message_id = source_message_id or row.source_message_id
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
    limit = max(1, min(limit or settings.memory_max_context_items, 50))
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
            .limit(10)
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
    rows = preferences + others
    return rows[:limit] if len(preferences) >= limit else preferences + others[: limit - len(preferences)]


async def build_memory_context(*, session_id: str, file_id: str | None, query: str) -> dict[str, Any]:
    """读取摘要、三层相关记忆和结构化用户输出偏好。"""
    summary = await get_latest_summary(session_id)
    memories_rows = await retrieve_memories(query=query, session_id=session_id, file_id=file_id)
    memories = tuple(memory_to_dict(row) for row in memories_rows)
    preference_policies = []
    for item in memories:
        if item.get("memory_type") != "user_preference":
            continue
        metadata = item.get("metadata") or {}
        if isinstance(metadata, dict):
            preference_policies.append(metadata.get("output_policy"))
    output_policy = merge_output_policy(*preference_policies)
    return MemoryContext(
        summary=summary.summary if summary else "",
        summary_id=summary.id if summary else None,
        memories=memories,
        output_policy=output_policy,
    ).as_dict()


async def upsert_preference(
    update: dict[str, Any],
    *,
    source_message_id: int | None = None,
) -> AgentMemory | None:
    """立即保存用户本轮明确偏好；后续重复设置同一 key 只更新。"""
    if not isinstance(update, dict) or update.get("preference_key") != "answer_presentation":
        return None
    value = update.get("value")
    if not isinstance(value, dict):
        return None
    content = "默认只提供总结，不展示小说原文或逐字引语"
    metadata = {
        "output_policy": {key: value[key] for key in DEFAULT_OUTPUT_POLICY if key in value},
        "source": update.get("source", "explicit_user_instruction"),
        "confidence": float(update.get("confidence", 0.99)),
    }
    return await save_memory(
        content,
        "user_preference",
        importance=1.0,
        source_message_id=source_message_id,
        metadata=metadata,
        preference_key="answer_presentation",
    )


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
        f"已有摘要：{previous or '无'}\n\n新增消息：\n{transcript}"
    )
    return (await _llm_text(prompt, settings.memory_extract_max_tokens)).strip()


async def _extract_memories(user_text: str, assistant_text: str, file_id: str | None) -> list[dict[str, Any]]:
    prompt = (
        "从这一轮用户与助手对话中抽取值得跨轮保留的稳定记忆，只输出 JSON 数组。"
        "允许的 memory_type 只有 user_preference、novel_fact、session_fact。"
        "user_preference 必须来自用户明确表达的偏好；novel_fact 必须是对当前小说有帮助的稳定事实；"
        "session_fact 仅保存当前会话仍会使用的明确事实。不要保存问候、临时推断、助手自创内容或普通回答。"
        f"最多输出 {settings.memory_extract_max_items} 条，每条包含 content、memory_type、importance。\n\n"
        f"当前小说 file_id：{file_id or '未知'}\n用户：{user_text}\n助手：{assistant_text}"
    )
    payload = _json_payload(await _llm_text(prompt, settings.memory_extract_max_tokens))
    if not isinstance(payload, list):
        return []
    allowed = {"user_preference", "novel_fact", "session_fact"}
    memories: list[dict[str, Any]] = []
    for item in payload[: settings.memory_extract_max_items]:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "").strip()
        memory_type = str(item.get("memory_type") or "").strip()
        try:
            importance = float(item.get("importance", 0.5))
        except (TypeError, ValueError):
            importance = 0.5
        if content and memory_type in allowed and importance >= settings.memory_min_importance:
            memories.append({"content": content, "memory_type": memory_type, "importance": importance})
    return memories


async def maintain_conversation_memory(
    *,
    session_id: str,
    file_id: str | None,
    user_text: str,
    assistant_text: str,
    assistant_message_id: int | None = None,
    preference_update: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """后台更新摘要和长期记忆；失败由调用方捕获，不影响回答。"""
    summary_updated = False
    memories_added: list[dict[str, Any]] = []
    if not settings.memory_enabled:
        return {"summary_updated": False, "memories_added": []}

    if preference_update:
        preference = await upsert_preference(preference_update, source_message_id=assistant_message_id)
        if preference is not None:
            memories_added.append(memory_to_dict(preference))

    if assistant_text.strip():
        for item in await _extract_memories(user_text, assistant_text, file_id):
            row = await save_memory(
                item["content"], item["memory_type"],
                session_id=session_id if item["memory_type"] == "session_fact" else None,
                file_id=file_id if item["memory_type"] == "novel_fact" else None,
                importance=item["importance"],
                source_message_id=assistant_message_id,
            )
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
    if pending and (
        len(pending) >= settings.memory_summary_trigger_messages
        or pending_chars >= settings.memory_summary_trigger_chars
    ):
        new_summary = await _generate_summary(summary.summary if summary else "", pending[-30:])
        if new_summary:
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


async def safe_upsert_preference(preference_update: dict, source_message_id: int | None) -> bool:
    """持久化用户偏好；失败仅记录告警并返回 False（不阻断问答）。"""
    try:
        await upsert_preference(preference_update, source_message_id=source_message_id)
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("chat.preference_update_failed", error=str(exc)[:200])
        return False


async def maintain_conversation_memory_safe(
    *,
    session_id: str,
    file_id: str | None,
    user_text: str,
    assistant_text: str,
    assistant_message_id: int | None,
    preference_update: dict | None,
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
                preference_update=preference_update,
            ),
            timeout=settings.memory_task_timeout,
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
