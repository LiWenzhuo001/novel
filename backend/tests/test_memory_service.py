"""memory tests: preference pool, TTL, token trigger."""
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime
from datetime import timedelta

import pytest
from sqlalchemy import select

from app.config import settings
from app.core.context import get_current_user
from app.core.context import set_current_user
from app.core.context import reset_current_user
from app.db import AsyncSessionLocal
from app.db.models import AgentMemory
from app.db.models import ChatMessage
from app.db.models import ChatSession
from app.db.models import ConversationSummary
from app.services import memory_service

# autouse fixture 会 mock 掉 LLM 抽取/摘要；需要测真实解析逻辑时用它恢复。
_REAL_EXTRACT = memory_service._extract_memories
_REAL_SUMMARY = memory_service._generate_summary

LOOP = asyncio.new_event_loop()


def run(coro):
    return LOOP.run_until_complete(coro)


@pytest.fixture
def memory_user():
    uid = f"test_mem_{uuid.uuid4().hex[:12]}"
    token = set_current_user(uid)
    try:
        yield uid
    finally:
        reset_current_user(token)


@pytest.fixture(autouse=True)
def no_embed(monkeypatch):
    async def fake_embed(content):
        return None

    async def no_extract(user_text, assistant_text, file_id, existing=None):
        return []

    async def fake_summary(previous, messages):
        return "new summary stub"

    monkeypatch.setattr(memory_service, "_embed_text", fake_embed)
    monkeypatch.setattr(memory_service, "_extract_memories", no_extract)
    monkeypatch.setattr(memory_service, "_generate_summary", fake_summary)


async def seed_messages(session_id, count, chars_per_msg):
    async with AsyncSessionLocal() as session:
        for i in range(count):
            session.add(ChatMessage(
                session_id=session_id,
                role="user",
                content=("a" * chars_per_msg) + f"#{i}",
            ))
        await session.commit()


async def ensure_session(session_id, user_id):
    async with AsyncSessionLocal() as session:
        session.add(ChatSession(
            id=session_id[:32],
            title="test session",
            role="user",
            user_id=user_id,
            domain="novel",
        ))
        await session.commit()


def test_preferences_do_not_crowd_out_facts(memory_user):
    async def scenario():
        sid = f"sess_{uuid.uuid4().hex[:8]}"
        await ensure_session(sid, get_current_user())
        for i in range(8):
            await memory_service.save_memory(
                f"pref {i} no verbatim quotes",
                "user_preference",
                session_id=sid,
                importance=1.0,
            )
        for i in range(6):
            await memory_service.save_memory(
                f"fact {i} the hero is Lin Mo",
                "session_fact",
                session_id=sid,
                importance=0.95,
            )

        rows = await memory_service.retrieve_memories(
            query="anything", session_id=sid, file_id=None,
        )
        assert len(rows) <=  8
        prefs = [r for r in rows if r.memory_type == "user_preference"]
        facts = [r for r in rows if r.memory_type == "session_fact"]
        assert len(prefs) == settings.memory_preference_top_k
        assert len(facts) == settings.memory_fact_pool_size

    run(scenario())


def test_ttl_write_refresh_filter_sweep(memory_user):
    async def scenario():
        sid = f"sess_{uuid.uuid4().hex[:8]}"
        await ensure_session(sid, get_current_user())
        row = await memory_service.save_memory(
            "temp fact question left open",
            "session_fact",
            session_id=sid,
            importance=0.9,
            ttl_minutes=60,
        )
        assert row.ttl_minutes == 60
        assert row.expires_at is not None
        assert row.expires_at > datetime.utcnow()

        hit = await memory_service.retrieve_memories(
            query="open", session_id=sid, file_id=None,
        )
        assert any(r.id == row.id for r in hit)

        async with AsyncSessionLocal() as session:

            obj = await session.get(AgentMemory, row.id)
            assert obj is not None
            obj.expires_at = datetime.utcnow() - timedelta(hours=1)
            await session.commit()

        miss = await memory_service.retrieve_memories(
            query="open", session_id=sid, file_id=None,
        )
        assert all(r.id != row.id for r in miss)
        deleted = await memory_service.sweep_expired_memories(batch=100)
        assert deleted >= 1

    run(scenario())


def test_summary_token_trigger(memory_user, monkeypatch):
    monkeypatch.setattr(settings, "memory_summary_trigger_messages",  10**9)
    monkeypatch.setattr(settings, "memory_summary_trigger_chars",  10**9)
    run(scenario_tokens())


async def scenario_tokens():
    sid = f"sess_{uuid.uuid4().hex[:8]}"
    await ensure_session(sid, get_current_user())
    await seed_messages(sid, count=2, chars_per_msg=1000)
    first = await memory_service.maintain_conversation_memory(
        session_id=sid, file_id=None, user_text="hi", assistant_text="ok",
    )
    assert first["summary_updated"] is False

    await seed_messages(sid, count=5, chars_per_msg=5000)
    second = await memory_service.maintain_conversation_memory(
        session_id=sid, file_id=None, user_text="hi", assistant_text="ok",
    )
    assert second["summary_updated"] is True


def test_no_new_summary_is_not_persisted(memory_user, monkeypatch):
    """"无新增"不能落库覆盖真实历史摘要。"""
    monkeypatch.setattr(memory_service, "_extract_memories", _REAL_EXTRACT)
    monkeypatch.setattr(memory_service, "_generate_summary", _REAL_SUMMARY)

    async def fake_llm_text(prompt, max_tokens):
        if "压缩为简洁" in prompt:
            return "无新增"
        return "[]"

    monkeypatch.setattr(memory_service, "_llm_text", fake_llm_text)

    async def scenario():
        sid = f"sess_{uuid.uuid4().hex[:8]}"
        await ensure_session(sid, get_current_user())
        await seed_messages(sid, count=5, chars_per_msg=5000)
        result = await memory_service.maintain_conversation_memory(
            session_id=sid, file_id=None, user_text="hi", assistant_text="ok",
        )
        assert result["summary_updated"] is False
        latest = await memory_service.get_latest_summary(sid)
        assert latest is None

    run(scenario())


def test_extract_memories_requires_importance(memory_user, monkeypatch):
    """缺 importance 的条目低于阈值会被丢弃，达标条目正常落库。"""
    monkeypatch.setattr(memory_service, "_extract_memories", _REAL_EXTRACT)
    monkeypatch.setattr(memory_service, "_generate_summary", _REAL_SUMMARY)
    payload = [
        {"content": "用户偏好简短回答", "memory_type": "user_preference"},
        {"content": "宝玉挨打后黛玉前去探望", "memory_type": "novel_fact", "importance": 0.9},
        {"content": "今天天气不错", "memory_type": "session_fact", "importance": 0.2},
    ]

    async def fake_llm_text(prompt, max_tokens):
        return json.dumps(payload, ensure_ascii=False)

    monkeypatch.setattr(memory_service, "_llm_text", fake_llm_text)

    async def scenario():
        sid = f"sess_{uuid.uuid4().hex[:8]}"
        await ensure_session(sid, get_current_user())
        result = await memory_service.maintain_conversation_memory(
            session_id=sid, file_id="file_x", user_text="hi", assistant_text="answer",
        )
        added = result["memories_added"]
        assert [m["content"] for m in added] == ["宝玉挨打后黛玉前去探望"]

    run(scenario())


def test_extract_memories_logs_parse_failure(memory_user, monkeypatch):
    """LLM 输出无法解析时返回空但不能完全静默。"""
    monkeypatch.setattr(memory_service, "_extract_memories", _REAL_EXTRACT)
    warnings: list[str] = []

    class FakeLog:
        def warning(self, event, **kwargs):
            warnings.append(event)

        def info(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

    monkeypatch.setattr(memory_service, "log", FakeLog())

    async def fake_llm_text(prompt, max_tokens):
        return "模型输出了一段没有 JSON 的文字"

    monkeypatch.setattr(memory_service, "_llm_text", fake_llm_text)

    async def scenario():
        memories = await memory_service._extract_memories("hi", "answer", None)
        assert memories == []
        assert "memory_extract.parse_failed" in warnings

    run(scenario())


def test_extract_applies_update_and_delete(memory_user, monkeypatch):
    """修正与废弃必须落到旧记忆上，而不是追加矛盾条目。"""
    monkeypatch.setattr(memory_service, "_extract_memories", _REAL_EXTRACT)
    monkeypatch.setattr(memory_service, "_generate_summary", _REAL_SUMMARY)

    async def scenario():
        sid = f"sess_{uuid.uuid4().hex[:8]}"
        await ensure_session(sid, get_current_user())
        old_pref = await memory_service.save_memory(
            "用户要求结尾加汪汪汪", "user_preference", importance=1.0,
        )
        old_fact = await memory_service.save_memory(
            "用户正在关注红楼梦", "session_fact", session_id=sid, importance=0.9,
        )

        async def fake_llm_text(prompt, max_tokens):
            assert old_pref.id in prompt and old_fact.id in prompt
            return json.dumps([
                {"op": "update", "id": old_pref.id, "content": "用户要求结尾加喵喵喵", "importance": 1.0},
                {"op": "delete", "id": old_fact.id},
                {"op": "add", "content": "用户喜欢红楼梦人物分析", "memory_type": "session_fact", "importance": 0.8},
            ], ensure_ascii=False)

        monkeypatch.setattr(memory_service, "_llm_text", fake_llm_text)
        result = await memory_service.maintain_conversation_memory(
            session_id=sid, file_id=None,
            user_text="把加汪汪汪改成喵喵喵", assistant_text="好的",
        )
        added = result["memories_added"]
        assert any(m["id"] == old_pref.id and "喵喵喵" in m["content"] for m in added)
        assert all(m["id"] != old_fact.id for m in added)
        assert any(m["content"] == "用户喜欢红楼梦人物分析" for m in added)

        async with AsyncSessionLocal() as session:
            gone = await session.get(AgentMemory, old_fact.id)
            kept = await session.get(AgentMemory, old_pref.id)
        assert gone is None
        assert kept is not None and "喵喵喵" in kept.content

    run(scenario())


def test_extract_rejects_unknown_ids(memory_user, monkeypatch):
    """update/delete 只能作用于召回集合内的 id，幻觉 id 被拒绝不执行。"""
    monkeypatch.setattr(memory_service, "_extract_memories", _REAL_EXTRACT)

    async def scenario():
        row = await memory_service.save_memory(
            "用户要求结尾加汪汪汪", "user_preference", importance=1.0,
        )

        async def fake_llm_text(prompt, max_tokens):
            return json.dumps([
                {"op": "delete", "id": "nonexistent_id"},
                {"op": "update", "id": "another_fake", "content": "x", "importance": 1.0},
                {"op": "add", "content": "用户要求结尾只加喵喵喵", "memory_type": "user_preference", "importance": 0.9},
            ], ensure_ascii=False)

        monkeypatch.setattr(memory_service, "_llm_text", fake_llm_text)
        ops = await memory_service._extract_memories("改成喵喵喵", "好的", None, [row])
        assert ops == [
            {"op": "add", "content": "用户要求结尾只加喵喵喵", "memory_type": "user_preference",
             "importance": 0.9, "preference_key": None},
        ]

    run(scenario())


def test_safe_wrapper_forwards_skip_extract(memory_user, monkeypatch):
    """回归：chat.py 调 _safe 时传 skip_extract，包裹层必须接受并转发给内层。

    曾经 _safe 不接受该参数，游离任务每轮抛 TypeError 且无人取回，
    记忆提取与摘要在生产路径整体失效，而测试只覆盖内层函数未能发现。
    """
    calls: list[dict] = []

    async def spy(**kwargs):
        calls.append(kwargs)
        return {"summary_updated": False, "memories_added": []}

    monkeypatch.setattr(memory_service, "maintain_conversation_memory", spy)

    async def scenario():
        await memory_service.maintain_conversation_memory_safe(
            session_id="sess_x", file_id=None,
            user_text="hi", assistant_text="ok",
            assistant_message_id=None, skip_extract=True,
        )
        assert len(calls) == 1
        assert calls[0]["skip_extract"] is True

    run(scenario())


def test_safe_wrapper_swallows_inner_failure(memory_user, monkeypatch):
    """内层异常只记日志不外抛——游离任务里外抛等于无声丢失。"""
    async def boom(**kwargs):
        raise RuntimeError("inner failure")

    monkeypatch.setattr(memory_service, "maintain_conversation_memory", boom)

    async def scenario():
        # 不抛即为通过；异常被吞并落 log.warning。
        await memory_service.maintain_conversation_memory_safe(
            session_id="sess_x", file_id=None,
            user_text="hi", assistant_text="ok",
            assistant_message_id=None,
        )

    run(scenario())


def test_summary_folding_covers_all_pending(memory_user, monkeypatch):
    """回归：pending 超过单窗口 30 条时必须顺序折叠覆盖全部，不能只摘要
    最后 30 条却把 covered_message_id 推进到最后——中间消息曾永久丢出摘要链。"""
    monkeypatch.setattr(settings, "memory_summary_trigger_messages", 1)
    monkeypatch.setattr(settings, "memory_summary_trigger_chars", 1)
    windows: list[list[str]] = []

    async def fake_summary(previous, messages):
        windows.append([m.content for m in messages])
        return f"fold#{len(windows)}"

    monkeypatch.setattr(memory_service, "_generate_summary", fake_summary)

    async def scenario():
        sid = f"sess_{uuid.uuid4().hex[:8]}"
        await ensure_session(sid, get_current_user())
        await seed_messages(sid, count=35, chars_per_msg=100)

        result = await memory_service.maintain_conversation_memory(
            session_id=sid, file_id=None, user_text="hi", assistant_text="ok",
        )
        assert result["summary_updated"] is True
        assert len(windows) == 2  # 35 条按 30 条一窗折叠两次
        assert len(windows[0]) == 30 and len(windows[1]) == 5
        assert windows[0][0].endswith("#0") and windows[1][-1].endswith("#34")

        latest = await memory_service.get_latest_summary(sid)
        assert latest is not None
        assert latest.summary == "fold#2"
        # 覆盖水位必须推进到最后一条消息
        async with AsyncSessionLocal() as session:
            last_id = max(
                row.id for row in (await session.execute(
                    select(ChatMessage).where(ChatMessage.session_id == sid)
                )).scalars()
            )
        assert latest.covered_message_id == last_id

    run(scenario())


def test_preference_key_upsert(memory_user, monkeypatch):
    """回归：偏好带 preference_key 时同一键更新原条而不是无限累积新行。"""
    monkeypatch.setattr(memory_service, "_extract_memories", _REAL_EXTRACT)
    monkeypatch.setattr(memory_service, "_generate_summary", _REAL_SUMMARY)
    payload_round1 = [
        {"op": "add", "content": "回答要引用原文", "memory_type": "user_preference",
         "importance": 0.9, "preference_key": "引用原文"},
    ]
    payload_round2 = [
        {"op": "add", "content": "回答必须引用原文并给章节", "memory_type": "user_preference",
         "importance": 0.9, "preference_key": "引用原文"},
    ]
    rounds = iter([payload_round1, payload_round2])

    async def fake_llm_text(prompt, max_tokens):
        return json.dumps(next(rounds), ensure_ascii=False)

    monkeypatch.setattr(memory_service, "_llm_text", fake_llm_text)

    async def scenario():
        sid = f"sess_{uuid.uuid4().hex[:8]}"
        await ensure_session(sid, get_current_user())
        await memory_service.maintain_conversation_memory(
            session_id=sid, file_id="file_x", user_text="要引用原文", assistant_text="好",
        )
        await memory_service.maintain_conversation_memory(
            session_id=sid, file_id="file_x", user_text="要引用原文", assistant_text="好",
        )
        async with AsyncSessionLocal() as session:
            rows = (await session.execute(
                select(AgentMemory).where(
                    AgentMemory.user_id == get_current_user(),
                    AgentMemory.memory_type == "user_preference",
                    AgentMemory.preference_key == "引用原文",
                )
            )).scalars().all()
        assert len(rows) == 1
        assert "章节" in rows[0].content
        assert rows[0].memory_version == 2

    run(scenario())
