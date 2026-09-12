"""「进入小说世界」：章节截断检索、人物名册、角色卡与 roleplay 流。

异步用例使用模块级统一事件循环（与 test_memory_service 同模式）：
全局 asyncpg 连接池不能跨多个已关闭的循环复用。
"""
from __future__ import annotations

import asyncio
import json
import re
import uuid

import pytest

from app.services import world_service

LOOP = asyncio.new_event_loop()


def run(coro):
    return LOOP.run_until_complete(coro)


# ===== TestClient 夹具（沿用已移出仓库的 conftest / test_chat_api 原模式）=====

@pytest.fixture(scope="session")
def client():
    """TestClient 保持 session 级，避免 asyncpg 全局连接池跨多个已关闭事件循环复用。"""
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth_headers(client):
    username = f"test_{uuid.uuid4().hex[:8]}"
    password = f"pw_{uuid.uuid4().hex[:8]}"
    r = client.post("/api/auth/register", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    token = r.json()["data"]["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _fake_rewrite_result(message: str) -> "RewriteResult":
    from app.core.query_rewriter import RewriteResult

    return RewriteResult(
        original=message,
        standalone_query=message,
        retrieval_query=message,
        applied=False,
        reason="test_stub",
        intent="other",
        entities=[],
        evidence_focus=[],
        confidence=0.0,
        needs_retrieval=True,
        answer_mode="novel_evidence",
        retrieval_reason="test_lookup",
        output_policy={},
        preference_update=None,
    )


@pytest.fixture
def chat_env(monkeypatch, client, auth_headers):
    """打桩 query 改写与嵌入；为认证用户种一本可对话的空索引书，返回 file_id。

    种子/清理用同步引擎：asyncpg 连接池绑定 TestClient 的事件循环，
    在夹具里另起 asyncio.run() 复用池会让 pytest 退出时挂死。"""
    from sqlalchemy import create_engine

    from app.api import chat as chat_module
    from app.config import settings
    from app.db.models import KnowledgeFile

    me = client.get("/api/auth/me", headers=auth_headers).json()["data"]
    user_id = me["id"]
    file_id = uuid.uuid4().hex[:12]

    engine = create_engine(settings.database_url)
    with engine.begin() as conn:
        conn.execute(KnowledgeFile.__table__.insert().values(
            id=file_id, filename="测试小说.txt", filetype="txt", size=100,
            chunks=0, status="indexed", user_id=user_id, domain="novel",
        ))

    async def fake_rewrite(query, history, memory_context=None, **kwargs):
        return _fake_rewrite_result(query)

    monkeypatch.setattr(chat_module, "rewrite_query", fake_rewrite)



    # 假嵌入：让检索在"无向量书"上确定性走空结果（缺桩时真实 embedding
    # API 会拖慢用例并留下环境依赖）。
    from app.core import rag as rag_module

    class _FakeEmbeddings:
        def embed_query(self, query):
            return [0.01] * 1024

    monkeypatch.setattr(rag_module, "get_embeddings", lambda: _FakeEmbeddings())

    try:
        yield {"file_id": file_id, "user_id": user_id}
    finally:
        with engine.begin() as conn:
            conn.execute(KnowledgeFile.__table__.delete().where(
                KnowledgeFile.__table__.c.id == file_id
            ))
        engine.dispose()


def _parse_sse_events(raw: str) -> list[tuple[str, str]]:
    events: list[tuple[str, str]] = []
    for block in re.split(r"\r?\n\r?\n", raw):
        event, data_lines = "", []
        for line in block.replace("\r", "").splitlines():
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())
        if event:
            events.append((event, "\n".join(data_lines)))
    return events


def _event_types(events) -> list[str]:
    return [name for name, _data in events]


class _Doc:
    """轻量 Document 替身：只带 metadata 与 page_content。"""

    def __init__(self, chapter_no, content="片段"):
        self.metadata = {"chapter_no": chapter_no}
        self.page_content = content


def test_chapter_within_filters():
    from app.core.rag import _chapter_within

    assert _chapter_within(_Doc(3), 5) is True
    assert _chapter_within(_Doc(9), 5) is False
    # 未分章片段无法判定归属，保守保留。
    assert _chapter_within(_Doc(None), 5) is True


def test_retrieve_novel_context_truncates_chapters(monkeypatch):
    """chapter_until 透传到底层检索，并对最终结果兜底过滤。"""
    from app.core import rag

    captured = {}

    async def fake_similarity_search(query, k=None, domain="novel", file_id=None,
                                     retrieval_query=None, chapter_until=None):
        captured["chapter_until"] = chapter_until
        return [_Doc(3), _Doc(9)]

    async def fake_expand(primary, neighbor_window, user_id):
        # 模拟邻居扩展重新引入越章内容：兜底过滤必须拦下。
        return list(primary) + [_Doc(12)]

    monkeypatch.setattr(rag, "similarity_search", fake_similarity_search)
    monkeypatch.setattr(rag, "expand_novel_context", fake_expand)
    monkeypatch.setattr(rag.settings, "enable_chapter_local_retrieval", False)

    docs = run(rag.retrieve_novel_context("问", file_id="f", chapter_until=5))

    assert captured["chapter_until"] == 5
    assert [d.metadata["chapter_no"] for d in docs] == [3]


def _patch_world_llm(monkeypatch, payload):
    async def fake_llm_json(prompt, system, max_tokens):
        return payload

    monkeypatch.setattr(world_service, "_llm_json", fake_llm_json)


def test_roster_extraction_dedupes(monkeypatch):
    """名册提取：去重、剔除空名；file_id 随机避免命中历史缓存。"""
    from app.core.context import set_current_user, reset_current_user
    token = set_current_user(f"world_{uuid.uuid4().hex[:8]}")
    file_id = f"file_{uuid.uuid4().hex[:10]}"
    try:
        async def fake_visible(fid):
            return True

        async def fake_hash(fid):
            return "hash1"

        async def fake_retrieve(query, k=None, neighbor_window=None, file_id=None,
                                retrieval_query=None, user_id=None, chapter_until=None):
            return [_Doc(1, "宝玉初见黛玉")]

        monkeypatch.setattr(world_service, "_visible_file", fake_visible)
        monkeypatch.setattr(world_service, "_file_source_hash", fake_hash)
        monkeypatch.setattr(world_service, "retrieve_novel_context", fake_retrieve)
        _patch_world_llm(monkeypatch, ["贾宝玉", "林黛玉", "贾宝玉", "", "薛宝钗"])

        async def scenario():
            data = await world_service.list_characters(file_id, None)
            assert data["characters"] == ["贾宝玉", "林黛玉", "薛宝钗"]
            assert data["cached"] is False
            # 二次调用命中缓存。
            data2 = await world_service.list_characters(file_id, None)
            assert data2["cached"] is True

        run(scenario())
    finally:
        reset_current_user(token)


def test_roster_failure_raises(monkeypatch):
    from app.core.context import set_current_user, reset_current_user
    token = set_current_user(f"world_{uuid.uuid4().hex[:8]}")
    file_id = f"file_{uuid.uuid4().hex[:10]}"
    try:
        async def fake_visible(fid):
            return True

        async def fake_hash(fid):
            return None

        async def fake_retrieve(query, k=None, neighbor_window=None, file_id=None,
                                retrieval_query=None, user_id=None, chapter_until=None):
            return []

        monkeypatch.setattr(world_service, "_visible_file", fake_visible)
        monkeypatch.setattr(world_service, "_file_source_hash", fake_hash)
        monkeypatch.setattr(world_service, "retrieve_novel_context", fake_retrieve)
        _patch_world_llm(monkeypatch, [])

        async def scenario():
            with pytest.raises(ValueError):
                await world_service.list_characters(file_id, None)

        run(scenario())
    finally:
        reset_current_user(token)


def test_character_card_generated_then_cached(monkeypatch):
    """角色卡+开场情景生成后落库，第二次直接命中缓存（LLM 坏掉也不影响）。"""
    from app.core.context import set_current_user, reset_current_user
    token = set_current_user(f"world_{uuid.uuid4().hex[:8]}")
    file_id = f"file_{uuid.uuid4().hex[:10]}"
    try:
        async def fake_visible(fid):
            return True

        async def fake_hash(fid):
            return "hash1"

        async def fake_retrieve(query, k=None, neighbor_window=None, file_id=None,
                                retrieval_query=None, user_id=None, chapter_until=None):
            assert chapter_until == 5
            return [_Doc(2, "黛玉葬花")]

        monkeypatch.setattr(world_service, "_visible_file", fake_visible)
        monkeypatch.setattr(world_service, "_file_source_hash", fake_hash)
        monkeypatch.setattr(world_service, "retrieve_novel_context", fake_retrieve)

        async def fake_llm_json(prompt, system, max_tokens):
            if "开场情景" in prompt:
                return {"scenario": "暮春，大观园。你沿着小径走来。"}
            return {
                "in_novel": True, "persona": "多愁善感", "style": "敏感细腻带机锋",
                "background": "寄居贾府；葬花", "greeting": "你来了。",
            }

        monkeypatch.setattr(world_service, "_llm_json", fake_llm_json)

        async def scenario():
            result = await world_service.get_character_cards(file_id, ["林黛玉"], 5)
            assert result["cards"][0]["name"] == "林黛玉"
            assert result["cards"][0]["greeting"] == "你来了。"
            assert "大观园" in result["scenario"]

            # 缓存命中：LLM 坏掉也不影响。
            async def broken_llm(prompt, system, max_tokens):
                raise RuntimeError("LLM 挂了")

            monkeypatch.setattr(world_service, "_llm_json", broken_llm)
            result2 = await world_service.get_character_cards(file_id, ["林黛玉"], 5)
            assert result2["cards"][0]["persona"] == "多愁善感"
            assert "大观园" in result2["scenario"]

        run(scenario())
    finally:
        reset_current_user(token)


def test_character_rejected_when_not_in_novel(monkeypatch):
    """LLM 判定人物不存在（in_novel=false）时拒绝生成。"""
    from app.core.context import set_current_user, reset_current_user
    token = set_current_user(f"world_{uuid.uuid4().hex[:8]}")
    file_id = f"file_{uuid.uuid4().hex[:10]}"
    try:
        async def fake_visible(fid):
            return True

        async def fake_hash(fid):
            return None

        async def fake_retrieve(query, k=None, neighbor_window=None, file_id=None,
                                retrieval_query=None, user_id=None, chapter_until=None):
            return [_Doc(1, "与她无关的片段")]

        monkeypatch.setattr(world_service, "_visible_file", fake_visible)
        monkeypatch.setattr(world_service, "_file_source_hash", fake_hash)
        monkeypatch.setattr(world_service, "retrieve_novel_context", fake_retrieve)
        _patch_world_llm(monkeypatch, {"in_novel": False, "persona": ""})

        async def scenario():
            with pytest.raises(ValueError, match="未找到"):
                await world_service.get_character_cards(file_id, ["不存在的人"], None)

        run(scenario())
    finally:
        reset_current_user(token)


def test_character_rejected_when_no_fragments(monkeypatch):
    """检索宽松重试后仍零命中 → 拒绝。"""
    from app.core.context import set_current_user, reset_current_user
    token = set_current_user(f"world_{uuid.uuid4().hex[:8]}")
    file_id = f"file_{uuid.uuid4().hex[:10]}"
    calls = []

    try:
        async def fake_visible(fid):
            return True

        async def fake_hash(fid):
            return None

        async def fake_retrieve(query, k=None, neighbor_window=None, file_id=None,
                                retrieval_query=None, user_id=None, chapter_until=None):
            calls.append(query)
            return []

        monkeypatch.setattr(world_service, "_visible_file", fake_visible)
        monkeypatch.setattr(world_service, "_file_source_hash", fake_hash)
        monkeypatch.setattr(world_service, "retrieve_novel_context", fake_retrieve)

        async def scenario():
            with pytest.raises(ValueError, match="未找到"):
                await world_service.get_character_cards(file_id, ["无名氏"], None)
            assert len(calls) == 2  # 精确 + 宽松各一次

        run(scenario())
    finally:
        reset_current_user(token)


def test_roleplay_stream_has_no_retrieval_step(client, auth_headers, chat_env, monkeypatch):
    """对话阶段零检索：不再下发 tool_start/tool_end 事件。"""
    from app.api import chat as chat_module

    async def fake_stream(file_id, personas, chapter_until, message, history):
        assert isinstance(history, list)
        yield {"type": "token", "data": "嗯。"}
        yield {"type": "meta", "data": {"strategy": "roleplay", "personas": personas, "steps": 1}}

    monkeypatch.setattr(chat_module.world_service, "stream_roleplay", fake_stream)
    response = client.post("/api/chat", json={
        "message": "你好",
        "strategy": "roleplay",
        "memory_mode": "off",
        "file_id": chat_env["file_id"],
        "personas": ["宝玉"],
    }, headers=auth_headers)
    assert response.status_code == 200
    types = _event_types(_parse_sse_events(response.text))
    assert "token" in types
    assert "tool_start" not in types and "tool_end" not in types


def test_roleplay_stream_and_session_persistence(client, auth_headers, chat_env, monkeypatch):
    """roleplay 请求走专属管线，会话持久化 personas 与 chapter_until。"""
    from app.api import chat as chat_module

    async def fake_stream(file_id, personas, chapter_until, message, history):
        assert personas == ["林黛玉"]
        assert chapter_until == 12
        yield {"type": "tool_start", "data": {"id": "rp_retrieve", "tool": "retrieve_novel", "label": "人物记忆检索"}}
        for chunk in ["**林黛玉**：", "你来了。"]:
            yield {"type": "token", "data": chunk}
        yield {"type": "meta", "data": {"strategy": "roleplay", "personas": personas, "chapter_until": chapter_until, "steps": 2}}

    monkeypatch.setattr(chat_module.world_service, "stream_roleplay", fake_stream)

    response = client.post("/api/chat", json={
        "message": "你好",
        "strategy": "roleplay",
        "memory_mode": "off",
        "file_id": chat_env["file_id"],
        "personas": ["林黛玉"],
        "chapter_until": 12,
    }, headers=auth_headers)
    assert response.status_code == 200

    events = _parse_sse_events(response.text)
    types = _event_types(events)
    for expected in ("session", "tool_start", "token", "meta", "done"):
        assert expected in types, f"缺少 {expected} 事件，实际：{types}"
    token_text = "".join(data for name, data in events if name == "token")
    assert "林黛玉" in token_text

    session_id = next(data for name, data in events if name == "session")
    rows = client.get("/api/chat/sessions", headers=auth_headers).json()["data"]
    row = next(r for r in rows if r["id"] == session_id)
    assert row["personas"] == ["林黛玉"]
    assert row["chapter_until"] == 12


def test_roleplay_rejects_more_than_three(client, auth_headers, chat_env):
    response = client.post("/api/chat", json={
        "message": "你好",
        "strategy": "roleplay",
        "memory_mode": "off",
        "file_id": chat_env["file_id"],
        "personas": ["甲", "乙", "丙", "丁"],
    }, headers=auth_headers)
    assert response.status_code == 422


def test_history_lines_accepts_dicts_and_lc_messages():
    """第二轮对话的历史是 LangChain 消息对象，第一轮是 dict——两种都要能拼。"""
    from langchain_core.messages import AIMessage, HumanMessage

    from app.services.world_service import _history_lines

    dict_form = _history_lines([
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "来了"},
    ])
    lc_form = _history_lines([HumanMessage(content="你好"), AIMessage(content="来了")])
    assert dict_form == lc_form
    assert "访客：你好" in dict_form and "角色：来了" in dict_form


def test_roleplay_accepts_lc_message_history(client, auth_headers, chat_env, monkeypatch):
    """第二轮带 LangChain 消息对象历史时管线不再报 'no attribute get'。"""
    from langchain_core.messages import HumanMessage

    from app.api import chat as chat_module

    async def fake_stream(file_id, personas, chapter_until, message, history):
        # 生产形态：history 为 _to_lc_messages 的产物（首轮可能为空），
        # 兼容性本身由 test_history_lines_accepts_dicts_and_lc_messages 锁定。
        assert isinstance(history, list)
        yield {"type": "token", "data": "好"}
        yield {"type": "meta", "data": {"strategy": "roleplay", "personas": personas, "steps": 2}}

    monkeypatch.setattr(chat_module.world_service, "stream_roleplay", fake_stream)
    response = client.post("/api/chat", json={
        "message": "心情怎么样",
        "strategy": "roleplay",
        "memory_mode": "off",
        "file_id": chat_env["file_id"],
        "personas": ["宝玉"],
        "history": [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "来了"},
        ],
    }, headers=auth_headers)
    assert response.status_code == 200
    assert "token" in _event_types(_parse_sse_events(response.text))
