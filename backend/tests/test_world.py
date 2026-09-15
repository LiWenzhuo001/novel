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


def _json_loads(data: str):
    import json
    return json.loads(data)


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


_ROLEPLAY_CARDS = {
    "cards": [{
        "name": "宝玉", "persona": "温润多情的公子",
        "style": "口齿伶俐，称人妹妹", "background": "大观园中与众姊妹起居", "greeting": "妹妹来了",
    }],
    "scenario": "你走进怡红院，宝玉正等着你。",
}


def _mock_roleplay_graph(monkeypatch):
    """把统一图的角色扮演链路打到假实现：角色卡免 LLM，模型两轮后直接给出角色回复。

    轮 1：模型调用 load_character_context（真实 registry 通道 → 假角色卡）；
    轮 2：模型不再调用工具，最终内容即候选答案（单一生成语义）。
    """
    import json as _json

    from app.agent import runtime as runtime_module
    from app.core.llm import ModelTurnDelta
    from app.services import world_service as world_module

    async def fake_cards(file_id, names, chapter_until):
        assert isinstance(names, list) and 1 <= len(names) <= 3
        return {**_ROLEPLAY_CARDS, "cards": [
            {**card, "name": names[0]} for card in _ROLEPLAY_CARDS["cards"]
        ]}

    monkeypatch.setattr(world_module, "get_character_cards", fake_cards)

    rounds = {"n": 0}

    async def fake_stream(messages, purpose, *, tools=None, max_tokens=None):
        rounds["n"] += 1
        if rounds["n"] == 1:
            yield ModelTurnDelta(kind="tool_call", tool_call_chunk={
                "index": 0, "id": "rp1", "name": "load_character_context", "args_str": "{}",
            })
        else:
            yield ModelTurnDelta(kind="content", text="**宝玉**：妹妹今日气色不错。")

    monkeypatch.setattr(runtime_module, "astream_model_turn", fake_stream)
    return rounds


def test_roleplay_runs_in_unified_graph(client, auth_headers, chat_env, monkeypatch):
    """角色扮演并入统一图：interaction_mode=roleplay 走同一 SSE 协议，角色回复即答案。"""
    _mock_roleplay_graph(monkeypatch)
    response = client.post("/api/chat", json={
        "message": "你好",
        "interaction_mode": "roleplay",
        "strategy": "auto",
        "memory_mode": "off",
        "file_id": chat_env["file_id"],
        "personas": ["宝玉"],
    }, headers=auth_headers)
    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    types = _event_types(events)
    # 统一协议：run_started/route/meta/done 都在，角色回复以 token 下发
    for expected in ("session", "run_started", "route", "token", "meta", "done"):
        assert expected in types, f"缺少 {expected} 事件，实际：{types}"
    token_text = "".join(data for name, data in events if name == "token")
    assert "宝玉" in token_text
    meta = _json_loads(next(data for name, data in events if name == "meta"))
    assert meta["interaction_mode"] == "roleplay"
    assert meta["effective_strategy"] == "react"
    assert meta["answer_mode"] == "roleplay"
    assert meta["personas"] == ["宝玉"]


def test_roleplay_session_persistence_and_chapter_boundary(client, auth_headers, chat_env, monkeypatch):
    """roleplay 请求会话持久化 personas 与 chapter_until；meta 携带时间边界。"""
    _mock_roleplay_graph(monkeypatch)
    response = client.post("/api/chat", json={
        "message": "你好",
        "interaction_mode": "roleplay",
        "memory_mode": "off",
        "file_id": chat_env["file_id"],
        "personas": ["林黛玉"],
        "chapter_until": 12,
    }, headers=auth_headers)
    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    meta = _json_loads(next(data for name, data in events if name == "meta"))
    assert meta["chapter_until"] == 12
    assert meta["personas"] == ["林黛玉"]

    session_id = next(data for name, data in events if name == "session")
    rows = client.get("/api/chat/sessions", headers=auth_headers).json()["data"]
    row = next(r for r in rows if r["id"] == session_id)
    assert row["personas"] == ["林黛玉"]
    assert row["chapter_until"] == 12


def test_roleplay_legacy_strategy_value_maps_to_interaction_mode(client, auth_headers, chat_env, monkeypatch):
    """过渡兼容：strategy=roleplay 自动映射为 interaction_mode=roleplay，并在 meta.deprecations 提示。"""
    _mock_roleplay_graph(monkeypatch)
    response = client.post("/api/chat", json={
        "message": "你好",
        "strategy": "roleplay",
        "memory_mode": "off",
        "file_id": chat_env["file_id"],
        "personas": ["宝玉"],
    }, headers=auth_headers)
    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    meta = _json_loads(next(data for name, data in events if name == "meta"))
    assert meta["interaction_mode"] == "roleplay"
    assert any("roleplay" in d for d in meta.get("deprecations", []))


def test_roleplay_rejects_more_than_three(client, auth_headers, chat_env):
    response = client.post("/api/chat", json={
        "message": "你好",
        "interaction_mode": "roleplay",
        "memory_mode": "off",
        "file_id": chat_env["file_id"],
        "personas": ["甲", "乙", "丙", "丁"],
    }, headers=auth_headers)
    assert response.status_code == 422


def test_roleplay_qa_mode_rejects_personas(client, auth_headers, chat_env):
    """qa 模式带 personas 直接 422：交互模式与人物集合互斥。"""
    response = client.post("/api/chat", json={
        "message": "你好",
        "strategy": "auto",
        "memory_mode": "off",
        "file_id": chat_env["file_id"],
        "personas": ["宝玉"],
    }, headers=auth_headers)
    assert response.status_code == 422


def test_history_lines_accepts_dicts_and_lc_messages():
    """第二轮对话的历史是 LangChain 消息对象，第一轮是 dict——两种都要能拼。"""
    from langchain_core.messages import AIMessage, HumanMessage

    from app.services.world_service import format_roleplay_history

    dict_form = format_roleplay_history([
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "来了"},
    ])
    lc_form = format_roleplay_history([HumanMessage(content="你好"), AIMessage(content="来了")])
    assert dict_form == lc_form
    assert "访客：你好" in dict_form and "角色：来了" in dict_form


def test_roleplay_accepts_lc_message_history(client, auth_headers, chat_env, monkeypatch):
    """第二轮带 LangChain 消息对象历史时统一图正常作答（历史经 roleplay_history 注入）。"""
    _mock_roleplay_graph(monkeypatch)
    response = client.post("/api/chat", json={
        "message": "心情怎么样",
        "interaction_mode": "roleplay",
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
