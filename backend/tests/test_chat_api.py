"""聊天 SSE 主链路与记忆 API 的集成测试（TestClient + 真实库 + mock 改写）。

覆盖此前零测试的 chat.py SSE 编排层：事件序列、回答持久化、记忆端点。
测试夹具为认证用户创建一本"已索引"但无向量的书——检索必然走空结果分支，
最终答案为固定空检索提示，全程不依赖任何外部 LLM/嵌入 API。
"""
from __future__ import annotations

import json
import uuid

import pytest

from app.core.query_rewriter import RewriteResult


def _fake_rewrite_result(message: str) -> RewriteResult:
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


def _parse_sse_events(raw: str) -> list[tuple[str, str]]:
    """sse_starlette 使用 \\r\\n 行尾，块间以空行分隔——按正则切分并剥掉 \\r。"""
    import re

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


@pytest.fixture
def chat_env(monkeypatch, client, auth_headers):
    """打桩 query 改写；为认证用户种一本可对话的空索引书，返回 file_id。

    种子/清理用同步引擎：asyncpg 连接池绑定 TestClient 的事件循环，
    在夹具里另起 asyncio.run() 复用池会让 pytest 退出时挂死。
    """
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

    # 假嵌入：让检索在"无向量书"上确定性走空结果（conftest 的 ST stub 会抛
    # TypeError，导致 fallback_reason 在不同环境下不一致）。
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


def _chat_payload(message: str, file_id: str, **overrides) -> dict:
    payload = {
        "message": message,
        "strategy": "direct",
        "memory_mode": "off",
        "file_id": file_id,
    }
    payload.update(overrides)
    return payload


def test_chat_sse_stream_persists_answer(client, auth_headers, chat_env):
    """完整 SSE 链路：事件序列完整、空检索提示持久化、meta 携带策略信息。"""
    response = client.post(
        "/api/chat",
        json=_chat_payload("林舟为什么要离开故乡？", chat_env["file_id"]),
        headers=auth_headers,
    )
    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    types = _event_types(events)

    # 事件序列：会话/路由/计划/来源/回答/元数据/结束
    for expected in ("session", "route", "plan", "tool_start", "tool_end", "token", "meta", "done"):
        assert expected in types, f"缺少 {expected} 事件，实际：{types}"

    meta_payload = next(json.loads(data) for name, data in events if name == "meta")
    assert meta_payload["strategy"] == "direct"
    assert meta_payload["answer_mode"] == "novel_evidence"
    # 无向量索引 → 空检索兜底提示
    assert meta_payload["fallback_reason"] == "empty_retrieval"

    token_text = "".join(data for name, data in events if name == "token")
    assert "没有检索到" in token_text

    # 回答已持久化：会话消息接口可取回
    session_id = next(data for name, data in events if name == "session")
    messages = client.get(
        f"/api/chat/sessions/{session_id}/messages", headers=auth_headers
    ).json()["data"]
    assert {m["role"] for m in messages} == {"user", "assistant"}
    assistant = next(m for m in messages if m["role"] == "assistant")
    assert "没有检索到" in assistant["content"]


def test_chat_memory_off_has_no_memory_context(client, auth_headers, chat_env):
    """memory_mode=off 时不注入记忆上下文事件。"""
    response = client.post(
        "/api/chat",
        json=_chat_payload("随便聊聊", chat_env["file_id"], memory_mode="off"),
        headers=auth_headers,
    )
    assert response.status_code == 200
    assert "memory_context" not in _event_types(_parse_sse_events(response.text))


def test_chat_requires_selected_file(client, auth_headers):
    """未选择小说（file_id 缺失）应被 400 拒绝。"""
    response = client.post(
        "/api/chat",
        json={"message": "问题", "strategy": "direct", "memory_mode": "off"},
        headers=auth_headers,
    )
    assert response.status_code == 400


def test_memory_list_endpoint(client, auth_headers):
    """新用户记忆列表为空但接口可用。"""
    response = client.get("/api/memories", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["data"] == []


def test_session_memory_context_endpoint(client, auth_headers):
    """会话记忆上下文接口返回默认结构（无摘要、无记忆）。"""
    created = client.post("/api/chat/sessions", headers=auth_headers).json()["data"]
    session_id = created["id"] if isinstance(created, dict) else created
    response = client.get(
        f"/api/chat/sessions/{session_id}/memory-context", headers=auth_headers
    )
    assert response.status_code == 200
    payload = response.json()["data"]
    assert "summary" in payload and "memories" in payload
