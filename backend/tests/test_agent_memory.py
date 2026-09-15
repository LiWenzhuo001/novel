"""模型自主记忆（memory_agent）：工具注册、节点循环与后台提取联动。"""
from __future__ import annotations

import asyncio
import uuid

import pytest
from langchain_core.messages import AIMessage

from app.agent import runtime
from app.agent.tools import MEMORY_AGENT_TOOLS, registry
from app.core.context import (
    get_memory_session,
    set_current_user,
    set_memory_session,
    reset_current_user,
    reset_memory_session,
)
from app.services import memory_service

LOOP = asyncio.new_event_loop()


def run(coro):
    return LOOP.run_until_complete(coro)


async def _purge_memory_jobs(user_id: str) -> None:
    """清掉全部残留记忆任务：领取是全局 FIFO，跨用户残留会让断言错位。

    用"创建时间早于未来 1 小时"的参数化条件覆盖全表，避免无 WHERE 的全表删除。
    """
    from datetime import datetime, timedelta
    from sqlalchemy import text as sql_text
    from app.db import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        await session.execute(
            sql_text("DELETE FROM memory_jobs WHERE created_at < :cutoff").bindparams(
                cutoff=datetime.utcnow() + timedelta(hours=1),
            )
        )
        await session.commit()


@pytest.fixture
def no_embed(monkeypatch):
    """记忆写入不调用真实 embedding API。"""
    async def fake_embed(content):
        return None

    monkeypatch.setattr(memory_service, "_embed_text", fake_embed)


def test_memory_tools_registered():
    """四个记忆工具挂进 registry，白名单与 schema 名一致。"""
    names = [spec.name for spec in registry.specs()]
    for tool in MEMORY_AGENT_TOOLS:
        assert tool in names, f"{tool} 未注册进 registry"


def test_add_update_delete_memory_tools_roundtrip(no_embed):
    """写工具入队 → worker 消费 → 落库全链路（真实库，随机会话）。

    回答路径零直接写库：add/update/delete 立即返回 deferred，记忆效果由
    worker 消费 memory_jobs 后生效；search 直查可见已落库记忆。
    """
    from app.services import memory_worker
    from app.db.models import ChatSession

    session_id = f"sess_{uuid.uuid4().hex[:8]}"
    user_id = f"memtool_{uuid.uuid4().hex[:8]}"
    user_token = set_current_user(user_id)

    async def ensure_session():
        # memory_jobs.session_id 有 FK：先落一行真实会话（生产路径会话必然已存在）。
        # 顺带清掉本用户历史残留任务（FIFO 污染防护）。
        async with memory_service.AsyncSessionLocal() as session:
            session.add(ChatSession(id=session_id, user_id=user_id, domain="novel"))
            await session.commit()
        await _purge_memory_jobs(user_id)

    run(ensure_session())
    token = set_memory_session(session_id, None)
    try:
        async def scenario():
            # add：立即 ok + deferred，先不落库
            add_result = await registry.execute(
                "add_memory", allowed_tools=MEMORY_AGENT_TOOLS,
                content="用户喜欢简短回答", memory_type="user_preference", importance=0.9,
            )
            assert add_result.status == "ok" and add_result.output.get("deferred") is True
            search_result = await registry.execute(
                "search_memories", allowed_tools=MEMORY_AGENT_TOOLS, query="简短回答",
            )
            assert "没有找到相关记忆" in search_result.output["message"]

            # worker 消费后记忆落库，search 可见
            job = await memory_worker._claim_due_job()
            assert job is not None and job.kind == "memory_op"
            await memory_worker._run_job(job)
            search_result = await registry.execute(
                "search_memories", allowed_tools=MEMORY_AGENT_TOOLS, query="简短回答",
            )
            assert "用户喜欢简短回答" in search_result.output["message"]
            memory_id = search_result.output["memory_ids"][0]

            # update：归属校验通过后入队；未知 id 直接拒绝（同步反馈）
            bad_update = await registry.execute(
                "update_memory", allowed_tools=MEMORY_AGENT_TOOLS,
                memory_id="nonexistent_id", content="x",
            )
            assert bad_update.status == "error" and bad_update.error_code == "memory_not_found"
            update_result = await registry.execute(
                "update_memory", allowed_tools=MEMORY_AGENT_TOOLS,
                memory_id=memory_id, content="用户喜欢极简回答",
            )
            assert update_result.status == "ok" and update_result.output.get("deferred") is True
            job = await memory_worker._claim_due_job()
            await memory_worker._run_job(job)
            search_result = await registry.execute(
                "search_memories", allowed_tools=MEMORY_AGENT_TOOLS, query="极简回答",
            )
            assert "用户喜欢极简回答" in search_result.output["message"]

            # delete：入队 → worker 执行 → 记忆消失
            delete_result = await registry.execute(
                "delete_memory", allowed_tools=MEMORY_AGENT_TOOLS, memory_id=memory_id,
            )
            assert delete_result.status == "ok"
            job = await memory_worker._claim_due_job()
            await memory_worker._run_job(job)
            search_result = await registry.execute(
                "search_memories", allowed_tools=MEMORY_AGENT_TOOLS, query="极简回答",
            )
            assert "没有找到相关记忆" in search_result.output["message"]

        run(scenario())
    finally:
        reset_memory_session(token)
        reset_current_user(user_token)


def test_memory_tools_reject_without_session_context():
    """无会话上下文时写操作直接拒绝（不落库）。"""
    reset_token = None
    if get_memory_session() is not None:
        reset_token = None
    token = set_memory_session  # noqa: F841  确保无上下文时不注入

    async def scenario():
        result = await registry.execute(
            "add_memory", allowed_tools=MEMORY_AGENT_TOOLS,
            content="测试", memory_type="session_fact", importance=0.9,
        )
        assert result.status == "error"
        assert "不可用" in result.output

    run(scenario())


def test_memory_agent_node_executes_model_tool_calls(monkeypatch):
    """memory_agent 节点：模型发起 add_memory → registry 入队 → ToolMessage 回喂。"""
    from app.db.models import ChatSession

    session_id = f"sess_{uuid.uuid4().hex[:8]}"

    async def ensure_session():
        # memory_jobs.session_id 有 FK：先落一行真实会话。
        async with memory_service.AsyncSessionLocal() as session:
            session.add(ChatSession(id=session_id, user_id="memagent_test", domain="novel"))
            await session.commit()

    run(ensure_session())
    state = {
        "session_id": session_id,
        "file_id": None,
        "original_query": "记住我喜欢简短回答",
        "standalone_query": "记住我喜欢简短回答",
        "synthesis_context": {"memories": []},
        "current_step": 5,
        "memory_agent_active": True,
    }

    class FakeLLM:
        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages):
            nonlocal rounds
            rounds += 1
            if rounds == 1:
                return AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "add_memory",
                        "args": {"content": "用户喜欢简短回答", "memory_type": "user_preference", "importance": 0.9},
                        "id": "call_1",
                    }],
                )
            return AIMessage(content="已记录用户偏好。")

    rounds = 0
    monkeypatch.setattr(runtime, "get_llm", lambda **kwargs: FakeLLM())

    async def scenario():
        result = await runtime._memory_agent_node(state)
        assert rounds >= 2
        assert len(result["memory_ops"]) == 1
        assert result["memory_ops"][0]["tool"] == "add_memory"
        assert result["memory_ops"][0]["status"] == "ok"
        return result["memory_ops"][0]

    op = run(scenario())

    # 写操作不再直接落库：入库凭证是一条高优先级 memory_jobs 任务。
    async def verify_job():
        from sqlalchemy import select
        from app.db.models import MemoryJob
        async with memory_service.AsyncSessionLocal() as session:
            rows = (await session.execute(
                select(MemoryJob).where(MemoryJob.kind == "memory_op")
            )).scalars().all()
        return rows

    jobs = run(verify_job())
    assert any("简短回答" in (job.payload or "") for job in jobs)
    assert op["summary"] == "已提交记忆写入（后台生效）"


def test_maintain_skips_extract_when_model_operated(monkeypatch):
    """模型做过记忆操作时，后台提取跳过（摘要照常）。"""
    calls = []

    from app.core.context import set_current_user, reset_current_user
    token = set_current_user(f"memagent_{uuid.uuid4().hex[:8]}")

    async def fake_extract(user_text, assistant_text, file_id, existing=None):
        calls.append(1)
        return []

    monkeypatch.setattr(memory_service, "_extract_memories", fake_extract)

    async def scenario():
        result = await memory_service.maintain_conversation_memory(
            session_id=f"sess_{uuid.uuid4().hex[:8]}",
            file_id=None,
            user_text="hi",
            assistant_text="ok",
            skip_extract=True,
        )
        assert result["summary_updated"] is False
        assert not calls

        result2 = await memory_service.maintain_conversation_memory(
            session_id=f"sess_{uuid.uuid4().hex[:8]}",
            file_id=None,
            user_text="hi",
            assistant_text="ok",
        )
        assert result2["summary_updated"] is False
        assert len(calls) == 1

    run(scenario())
    reset_current_user(token)
