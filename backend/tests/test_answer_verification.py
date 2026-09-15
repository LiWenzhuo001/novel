"""最终答案验证链（compose → verify → repair → finalize）的专项测试（无 DB、无真实 LLM）。

确定性检查：[S#] 合法性、required 引用门槛、角色上下文存在性；
修复路径：纯坏引用零成本剥除；语义不可支持走一次带诊断的重新生成。
"""
from __future__ import annotations

import json

from app.agent import runtime
from app.agent.types import AgentState


def _sources(*ids):
    return [{"id": f"S{i}", "source": "红楼梦.txt", "chapter_no": i, "chunk_no": i,
             "char_start": i * 10, "char_end": i * 10 + 5} for i in ids]


def _state(**overrides) -> AgentState:
    state: AgentState = {
        "strategy": "react",
        "retrieval_policy": "optional",
        "answer_mode": "novel_evidence",
        "interaction_mode": "qa",
        "sources": _sources(1, 2),
        "candidate_answer": "宝玉初见黛玉 [S1]。",
        "candidate_streamed": True,
        "repairs_used": 0,
        "synthesis_context": {"evidence": [], "memories": []},
        "current_step": 3,
    }
    state.update(overrides)
    return state


def _events(state: AgentState) -> list[dict]:
    events: list[dict] = []

    class Queue:
        async def put(self, item):
            events.append(item)

    return {**state, "event_queue": Queue()} and events


# ---------- 确定性检查 ----------

def test_invalid_citation_detected():
    state = _state(candidate_answer="引用了不存在的编号 [S9]。")
    issues = runtime._deterministic_checks(state, state["candidate_answer"])
    assert issues and issues[0].startswith("invalid_citations:S9")


def test_valid_citations_pass():
    state = _state(candidate_answer="宝玉初见黛玉 [S1]，黛玉入府 [S2]。")
    assert runtime._deterministic_checks(state, state["candidate_answer"]) == []


def test_required_policy_requires_citation():
    state = _state(retrieval_policy="required", candidate_answer="无引用的回答。")
    issues = runtime._deterministic_checks(state, state["candidate_answer"])
    assert "missing_citation" in issues


def test_roleplay_requires_character_context():
    state = _state(interaction_mode="roleplay", answer_mode="roleplay")
    issues = runtime._deterministic_checks(state, "你好呀")
    assert "missing_character_context" in issues


# ---------- verify 节点 ----------

async def test_verify_marks_failed_on_invalid_citation():
    state = _state(candidate_answer="坏引用 [S7]。")
    result = await runtime._verify_node(state)
    assert result["verification"]["grounding_status"] == "failed"


async def test_verify_empty_answer_is_insufficient_not_failed():
    state = _state(candidate_answer=runtime._EMPTY_MESSAGE, sources=[])
    result = await runtime._verify_node(state)
    assert result["verification"]["grounding_status"] == "insufficient_evidence"


# ---------- repair 节点 ----------

async def test_repair_strips_invalid_citations_without_llm():
    """仅坏引用：确定性剥除即可通过，不调用模型。"""
    events = _events(_state())
    state = {**_state(candidate_answer="答案 [S1] 与 [S9]。"), "event_queue": events and None}
    state = _state(candidate_answer="答案 [S1] 与 [S9]。")
    verification = {"grounding_status": "failed", "issues": ["invalid_citations:S9"],
                    "semantic_checked": False, "semantic_unsupported": []}
    result = await runtime._repair_node({**state, "verification": verification})
    assert result["verification"]["grounding_status"] == "repaired"
    assert "[S9]" not in result["candidate_answer"]
    assert "[S1]" in result["candidate_answer"]
    assert result["repairs_used"] == 1


async def test_repair_regenerates_with_diagnostics(monkeypatch):
    """语义不可支持：带诊断重新生成一次，生成结果再过确定性检查。"""

    class FakeLLM:
        def __init__(self, content):
            self._content = content

        async def ainvoke(self, messages):
            class R:
                content = self._content
            return R()

    monkeypatch.setattr(runtime, "get_llm", lambda **kwargs: FakeLLM("修复后的回答 [S1]。"))
    state = _state()
    verification = {
        "grounding_status": "failed",
        "issues": ["unsupported_facts:宝玉比黛玉大三岁"],
        "semantic_checked": True,
        "semantic_unsupported": ["宝玉比黛玉大三岁"],
    }
    result = await runtime._repair_node({**state, "verification": verification})
    assert result["verification"]["grounding_status"] == "repaired"
    assert result["candidate_answer"] == "修复后的回答 [S1]。"


# ---------- finalize 节点 ----------

async def test_finalize_streams_buffered_answer_and_meta():
    """required 缓冲路径：验证通过后 finalize 才发 token，meta 带完整新字段。"""
    events: list[dict] = []

    class Queue:
        async def put(self, item):
            events.append(item)

    state = _state(
        candidate_answer="答案 [S1]。",
        candidate_streamed=False,
        retrieval_policy="required",
        event_queue=Queue(),
        requested_strategy="auto",
        strategy="react",
        strategy_adjusted=True,
        adjustment_reason="auto_routing",
        deprecations=["旧值提示"],
        tool_calls_used=2,
        rag_call_count=1,
        rag_called=True,
        verification={"grounding_status": "verified", "issues": [],
                      "semantic_checked": True, "semantic_unsupported": []},
    )
    result = await runtime._finalize_node(state)
    kinds = [e["type"] for e in events]
    assert "token" in kinds and "meta" in kinds
    assert result["answer"] == "答案 [S1]。"
    assert result["stop_reason"] == "answer_verified"
    meta = next(e["data"] for e in events if e["type"] == "meta")
    assert meta["requested_strategy"] == "auto"
    assert meta["effective_strategy"] == "react"
    assert meta["strategy_adjusted"] is True
    assert meta["tool_calls_used"] == 2
    assert meta["grounding_status"] == "verified"
    assert meta["completion_status"] == "completed"
    assert meta["deprecations"] == ["旧值提示"]


async def test_finalize_replaces_failed_buffered_answer():
    """缓冲路径修复仍失败：输出证据不足口径，不下发未验证内容。"""
    events: list[dict] = []

    class Queue:
        async def put(self, item):
            events.append(item)

    state = _state(
        candidate_answer="不可靠的答案",
        candidate_streamed=False,
        retrieval_policy="required",
        event_queue=Queue(),
        repairs_used=1,
        verification={"grounding_status": "failed", "issues": ["unsupported_facts:x"],
                      "semantic_checked": True, "semantic_unsupported": ["x"]},
    )
    result = await runtime._finalize_node(state)
    kinds = [e["type"] for e in events]
    assert "token" in kinds
    token = next(e["data"] for e in events if e["type"] == "token")
    assert token == runtime._INSUFFICIENT_MESSAGE
    assert result["stop_reason"] == "evidence_insufficient"


def test_after_verify_routes_repair_once():
    """验证失败且未修复 → repair；额度用尽 → finalize。"""
    failed = {"grounding_status": "failed", "issues": ["x"]}
    assert runtime._after_verify({**_state(), "verification": failed, "repairs_used": 0}) == "repair_answer"
    assert runtime._after_verify({**_state(), "verification": failed, "repairs_used": 1}) == "finalize"
    ok = {"grounding_status": "verified", "issues": []}
    assert runtime._after_verify({**_state(), "verification": ok, "repairs_used": 0}) == "finalize"
