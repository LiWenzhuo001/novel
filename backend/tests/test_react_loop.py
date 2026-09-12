"""ReAct 执行环的纯逻辑测试（无 DB、无 LLM 调用）。"""
from __future__ import annotations

from app.agent import runtime
from app.agent.types import AgentState


def test_after_reflect_continues_within_budget():
    state: AgentState = {"current_step": 2, "max_steps": 6}
    assert runtime._after_reflect(state) == "execute"


def test_after_reflect_stops_on_budget():
    state: AgentState = {"current_step": 6, "max_steps": 6}
    assert runtime._after_reflect(state) == "supervisor"


def test_after_reflect_stops_on_done_or_fallback():
    assert runtime._after_reflect({"react_done": True, "current_step": 1, "max_steps": 6}) == "supervisor"
    assert runtime._after_reflect({"fallback_reason": "tool_failed", "current_step": 1, "max_steps": 6}) == "supervisor"


def test_graph_contains_react_back_edge():
    """编译后的图必须含 reflect→execute 回边：这是 ReAct 循环真实存在的最低要求。"""
    pairs = {(e.source, e.target) for e in runtime.agent_graph.get_graph().edges}
    assert ("reflect", "execute") in pairs
    assert ("execute", "reflect") in pairs


async def test_execute_budget_guard_short_circuits():
    """步数预算在入口拦截：max_steps 从装饰性变成真实约束。"""
    state: AgentState = {
        "current_step": 6, "max_steps": 6, "strategy": "react",
        "allowed_tools": ["retrieve_novel"], "standalone_query": "q",
    }
    result = await runtime._execute_node(state)
    assert result["react_done"] is True
    assert result["fallback_reason"] == "step_budget_exceeded"


def test_parse_plan_text_caps_five_steps():
    steps = runtime._parse_plan_text("\n".join(f"第{i}步：做事{i}" for i in range(9)))
    assert len(steps) == 5
    assert steps[0]["action"] == "model_step"
