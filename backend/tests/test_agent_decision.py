"""Agent-first 决策逻辑的纯逻辑测试（无 DB、无真实 LLM）。

映射验收计划：闲聊直答不调 RAG、小说事实主动检索、低置信/失效 prep 保守兜底、
auto 混合路由、单轮工具上限、消息回灌、预算终止、supervisor 动态 answer_mode。
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from app.agent import runtime
from app.agent.router import _routing_choice, normalize_strategy, route_query
from app.agent.types import AgentState, Strategy, ToolResult
from app.config import settings


# ---------- 路由三档政策 ----------

def _hint(**overrides):
    base = {
        "standalone_query": "q",
        "retrieval_query": "",
        "reason": "rewritten",
        "intent": "other",
        "needs_retrieval": False,
        "answer_mode": "conversation",
        "retrieval_reason": "conversation_only",
        "confidence": 0.9,
    }
    base.update(overrides)
    return base


def test_chitchat_high_confidence_is_forbidden():
    needs, policy, reason, mode, *_ = _routing_choice("你好呀", _hint())
    assert needs is False and policy == "forbidden" and mode == "conversation"


def test_novel_fact_hint_is_optional_advice():
    """LLM 判需要检索只是建议（纯提示驱动），policy=optional 不预检索。"""
    needs, policy, _, mode, *_ = _routing_choice(
        "宝玉初见黛玉是第几章", _hint(needs_retrieval=True, answer_mode="novel_evidence",
                                      retrieval_query="宝玉初见黛玉", retrieval_reason="chapter_locator"),
    )
    assert needs is True and policy == "optional" and mode == "novel_evidence"


def test_low_confidence_novel_hint_falls_back_to_required():
    needs, policy, reason, *_ = _routing_choice(
        "宝玉几岁", _hint(needs_retrieval=True, answer_mode="novel_evidence",
                          retrieval_query="宝玉 年龄", confidence=0.2),
    )
    assert needs is True and policy == "required" and reason == "forced_by_low_confidence"


def test_invalid_hint_is_required_fallback():
    needs, policy, reason, *_ = _routing_choice("任意问题", {"standalone_query": "q"})
    assert needs is True and policy == "required" and reason == "query_preparation_failed"


def test_non_rewritten_hint_is_required():
    needs, policy, *_ = _routing_choice("任意问题", _hint(reason="error"))
    assert policy == "required"


def test_strong_novel_signal_overrides_skip():
    needs, policy, reason, *_ = _routing_choice("帮我梳理宝玉的人物关系", _hint())
    assert needs is True and policy == "required" and reason == "forced_by_strong_novel_signal"


def test_rules_without_llm():
    needs, policy, reason, *_ = _routing_choice("宝玉和黛玉什么关系", None)
    assert needs is True and policy == "required"
    needs, policy, reason, mode, *_ = _routing_choice("嗯嗯好的谢谢", None)
    assert needs is False and policy == "forbidden" and mode == "conversation"


def test_auto_routes_complex_novel_to_multi_expert_else_react():
    assert normalize_strategy("auto", "请全面梳理主要人物之间的关系、情节因果与时间线变化", True) is Strategy.MULTI_EXPERT
    assert normalize_strategy("auto", "你好", False) is Strategy.REACT
    assert normalize_strategy("auto", "随便聊聊今天天气", True) is Strategy.REACT
    assert normalize_strategy("react", "x", True) is Strategy.REACT


def test_route_query_tools_include_react_set():
    decision = route_query("你好", "direct", _hint())
    assert decision.retrieval_policy == "forbidden"
    assert set(decision.allowed_tools) == {"retrieve_novel", "get_chapter_context", "calculator"}
    assert decision.max_steps == 2  # direct 短路径


def test_multi_expert_downgrade_keeps_tools():
    decision = route_query("你好呀", "multi_expert", _hint())
    assert decision.strategy is Strategy.DIRECT
    assert set(decision.allowed_tools) == {"retrieve_novel", "get_chapter_context", "calculator"}


# ---------- 图分派 ----------

def test_after_plan_dispatch():
    assert runtime._after_plan({"strategy": "multi_expert", "retrieval_policy": "optional"}) == "retrieve"
    assert runtime._after_plan({"strategy": "react", "retrieval_policy": "required"}) == "retrieve"
    assert runtime._after_plan({"strategy": "react", "retrieval_policy": "optional"}) == "execute"
    assert runtime._after_plan({"strategy": "direct", "retrieval_policy": "forbidden"}) == "execute"


def test_graph_has_react_loop_and_no_fixed_retrieve_for_react():
    pairs = {(e.source, e.target) for e in runtime.agent_graph.get_graph().edges}
    assert ("execute", "reflect") in pairs and ("reflect", "execute") in pairs
    # plan 条件边必须允许直连 execute（普通路径不再强制先检索）
    cond = [e for e in runtime.agent_graph.get_graph().edges if e.source == "plan"]
    assert {e.target for e in cond} == {"retrieve", "execute"}


# ---------- 提示词构造 ----------

def test_initial_human_injects_required_and_hint():
    state: AgentState = {
        "original_query": "宝玉几岁", "standalone_query": "宝玉几岁",
        "retrieval_policy": "required",
        "query_preparation": {"needs_retrieval": True, "retrieval_reason": "factual"},
    }
    text = runtime._react_initial_human(state)
    assert "retrieve_novel" in text and "保守兜底" in text and "查询准备建议" in text


def test_system_prompt_direct_short_path():
    text = runtime._react_system_prompt({"strategy": "direct"})
    assert "短路径" in text
    assert "不得以常识、会话记忆或猜测冒充小说事实" in runtime._react_system_prompt({"strategy": "react"})


# ---------- execute 节点（假 delta 流 + 假工具通道） ----------

def _fake_stream(content="", tool_calls=None, reasoning=None):
    """构造 astream_model_turn 替身：按 ModelTurnDelta 协议产出增量。"""
    from app.core.llm import ModelTurnDelta

    async def stream(messages, purpose, *, tools=None, max_tokens=None):
        if reasoning:
            yield ModelTurnDelta(kind="reasoning", text=reasoning)
        if content:
            yield ModelTurnDelta(kind="content", text=content)
        for i, call in enumerate(tool_calls or []):
            yield ModelTurnDelta(kind="tool_call", tool_call_chunk={
                "index": i, "id": call["id"], "name": call["name"],
                "args_str": json.dumps(call.get("args") or {}, ensure_ascii=False),
            })

    return stream


def _fake_registry(result, calls):
    """构造 runtime.registry 替身：记录调用并返回固定 ToolResult。"""

    async def _dispatch(name, *, allowed_tools, query=None, expression=None, file_id=None, neighbor_window=None):
        calls.append({"name": name, "query": query, "file_id": file_id})
        return result

    return SimpleNamespace(execute=_dispatch)


def _retrieval_result(chapter="第三回"):
    source = {"id": "S1", "source": "红楼梦.txt", "chapter": chapter, "chapter_no": 3, "chunk_no": 1}
    return ToolResult(status="ok", output={"evidence": [{"source": source, "content": "宝玉初见黛玉"}],
                                           "sources": [source]}, citations=[source])


def _state(**overrides):
    state: AgentState = {
        "strategy": "react", "allowed_tools": ["retrieve_novel", "get_chapter_context", "calculator"],
        "max_steps": 6, "standalone_query": "宝玉初见黛玉是第几章", "original_query": "宝玉初见黛玉是第几章",
        "file_id": "f1", "current_step": 1, "retrieval_policy": "optional",
        "query_preparation": {}, "react_messages": [],
    }
    state.update(overrides)
    return state


async def test_direct_answer_without_tools(monkeypatch):
    """闲聊：模型不调工具 → 循环终止、零 RAG、stop_reason=model_answer。"""
    monkeypatch.setattr(runtime, "astream_model_turn", _fake_stream(content="你好！"))
    result = await runtime._execute_node(_state(original_query="你好", standalone_query="你好"))
    assert result["react_done"] is True
    assert result["rag_called"] is False
    assert result["stop_reason"] == "model_answer"


async def test_reasoning_stream_lifecycle(monkeypatch):
    """reasoning → thinking_start/token/end 生命周期；decide content 不进答案流。"""
    events = []

    class Queue:
        async def put(self, item):
            events.append(item)

    monkeypatch.setattr(settings, "expose_raw_reasoning", True, raising=False)
    monkeypatch.setattr(runtime, "astream_model_turn", _fake_stream(
        content="仅内部辅助文本", tool_calls=None, reasoning="先判断这算不算小说问题",
    ))
    await runtime._execute_node(_state(original_query="你好", standalone_query="你好", event_queue=Queue()))
    kinds = [e["type"] for e in events]
    assert kinds[0] == "thinking_start" and "thinking_token" in kinds
    assert kinds.index("thinking_end") < kinds.index("agent_decision")  # 先收尾思考，再发决策
    tokens = [e for e in events if e["type"] == "thinking_token"]
    assert tokens[0]["data"]["stream"] == "main_agent" and tokens[0]["data"]["phase"] == "decide"
    end = [e for e in events if e["type"] == "thinking_end"][0]["data"]
    assert end["status"] == "completed" and end["reasoning_chars"] == len("先判断这算不算小说问题")
    # decide 阶段 content 不得进入答案：无 token 事件
    assert "token" not in kinds


async def test_reasoning_hidden_when_expose_disabled(monkeypatch):
    """EXPOSE_RAW_REASONING=false：不发原文 delta，但 thinking_end 仍带统计。"""
    events = []

    class Queue:
        async def put(self, item):
            events.append(item)

    monkeypatch.setattr(settings, "expose_raw_reasoning", False, raising=False)
    monkeypatch.setattr(runtime, "astream_model_turn", _fake_stream(reasoning="敏感推理原文"))
    await runtime._execute_node(_state(event_queue=Queue()))
    assert not [e for e in events if e["type"] == "thinking_token"]
    ends = [e for e in events if e["type"] == "thinking_end"]
    assert ends and ends[0]["data"]["reasoning_chars"] == len("敏感推理原文")


async def test_tool_call_without_content_emits_decision(monkeypatch):
    """回归：模型只返回工具调用、无正文时，agent_decision 的 reason 兜底为工具名。

    曾用 c.get("name") 读取 ToolCall 数据类导致 AttributeError，整个图失败。
    """
    events = []

    class Queue:
        async def put(self, item):
            events.append(item)

    monkeypatch.setattr(runtime, "astream_model_turn", _fake_stream(
        content="", tool_calls=[{"name": "retrieve_novel", "args": {"query": "宝玉初见黛玉"}, "id": "c1"}],
    ))
    monkeypatch.setattr(runtime, "registry", _fake_registry(_retrieval_result(), []))
    result = await runtime._execute_node(_state(event_queue=Queue()))
    decisions = [e["data"] for e in events if e["type"] == "agent_decision"]
    assert decisions and decisions[0]["reason"] == "retrieve_novel"
    assert result["react_done"] is False


async def test_novel_fact_triggers_retrieve_and_feedback(monkeypatch):
    """小说事实：模型调 retrieve_novel → 证据入状态、ToolMessage 回灌、RAG 计数。"""
    calls: list[dict] = []
    monkeypatch.setattr(runtime, "astream_model_turn", _fake_stream(
        content="查一下",
        tool_calls=[{"name": "retrieve_novel", "args": {"query": "宝玉初见黛玉"}, "id": "c1"}],
    ))
    monkeypatch.setattr(runtime, "registry", _fake_registry(_retrieval_result(), calls))
    result = await runtime._execute_node(_state())
    assert result["react_done"] is False
    assert result["rag_called"] is True and result["rag_call_count"] == 1
    assert len(result["evidence"]) == 1
    # 消息回灌：assistant(tool_calls) + ToolMessage
    assert len(result["react_messages"]) == 3
    kinds = [type(m).__name__ for m in result["react_messages"]]
    assert kinds == ["HumanMessage", "AIMessage", "ToolMessage"]
    # file_id 由状态注入，不来自模型参数
    assert calls[0]["file_id"] == "f1"
    assert calls[0]["query"] == "宝玉初见黛玉"


async def test_turn_tool_budget_caps_at_two(monkeypatch):
    """单轮最多执行 2 个工具调用，多余调用不产生悬空 ToolMessage。"""
    calls: list[dict] = []
    requests = [{"name": "retrieve_novel", "args": {"query": f"q{i}"}, "id": f"c{i}"} for i in range(4)]
    monkeypatch.setattr(runtime, "astream_model_turn", _fake_stream(content="多查几轮", tool_calls=requests))
    monkeypatch.setattr(runtime, "registry", _fake_registry(_retrieval_result(), calls))
    result = await runtime._execute_node(_state())
    assert len(calls) == 2
    assert result["current_step"] == 3
    assert result["react_done"] is False
    assert len(result["react_messages"]) == 4  # human + assistant(tool_calls) + 2×tool


async def test_budget_exhausted_terminates(monkeypatch):
    result = await runtime._execute_node(_state(current_step=6, max_steps=6))
    assert result["react_done"] is True
    assert result["fallback_reason"] == "step_budget_exceeded"
    assert result["stop_reason"] == "step_budget_exceeded"


async def test_plan_execute_first_turn_only_plans(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(runtime, "astream_model_turn", _fake_stream(
        content="1. 检索初见章节\n2. 计算年龄差",
        tool_calls=[{"name": "retrieve_novel", "args": {"query": "x"}, "id": "c0"}],
    ))
    monkeypatch.setattr(runtime, "registry", _fake_registry(_retrieval_result(), calls))
    result = await runtime._execute_node(_state(strategy="plan_execute"))
    assert result["plan_committed"] is True
    assert len(result["plan"]) == 2
    assert calls == []  # 计划轮不执行任何工具
    assert result["react_done"] is False


async def test_spontaneous_react_plan_emitted(monkeypatch):
    """react 首轮模型带多行计划文本 + 工具调用 → 计划经 plan 事件展示。"""
    events = []

    class Queue:
        async def put(self, item):
            events.append(item)

    monkeypatch.setattr(runtime, "astream_model_turn", _fake_stream(
        content="1. 检索初见回目\n2. 检索年龄描写",
        tool_calls=[{"name": "retrieve_novel", "args": {"query": "初见"}, "id": "c1"}],
    ))
    monkeypatch.setattr(runtime, "registry", _fake_registry(_retrieval_result(), []))
    result = await runtime._execute_node(_state(event_queue=Queue()))
    assert result["react_done"] is False
    plans = [e for e in events if e["type"] == "plan"]
    assert plans and plans[0]["data"]["steps"][0]["action"] == "model_step"


# ---------- supervisor 动态 answer_mode ----------

async def test_supervisor_effective_mode():
    state: AgentState = {
        "strategy": "react", "answer_mode": "novel_evidence", "rag_called": True,
        "needs_retrieval": True, "evidence": [], "sources": [], "reports": [],
        "report_validation": {}, "observations": [], "memory_context": {},
        "current_step": 3,
    }
    result = await runtime._supervisor_node(state)
    assert result["synthesis_context"]["effective_answer_mode"] == "novel_evidence"

    state["rag_called"] = False
    result = await runtime._supervisor_node(state)
    ctx = result["synthesis_context"]
    # 模型跳过检索：走对话分支，且标记"路由建议过检索"
    assert ctx["effective_answer_mode"] == "conversation"
    assert ctx["route_suggested_retrieval"] is True
    assert ctx["observations"] == []
