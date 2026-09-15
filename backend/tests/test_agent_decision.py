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
from app.core.llm import LLMPurpose, ModelTurnDelta


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


def test_auto_routes_by_complexity_three_tiers():
    """auto 路由：复杂→plan_execute；闲聊→direct；普通小说问题→react。"""
    assert normalize_strategy("auto", "请全面梳理主要人物之间的关系、情节因果与时间线变化", True) is Strategy.PLAN_EXECUTE
    assert normalize_strategy("auto", "你好", False) is Strategy.DIRECT
    assert normalize_strategy("auto", "请讲一讲宝玉挨打之后贾府众人的反应", True) is Strategy.REACT
    assert normalize_strategy("react", "x", True) is Strategy.REACT
    # 简单单跳：短查询 + 至多一个小说信号 → direct
    assert normalize_strategy("auto", "林黛玉是谁", True) is Strategy.DIRECT


def test_route_query_direct_budget_is_one_tool_call():
    decision = route_query("你好", "direct", _hint())
    assert decision.retrieval_policy == "forbidden"
    assert set(decision.allowed_tools) == {"retrieve_novel", "get_chapter_context", "calculator"}
    assert decision.max_steps == 1  # direct 短路径：最多 1 次工具调用


def test_roleplay_interaction_mode_routing():
    """角色扮演：interaction_mode 决定工具集与输出策略，strategy 固定 react。"""
    decision = route_query("今天府里如何？", "auto", None, interaction_mode="roleplay")
    assert decision.strategy is Strategy.REACT
    assert decision.answer_mode == "roleplay"
    assert decision.retrieval_policy == "optional"
    assert decision.allowed_tools[0] == "load_character_context"
    assert decision.output_policy["allow_direct_quotes"] is True
    assert decision.output_policy["show_citations"] is False


def test_legacy_multi_expert_value_falls_back_to_direct():
    """router 不再认识 multi_expert（schemas 层已映射为 plan_execute）：未知值回退 direct。"""
    decision = route_query("你好呀", "multi_expert", _hint())
    assert decision.strategy is Strategy.DIRECT
    assert set(decision.allowed_tools) == {"retrieve_novel", "get_chapter_context", "calculator"}


# ---------- 图分派 ----------

def test_after_plan_dispatch():
    assert runtime._after_plan({"strategy": "plan_execute", "retrieval_policy": "required"}) == "execute"
    assert runtime._after_plan({"strategy": "plan_execute", "retrieval_policy": "optional"}) == "execute"
    assert runtime._after_plan({"strategy": "react", "retrieval_policy": "required", "answer_mode": "novel_evidence"}) == "retrieve"
    assert runtime._after_plan({"strategy": "react", "retrieval_policy": "optional", "answer_mode": "novel_evidence"}) == "execute"
    assert runtime._after_plan({"strategy": "direct", "retrieval_policy": "forbidden", "answer_mode": "conversation"}) == "execute"
    assert runtime._after_plan({"strategy": "react", "retrieval_policy": "required", "answer_mode": "conversation"}) == "execute"


# ---------- 计划校验器（纯函数，无 LLM） ----------

def _vp(steps, policy="optional", query="默认检索词"):
    return runtime._validate_plan(steps, retrieval_policy=policy, standalone_query=query)


def test_validate_plan_forbidden_skips_retrieval():
    plan, fb = _vp([{"action": "retrieve", "query": "q", "objective": "x"}], policy="forbidden")
    assert fb is False
    assert plan[0]["status"] == "skipped" and plan[0]["reason"] == "forbidden_policy"


def test_validate_plan_invalid_action_not_coerced():
    """非法 action 只标 invalid，绝不静默转成检索步骤。"""
    plan, _ = _vp([{"action": "summarize_everything", "objective": "x"}])
    assert plan[0]["status"] == "invalid" and plan[0]["reason"] == "invalid_action"


def test_validate_plan_tool_step_requires_query():
    plan, _ = _vp([{"action": "retrieve", "objective": "缺检索词"}])
    assert plan[0]["status"] == "invalid" and plan[0]["reason"] == "missing_query"


def test_validate_plan_dedupes_same_query():
    plan, _ = _vp([
        {"action": "retrieve", "query": "宝玉 挨打", "objective": "a"},
        {"action": "retrieve", "query": "宝玉 挨打", "objective": "b"},
    ])
    assert plan[0]["status"] == "pending"
    assert plan[1]["status"] == "skipped" and plan[1]["reason"] == "duplicate_query"


def test_validate_plan_no_retrieval_step_cap():
    """检索步骤数量无额外上限：4 个全部保留，由执行期全局预算硬闸。"""
    steps = [{"action": "retrieve", "query": f"q{i}", "objective": f"o{i}"} for i in range(4)]
    plan, _ = _vp(steps)
    assert all(s["status"] == "pending" for s in plan)
    assert len(plan) == 4


def test_validate_plan_required_inserts_retrieve():
    plan, fb = _vp([{"action": "analyze", "objective": "总结"}], policy="required")
    assert fb is False
    assert plan[0]["action"] == "retrieve" and plan[0]["status"] == "pending"
    assert plan[1]["status"] == "delegated"


def test_validate_plan_fallback_single_analyze():
    """解析失败：单步 analyze 兜底（required 再补检索），不再按行生成检索。"""
    plan, fb = _vp([], policy="optional")
    assert fb is True and len(plan) == 1
    assert plan[0]["action"] == "analyze" and plan[0]["status"] == "delegated"
    plan, fb = _vp([], policy="required")
    assert fb is True and plan[0]["action"] == "retrieve" and plan[1]["action"] == "analyze"


def test_validate_plan_truncates_to_six():
    steps = [{"action": "analyze", "objective": f"o{i}"} for i in range(8)]
    plan, _ = _vp(steps)
    assert len(plan) == 6


def test_validate_plan_dependency_cycle_cleared():
    plan, _ = _vp([
        {"action": "analyze", "objective": "a", "depends_on": [2]},
        {"action": "analyze", "objective": "b", "depends_on": [1]},
    ])
    assert all(not s["depends_on"] for s in plan)


def test_validate_plan_truncates_and_renumbers_ids():
    plan, _ = _vp([
        {"action": "retrieve", "query": "q1", "objective": "a"},
        {"action": "retrieve", "query": "q1", "objective": "重复"},
        {"action": "analyze", "objective": "b"},
    ])
    assert [s["id"] for s in plan] == [1, 2, 3]
    assert plan[0]["tool"] == "retrieve_novel"


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
    """构造 runtime.registry 替身：记录调用并返回固定 ToolResult。

    显式分发按工具名透传参数，替身按 registry.execute 的实际签名接收。
    """

    async def _dispatch(name, *, allowed_tools, query=None, expression=None,
                        file_id=None, neighbor_window=None, chapter_until=None,
                        retrieval_query=None, personas=None, **_):
        calls.append({"name": name, "query": query, "file_id": file_id,
                      "chapter_until": chapter_until, "personas": personas})
        return result

    return SimpleNamespace(execute=_dispatch, spec=lambda name: None)


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


async def test_plan_node_commits_task_plan(monkeypatch):
    """计划唯一来源是 _plan_node：关工具绑定、发唯一 plan 事件、reasoning_round=1。

    plan_execute 的计划不再由执行环生成——"展示计划"与"实际计划"从此一致。
    """
    events = []
    captured: dict = {}

    class Queue:
        async def put(self, item):
            events.append(item)

    async def stream(messages, purpose, *, tools=None, max_tokens=None):
        captured["tools"] = tools
        captured["purpose"] = purpose
        yield ModelTurnDelta(kind="reasoning", text="先理解任务")
        yield ModelTurnDelta(kind="content", text=(
            '[{"action": "retrieve", "query": "宝玉初见黛玉", "objective": "查找原文证据"},'
            '{"action": "compare", "objective": "比较人物立场"}]'
        ))

    monkeypatch.setattr(runtime, "astream_model_turn", stream)
    state = _state(strategy="plan_execute", event_queue=Queue(), retrieval_policy="optional")
    result = await runtime._plan_node(state)
    plans = [e["data"] for e in events if e["type"] == "plan"]
    starts = [e["data"] for e in events if e["type"] == "thinking_start"]
    assert plans and len(plans) == 1  # 唯一一次 plan 事件
    assert plans[0]["steps"][0]["status"] == "pending"       # 工具步骤
    assert plans[0]["steps"][1]["status"] == "delegated"     # 分析步骤委托 compose
    assert starts[0]["id"] == "agent-round-1" and starts[0]["reasoning_round"] == 1
    assert result["plan_committed"] is True and result["reasoning_round"] == 1
    # 规划阶段关闭工具绑定：模型不可能在规划时直接触发 RAG
    assert captured["tools"] is None
    assert captured["purpose"] == LLMPurpose.AGENT_PLAN


async def test_plan_node_falls_back_to_single_analyze(monkeypatch):
    """规划输出不是 JSON：单步 analyze 兜底，plan 事件标记 fallback_used。"""
    events = []

    class Queue:
        async def put(self, item):
            events.append(item)

    async def stream(messages, purpose, *, tools=None, max_tokens=None):
        yield ModelTurnDelta(kind="content", text="我先把任务理解一下，然后分两步处理……")

    monkeypatch.setattr(runtime, "astream_model_turn", stream)
    state = _state(strategy="plan_execute", event_queue=Queue(), retrieval_policy="optional")
    result = await runtime._plan_node(state)
    plans = [e["data"] for e in events if e["type"] == "plan"]
    assert plans[0]["fallback_used"] is True
    assert len(result["plan"]) == 1 and result["plan"][0]["action"] == "analyze"
    assert result["plan"][0]["status"] == "delegated"


async def test_execute_node_defers_to_committed_plan(monkeypatch):
    """执行环不再生成计划：plan_execute 无 pending 步骤时直接进入决策轮。"""
    monkeypatch.setattr(runtime, "astream_model_turn", _fake_stream(content="证据足够"))
    steps = [
        {"id": 1, "action": "retrieve", "tool": "retrieve_novel", "query": "q", "status": "done", "depends_on": []},
        {"id": 2, "action": "analyze", "objective": "比较", "tool": None, "query": None, "status": "delegated", "depends_on": []},
    ]
    result = await runtime._execute_node(_state(
        strategy="plan_execute", plan=steps, plan_committed=True,
        reasoning_round=1, tool_calls_used=1, current_step=2,
    ))
    assert result["react_done"] is True          # 模型宣布完成 → supervisor/compose
    assert result["reasoning_round"] == 2


async def test_parallel_plan_steps_keep_separate_ids(monkeypatch):
    """4 个并行计划步骤：plan_step_id = 1..4、tool_call_index 连续、不产生新推理轮次。"""
    events = []

    class Queue:
        async def put(self, item):
            events.append(item)

    steps = [{"id": i, "action": "retrieve", "tool": "retrieve_novel", "query": f"q{i}",
              "status": "pending", "depends_on": []} for i in range(1, 5)]
    state = _state(
        strategy="plan_execute",
        plan=steps,
        plan_committed=True,
        reasoning_round=1,
        tool_calls_used=0,
        current_step=1,
        allowed_tools=["retrieve_novel", "get_chapter_context", "calculator"],
        event_queue=Queue(),
    )
    result = await runtime._execute_plan_steps(state, steps)
    starts = [e["data"] for e in events if e["type"] == "tool_start"]
    decisions = [e["data"] for e in events if e["type"] == "agent_decision"]
    assert [s["plan_step_id"] for s in starts] == [1, 2, 3, 4]
    assert [s["tool_call_index"] for s in starts] == [1, 2, 3, 4]
    assert decisions and decisions[0]["action"] == "plan_execute"
    # 关键断言：并行步骤不新开推理轮次，仍归属产出计划的第 1 轮
    assert decisions[0]["reasoning_round"] == 1
    assert result["reasoning_round"] == 1
    assert [s["status"] for s in steps] == ["done"] * 4
    # 执行完成后必须重发 plan 事件：否则前端快照永远停留在"待执行"
    plan_events = [e["data"] for e in events if e["type"] == "plan"]
    assert plan_events and plan_events[-1]["steps"][0]["status"] == "done"


async def test_zero_retrieval_plan_executes_no_rag(monkeypatch):
    """用户材料齐全：计划只含 delegated 步骤 → 零工具调用、零 RAG。"""
    monkeypatch.setattr(runtime, "astream_model_turn", _fake_stream(content="证据足够"))
    steps = [
        {"id": 1, "action": "understand", "objective": "识别用户要求", "tool": None, "query": None, "status": "delegated", "depends_on": []},
        {"id": 2, "action": "synthesize", "objective": "总结结论", "tool": None, "query": None, "status": "delegated", "depends_on": []},
    ]
    state = _state(strategy="plan_execute", plan=steps, plan_committed=True, reasoning_round=1)
    result = await runtime._execute_node(state)
    assert result["react_done"] is True
    assert result.get("rag_call_count", 0) == 0 and result["tool_calls_used"] == 0


async def test_plan_steps_beyond_retrieval_budget_denied(monkeypatch):
    """检索步骤总量由全局预算硬闸：超限步骤 skipped 并带 reason。"""
    monkeypatch.setattr(runtime, "registry", _fake_registry(_retrieval_result(), []))
    steps = [{"id": i, "action": "retrieve", "tool": "retrieve_novel", "query": f"q{i}",
              "status": "pending", "depends_on": []} for i in range(1, 5)]
    state = _state(
        strategy="plan_execute", plan=steps, plan_committed=True,
        reasoning_round=1, tool_calls_used=0, current_step=1,
        budget={"max_steps": 8, "max_tool_calls": 8, "max_retrieval_calls": 2},
        allowed_tools=["retrieve_novel"],
    )
    await runtime._execute_plan_steps(state, steps)
    statuses = [s["status"] for s in steps]
    assert statuses == ["done", "done", "skipped", "skipped"]
    assert steps[2]["reason"] == "exceeds_retrieval_budget"


async def test_next_reasoning_round_after_parallel_steps(monkeypatch):
    """并行步骤后下一次主 Agent 推理是第 2 轮——不再出现"第 5 步"。"""
    events = []

    class Queue:
        async def put(self, item):
            events.append(item)

    monkeypatch.setattr(runtime, "astream_model_turn", _fake_stream(content="证据足够，直接回答", reasoning="收尾判断"))
    state = _state(
        reasoning_round=1, tool_calls_used=4, current_step=5,
        event_queue=Queue(),
    )
    result = await runtime._execute_node(state)
    starts = [e["data"] for e in events if e["type"] == "thinking_start"]
    decisions = [e["data"] for e in events if e["type"] == "agent_decision"]
    assert starts and starts[0]["id"] == "agent-round-2"
    assert starts[0]["reasoning_round"] == 2
    assert decisions and decisions[0]["reasoning_round"] == 2
    assert result["reasoning_round"] == 2
    assert result["react_done"] is True


async def test_tool_call_index_counts_calls_not_rounds(monkeypatch):
    """工具调用序号按调用次数递增，与推理轮次相互独立。"""
    calls: list[dict] = []
    requests = [{"name": "retrieve_novel", "args": {"query": f"q{i}"}, "id": f"c{i}"} for i in range(2)]
    monkeypatch.setattr(runtime, "astream_model_turn", _fake_stream(content="查", tool_calls=requests))
    monkeypatch.setattr(runtime, "registry", _fake_registry(_retrieval_result(), calls))
    result = await runtime._execute_node(_state())
    assert result["tool_calls_used"] == 2    # 两次调用
    assert result["reasoning_round"] == 1    # 仍是一个推理轮


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
