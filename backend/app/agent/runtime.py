"""LangGraph Agent Runtime：统一单图编排（route → plan → execute ↔ reflect →
supervisor → memory_agent → compose → verify → finalize）。

- 三种实际策略：direct（≤1 次工具调用）/ react（默认自主循环）/
  plan_execute（结构化计划 + 并行步骤）；auto 必须解析为其中之一。
- 角色扮演是交互模式不是策略：load_character_context 工具 + 专用提示词，
  最终角色的回复内容直接作为候选答案，不再二次生成。
- 预算由运行时代码计数（ExecutionBudget），不依赖 Prompt 声明。
- 候选答案必须经 verify_answer 校验后才允许输出/持久化；required 策略的
  答案先缓冲再验证，direct/optional 直接流式 + 确定性校验 + token_replace 修补。
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from app.agent.router import route_query
from app.agent.tools import (
    MEMORY_AGENT_TOOLS,
    MEMORY_AGENT_TOOL_SPECS,
    REACT_TOOL_LABELS,
    _react_payload,
    model_tool_specs,
    registry,
)
from app.agent.types import (
    AgentState,
    DEFAULT_OUTPUT_POLICY,
    ExecutionBudget,
    Strategy,
    ToolResult,
)
from app.config import settings
from app.core.context import get_memory_session, set_memory_session
from app.core.llm import LLMPurpose, ModelTurnBuilder, astream_model_turn, get_llm, tool_call_field
from app.core.logging_config import get_logger
from app.core.metrics import metrics
from app.services import world_service

log = get_logger("agent_runtime")
_EMPTY_MESSAGE = "当前小说知识库中没有检索到足以回答该问题的原文。请确认作品已完成索引，或补充人物名、事件名、章节等线索。"
_INSUFFICIENT_MESSAGE = "抱歉，现有检索证据不足以支持该答案中的关键事实。请补充人物名、章节等线索后重试，或换一个更具体的问题。"
_STREAM_DONE = object()
# 单轮 reasoning 累计字符上限：防异常模型无限输出（reasoning 只展示不持久化）。
_THINKING_MAX_CHARS = 8000
# ReAct 证据累积上限（块数）：循环成立后证据跨轮增长，系统没有全局 token
# 计数器，必须在源头封顶，否则证据滚雪球会放大 summary prompt 成本。
_REACT_MAX_EVIDENCE = 24
# plan_execute 任务计划：步骤数上限（非工具步骤委托 compose，不等于 RAG 次数）。
_PLAN_MAX_STEPS = 6
# action → registry 工具的确定性映射；只有这三类步骤会触发真实工具调用。
_PLAN_TOOL_ACTIONS = {
    "retrieve": "retrieve_novel",
    "chapter_context": "get_chapter_context",
    "calculate": "calculator",
}
# 非工具步骤：分析/比较/总结类目标，委托给 compose_answer 完成。
_PLAN_DELEGATED_ACTIONS = {"understand", "analyze", "compare", "synthesize"}
_PLAN_ALL_ACTIONS = set(_PLAN_TOOL_ACTIONS) | _PLAN_DELEGATED_ACTIONS
_RETRIEVAL_TOOLS = {"retrieve_novel", "get_chapter_context"}


class _ThinkingSpan:
    """thinking 流生命周期助手：start/token/end 各至多一次，覆盖取消与异常收尾。

    reasoning 原文是否下发由 settings.expose_raw_reasoning 决定；
    统计（字符数/截断）无论如何都随 thinking_end 上报。
    """

    def __init__(
        self,
        state: AgentState,
        *,
        stream: str,
        span_id: str,
        phase: str,
        agent: str | None = None,
        label: str | None = None,
        reasoning_round: int | None = None,
        retry: int | None = None,
    ) -> None:
        self._state = state
        self._stream = stream
        self._id = span_id
        self._phase = phase
        self._agent = agent
        self._label = label
        self._round = reasoning_round
        self._retry = retry
        self._started = False
        self._ended = False
        self.chars = 0
        self.truncated = False
        self.started_at = time.perf_counter()

    def _payload(self, **extra: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {"stream": self._stream, "id": self._id, "phase": self._phase}
        if self._agent is not None:
            payload["agent"] = self._agent
        if self._label is not None:
            payload["label"] = self._label
        if self._round is not None:
            # 主 Agent 推理只按"轮次"编号：与工具调用序号、计划步骤编号语义分离。
            payload["reasoning_round"] = self._round
        if self._retry is not None:
            payload["retry"] = self._retry
        payload.update(extra)
        return payload

    async def token(self, text: str) -> None:
        if self._ended or not text:
            return
        if not self._started:
            self._started = True
            await _emit(self._state, "thinking_start", self._payload(status="running"))
        if self.chars >= _THINKING_MAX_CHARS:
            self.truncated = True
            return
        self.chars += len(text)
        if settings.expose_raw_reasoning:
            await _emit(self._state, "thinking_token", self._payload(delta=text))

    async def end(self, status: str) -> None:
        if self._ended or not self._started:
            self._ended = True
            return
        self._ended = True
        await _emit(self._state, "thinking_end", self._payload(
            status=status,
            reasoning_chars=self.chars,
            truncated=self.truncated,
            latency_ms=round((time.perf_counter() - self.started_at) * 1000, 1),
        ))


async def _emit(state: AgentState, event_type: str, data: Any) -> None:
    queue = state.get("event_queue")
    if queue is not None:
        await queue.put({"type": event_type, "data": data})


def _plan(strategy: Strategy) -> list[dict[str, Any]]:
    """direct/react 的展示骨架；plan_execute 的真实任务计划由 _plan_node 产出。"""
    if strategy is Strategy.DIRECT:
        return [
            {"step": 1, "action": "react_loop", "purpose": "短路径决策：最多一次工具调用"},
            {"step": 2, "action": "compose_answer", "purpose": "生成并校验最终答案"},
        ]
    return [
        {"step": 1, "action": "react_loop", "purpose": "模型自主决定是否检索、计算或直接回答"},
        {"step": 2, "action": "compose_answer", "purpose": "生成并校验最终答案"},
    ]


def _evidence_key(original: dict[str, Any]) -> tuple[Any, ...] | None:
    """证据去重的回退键：file 内 source + 章节号 + 片段号 + 字符区间。

    元数据不全（缺 chunk_no 或 char_start）时不参与去重——绝不把缺少定位
    信息的不同片段合并成一条。
    """
    chunk_no = original.get("chunk_no")
    char_start = original.get("char_start")
    if chunk_no is None or char_start is None:
        return None
    return (
        original.get("source"),
        original.get("chapter_no"),
        chunk_no,
        char_start,
        original.get("char_end"),
    )


def _normalize_evidence(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """按稳定回退键去重证据，并重新编号为稳定的 [S#] 引用。"""
    seen: set[tuple[Any, ...]] = set()
    evidence: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for item in items:
        original = item.get("source", {})
        key = _evidence_key(original)
        if key is not None:
            if key in seen:
                continue
            seen.add(key)
        source = dict(original)
        source["id"] = f"S{len(sources) + 1}"
        sources.append(source)
        evidence.append({"source": source, "content": item.get("content", "")})
    return evidence, sources


def _evidence_text(evidence: list[dict[str, Any]]) -> str:
    """把结构化证据拼接成模型可读的共享原文上下文。"""
    return "\n\n".join(
        f"[{item['source']['id']}] {item['source'].get('source', '未知')} / "
        f"{item['source'].get('chapter') or '未分章'}\n{item['content']}"
        for item in evidence
    )


async def _stream_llm(messages: list[Any], max_tokens: int):
    """以流式方式读取模型输出，并兼容字符串和多模态内容块。"""
    async for chunk in get_llm(streaming=True, temperature=0, max_tokens=max_tokens).astream(messages):
        content = getattr(chunk, "content", "")
        if isinstance(content, str) and content:
            yield content
        elif isinstance(content, list):
            text = "".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
            if text:
                yield text


async def _route_node(state: AgentState) -> dict[str, Any]:
    """LangGraph 路由节点：决定执行策略、检索政策与统一预算。"""
    decision = route_query(
        state["standalone_query"],
        state.get("requested_strategy"),
        routing_hint=state.get("query_preparation") or state.get("query_rewrite") or None,
        interaction_mode=state.get("interaction_mode", "qa"),
    )
    # 客户端 max_steps 只能收紧预算：取 min(请求值, 服务端档位, 全局上限)。
    requested = state.get("requested_max_steps")
    max_steps = min(
        requested or decision.max_steps,
        decision.max_steps,
        settings.agent_max_steps,
    )
    budget = ExecutionBudget(
        max_steps=max_steps,
        max_tool_calls=settings.agent_max_tool_calls,
        max_retrieval_calls=settings.agent_max_retrieval_calls,
        max_plan_adjustments=1,
        deadline_seconds=settings.agent_request_timeout,
    )
    adjusted = (state.get("requested_strategy") or "auto") == "auto"
    route = decision.as_dict()
    route.update({
        "requested_strategy": state.get("requested_strategy", "auto"),
        "effective_strategy": decision.strategy.value,
        "interaction_mode": state.get("interaction_mode", "qa"),
        "max_steps": max_steps,
        "budget": budget.as_dict(),
        "retrieval_skipped": decision.retrieval_policy != "required",
        "llm_needs_retrieval": decision.llm_needs_retrieval,
        "routing_override": decision.routing_override,
        "routing_override_reason": decision.routing_override_reason,
        "routing_confidence": decision.routing_confidence,
    })
    await _emit(state, "route", route)
    return {
        "strategy": decision.strategy.value,
        "intent": decision.intent,
        "allowed_tools": list(decision.allowed_tools),
        "max_steps": max_steps,
        "budget": budget.as_dict(),
        "needs_retrieval": decision.needs_retrieval,
        "retrieval_policy": decision.retrieval_policy,
        "retrieval_reason": decision.retrieval_reason,
        "answer_mode": decision.answer_mode,
        "llm_needs_retrieval": decision.llm_needs_retrieval,
        "routing_override": decision.routing_override,
        "routing_override_reason": decision.routing_override_reason,
        "routing_confidence": decision.routing_confidence,
        "output_policy": decision.output_policy,
        "preference_update": decision.preference_update,
        "strategy_adjusted": adjusted,
        "adjustment_reason": "auto_routing" if adjusted else "",
        "evidence": [],
        "sources": [],
        "observations": [],
        "fallback_reason": "",
    }


async def _plan_node(state: AgentState) -> dict[str, Any]:
    """计划节点：plan_execute 产出真实任务计划（唯一来源），direct/react 发展示骨架。

    规划调用关闭工具绑定（tools=None）：模型在规划阶段只能"纸上谈兵"，
    不可能直接触发 RAG；分析/比较/总结类目标委托给 compose，不转成检索。
    """
    strategy = Strategy(state["strategy"])
    if strategy is not Strategy.PLAN_EXECUTE:
        plan = _plan(strategy)
        if state.get("retrieval_policy") == "required" and state.get("answer_mode") == "novel_evidence":
            # required 兜底：预检索只是初始证据来源，之后仍进入同一决策循环。
            plan = [
                {"step": 1, "action": "retrieve_novel", "purpose": "保守兜底预检索（查询准备不可靠）"},
                {"step": 2, "action": "react_loop", "purpose": "模型自主决定补充检索、计算或直接回答"},
                {"step": 3, "action": "compose_answer", "purpose": "生成并校验最终答案"},
            ]
        await _emit(state, "plan", {
            "steps": plan,
            "max_steps": state["max_steps"],
            "retrieval_policy": state.get("retrieval_policy", "optional"),
        })
        return {"plan": plan}

    # ===== 任务规划：理解任务 → 判断材料 → 按需检索 → 分析/比较/总结 =====
    reasoning_round = int(state.get("reasoning_round", 0)) + 1
    thinking = _ThinkingSpan(
        state, stream="main_agent", span_id=f"agent-round-{reasoning_round}",
        phase="plan", reasoning_round=reasoning_round,
    )
    prep = state.get("query_preparation") or {}
    prep_hint = ""
    if prep.get("needs_retrieval"):
        reason = prep.get("retrieval_reason") or "novel_evidence"
        prep_hint = f"\n查询准备提示：该问题可能需要小说原文检索（{reason}）。"
    policy_text = {
        "required": "检索政策：required——无论计划如何，系统会保证至少一次成功检索。",
        "optional": "检索政策：optional——模型确信无需原文时，可以生成完全不含检索步骤的计划。",
        "forbidden": "检索政策：forbidden——本轮禁止检索，只能安排分析/比较/总结类步骤。",
    }.get(state.get("retrieval_policy", "optional"), "")
    prompt = (
        "你是任务规划助手。请为完成用户的请求制定工作计划，只输出 JSON 数组，不要多余文本。\n"
        "每个步骤形如："
        '{"action": "...", "objective": "这一步要完成什么", "query": "检索词(仅检索步骤)", "depends_on": []}。\n'
        f"允许的 action：{', '.join(sorted(_PLAN_ALL_ACTIONS))}。\n"
        "规则：\n"
        "- 先理解用户真正要的最终产物，再判断已有上下文是否足够；\n"
        "- 只有缺少小说原文事实时才生成 retrieve/chapter_context 步骤（必须带 query）；\n"
        "- 分析、比较、总结永远不要转换成检索——它们由汇总节点完成；\n"
        "- 最多 6 个步骤；用户已提供完整材料时，生成不含检索的计划。\n"
        f"{policy_text}{prep_hint}\n\n"
        f"用户请求：{state.get('original_query', '') or state.get('standalone_query', '')}"
    )
    builder = ModelTurnBuilder()
    try:
        async for delta in astream_model_turn(
            [SystemMessage(content="你是严谨的任务规划助手，只输出 JSON 计划。"),
             HumanMessage(content=prompt)],
            LLMPurpose.AGENT_PLAN, tools=None, max_tokens=600,
        ):
            if delta.kind == "reasoning":
                await thinking.token(delta.text)
            else:
                builder.add_delta(delta)
        await thinking.end("completed")
    except asyncio.CancelledError:
        await thinking.end("cancelled")
        raise
    except Exception:
        await thinking.end("error")
        raise
    turn = builder.build()
    raw_steps = _parse_plan(turn.content)
    plan, fallback_used = _validate_plan(
        raw_steps,
        retrieval_policy=state.get("retrieval_policy", "optional"),
        standalone_query=state.get("standalone_query", ""),
    )
    await _emit(state, "plan", {
        "steps": plan,
        "max_steps": state["max_steps"],
        "retrieval_policy": state.get("retrieval_policy", "optional"),
        "fallback_used": fallback_used,
    })
    await _emit(state, "agent_decision", {
        "action": "plan",
        "reason": "已生成任务工作计划" + ("（解析失败，使用安全兜底）" if fallback_used else ""),
        "step": 0,
        "reasoning_round": reasoning_round,
    })
    return {
        "plan": plan,
        "plan_committed": True,
        "reasoning_round": reasoning_round,
    }


def _after_plan(state: AgentState) -> str:
    """plan_execute 直接执行已提交计划（required 由校验器补齐检索步骤）；
    direct/react 的 required 兜底先预检索一次。"""
    if Strategy(state["strategy"]) is Strategy.PLAN_EXECUTE:
        return "execute"
    is_required = state.get("retrieval_policy") == "required" and state.get("answer_mode") == "novel_evidence"
    return "retrieve" if is_required else "execute"


async def _retrieve_node(state: AgentState) -> dict[str, Any]:
    """required 兜底路径的共享预检索：后续循环不再强制重复调用。"""
    call_id = "agent-retrieve"
    await _emit(state, "step_start", {"step": 1, "action": "retrieve_novel", "purpose": "召回共享小说证据"})
    await _emit(state, "tool_start", {"id": call_id, "tool": "retrieve_novel", "label": "小说证据检索", "step": 1})
    result = await registry.execute(
        "retrieve_novel",
        allowed_tools=state["allowed_tools"],
        query=state["standalone_query"],
        retrieval_query=state.get("retrieval_query") or state["standalone_query"],
        file_id=state.get("file_id"),
    )
    raw = result.output.get("evidence", []) if result.status == "ok" and isinstance(result.output, dict) else []
    evidence, sources = _normalize_evidence(raw)
    observation = result.as_dict()
    observation["query"] = state["standalone_query"]
    await _emit(state, "observation", {"step": 1, **observation})
    await _emit(state, "tool_end", {
        "id": call_id,
        "tool": "retrieve_novel",
        "label": "小说证据检索",
        "step": 1,
        "status": result.status,
        "summary": result.error_code or f"召回 {len(sources)} 条证据，耗时 {result.latency_ms}ms",
    })
    await _emit(state, "sources", sources)
    return {
        "evidence": evidence,
        "sources": sources,
        "observations": [observation],
        "current_step": 1,
        "tool_calls_used": 1,
        "rag_called": True,
        "rag_call_count": 1,
        "fallback_reason": "" if evidence else (result.error_code or "empty_retrieval"),
    }


def _after_retrieve(state: AgentState) -> str:
    """预检索完成后进入同一决策循环。"""
    return "execute"


def _react_system_prompt(state: AgentState) -> str:
    """决策循环的系统提示：按交互模式与策略给出行动边界。"""
    if state.get("interaction_mode") == "roleplay":
        cards = state.get("character_cards") or []
        if not cards:
            return (
                "你是角色扮演编排助手。第一步必须调用 load_character_context 装载登场角色卡，"
                "之后以角色身份直接回复访客。不要输出答案之外的解释。"
            )
        return world_service.roleplay_system_prompt(cards, state.get("chapter_until"))
    parts = [
        "你是小说阅读助手 Agent，自主决定下一步行动。",
        "调用工具的条件：",
        "- 问题涉及小说人物、关系、情节、时间线、章节、伏笔、动机或原文核验，且当前上下文没有足够证据；",
        "- 问题包含跨轮小说指代，会话记忆不足以可靠回答；",
        "- 对关键小说事实不确定，而检索能够消除该不确定性；",
        "- 已有命中不足以理解前因后果时调用 get_chapter_context；",
        "- 需要时间跨度、数量等确定性计算时调用 calculator。",
        "不调用工具、直接回答的条件：",
        "- 问候、闲聊、感谢、输出偏好或记忆操作；",
        "- 仅需处理用户本轮提供的完整文本（改写、总结、分析）；",
        "- 当前上下文与已有工具结果已足以回答；",
        "- 与小说内容无关的一般问题。",
        "约束：",
        "- 「不展示原文」只影响最终输出策略，不禁止你检索；",
        "- 检索无结果或失败时，不得以常识、会话记忆或猜测冒充小说事实；",
        "- 工具结果只是证据，最终回答由汇总节点生成，不要在决策阶段写答案；",
        "- 证据足以回答时立即结束（不调用任何工具）；不要重复调用已覆盖相同内容的检索。",
    ]
    if Strategy(state["strategy"]) is Strategy.DIRECT:
        parts.append("- 本轮为短路径模式：优先直接回答，仅在明显缺少小说事实时做一次检索。")
    return "\n".join(parts)


def _react_initial_human(state: AgentState) -> str:
    """决策循环的首条用户消息：问题 + 查询准备建议 + 记忆/历史上下文。"""
    original = state.get("original_query", "")
    if state.get("interaction_mode") == "roleplay":
        history_text = world_service.format_roleplay_history(state.get("roleplay_history"))
        if state.get("chapter_until"):
            boundary = f"剧情时间线：故事进行到第 {state.get('chapter_until')} 章，之后情节尚未发生。"
        else:
            boundary = "剧情时间线：全书完结后的世界。"
        lines = []
        if history_text:
            lines.append(f"【此前对话】\n{history_text}")
        lines.append(f"{boundary}\n访客说：{original or state.get('standalone_query', '')}\n请以角色身份继续对话。")
        return "\n\n".join(lines)
    lines = [f"用户问题：{original or state.get('standalone_query', '')}"]
    prep = state.get("query_preparation") or {}
    if prep.get("needs_retrieval"):
        reason = prep.get("retrieval_reason") or "novel_evidence"
        lines.append(f"查询准备建议：该问题可能需要小说原文检索（{reason}），请优先考虑 retrieve_novel。")
    memories = (state.get("memory_context") or {}).get("memories") or []
    memory_lines = [f"- {m.get('content', '')}" for m in memories[:5] if m.get("content")]
    if memory_lines:
        lines.append("长期记忆（仅供理解上下文，不是小说原文证据）：\n" + "\n".join(memory_lines))
    if state.get("retrieval_policy") == "required":
        lines.append("系统保守兜底判定：该问题需要小说证据，请先调用 retrieve_novel。")
    return "\n\n".join(lines)


def _parse_plan(text: str) -> list[Any]:
    """解析规划模型输出：只接受 JSON 数组，失败返回空列表（由校验器兜底为单 analyze）。

    不再"按行生成检索步骤"——自由文本解析成检索动作正是 plan_execute 沦为
    "检索问题拆分器"的根源。
    """
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I | re.S).strip()
    try:
        payload = json.loads(cleaned)
        return payload if isinstance(payload, list) else []
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", cleaned)
        if match:
            try:
                payload = json.loads(match.group(0))
                return payload if isinstance(payload, list) else []
            except json.JSONDecodeError:
                return []
    return []


def _normalize_depends_on(raw: Any, valid_ids: set[int]) -> list[int]:
    """把 depends_on 归一化为合法 id 列表：引用不存在/自引用/重复一律剔除。"""
    if not isinstance(raw, list):
        return []
    result: list[int] = []
    for item in raw:
        try:
            dep = int(item)
        except (TypeError, ValueError):
            continue
        if dep in valid_ids and dep not in result:
            result.append(dep)
    return result


def _has_dependency_cycle(steps: list[dict[str, Any]]) -> bool:
    """依赖图环检测（deps 只做校验不做调度，但带环的元数据宁可清空）。"""
    graph = {s["id"]: set(s.get("depends_on") or []) for s in steps}
    visited: set[int] = set()
    done: set[int] = set()

    def visit(node: int) -> bool:
        if node in done:
            return False
        if node in visited:
            return True
        visited.add(node)
        if any(visit(dep) for dep in graph.get(node, ()) if dep in graph):
            return True
        done.add(node)
        return False

    return any(visit(node) for node in graph if node not in done)


def _validate_plan(
    raw_steps: list[Any],
    *,
    retrieval_policy: str,
    standalone_query: str,
) -> tuple[list[dict[str, Any]], bool]:
    """计划提交前的确定性校验：只拒绝或跳过，不静默改写步骤语义。

    - 非法 action / 工具步骤缺 query → invalid（不能静默转成 retrieve_novel）；
    - forbidden 策略 → 检索步骤 skipped(reason=forbidden_policy)；
    - 相同 (action, query) 去重；检索步骤数量不设额外上限——由全局
      agent_max_retrieval_calls 预算在执行期硬闸；
    - required 且无有效检索步骤 → 自动插入最小 retrieve；
    - 解析失败 → 单步 analyze 兜底（required 再补检索）。
    返回 (校验后步骤, 是否走了兜底)。
    """
    if not isinstance(raw_steps, list) or not raw_steps:
        steps: list[dict[str, Any]] = [{
            "id": 1, "action": "analyze", "tool": None, "query": None,
            "objective": "根据当前用户输入和已有上下文完成任务",
            "depends_on": [], "status": "delegated",
        }]
        if retrieval_policy == "required":
            steps.insert(0, {
                "id": 1, "action": "retrieve", "tool": "retrieve_novel",
                "query": standalone_query[:120],
                "objective": "根据用户问题检索小说原文证据",
                "depends_on": [], "status": "pending",
            })
        return [{**s, "id": i + 1} for i, s in enumerate(steps)], True

    validated: list[dict[str, Any]] = []
    seen_signatures: set[tuple[str, str]] = set()
    for raw in raw_steps[:_PLAN_MAX_STEPS]:
        if not isinstance(raw, dict):
            continue
        action = str(raw.get("action") or "").strip().lower()
        objective = str(raw.get("objective") or raw.get("purpose") or "").strip()[:200]
        query = str(raw.get("query") or "").strip()[:120]
        tool = _PLAN_TOOL_ACTIONS.get(action)
        raw_depends = raw.get("depends_on")
        if action not in _PLAN_ALL_ACTIONS:
            validated.append({"action": action, "objective": objective,
                              "tool": tool, "query": query or None,
                              "depends_on": raw_depends if isinstance(raw_depends, list) else [],
                              "status": "invalid", "reason": "invalid_action"})
            continue
        if tool and not query:
            validated.append({"action": action, "objective": objective,
                              "tool": tool, "query": None,
                              "depends_on": raw_depends if isinstance(raw_depends, list) else [],
                              "status": "invalid", "reason": "missing_query"})
            continue
        if tool and retrieval_policy == "forbidden":
            validated.append({"action": action, "objective": objective,
                              "tool": tool, "query": query,
                              "depends_on": raw_depends if isinstance(raw_depends, list) else [],
                              "status": "skipped", "reason": "forbidden_policy"})
            continue
        signature = (action, query) if tool else None
        if signature and signature in seen_signatures:
            validated.append({"action": action, "objective": objective,
                              "tool": tool, "query": query,
                              "depends_on": raw_depends if isinstance(raw_depends, list) else [],
                              "status": "skipped", "reason": "duplicate_query"})
            continue
        if signature:
            seen_signatures.add(signature)
        validated.append({
            "action": action,
            "objective": objective or ("根据用户问题检索小说原文证据" if tool else "完成该计划目标"),
            "tool": tool,
            "query": query or None,
            "depends_on": raw_depends if isinstance(raw_depends, list) else [],
            "status": "pending" if tool else "delegated",
        })

    # 依赖元数据清洗：模型按自己的输出序号声明 depends_on，与重编号后的 id 同序
    # 近似对齐；引用越界/自引用剔除，检测到环则整体清空（deps 不参与调度）。
    ids = set(range(1, len(validated) + 1))
    for idx, step in enumerate(validated):
        step["depends_on"] = _normalize_depends_on(step.get("depends_on"), ids - {idx + 1})
    if _has_dependency_cycle([{"id": i + 1, "depends_on": s["depends_on"]} for i, s in enumerate(validated)]):
        log.warning("plan_validate.dependency_cycle_cleared", steps=len(validated))
        for s in validated:
            s["depends_on"] = []

    # required 策略：计划里没有任何有效检索步骤时，系统自动补一个最小检索。
    has_retrieval = any(
        s["action"] in _PLAN_TOOL_ACTIONS and s["status"] == "pending"
        for s in validated
    )
    if retrieval_policy == "required" and not has_retrieval:
        validated.insert(0, {
            "action": "retrieve", "tool": "retrieve_novel",
            "query": standalone_query[:120],
            "objective": "根据用户问题检索小说原文证据",
            "depends_on": [], "status": "pending",
        })
    return [{**s, "id": i + 1} for i, s in enumerate(validated)], False


async def _execute_plan_steps(state: AgentState, steps: list[dict[str, Any]]) -> dict[str, Any]:
    """执行已提交计划中的工具步骤：信号量限流并发，非工具步骤不经过这里。

    只有校验后 status=pending 的工具步骤（retrieve/chapter_context/calculate）
    会真实调用 registry；分析/比较/总结已标 delegated 委托给 compose。
    结果以 ToolMessage 回灌，补充决策仍归模型（react 循环继续）。
    """
    budget = ExecutionBudget.from_dict(state.get("budget"))
    current_step = int(state.get("current_step", 0))
    tool_calls_used = int(state.get("tool_calls_used", 0))
    rag_count = int(state.get("rag_call_count", 0))
    reasoning_round = int(state.get("reasoning_round", 0))
    evidence = list(state.get("evidence", []))
    observations = list(state.get("observations", []))
    react_messages = list(state.get("react_messages", []))
    fallback = state.get("fallback_reason", "")

    runnable: list[dict[str, Any]] = []
    for step in steps:
        if step.get("status") != "pending":
            continue                    # delegated/invalid/skipped 由校验器定案，执行环不碰
        if current_step + len(runnable) >= budget.max_steps:
            step["status"] = "skipped"; step["reason"] = "exceeds_step_budget"
            continue
        if tool_calls_used + len(runnable) >= budget.max_tool_calls:
            step["status"] = "skipped"; step["reason"] = "exceeds_tool_budget"
            continue
        retrieval_runnable = sum(1 for r in runnable if r["tool"] in _RETRIEVAL_TOOLS)
        if step["tool"] in _RETRIEVAL_TOOLS and rag_count + retrieval_runnable >= budget.max_retrieval_calls:
            step["status"] = "skipped"; step["reason"] = "exceeds_retrieval_budget"
            continue
        runnable.append(step)

    semaphore = asyncio.Semaphore(settings.agent_plan_max_concurrency)

    async def _run_one(index: int, step: dict[str, Any]) -> tuple[int, str, ToolResult]:
        call_id = f"plan-step-{step['id']}"
        label = REACT_TOOL_LABELS.get(step["tool"], step["action"])
        await _emit(state, "tool_start", {
            "id": call_id, "tool": step["tool"], "action": step["action"], "label": label,
            "step": current_step + index + 1,       # 兼容字段（内部步号）
            "plan_step_id": step["id"],             # 计划项编号：前端展示"计划项 N"
            "tool_call_index": tool_calls_used + index + 1,  # 全局工具调用序号
        })
        async with semaphore:
            if step["tool"] == "calculator":
                result = await registry.execute(
                    "calculator", allowed_tools=state["allowed_tools"],
                    expression=str(step.get("query") or ""),
                )
            else:
                result = await registry.execute(
                    step["tool"],
                    allowed_tools=state["allowed_tools"],
                    query=str(step.get("query") or ""),
                    retrieval_query=str(step.get("query") or ""),
                    file_id=state.get("file_id"),
                    chapter_until=state.get("chapter_until"),
                )
        return index, call_id, result

    if runnable:
        await _emit(state, "step_start", {
            "step": current_step + 1,
            "action": "execute_plan",
            "purpose": f"并行执行 {len(runnable)} 个计划工具步骤",
        })
        gathered = await asyncio.gather(*[
            _run_one(index, step) for index, step in enumerate(runnable)
        ], return_exceptions=True)

        executed: list[tuple[dict[str, Any], str, ToolResult]] = []
        for entry in gathered:
            if isinstance(entry, BaseException):
                log.warning("plan_step.failed", error=str(entry)[:200])
                continue
            index, call_id, result = entry
            step = runnable[index]
            step["status"] = "done" if result.status == "ok" else "failed"
            label = REACT_TOOL_LABELS.get(step["tool"], step["action"])
            observation = result.as_dict()
            observation["query"] = step["query"]
            observation["plan_step_id"] = step["id"]
            observation["action"] = step["action"]
            observation["tool_call_index"] = tool_calls_used + 1
            observation["tool_called"] = True
            observation["retrieval_used"] = step["tool"] in _RETRIEVAL_TOOLS
            observations.append(observation)
            if result.status == "ok" and isinstance(result.output, dict):
                evidence.extend(result.output.get("evidence", []))
            else:
                fallback = fallback or result.error_code or "plan_step_failed"
            if step["tool"] in _RETRIEVAL_TOOLS:
                rag_count += 1
            tool_calls_used += 1
            current_step += 1
            await _emit(state, "tool_end", {
                "id": call_id,
                "tool": step["tool"],
                "label": label,
                "step": current_step,
                "plan_step_id": step["id"],
                "tool_call_index": tool_calls_used,
                "status": result.status,
                "summary": result.error_code or f"完成，耗时 {result.latency_ms}ms",
            })
            executed.append((step, call_id, result))

        if executed:
            await _emit(state, "agent_decision", {
                "action": "plan_execute",
                "reason": f"已并行执行 {len(executed)} 个计划步骤",
                "step": current_step,
                # 并行步骤不新开推理轮次：沿用产出计划的那一轮。
                "reasoning_round": reasoning_round,
            })
            threaded_calls = [
                {"name": step["tool"],
                 "args": {"expression": step["query"]} if step["tool"] == "calculator" else {"query": step["query"]},
                 "id": call_id}
                for step, call_id, _ in executed
            ]
            threaded = AIMessage(content="已按计划并行执行工具步骤。", tool_calls=threaded_calls)
            react_messages = react_messages + [threaded] + [
                ToolMessage(content=_react_payload(result), tool_call_id=call_id)
                for _, call_id, result in executed
            ]

    normalized, sources = _normalize_evidence(evidence)
    normalized = normalized[:_REACT_MAX_EVIDENCE]
    sources = sources[:len(normalized)]
    await _emit(state, "sources", sources)
    updated_plan = [
        {**step, "status": step.get("status", "pending")}
        for step in state.get("plan", [])
    ]
    # 计划项状态变化只发生在图状态里：重发一次 plan 事件让前端快照刷新，
    # 否则工具步骤永远显示提交时刻的"待执行"。
    await _emit(state, "plan", {
        "steps": updated_plan,
        "max_steps": budget.max_steps,
        "retrieval_policy": state.get("retrieval_policy", "optional"),
    })
    return {
        "evidence": normalized,
        "sources": sources,
        "observations": observations,
        "current_step": current_step,
        "tool_calls_used": tool_calls_used,
        # 并行步骤不消耗推理轮次：原样透传，供后续决策轮递增。
        "reasoning_round": reasoning_round,
        "rag_called": rag_count > 0 or bool(state.get("rag_called")),
        "rag_call_count": rag_count,
        "react_messages": react_messages,
        "plan": updated_plan,
        "fallback_reason": fallback if not normalized else "",
        "react_done": False,
        "stop_reason": "",
    }


async def _execute_node(state: AgentState) -> dict[str, Any]:
    """Agent 决策循环的单步：observe → decide（模型选工具/直答/计划）→ act。

    react/direct/plan_execute/roleplay 共用本节点。工具结果经 ToolMessage 回灌
    （react_messages 跨轮保留），模型每轮都能看到此前全部工具结果。
    """
    strategy = Strategy(state["strategy"])
    interaction_mode = state.get("interaction_mode", "qa")

    # plan_execute：计划已提交且存在待执行步骤 → 并行执行计划步骤。
    if strategy is Strategy.PLAN_EXECUTE and state.get("plan_committed"):
        pending = [step for step in state.get("plan", []) if step.get("status") == "pending"]
        if pending:
            return await _execute_plan_steps(state, pending)

    evidence = list(state.get("evidence", []))
    observations = list(state.get("observations", []))
    current_step = int(state.get("current_step", 0))
    fallback = state.get("fallback_reason", "")
    budget = ExecutionBudget.from_dict(state.get("budget"))
    tool_calls_used = int(state.get("tool_calls_used", 0))
    rag_count = int(state.get("rag_call_count", 0))
    react_messages = list(state.get("react_messages", []))

    if current_step >= budget.max_steps:
        return {
            "current_step": current_step,
            "fallback_reason": fallback or "step_budget_exceeded",
            "react_done": True,
            "stop_reason": fallback or "step_budget_exceeded",
            "react_messages": react_messages,
        }
    if tool_calls_used >= budget.max_tool_calls:
        return {
            "current_step": current_step,
            "fallback_reason": fallback or "tool_budget_exceeded",
            "react_done": True,
            "stop_reason": "tool_budget_exceeded",
            "react_messages": react_messages,
        }

    # plan_execute 的计划由 _plan_node 提交；执行环只执行计划，不再生成计划。
    if strategy is Strategy.PLAN_EXECUTE and not state.get("plan_committed"):
        # 防御分支：正常路径不可达（计划节点必定提交）。兜底为直接汇总。
        return {
            "current_step": current_step,
            "reasoning_round": int(state.get("reasoning_round", 0)),
            "react_done": True,
            "stop_reason": "empty_plan",
            "fallback_reason": "empty_plan",
            "react_messages": react_messages,
        }

    if not react_messages:
        react_messages = [HumanMessage(content=_react_initial_human(state))]

    # 主 Agent 推理轮次：每个决策轮 +1，与工具调用计数/计划步骤编号语义分离。
    reasoning_round = int(state.get("reasoning_round", 0)) + 1
    system = _react_system_prompt(state)
    messages = [SystemMessage(content=system)] + react_messages
    thinking = _ThinkingSpan(
        state,
        stream="main_agent",
        span_id=f"agent-round-{reasoning_round}",
        phase="decide",
        reasoning_round=reasoning_round,
    )
    builder = ModelTurnBuilder()
    purpose = LLMPurpose.AGENT_DECISION
    try:
        async for delta in astream_model_turn(
            messages, purpose,
            tools=model_tool_specs(state["allowed_tools"]),
            max_tokens=700 if interaction_mode == "roleplay" else 300,
        ):
            if delta.kind == "reasoning":
                await thinking.token(delta.text)
                builder.add_delta(delta)
            else:
                builder.add_delta(delta)
        await thinking.end("completed")
    except asyncio.CancelledError:
        await thinking.end("cancelled")
        raise
    except Exception:
        await thinking.end("error")
        raise
    turn = builder.build()
    calls = turn.tool_calls
    content_text = turn.content

    if not calls:
        # 模型不再调用工具即判定上下文足够。角色扮演的最终内容就是候选答案本身
        #（单一生成，不二次汇总）；其余策略交给 compose 节点生成。
        await _emit(state, "agent_decision", {
            "action": "answer",
            "reason": (content_text.strip()[:120] or "当前上下文足以回答"),
            "step": current_step,
            "reasoning_round": reasoning_round,
        })
        update: dict[str, Any] = {
            "current_step": current_step,
            "reasoning_round": reasoning_round,
            "react_done": True,
            "stop_reason": fallback or "model_answer",
            "react_messages": react_messages,
            "tool_calls_used": tool_calls_used,
            "rag_called": bool(state.get("rag_called")),
            "rag_call_count": rag_count,
        }
        if interaction_mode == "roleplay" and content_text.strip():
            update["answer"] = content_text.strip()
        return update

    await _emit(state, "agent_decision", {
        "action": "tool_call",
        # calls 是 llm.ToolCall 数据类：必须属性访问，字典 .get 会在"无正文纯工具调用"时崩溃。
        "reason": (content_text.strip()[:120] or "、".join(call.name for call in calls[:2])),
        "step": current_step + 1,
        "reasoning_round": reasoning_round,
    })

    # 单轮最多执行 min(2, 剩余步数, 剩余总调用) 个工具调用，防止单轮爆发。
    budget_this_turn = min(
        2,
        budget.max_steps - current_step,
        budget.max_tool_calls - tool_calls_used,
    )
    prior_ok_queries = {
        (obs.get("tool"), obs.get("query"))
        for obs in observations
        if obs.get("status") == "ok" and obs.get("query")
    }
    executed_calls: list[dict[str, Any]] = []
    executed_results: list[ToolResult] = []
    for call in calls:
        if len(executed_calls) >= budget_this_turn:
            break
        name = call.name
        call_id = call.id or f"agent-step-{current_step + len(executed_calls) + 1}"
        # 模型只提供业务参数（query/expression）；检索范围（file_id）与角色扮演
        # 上下文（personas/chapter_until）由状态注入，不进模型 schema。
        kwargs = {k: v for k, v in call.args.items() if k in {"query", "expression"}}
        if name in _RETRIEVAL_TOOLS and not kwargs.get("query"):
            kwargs["query"] = state["standalone_query"]
        # 预算硬闸：检索调用超限或与已成功检索同参时，不再执行，直接回灌拒绝。
        if name in _RETRIEVAL_TOOLS:
            ok_retrievals_this_turn = len([
                r for r in executed_results
                if r.tool in _RETRIEVAL_TOOLS and r.status == "ok"
            ])
            if rag_count + ok_retrievals_this_turn >= budget.max_retrieval_calls:
                executed_calls.append({"id": call_id, "name": name, "args": dict(call.args)})
                executed_results.append(ToolResult(status="denied", error_code="retrieval_budget_exceeded", tool=name))
                continue
            signature = (name, kwargs.get("query") or "")
            if signature in prior_ok_queries:
                executed_calls.append({"id": call_id, "name": name, "args": dict(call.args)})
                executed_results.append(ToolResult(status="denied", error_code="duplicate_retrieval", tool=name))
                continue
        current_step += 1
        tool_calls_used += 1
        await _emit(state, "step_start", {
            "step": current_step, "action": name,
            "purpose": REACT_TOOL_LABELS.get(name, name),
            "tool_call_index": tool_calls_used,
        })
        await _emit(state, "tool_start", {
            "id": call_id, "tool": name,
            "label": REACT_TOOL_LABELS.get(name, name),
            "step": current_step,
            "tool_call_index": tool_calls_used,
        })

        # 按工具名显式分发：受信任上下文（file_id/chapter_until）由状态注入，
        # 模型参数逐个透传，不做动态 kwargs 展开。
        if name == "retrieve_novel":
            def _invoke():
                return registry.execute(
                    "retrieve_novel", allowed_tools=state["allowed_tools"],
                    query=str(kwargs.get("query", "")),
                    retrieval_query=str(kwargs.get("query", "")),
                    file_id=state.get("file_id"),
                    chapter_until=state.get("chapter_until"),
                )
        elif name == "get_chapter_context":
            def _invoke():
                return registry.execute(
                    "get_chapter_context", allowed_tools=state["allowed_tools"],
                    query=str(kwargs.get("query", "")),
                    file_id=state.get("file_id"),
                    chapter_until=state.get("chapter_until"),
                )
        elif name == "calculator":
            def _invoke():
                return registry.execute(
                    "calculator", allowed_tools=state["allowed_tools"],
                    expression=str(kwargs.get("expression", "")),
                )
        elif name == "load_character_context":
            def _invoke():
                return registry.execute(
                    "load_character_context", allowed_tools=state["allowed_tools"],
                    file_id=state.get("file_id"),
                    personas=list(state.get("personas") or []),
                    chapter_until=state.get("chapter_until"),
                )
        else:
            async def _invoke():
                return ToolResult(status="denied", error_code="tool_not_allowed", tool=name)

        spec = registry.spec(name)
        result = await _invoke()
        attempt = 0
        while result.status in {"timeout", "error"} and spec is not None and spec.retryable and attempt < 1:
            attempt += 1
            log.info("tool.retry", tool=name, attempt=attempt, previous=result.error_code)
            result = await _invoke()
        observation = result.as_dict()
        observation["query"] = kwargs.get("query")
        observations.append(observation)
        if name in _RETRIEVAL_TOOLS:
            rag_count += 1
            if result.status == "ok":
                prior_ok_queries.add((name, kwargs.get("query") or ""))
        if name == "load_character_context" and result.status == "ok" and isinstance(result.output, dict):
            # 角色卡进入受信任状态：compose 与校验节点都用它，不依赖模型转述。
            cards_loaded = result.output
            await _emit(state, "observation", {
                "step": current_step, "tool_call_index": tool_calls_used,
                "cards_loaded": len(cards_loaded.get("cards") or []),
            })
        else:
            await _emit(state, "observation", {"step": current_step, "tool_call_index": tool_calls_used, **observation})
            if result.status == "ok" and isinstance(result.output, dict):
                evidence.extend(result.output.get("evidence", []))
        await _emit(state, "tool_end", {
            "id": call_id,
            "tool": name,
            "label": REACT_TOOL_LABELS.get(name, name),
            "step": current_step,
            "tool_call_index": tool_calls_used,
            "status": result.status,
            "summary": result.error_code or f"完成，耗时 {result.latency_ms}ms",
        })
        executed_calls.append({"id": call_id, "name": name, "args": dict(call.args)})
        executed_results.append(result)
        if result.status not in {"ok", "denied"}:
            fallback = result.error_code or "tool_failed"

    # 工具结果以 ToolMessage 回灌；只保留实际执行过的调用，保证消息序列合法。
    threaded_response = AIMessage(content=content_text, tool_calls=executed_calls)
    react_messages = react_messages + [threaded_response] + [
        ToolMessage(content=_react_payload(result), tool_call_id=call_spec["id"])
        for call_spec, result in zip(executed_calls, executed_results)
    ]

    normalized, sources = _normalize_evidence(evidence)
    # _normalize_evidence 中 evidence 与 sources 一一对应，同步截断保持一致。
    normalized = normalized[:_REACT_MAX_EVIDENCE]
    sources = sources[:len(normalized)]
    if sources != state.get("sources", []):
        await _emit(state, "sources", sources)
    update: dict[str, Any] = {
        "evidence": normalized,
        "sources": sources,
        "observations": observations,
        "current_step": current_step,
        "tool_calls_used": tool_calls_used,
        "reasoning_round": reasoning_round,
        "fallback_reason": fallback,
        "react_done": False,
        "react_messages": react_messages,
        "rag_called": rag_count > 0 or bool(state.get("rag_called")),
        "rag_call_count": rag_count,
        "stop_reason": "",
    }
    if interaction_mode == "roleplay" and any(call["name"] == "load_character_context" for call in executed_calls):
        # 角色卡装载成功后写入状态（cards 从最后一条 ok 结果提取）。
        for result in reversed(executed_results):
            if result.tool == "load_character_context" and result.status == "ok" and isinstance(result.output, dict):
                update["character_cards"] = result.output.get("cards") or []
                update["scenario"] = result.output.get("scenario", "")
                break
    return update


async def _reflect_node(state: AgentState) -> dict[str, Any]:
    """评估证据与预算，决定继续执行还是进入汇总；本节点驱动条件回边。"""
    if state.get("react_done"):
        decision, reason = "final", "模型判定证据足以回答"
    elif state.get("fallback_reason"):
        decision, reason = "final", state["fallback_reason"]
    elif state.get("current_step", 0) >= state.get("max_steps", 0):
        decision, reason = "final", "step_budget_exceeded"
    else:
        decision, reason = "continue", "证据或计划尚未完成，继续执行"
    # reflection 跟随刚结束的决策轮：编号用轮次，不用会把并行步骤算进来的步号。
    await _emit(state, "reflection", {
        "decision": decision, "reason": reason,
        "step": state.get("current_step", 0),
        "reasoning_round": state.get("reasoning_round", 0),
    })
    return {}


def _after_reflect(state: AgentState) -> str:
    """ReAct 条件回边：未终止且预算未耗尽时回到 execute 继续执行。"""
    if state.get("react_done") or state.get("fallback_reason"):
        return "supervisor"
    if state.get("current_step", 0) >= state.get("max_steps", 0):
        return "supervisor"
    return "execute"


async def _memory_agent_node(state: AgentState) -> dict[str, Any]:
    """模型自主发起的记忆维护：判断本轮是否需要新增、更新或遗忘长期记忆。

    记忆工具（search/add/update/delete）通过 bind_tools 交给模型，模型可多轮
    调用；执行走与 retrieve_novel 相同的 registry 通道（白名单为记忆工具集）。
    """
    step = state.get("current_step", 0) + 1
    await _emit(state, "step_start", {
        "step": step,
        "action": "memory_agent",
        "purpose": "判断是否需要记录、更新或遗忘记忆",
    })
    # 记忆工具需要会话上下文（session_id/file_id），通过 ContextVar 注入。
    session_id = state.get("session_id")
    if session_id:
        set_memory_session(session_id, state.get("file_id"))
    if get_memory_session() is None:
        return {"memory_ops": [], "current_step": step}

    context = state.get("synthesis_context") or {}
    memories = context.get("memories") or []
    existing = "\n".join(
        f"- id={m.get('id')} [{m.get('memory_type')}] {m.get('content', '')}"
        for m in memories if m.get("content")
    ) or "（暂无长期记忆）"
    system = (
        "你是记忆维护助手。根据本轮对话判断是否需要操作用户的长期记忆：\n"
        "- 用户表达了稳定偏好或重要事实 → add_memory（memory_type："
        "user_preference=用户偏好；novel_fact=对咨询这本小说有用的事实；session_fact=仅本会话使用的事实）；\n"
        "- 本轮信息与现有记忆矛盾或需要细化 → update_memory（先 search_memories 拿 id）；\n"
        "- 用户明确要求忘记或撤回 → delete_memory；\n"
        "- 不确定时先 search_memories 查看现有记忆再决定，避免重复记录；\n"
        "- 普通闲聊、一次性问题不需要记忆：直接回复「无需操作」，不调用任何工具。\n"
        "最多进行 3 次工具调用，完成后用一句话总结做了什么（或说明无需操作）。"
    )
    user = (
        f"本轮用户说：{state.get('original_query', '')}\n"
        f"助手即将回答的问题：{state.get('standalone_query', '')}\n"
        f"现有长期记忆：\n{existing}"
    )
    llm = get_llm(temperature=0, max_tokens=500).bind_tools(list(MEMORY_AGENT_TOOL_SPECS))
    messages: list[Any] = [SystemMessage(content=system), HumanMessage(content=user)]
    ops: list[dict[str, Any]] = []
    for _ in range(3):
        response = await llm.ainvoke(messages)
        calls = getattr(response, "tool_calls", None) or []
        if not calls:
            break
        messages.append(response)
        for call in calls[:3]:
            name = str(tool_call_field(call, "name") or "")
            args = tool_call_field(call, "args") or {}
            call_id = tool_call_field(call, "id") or f"mem-{len(ops) + 1}"
            await _emit(state, "tool_start", {"id": call_id, "tool": name, "label": "记忆操作"})
            # 按工具名显式分发：模型参数逐个透传，不做动态 kwargs 展开。
            if name == "search_memories":
                result = await registry.execute(
                    "search_memories", allowed_tools=MEMORY_AGENT_TOOLS,
                    query=str(args.get("query", "")),
                )
            elif name == "add_memory":
                result = await registry.execute(
                    "add_memory", allowed_tools=MEMORY_AGENT_TOOLS,
                    content=str(args.get("content", "")),
                    memory_type=str(args.get("memory_type", "session_fact")),
                    importance=float(args.get("importance", 0.7)),
                )
            elif name == "update_memory":
                result = await registry.execute(
                    "update_memory", allowed_tools=MEMORY_AGENT_TOOLS,
                    memory_id=str(args.get("memory_id", "")),
                    content=str(args.get("content", "")),
                )
            elif name == "delete_memory":
                result = await registry.execute(
                    "delete_memory", allowed_tools=MEMORY_AGENT_TOOLS,
                    memory_id=str(args.get("memory_id", "")),
                )
            else:
                result = ToolResult(status="denied", error_code="tool_not_allowed", tool=name)
            # 前端只展示操作语义，不展示记忆内容与参数，避免把用户信息铺在界面上。
            if name == "search_memories":
                brief = "已核对现有记忆"
            elif name == "add_memory":
                brief = "已提交记忆写入（后台生效）"
            elif name == "update_memory":
                brief = "已提交记忆更新（后台生效）"
            elif name == "delete_memory":
                brief = "已提交记忆删除（后台生效）"
            else:
                brief = "记忆操作完成"
            summary = brief if result.status == "ok" else "操作未完成"
            ops.append({"tool": name, "status": result.status, "summary": summary})
            await _emit(state, "tool_end", {
                "id": call_id, "tool": name, "status": result.status, "summary": summary,
            })
            messages.append(ToolMessage(
                content=json.dumps(result.as_dict(), ensure_ascii=False),
                tool_call_id=call_id,
            ))
    if ops:
        log.info("chat.memory_agent_ops", session_id=state.get("session_id"), ops=len(ops))
    return {"memory_ops": ops, "current_step": step}


async def _supervisor_node(state: AgentState) -> dict[str, Any]:
    """内部 Supervisor：验证证据、确定实际回答模式并准备 compose 上下文。"""
    step = state.get("current_step", 0) + 1
    await _emit(state, "step_start", {
        "step": step,
        "action": "supervisor",
        "purpose": "内部验证与冲突消解",
    })
    evidence = state.get("evidence", [])
    fallback = state.get("fallback_reason", "")
    rag_called = bool(state.get("rag_called"))
    answer_mode = state.get("answer_mode", "novel_evidence")
    if state.get("answer_mode", "novel_evidence") == "novel_evidence" and not evidence and rag_called:
        fallback = fallback or "empty_retrieval"
    # Agent-first 语义：实际执行过 RAG 才走原文证据汇总；模型跳过检索时按
    # 路由判定的会话/记忆模式回答（compose 侧对"建议检索却未检索"加强提示）。
    if state.get("interaction_mode") == "roleplay":
        effective_answer_mode = "roleplay"
    elif rag_called:
        effective_answer_mode = "novel_evidence"
    else:
        effective_answer_mode = answer_mode if answer_mode in {"memory_context", "conversation"} else "conversation"
    plan_steps = state.get("plan", [])
    synthesis_context = {
        "question": state.get("standalone_query", ""),
        "evidence": evidence,
        "sources": state.get("sources", []),
        "observations": state.get("observations", []),
        "scenario": state.get("scenario", ""),
        "summary": (state.get("memory_context") or {}).get("summary", ""),
        "memories": (state.get("memory_context") or {}).get("memories", []),
        "effective_answer_mode": effective_answer_mode,
        "route_suggested_retrieval": bool(state.get("needs_retrieval")) and not rag_called,
        # 任务计划消费：delegated 步骤的目标交给 compose 完成（无需再调工具）。
        "task_objectives": [
            s.get("objective") for s in plan_steps
            if s.get("status") == "delegated" and s.get("objective")
        ],
        "completed_plan_steps": [s for s in plan_steps if s.get("status") == "done"],
        "failed_plan_steps": [s for s in plan_steps if s.get("status") in {"failed", "invalid"}],
        "retrieval_used": rag_called,
        "retrieval_count": int(state.get("rag_call_count", 0)),
    }
    return {
        "synthesis_context": synthesis_context,
        "fallback_reason": fallback,
        "current_step": step,
    }


def _after_supervisor(state: AgentState) -> str:
    """Supervisor 之后先让模型自主维护记忆（可跳过），再生成候选答案。"""
    if state.get("memory_agent_active"):
        return "memory_agent"
    return "compose_answer"


def _sanitize_summary(text: str, policy: dict[str, Any]) -> str:
    """应用最终输出护栏，避免模型泄漏原文摘录或内部执行说明。"""
    result = text.strip()
    result = re.sub(r"(?im)^\s*(?:根据共享原文(?:与专家报告)?|根据专家报告|根据检索结果)[：:]?\s*", "", result)
    if policy.get("summary_only") or not policy.get("show_source_text"):
        result = re.sub(r"[“\"]([^”\"]{24,})[”\"]", lambda m: m.group(1)[:20] + "……", result)
        result = re.sub(r"(?m)^>\s?", "", result)
    if not policy.get("allow_direct_quotes"):
        result = re.sub(r"[“\"]([^”\"]{1,200})[”\"]", "", result)
    if not policy.get("show_citations") or policy.get("citation_style") == "hidden":
        result = re.sub(r"\s*\[S\d+\]", "", result)
    return re.sub(r"\n{3,}", "\n\n", result).strip()


def _should_buffer(state: AgentState) -> bool:
    """required 策略且开启验证时缓冲候选答案：验证通过后再输出，避免用户看到未验证内容。"""
    return bool(
        settings.answer_verify_enabled
        and state.get("retrieval_policy") == "required"
        and state.get("answer_mode") == "novel_evidence"
    )


async def _compose_node(state: AgentState) -> dict[str, Any]:
    """compose_answer：生成候选答案（不直接作为最终消息输出/持久化）。

    - 角色扮演：执行环的最终回复内容已是答案（单一生成），直接采用；
      为空时用角色卡 + 高温流式补一次生成。
    - required 缓冲路径：完整生成不发 token，等 verify/finalize。
    - direct/optional 流式路径：边生成边发 token（与现有体验一致）。
    """
    step = state.get("current_step", 0) + 1
    await _emit(state, "step_start", {
        "step": step,
        "action": "compose_answer",
        "purpose": "生成候选答案并交由验证节点校验",
    })
    policy = {**DEFAULT_OUTPUT_POLICY, **(state.get("output_policy") or {})}
    context = state.get("synthesis_context") or {}
    evidence = context.get("evidence") or []
    answer_mode = context.get("effective_answer_mode") or state.get("answer_mode", "novel_evidence")
    fallback = state.get("fallback_reason", "")
    buffered = _should_buffer(state)

    # 角色扮演：执行环最终内容直接作为候选答案（不展示 [S#]）。
    if answer_mode == "roleplay":
        candidate = (state.get("answer") or "").strip()
        if candidate:
            candidate = re.sub(r"\s*\[S\d+\]", "", candidate)
            # 执行环阶段不流式：答案在此处一次性下发（角色扮演不缓冲）。
            if not buffered:
                await _emit(state, "token", candidate)
        else:
            cards = state.get("character_cards") or []
            if cards:
                system = world_service.roleplay_system_prompt(cards, state.get("chapter_until"))
            else:
                system = "你正在扮演小说角色与访客交谈，始终保持角色内。"
            parts: list[str] = []
            async for token in _stream_llm([
                SystemMessage(content=system),
                HumanMessage(content=f"访客说：{context.get('question', '')}\n请以角色身份继续对话。"),
            ], settings.agent_synthesis_max_tokens):
                parts.append(token)
                if not buffered:
                    await _emit(state, "token", token)
            candidate = "".join(parts).strip()
        return {
            "candidate_answer": candidate,
            "candidate_streamed": not buffered,
            "fallback_reason": fallback,
            "current_step": step,
        }

    if answer_mode == "novel_evidence" and not evidence:
        # 无证据：固定口径的"证据不足"响应，不进入模型生成。
        if not buffered:
            await _emit(state, "token", _EMPTY_MESSAGE)
        return {
            "candidate_answer": _EMPTY_MESSAGE,
            "candidate_streamed": not buffered,
            "fallback_reason": fallback or "empty_retrieval",
            "current_step": step,
        }

    # 任务计划目标：delegated 步骤（理解/分析/比较/总结）由本节点完成。
    task_objectives = context.get("task_objectives") or []
    objectives_text = "\n".join(f"- {item}" for item in task_objectives) or "（无）"
    objectives_block = (
        f"\n\n工作计划目标（请在答案中依次完成这些分析，无需再调用工具）：\n{objectives_text}"
        if task_objectives else ""
    )

    # 会话/记忆模式：不依赖原文证据，禁止编造小说事实、禁止 [S#]。
    if answer_mode in {"memory_context", "conversation"}:
        memory_text = "\n".join(
            f"- [{item.get('memory_type', 'memory')}] {item.get('content', '')}"
            for item in (context.get("memories") or []) if item.get("content")
        ) or "（无可用长期记忆）"
        summary_text = context.get("summary") or "（无会话摘要）"
        prompt = (
            "当前回答不依赖小说原文检索。请基于会话摘要、长期记忆和用户本轮提供的内容自然回答，"
            "不要编造小说事实，不要生成 [S#]。如果用户是在设置偏好，简洁确认即可。"
            + (objectives_block or (
                "\n\n注意：查询准备判定该问题可能需要小说原文证据，但本轮未执行检索。"
                "回答中涉及小说事实的部分必须明确说明“未核对原文、无法确认”，"
                "严禁凭记忆给出具体情节、数字或引文；如需准确答案请建议用户追问以触发检索。"
            ) if context.get("route_suggested_retrieval") else "")
            + f"\n\n问题：{context.get('question', '')}"
            + f"\n\n会话摘要：\n{summary_text}\n\n长期记忆：\n{memory_text}"
        )
        parts = []
        async for token in _stream_llm([
            SystemMessage(content="你是能够保持会话连续性的小说阅读助手。"),
            HumanMessage(content=prompt),
        ], settings.agent_synthesis_max_tokens):
            parts.append(token)
            if not buffered:
                await _emit(state, "token", token)
        raw = "".join(parts)
        answer = _sanitize_summary(raw, policy)
        if not buffered and answer != raw:
            await _emit(state, "token_replace", answer)
        return {
            "candidate_answer": answer,
            "candidate_streamed": not buffered,
            "fallback_reason": fallback,
            "current_step": step,
        }

    memory_text = "\n".join(
        f"- [{item.get('memory_type', 'memory')}] {item.get('content', '')}"
        for item in (context.get("memories") or []) if item.get("content")
    ) or "（无可用长期记忆）"
    summary_text = context.get("summary") or "（无会话摘要）"
    policy_text = (
        "只输出总结、结论和分析；禁止展示来源片段、复制原文、连续复述人物原话，"
        "禁止以‘根据共享原文’描述内部过程。"
        if policy.get("summary_only") or not policy.get("show_source_text")
        else "按用户本轮要求提供必要的原文依据。"
    )
    if not policy.get("allow_direct_quotes"):
        policy_text += "禁止长引号和直接人物原话。"
    if policy.get("show_citations") and policy.get("citation_style") == "chapter_only":
        policy_text += "出处只保留章节、回目、页码或来源编号，不展示原文片段。"
    elif not policy.get("show_citations"):
        policy_text += "不要输出 [S#] 来源标记。"
    tool_text = "\n".join(
        f"- {obs.get('tool')}: {json.dumps(obs.get('output'), ensure_ascii=False)[:200]}"
        for obs in context.get("observations", [])
        if obs.get("tool") not in (_RETRIEVAL_TOOLS | {
            "specialist", "load_character_context",
            "search_memories", "add_memory", "update_memory", "delete_memory",
        })
        and obs.get("status") == "ok" and obs.get("output") is not None
    ) or "（无）"
    prompt = (
        f"{policy_text}\n请基于经过内部校验的小说证据回答用户问题。事实优先于推断；"
        "若证据不足请明确说明。不要展示内部过程。关键事实必须使用 [S#] 引用对应的证据编号，"
        "不得编造编号。\n\n"
        f"问题：{context.get('question', '')}\n\n共享证据：\n{_evidence_text(evidence)}\n\n"
        f"工具计算结果：\n{tool_text}\n\n"
        f"会话摘要：\n{summary_text}\n\n长期记忆：\n{memory_text}"
        + objectives_block
    )
    system = "你是严谨的小说问答总结助手。"
    if context.get("route_suggested_retrieval"):
        # 路由建议检索而 Agent 未检索：强制声明证据边界，禁止参数记忆冒充原文。
        prompt += (
            "\n\n注意：查询准备判定该问题可能需要小说原文证据，但本轮未执行检索。"
            "回答中涉及小说事实的部分必须明确说明“未核对原文、无法确认”，"
            "严禁凭记忆给出具体情节、数字或引文；如需准确答案请建议用户追问以触发检索。"
        )
    parts = []
    async for token in _stream_llm([
        SystemMessage(content=system),
        HumanMessage(content=prompt),
    ], settings.agent_synthesis_max_tokens):
        parts.append(token)
        if not buffered:
            await _emit(state, "token", token)
    raw = "".join(parts)
    answer = _sanitize_summary(raw, policy)
    if not buffered and answer != raw:
        # 输出护栏净化改变了内容时，以完整净化稿覆盖已流式渲染的内容。
        await _emit(state, "token_replace", answer)
    return {
        "candidate_answer": answer,
        "candidate_streamed": not buffered,
        "fallback_reason": fallback,
        "current_step": step,
    }


_CITATION_RE = re.compile(r"\[(S\d+)\]")


def _deterministic_checks(state: AgentState, candidate: str) -> list[str]:
    """verify_answer 的确定性检查：引用合法性、required 引用门槛、角色上下文。"""
    issues: list[str] = []
    sources = state.get("sources", [])
    valid_ids = {source.get("id") for source in sources}
    cited = _CITATION_RE.findall(candidate)
    invalid = sorted({c for c in cited if c not in valid_ids})
    if invalid:
        issues.append(f"invalid_citations:{','.join(invalid)}")
    if (
        state.get("retrieval_policy") == "required"
        and state.get("answer_mode") == "novel_evidence"
        and sources
        and not any(c in valid_ids for c in cited)
    ):
        issues.append("missing_citation")
    if state.get("interaction_mode") == "roleplay" and not state.get("character_cards"):
        issues.append("missing_character_context")
    return issues


async def _semantic_check(state: AgentState, candidate: str) -> list[str]:
    """轻量语义验证：把答案拆成关键事实，判定是否被引用证据支持。

    返回 unsupported 事实列表。验证器自身失败（解析失败/超时）不阻断答案：
    记录告警后按"无 unsupported"处理，避免验证器故障放大为回答失败。
    """
    evidence = (state.get("synthesis_context") or {}).get("evidence") or []
    if not evidence:
        return []
    evidence_text = _evidence_text(evidence)[:6000]
    answer_text = candidate[:2000]
    prompt = (
        "你是事实核查器。判断候选答案中的每个关键小说事实是否被给定证据支持。\n"
        '只输出 JSON：{"unsupported": ["不被证据支持的事实", ...]}。'
        "推断性表述（合理归纳但无直接证据）不算 unsupported；只有与证据矛盾或证据中"
        "完全不存在依据的具体事实（人名、情节、数字、因果）才算。\n\n"
        f"证据：\n{evidence_text}\n\n候选答案：\n{answer_text}"
    )
    try:
        model = get_llm(temperature=0, max_tokens=400, timeout=settings.llm_timeout, max_retries=0)
        response = await model.ainvoke([{"role": "user", "content": prompt}])
        content = getattr(response, "content", "")
        if isinstance(content, list):
            content = "".join(block.get("text", "") for block in content if isinstance(block, dict))
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(content or "").strip(), flags=re.I | re.S).strip()
        payload = json.loads(cleaned)
        if not isinstance(payload, dict):
            return []
        unsupported = payload.get("unsupported")
        if isinstance(unsupported, list):
            return [str(item)[:120] for item in unsupported if str(item or "").strip()][:5]
    except Exception as exc:  # noqa: BLE001
        log.warning("answer_verify.semantic_failed", error=str(exc)[:200])
    return []


async def _verify_node(state: AgentState) -> dict[str, Any]:
    """verify_answer：确定性检查 +（required 策略）语义支持性检查。

    校验结果只写 state.verification；是否进入 repair 由条件边决定。
    """
    step = state.get("current_step", 0) + 1
    await _emit(state, "step_start", {
        "step": step,
        "action": "verify_answer",
        "purpose": "校验引用与证据支持",
    })
    candidate = state.get("candidate_answer", "")
    issues = _deterministic_checks(state, candidate)
    semantic_unsupported: list[str] = []
    semantic_checked = False
    if (
        settings.answer_verify_enabled
        and not issues
        and state.get("retrieval_policy") == "required"
        and state.get("answer_mode") == "novel_evidence"
        and (state.get("synthesis_context") or {}).get("evidence")
        and candidate and candidate != _EMPTY_MESSAGE
    ):
        semantic_checked = True
        semantic_unsupported = await _semantic_check(state, candidate)
        if semantic_unsupported:
            unsupported_text = "；".join(semantic_unsupported)
            issues.append(f"unsupported_facts:{unsupported_text}")

    if candidate in {_EMPTY_MESSAGE, _INSUFFICIENT_MESSAGE}:
        grounding = "insufficient_evidence"
    elif issues:
        grounding = "failed"
    elif semantic_checked:
        grounding = "verified"
    else:
        grounding = "unverified" if state.get("candidate_streamed") else "verified"

    verification = {
        "grounding_status": grounding,
        "issues": issues,
        "semantic_checked": semantic_checked,
        "semantic_unsupported": semantic_unsupported,
    }
    await _emit(state, "validation", {
        "grounding_status": grounding,
        "issues": issues,
        "repairs_used": state.get("repairs_used", 0),
    })
    return {"verification": verification, "current_step": step}


def _after_verify(state: AgentState) -> str:
    """验证失败且还有修复额度 → repair_answer；否则 finalize。"""
    verification = state.get("verification") or {}
    if (
        verification.get("grounding_status") == "failed"
        and state.get("repairs_used", 0) < settings.answer_verify_max_repairs
    ):
        return "repair_answer"
    return "finalize"


def _strip_invalid_citations(candidate: str, valid_ids: set[str]) -> str:
    """移除答案中不存在的 [S#] 引用标记（流式路径的轻量修复）。"""

    def _keep(match: re.Match[str]) -> str:
        # 合法引用保留完整匹配（含方括号）；非法引用整体移除。
        return match.group(0) if match.group(1) in valid_ids else ""

    return _CITATION_RE.sub(_keep, candidate).strip()


async def _repair_node(state: AgentState) -> dict[str, Any]:
    """repair_answer：对未通过验证的候选答案修复一次。

    - 仅引用编号非法 → 确定性修复（剥除坏引用），零成本。
    - 缺引用/事实不被支持 → 带诊断信息重新生成一次（非流式）。
    修复后重新执行确定性检查；仍失败由 finalize 输出证据不足口径。
    """
    step = state.get("current_step", 0) + 1
    await _emit(state, "step_start", {
        "step": step,
        "action": "repair_answer",
        "purpose": "修复未通过校验的候选答案（最多一次）",
    })
    repairs_used = state.get("repairs_used", 0) + 1
    verification = state.get("verification") or {}
    issues = list(verification.get("issues", []))
    candidate = state.get("candidate_answer", "")
    context = state.get("synthesis_context") or {}
    evidence = context.get("evidence") or []
    valid_ids = {source.get("id") for source in state.get("sources", [])}
    invalid_only = bool(issues) and all(issue.startswith("invalid_citations:") for issue in issues)

    if invalid_only:
        repaired = _strip_invalid_citations(candidate, valid_ids)
        issues_left = _deterministic_checks(state, repaired)
    else:
        policy = {**DEFAULT_OUTPUT_POLICY, **(state.get("output_policy") or {})}
        unsupported = verification.get("semantic_unsupported") or []
        unsupported_text = "；".join(unsupported)[:400] if unsupported else "无"
        issues_text = "；".join(issues)[:600] or "无"
        evidence_text = _evidence_text(evidence)[:6000]
        answer_text = candidate[:2000]
        question = context.get("question", "")
        prompt = (
            "你的上一个回答未通过事实校验，请修复后重新输出完整答案。\n"
            f"问题：{question}\n\n"
            f"上一回答：\n{answer_text}\n\n"
            f"校验发现的问题：{issues_text}\n"
            f"不被支持的事实：{unsupported_text}\n"
            "要求：只依据下面的证据重写；删除或改写不被支持的内容；"
            "关键事实使用 [S#] 引用且编号必须来自证据列表；证据不足时明确说明。\n\n"
            f"证据：\n{evidence_text}"
        )
        response = await get_llm(temperature=0, max_tokens=settings.agent_synthesis_max_tokens).ainvoke([
            SystemMessage(content="你是严谨的小说问答总结助手，输出修复后的最终答案。"),
            HumanMessage(content=prompt),
        ])
        content = response.content if isinstance(response.content, str) else ""
        repaired = _sanitize_summary(content.strip(), policy)
        issues_left = _deterministic_checks(state, repaired)

    grounding = "repaired" if not issues_left else "failed"
    verification = {
        "grounding_status": grounding,
        "issues": issues_left,
        "semantic_checked": verification.get("semantic_checked", False),
        "semantic_unsupported": verification.get("semantic_unsupported", []),
        "repaired": True,
    }
    await _emit(state, "validation", {
        "grounding_status": grounding,
        "issues": issues_left,
        "repairs_used": repairs_used,
        "repaired": True,
    })
    return {
        "candidate_answer": repaired,
        "verification": verification,
        "repairs_used": repairs_used,
        "current_step": step,
    }


async def _finalize_node(state: AgentState) -> dict[str, Any]:
    """finalize：只输出通过验证的答案或明确的证据不足响应，并统一上报 meta。"""
    step = state.get("current_step", 0) + 1
    verification = state.get("verification") or {}
    grounding = verification.get("grounding_status", "unverified")
    candidate = state.get("candidate_answer", "")
    streamed = bool(state.get("candidate_streamed"))
    buffered = _should_buffer(state)
    fallback = state.get("fallback_reason", "")

    # 缓冲路径验证失败且无修复额度：替换为证据不足口径，不输出未验证内容。
    if grounding == "failed" and not streamed:
        candidate = _INSUFFICIENT_MESSAGE
        grounding = "insufficient_evidence"

    if buffered and not streamed:
        # 缓冲答案验证通过：此时才下发。
        await _emit(state, "token", candidate)
    elif streamed and grounding == "repaired" and candidate:
        # 流式路径修复过：以修复稿覆盖已渲染内容。
        await _emit(state, "token_replace", candidate)

    if grounding == "repaired":
        stop_reason = "answer_repaired"
    elif grounding == "insufficient_evidence":
        stop_reason = "evidence_insufficient"
    else:
        stop_reason = "answer_verified"

    context = state.get("synthesis_context") or {}
    meta = {
        "requested_strategy": state.get("requested_strategy", "auto"),
        "effective_strategy": state.get("strategy"),
        "strategy_adjusted": bool(state.get("strategy_adjusted")),
        "adjustment_reason": state.get("adjustment_reason", ""),
        "interaction_mode": state.get("interaction_mode", "qa"),
        "deprecations": state.get("deprecations", []),
        "intent": state.get("intent"),
        "original_query": state.get("original_query"),
        "standalone_query": state.get("standalone_query"),
        "retrieval_query": state.get("retrieval_query"),
        "query_preparation": state.get("query_preparation") or state.get("query_rewrite", {}),
        "plan": state.get("plan", []),
        "steps": step,
        "tool_calls_used": int(state.get("tool_calls_used", 0)),
        "halt_reason": fallback,
        "needs_retrieval": bool(state.get("rag_called")),
        "rag_called": bool(state.get("rag_called")),
        "rag_call_count": int(state.get("rag_call_count", 0)),
        "retrieval_policy": state.get("retrieval_policy", "optional"),
        "retrieval_skipped": not bool(state.get("rag_called")),
        "retrieval_reason": state.get("retrieval_reason", ""),
        "answer_mode": context.get("effective_answer_mode") or state.get("answer_mode", "novel_evidence"),
        "output_policy": {**DEFAULT_OUTPUT_POLICY, **(state.get("output_policy") or {})},
        "preference_update": state.get("preference_update"),
        "llm_needs_retrieval": state.get("llm_needs_retrieval"),
        "routing_override": state.get("routing_override", False),
        "routing_override_reason": state.get("routing_override_reason", ""),
        "routing_confidence": state.get("routing_confidence"),
        "stop_reason": stop_reason,
        "grounding_status": grounding,
        "completion_status": "completed",
        "personas": [card.get("name") for card in (state.get("character_cards") or [])],
        "chapter_until": state.get("chapter_until"),
        "memory_used_count": len(context.get("memories") or []),
        "summary_used": bool(context.get("summary")),
        "memory_ops": state.get("memory_ops", []),
        "validation_issues": verification.get("issues", []),
    }
    await _emit(state, "meta", meta)
    return {
        "answer": candidate,
        "status": "completed",
        "stop_reason": stop_reason,
        "current_step": step,
    }


def _build_graph():
    """构建并编译 LangGraph 唯一编排入口。"""
    graph = StateGraph(AgentState)
    graph.add_node("route", _route_node)
    graph.add_node("plan", _plan_node)
    graph.add_node("retrieve", _retrieve_node)
    graph.add_node("execute", _execute_node)
    graph.add_node("reflect", _reflect_node)
    graph.add_node("supervisor", _supervisor_node)
    graph.add_node("memory_agent", _memory_agent_node)
    graph.add_node("compose_answer", _compose_node)
    graph.add_node("verify_answer", _verify_node)
    graph.add_node("repair_answer", _repair_node)
    graph.add_node("finalize", _finalize_node)
    graph.add_edge(START, "route")
    graph.add_edge("route", "plan")
    graph.add_conditional_edges(
        "plan",
        _after_plan,
        {"retrieve": "retrieve", "execute": "execute"},
    )
    graph.add_edge("retrieve", "execute")
    # ReAct 执行环：execute → reflect → (execute | supervisor)。
    # 这是全图唯一的回边，构成真实的"执行—评估—再执行"循环；
    # 终止由执行预算（步数/工具数/检索数）与 react_done 双重保证，不会无限循环。
    graph.add_edge("execute", "reflect")
    graph.add_conditional_edges(
        "reflect",
        _after_reflect,
        {"execute": "execute", "supervisor": "supervisor"},
    )
    graph.add_conditional_edges(
        "supervisor",
        _after_supervisor,
        {"memory_agent": "memory_agent", "compose_answer": "compose_answer"},
    )
    graph.add_edge("memory_agent", "compose_answer")
    graph.add_edge("compose_answer", "verify_answer")
    graph.add_conditional_edges(
        "verify_answer",
        _after_verify,
        {"repair_answer": "repair_answer", "finalize": "finalize"},
    )
    graph.add_edge("repair_answer", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


agent_graph = _build_graph()


async def stream_agent_question(
    query: str,
    strategy: str = "auto",
    file_id: str | None = None,
    max_steps: int | None = None,
    original_query: str | None = None,
    retrieval_query: str | None = None,
    query_preparation: dict[str, Any] | None = None,
    query_rewrite: dict[str, Any] | None = None,
    memory_context: dict[str, Any] | None = None,
    session_id: str | None = None,
    memory_agent_active: bool = False,
    interaction_mode: str = "qa",
    personas: list[str] | None = None,
    chapter_until: int | None = None,
    roleplay_history: list[Any] | None = None,
    deprecations: list[str] | None = None,
):
    """Run LangGraph in the background and bridge live node events to SSE."""
    queue: asyncio.Queue = asyncio.Queue()
    initial: AgentState = {
        "query": query,
        "original_query": original_query or query,
        "standalone_query": query,
        "retrieval_query": retrieval_query or query,
        "query_preparation": query_preparation or query_rewrite or {},
        "query_rewrite": query_preparation or query_rewrite or {},
        "output_policy": (query_preparation or query_rewrite or {}).get("output_policy", dict(DEFAULT_OUTPUT_POLICY)),
        "preference_update": (query_preparation or query_rewrite or {}).get("preference_update"),
        "memory_context": memory_context or {},
        "requested_strategy": strategy,
        "requested_max_steps": max_steps,
        "file_id": file_id,
        "session_id": session_id,
        "memory_agent_active": memory_agent_active,
        "interaction_mode": interaction_mode,
        "personas": personas or [],
        "chapter_until": chapter_until,
        "roleplay_history": roleplay_history or [],
        "deprecations": deprecations or [],
        "react_messages": [],
        "rag_called": False,
        "rag_call_count": 0,
        "tool_calls_used": 0,
        "reasoning_round": 0,
        "repairs_used": 0,
        "stop_reason": "",
        "plan_adjusted": False,
        "event_queue": queue,
    }

    async def run_graph() -> None:
        try:
            await agent_graph.ainvoke(initial)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.error("agent_graph.failed", error=str(exc)[:300])
            await queue.put({"type": "error", "data": {"code": "agent_graph_failed", "message": "Agent 执行失败，请稍后重试"}})
        finally:
            await queue.put(_STREAM_DONE)

    await queue.put({"type": "run_started", "data": {
        "requested_strategy": strategy,
        "interaction_mode": interaction_mode,
        "personas": personas or [],
    }})
    graph_task = asyncio.create_task(run_graph())
    try:
        while True:
            event = await queue.get()
            if event is _STREAM_DONE:
                break
            yield event
        await graph_task
    finally:
        # 客户端断开 SSE 时取消图任务，避免后台继续消耗模型和检索资源。
        if not graph_task.done():
            graph_task.cancel()
            await asyncio.gather(graph_task, return_exceptions=True)
