"""LangGraph Agent Runtime with query decomposition, validation and live SSE events."""
from __future__ import annotations

import asyncio
import re
import time
from typing import Any

import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from app.agent.contracts import EXPERT_CONTRACTS, SPECIALIST_ORDER, SpecialistContract
from app.agent.dispatcher import dispatch_expert_tasks
from app.agent.router import route_query
from app.agent.tools import MEMORY_AGENT_TOOLS, MEMORY_AGENT_TOOL_SPECS, REACT_TOOL_LABELS, REACT_TOOL_SPECS, _react_payload, registry
from app.agent.types import AgentState, DEFAULT_OUTPUT_POLICY, Strategy, ToolResult
from app.agent.validation import validate_reports
from app.config import settings
from app.core.context import get_memory_session, set_memory_session
from app.core.llm import LLMPurpose, ModelTurnBuilder, astream_model_turn, get_llm, tool_call_field
from app.core.logging_config import get_logger
from app.core.metrics import metrics

log = get_logger("agent_runtime")
_EMPTY_MESSAGE = "当前小说知识库中没有检索到足以回答该问题的原文。请确认作品已完成索引，或补充人物名、事件名、章节等线索。"
_STREAM_DONE = object()
# 单轮 reasoning 累计字符上限：防异常模型无限输出（reasoning 只展示不持久化）。
_THINKING_MAX_CHARS = 8000


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
        step: int | None = None,
        retry: int | None = None,
    ) -> None:
        self._state = state
        self._stream = stream
        self._id = span_id
        self._phase = phase
        self._agent = agent
        self._label = label
        self._step = step
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
        if self._step is not None:
            payload["step"] = self._step
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
    if strategy is Strategy.MULTI_EXPERT:
        return [
            {"step": 1, "action": "retrieve_novel", "purpose": "召回所有专家共享的小说证据"},
            {"step": 2, "action": "dispatch_expert_tasks", "purpose": "将问题拆成四个互补子任务"},
            {"step": 3, "action": "multi_expert", "purpose": "四类专家并发分析共享证据"},
            {"step": 4, "action": "validate_reports", "purpose": "检查职责契约与报告重复度"},
            {"step": 5, "action": "supervisor", "purpose": "去重、消解冲突并汇总最终答案"},
        ]
    if strategy is Strategy.PLAN_EXECUTE:
        return [
            {"step": 1, "action": "make_plan", "purpose": "模型依据上下文产出执行计划"},
            {"step": 2, "action": "react_loop", "purpose": "模型自主决定检索、计算或直接回答"},
            {"step": 3, "action": "supervisor", "purpose": "生成带引用的最终答案"},
        ]
    if strategy is Strategy.DIRECT:
        return [
            {"step": 1, "action": "react_loop", "purpose": "短路径决策：模型自主决定检索或直接回答"},
            {"step": 2, "action": "supervisor", "purpose": "生成最终答案"},
        ]
    return [
        {"step": 1, "action": "react_loop", "purpose": "模型自主决定是否检索、计算或直接回答"},
        {"step": 2, "action": "supervisor", "purpose": "生成带引用的最终答案"},
    ]


def _normalize_evidence(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """按来源、章节和片段去重证据，并重新编号为稳定的 [S#] 引用。"""
    seen: set[tuple[Any, Any, Any]] = set()
    evidence: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for item in items:
        original = item.get("source", {})
        key = (original.get("source"), original.get("chapter_no"), original.get("chunk_no"))
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
    """LangGraph 路由节点：同时决定执行策略和是否调用小说 RAG。"""
    decision = route_query(
        state["standalone_query"],
        state.get("requested_strategy"),
        routing_hint=state.get("query_preparation") or state.get("query_rewrite") or None,
    )
    max_steps = min(state.get("requested_max_steps") or decision.max_steps, settings.agent_max_steps)
    route = decision.as_dict()
    route.update({
        "requested_strategy": state.get("requested_strategy", "auto"),
        "max_steps": max_steps,
        "max_experts": settings.agent_max_experts,
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
        "max_experts": settings.agent_max_experts,
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
        "evidence": [],
        "sources": [],
        "observations": [],
        "fallback_reason": "",
    }


async def _plan_node(state: AgentState) -> dict[str, Any]:
    """写入可展示的执行计划；required 兜底路径保留预检索步骤。"""
    strategy = Strategy(state["strategy"])
    if strategy is Strategy.MULTI_EXPERT or state.get("retrieval_policy") == "required":
        plan = _plan(strategy)
        if strategy is not Strategy.MULTI_EXPERT:
            # required 兜底：预检索只是初始证据来源，之后仍进入同一决策循环。
            plan = [
                {"step": 1, "action": "retrieve_novel", "purpose": "保守兜底预检索（查询准备不可靠）"},
                {"step": 2, "action": "react_loop", "purpose": "模型自主决定补充检索、计算或直接回答"},
                {"step": 3, "action": "supervisor", "purpose": "生成带引用的最终答案"},
            ]
    else:
        plan = _plan(strategy)
    await _emit(state, "plan", {
        "steps": plan,
        "max_steps": state["max_steps"],
        "retrieval_policy": state.get("retrieval_policy", "optional"),
    })
    return {"plan": plan}


def _after_plan(state: AgentState) -> str:
    """multi_expert 与 required 兜底先跑共享预检索；其余直接进入决策循环。"""
    if Strategy(state["strategy"]) is Strategy.MULTI_EXPERT:
        return "retrieve"
    return "retrieve" if state.get("retrieval_policy") == "required" else "execute"


async def _retrieve_node(state: AgentState) -> dict[str, Any]:
    """执行一次共享小说检索，后续专家不再分别调用 RAG。"""
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
        "fallback_reason": "" if evidence else (result.error_code or "empty_retrieval"),
        # 预检索也算"实际执行过 RAG"：multi_expert 与 required 兜底都经此路径。
        "rag_called": True,
        "rag_call_count": 1,
    }


def _after_retrieve(state: AgentState) -> str:
    """预检索完成后：multi_expert 去分派专家，其余进入同一决策循环。"""
    if Strategy(state["strategy"]) is Strategy.MULTI_EXPERT:
        return "dispatch"
    return "execute"


async def _dispatch_node(state: AgentState) -> dict[str, Any]:
    """为 multi_expert 分支生成四个互斥的动态子任务。"""
    await _emit(state, "step_start", {
        "step": 2,
        "action": "dispatch_expert_tasks",
        "purpose": "生成四个职责互斥的专家子任务",
    })
    result = await dispatch_expert_tasks(state["standalone_query"])
    public_tasks = {
        name: {"label": task["label"], "task": task["task"]}
        for name, task in result.tasks.items()
    }
    await _emit(state, "expert_tasks", {
        "tasks": public_tasks,
        "mode": result.mode,
        "reason": result.reason,
    })
    return {
        "expert_tasks": result.tasks,
        "dispatch_mode": result.mode,
        "dispatch_reason": result.reason,
        "current_step": 2,
        "expert_retry_count": {name: 0 for name in SPECIALIST_ORDER},
    }


def _specialist_prompt(contract: SpecialistContract, state: AgentState, correction: dict[str, Any] | None) -> str:
    """构造带固定契约、专属任务和共享证据的专家提示词。"""
    task = state["expert_tasks"][contract.name]
    base = (
        f"你是小说问答系统的{contract.label}。\n\n"
        f"原始用户问题仅用于理解背景，不要求你完整回答：\n{state['standalone_query']}\n\n"
        f"你的本轮专属任务：\n{task['task']}\n\n"
        f"固定职责：\n- " + "\n- ".join(task["focus"]) + "\n\n"
        "禁止事项：\n- " + "\n- ".join(task["forbidden"]) + "\n\n"
        f"必须使用的输出格式：\n{task['output_format']}\n\n"
        "只能完成本专家的专属任务，不得覆盖其他专家职责。关键结论必须使用 [S#] 引用。"
        "没有本维度发现时直接写“本维度证据不足”。请将整段分析写成 Markdown 引用块，每一行以 '> ' 开头。\n\n"
        f"共享证据：\n{_evidence_text(state.get('evidence', []))}"
    )
    if not correction:
        return base
    flags = correction.get("similarity_flags", [])
    duplicate_agents = "、".join(flag["agent"] for flag in flags) or "无"
    return (
        f"{base}\n\n你的上一份报告未通过校验，需要纠偏一次。\n"
        f"上一份报告：\n{correction.get('previous_report', '')}\n\n"
        f"缺失项：{'；'.join(correction.get('missing_sections', [])) or '无'}\n"
        f"越界项：{'；'.join(correction.get('forbidden_hits', [])) or '无'}\n"
        f"高度重复对象：{duplicate_agents}\n"
        "请删除与其他专家重复的完整总述，只保留本专家独有贡献，并严格遵守输出格式。"
    )


async def _generate_fallback_report(contract: SpecialistContract, state: AgentState) -> str:
    """专家 reasoning-only / 空输出时的最终报告补生成。

    走主模型（get_llm 不带 purpose → 全局关闭 thinking）一次非流式调用：
    不带纠偏诊断、不暴露内部推理，直接按原任务与共享证据输出报告。
    """
    response = await get_llm(temperature=0).ainvoke([
        SystemMessage(content="你只完成被分配的专家子任务，共享原文是唯一事实边界。直接输出最终报告，不要输出思考过程、任务说明或解释。"),
        HumanMessage(content=_specialist_prompt(contract, state, None)),
    ])
    content = response.content if isinstance(response.content, str) else ""
    return content.strip()


async def _run_specialist(
    contract: SpecialistContract,
    state: AgentState,
    agent_id: str,
    *,
    retry: int = 0,
    correction: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """执行单个专家并收集流式报告；失败只影响当前专家。

    reasoning 走 thinking 事件（stream=multi_agent，按 agent 分组），最终报告
    仍走 tool_token——前端报告渲染不受影响；纠偏使用新 stream id 防拼接旧流。
    正文为空（reasoning-only 或静默空回包）不是契约问题：不进入契约纠偏，
    由主模型补生成一次，仍为空才标记 empty_output 并交给 Supervisor 降级。
    """
    started = time.perf_counter()
    first_token_ms: float | None = None
    parts: list[str] = []
    thinking = _ThinkingSpan(
        state,
        stream="multi_agent",
        span_id=f"expert-{contract.name}" + (f"-retry{retry}" if retry else ""),
        phase="reasoning",
        agent=contract.name,
        label=contract.label,
        retry=retry or None,
    )
    recovered = False
    try:
        async for delta in astream_model_turn([
            SystemMessage(content="你只完成被分配的专家子任务，共享原文是唯一事实边界。"),
            HumanMessage(content=_specialist_prompt(contract, state, correction)),
        ], LLMPurpose.EXPERT, max_tokens=settings.agent_expert_max_tokens):
            if delta.kind == "reasoning":
                await thinking.token(delta.text)
                continue
            if delta.kind != "content":
                continue
            token = delta.text
            if first_token_ms is None:
                first_token_ms = round((time.perf_counter() - started) * 1000, 1)
            parts.append(token)
            await _emit(state, "tool_token", {
                "id": agent_id,
                "tool": "specialist",
                "agent": contract.name,
                "label": contract.label,
                "retry": retry,
                "delta": token,
            })
        await thinking.end("corrected" if retry else "completed")
        metrics.incr("agent_expert_reasoning_chars", thinking.chars)

        report_text = "".join(parts).strip()
        if not report_text:
            await _emit(state, "tool_end", {
                "id": agent_id,
                "tool": "specialist",
                "agent": contract.name,
                "label": contract.label,
                "status": "fallback_generation",
                "reason": "expert_final_content_empty",
                "summary": "专家最终报告未生成，正在重新生成",
                "retry": retry,
            })
            metrics.incr("agent_expert_fallback_generation_count")
            report_text = await _generate_fallback_report(contract, state)
            recovered = bool(report_text)
            if report_text:
                # 补生成是非流式调用：报告整段回灌 tool_token，让前端「调用过程」
                # 里能看到正文（与主流式报告同一条渲染路径）。
                await _emit(state, "tool_token", {
                    "id": agent_id,
                    "tool": "specialist",
                    "agent": contract.name,
                    "label": contract.label,
                    "retry": retry,
                    "delta": report_text,
                })
        metrics.incr("agent_expert_report_chars", len(report_text))

        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        if not report_text:
            # 补生成仍为空：标记 empty_output 退出；校验集合只收 ok 报告，天然不进纠偏。
            metrics.incr("agent_expert_empty_output_count")
            await _emit(state, "tool_end", {
                "id": agent_id,
                "tool": "specialist",
                "agent": contract.name,
                "label": contract.label,
                "status": "empty_output",
                "error_code": "expert_final_content_empty",
                "summary": "专家最终报告未生成",
                "latency_ms": latency_ms,
                "retry": retry,
            })
            return {
                "agent": contract.name,
                "label": contract.label,
                "status": "empty_output",
                "report": "",
                "error_code": "expert_final_content_empty",
                "latency_ms": latency_ms,
                "corrected": bool(retry),
            }

        summary = (
            f"{contract.label}纠偏完成" if retry
            else f"{contract.label}报告补生成完成" if recovered
            else f"{contract.label}分析完成"
        )
        await _emit(state, "tool_end", {
            "id": agent_id,
            "tool": "specialist",
            "agent": contract.name,
            "label": contract.label,
            "status": "corrected" if retry else "ok",
            "summary": summary,
            "latency_ms": latency_ms,
            "first_token_ms": first_token_ms,
            "retry": retry,
            "recovered": recovered,
        })
        return {
            "agent": contract.name,
            "label": contract.label,
            "status": "ok",
            "report": report_text,
            "latency_ms": latency_ms,
            "first_token_ms": first_token_ms,
            "corrected": bool(retry),
            "recovered": recovered,
        }
    except asyncio.CancelledError:
        await thinking.end("cancelled")
        raise
    except Exception as exc:  # noqa: BLE001
        await thinking.end("error")
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        await _emit(state, "tool_end", {
            "id": agent_id,
            "tool": "specialist",
            "agent": contract.name,
            "label": contract.label,
            "status": "error",
            "summary": f"{contract.label}失败",
            "latency_ms": latency_ms,
            "retry": retry,
        })
        log.warning("specialist.failed", agent=contract.name, retry=retry, error=str(exc)[:200])
        return {
            "agent": contract.name,
            "label": contract.label,
            "status": "error",
            "report": "".join(parts),
            "error": str(exc)[:200],
            "corrected": bool(retry),
        }


async def _run_specialists_concurrently(
    names: list[str],
    state: AgentState,
    *,
    retry: int = 0,
    corrections: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """并发运行专家任务并隔离各自的输出事件、超时和异常。"""
    task_map: dict[asyncio.Task, str] = {}
    for name in names:
        contract = EXPERT_CONTRACTS[name]
        expert_task = state["expert_tasks"][name]
        await _emit(state, "tool_start", {
            "id": f"expert-{name}",
            "tool": "specialist",
            "agent": name,
            "label": contract.label,
            "task": expert_task["task"],
            "step": 5 if retry else 3,
            "retry": retry,
            "reset": bool(retry),
            "reason": "report_correction" if retry else "initial_analysis",
        })
        task_map[asyncio.create_task(_run_specialist(
            contract,
            state,
            f"expert-{name}",
            retry=retry,
            correction=(corrections or {}).get(name),
        ))] = name

    reports: dict[str, dict[str, Any]] = {}
    try:
        # 专家并发等待统一超时；已完成结果保留，未完成任务标记 timeout 而不是拖垮整轮问答。
        done, pending = await asyncio.wait(task_map, timeout=settings.agent_multi_expert_timeout)
        for task in done:
            name = task_map[task]
            reports[name] = task.result()
        for task in pending:
            name = task_map[task]
            task.cancel()
            reports[name] = {
                "agent": name,
                "label": EXPERT_CONTRACTS[name].label,
                "status": "timeout",
                "report": "",
                "error": "expert_timeout",
                "corrected": bool(retry),
            }
            await _emit(state, "tool_end", {
                "id": f"expert-{name}",
                "tool": "specialist",
                "agent": name,
                "label": EXPERT_CONTRACTS[name].label,
                "status": "timeout",
                "summary": f"{EXPERT_CONTRACTS[name].label}超时，已忽略该结果",
                "retry": retry,
            })
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
    except asyncio.CancelledError:
        for task in task_map:
            task.cancel()
        await asyncio.gather(*task_map, return_exceptions=True)
        raise
    return [reports[name] for name in names]


async def _experts_node(state: AgentState) -> dict[str, Any]:
    """运行四专家节点并将成功报告写入共享状态。"""
    names = list(SPECIALIST_ORDER[: state.get("max_experts", 4)])
    assignments = [EXPERT_CONTRACTS[name].label for name in names]
    if not state.get("evidence"):
        return {"assignments": assignments, "reports": [], "fallback_reason": "empty_retrieval"}
    metrics.incr("agent_multi_expert_runs")
    await _emit(state, "step_start", {"step": 3, "action": "multi_expert", "purpose": "并发运行专属专家子任务"})
    reports = await _run_specialists_concurrently(names, state)
    successful = [report for report in reports if report.get("status") == "ok"]
    fallback = "" if successful else "all_experts_failed"
    if fallback:
        metrics.incr("agent_multi_expert_fallbacks")
    return {"assignments": assignments, "reports": reports, "fallback_reason": fallback, "current_step": 3}


async def _validate_reports_node(state: AgentState) -> dict[str, Any]:
    """校验专家报告并决定是否启动一次局部纠偏。"""
    validations, refine_agents = validate_reports(
        state.get("reports", []),
        settings.agent_report_similarity_threshold,
    )
    await _emit(state, "validation", {
        "reports": validations,
        "refine_agents": refine_agents,
        "retry": 0,
    })
    return {
        "report_validation": validations,
        "refine_agents": refine_agents,
        "current_step": 4,
    }


def _after_validation(state: AgentState) -> str:
    """根据报告校验结果路由到纠偏或 Supervisor。"""
    if state.get("refine_agents") and settings.agent_expert_correction_retries > 0:
        return "refine"
    return "supervisor"


async def _refine_experts_node(state: AgentState) -> dict[str, Any]:
    """仅重试被标记的专家一次，并通过 reset 事件让前端清理旧文本。"""
    names = [
        name for name in state.get("refine_agents", [])
        if state.get("expert_retry_count", {}).get(name, 0) < settings.agent_expert_correction_retries
    ]
    if not names:
        return {"refine_agents": []}

    by_agent = {report["agent"]: report for report in state.get("reports", [])}
    corrections: dict[str, dict[str, Any]] = {}
    for name in names:
        validation = state["report_validation"][name]
        if validation.get("similarity_flags"):
            metrics.incr("agent_expert_similarity_correction_count")
        else:
            metrics.incr("agent_expert_contract_correction_count")
        corrections[name] = {
            **validation,
            "previous_report": by_agent[name].get("report", ""),
        }
    refined = await _run_specialists_concurrently(names, state, retry=1, corrections=corrections)
    for report in refined:
        by_agent[report["agent"]] = report

    reports = [by_agent[name] for name in SPECIALIST_ORDER if name in by_agent]
    validations, _ = validate_reports(reports, settings.agent_report_similarity_threshold)
    for name in names:
        validation = validations[name]
        if not validation["contract_ok"] or validation["similarity_flags"]:
            by_agent[name]["status"] = "invalid"
            validation["contract_ok"] = False
            if validation["similarity_flags"]:
                validation["missing_sections"].append("纠偏后仍与其他专家高度重复")
    reports = [by_agent[name] for name in SPECIALIST_ORDER if name in by_agent]
    retry_count = dict(state.get("expert_retry_count", {}))
    for name in names:
        retry_count[name] = retry_count.get(name, 0) + 1

    await _emit(state, "validation", {
        "reports": validations,
        "refine_agents": [],
        "retry": 1,
    })
    return {
        "reports": reports,
        "report_validation": validations,
        "refine_agents": [],
        "expert_retry_count": retry_count,
        "current_step": 5,
    }


# ReAct 证据累积上限（块数）：循环成立后证据跨轮增长，系统没有全局 token
# 计数器，必须在源头封顶，否则证据滚雪球会放大 summary prompt 成本。
_REACT_MAX_EVIDENCE = 24


def _react_system_prompt(state: AgentState) -> str:
    """Agent 决策循环的系统提示：是否调用 RAG 的判定条件集中在此。"""
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
    """决策循环的首条用户消息：问题 + 查询准备建议 + 记忆上下文。"""
    original = state.get("original_query", "")
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


def _parse_plan_text(text: str) -> list[dict[str, Any]]:
    """把模型输出的计划文本解析为可展示的计划步骤。"""
    steps: list[dict[str, Any]] = []
    for line in (text or "").splitlines():
        line = line.strip().lstrip("-*· ")
        if not line:
            continue
        steps.append({"step": len(steps) + 1, "action": "model_step", "purpose": line[:120]})
        if len(steps) >= 5:
            break
    return steps


async def _execute_node(state: AgentState) -> dict[str, Any]:
    """Agent 决策循环的单步：observe（消息历史）→ decide（模型选工具/直答/计划）→ act。

    react/direct/plan_execute 共用本节点。工具结果经 ToolMessage 回灌
    （react_messages 跨轮保留），模型每轮都能看到此前全部工具结果。
    """
    evidence = list(state.get("evidence", []))
    observations = list(state.get("observations", []))
    # 步号从 0 起计：仅当共享预检索（required 兜底/multi_expert）已占用第 1 步时，
    # 首个决策才是第 2 步；检索被跳过的轮次不再凭空空缺"第 1 步"。
    current_step = state.get("current_step", 0)
    fallback = state.get("fallback_reason", "")
    max_steps = state["max_steps"]
    strategy = Strategy(state["strategy"])
    react_messages = list(state.get("react_messages", []))
    rag_called = bool(state.get("rag_called"))
    rag_count = int(state.get("rag_call_count", 0))

    if current_step >= max_steps:
        return {
            "current_step": current_step,
            "fallback_reason": fallback or "step_budget_exceeded",
            "react_done": True,
            "stop_reason": fallback or "step_budget_exceeded",
            "react_messages": react_messages,
        }

    if not react_messages:
        react_messages = [HumanMessage(content=_react_initial_human(state))]

    system = _react_system_prompt(state)
    plan_phase = strategy is Strategy.PLAN_EXECUTE and not state.get("plan_committed")
    if plan_phase:
        system = (
            "你是小说问答的规划助手。请给出不超过 5 行的中文执行计划，每行一个动作，"
            "说明为回答用户问题需要哪些证据或计算。不要调用工具，只输出计划本身。"
        )
    messages = [SystemMessage(content=system)] + react_messages
    thinking = _ThinkingSpan(
        state,
        stream="main_agent",
        span_id=f"agent-step-{current_step + 1}",
        phase="plan" if plan_phase else "decide",
        step=current_step + 1,
    )
    builder = ModelTurnBuilder()
    purpose = LLMPurpose.AGENT_PLAN if plan_phase else LLMPurpose.AGENT_DECISION
    try:
        async for delta in astream_model_turn(
            messages, purpose,
            tools=list(REACT_TOOL_SPECS),
            max_tokens=400 if plan_phase else 300,
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

    # plan_execute 首轮强制先产出计划；计划本身不触发任何工具。
    if strategy is Strategy.PLAN_EXECUTE and not state.get("plan_committed"):
        steps = _parse_plan_text(content_text)
        await _emit(state, "plan", {"steps": steps, "max_steps": max_steps, "plan_adjusted": False})
        await _emit(state, "agent_decision", {"action": "plan", "reason": "复杂任务先生成执行计划", "step": current_step})
        return {
            "plan": steps or state.get("plan", []),
            "plan_committed": True,
            "current_step": current_step,
            "react_done": False,
            "react_messages": react_messages + [AIMessage(content=content_text, tool_calls=[])],
        }

    if not calls:
        # 模型不再调用工具即判定上下文足够，循环终止，进入汇总。
        await _emit(state, "agent_decision", {
            "action": "answer",
            "reason": (content_text.strip()[:120] or "当前上下文足以回答"),
            "step": current_step,
        })
        return {
            "current_step": current_step,
            "react_done": True,
            "stop_reason": fallback or "model_answer",
            "react_messages": react_messages,
            "rag_called": rag_called,
            "rag_call_count": rag_count,
        }

    # react 自发计划：第一轮模型在调用工具的同时给出多行计划文本，则展示之。
    if strategy is Strategy.REACT and not state.get("plan_committed") and content_text.count("\n") >= 1:
        candidate = _parse_plan_text(content_text)
        if len(candidate) >= 2:
            await _emit(state, "plan", {"steps": candidate, "max_steps": max_steps, "plan_adjusted": False})
            await _emit(state, "agent_decision", {"action": "plan", "reason": "复杂任务先生成执行计划", "step": current_step})

    await _emit(state, "agent_decision", {
        "action": "tool_call",
        # calls 是 llm.ToolCall 数据类：必须属性访问，字典 .get 会在"无正文纯工具调用"时崩溃。
        "reason": (content_text.strip()[:120] or "、".join(call.name for call in calls[:2])),
        "step": current_step + 1,
    })

    # 单轮最多执行 min(2, 剩余预算) 个工具调用，防止单轮爆发。
    budget_this_turn = min(2, max_steps - current_step)
    executed_calls: list[dict[str, Any]] = []
    executed_results: list[ToolResult] = []
    for call in calls:
        if len(executed_calls) >= budget_this_turn:
            break
        name = call.name
        call_id = call.id or f"agent-step-{current_step + len(executed_calls) + 1}"
        # 模型只提供业务参数（query/expression）；检索范围（file_id）由状态注入，
        # 不进模型 schema，避免伪造或跨书检索。
        kwargs = {k: v for k, v in call.args.items() if k in {"query", "expression"}}
        if name in {"retrieve_novel", "get_chapter_context"} and not kwargs.get("query"):
            kwargs["query"] = state["standalone_query"]
        current_step += 1
        await _emit(state, "step_start", {"step": current_step, "action": name, "purpose": REACT_TOOL_LABELS.get(name, name)})
        await _emit(state, "tool_start", {"id": call_id, "tool": name, "label": REACT_TOOL_LABELS.get(name, name), "step": current_step})
        result = await registry.execute(
            name,
            allowed_tools=state["allowed_tools"],
            file_id=state.get("file_id"),
            **kwargs,
        )
        observations.append(result.as_dict())
        if name in {"retrieve_novel", "get_chapter_context"}:
            rag_called = True
            rag_count += 1
        if result.status == "ok" and isinstance(result.output, dict):
            evidence.extend(result.output.get("evidence", []))
        await _emit(state, "observation", {"step": current_step, **result.as_dict()})
        await _emit(state, "tool_end", {
            "id": call_id,
            "tool": name,
            "label": REACT_TOOL_LABELS.get(name, name),
            "step": current_step,
            "status": result.status,
            "summary": result.error_code or f"完成，耗时 {result.latency_ms}ms",
        })
        executed_calls.append({"id": call_id, "name": name, "args": dict(call.args)})
        executed_results.append(result)
        if result.status != "ok":
            fallback = result.error_code or "tool_failed"
            break

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
    return {
        "evidence": normalized,
        "sources": sources,
        "observations": observations,
        "current_step": current_step,
        "fallback_reason": fallback,
        "react_done": False,
        "react_messages": react_messages,
        "rag_called": rag_called,
        "rag_call_count": rag_count,
        "stop_reason": "",
    }


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
    await _emit(state, "reflection", {"decision": decision, "reason": reason, "step": state.get("current_step", 0)})
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
            result = await registry.execute(name, allowed_tools=MEMORY_AGENT_TOOLS, **args)
            # 前端只展示操作语义，不展示记忆内容与参数，避免把用户信息铺在界面上。
            brief = {
                "search_memories": "已核对现有记忆",
                "add_memory": "已记录新的记忆",
                "update_memory": "已更新既有记忆",
                "delete_memory": "已遗忘对应记忆",
            }.get(name, "记忆操作完成")
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


def _strip_quote_markers(text: str) -> str:
    return re.sub(r"(?m)^>\s?", "", text).strip()


async def _supervisor_node(state: AgentState) -> dict[str, Any]:
    """内部 Supervisor：验证证据、去重报告并准备给 Summary Node 的上下文。"""
    step = state.get("current_step", 0) + 1
    await _emit(state, "step_start", {
        "step": step,
        "action": "supervisor",
        "purpose": "内部验证、去重和冲突消解",
    })
    evidence = state.get("evidence", [])
    reports = state.get("reports", [])
    validations = state.get("report_validation", {})
    successful = [
        report for report in reports
        if report.get("status") == "ok"
        and validations.get(report.get("agent"), {}).get("contract_ok", True)
    ]
    fallback = state.get("fallback_reason", "")
    rag_called = bool(state.get("rag_called"))
    answer_mode = state.get("answer_mode", "novel_evidence")
    if state.get("answer_mode", "novel_evidence") == "novel_evidence" and not evidence and rag_called:
        fallback = fallback or "empty_retrieval"
    # Agent-first 语义：实际执行过 RAG 才走原文证据汇总；模型跳过检索时按
    # 路由判定的会话/记忆模式回答（summary 侧对"建议检索却未检索"加强提示）。
    if rag_called:
        effective_answer_mode = "novel_evidence"
    else:
        effective_answer_mode = answer_mode if answer_mode in {"memory_context", "conversation"} else "conversation"
    synthesis_context = {
        "question": state.get("standalone_query", ""),
        "evidence": evidence,
        "sources": state.get("sources", []),
        "reports": successful,
        "report_validation": validations,
        "observations": state.get("observations", []),
        "summary": (state.get("memory_context") or {}).get("summary", ""),
        "memories": (state.get("memory_context") or {}).get("memories", []),
        "effective_answer_mode": effective_answer_mode,
        "route_suggested_retrieval": bool(state.get("needs_retrieval")) and not rag_called,
    }
    return {
        "synthesis_context": synthesis_context,
        "fallback_reason": fallback,
        "current_step": step,
    }


def _sanitize_summary(text: str, policy: dict[str, Any]) -> str:
    """应用最终输出护栏，避免模型泄漏原文摘录或内部执行说明。"""
    result = text.strip()
    result = re.sub(r"(?im)^\s*(?:根据共享原文(?:与四份专家报告)?|根据专家报告)[：:]?\s*", "", result)
    if policy.get("summary_only") or not policy.get("show_source_text"):
        result = re.sub(r"[“\"]([^”\"]{24,})[”\"]", lambda m: m.group(1)[:20] + "……", result)
        result = re.sub(r"(?m)^>\s?", "", result)
    if not policy.get("allow_direct_quotes"):
        result = re.sub(r"[“\"]([^”\"]{1,200})[”\"]", "", result)
    if not policy.get("show_citations") or policy.get("citation_style") == "hidden":
        result = re.sub(r"\s*\[S\d+\]", "", result)
    return re.sub(r"\n{3,}", "\n\n", result).strip()


async def _summary_node(state: AgentState) -> dict[str, Any]:
    """最终 Summary Node：只向用户输出符合 output_policy 的总结。"""
    step = state.get("current_step", 0) + 1
    await _emit(state, "step_start", {
        "step": step,
        "action": "summary",
        "purpose": "按用户偏好生成最终总结",
    })
    policy = {**DEFAULT_OUTPUT_POLICY, **(state.get("output_policy") or {})}
    context = state.get("synthesis_context") or {}
    evidence = context.get("evidence") or []
    reports = context.get("reports") or []
    answer_mode = context.get("effective_answer_mode") or state.get("answer_mode", "novel_evidence")
    fallback = state.get("fallback_reason", "")
    if answer_mode == "novel_evidence" and not evidence:
        answer = _EMPTY_MESSAGE
        await _emit(state, "token", answer)
    else:
        memory_text = "\n".join(
            f"- [{item.get('memory_type', 'memory')}] {item.get('content', '')}"
            for item in (context.get("memories") or []) if item.get("content")
        ) or "（无可用长期记忆）"
        summary_text = context.get("summary") or "（无会话摘要）"
        policy_text = (
            "只输出总结、结论和分析；禁止展示来源片段、复制原文、连续复述人物原话，"
            "禁止以‘根据共享原文与专家报告’描述内部过程。"
            if policy.get("summary_only") or not policy.get("show_source_text")
            else "按用户本轮要求提供必要的原文依据。"
        )
        if not policy.get("allow_direct_quotes"):
            policy_text += "禁止长引号和直接人物原话。"
        if policy.get("show_citations") and policy.get("citation_style") == "chapter_only":
            policy_text += "出处只保留章节、回目、页码或来源编号，不展示原文片段。"
        elif not policy.get("show_citations"):
            policy_text += "不要输出 [S#] 来源标记。"
        if answer_mode == "novel_evidence":
            reports_text = "\n\n".join(
                f"【{report.get('label', report.get('agent', '专家'))}】\n{_strip_quote_markers(report.get('report', ''))}"
                for report in reports
            ) or "（无专家报告）"
            tool_text = "\n".join(
                f"- {obs.get('tool')}: {json.dumps(obs.get('output'), ensure_ascii=False)[:200]}"
                for obs in context.get("observations", [])
                if obs.get("tool") not in {
                    "retrieve_novel", "get_chapter_context", "specialist",
                    "search_memories", "add_memory", "update_memory", "delete_memory",
                }
                and obs.get("status") == "ok" and obs.get("output") is not None
            ) or "（无）"
            prompt = (
                f"{policy_text}\n请基于经过内部校验的小说证据回答用户问题。事实优先于推断；"
                "若证据不足请明确说明。不要展示内部过程。关键事实可使用 [S#]，但严格遵守展示策略。\n\n"
                f"问题：{context.get('question', '')}\n\n共享证据：\n{_evidence_text(evidence)}\n\n"
                f"专家内部结论：\n{reports_text}\n\n工具计算结果：\n{tool_text}\n\n"
                f"会话摘要：\n{summary_text}\n\n长期记忆：\n{memory_text}"
            )
            system = "你是严谨的小说问答总结助手。"
        else:
            prompt = (
                f"{policy_text}\n当前回答不依赖小说原文检索。请基于会话摘要、长期记忆和用户本轮提供的内容自然回答，"
                "不要编造小说事实，不要生成 [S#]。如果用户是在设置偏好，简洁确认即可。\n\n"
                f"问题：{context.get('question', '')}\n\n会话摘要：\n{summary_text}\n\n长期记忆：\n{memory_text}"
            )
            if context.get("route_suggested_retrieval"):
                # 路由建议检索而 Agent 未检索：强制声明证据边界，禁止参数记忆冒充原文。
                prompt += (
                    "\n\n注意：查询准备判定该问题可能需要小说原文证据，但本轮未执行检索。"
                    "回答中涉及小说事实的部分必须明确说明“未核对原文、无法确认”，"
                    "严禁凭记忆给出具体情节、数字或引文；如需准确答案请建议用户追问以触发检索。"
                )
            system = "你是能够保持会话连续性的小说阅读助手。"
        parts: list[str] = []
        # 流式下发总结 token：专家报告已流式展示，最终答案同样逐段推送，
        # 避免长答案整段等待。
        async for token in _stream_llm([
            SystemMessage(content=system),
            HumanMessage(content=prompt),
        ], settings.agent_synthesis_max_tokens):
            parts.append(token)
            await _emit(state, "token", token)
        answer = _sanitize_summary("".join(parts), policy)
        if answer != "".join(parts):
            # 输出护栏净化改变了内容（去引用 / 截断引语 / 隐藏 [S#]）时，
            # 以完整净化稿覆盖已流式渲染的内容：流式体验与护栏语义同时成立。
            await _emit(state, "token_replace", answer)

    meta = {
        "strategy": state.get("strategy"),
        "intent": state.get("intent"),
        "original_query": state.get("original_query"),
        "standalone_query": state.get("standalone_query"),
        "retrieval_query": state.get("retrieval_query"),
        "query_preparation": state.get("query_preparation") or state.get("query_rewrite", {}),
        "dispatch_mode": state.get("dispatch_mode"),
        "dispatch_reason": state.get("dispatch_reason"),
        "plan": state.get("plan", []),
        "steps": step,
        "assignments": state.get("assignments", []),
        "reports": [{key: value for key, value in report.items() if key != "report"} for report in state.get("reports", [])],
        "report_validation": state.get("report_validation", {}),
        "expert_count": len(state.get("assignments", [])),
        "fallback_reason": fallback,
        # Agent-first 语义：needs_retrieval 表示"实际是否执行过 RAG"，
        # 路由建议与实际执行分开上报（retrieval_policy / llm_needs_retrieval）。
        "needs_retrieval": bool(state.get("rag_called")),
        "rag_called": bool(state.get("rag_called")),
        "rag_call_count": int(state.get("rag_call_count", 0)),
        "retrieval_policy": state.get("retrieval_policy", "optional"),
        "stop_reason": state.get("stop_reason", ""),
        "plan_adjusted": bool(state.get("plan_adjusted")),
        "retrieval_skipped": not bool(state.get("rag_called")),
        "retrieval_reason": state.get("retrieval_reason", ""),
        "answer_mode": answer_mode,
        "output_policy": policy,
        "preference_update": state.get("preference_update"),
        "llm_needs_retrieval": state.get("llm_needs_retrieval"),
        "routing_override": state.get("routing_override", False),
        "routing_override_reason": state.get("routing_override_reason", ""),
        "routing_confidence": state.get("routing_confidence"),
        "memory_used_count": len(context.get("memories") or []),
        "summary_used": bool(context.get("summary")),
        "memory_ops": state.get("memory_ops", []),
    }
    await _emit(state, "meta", meta)
    return {"answer": answer, "fallback_reason": fallback, "status": "completed", "current_step": step}


def _after_supervisor(state: AgentState) -> str:
    """Supervisor 之后先让模型自主维护记忆（可跳过），再进入最终总结。"""
    if state.get("memory_agent_active"):
        return "memory_agent"
    return "summary"


def _build_graph():
    """构建并编译 LangGraph 唯一编排入口。"""
    graph = StateGraph(AgentState)
    graph.add_node("route", _route_node)
    graph.add_node("plan", _plan_node)
    graph.add_node("retrieve", _retrieve_node)
    graph.add_node("dispatch", _dispatch_node)
    graph.add_node("experts", _experts_node)
    graph.add_node("validate_reports", _validate_reports_node)
    graph.add_node("refine_experts", _refine_experts_node)
    graph.add_node("execute", _execute_node)
    graph.add_node("reflect", _reflect_node)
    graph.add_node("supervisor", _supervisor_node)
    graph.add_node("memory_agent", _memory_agent_node)
    graph.add_node("summary", _summary_node)
    graph.add_edge(START, "route")
    graph.add_edge("route", "plan")
    graph.add_conditional_edges(
        "plan",
        _after_plan,
        {"retrieve": "retrieve", "execute": "execute"},
    )
    graph.add_conditional_edges(
        "retrieve",
        _after_retrieve,
        {"dispatch": "dispatch", "execute": "execute", "supervisor": "supervisor"},
    )
    graph.add_edge("dispatch", "experts")
    graph.add_edge("experts", "validate_reports")
    graph.add_conditional_edges(
        "validate_reports",
        _after_validation,
        {"refine": "refine_experts", "supervisor": "supervisor"},
    )
    graph.add_edge("refine_experts", "supervisor")
    # ReAct 执行环：execute → reflect → (execute | supervisor)。
    # 这是全图唯一的回边，构成真实的"执行—评估—再执行"循环；
    # 终止由 max_steps 预算与 react_done 双重保证，不会无限循环。
    graph.add_edge("execute", "reflect")
    graph.add_conditional_edges(
        "reflect",
        _after_reflect,
        {"execute": "execute", "supervisor": "supervisor"},
    )
    graph.add_conditional_edges(
        "supervisor",
        _after_supervisor,
        {"memory_agent": "memory_agent", "summary": "summary"},
    )
    graph.add_edge("memory_agent", "summary")
    graph.add_edge("summary", END)
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
        "react_messages": [],
        "rag_called": False,
        "rag_call_count": 0,
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
            await queue.put({"type": "error", "data": {"code": "agent_graph_failed", "message": f"Agent 执行失败：{exc}"}})
        finally:
            await queue.put(_STREAM_DONE)

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
