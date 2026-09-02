"""LangGraph Agent Runtime with query decomposition, validation and live SSE events."""
from __future__ import annotations

import asyncio
import re
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from app.agent.contracts import EXPERT_CONTRACTS, SPECIALIST_ORDER, SpecialistContract
from app.agent.dispatcher import dispatch_expert_tasks
from app.agent.router import route_query
from app.agent.tools import registry
from app.agent.types import AgentState, DEFAULT_OUTPUT_POLICY, Strategy
from app.agent.validation import validate_reports
from app.config import settings
from app.core.llm import get_llm
from app.core.logging_config import get_logger
from app.core.metrics import metrics

log = get_logger("agent_runtime")
_EMPTY_MESSAGE = "当前小说知识库中没有检索到足以回答该问题的原文。请确认作品已完成索引，或补充人物名、事件名、章节等线索。"
_STREAM_DONE = object()


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
    if strategy is Strategy.REACT:
        return [
            {"step": 1, "action": "retrieve_novel", "purpose": "检索与问题直接相关的小说原文"},
            {"step": 2, "action": "reflect", "purpose": "判断证据是否足以回答"},
            {"step": 3, "action": "supervisor", "purpose": "生成带引用的最终答案"},
        ]
    if strategy is Strategy.PLAN_EXECUTE:
        return [
            {"step": 1, "action": "retrieve_novel", "purpose": "召回主要证据"},
            {"step": 2, "action": "get_chapter_context", "purpose": "补充命中章节的前后文"},
            {"step": 3, "action": "reflect", "purpose": "检查证据完整性"},
            {"step": 4, "action": "supervisor", "purpose": "生成带引用的最终答案"},
        ]
    return [
        {"step": 1, "action": "retrieve_novel", "purpose": "召回答案所需的小说原文"},
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
        "retrieval_skipped": not decision.needs_retrieval,
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
    """根据路由结果写入可展示的执行计划；非 RAG 问题跳过检索步骤。"""
    if not state.get("needs_retrieval", True):
        plan = [{
            "step": 1,
            "action": "supervisor",
            "purpose": "基于会话和记忆上下文直接回答，无需检索小说原文",
        }]
    else:
        plan = _plan(Strategy(state["strategy"]))
    await _emit(state, "plan", {
        "steps": plan,
        "max_steps": state["max_steps"],
        "retrieval_skipped": not state.get("needs_retrieval", True),
    })
    return {"plan": plan}


def _after_plan(state: AgentState) -> str:
    """路由节点决定是否进入共享小说 RAG。"""
    return "retrieve" if state.get("needs_retrieval", True) else "supervisor"


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
    }


def _after_retrieve(state: AgentState) -> str:
    """根据当前策略和检索结果选择专家、执行或直接汇总分支。"""
    strategy = Strategy(state["strategy"])
    # 多专家分支先完成一次共享检索，再拆分任务，避免四个专家重复调用 RAG。
    if strategy is Strategy.MULTI_EXPERT:
        return "dispatch"
    if strategy is Strategy.DIRECT:
        return "supervisor"
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


async def _run_specialist(
    contract: SpecialistContract,
    state: AgentState,
    agent_id: str,
    *,
    retry: int = 0,
    correction: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """执行单个专家并收集流式报告；失败只影响当前专家。"""
    started = time.perf_counter()
    first_token_ms: float | None = None
    parts: list[str] = []
    try:
        async for token in _stream_llm([
            SystemMessage(content="你只完成被分配的专家子任务，共享原文是唯一事实边界。"),
            HumanMessage(content=_specialist_prompt(contract, state, correction)),
        ], settings.agent_expert_max_tokens):
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
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        await _emit(state, "tool_end", {
            "id": agent_id,
            "tool": "specialist",
            "agent": contract.name,
            "label": contract.label,
            "status": "corrected" if retry else "ok",
            "summary": f"{contract.label}{'纠偏' if retry else '分析'}完成",
            "latency_ms": latency_ms,
            "first_token_ms": first_token_ms,
            "retry": retry,
        })
        return {
            "agent": contract.name,
            "label": contract.label,
            "status": "ok",
            "report": "".join(parts),
            "latency_ms": latency_ms,
            "first_token_ms": first_token_ms,
            "corrected": bool(retry),
        }
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
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


async def _execute_node(state: AgentState) -> dict[str, Any]:
    """执行 react 或 plan_execute 策略中的下一步工具调用。"""
    evidence = list(state.get("evidence", []))
    observations = list(state.get("observations", []))
    current_step = state.get("current_step", 1)
    fallback = state.get("fallback_reason", "")
    for item in state.get("plan", []):
        action = item["action"]
        if action not in {"get_chapter_context", "calculator"}:
            continue
        if current_step >= state["max_steps"]:
            fallback = "step_budget_exceeded"
            break
        current_step += 1
        call_id = f"agent-step-{current_step}"
        await _emit(state, "step_start", {"step": current_step, "action": action, "purpose": item["purpose"]})
        await _emit(state, "tool_start", {"id": call_id, "tool": action, "label": item["purpose"], "step": current_step})
        result = await registry.execute(
            action,
            allowed_tools=state["allowed_tools"],
            query=state["standalone_query"],
            file_id=state.get("file_id"),
        )
        observations.append(result.as_dict())
        if result.status == "ok" and isinstance(result.output, dict):
            evidence.extend(result.output.get("evidence", []))
        await _emit(state, "observation", {"step": current_step, **result.as_dict()})
        await _emit(state, "tool_end", {
            "id": call_id,
            "tool": action,
            "label": item["purpose"],
            "step": current_step,
            "status": result.status,
            "summary": result.error_code or f"完成，耗时 {result.latency_ms}ms",
        })
        if result.status != "ok":
            fallback = result.error_code or "tool_failed"
            break
    normalized, sources = _normalize_evidence(evidence)
    if sources != state.get("sources", []):
        await _emit(state, "sources", sources)
    return {
        "evidence": normalized,
        "sources": sources,
        "observations": observations,
        "current_step": current_step,
        "fallback_reason": fallback,
    }


async def _reflect_node(state: AgentState) -> dict[str, Any]:
    """检查多步执行结果是否足以进入最终汇总。"""
    if state.get("fallback_reason"):
        decision, reason = "fallback", state["fallback_reason"]
    elif state.get("evidence"):
        decision, reason = "final", "已有可引用证据"
    else:
        decision, reason = "fallback", "没有召回有效证据"
    await _emit(state, "reflection", {"decision": decision, "reason": reason, "step": state.get("current_step", 0)})
    return {"fallback_reason": state.get("fallback_reason") or ("empty_retrieval" if not state.get("evidence") else "")}


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
    if state.get("answer_mode", "novel_evidence") == "novel_evidence" and not evidence:
        fallback = fallback or "empty_retrieval"
    synthesis_context = {
        "question": state.get("standalone_query", ""),
        "evidence": evidence,
        "sources": state.get("sources", []),
        "reports": successful,
        "report_validation": validations,
        "summary": (state.get("memory_context") or {}).get("summary", ""),
        "memories": (state.get("memory_context") or {}).get("memories", []),
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
    answer_mode = state.get("answer_mode", "novel_evidence")
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
            prompt = (
                f"{policy_text}\n请基于经过内部校验的小说证据回答用户问题。事实优先于推断；"
                "若证据不足请明确说明。不要展示内部过程。关键事实可使用 [S#]，但严格遵守展示策略。\n\n"
                f"问题：{context.get('question', '')}\n\n共享证据：\n{_evidence_text(evidence)}\n\n"
                f"专家内部结论：\n{reports_text}\n\n会话摘要：\n{summary_text}\n\n长期记忆：\n{memory_text}"
            )
            system = "你是严谨的小说问答总结助手。"
        else:
            prompt = (
                f"{policy_text}\n当前问题不需要检索小说原文。请基于会话摘要和长期记忆自然回答，"
                "不要编造小说事实，不要生成 [S#]。如果用户是在设置偏好，简洁确认即可。\n\n"
                f"问题：{context.get('question', '')}\n\n会话摘要：\n{summary_text}\n\n长期记忆：\n{memory_text}"
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
        "needs_retrieval": state.get("needs_retrieval", True),
        "retrieval_skipped": not state.get("needs_retrieval", True),
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
    }
    await _emit(state, "meta", meta)
    return {"answer": answer, "fallback_reason": fallback, "status": "completed", "current_step": step}


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
    graph.add_node("summary", _summary_node)
    graph.add_edge(START, "route")
    graph.add_edge("route", "plan")
    graph.add_conditional_edges(
        "plan",
        _after_plan,
        {"retrieve": "retrieve", "supervisor": "supervisor"},
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
    graph.add_edge("execute", "reflect")
    graph.add_edge("reflect", "supervisor")
    graph.add_edge("supervisor", "summary")
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
