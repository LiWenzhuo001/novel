"""LLM-assisted semantic routing with deterministic safety guards."""
from __future__ import annotations

import re
from typing import Any

from app.agent.types import AnswerMode, RouteDecision, Strategy, merge_output_policy
from app.config import settings

_NOVEL_SIGNALS = (
    "人物", "角色", "关系", "情节", "剧情", "事件", "时间线", "先后", "之前", "之后",
    "章节", "第几章", "页码", "片段", "梳理", "综合", "对比", "完整", "跨章节",
    "为什么", "为何", "伏笔", "动机", "因果", "谁", "何时", "后来", "结局",
    "根据原文", "原文依据", "核对", "验证", "可靠", "正确吗", "真实吗",
)
# 强信号闸专用子集：只收录明确指向小说内容的词，用于覆盖 LLM"不需要检索"的判定。
# 谁/为什么/之前/之后/事件等日常词保留在 _NOVEL_SIGNALS 供规则兜底与复杂度判断，
# 但不得进入强制闸，否则"你觉得谁写得更好"这类闲聊也会被强制检索。
_FORCED_SIGNALS = (
    "人物", "角色", "关系", "情节", "剧情", "时间线", "章节", "第几章", "页码",
    "片段", "跨章节", "伏笔", "动机", "因果", "结局", "根据原文", "原文依据",
    "核对", "验证", "梳理", "对比", "综合",
)
_CONVERSATION_SIGNALS = (
    "你好", "嗨", "谢谢", "感谢", "好的", "明白", "知道了", "记住", "记得我的",
    "我的偏好", "我的风格", "语气", "格式", "简短一点", "详细一点", "继续刚才",
    "刚才的风格", "不要引用", "不用检索", "闲聊", "只给总结", "直接告诉我总结",
)
_NEGATED_SOURCE_RE = re.compile(r"(?:不要|不用|无需|别|不必).{0,12}(?:展示|显示|贴|引用|给我|输出).{0,10}(?:原文|引语|原话)")
_NOVEL_REFERENCE_RE = re.compile(r"第\s*[一二三四五六七八九十0-9]+章|人物|角色|小说|这段|这个事件|这个情节|后来|为什么|为何|根据原文|原文依据|核对|验证")
_ALLOWED_ANSWER_MODES = {"novel_evidence", "memory_context", "conversation"}
_STRATEGY_ALIASES = {
    "direct": Strategy.DIRECT,
    "multi_expert": Strategy.MULTI_EXPERT,
    "react": Strategy.REACT,
    "plan_execute": Strategy.PLAN_EXECUTE,
}


def _routing_metadata(
    *,
    llm_needs_retrieval: bool | None,
    override: bool,
    reason: str,
    confidence: float | None,
) -> dict[str, Any]:
    """构造路由诊断字段，集中保证各分支输出结构一致。"""
    return {
        "llm_needs_retrieval": llm_needs_retrieval,
        "routing_override": override,
        "routing_override_reason": reason,
        "routing_confidence": confidence,
    }


def is_complex_query(query: str) -> bool:
    text = query.strip()
    return len(text) >= 32 or sum(1 for signal in _NOVEL_SIGNALS if signal in text) >= 2


def _rule_needs_retrieval(query: str) -> tuple[bool, str, AnswerMode]:
    text = query.strip()
    if not text:
        return False, "empty_conversation", "conversation"
    if _NEGATED_SOURCE_RE.search(text) and not _NOVEL_REFERENCE_RE.search(_NEGATED_SOURCE_RE.sub("", text)):
        return False, "user_output_preference", "memory_context"
    if any(signal in text for signal in _NOVEL_SIGNALS):
        return True, "novel_evidence", "novel_evidence"
    if any(signal in text for signal in _CONVERSATION_SIGNALS):
        if _NOVEL_REFERENCE_RE.search(text):
            return True, "ambiguous_novel_reference", "novel_evidence"
        return False, "conversation_only", "conversation"
    return True, "conservative_novel_lookup", "novel_evidence"


def _strong_novel_signal(query: str) -> bool:
    text = _NEGATED_SOURCE_RE.sub("", query.strip())
    return any(signal in text for signal in _FORCED_SIGNALS)


def _hint_is_valid(hint: dict[str, Any]) -> bool:
    try:
        confidence = float(hint.get("confidence"))
    except (TypeError, ValueError):
        return False
    return (
        isinstance(hint.get("needs_retrieval"), bool)
        and isinstance(hint.get("answer_mode"), str)
        and hint["answer_mode"] in _ALLOWED_ANSWER_MODES
        and isinstance(hint.get("retrieval_reason"), str)
        and bool(hint["retrieval_reason"].strip())
        and 0.0 <= confidence <= 1.0
        and isinstance(hint.get("retrieval_query"), str)
        and (hint["needs_retrieval"] == bool(hint["retrieval_query"].strip()))
        and (hint["needs_retrieval"] or hint["answer_mode"] != "novel_evidence")
        and (not hint["needs_retrieval"] or hint["answer_mode"] == "novel_evidence")
    )


def _routing_choice(query: str, routing_hint: dict[str, Any] | None):
    hint = routing_hint if isinstance(routing_hint, dict) else {}
    policy = merge_output_policy(hint.get("output_policy"))
    preference_update = hint.get("preference_update")

    if not settings.enable_llm_query_routing or not routing_hint:
        needs, reason, mode = _rule_needs_retrieval(query)
        return needs, reason, mode, policy, preference_update, _routing_metadata(
            llm_needs_retrieval=None,
            override=False,
            reason="llm_routing_disabled_or_unavailable",
            confidence=None,
        )
    if not _hint_is_valid(routing_hint):
        return True, "query_preparation_failed", "novel_evidence", policy, preference_update, _routing_metadata(
            llm_needs_retrieval=None,
            override=True,
            reason="invalid_query_preparation",
            confidence=None,
        )
    if routing_hint.get("reason") != "rewritten":
        return True, "query_preparation_failed", "novel_evidence", policy, preference_update, _routing_metadata(
            llm_needs_retrieval=None,
            override=True,
            reason=str(routing_hint.get("reason") or "query_preparation_failed"),
            confidence=float(routing_hint.get("confidence", 0.0)),
        )

    llm_needs = bool(routing_hint["needs_retrieval"])
    confidence = float(routing_hint["confidence"])
    mode: AnswerMode = routing_hint["answer_mode"]
    reasons: list[str] = []
    # 低置信度强制闸分级：会话/记忆类回答漏检索代价小，尊重大模型判定；
    # 只有明确走小说证据路径的判定才维持严格置信度闸。
    if mode == "novel_evidence" and confidence < settings.query_routing_confidence_threshold:
        reasons.append("low_confidence")
    if _strong_novel_signal(f"{query} {routing_hint.get('original', '')}"):
        reasons.append("strong_novel_signal")
    if mode == "novel_evidence":
        reasons.append("novel_evidence_mode")
    if llm_needs:
        return True, str(routing_hint["retrieval_reason"]), mode, policy, preference_update, _routing_metadata(
            llm_needs_retrieval=True,
            override=False,
            reason="",
            confidence=confidence,
        )
    if reasons:
        return True, "forced_by_" + reasons[0], "novel_evidence", policy, preference_update, _routing_metadata(
            llm_needs_retrieval=False,
            override=True,
            reason=";".join(reasons),
            confidence=confidence,
        )
    return False, str(routing_hint["retrieval_reason"]), mode, policy, preference_update, _routing_metadata(
        llm_needs_retrieval=False,
        override=False,
        reason="",
        confidence=confidence,
    )


def normalize_strategy(requested_strategy: str | None, query: str) -> Strategy:
    """将用户输入归一化为受支持的执行策略；未知值保持旧的 direct 回退。"""
    value = (requested_strategy or "auto").strip().lower()
    if value == "auto":
        return Strategy.MULTI_EXPERT if is_complex_query(query) else Strategy.DIRECT
    return _STRATEGY_ALIASES.get(value, Strategy.DIRECT)


def _decision_metadata(
    *,
    needs_retrieval: bool,
    reason: str,
    answer_mode: AnswerMode,
    output_policy: dict[str, Any],
    preference_update: dict[str, Any] | None,
    routing: dict[str, Any],
) -> dict[str, Any]:
    """返回 RouteDecision 的公共字段，避免不同策略分支漂移。"""
    return {
        "requires_citation": needs_retrieval,
        "needs_retrieval": needs_retrieval,
        "retrieval_reason": reason,
        "answer_mode": answer_mode,
        "output_policy": output_policy,
        "preference_update": preference_update,
        **routing,
    }


def route_query(query: str, requested_strategy: str | None = None, routing_hint: dict[str, Any] | None = None) -> RouteDecision:
    strategy = normalize_strategy(requested_strategy, query)
    needs_retrieval, reason, answer_mode, output_policy, preference_update, routing = _routing_choice(query, routing_hint)
    kwargs = _decision_metadata(
        needs_retrieval=needs_retrieval,
        reason=reason,
        answer_mode=answer_mode,
        output_policy=output_policy,
        preference_update=preference_update,
        routing=routing,
    )
    if strategy is Strategy.DIRECT:
        intent = "fact_lookup" if needs_retrieval else "conversation"
        tools = ("retrieve_novel",) if needs_retrieval else ()
        return RouteDecision(intent, strategy, tools, 2, **kwargs)
    if strategy is Strategy.MULTI_EXPERT:
        if not needs_retrieval:
            return RouteDecision(
                "conversation",
                Strategy.DIRECT,
                (),
                2,
                **{
                    **kwargs,
                    "requires_citation": False,
                    "needs_retrieval": False,
                    "retrieval_reason": f"{reason};strategy_downgraded",
                },
            )
        return RouteDecision("novel_analysis", strategy, ("retrieve_novel", "get_chapter_context"), max(3, settings.agent_max_steps), **kwargs)
    tools = ("retrieve_novel", "get_chapter_context", "calculator") if needs_retrieval else ("calculator",)
    max_steps = max(3, settings.agent_max_steps) if needs_retrieval else 2
    intent = "tool_augmented_question" if strategy is Strategy.REACT else "multi_step_novel_question"
    return RouteDecision(intent, strategy, tools, max_steps, **kwargs)
