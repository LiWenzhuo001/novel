"""面向中文小说 RAG 的 Query Preparation。"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.agent.types import DEFAULT_OUTPUT_POLICY, merge_output_policy
from app.config import settings
from app.core.llm import get_llm
from app.core.logging_config import get_logger

log = get_logger("query_rewriter")
_ALLOWED_INTENTS = {
    "character", "character_relation", "plot_causality", "timeline",
    "chapter_locator", "factual", "user_preference", "other",
}
_ALLOWED_ANSWER_MODES = {"novel_evidence", "memory_context", "conversation"}
_ALLOWED_CITATION_STYLES = {"chapter_only", "hidden", "normal"}
_ANSWER_MARKERS = ("答案：", "回答：", "综上", "因此可以看出", "分析如下", "总结：")
AnswerMode = Literal["novel_evidence", "memory_context", "conversation"]

_NO_SOURCE_RE = re.compile(
    r"(?:不要|不用|无需|别|不必).{0,12}(?:展示|显示|贴|引用|给我|输出).{0,10}(?:原文|引语|原话)"
    r"|只(?:要|需).{0,8}(?:总结|结论)|直接告诉我.{0,12}(?:总结|结论)"
)
_TEMP_SOURCE_RE = re.compile(r"(?:给我|展示|引用|贴出).{0,8}(?:原文|原话|依据)")


@dataclass(frozen=True)
class RewriteResult:
    original: str
    standalone_query: str
    retrieval_query: str
    applied: bool
    reason: str
    intent: str = "other"
    entities: list[str] = field(default_factory=list)
    evidence_focus: list[str] = field(default_factory=list)
    confidence: float = 0.0
    needs_retrieval: bool = True
    answer_mode: AnswerMode = "novel_evidence"
    retrieval_reason: str = "query_preparation_failed"
    output_policy: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_OUTPUT_POLICY))
    preference_update: dict[str, Any] | None = None

    @property
    def query(self) -> str:
        return self.standalone_query

    def as_dict(self) -> dict[str, Any]:
        return {
            "original": self.original,
            "standalone_query": self.standalone_query,
            "retrieval_query": self.retrieval_query,
            "applied": self.applied,
            "reason": self.reason,
            "intent": self.intent,
            "entities": list(self.entities),
            "evidence_focus": list(self.evidence_focus),
            "confidence": self.confidence,
            "needs_retrieval": self.needs_retrieval,
            "answer_mode": self.answer_mode,
            "retrieval_reason": self.retrieval_reason,
            "output_policy": dict(self.output_policy),
            "preference_update": self.preference_update,
            "prompt_version": getattr(settings, "query_routing_prompt_version", settings.query_rewrite_prompt_version),
        }


_NOVEL_REQUEST_RE = re.compile(
    r"人物|角色|关系|情节|剧情|事件|时间线|章节|回目|页码|片段|为什么|为何|谁|何时|后来|之后|"
    r"发生了什么|怎么回事|如何|是否正确|可靠吗|核对|验证|原文依据|根据原文"
)


def _explicit_output_policy(original: str) -> tuple[dict[str, Any], dict[str, Any] | None, bool, bool]:
    """识别当前轮展示偏好；返回 policy、更新、临时放宽、是否纯偏好。"""
    text = original.strip()
    policy = dict(DEFAULT_OUTPUT_POLICY)
    if _NO_SOURCE_RE.search(text):
        update = {
            "preference_key": "answer_presentation",
            "operation": "upsert",
            "value": dict(policy),
            "source": "explicit_user_instruction",
            "confidence": 0.99,
        }
        return policy, update, False, not bool(_NOVEL_REQUEST_RE.search(text))
    if _TEMP_SOURCE_RE.search(text):
        policy.update({
            "summary_only": False,
            "show_source_text": True,
            "allow_direct_quotes": True,
            "citation_style": "normal",
        })
        return policy, None, True, False
    return policy, None, False, False


def _history_text(history: Sequence[BaseMessage], limit: int) -> str:
    if limit <= 0:
        return ""
    lines: list[str] = []
    for message in list(history)[-limit:]:
        content = message.content if isinstance(message.content, str) else str(message.content)
        content = content.strip()
        if not content:
            continue
        role = "user" if isinstance(message, HumanMessage) else "assistant" if isinstance(message, AIMessage) else ""
        if role:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else item if isinstance(item, str) else ""
            for item in content
        ).strip()
    return str(content or "").strip()


def _extract_json(content: object) -> dict[str, Any] | None:
    if isinstance(content, dict):
        return content
    text = _content_text(content)
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start:end + 1])
    except (TypeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _as_string_list(value: object, *, limit: int, max_chars: int) -> list[str] | None:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > limit:
        return None
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            return None
        item = item.strip()
        if not item or len(item) > max_chars:
            return None
        result.append(item)
    return result


def _validate_policy(value: object, base: dict[str, Any]) -> dict[str, Any] | None:
    if value is None:
        return dict(base)
    if not isinstance(value, dict):
        return None
    policy = merge_output_policy(base, value)
    for key in ("summary_only", "show_source_text", "allow_direct_quotes", "show_citations", "show_agent_details"):
        if not isinstance(policy[key], bool):
            return None
    if policy["citation_style"] not in _ALLOWED_CITATION_STYLES:
        return None
    return policy


def _validate_preference_update(value: object, policy: dict[str, Any]) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or value.get("preference_key") != "answer_presentation":
        return None
    if value.get("operation", "upsert") != "upsert":
        return None
    stored = value.get("value", policy)
    validated = _validate_policy(stored, DEFAULT_OUTPUT_POLICY)
    if validated is None:
        return None
    try:
        confidence = float(value.get("confidence", 0.99))
    except (TypeError, ValueError):
        return None
    return {
        "preference_key": "answer_presentation",
        "operation": "upsert",
        "value": validated,
        "source": str(value.get("source") or "explicit_user_instruction")[:80],
        "confidence": max(0.0, min(1.0, confidence)),
    }


def _validate_payload(
    payload: dict[str, Any],
    original: str,
    history_text: str,
    memory_text: str = "",
    memory_policy: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, str]:
    standalone = payload.get("standalone_query")
    retrieval = payload.get("retrieval_query")
    intent = payload.get("intent", "other")
    confidence = payload.get("confidence", 0.0)
    needs_retrieval = payload.get("needs_retrieval", True)
    answer_mode = payload.get("answer_mode", "novel_evidence")
    retrieval_reason = payload.get("retrieval_reason", "legacy_query_preparation")
    explicit_policy, explicit_update, _, preference_only = _explicit_output_policy(original)
    base_policy = merge_output_policy(memory_policy, explicit_policy if explicit_update else None)

    if not isinstance(standalone, str) or not isinstance(retrieval, str):
        return None, "missing_query"
    standalone, retrieval = standalone.strip(), retrieval.strip()
    # An explicit preference-only message must never enter the novel evidence path,
    # even when an older model omits the new routing fields or returns a stale query.
    if explicit_update and preference_only:
        needs_retrieval = False
        answer_mode = "memory_context"
        retrieval_reason = "user_output_preference"
        retrieval = ""
        intent = "user_preference"
    if not standalone:
        return None, "empty_query"
    if len(standalone) > settings.query_rewrite_max_chars or len(retrieval) > settings.query_rewrite_max_chars:
        return None, "query_too_long"
    if not isinstance(intent, str) or intent not in _ALLOWED_INTENTS:
        return None, "invalid_intent"
    if not isinstance(needs_retrieval, bool):
        return None, "invalid_needs_retrieval"
    if not isinstance(answer_mode, str) or answer_mode not in _ALLOWED_ANSWER_MODES:
        return None, "invalid_answer_mode"
    if not isinstance(retrieval_reason, str) or not retrieval_reason.strip() or len(retrieval_reason) > 80:
        return None, "invalid_retrieval_reason"
    if needs_retrieval and not retrieval:
        return None, "missing_retrieval_query"
    if not needs_retrieval and retrieval:
        return None, "unexpected_retrieval_query"
    if needs_retrieval and answer_mode != "novel_evidence":
        return None, "inconsistent_answer_mode"
    if not needs_retrieval and answer_mode == "novel_evidence":
        return None, "inconsistent_answer_mode"
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        return None, "invalid_confidence"
    if not 0.0 <= confidence <= 1.0:
        return None, "invalid_confidence"
    entities = _as_string_list(payload.get("entities"), limit=12, max_chars=40)
    evidence_focus = _as_string_list(payload.get("evidence_focus"), limit=8, max_chars=60)
    if entities is None or evidence_focus is None:
        return None, "invalid_list"
    policy = _validate_policy(payload.get("output_policy"), base_policy)
    if policy is None:
        return None, "invalid_output_policy"
    preference_update = _validate_preference_update(payload.get("preference_update"), policy)
    if payload.get("preference_update") is not None and preference_update is None:
        return None, "invalid_preference_update"
    if explicit_update:
        preference_update = explicit_update
        policy = merge_output_policy(policy, explicit_update["value"])
        # A direct no-source instruction is a presentation policy even when the
        # same turn also asks a novel question; it must not disable RAG.
        policy["summary_only"] = True
        policy["show_source_text"] = False
        policy["allow_direct_quotes"] = False
        if not needs_retrieval:
            intent = "user_preference"

    source_text = f"{original}\n{history_text}\n{memory_text}"
    missing_entities = [item for item in entities if len(item) >= 2 and item not in source_text]
    if any(item in standalone for item in missing_entities):
        return None, "invented_entity"
    if any(marker in retrieval or marker in standalone for marker in _ANSWER_MARKERS):
        return None, "answer_like"
    if "\n" in retrieval or "\n" in standalone:
        return None, "multiline_query"
    return {
        "standalone_query": standalone,
        "retrieval_query": retrieval,
        "intent": intent,
        "entities": [item for item in entities if item not in missing_entities],
        "evidence_focus": evidence_focus,
        "confidence": round(confidence, 4),
        "needs_retrieval": needs_retrieval,
        "answer_mode": answer_mode,
        "retrieval_reason": retrieval_reason.strip(),
        "output_policy": policy,
        "preference_update": preference_update,
    }, "ok"


def _fallback(original: str, reason: str, memory_policy: dict[str, Any] | None = None) -> RewriteResult:
    policy, preference_update, _, preference_only = _explicit_output_policy(original)
    policy = merge_output_policy(memory_policy, policy)
    if preference_update and preference_only:
        return RewriteResult(
            original=original,
            standalone_query=original,
            retrieval_query="",
            applied=False,
            reason=reason,
            intent="user_preference",
            needs_retrieval=False,
            answer_mode="memory_context",
            retrieval_reason="user_output_preference",
            output_policy=policy,
            preference_update=preference_update,
        )
    return RewriteResult(
        original=original,
        standalone_query=original,
        retrieval_query=original,
        applied=False,
        reason=reason,
        needs_retrieval=True,
        answer_mode="novel_evidence",
        retrieval_reason="query_preparation_failed",
        output_policy=policy,
    )


def _prompt(original: str, history_text: str, *, correction: bool = False, memory_text: str = "", memory_policy: dict[str, Any] | None = None) -> str:
    history = history_text or "无（这是首轮问题）"
    memory = memory_text or "无可用记忆"
    policy = json.dumps(merge_output_policy(memory_policy), ensure_ascii=False)
    correction_hint = "上一轮输出未通过校验。请只输出合法 JSON，不要添加解释。\n" if correction else ""
    return f"""你是中文小说问答 Agent 的 Query Preparation 模块，不直接回答用户。
{correction_hint}
请基于用户问题、有限历史和记忆上下文，一次性完成：
1. standalone_query：解决指代和省略，供 Agent 理解用户意图；
2. retrieval_query：只有需要小说原文证据时才生成一条检索 Query，否则必须为空字符串；
3. needs_retrieval：判断本轮是否必须调用小说 RAG；
4. answer_mode：novel_evidence、memory_context 或 conversation；
5. retrieval_reason、output_policy、preference_update、intent、entities、evidence_focus、confidence。

需要 RAG：小说事实、人物关系、情节因果、时间线、章节定位、伏笔动机、原文依据、核对结论，
以及结合历史后仍指向小说内容的指代问题。边界不清或置信度不足时选择 RAG。
不需要 RAG：问候、感谢、闲聊、用户偏好、回答风格/格式/长度调整、纯会话承接和记忆操作。

特别注意：
- “不要展示原文，只给总结”是用户输出偏好，不是原文证据请求；
- “不要展示原文，但回答孙悟空为什么离开”仍需要 RAG，但 output_policy 必须禁止展示原文；
- 是否调用 RAG 与是否展示原文是两个独立决策；
- 历史和记忆只用于解决指代与保持偏好，不是小说原文证据；
- 当前已知用户输出偏好：{policy}。

严格约束：
- needs_retrieval=true 时 answer_mode 必须为 novel_evidence，retrieval_query 不得为空；
- needs_retrieval=false 时 answer_mode 必须为 memory_context 或 conversation，retrieval_query 必须为空；
- summary_only=true 时 show_source_text=false、allow_direct_quotes=false；
- standalone_query 不得添加原输入、历史和记忆中没有的新实体；
- 只输出合法 JSON，不要 Markdown 或答案文本；
- intent 必须是 character、character_relation、plot_causality、timeline、chapter_locator、factual、user_preference、other 之一；
- confidence 必须是 0 到 1 的数字。

输出格式：
{{
  "standalone_query": "...",
  "retrieval_query": "",
  "intent": "user_preference",
  "needs_retrieval": false,
  "answer_mode": "memory_context",
  "retrieval_reason": "user_output_preference",
  "output_policy": {json.dumps(DEFAULT_OUTPUT_POLICY, ensure_ascii=False)},
  "preference_update": null,
  "entities": [],
  "evidence_focus": [],
  "confidence": 0.0
}}

用户原始问题：
{original}

历史对话：
{history}

长期记忆：
{memory}
"""


async def rewrite_query(query: str, history: Sequence[BaseMessage], *, llm=None, memory_context: dict | None = None) -> RewriteResult:
    original = query.strip()
    if not original:
        return _fallback(query, "empty")
    memory_context = memory_context or {}
    memory_policy = memory_context.get("output_policy") if isinstance(memory_context.get("output_policy"), dict) else None
    if not settings.enable_query_rewrite:
        return _fallback(original, "disabled", memory_policy)

    history_text = _history_text(history, settings.query_rewrite_history_messages)
    memory_lines = [
        f"[{item.get('memory_type', 'memory')}] {item.get('content', '')}"
        for item in (memory_context.get("memories") or [])
        if item.get("content")
    ]
    memory_text = "\n".join(memory_lines)
    if memory_context.get("summary"):
        memory_text = f"会话摘要：{memory_context['summary']}\n" + memory_text
    model = llm or get_llm(
        temperature=0,
        max_tokens=settings.query_rewrite_max_tokens,
        timeout=settings.query_rewrite_timeout,
        max_retries=settings.query_rewrite_retries,
    )
    messages = [
        SystemMessage(content="你只负责生成结构化 Query Preparation、输出偏好和路由判断，不回答用户问题。"),
        HumanMessage(content=_prompt(original, history_text, memory_text=memory_text, memory_policy=memory_policy)),
    ]
    try:
        response = await asyncio.wait_for(model.ainvoke(messages), timeout=settings.query_rewrite_timeout)
        payload = _extract_json(getattr(response, "content", response))
        validated, reason = _validate_payload(payload or {}, original, history_text, memory_text, memory_policy)
        if validated is None:
            log.warning("query_rewrite.invalid", reason=reason)
            return _fallback(original, reason, memory_policy)
        return RewriteResult(
            original=original,
            standalone_query=validated["standalone_query"],
            retrieval_query=validated["retrieval_query"],
            applied=validated["standalone_query"] != original or validated["retrieval_query"] != original,
            reason="rewritten",
            intent=validated["intent"],
            entities=validated["entities"],
            evidence_focus=validated["evidence_focus"],
            confidence=validated["confidence"],
            needs_retrieval=validated["needs_retrieval"],
            answer_mode=validated["answer_mode"],
            retrieval_reason=validated["retrieval_reason"],
            output_policy=validated["output_policy"],
            preference_update=validated["preference_update"],
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        log.warning("query_rewrite.failed", error=str(exc)[:200])
        return _fallback(original, "error", memory_policy)
