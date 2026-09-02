"""Agent 运行时使用的状态、策略和工具结果类型定义。"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, TypedDict

AnswerMode = Literal["novel_evidence", "memory_context", "conversation"]
CitationStyle = Literal["chapter_only", "hidden", "normal"]

DEFAULT_OUTPUT_POLICY: dict[str, Any] = {
    "summary_only": True,
    "show_source_text": False,
    "allow_direct_quotes": False,
    "show_citations": True,
    "citation_style": "chapter_only",
    # 专家分析过程默认展示：报告已按维度互斥生成，隐藏会让"调用过程"里的
    # 子智能体看起来毫无产出（前端按该字段 === true 渲染流式文本）。
    "show_agent_details": True,
}


def merge_output_policy(*policies: dict[str, Any] | None) -> dict[str, Any]:
    """按默认键合并输出策略，忽略未知字段并始终返回新字典。"""
    merged = dict(DEFAULT_OUTPUT_POLICY)
    for policy in policies:
        if not isinstance(policy, dict):
            continue
        merged.update({key: policy[key] for key in DEFAULT_OUTPUT_POLICY if key in policy})
    return merged


class Strategy(StrEnum):
    """Public execution strategies exposed by the Agent API."""

    DIRECT = "direct"
    MULTI_EXPERT = "multi_expert"
    REACT = "react"
    PLAN_EXECUTE = "plan_execute"


class AgentState(TypedDict, total=False):
    """LangGraph 在各节点之间传递的共享状态。字段允许按执行路径逐步填充。"""
    query: str
    original_query: str
    standalone_query: str
    retrieval_query: str
    query_preparation: dict[str, Any]
    # 兼容旧调用方，正式语义为 query_preparation。
    query_rewrite: dict[str, Any]
    memory_context: dict[str, Any]
    needs_retrieval: bool
    retrieval_reason: str
    answer_mode: str
    output_policy: dict[str, Any]
    preference_update: dict[str, Any] | None
    synthesis_context: dict[str, Any]
    llm_needs_retrieval: bool | None
    routing_override: bool
    routing_override_reason: str
    routing_confidence: float | None
    file_id: str | None
    requested_strategy: str
    requested_max_steps: int | None
    strategy: str
    intent: str
    max_steps: int
    current_step: int
    max_experts: int
    allowed_tools: list[str]
    plan: list[dict[str, Any]]
    observations: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    sources: list[dict[str, Any]]
    assignments: list[str]
    expert_tasks: dict[str, dict[str, Any]]
    dispatch_mode: str
    dispatch_reason: str
    reports: list[dict[str, Any]]
    report_validation: dict[str, dict[str, Any]]
    refine_agents: list[str]
    expert_retry_count: dict[str, int]
    answer: str
    fallback_reason: str
    status: str
    event_queue: Any


@dataclass(frozen=True)
class RouteDecision:
    """路由节点输出的执行策略、工具白名单和最大步数。"""
    intent: str
    strategy: Strategy
    allowed_tools: tuple[str, ...]
    max_steps: int
    requires_citation: bool = True
    needs_retrieval: bool = True
    retrieval_reason: str = "novel_evidence"
    answer_mode: AnswerMode = "novel_evidence"
    output_policy: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_OUTPUT_POLICY))
    preference_update: dict[str, Any] | None = None
    llm_needs_retrieval: bool | None = None
    routing_override: bool = False
    routing_override_reason: str = ""
    routing_confidence: float | None = None

    def as_dict(self) -> dict[str, Any]:
        """将路由决策转换为可通过 SSE 发送的普通字典。"""
        return {
            "intent": self.intent,
            "strategy": self.strategy.value,
            "allowed_tools": list(self.allowed_tools),
            "max_steps": self.max_steps,
            "requires_citation": self.requires_citation,
            "needs_retrieval": self.needs_retrieval,
            "retrieval_reason": self.retrieval_reason,
            "answer_mode": self.answer_mode,
            "output_policy": dict(self.output_policy),
            "preference_update": self.preference_update,
            "llm_needs_retrieval": self.llm_needs_retrieval,
            "routing_override": self.routing_override,
            "routing_override_reason": self.routing_override_reason,
            "routing_confidence": self.routing_confidence,
        }


@dataclass
class ToolResult:
    """Agent 工具的统一执行结果，包含状态、输出、引用和耗时。"""
    status: Literal["ok", "error", "timeout", "denied"]
    output: Any = None
    citations: list[dict[str, Any]] = field(default_factory=list)
    error_code: str | None = None
    latency_ms: float = 0.0
    tool: str = ""

    def as_dict(self) -> dict[str, Any]:
        """将工具结果转换为事件和日志可使用的字典。"""
        return {
            "status": self.status,
            "output": self.output,
            "citations": self.citations,
            "error_code": self.error_code,
            "latency_ms": self.latency_ms,
            "tool": self.tool,
        }
