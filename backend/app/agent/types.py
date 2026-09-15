"""Agent 运行时使用的状态、策略、预算和工具结果类型定义。"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, TypedDict

AnswerMode = Literal["novel_evidence", "memory_context", "conversation", "roleplay"]
CitationStyle = Literal["chapter_only", "hidden", "normal"]
# 三档检索政策（运行时硬约束）：
# required=执行答案生成前必须至少完成一次成功检索（预检索兜底）；
# optional=模型在工具预算内自主决定是否检索；
# forbidden=纯会话/偏好，检索工具不进入模型可见工具列表。
RetrievalPolicy = Literal["required", "optional", "forbidden"]

DEFAULT_OUTPUT_POLICY: dict[str, Any] = {
    "summary_only": True,
    "show_source_text": False,
    "allow_direct_quotes": False,
    "show_citations": True,
    "citation_style": "chapter_only",
    # 内部过程（计划、工具调用）默认展示，隐藏会让执行过程看起来毫无产出。
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
    """运行时实际执行的策略；auto 在路由层必须解析为其中之一。"""

    DIRECT = "direct"
    REACT = "react"
    PLAN_EXECUTE = "plan_execute"


class InteractionMode(StrEnum):
    """交互模式：普通问答 / 角色扮演。角色扮演不是策略，而是工具与提示的差异。"""

    QA = "qa"
    ROLEPLAY = "roleplay"


@dataclass(frozen=True)
class ExecutionBudget:
    """单次运行的统一执行预算；计数发生在运行时代码，不依赖 Prompt 声明。

    max_steps：工具调用步数上限（每次成功发起的工具调用计 1 步）；
    max_tool_calls：总工具调用次数硬上限；
    max_retrieval_calls：RAG 检索类工具调用上限（防重复检索滚雪球）；
    max_plan_adjustments：plan_execute 计划重排次数上限；
    deadline_seconds：图执行墙钟上限（超时由外层 asyncio.timeout 兜底）。
    """

    max_steps: int
    max_tool_calls: int
    max_retrieval_calls: int
    max_plan_adjustments: int = 1
    deadline_seconds: float = 240.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_steps": self.max_steps,
            "max_tool_calls": self.max_tool_calls,
            "max_retrieval_calls": self.max_retrieval_calls,
            "max_plan_adjustments": self.max_plan_adjustments,
            "deadline_seconds": self.deadline_seconds,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "ExecutionBudget":
        data = value or {}
        return cls(
            max_steps=int(data.get("max_steps", 6)),
            max_tool_calls=int(data.get("max_tool_calls", 8)),
            max_retrieval_calls=int(data.get("max_retrieval_calls", 4)),
            max_plan_adjustments=int(data.get("max_plan_adjustments", 1)),
            deadline_seconds=float(data.get("deadline_seconds", 240.0)),
        )


class AgentState(TypedDict, total=False):
    """LangGraph 在各节点之间传递的共享状态。字段允许按执行路径逐步填充。"""
    query: str
    session_id: str | None
    interaction_mode: str
    memory_agent_active: bool
    memory_ops: list
    original_query: str
    standalone_query: str
    retrieval_query: str
    query_preparation: dict[str, Any]
    # 兼容旧调用方，正式语义为 query_preparation。
    query_rewrite: dict[str, Any]
    memory_context: dict[str, Any]
    needs_retrieval: bool
    retrieval_reason: str
    retrieval_policy: str
    answer_mode: str
    output_policy: dict[str, Any]
    preference_update: dict[str, Any] | None
    synthesis_context: dict[str, Any]
    llm_needs_retrieval: bool | None
    routing_override: bool
    routing_override_reason: str
    routing_confidence: float | None
    file_id: str | None
    # 主 Agent 模型决策轮次（1 起计，每个决策轮 +1）：与工具调用计数、
    # 计划步骤编号语义分离——并行 RAG 数量不影响推理轮次，前端展示专用。
    reasoning_round: int
    # 角色扮演上下文：受信任数据由 API 层注入，模型不可见、不可改。
    personas: list[str]
    chapter_until: int | None
    roleplay_history: list[Any]
    requested_strategy: str
    requested_max_steps: int | None
    strategy: str
    strategy_adjusted: bool
    adjustment_reason: str
    deprecations: list[str]
    intent: str
    max_steps: int
    budget: dict[str, Any]
    current_step: int
    allowed_tools: list[str]
    plan: list[dict[str, Any]]
    plan_committed: bool
    plan_adjusted: bool
    # 角色扮演：load_character_context 装载的角色卡与开场情景。
    character_cards: list[dict[str, Any]]
    scenario: str
    # ReAct 循环状态：模型判定证据充分（react_done）或 plan_execute 已产出计划。
    react_done: bool
    # Agent 决策循环跨轮消息（模型 response 与 ToolMessage 回灌）与执行统计。
    react_messages: list
    rag_called: bool
    rag_call_count: int
    tool_calls_used: int
    stop_reason: str
    observations: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    sources: list[dict[str, Any]]
    answer: str
    # compose_answer 产出的候选答案；verify_answer 校验后才允许输出/持久化。
    candidate_answer: str
    candidate_streamed: bool
    verification: dict[str, Any]
    repairs_used: int
    fallback_reason: str
    status: str
    event_queue: Any


@dataclass(frozen=True)
class RouteDecision:
    """路由节点输出的执行策略、工具白名单和预算。"""
    intent: str
    strategy: Strategy
    allowed_tools: tuple[str, ...]
    max_steps: int
    requires_citation: bool = True
    needs_retrieval: bool = True
    retrieval_policy: str = "optional"
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
            "retrieval_policy": self.retrieval_policy,
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
