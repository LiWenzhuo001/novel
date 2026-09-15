"""Agent 工具注册、权限控制、超时处理和小说检索工具。

写类工具（记忆增删改）不在回答路径直接落库：registry 层将其转为高优先级
memory_jobs 任务，由独立 worker 消费——回答过程与数据写入解耦。
"""
from __future__ import annotations

import ast
import asyncio
import json
import operator
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from langchain_core.tools import tool

from app.agent.types import ToolResult
from app.config import settings
from app.core.context import get_memory_session
from app.core.logging_config import get_logger
from app.core.rag import retrieve_novel_context
from app.services import memory_service, world_service

log = get_logger("agent_tools")

ToolHandler = Callable[..., Awaitable[ToolResult]]


@dataclass(frozen=True)
class ToolSpec:
    """工具的静态描述：超时、权限、幂等、重试语义、成本档与适用交互模式。"""
    name: str
    description: str
    timeout_seconds: float = 20.0
    permission: str = "read"
    idempotent: bool = True
    # 失败是否允许执行环重试一次（denied 永不重试）。
    retryable: bool = False
    cost_class: str = "low"  # low | medium | high
    supported_modes: frozenset = field(default_factory=lambda: frozenset({"qa", "roleplay"}))


class ToolRegistry:
    """按名称保存工具定义和处理器，并统一执行权限、超时和异常转换。"""
    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self._handlers: dict[str, ToolHandler] = {}

    def register(self, spec: ToolSpec, handler: ToolHandler) -> None:
        """注册或覆盖一个工具处理器。"""
        self._specs[spec.name] = spec
        self._handlers[spec.name] = handler

    def specs(self) -> list[ToolSpec]:
        return list(self._specs.values())

    def spec(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def tools_for_mode(self, interaction_mode: str) -> list[str]:
        """返回对指定交互模式可见的工具名（supported_modes 硬过滤）。"""
        return [
            name for name, spec in self._specs.items()
            if interaction_mode in spec.supported_modes
        ]

    async def execute(self, name: str, *, allowed_tools: list[str] | tuple[str, ...], **kwargs: Any) -> ToolResult:
        """执行指定工具，并将拒绝、超时和异常统一转换为 ToolResult。"""
        if name not in self._specs or name not in allowed_tools:
            return ToolResult(status="denied", error_code="tool_not_allowed", tool=name)
        started = time.perf_counter()
        spec = self._specs[name]
        try:
            result = await asyncio.wait_for(self._handlers[name](**kwargs), timeout=spec.timeout_seconds)
        except asyncio.TimeoutError:
            log.error("tool.failed", tool=name, error_code="tool_timeout", output="工具执行超时")
            return ToolResult(
                status="timeout",
                error_code="tool_timeout",
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
                tool=name,
            )
        except Exception as exc:  # noqa: BLE001
            # 失败必须带消息落日志：此前异常类型名之外的信息全部丢失，
            # 线上只能看到 "RuntimeError" 这类空指针式线索，无法定位根因。
            log.error("tool.failed", tool=name, error_code=type(exc).__name__, output=str(exc)[:200])
            return ToolResult(
                status="error",
                error_code=type(exc).__name__,
                output=str(exc)[:200],
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
                tool=name,
            )
        result.tool = name
        result.latency_ms = round((time.perf_counter() - started) * 1000, 1)
        return result


async def _retrieve_novel(
    *,
    query: str,
    retrieval_query: str | None = None,
    file_id: str | None = None,
    neighbor_window: int | None = None,
    chapter_until: int | None = None,
    **_: Any,
) -> ToolResult:
    """调用共享小说 RAG，生成专家和 Supervisor 共用的 evidence 与 sources。"""
    retrieve_kwargs = {
        "k": settings.novel_context_k,
        "neighbor_window": neighbor_window,
        "file_id": file_id,
    }
    # 仅在确实存在改写 Query 时传递新参数，兼容旧的工具替身和外部调用方。
    if retrieval_query:
        retrieve_kwargs["retrieval_query"] = retrieval_query
    # 角色扮演的剧情时间线：人物记忆只召回截至章节及之前的内容。
    if chapter_until is not None:
        retrieve_kwargs["chapter_until"] = chapter_until
    docs = await retrieve_novel_context(query, **retrieve_kwargs)
    sources: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for index, doc in enumerate(docs, start=1):
        meta = doc.metadata
        source_name = str(meta.get("source", "未知"))
        source_type = str(meta.get("source_type") or source_name.rsplit(".", 1)[-1]).lower()
        has_real_page = bool(meta.get("has_real_page", source_type == "pdf"))
        source = {
            "id": f"S{index}",
            "source": source_name,
            "source_type": source_type,
            "chapter": meta.get("chapter"),
            "chapter_no": meta.get("chapter_no"),
            "page": meta.get("page") if has_real_page else None,
            "chunk_no": meta.get("chunk_no"),
            "char_start": meta.get("char_start"),
            "char_end": meta.get("char_end"),
            "score": float(meta.get("score", 0.0) or 0.0),
            "score_type": meta.get("score_type"),
            "neighbor": bool(meta.get("neighbor", False)),
            "retrieval_rank": meta.get("retrieval_rank"),
            "vector_score": meta.get("vector_score"),
            "fts_score": meta.get("fts_score"),
            "rrf_score": meta.get("rrf_score"),
            "reranked": bool(meta.get("reranked", False)),
            "snippet": doc.page_content[:240].strip(),
        }
        sources.append(source)
        evidence.append({"source": source, "content": doc.page_content})
    return ToolResult(status="ok", output={"evidence": evidence, "sources": sources}, citations=sources)


async def _chapter_context(*, query: str, file_id: str | None = None, chapter_until: int | None = None, **_: Any) -> ToolResult:
    """以较大的邻居窗口检索命中章节的前后文。"""
    return await _retrieve_novel(query=query, file_id=file_id, neighbor_window=2, chapter_until=chapter_until)


_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float:
    """递归计算受限 AST 表达式；不执行变量、函数调用或任意 Python 代码。"""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 8:
            raise ValueError("exponent_too_large")
        return _ALLOWED_OPERATORS[type(node.op)](left, right)
    raise ValueError("unsupported_expression")


async def _calculator(*, expression: str, **_: Any) -> ToolResult:
    """执行长度和运算符均受限的数值计算工具。"""
    if not expression or len(expression) > 100:
        return ToolResult(status="error", error_code="invalid_expression")
    try:
        value = _safe_eval(ast.parse(expression, mode="eval"))
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
        return ToolResult(status="error", error_code="unsupported_expression")
    return ToolResult(status="ok", output={"expression": expression, "value": value})


def build_default_registry() -> ToolRegistry:
    """创建并注册当前 Agent Runtime 可用的默认工具集合。"""
    registry = ToolRegistry()
    registry.register(ToolSpec(
        "retrieve_novel", "混合检索小说原文并返回引用",
        timeout_seconds=settings.agent_tool_timeout, retryable=True, cost_class="medium",
    ), _retrieve_novel)
    registry.register(ToolSpec(
        "get_chapter_context", "检索命中章节的相邻片段",
        timeout_seconds=settings.agent_tool_timeout, retryable=True, cost_class="medium",
    ), _chapter_context)
    registry.register(ToolSpec(
        "calculator", "执行受限数值计算", timeout_seconds=settings.agent_tool_timeout, cost_class="low",
    ), _calculator)
    registry.register(ToolSpec(
        "load_character_context", "装载登场角色卡与开场情景",
        timeout_seconds=_CHARACTER_CONTEXT_TIMEOUT, retryable=True, cost_class="high",
        supported_modes=frozenset({"roleplay"}),
    ), _load_character_context_impl)
    _register_memory_tools(registry)
    return registry


# ===== 角色扮演工具（interaction_mode=roleplay 专用；personas/chapter_until 由受信任状态注入） =====

# 角色卡生成包含检索 + 多次 LLM 调用，普通工具超时不够；命中缓存时通常毫秒级。
_CHARACTER_CONTEXT_TIMEOUT = 90.0


async def _load_character_context_impl(
    *, file_id: str | None = None, personas: list[str] | None = None,
    chapter_until: int | None = None, **_: Any,
) -> ToolResult:
    """装载角色卡与开场情景；模型不提供任何参数，全部由执行器从状态注入。"""
    if not file_id or not personas:
        return ToolResult(status="error", error_code="missing_roleplay_context",
                          output="（缺少角色扮演上下文：personas 或 file_id 未注入）")
    result = await world_service.get_character_cards(file_id, personas, chapter_until)
    return ToolResult(
        status="ok",
        output={"cards": result["cards"], "scenario": result["scenario"], "chapter_until": chapter_until},
    )


# ===== 记忆工具（模型自主发起，memory_agent 节点执行） =====
# 会话上下文（session_id/file_id）由记忆决策节点通过 ContextVar 注入，
# 工具 schema 只暴露业务参数；handler 的 **_ 吞掉模型多余参数。


async def _search_memories_impl(query: str, **_: Any) -> ToolResult:
    ctx = get_memory_session()
    if ctx is None:
        return ToolResult(status="ok", output={"message": "（会话上下文不可用，无法检索记忆）"})
    session_id, file_id = ctx
    rows = await memory_service.retrieve_memories(query=query, session_id=session_id, file_id=file_id)
    if not rows:
        return ToolResult(status="ok", output={"message": "（没有找到相关记忆）"})
    listing = "\n".join(f"- id={row.id} [{row.memory_type}] {row.content}" for row in rows)
    return ToolResult(status="ok", output={
        "message": listing, "count": len(rows), "memory_ids": [row.id for row in rows],
    })


async def _add_memory_impl(content: str, memory_type: str = "session_fact", importance: float = 0.7, **_: Any) -> ToolResult:
    ctx = get_memory_session()
    if ctx is None:
        return ToolResult(status="error", error_code="no_session_context", output="（会话上下文不可用，无法保存记忆）")
    session_id, file_id = ctx
    if memory_type not in {"user_preference", "novel_fact", "session_fact"}:
        return ToolResult(status="error", error_code="invalid_memory_type",
                          output=f"（无效的 memory_type：{memory_type}）")
    # 回答路径零直接写库：入队高优先级任务，由 worker 校验并执行。
    job_id = await memory_service.enqueue_memory_job(
        kind="memory_op",
        payload={"op": "add", "content": content, "memory_type": memory_type, "importance": importance},
        session_id=session_id,
        file_id=file_id,
        priority="high",
    )
    return ToolResult(status="ok", output={
        "message": f"已提交记忆写入（后台生效）：{content}", "job_id": job_id, "deferred": True,
    })


async def _update_memory_impl(memory_id: str, content: str, **_: Any) -> ToolResult:
    ctx = get_memory_session()
    if ctx is None:
        return ToolResult(status="error", error_code="no_session_context", output="（会话上下文不可用，无法更新记忆）")
    session_id, file_id = ctx
    # 同步做归属校验，让模型立刻拿到"记忆不存在"的反馈，而不是等 worker 失败。
    if not await memory_service.memory_owned_by_current_user(memory_id):
        return ToolResult(status="error", error_code="memory_not_found",
                          output=f"（未找到 id={memory_id} 的记忆，或该记忆不属于当前用户）")
    job_id = await memory_service.enqueue_memory_job(
        kind="memory_op",
        payload={"op": "update", "id": memory_id, "content": content},
        session_id=session_id,
        file_id=file_id,
        priority="high",
    )
    return ToolResult(status="ok", output={
        "message": f"已提交记忆更新（后台生效）：{content}", "job_id": job_id, "deferred": True,
    })


async def _delete_memory_impl(memory_id: str, **_: Any) -> ToolResult:
    ctx = get_memory_session()
    if ctx is None:
        return ToolResult(status="error", error_code="no_session_context", output="（会话上下文不可用，无法删除记忆）")
    session_id, file_id = ctx
    if not await memory_service.memory_owned_by_current_user(memory_id):
        return ToolResult(status="error", error_code="memory_not_found",
                          output=f"（未找到 id={memory_id} 的记忆，或该记忆不属于当前用户）")
    job_id = await memory_service.enqueue_memory_job(
        kind="memory_op",
        payload={"op": "delete", "id": memory_id},
        session_id=session_id,
        file_id=file_id,
        priority="high",
    )
    return ToolResult(status="ok", output={"message": f"已提交记忆删除（后台生效）：{memory_id}", "job_id": job_id, "deferred": True})


@tool
async def search_memories(query: str) -> str:
    """按关键词检索当前用户的长期记忆（返回记忆 id 与内容）。在新增/修改/删除记忆之前，先调用本工具确认已有记忆，避免重复或遗漏。"""
    return await _search_memories_impl(query)


@tool
async def add_memory(content: str, memory_type: str = "session_fact", importance: float = 0.7) -> str:
    """保存一条值得跨轮记住的稳定信息。memory_type：user_preference（用户偏好）/ novel_fact（小说事实）/ session_fact（会话事实）。仅当用户明确表达偏好或重要事实时调用。"""
    return await _add_memory_impl(content, memory_type=memory_type, importance=importance)


@tool
async def update_memory(memory_id: str, content: str) -> str:
    """更新一条已有记忆的内容（先用 search_memories 获取记忆 id）。当用户修正、细化或改变了之前的信息时调用。"""
    return await _update_memory_impl(memory_id, content)


@tool
async def delete_memory(memory_id: str) -> str:
    """删除一条已有记忆（先用 search_memories 获取记忆 id）。仅当用户明确要求忘记某事或撤回偏好时调用。"""
    return await _delete_memory_impl(memory_id)


# bind_tools 用的模型侧工具清单（名字与 registry 白名单一致）。
MEMORY_AGENT_TOOL_SPECS = (search_memories, add_memory, update_memory, delete_memory)
MEMORY_AGENT_TOOLS = ("search_memories", "add_memory", "update_memory", "delete_memory")


def _register_memory_tools(registry: ToolRegistry) -> None:
    memory_modes = frozenset({"qa"})
    registry.register(ToolSpec(
        "search_memories", "检索当前用户的长期记忆",
        timeout_seconds=settings.memory_task_timeout, cost_class="low", supported_modes=memory_modes,
    ), _search_memories_impl)
    registry.register(ToolSpec(
        "add_memory", "保存一条长期记忆",
        timeout_seconds=settings.memory_task_timeout, cost_class="low", supported_modes=memory_modes,
    ), _add_memory_impl)
    registry.register(ToolSpec(
        "update_memory", "更新一条长期记忆",
        timeout_seconds=settings.memory_task_timeout, cost_class="low", supported_modes=memory_modes,
    ), _update_memory_impl)
    registry.register(ToolSpec(
        "delete_memory", "删除一条长期记忆",
        timeout_seconds=settings.memory_task_timeout, cost_class="low", supported_modes=memory_modes,
    ), _delete_memory_impl)


# ===== ReAct 执行环的模型侧工具（bind_tools 用；执行仍走 registry 以复用超时/白名单） =====


def _react_payload(result: ToolResult) -> str:
    """把工具结果压缩成模型可读的 JSON 文本。仅用于模型侧描述；状态证据以 registry 返回值为准。"""
    if result.status != "ok":
        return json.dumps({"status": result.status, "error": result.error_code}, ensure_ascii=False)
    output = result.output if isinstance(result.output, dict) else {}
    if "evidence" in output:
        return json.dumps({
            "status": "ok",
            "evidence": [
                {
                    "id": item.get("source", {}).get("id"),
                    "source": item.get("source", {}).get("source"),
                    "chapter": item.get("source", {}).get("chapter"),
                    "content": item.get("content", ""),
                }
                for item in output.get("evidence", [])
            ],
        }, ensure_ascii=False)
    return json.dumps({"status": "ok", **output}, ensure_ascii=False, default=str)


@tool("retrieve_novel")
async def react_retrieve_novel(query: str) -> str:
    """外置小说知识库检索：仅在缺少可靠小说证据（人物、关系、情节、时间线、章节、原文核验）且上下文不足时调用。query 填写你认为最可能命中的检索词，可换用新的表述、人物名或事件名。"""
    return _react_payload(await _retrieve_novel(query=query))


@tool("get_chapter_context")
async def react_chapter_context(query: str) -> str:
    """检索命中章节的相邻前后文片段。仅在已有命中不足以理解前因后果时调用，不是固定后置步骤。"""
    return _react_payload(await _chapter_context(query=query))


@tool("calculator")
async def react_calculator(expression: str) -> str:
    """执行受限数值计算（加减乘除、取模、乘方）。expression 填写算式，如 (1908-1912)*12。"""
    return _react_payload(await _calculator(expression=expression))


@tool("load_character_context")
async def react_load_character_context() -> str:
    """加载当前场景的登场角色卡与开场情景（人物知识边界已在卡内固化）。角色扮演对话开始时调用一次。"""
    return _react_payload(await _load_character_context_impl())


# 模型侧工具描述按名字索引；执行环据此把 allowed_tools 翻译为 bind_tools 清单。
MODEL_TOOL_SPECS = {
    "retrieve_novel": react_retrieve_novel,
    "get_chapter_context": react_chapter_context,
    "calculator": react_calculator,
    "load_character_context": react_load_character_context,
}
REACT_TOOL_SPECS = (react_retrieve_novel, react_chapter_context, react_calculator)  # 兼容旧引用
REACT_TOOL_LABELS = {
    "retrieve_novel": "补充检索",
    "get_chapter_context": "章节上下文",
    "calculator": "数值计算",
    "load_character_context": "装载角色卡",
}


def model_tool_specs(allowed_tools: list[str] | tuple[str, ...]) -> list[Any]:
    """按白名单返回 bind_tools 用的模型侧工具清单（保持声明顺序）。"""
    return [spec for name, spec in MODEL_TOOL_SPECS.items() if name in allowed_tools]


registry = build_default_registry()
