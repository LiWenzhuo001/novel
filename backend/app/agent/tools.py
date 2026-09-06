"""Agent 工具注册、权限控制、超时处理和小说检索工具。"""
from __future__ import annotations

import ast
import asyncio
import operator
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from langchain_core.tools import tool

from app.agent.types import ToolResult
from app.config import settings
from app.core.context import get_memory_session
from app.core.rag import retrieve_novel_context
from app.services import memory_service

ToolHandler = Callable[..., Awaitable[ToolResult]]


@dataclass(frozen=True)
class ToolSpec:
    """工具的静态描述，包括超时、权限和幂等性信息。"""
    name: str
    description: str
    timeout_seconds: float = 20.0
    permission: str = "read"
    idempotent: bool = True


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

    async def execute(self, name: str, *, allowed_tools: list[str] | tuple[str, ...], **kwargs: Any) -> ToolResult:
        """执行指定工具，并将拒绝、超时和异常统一转换为 ToolResult。"""
        if name not in self._specs or name not in allowed_tools:
            return ToolResult(status="denied", error_code="tool_not_allowed", tool=name)
        started = time.perf_counter()
        spec = self._specs[name]
        try:
            result = await asyncio.wait_for(self._handlers[name](**kwargs), timeout=spec.timeout_seconds)
        except asyncio.TimeoutError:
            return ToolResult(
                status="timeout",
                error_code="tool_timeout",
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
                tool=name,
            )
        except Exception as exc:  # noqa: BLE001
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
    registry.register(ToolSpec("retrieve_novel", "混合检索小说原文并返回引用", timeout_seconds=settings.agent_tool_timeout), _retrieve_novel)
    registry.register(ToolSpec("get_chapter_context", "检索命中章节的相邻片段", timeout_seconds=settings.agent_tool_timeout), _chapter_context)
    registry.register(ToolSpec("calculator", "执行受限数值计算", timeout_seconds=settings.agent_tool_timeout), _calculator)
    _register_memory_tools(registry)
    return registry


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
    row = await memory_service.save_memory(
        content, memory_type,
        session_id=session_id if memory_type == "session_fact" else None,
        file_id=file_id if memory_type == "novel_fact" else None,
        importance=importance,
    )
    return ToolResult(status="ok", output={"message": f"已保存记忆：{content}", "memory_id": row.id})


async def _update_memory_impl(memory_id: str, content: str, **_: Any) -> ToolResult:
    row = await memory_service.update_memory(memory_id, content=content)
    if row is None:
        return ToolResult(status="error", error_code="memory_not_found",
                          output=f"（未找到 id={memory_id} 的记忆，或该记忆不属于当前用户）")
    return ToolResult(status="ok", output={"message": f"已更新记忆：{content}", "memory_id": memory_id})


async def _delete_memory_impl(memory_id: str, **_: Any) -> ToolResult:
    deleted = await memory_service.delete_memory(memory_id)
    if not deleted:
        return ToolResult(status="error", error_code="memory_not_found",
                          output=f"（未找到 id={memory_id} 的记忆，或该记忆不属于当前用户）")
    return ToolResult(status="ok", output={"message": f"已删除记忆：{memory_id}"})


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
    registry.register(ToolSpec("search_memories", "检索当前用户的长期记忆", timeout_seconds=settings.memory_task_timeout), _search_memories_impl)
    registry.register(ToolSpec("add_memory", "保存一条长期记忆", timeout_seconds=settings.memory_task_timeout), _add_memory_impl)
    registry.register(ToolSpec("update_memory", "更新一条长期记忆", timeout_seconds=settings.memory_task_timeout), _update_memory_impl)
    registry.register(ToolSpec("delete_memory", "删除一条长期记忆", timeout_seconds=settings.memory_task_timeout), _delete_memory_impl)


registry = build_default_registry()
