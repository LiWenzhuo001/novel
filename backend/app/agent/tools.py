"""Agent 工具注册、权限控制、超时处理和小说检索工具。"""
from __future__ import annotations

import ast
import asyncio
import operator
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from app.agent.types import ToolResult
from app.config import settings
from app.core.rag import retrieve_novel_context

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


async def _chapter_context(*, query: str, file_id: str | None = None, **_: Any) -> ToolResult:
    """以较大的邻居窗口检索命中章节的前后文。"""
    return await _retrieve_novel(query=query, file_id=file_id, neighbor_window=2)


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
    return registry


registry = build_default_registry()
