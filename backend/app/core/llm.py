"""统一创建聊天模型实例，并集中处理模型、超时、API 配置与 reasoning 解析。

思考流（实验能力）：agent_decision / agent_plan / expert 用途可开启 reasoning
（需配置 AGENT_REASONING_MODEL 指向推理模型）；answer 用途强制关闭 thinking，
保证最终答案流干净。

流式通道说明：reasoning 路径（astream_model_turn）直接使用 raw OpenAI SDK——
langchain-openai 的 ChatOpenAI 会丢弃第三方 provider 的非标准字段（如 DeepSeek
的 delta.reasoning_content，其 1.4.1 模块头明确声明），而 SDK 对未知字段透传，
可原生读到 reasoning_content，且天然兼容任意 OpenAI-compatible 服务商。
非流式与其余用途仍走 get_llm（ChatOpenAI），行为不变。
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, AsyncIterator, Literal
from urllib.parse import urlsplit

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI

from app.config import settings
from app.core.logging_config import get_logger

log = get_logger("llm")


class LLMPurpose(StrEnum):
    """模型调用的用途；决定是否启用 reasoning 以及用哪个模型。"""

    AGENT_DECISION = "agent_decision"
    AGENT_PLAN = "agent_plan"
    AGENT_REFLECT = "agent_reflect"  # 本期 reflect 是规则节点，预留
    EXPERT = "expert"
    ANSWER = "answer"


def get_llm(
    streaming: bool = False,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
    model: str | None = None,
    purpose: LLMPurpose | str | None = None,
) -> ChatOpenAI:
    """返回 LangChain ChatOpenAI（兼容 DeepSeek / 通义 / 智谱等 OpenAI 接口）。

    ``purpose`` 决定 reasoning 行为：answer 强制关闭 thinking；agent/expert 用途
    在配置开启且配置了推理模型时不注入 disable（由 build_reasoning_extra_body
    决定 extra_body）。不传 purpose 保持旧行为（跟随全局 disable 开关）。
    注意：需要 reasoning 输出的流式调用请走 astream_model_turn（raw SDK 通道），
    ChatOpenAI 不透传 reasoning_content。
    """
    settings.validate()
    purpose_enum = LLMPurpose(purpose) if purpose else None
    thinking_enabled = _purpose_thinking_enabled(purpose_enum)
    kwargs: dict[str, Any] = {
        "model": model or _purpose_model(purpose_enum),
        "api_key": settings.llm_api_key,
        "base_url": settings.llm_base_url,
        # 推理模型的 temperature 兼容性由服务商侧保证（DeepSeek reasoner 忽略该参数）。
        "temperature": temperature,
        "streaming": streaming,
        "timeout": settings.llm_timeout if timeout is None else timeout,
        "max_retries": settings.llm_max_retries if max_retries is None else max_retries,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    extra_body = _purpose_extra_body(purpose_enum, thinking_enabled)
    if extra_body:
        kwargs["extra_body"] = extra_body
    elif settings.llm_disable_thinking and not thinking_enabled:
        # DeepSeek V4 等推理模型关闭思考模式（thinking.type=disabled）：
        # 直接返回 content，避免思维链空 content + 长延迟。不支持该参数的服务端会忽略。
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    return ChatOpenAI(**kwargs)


def _purpose_model(purpose: LLMPurpose | None) -> str:
    """按用途选择模型：agent 类用途优先用配置的推理模型。"""
    if purpose in {LLMPurpose.AGENT_DECISION, LLMPurpose.AGENT_PLAN}:
        if settings.agent_reasoning_enabled and settings.agent_reasoning_model:
            return settings.agent_reasoning_model
    return settings.llm_model


def _purpose_thinking_enabled(purpose: LLMPurpose | None) -> bool:
    """该用途是否应允许 reasoning 输出。answer 永远关闭（答案流干净）。"""
    if purpose is None:
        return False
    if purpose is LLMPurpose.ANSWER:
        return bool(settings.agent_answer_reasoning_enabled)
    if purpose is LLMPurpose.AGENT_DECISION:
        return bool(settings.agent_reasoning_enabled and settings.agent_reasoning_model)
    if purpose is LLMPurpose.AGENT_PLAN:
        return bool(settings.agent_reasoning_enabled and settings.agent_reasoning_model
                    and settings.agent_plan_reasoning_enabled)
    return False  # AGENT_REFLECT 本期惰性；EXPERT 随专家链移除而退役


def _infer_provider(base_url: str) -> str:
    host = urlsplit(base_url).netloc.lower()
    if "deepseek" in host:
        return "deepseek"
    if "openai" in host:
        return "openai"
    if "dashscope" in host or "aliyun" in host:
        return "qwen"
    if "bigmodel" in host or "zhipu" in host:
        return "zhipu"
    if "siliconflow" in host:
        return "siliconflow"
    return "unknown"


def build_reasoning_extra_body(provider: str, effort: str, budget: int) -> dict[str, Any]:
    """按 provider 生成 reasoning extra_body；未知 provider 只发最小开关。

    EFFORT/BUDGET 不是 DeepSeek 字段：仅 OpenAI 系发送 reasoning_effort，
    其余 provider 忽略——不无条件拼装，避免 unknown field 报错。
    """
    body: dict[str, Any] = {"thinking": {"type": "enabled"}}
    if provider == "openai" and effort:
        body["reasoning_effort"] = effort
    return body


def _purpose_extra_body(purpose: LLMPurpose | None, thinking_enabled: bool) -> dict[str, Any] | None:
    """reasoning 用途注入 provider 适配的 extra_body；其余场景返回 None 走旧逻辑。"""
    if purpose is None or not thinking_enabled:
        return None
    provider = _infer_provider(settings.llm_base_url)
    return build_reasoning_extra_body(provider, settings.agent_reasoning_effort, settings.agent_reasoning_budget)


# ===== reasoning / content / tool_call 流式解析 =====
# 消息与工具格式转换、tool_call 分片聚合、供应商差异全部收在本模块，
# runtime 只消费 ModelTurnDelta 并用 ModelTurnBuilder 累积 ModelTurn。


@dataclass
class ModelTurnDelta:
    """流式响应的单个增量事件；调用方据此发 thinking_token / 累积 ModelTurn。"""

    kind: Literal["reasoning", "content", "tool_call"]
    text: str = ""
    tool_call_chunk: dict[str, Any] | None = None  # {index, id?, name?, args_str?}


@dataclass
class ToolCall:
    """聚合完成的单次工具调用。"""

    id: str
    name: str
    args: dict[str, Any]


def tool_call_field(call: Any, key: str, default: Any = None) -> Any:
    """兼容读取工具调用字段：langchain 各版本的 ToolCall 可能是 dict 或 dataclass。"""
    if isinstance(call, dict):
        return call.get(key, default)
    return getattr(call, key, default)


@dataclass
class ModelTurn:
    """一次模型调用的最终结果；reasoning 仅供统计/展示，不进答案。"""

    content: str
    reasoning: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    truncated: bool = False  # reasoning 超过上限被截断


class ModelTurnBuilder:
    """从 ModelTurnDelta 流构造 ModelTurn：tool_call 分片聚合与容错在此完成。"""

    def __init__(self) -> None:
        self._content_parts: list[str] = []
        self._reasoning_parts: list[str] = []
        self._finish_reason: str | None = None
        # index -> {"id": str, "name": str, "args": str}
        self._call_chunks: dict[int, dict[str, str]] = {}
        self._call_order: list[int] = []
        self._response_tool_calls: list[Any] | None = None

    def add_response(self, response: Any) -> None:
        """记录最终聚合响应（AIMessage/AIMessageChunk），作为 tool_call 回退来源。"""
        self._response_tool_calls = getattr(response, "tool_calls", None) or None
        if self._finish_reason is None:
            response_metadata = getattr(response, "response_metadata", None) or {}
            self._finish_reason = response_metadata.get("finish_reason")

    def add_delta(self, delta: ModelTurnDelta) -> None:
        if delta.kind == "reasoning":
            self._reasoning_parts.append(delta.text)
        elif delta.kind == "content":
            self._content_parts.append(delta.text)
        elif delta.kind == "tool_call" and delta.tool_call_chunk is not None:
            chunk = delta.tool_call_chunk
            index = int(chunk.get("index") or 0)
            entry = self._call_chunks.get(index)
            if entry is None:
                entry = {"id": "", "name": "", "args": ""}
                self._call_chunks[index] = entry
                self._call_order.append(index)
            if chunk.get("id"):
                entry["id"] = chunk["id"]
            if chunk.get("name"):
                entry["name"] += str(chunk["name"])
            if chunk.get("args_str"):
                entry["args"] += str(chunk["args_str"])

    def build(self) -> ModelTurn:
        tool_calls = self._aggregate_tool_calls()
        return ModelTurn(
            content="".join(self._content_parts),
            reasoning="".join(self._reasoning_parts),
            tool_calls=tool_calls,
            finish_reason=self._finish_reason,
        )

    def _aggregate_tool_calls(self) -> list[ToolCall]:
        if self._call_chunks:
            calls: list[ToolCall] = []
            for seq, index in enumerate(self._call_order):
                entry = self._call_chunks[index]
                call_id = entry["id"] or f"call-{index}-{seq}"
                name = entry["name"].strip()
                args = self._parse_args(entry["args"], call_id)
                if not name:
                    continue
                calls.append(ToolCall(id=call_id, name=name, args=args))
            if calls:
                return calls
        # 聚合结果为空时回退最终响应的 tool_calls（部分 provider 不走增量分片）。
        fallback: list[ToolCall] = []
        for seq, call in enumerate(self._response_tool_calls or []):
            name = str(call.get("name") or "").strip()
            if not name:
                continue
            fallback.append(ToolCall(
                id=str(call.get("id") or f"call-fb-{seq}"),
                name=name,
                args=call.get("args") if isinstance(call.get("args"), dict) else {},
            ))
        return fallback

    @staticmethod
    def _parse_args(args_str: str, call_id: str) -> dict[str, Any]:
        if not args_str.strip():
            return {}
        try:
            parsed = json.loads(args_str)
        except json.JSONDecodeError:
            # 参数非法时返回空 dict：工具侧参数校验会兜底（如 calculator invalid_expression）。
            log.warning("llm.tool_call_args_invalid", call_id=call_id, length=len(args_str))
            return {}
        return parsed if isinstance(parsed, dict) else {}


def _to_openai_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """把 LangChain 消息转换为 OpenAI chat completions 的 messages 数组。"""
    out: list[dict[str, Any]] = []
    for message in messages:
        content = message.content if isinstance(message.content, str) else str(message.content or "")
        if isinstance(message, SystemMessage):
            out.append({"role": "system", "content": content})
        elif isinstance(message, HumanMessage):
            out.append({"role": "user", "content": content})
        elif isinstance(message, AIMessage):
            entry: dict[str, Any] = {"role": "assistant", "content": content}
            tool_calls = getattr(message, "tool_calls", None) or []
            if tool_calls:
                entry["tool_calls"] = [{
                    "id": str(tool_call_field(call, "id") or f"call-{index}"),
                    "type": "function",
                    "function": {
                        "name": str(tool_call_field(call, "name") or ""),
                        "arguments": json.dumps(tool_call_field(call, "args") or {}, ensure_ascii=False),
                    },
                } for index, call in enumerate(tool_calls)]
            out.append(entry)
        elif isinstance(message, ToolMessage):
            out.append({"role": "tool", "tool_call_id": str(message.tool_call_id), "content": content})
        else:
            raise TypeError(f"不支持的消息类型：{type(message).__name__}")
    return out


def _convert_tools(tools: list[Any] | None) -> list[dict[str, Any]] | None:
    """LangChain 工具（@tool / ToolSpec spec 对象）→ OpenAI tools 数组。"""
    if not tools:
        return None
    return [convert_to_openai_tool(tool) for tool in tools]


def _is_capability_error(exc: BaseException) -> bool:
    """判定是否为"模型/参数不支持"类错误：才允许降级主模型。

    超时、429、5xx、网络错误一律不回退，按现有重试策略处理。
    """
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        return status_code in {400, 404, 422}
    text = str(exc).lower()
    capability_markers = (
        "400", "bad request", "unknown field", "unsupported", "not supported",
        "invalid parameter", "invalid_request", "404", "model_not_found",
        "not_found", "does not exist", "unknown model",
    )
    transient_markers = ("timeout", "429", "rate limit", "500", "502", "503", "504",
                         "connection", "network", "temporarily")
    if any(marker in text for marker in transient_markers):
        return False
    if any(marker in text for marker in capability_markers):
        return True
    exc_type = type(exc).__name__.lower()
    return "badrequest" in exc_type or "notfound" in exc_type


# 进程内能力缓存：(base_url, model) -> 是否支持 reasoning；避免每轮先失败一次再回退。
reasoning_capability_cache: dict[tuple[str, str], bool] = {}


async def _stream_openai_chunks(
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    max_tokens: int | None,
    extra_body: dict[str, Any] | None,
) -> AsyncIterator[Any]:
    """raw OpenAI SDK 流式通道：透传非标准字段（delta.reasoning_content 等）。"""
    client = AsyncOpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": 0,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if tools:
        kwargs["tools"] = tools
    if extra_body:
        kwargs["extra_body"] = extra_body
    stream = await client.chat.completions.create(**kwargs)
    async for event in stream:
        yield event


def _raw_chunk_to_delta(chunk: Any) -> list[ModelTurnDelta]:
    """把 raw SDK chunk 转成 0..n 个 ModelTurnDelta（reasoning 优先，其次工具，最后正文）。"""
    choices = getattr(chunk, "choices", None) or []
    if not choices:
        return []
    delta = getattr(choices[0], "delta", None)
    if delta is None:
        return []
    deltas: list[ModelTurnDelta] = []
    reasoning = getattr(delta, "reasoning_content", None)
    if isinstance(reasoning, str) and reasoning:
        deltas.append(ModelTurnDelta(kind="reasoning", text=reasoning))
    for call in getattr(delta, "tool_calls", None) or []:
        fn = getattr(call, "function", None)
        args_str = getattr(fn, "arguments", "") or ""
        deltas.append(ModelTurnDelta(kind="tool_call", tool_call_chunk={
            "index": getattr(call, "index", None) or 0,
            "id": getattr(call, "id", None),
            "name": getattr(fn, "name", None),
            "args_str": args_str,
        }))
    content = getattr(delta, "content", None)
    if isinstance(content, str) and content:
        deltas.append(ModelTurnDelta(kind="content", text=content))
    elif isinstance(content, list):
        # 部分 provider 以 content blocks 传输：reasoning/thinking + text。
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = str(block.get("type") or "")
            if block_type in {"reasoning", "thinking"}:
                text = str(block.get("reasoning") or block.get("thinking") or block.get("text") or "")
                if text:
                    deltas.append(ModelTurnDelta(kind="reasoning", text=text))
            elif block_type == "text":
                text = str(block.get("text") or "")
                if text:
                    deltas.append(ModelTurnDelta(kind="content", text=text))
    return deltas


async def astream_model_turn(
    messages: list[Any],
    purpose: LLMPurpose | str,
    *,
    tools: list[Any] | None = None,
    max_tokens: int | None = None,
) -> AsyncIterator[ModelTurnDelta]:
    """按用途创建模型并流式产出 ModelTurnDelta（raw OpenAI SDK 通道）。

    ``tools`` 为模型侧工具清单（决策环必须传入，否则模型无从选工具）。
    能力回退：purpose 模型的**首个 chunk 之前**若发生能力错误，降级主模型普通
    模式重启流，并写入能力缓存；已产出 delta 后的异常按瞬时错误抛出。
    """
    purpose_enum = LLMPurpose(purpose)
    primary_model = _purpose_model(purpose_enum)
    cache_key = (settings.llm_base_url, primary_model)
    use_reasoning = _purpose_thinking_enabled(purpose_enum) and reasoning_capability_cache.get(cache_key, True)
    openai_messages = _to_openai_messages(messages)
    openai_tools = _convert_tools(tools)
    # 推理模型的思维链计入 max_tokens（DeepSeek reasoner 实测行为）：为思考预留
    # reasoning 预算，否则推理耗尽预算后正文为零，报告/决策全部为空。
    effective_max_tokens = max_tokens
    if use_reasoning and max_tokens is not None:
        effective_max_tokens = max_tokens + settings.agent_reasoning_budget

    try:
        iterator = _stream_openai_chunks(
            model=primary_model if use_reasoning else settings.llm_model,
            messages=openai_messages,
            tools=openai_tools,
            max_tokens=effective_max_tokens,
            extra_body=_purpose_extra_body(purpose_enum, use_reasoning),
        ).__aiter__()
        first_chunk = await iterator.__anext__()
    except StopAsyncIteration:
        return
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        if use_reasoning and _is_capability_error(exc):
            log.warning("llm.reasoning_unsupported", model=primary_model, error=str(exc)[:200])
            reasoning_capability_cache[cache_key] = False
            iterator = _stream_openai_chunks(
                model=settings.llm_model,
                messages=openai_messages,
                tools=openai_tools,
                max_tokens=max_tokens,
                extra_body=None,
            ).__aiter__()
            async for chunk in iterator:
                for delta in _raw_chunk_to_delta(chunk):
                    yield delta
            return
        raise

    for delta in _raw_chunk_to_delta(first_chunk):
        yield delta
    async for chunk in iterator:
        for delta in _raw_chunk_to_delta(chunk):
            yield delta
