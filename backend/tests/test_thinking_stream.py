"""思考流模型响应层测试：解析器、tool_call 聚合协议、能力回退与缓存（raw SDK 路径）。"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.agent.tools import REACT_TOOL_SPECS
from app.config import settings
from app.core import llm as llm_module
from app.core.llm import (
    LLMPurpose,
    ModelTurnBuilder,
    ModelTurnDelta,
    astream_model_turn,
    build_reasoning_extra_body,
    reasoning_capability_cache,
    _convert_tools,
    _is_capability_error,
    _raw_chunk_to_delta,
    _to_openai_messages,
)


# ---------- Builder：reasoning / content 累积 ----------

def test_builder_accumulates_reasoning_and_content():
    builder = ModelTurnBuilder()
    builder.add_delta(ModelTurnDelta(kind="reasoning", text="先想"))
    builder.add_delta(ModelTurnDelta(kind="content", text="决定"))
    builder.add_delta(ModelTurnDelta(kind="reasoning", text="一下"))
    turn = builder.build()
    assert turn.reasoning == "先想一下"
    assert turn.content == "决定"
    assert turn.tool_calls == []


def test_tool_call_args_split_across_chunks():
    """同一调用的 name/args 跨多分片按 index 聚合，args 结束后一次性解析。"""
    builder = ModelTurnBuilder()
    builder.add_delta(ModelTurnDelta(kind="tool_call", tool_call_chunk={"index": 0, "id": "c1", "name": "retrieve_", "args_str": ""}))
    builder.add_delta(ModelTurnDelta(kind="tool_call", tool_call_chunk={"index": 0, "name": "novel", "args_str": '{"qu'}))
    builder.add_delta(ModelTurnDelta(kind="tool_call", tool_call_chunk={"index": 0, "name": "", "args_str": 'ery": "宝玉"}'}))
    turn = builder.build()
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].id == "c1"
    assert turn.tool_calls[0].name == "retrieve_novel"
    assert turn.tool_calls[0].args == {"query": "宝玉"}


def test_two_parallel_tool_calls_keep_order():
    builder = ModelTurnBuilder()
    builder.add_delta(ModelTurnDelta(kind="tool_call", tool_call_chunk={"index": 0, "id": "a", "name": "t1", "args_str": "{}"}))
    builder.add_delta(ModelTurnDelta(kind="tool_call", tool_call_chunk={"index": 1, "id": "b", "name": "t2", "args_str": "{}"}))
    assert [c.name for c in builder.build().tool_calls] == ["t1", "t2"]


def test_invalid_args_json_becomes_empty_dict():
    builder = ModelTurnBuilder()
    builder.add_delta(ModelTurnDelta(kind="tool_call", tool_call_chunk={"index": 0, "id": "x", "name": "calculator", "args_str": '{"expr'}))
    assert builder.build().tool_calls[0].args == {}


def test_missing_call_id_gets_stable_generated_id():
    builder = ModelTurnBuilder()
    builder.add_delta(ModelTurnDelta(kind="tool_call", tool_call_chunk={"index": 0, "name": "calculator", "args_str": "{}"}))
    call = builder.build().tool_calls[0]
    assert call.id == "call-0-0"


def test_empty_aggregation_falls_back_to_response_tool_calls():
    builder = ModelTurnBuilder()
    builder.add_response(AIMessage(content="", tool_calls=[
        {"name": "calculator", "args": {"expression": "1+1"}, "id": "fb1"},
    ]))
    calls = builder.build().tool_calls
    assert len(calls) == 1 and calls[0].name == "calculator" and calls[0].args == {"expression": "1+1"}


# ---------- 消息与工具格式转换 ----------

def test_to_openai_messages_covers_all_roles():
    messages = [
        SystemMessage(content="系统提示"),
        HumanMessage(content="用户问题"),
        AIMessage(content="", tool_calls=[{"name": "retrieve_novel", "args": {"query": "宝玉"}, "id": "c1"}]),
        ToolMessage(content="证据内容", tool_call_id="c1"),
    ]
    converted = _to_openai_messages(messages)
    assert converted[0] == {"role": "system", "content": "系统提示"}
    assert converted[1] == {"role": "user", "content": "用户问题"}
    assert converted[2]["role"] == "assistant"
    assert converted[2]["tool_calls"][0]["function"] == {
        "name": "retrieve_novel", "arguments": '{"query": "宝玉"}',
    }
    assert converted[3] == {"role": "tool", "tool_call_id": "c1", "content": "证据内容"}


def test_to_openai_messages_rejects_unknown_type():
    with pytest.raises(TypeError):
        _to_openai_messages([SimpleNamespace(content="x")])


def test_convert_tools_maps_react_specs():
    converted = _convert_tools(list(REACT_TOOL_SPECS))
    names = {tool["function"]["name"] for tool in converted}
    assert names == {"retrieve_novel", "get_chapter_context", "calculator"}
    assert all(tool["type"] == "function" for tool in converted)
    assert converted[0]["function"]["parameters"]["type"] == "object"


# ---------- 能力错误分类 ----------

class _APIError(Exception):
    def __init__(self, status_code: int | None, message: str):
        super().__init__(message)
        self.status_code = status_code


def test_capability_error_classification():
    assert _is_capability_error(Exception("400 unsupported parameter: thinking"))
    assert _is_capability_error(Exception("unknown field reasoning_effort"))
    assert _is_capability_error(Exception("model not_found: reasoner-x"))
    assert _is_capability_error(_APIError(400, "bad request"))
    assert _is_capability_error(_APIError(404, "model missing"))
    assert not _is_capability_error(_APIError(429, "rate limited"))
    assert not _is_capability_error(_APIError(503, "server busy"))
    assert not _is_capability_error(Exception("request timeout after 30s"))
    assert not _is_capability_error(Exception("connection reset by peer"))


def test_provider_extra_body():
    assert build_reasoning_extra_body("deepseek", "medium", 2048) == {"thinking": {"type": "enabled"}}
    body = build_reasoning_extra_body("openai", "high", 2048)
    assert body["reasoning_effort"] == "high"
    assert build_reasoning_extra_body("unknown", "high", 2048) == {"thinking": {"type": "enabled"}}


# ---------- raw chunk → delta ----------

def _raw_delta(reasoning=None, content=None, tool_calls=None, finish_reason=None):
    delta = SimpleNamespace(reasoning_content=reasoning, content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)])


def _raw_tool_call(index=0, id=None, name=None, arguments=""):
    return SimpleNamespace(index=index, id=id, function=SimpleNamespace(name=name, arguments=arguments))


def test_raw_chunk_reasoning_attr_passthrough():
    deltas = _raw_chunk_to_delta(_raw_delta(reasoning="透传推理"))
    assert [d.kind for d in deltas] == ["reasoning"]
    assert deltas[0].text == "透传推理"


def test_raw_chunk_content_blocks_reasoning_and_text():
    deltas = _raw_chunk_to_delta(_raw_delta(content=[
        {"type": "reasoning", "reasoning": "块内推理"},
        {"type": "text", "text": "正文"},
    ]))
    assert [(d.kind, d.text) for d in deltas] == [("reasoning", "块内推理"), ("content", "正文")]


def test_raw_chunk_tool_calls_yield_per_call():
    deltas = _raw_chunk_to_delta(_raw_delta(tool_calls=[
        _raw_tool_call(0, "a", "t1", "{}"),
        _raw_tool_call(1, "b", "t2", "{}"),
    ]))
    assert [d.kind for d in deltas] == ["tool_call", "tool_call"]
    assert deltas[0].tool_call_chunk["name"] == "t1"


# ---------- astream_model_turn：流式解析与能力回退 ----------

def _patch_stream(monkeypatch, chunks, error=None):
    """monkeypatch _stream_openai_chunks，捕获请求参数并按脚本产出 raw chunks。"""
    calls: list[dict] = []

    def factory(**kwargs):
        calls.append(kwargs)

        async def stream():
            if error is not None:
                raise error
            for chunk in chunks:
                yield chunk

        return stream()

    monkeypatch.setattr(llm_module, "_stream_openai_chunks", factory)
    return calls


async def _collect(purpose, monkeypatch, chunks=None, error=None):
    calls = _patch_stream(monkeypatch, chunks or [], error=error)
    deltas = []
    async for delta in astream_model_turn([HumanMessage(content="q")], purpose):
        deltas.append(delta)
    return deltas, calls


async def test_stream_reasoning_content_and_tool_call_interleaved(monkeypatch):
    monkeypatch.setattr(settings, "agent_reasoning_model", "reasoner-x", raising=False)
    chunks = [
        _raw_delta(reasoning="推理A"),
        _raw_delta(content="部分"),
        _raw_delta(tool_calls=[_raw_tool_call(0, "c9", "retrieve_", "")]),
        _raw_delta(tool_calls=[_raw_tool_call(0, None, "novel", '{"query": "宝玉"}')]),
        _raw_delta(content="结论"),
    ]
    deltas, calls = await _collect(LLMPurpose.AGENT_DECISION, monkeypatch, chunks)
    assert calls[0]["model"] == "reasoner-x"
    assert [d.kind for d in deltas] == ["reasoning", "content", "tool_call", "tool_call", "content"]
    builder = ModelTurnBuilder()
    for d in deltas:
        builder.add_delta(d)
    turn = builder.build()
    assert turn.reasoning == "推理A" and turn.content == "部分结论"
    assert turn.tool_calls[0].name == "retrieve_novel" and turn.tool_calls[0].args == {"query": "宝玉"}


async def test_decide_call_passes_tools_and_temperature(monkeypatch):
    monkeypatch.setattr(settings, "agent_reasoning_model", "", raising=False)
    calls = _patch_stream(monkeypatch, [_raw_delta(content="ok")])
    deltas = []
    async for delta in astream_model_turn(
        [HumanMessage(content="q")], LLMPurpose.AGENT_DECISION, tools=list(REACT_TOOL_SPECS),
    ):
        deltas.append(delta)
    assert calls[0]["tools"] is not None
    assert {t["function"]["name"] for t in calls[0]["tools"]} == {
        "retrieve_novel", "get_chapter_context", "calculator",
    }
    assert any(d.kind == "content" and d.text == "ok" for d in deltas)


async def test_capability_error_falls_back_and_caches(monkeypatch):
    monkeypatch.setattr(settings, "agent_reasoning_model", "reasoner-x", raising=False)
    monkeypatch.setattr(settings, "agent_reasoning_enabled", True, raising=False)
    reasoning_capability_cache.clear()
    scripted = iter([
        (None, _APIError(400, "unsupported parameter: thinking")),
        ([_raw_delta(content="降级回答")], None),
        ([_raw_delta(content="第二次")], None),
    ])
    calls: list[dict] = []

    def factory(**kwargs):
        chunks, error = next(scripted)
        calls.append({"model": kwargs["model"], "extra_body": kwargs["extra_body"]})

        async def stream():
            if error is not None:
                raise error
            for chunk in chunks:
                yield chunk

        return stream()

    monkeypatch.setattr(llm_module, "_stream_openai_chunks", factory)
    deltas = []
    async for delta in astream_model_turn([HumanMessage(content="q")], LLMPurpose.AGENT_DECISION):
        deltas.append(delta)
    # 第一次走 reasoning 模型失败 → 降级主模型（extra_body 清空）
    assert calls[0]["model"] == "reasoner-x" and calls[0]["extra_body"] is not None
    assert calls[1]["model"] == settings.llm_model and calls[1]["extra_body"] is None
    assert any(d.kind == "content" and d.text == "降级回答" for d in deltas)
    assert reasoning_capability_cache[(settings.llm_base_url, "reasoner-x")] is False

    # 缓存生效：下一次直接主模型，不再探测 reasoning 模型。
    async for _ in astream_model_turn([HumanMessage(content="q")], LLMPurpose.AGENT_DECISION):
        pass
    assert len(calls) == 3 and calls[2]["model"] == settings.llm_model


async def test_transient_error_does_not_fallback(monkeypatch):
    monkeypatch.setattr(settings, "agent_reasoning_model", "reasoner-x", raising=False)
    monkeypatch.setattr(settings, "agent_reasoning_enabled", True, raising=False)
    reasoning_capability_cache.clear()
    calls = _patch_stream(monkeypatch, [], error=_APIError(429, "rate limited"))
    with pytest.raises(Exception, match="rate limited"):
        async for _ in astream_model_turn([HumanMessage(content="q")], LLMPurpose.AGENT_DECISION):
            pass
    assert calls[0]["model"] == "reasoner-x"
    assert reasoning_capability_cache == {}


async def test_reasoning_budget_expands_max_tokens(monkeypatch):
    """推理模型的思维链计入 max_tokens：思考开启时须为正文预留 reasoning 预算。"""
    monkeypatch.setattr(settings, "agent_reasoning_model", "reasoner-x", raising=False)
    monkeypatch.setattr(settings, "agent_reasoning_budget", 2048, raising=False)
    reasoning_capability_cache.clear()
    calls = _patch_stream(monkeypatch, [_raw_delta(content="ok")])
    async for _ in astream_model_turn([HumanMessage(content="q")], LLMPurpose.AGENT_DECISION, max_tokens=800):
        pass
    assert calls[0]["max_tokens"] == 800 + 2048


async def test_max_tokens_untouched_without_reasoning(monkeypatch):
    monkeypatch.setattr(settings, "agent_reasoning_model", "", raising=False)
    calls = _patch_stream(monkeypatch, [_raw_delta(content="ok")])
    async for _ in astream_model_turn([HumanMessage(content="q")], LLMPurpose.AGENT_DECISION, max_tokens=800):
        pass
    assert calls[0]["max_tokens"] == 800
