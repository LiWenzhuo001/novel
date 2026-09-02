import asyncio
import time

import pytest

from app.agent import runtime
from app.agent.contracts import template_expert_tasks
from app.agent.dispatcher import DispatchResult
from app.agent.types import ToolResult


async def _fake_tool(_name, **kwargs):
    return ToolResult(
        status="ok",
        output={
            "evidence": [{
                "source": {
                    "id": "S1",
                    "source": "book.txt",
                    "chapter": "第一章",
                    "chapter_no": 1,
                    "chunk_no": 1,
                },
                "content": "林舟醒来，随后离开故乡。",
            }],
            "sources": [{
                "id": "S1",
                "source": "book.txt",
                "chapter": "第一章",
                "chapter_no": 1,
                "chunk_no": 1,
            }],
        },
    )


async def _fake_dispatch(query):
    return DispatchResult(template_expert_tasks(query), "template", "test_dispatch")


async def _fake_stream(messages, _max_tokens):
    prompt = str(messages[-1].content)
    if "人物关系专家" in prompt:
        yield "> 关系与立场：林舟从依赖故乡转为主动离开。事实是他已经离开，人物动机与关系变化阶段仍需更多证据，因此部分内容属于推断 [S1]"
    elif "情节发展专家" in prompt:
        yield "> 起因：林舟醒来并面对新的处境；事件与冲突：他决定离开故乡；转折是行动正式开始；结果与后续影响是旅程由此展开 [S1]"
    elif "时间线专家" in prompt:
        yield (
            "> 1. 首先，林舟醒来并确认当前处境 [S1]\n"
            "> 2. 随后，他离开故乡，事件顺序由停留转为出发，人物关系也随行动发生变化 [S1]"
        )
    elif "章节定位专家" in prompt:
        yield "> | 结论 | 来源ID | 章节 | 页码 | 片段号 |\n> |---|---|---|---|---|\n> | 林舟醒来并离开故乡 | [S1] | 第一章 | 定位信息不足 | 1 |\n> 定位说明：当前共享证据未提供更精确页码。"
    else:
        yield "最终回答 [S1]"


@pytest.fixture(autouse=True)
def _runtime_fakes(monkeypatch):
    monkeypatch.setattr(runtime.registry, "execute", _fake_tool)
    monkeypatch.setattr(runtime, "dispatch_expert_tasks", _fake_dispatch)
    monkeypatch.setattr(runtime, "_stream_llm", _fake_stream)


@pytest.mark.asyncio
async def test_react_runtime_emits_live_graph_events():
    events = [event async for event in runtime.stream_agent_question("谁是林舟？", "react", "file-1")]
    types = [event["type"] for event in events]
    assert types[:2] == ["route", "plan"]
    assert {"tool_start", "observation", "reflection", "sources", "token", "meta"}.issubset(types)


@pytest.mark.asyncio
async def test_multi_expert_dispatches_distinct_tasks_before_experts():
    events = [event async for event in runtime.stream_agent_question(
        "梳理林舟的人物关系、情节发展和时间线，并给出章节位置",
        "multi_expert",
        "file-1",
    )]
    task_index = next(i for i, event in enumerate(events) if event["type"] == "expert_tasks")
    first_expert_index = next(
        i for i, event in enumerate(events)
        if event["type"] == "tool_start" and event["data"].get("tool") == "specialist"
    )
    tasks = events[task_index]["data"]["tasks"]
    assert task_index < first_expert_index
    assert len({item["task"] for item in tasks.values()}) == 4


@pytest.mark.asyncio
async def test_multi_expert_streams_four_experts_before_final_answer():
    events = [event async for event in runtime.stream_agent_question(
        "梳理林舟的人物关系、情节发展和时间线，并给出章节位置",
        "multi_expert",
        "file-1",
    )]
    starts = [event for event in events if event["type"] == "tool_start" and event["data"].get("tool") == "specialist"]
    expert_tokens = [event for event in events if event["type"] == "tool_token"]
    final_token_index = next(i for i, event in enumerate(events) if event["type"] == "token")
    validation_index = max(i for i, event in enumerate(events) if event["type"] == "validation")
    assert len(starts) == 4
    assert {event["data"]["agent"] for event in expert_tokens} == {"character", "plot", "timeline", "locator"}
    assert validation_index < final_token_index
    meta = next(event["data"] for event in events if event["type"] == "meta")
    assert meta["expert_count"] == 4
    assert all(report["status"] == "ok" for report in meta["reports"])


@pytest.mark.asyncio
async def test_experts_are_concurrent(monkeypatch):
    async def delayed_specialist(contract, state, agent_id, **kwargs):
        await runtime._emit(state, "tool_token", {
            "id": agent_id,
            "agent": contract.name,
            "label": contract.label,
            "delta": f"> {contract.label} [S1]",
        })
        await asyncio.sleep(0.05)
        await runtime._emit(state, "tool_end", {
            "id": agent_id,
            "tool": "specialist",
            "agent": contract.name,
            "label": contract.label,
            "status": "ok",
            "summary": "完成",
        })
        return {"agent": contract.name, "label": contract.label, "status": "ok", "report": _valid_report(contract.name)}

    monkeypatch.setattr(runtime, "_run_specialist", delayed_specialist)
    started = time.perf_counter()
    events = [event async for event in runtime.stream_agent_question("梳理人物关系和时间线", "multi_expert")]
    assert time.perf_counter() - started < 0.18
    assert sum(1 for event in events if event["type"] == "tool_token") == 4


def _valid_report(agent):
    return {
        "character": "关系与立场发生明显变化，人物从依赖转向疏离。事实与推断已经分开说明，人物动机和关系阶段仍有证据不足之处 [S1]",
        "plot": "起因是人物面对新的处境，事件引发冲突并形成关键转折，结果推动旅程展开，并对后续关系产生持续影响 [S1]",
        "timeline": "1. 首先人物确认处境并发生第一个事件 [S1]；2. 随后人物采取行动并造成关系变化；3. 最终形成清晰的事件顺序 [S1]",
        "locator": "| 结论 | 来源 | 章节 | 页码 | 片段 |\n|---|---|---|---|---|\n| 人物采取关键行动并推动后续发展 | [S1] | 第一章 | 1 | 1 |\n定位信息已经与来源片段对齐，章节、页码和片段号均可用于回查原文。",
    }[agent]


@pytest.mark.asyncio
async def test_partial_expert_failure_still_synthesizes(monkeypatch):
    async def mixed_specialist(contract, state, agent_id, **kwargs):
        status = "error" if contract.name == "timeline" else "ok"
        await runtime._emit(state, "tool_end", {
            "id": agent_id,
            "tool": "specialist",
            "agent": contract.name,
            "label": contract.label,
            "status": status,
            "summary": status,
        })
        return {
            "agent": contract.name,
            "label": contract.label,
            "status": status,
            "report": _valid_report(contract.name) if status == "ok" else "",
        }

    monkeypatch.setattr(runtime, "_run_specialist", mixed_specialist)
    events = [event async for event in runtime.stream_agent_question("梳理人物关系和时间线", "multi_expert")]
    meta = next(event["data"] for event in events if event["type"] == "meta")
    assert any(event["type"] == "token" for event in events)
    assert sum(report["status"] == "ok" for report in meta["reports"]) == 3
    assert any(report["status"] == "error" for report in meta["reports"])


@pytest.mark.asyncio
async def test_empty_retrieval_reports_fallback(monkeypatch):
    async def empty_tool(_name, **kwargs):
        return ToolResult(status="ok", output={"evidence": [], "sources": []})

    monkeypatch.setattr(runtime.registry, "execute", empty_tool)
    events = [event async for event in runtime.stream_agent_question("未知问题", "direct")]
    meta = next(event["data"] for event in events if event["type"] == "meta")
    assert meta["fallback_reason"] == "empty_retrieval"
    assert any(event["type"] == "token" and "没有检索到" in event["data"] for event in events)


@pytest.mark.asyncio
async def test_duplicate_report_is_corrected_once_with_reset_event(monkeypatch):
    calls: dict[str, int] = {}

    async def duplicate_then_correct(contract, state, agent_id, **kwargs):
        retry = kwargs.get("retry", 0)
        calls[contract.name] = calls.get(contract.name, 0) + 1
        if contract.name == "plot" and retry == 0:
            report = _valid_report("character")
        else:
            report = _valid_report(contract.name)
        return {
            "agent": contract.name,
            "label": contract.label,
            "status": "ok",
            "report": report,
            "corrected": bool(retry),
        }

    monkeypatch.setattr(runtime, "_run_specialist", duplicate_then_correct)
    events = [event async for event in runtime.stream_agent_question("梳理人物关系和时间线", "multi_expert")]
    retry_starts = [
        event for event in events
        if event["type"] == "tool_start"
        and event["data"].get("agent") == "plot"
        and event["data"].get("retry") == 1
    ]
    meta = next(event["data"] for event in events if event["type"] == "meta")
    plot_report = next(report for report in meta["reports"] if report["agent"] == "plot")

    assert len(retry_starts) == 1
    assert retry_starts[0]["data"]["reset"] is True
    assert calls["plot"] == 2
    assert plot_report["status"] == "ok"
    assert plot_report["corrected"] is True
    assert meta["report_validation"]["plot"]["contract_ok"] is True


@pytest.mark.asyncio
async def test_non_rag_route_skips_retrieval_and_synthesizes(monkeypatch):
    calls = 0

    async def forbidden_tool(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("RAG should be skipped")

    async def conversation_stream(messages, _max_tokens):
        yield "好的，我会保持简洁。"

    monkeypatch.setattr(runtime.registry, "execute", forbidden_tool)
    monkeypatch.setattr(runtime, "_stream_llm", conversation_stream)
    events = [event async for event in runtime.stream_agent_question("你好，谢谢", "auto")]

    route = next(event["data"] for event in events if event["type"] == "route")
    meta = next(event["data"] for event in events if event["type"] == "meta")
    assert calls == 0
    assert route["needs_retrieval"] is False
    assert route["retrieval_skipped"] is True
    assert meta["retrieval_reason"] == "conversation_only"
    assert meta["fallback_reason"] == ""
    assert any(event["type"] == "token" and event["data"] == "好的，我会保持简洁。" for event in events)


@pytest.mark.asyncio
async def test_runtime_uses_query_preparation_routing_hint(monkeypatch):
    async def forbidden_tool(*args, **kwargs):
        raise AssertionError("RAG should be skipped from the LLM routing hint")

    async def conversation_stream(messages, _max_tokens):
        yield "已按你的偏好继续。"

    monkeypatch.setattr(runtime.registry, "execute", forbidden_tool)
    monkeypatch.setattr(runtime, "_stream_llm", conversation_stream)
    preparation = {
        "original": "以后回答简短一点",
        "standalone_query": "用户希望后续回答更简短",
        "retrieval_query": "",
        "reason": "rewritten",
        "needs_retrieval": False,
        "answer_mode": "memory_context",
        "retrieval_reason": "user_preference",
        "confidence": 0.98,
    }
    events = [event async for event in runtime.stream_agent_question(
        "用户希望后续回答更简短",
        "auto",
        query_preparation=preparation,
    )]
    route = next(event["data"] for event in events if event["type"] == "route")
    meta = next(event["data"] for event in events if event["type"] == "meta")
    assert route["llm_needs_retrieval"] is False
    assert route["needs_retrieval"] is False
    assert route["routing_override"] is False
    assert meta["answer_mode"] == "memory_context"


@pytest.mark.asyncio
async def test_summary_node_applies_summary_only_policy(monkeypatch):
    async def fake_stream(messages, _max_tokens):
        yield "根据共享原文与四份专家报告：\n“幸亏他大徒弟慨然见允” [S1]"

    monkeypatch.setattr(runtime, "_stream_llm", fake_stream)
    state = {
        "current_step": 1,
        "strategy": "direct",
        "intent": "fact_lookup",
        "original_query": "问题",
        "standalone_query": "问题",
        "retrieval_query": "问题",
        "query_preparation": {},
        "needs_retrieval": True,
        "retrieval_reason": "novel_fact",
        "answer_mode": "novel_evidence",
        "output_policy": {
            "summary_only": True,
            "show_source_text": False,
            "allow_direct_quotes": False,
            "show_citations": True,
            "citation_style": "chapter_only",
            "show_agent_details": False,
        },
        "evidence": [{"source": {"id": "S1", "chapter": "第五十四回"}, "content": "原文"}],
        "sources": [{"id": "S1", "chapter": "第五十四回"}],
        "reports": [],
        "report_validation": {},
        "memory_context": {},
        "event_queue": asyncio.Queue(),
    }
    result = await runtime._supervisor_node(state)
    state.update(result)
    events = []
    async def capture_emit(current, event_type, data):
        events.append((event_type, data))
    monkeypatch.setattr(runtime, "_emit", capture_emit)
    result = await runtime._summary_node(state)
    answer = result["answer"]
    assert "根据共享原文" not in answer
    assert "幸亏他大徒弟慨然见允" not in answer
    assert any(event_type == "meta" for event_type, _ in events)
