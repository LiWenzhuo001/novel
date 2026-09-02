import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from app.agent import dispatcher, runtime
from app.agent.contracts import EXPERT_CONTRACTS, template_expert_tasks
from app.agent.validation import report_similarity, validate_report, validate_reports


@pytest.mark.asyncio
async def test_dispatcher_generates_four_distinct_tasks(monkeypatch):
    monkeypatch.setattr(dispatcher.settings, "agent_expert_dispatch_mode", "hybrid")
    payload = '''{"character":"分析林舟与沈月的关系边和动机变化","plot":"提取救援事件导致关系变化的因果链","timeline":"按初遇、冲突、和解的先后顺序整理","locator":"核对关键关系结论对应的章节页码和片段"}'''
    result = await dispatcher.dispatch_expert_tasks(
        "梳理林舟和沈月的关系变化",
        llm=RunnableLambda(lambda _: AIMessage(content=payload)),
    )
    assert result.mode == "hybrid"
    assert len({task["task"] for task in result.tasks.values()}) == 4
    assert result.tasks["locator"]["output_format"].startswith("Markdown")


@pytest.mark.asyncio
async def test_dispatcher_falls_back_on_invalid_json(monkeypatch):
    monkeypatch.setattr(dispatcher.settings, "agent_expert_dispatch_mode", "hybrid")
    result = await dispatcher.dispatch_expert_tasks(
        "梳理人物关系",
        llm=RunnableLambda(lambda _: AIMessage(content="无法生成")),
    )
    assert result.mode == "template"
    assert result.reason == "dispatcher_fallback"
    assert set(result.tasks) == {"character", "plot", "timeline", "locator"}


@pytest.mark.asyncio
async def test_template_mode_does_not_call_llm(monkeypatch):
    monkeypatch.setattr(dispatcher.settings, "agent_expert_dispatch_mode", "template")
    called = False

    async def should_not_call(_):
        nonlocal called
        called = True
        return AIMessage(content="{}")

    result = await dispatcher.dispatch_expert_tasks("梳理人物关系", llm=RunnableLambda(should_not_call))
    assert result.mode == "template"
    assert called is False


def test_locator_contract_rejects_report_without_location():
    result = validate_report("locator", "人物关系发生变化 [S1]，但没有进一步信息。")
    assert result["contract_ok"] is False
    assert any("定位" in item or "章节" in item for item in result["missing_sections"])


def test_similarity_detection_selects_lower_contract_report():
    shared = "关系与立场发生变化，事实与推断分开，人物动机证据不足 [S1]"
    reports = [
        {"agent": "character", "status": "ok", "report": shared},
        {"agent": "plot", "status": "ok", "report": shared},
    ]
    validations, refine = validate_reports(reports, 0.72)
    assert report_similarity(shared, shared) == 1.0
    assert "plot" in refine
    assert validations["plot"]["similarity_flags"]


def test_template_tasks_keep_contracts_distinct():
    tasks = template_expert_tasks("梳理主要人物关系")
    assert len({task["task"] for task in tasks.values()}) == 4
    assert tasks["character"]["focus"] != tasks["timeline"]["focus"]


@pytest.mark.asyncio
async def test_dispatcher_falls_back_when_four_tasks_are_identical(monkeypatch):
    monkeypatch.setattr(dispatcher.settings, "agent_expert_dispatch_mode", "hybrid")
    same = "完整分析所有人物关系和全部情节发展"
    payload = {name: same for name in ("character", "plot", "timeline", "locator")}
    result = await dispatcher.dispatch_expert_tasks(
        "梳理人物关系",
        llm=RunnableLambda(lambda _: AIMessage(content=__import__("json").dumps(payload, ensure_ascii=False))),
    )
    assert result.mode == "template"
    assert result.reason == "dispatcher_fallback"


def test_specialist_prompt_uses_exclusive_task_and_original_only_as_context():
    tasks = template_expert_tasks("梳理人物关系和事件顺序")
    state = {
        "standalone_query": "梳理人物关系和事件顺序",
        "expert_tasks": tasks,
        "evidence": [{"source": {"id": "S1", "source": "book.txt", "chapter": "第一章"}, "content": "原文"}],
    }
    character_prompt = runtime._specialist_prompt(EXPERT_CONTRACTS["character"], state, None)
    timeline_prompt = runtime._specialist_prompt(EXPERT_CONTRACTS["timeline"], state, None)
    assert tasks["character"]["task"] in character_prompt
    assert tasks["timeline"]["task"] in timeline_prompt
    assert "原始用户问题仅用于理解背景，不要求你完整回答" in character_prompt
    assert character_prompt != timeline_prompt
    assert "[S1]" in character_prompt and "[S1]" in timeline_prompt
