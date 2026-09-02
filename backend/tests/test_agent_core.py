from app.agent.router import normalize_strategy, route_query
from app.agent.tools import _calculator, build_default_registry
from app.agent.types import Strategy
from app.models.schemas import ChatRequest

import pytest


def test_strategy_values_are_normalized():
    assert normalize_strategy("auto", "谁是林舟？") is Strategy.DIRECT
    assert normalize_strategy("auto", "梳理人物关系和时间线") is Strategy.MULTI_EXPERT
    assert normalize_strategy("multi_expert", "谁是林舟？") is Strategy.MULTI_EXPERT
    assert normalize_strategy("plan_execute", "查找并计算") is Strategy.PLAN_EXECUTE


def test_multi_expert_router_assigns_shared_retrieval_tools():
    decision = route_query("跨章节梳理人物关系和时间线", requested_strategy="multi_expert")
    assert decision.strategy is Strategy.MULTI_EXPERT
    assert decision.intent == "novel_analysis"
    assert decision.allowed_tools == ("retrieve_novel", "get_chapter_context")
    assert decision.max_steps >= 3


@pytest.mark.asyncio
async def test_calculator_rejects_python_code():
    result = await _calculator(expression="__import__('os').system('echo bad')")
    assert result.status == "error"
    assert result.error_code == "unsupported_expression"


@pytest.mark.asyncio
async def test_registry_denies_unlisted_tool():
    registry = build_default_registry()
    result = await registry.execute("calculator", allowed_tools=["retrieve_novel"], expression="1+1")
    assert result.status == "denied"
    assert result.error_code == "tool_not_allowed"


def test_chat_request_accepts_multi_expert():
    request = ChatRequest(message="梳理人物关系", strategy="multi_expert")
    assert request.strategy == "multi_expert"


def test_legacy_agent_mode_is_rejected():
    with pytest.raises(ValueError):
        ChatRequest(message="测试", agent_mode="single")
