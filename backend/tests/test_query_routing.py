import json

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from app.core import llm, query_rewriter, rag
from app.agent.router import route_query
from app.agent.types import Strategy
from scripts.evaluate_rag_recall import trace_stage_candidates


def test_get_llm_uses_configured_runtime_options(monkeypatch):
    monkeypatch.setattr(llm.settings, "llm_model", "test-model")
    monkeypatch.setattr(llm.settings, "llm_api_key", "test-key")
    monkeypatch.setattr(llm.settings, "llm_base_url", "https://example.test/v1")
    monkeypatch.setattr(llm.settings, "llm_timeout", 12.0)
    monkeypatch.setattr(llm.settings, "llm_max_retries", 2)
    monkeypatch.setattr(llm.settings, "validate", lambda: None)

    model = llm.get_llm(temperature=0)

    assert model.model_name == "test-model"
    assert model.request_timeout == 12.0
    assert model.max_retries == 2


@pytest.mark.asyncio
async def test_rewrite_query_uses_history(monkeypatch):
    monkeypatch.setattr(query_rewriter.settings, "enable_query_rewrite", True)
    payload = {
        "standalone_query": "林舟经历了哪些转折？",
        "retrieval_query": "林舟 成长经历 关键转折 事件经过 原文证据",
        "intent": "plot_causality",
        "entities": ["林舟"],
        "evidence_focus": ["成长经历", "关键转折"],
        "confidence": 0.95,
    }
    model = RunnableLambda(lambda _: AIMessage(content=json.dumps(payload, ensure_ascii=False)))

    result = await query_rewriter.rewrite_query(
        "他经历了哪些转折？",
        [HumanMessage(content="介绍一下主角林舟的成长经历")],
        llm=model,
    )

    assert result.applied is True
    assert result.query == result.standalone_query
    assert "林舟" in result.standalone_query
    assert "关键转折" in result.retrieval_query


@pytest.mark.asyncio
async def test_rewrite_query_runs_on_first_question(monkeypatch):
    monkeypatch.setattr(query_rewriter.settings, "enable_query_rewrite", True)
    calls = 0
    payload = {
        "standalone_query": "孙悟空为什么被唐僧驱逐？",
        "retrieval_query": "孙悟空 唐僧 驱逐 师徒冲突 起因 经过 原文",
        "intent": "character_relation",
        "entities": ["孙悟空", "唐僧"],
        "evidence_focus": ["驱逐原因", "冲突经过"],
        "confidence": 0.92,
    }

    async def invoke(_):
        nonlocal calls
        calls += 1
        return AIMessage(content=json.dumps(payload, ensure_ascii=False))

    result = await query_rewriter.rewrite_query(
        "孙悟空为什么被唐僧赶走？",
        [],
        llm=RunnableLambda(invoke),
    )

    assert calls == 1
    assert result.reason == "rewritten"
    assert result.retrieval_query != result.original


@pytest.mark.asyncio
async def test_rewrite_query_invalid_payload_calls_llm_once(monkeypatch):
    """契约失败直接回退，不为同一问题发起第二次准备请求。"""
    monkeypatch.setattr(query_rewriter.settings, "enable_query_rewrite", True)
    calls = 0

    async def invoke(_):
        nonlocal calls
        calls += 1
        return AIMessage(content="not-json")

    result = await query_rewriter.rewrite_query("孙悟空是谁？", [], llm=RunnableLambda(invoke))

    assert calls == 1
    assert result.applied is False
    assert result.reason == "missing_query"



@pytest.mark.asyncio
async def test_rewrite_query_falls_back_on_error(monkeypatch):
    monkeypatch.setattr(query_rewriter.settings, "enable_query_rewrite", True)

    async def fail(_):
        raise RuntimeError("model unavailable")

    result = await query_rewriter.rewrite_query(
        "他后来去了哪里？",
        [HumanMessage(content="介绍一下情节发展")],
        llm=RunnableLambda(fail),
    )

    assert result.query == "他后来去了哪里？"
    assert result.retrieval_query == "他后来去了哪里？"
    assert result.reason == "error"


@pytest.mark.asyncio
async def test_rewrite_query_keeps_inferred_retrieval_clues(monkeypatch):
    """检索线索可包含模型推断的原文表达，但自然问题不能引入新实体。"""
    monkeypatch.setattr(query_rewriter.settings, "enable_query_rewrite", True)
    payload = {
        "standalone_query": "孙悟空为什么被唐僧驱逐？",
        "retrieval_query": "孙悟空 唐僧 三打白骨精 误会 驱逐 师徒冲突",
        "intent": "plot_causality",
        "entities": ["孙悟空", "唐僧", "三打白骨精"],
        "evidence_focus": ["驱逐原因", "冲突经过"],
        "confidence": 0.92,
    }
    result = await query_rewriter.rewrite_query(
        "孙悟空为什么被唐僧驱逐？",
        [],
        llm=RunnableLambda(lambda _: AIMessage(content=json.dumps(payload, ensure_ascii=False))),
    )

    assert result.applied is True
    assert result.retrieval_query == payload["retrieval_query"]
    assert result.entities == ["孙悟空", "唐僧"]



@pytest.mark.asyncio
async def test_rewrite_query_rejects_invented_entity(monkeypatch):
    monkeypatch.setattr(query_rewriter.settings, "enable_query_rewrite", True)
    payload = {
        "standalone_query": "林舟与未知角色的关系？",
        "retrieval_query": "林舟 未知角色 关系",
        "intent": "character_relation",
        "entities": ["林舟", "未知角色"],
        "evidence_focus": ["关系"],
        "confidence": 0.99,
    }
    result = await query_rewriter.rewrite_query(
        "林舟经历了什么？",
        [],
        llm=RunnableLambda(lambda _: AIMessage(content=json.dumps(payload, ensure_ascii=False))),
    )
    assert result.reason == "invented_entity"
    assert result.retrieval_query == result.original


def test_hybrid_rrf_keeps_fts_only_hit(monkeypatch):
    class Row:
        id = "fts-only"
        source = "novel.txt"
        file_id = "file-1"
        meta_json = "{}"
        content = "负责 FastAPI RAG 项目"
        domain = "novel"
        chapter = None
        chapter_no = None
        chunk_no = None
        page = None

    row = Row()
    monkeypatch.setattr(rag.settings, "similarity_threshold", 0.3)
    fused = [(row, 0.01)]
    vec_score_map = {}
    fts_score_map = {row.id: 0.8}
    pool = []
    for item, fused_score in fused:
        vec_score = vec_score_map.get(item.id)
        fts_score = fts_score_map.get(item.id)
        best_score = max(score for score in (vec_score, fts_score) if score is not None)
        if best_score >= rag.settings.similarity_threshold:
            meta = rag._parse_meta(item)
            meta["score"] = round(best_score, 4)
            meta["rrf_score"] = round(fused_score, 6)
            pool.append(Document(page_content=item.content, metadata=meta))

    assert len(pool) == 1
    assert pool[0].metadata["score"] == 0.8

def test_trace_stage_candidates_supports_single_query_trace():
    trace = {
        "vector_candidates": [
            {"file_id": "f1", "chapter_no": 1, "chunk_no": 2},
        ]
    }

    assert trace_stage_candidates(trace, "vector_candidates") == trace["vector_candidates"]


def test_trace_stage_candidates_merges_multi_query_trace_by_chunk():
    duplicate = {"file_id": "f1", "chapter_no": 1, "chunk_no": 2}
    unique = {"file_id": "f1", "chapter_no": 1, "chunk_no": 3}
    trace = {
        "variant_traces": [
            {"role": "base", "vector_candidates": [duplicate]},
            {"role": "retrieval", "vector_candidates": [duplicate, unique]},
        ]
    }

    assert trace_stage_candidates(trace, "vector_candidates") == [duplicate, unique]


@pytest.mark.asyncio
async def test_retrieve_tool_respects_zero_neighbor_window(monkeypatch):
    from app.agent import tools

    captured = {}

    async def fake_retrieve(query, *, k, neighbor_window, file_id):
        captured.update(query=query, k=k, neighbor_window=neighbor_window, file_id=file_id)
        return []

    monkeypatch.setattr(tools, "retrieve_novel_context", fake_retrieve)
    monkeypatch.setattr(tools.settings, "novel_neighbor_window", 0)

    result = await tools._retrieve_novel(query="测试", file_id="f1")

    assert result.status == "ok"
    assert captured["neighbor_window"] is None


@pytest.mark.asyncio
async def test_retrieve_tool_propagates_source_score_metadata(monkeypatch):
    from app.agent import tools

    async def fake_retrieve(*args, **kwargs):
        return [Document(page_content="上下文", metadata={
            "source": "novel.txt", "chunk_no": 2, "score": 0.0,
            "score_type": "neighbor", "neighbor": True,
            "retrieval_rank": None, "vector_score": None,
            "fts_score": None, "rrf_score": None, "reranked": False,
        })]

    monkeypatch.setattr(tools, "retrieve_novel_context", fake_retrieve)
    result = await tools._retrieve_novel(query="测试")
    source = result.output["sources"][0]

    assert source["neighbor"] is True
    assert source["score_type"] == "neighbor"
    assert source["score"] == 0.0


def test_route_skips_rag_for_conversation_only_messages():
    decision = route_query("你好，谢谢，继续刚才的风格", requested_strategy="auto")
    assert decision.needs_retrieval is False
    assert decision.answer_mode == "conversation"
    assert decision.allowed_tools == ()
    assert decision.requires_citation is False


def test_route_keeps_rag_for_novel_fact_questions():
    decision = route_query("这个关键转折最早出现在哪一章？", requested_strategy="direct")
    assert decision.needs_retrieval is True
    assert decision.answer_mode == "novel_evidence"
    assert "retrieve_novel" in decision.allowed_tools


def test_multi_expert_is_downgraded_without_rag():
    decision = route_query("记住我的回答偏好：请简短一点", requested_strategy="multi_expert")
    assert decision.strategy is Strategy.DIRECT
    assert decision.needs_retrieval is False
    assert "strategy_downgraded" in decision.retrieval_reason


def test_route_decision_serializes_retrieval_fields():
    payload = route_query("你好").as_dict()
    assert payload["needs_retrieval"] is False
    assert payload["retrieval_reason"] == "conversation_only"
    assert payload["answer_mode"] == "conversation"


@pytest.mark.asyncio
async def test_query_preparation_can_skip_rag_for_user_preference(monkeypatch):
    monkeypatch.setattr(query_rewriter.settings, "enable_query_rewrite", True)
    payload = {
        "standalone_query": "用户希望后续回答更简短",
        "retrieval_query": "",
        "intent": "other",
        "needs_retrieval": False,
        "answer_mode": "memory_context",
        "retrieval_reason": "user_preference",
        "entities": [],
        "evidence_focus": [],
        "confidence": 0.98,
    }
    result = await query_rewriter.rewrite_query(
        "以后回答简短一点",
        [],
        llm=RunnableLambda(lambda _: AIMessage(content=json.dumps(payload, ensure_ascii=False))),
    )
    assert result.reason == "rewritten"
    assert result.needs_retrieval is False
    assert result.retrieval_query == ""
    assert result.answer_mode == "memory_context"


@pytest.mark.asyncio
async def test_query_preparation_rejects_non_rag_with_retrieval_query(monkeypatch):
    monkeypatch.setattr(query_rewriter.settings, "enable_query_rewrite", True)
    payload = {
        "standalone_query": "用户希望后续回答更简短",
        "retrieval_query": "回答 简短",
        "intent": "other",
        "needs_retrieval": False,
        "answer_mode": "memory_context",
        "retrieval_reason": "user_preference",
        "entities": [],
        "evidence_focus": [],
        "confidence": 0.98,
    }
    result = await query_rewriter.rewrite_query(
        "以后回答简短一点",
        [],
        llm=RunnableLambda(lambda _: AIMessage(content=json.dumps(payload, ensure_ascii=False))),
    )
    assert result.reason == "unexpected_retrieval_query"
    assert result.needs_retrieval is True
    assert result.retrieval_reason == "query_preparation_failed"


def _routing_hint(**overrides):
    hint = {
        "original": "谢谢，继续用简短风格",
        "standalone_query": "继续使用简短回答风格",
        "retrieval_query": "",
        "needs_retrieval": False,
        "answer_mode": "memory_context",
        "retrieval_reason": "conversation_preference",
        "confidence": 0.95,
        "reason": "rewritten",
    }
    hint.update(overrides)
    return hint


def test_route_honors_high_confidence_llm_non_rag_hint(monkeypatch):
    monkeypatch.setattr(query_rewriter.settings, "enable_llm_query_routing", True)
    decision = route_query("继续使用简短回答风格", "auto", routing_hint=_routing_hint())
    assert decision.needs_retrieval is False
    assert decision.llm_needs_retrieval is False
    assert decision.routing_override is False


def test_route_overrides_llm_non_rag_hint_for_strong_novel_signal(monkeypatch):
    monkeypatch.setattr(query_rewriter.settings, "enable_llm_query_routing", True)
    hint = _routing_hint(original="请核对原文依据")
    decision = route_query("请核对原文依据", "direct", routing_hint=hint)
    assert decision.llm_needs_retrieval is False
    assert decision.needs_retrieval is True
    assert decision.routing_override is True
    assert "strong_novel_signal" in decision.routing_override_reason


def test_route_overrides_low_confidence_non_rag_hint(monkeypatch):
    monkeypatch.setattr(query_rewriter.settings, "enable_llm_query_routing", True)
    hint = _routing_hint(confidence=0.4)
    decision = route_query("继续使用简短回答风格", "direct", routing_hint=hint)
    assert decision.needs_retrieval is True
    assert decision.routing_override is True
    assert "low_confidence" in decision.routing_override_reason


def test_route_forces_rag_when_query_preparation_failed(monkeypatch):
    monkeypatch.setattr(query_rewriter.settings, "enable_llm_query_routing", True)
    hint = _routing_hint(reason="error", confidence=0.0, needs_retrieval=True,
                         answer_mode="novel_evidence", retrieval_query="未知问题")
    decision = route_query("未知问题", "direct", routing_hint=hint)
    assert decision.needs_retrieval is True
    assert decision.llm_needs_retrieval is None
    assert decision.routing_override is True
    assert decision.retrieval_reason == "query_preparation_failed"


@pytest.mark.asyncio
async def test_output_preference_is_detected_even_when_llm_returns_legacy_payload(monkeypatch):
    monkeypatch.setattr(query_rewriter.settings, "enable_query_rewrite", True)
    legacy = {
        "standalone_query": "以后只给总结",
        "retrieval_query": "展示原文",
        "intent": "other",
        "entities": [],
        "evidence_focus": [],
        "confidence": 0.9,
    }
    result = await query_rewriter.rewrite_query(
        "以后不要给我展示原文，直接告诉我总结好的就行",
        [],
        llm=RunnableLambda(lambda _: AIMessage(content=json.dumps(legacy, ensure_ascii=False))),
    )
    assert result.needs_retrieval is False
    assert result.answer_mode == "memory_context"
    assert result.retrieval_query == ""
    assert result.output_policy["summary_only"] is True
    assert result.preference_update["preference_key"] == "answer_presentation"


def test_route_does_not_force_rag_for_negated_source_preference(monkeypatch):
    monkeypatch.setattr(query_rewriter.settings, "enable_llm_query_routing", True)
    hint = {
        "reason": "rewritten",
        "needs_retrieval": False,
        "answer_mode": "memory_context",
        "retrieval_reason": "user_output_preference",
        "retrieval_query": "",
        "confidence": 0.99,
        "output_policy": {"summary_only": True, "show_source_text": False, "allow_direct_quotes": False},
        "preference_update": {"preference_key": "answer_presentation"},
    }
    decision = route_query("以后不要给我展示原文，直接告诉我总结好的就行", "auto", routing_hint=hint)
    assert decision.needs_retrieval is False
    assert decision.routing_override is False
    assert decision.output_policy["show_source_text"] is False


def test_route_keeps_summary_policy_when_mixed_with_novel_question(monkeypatch):
    monkeypatch.setattr(query_rewriter.settings, "enable_llm_query_routing", True)
    hint = {
        "reason": "rewritten",
        "needs_retrieval": True,
        "answer_mode": "novel_evidence",
        "retrieval_reason": "novel_fact",
        "retrieval_query": "孙悟空 离开女儿国 原因",
        "confidence": 0.98,
        "original": "不要展示原文，但告诉我孙悟空为什么离开女儿国",
        "output_policy": {"summary_only": True, "show_source_text": False, "allow_direct_quotes": False},
        "preference_update": {"preference_key": "answer_presentation"},
    }
    decision = route_query(hint["original"], "direct", routing_hint=hint)
    assert decision.needs_retrieval is True
    assert decision.output_policy["summary_only"] is True
    assert decision.output_policy["show_source_text"] is False
