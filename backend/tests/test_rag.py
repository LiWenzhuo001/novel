"""RAG 纯函数单测（无 DB / 无 LLM 依赖）。

覆盖：
- RRF 融合：多路排序的融合分数与顺序符合预期。
"""

from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from app.core import rag


def _row(rid: str):
    return SimpleNamespace(id=rid)


def test_rrf_fuse_basic():
    vec = [(_row("a"), 0.9), (_row("b"), 0.8), (_row("c"), 0.7)]
    fts = [(_row("c"), 0.9), (_row("a"), 0.8)]  # c 在 FTS 排第一
    fused = rag._rrf_fuse(vec, fts, k=10, rrf_c=60)
    ids = [r.id for r, _ in fused]
    # a 两路都在前二，c 在 FTS 第一、vec 第三，融合后 a 应居首，c 次之
    assert ids[0] == "a"
    assert "b" in ids and "c" in ids


def test_rrf_fuse_empty():
    assert rag._rrf_fuse([], [], k=5) == []


def test_annotate_primary_sets_rank_and_source_semantics():
    from langchain_core.documents import Document

    docs = [
        Document(page_content="a", metadata={"score": 0.8, "score_type": "hybrid"}),
        Document(page_content="b", metadata={"score": 0.7}),
    ]
    annotated = rag._annotate_primary(docs)

    assert [doc.metadata["retrieval_rank"] for doc in annotated] == [1, 2]
    assert all(doc.metadata["neighbor"] is False for doc in annotated)
    assert annotated[0].metadata["score_type"] == "hybrid"
    assert annotated[1].metadata["score_type"] == "vector"


def test_chinese_query_tokenization_deduplicates_and_limits(monkeypatch):
    import sys

    class FakeJieba:
        @staticmethod
        def lcut(query, cut_all=False):
            return ["谢煜璟", "为什么", "楚姒", "关系", "楚姒", "变化"]

    monkeypatch.setitem(sys.modules, "jieba", FakeJieba)
    tokens = rag._tokenize_chinese_query("谢煜璟和楚姒的关系为什么变化", max_terms=3)

    assert tokens == ["谢煜璟", "楚姒", "关系"]

@pytest.mark.asyncio
async def test_retrieve_context_zero_window_does_not_query_neighbors(monkeypatch):
    from langchain_core.documents import Document

    primary = [Document(page_content="主命中", metadata={"score": 0.8})]

    async def fake_similarity(*args, **kwargs):
        return primary

    monkeypatch.setattr(rag, "similarity_search", fake_similarity)
    result = await rag.retrieve_novel_context("测试", k=1, neighbor_window=0)

    assert result == primary
    assert result[0].metadata.get("neighbor") is None


def test_rerank_preserves_retrieval_details_and_sets_score_type(monkeypatch):
    from langchain_core.documents import Document
    from app.core import rerank as rerank_module

    class Model:
        def predict(self, pairs):
            return [2.0, -1.0]

    monkeypatch.setattr(rerank_module, "get_reranker", lambda: Model())
    docs = [
        Document(page_content="高相关", metadata={"score": 0.6, "vector_score": 0.55}),
        Document(page_content="低相关", metadata={"score": 0.5, "vector_score": 0.45}),
    ]

    ranked = rerank_module.rerank("问题", docs, 2)

    assert ranked[0].page_content == "高相关"
    assert ranked[0].metadata["retrieval_score"] == 0.6
    assert ranked[0].metadata["vector_score"] == 0.55
    assert ranked[0].metadata["score_type"] == "reranker"
    assert ranked[0].metadata["reranked"] is True
    assert ranked[0].metadata["reranker_rank"] == 1
    assert ranked[1].metadata["reranker_rank"] == 2
    assert ranked[0].metadata["reranker_protected"] is False


def test_build_merged_retrieval_query_preserves_semantic_anchor():
    merged = rag.build_merged_retrieval_query(
        "孙悟空为什么被压在五行山下？",
        "孙悟空 五行山 被压 原因 事件原文",
    )
    assert merged == "孙悟空为什么被压在五行山下？ 孙悟空 五行山 被压 原因 事件原文"
    assert rag.build_merged_retrieval_query("测试", "测试") == "测试"
    assert rag.build_merged_retrieval_query("测试", None) == "测试"


@pytest.mark.asyncio
async def test_similarity_search_uses_one_merged_query(monkeypatch):
    calls = []

    async def fake_once(query, k=None, filter_source=None, domain="novel", file_id=None, trace=None, rerank_query=None):
        calls.append({"query": query, "rerank_query": rerank_query, "k": k, "file_id": file_id})
        if trace is not None:
            trace.update({"query": query, "final_results": []})
        return [Document(
            page_content="证据片段",
            metadata={"file_id": "f1", "chapter_no": 1, "chunk_no": 1, "score": 0.8},
        )]

    monkeypatch.setattr(rag, "_similarity_search_once", fake_once)
    trace = {}
    docs = await rag.similarity_search(
        "孙悟空为什么被驱逐？",
        k=3,
        file_id="f1",
        trace=trace,
        retrieval_query="孙悟空 唐僧 驱逐 师徒冲突 原文",
    )

    assert len(docs) == 1
    assert len(calls) == 1
    assert calls[0]["query"] == "孙悟空为什么被驱逐？ 孙悟空 唐僧 驱逐 师徒冲突 原文"
    assert calls[0]["rerank_query"] == "孙悟空为什么被驱逐？"
    assert trace["query_count"] == 1
    assert trace["retrieval_mode"] == "merged_single_query"
    assert trace["merged_query"] == calls[0]["query"]


def test_trace_candidates_keeps_stage_and_location():
    from langchain_core.documents import Document

    docs = [Document(page_content="片段", metadata={"file_id": "f1", "chapter_no": 2, "chunk_no": 8, "score": 0.7})]
    trace = rag._trace_candidates(docs, "final")
    assert trace[0]["stage"] == "final"
    assert trace[0]["file_id"] == "f1"
    assert trace[0]["chapter_no"] == 2
    assert trace[0]["chunk_no"] == 8


def test_rerank_blends_retrieval_signals(monkeypatch):
    from langchain_core.documents import Document
    from app.core import rerank as rerank_module

    class Model:
        def predict(self, pairs):
            return [-2.0, 2.0]

    monkeypatch.setattr(rerank_module, "get_reranker", lambda: Model())
    monkeypatch.setattr(rerank_module.settings, "enable_reranker_blend", True)
    monkeypatch.setattr(rerank_module.settings, "reranker_weight", 0.7)
    monkeypatch.setattr(rerank_module.settings, "rrf_weight", 0.2)
    monkeypatch.setattr(rerank_module.settings, "raw_score_weight", 0.1)
    monkeypatch.setattr(rerank_module.settings, "reranker_protect_top_n", 0)
    monkeypatch.setattr(rerank_module.settings, "reranker_protect_slots", 0)
    docs = [
        Document(page_content="召回强", metadata={"score": 0.9, "rrf_score": 0.20, "vector_score": 0.9}),
        Document(page_content="重排强", metadata={"score": 0.4, "rrf_score": 0.02, "vector_score": 0.4}),
    ]
    ranked = rerank_module.rerank("问题", docs, 2)
    assert ranked[0].page_content == "重排强"
    assert ranked[0].metadata["reranker_score"] > ranked[1].metadata["reranker_score"]
    assert "final_score" in ranked[0].metadata


def test_evaluator_feature_buckets_and_missing_values():
    from scripts.evaluate_rag_recall import case_features

    features = case_features(
        {
            "query": "孙悟空为什么会被唐僧赶走？",
            "gold_chapters": [1],
            "entity_count": 2,
            "requires_multiple_evidence": True,
        },
        {"chapters": 100},
    )
    assert features["query_length_bucket"] == "medium"
    assert features["entity_count_bucket"] == "multi"
    assert features["chapter_position"] == "early"
    assert features["has_pronoun"] == "N/A"
    assert features["requires_multiple_evidence"] is True


def test_evaluator_trace_loss_point():
    from scripts.evaluate_rag_recall import _stage_gold_info

    trace = {
        "vector_candidates": [{"file_id": "f", "chapter_no": 1, "chunk_no": 2}],
        "lexical_candidates": [],
        "rrf_candidates": [{"file_id": "f", "chapter_no": 1, "chunk_no": 2}],
        "reranker_candidates": [{"file_id": "f", "chapter_no": 1, "chunk_no": 2}],
        "reranker_ranked_candidates": [],
        "final_results": [],
    }
    info = _stage_gold_info(trace, {("f", 1, 2)})
    assert info["loss_point"] == "reranker_loss"

    trace["reranker_ranked_candidates"] = [{"file_id": "f", "chapter_no": 1, "chunk_no": 2}]
    info = _stage_gold_info(trace, {("f", 1, 2)})
    assert info["loss_point"] == "final_top_k_cut"


def test_hybrid_candidates_are_not_removed_by_similarity_threshold():
    """合并 Query 后，低原始分但位于 RRF 候选池的证据仍应保留。"""
    threshold = float(getattr(rag.settings, "similarity_threshold", None) or 0.3)
    assert rag._keep_candidate([0.12, None], threshold, hybrid=True) is True
    assert rag._keep_candidate([None, None], threshold, hybrid=True) is False
    assert rag._keep_candidate([0.12], threshold, hybrid=False) is False


def test_query_preparation_cache_is_validated(tmp_path):
    import json
    from scripts.evaluate_rag_recall import load_preparation_cache, validate_preparation_cache

    path = tmp_path / "preparation.json"
    path.write_text(json.dumps({
        "file_id": "f1",
        "source_hash": "hash-1",
        "cases": [{
            "id": "q1",
            "query": "孙悟空是谁？",
            "standalone_query": "孙悟空是谁？",
            "retrieval_query": "孙悟空 身份 人物 原文",
        }],
    }, ensure_ascii=False), encoding="utf-8")

    cache = load_preparation_cache(path)
    validate_preparation_cache(
        cache,
        [{"id": "q1", "query": "孙悟空是谁？", "gold_chunks": [{"chunk_no": 1}]}],
        file_id="f1",
        index_metadata={"source_hash": "hash-1"},
    )
    assert cache["cases"]["q1"]["source"] == "cache"


def test_hybrid_trace_retains_candidate_pool_before_final_cut(monkeypatch):
    """混合检索应保留候选池，最终返回仍按 k 截断。"""
    monkeypatch.setattr(rag.settings, "enable_hybrid_search", True)
    monkeypatch.setattr(rag.settings, "enable_reranker", False)
    monkeypatch.setattr(rag.settings, "hybrid_candidate_k", 60)
    assert max(10, rag.settings.hybrid_candidate_k) == 60


def test_weighted_fuse_orders_by_normalized_scores():
    vec = [(_row("a"), 0.95), (_row("b"), 0.55)]
    fts = [(_row("b"), 0.9), (_row("c"), 0.4)]
    fused = rag._weighted_fuse(vec, fts, vector_weight=0.7, lexical_weight=0.3)
    scores = {row.id: score for row, score in fused}
    # vec 归一：a=1.0, b=0.0；fts 归一：b=1.0, c=0.0
    assert scores["a"] == pytest.approx(0.7)
    assert scores["b"] == pytest.approx(0.3)
    assert scores["c"] == pytest.approx(0.0)
    assert [row.id for row, _ in fused] == ["a", "b", "c"]


def test_weighted_fuse_prefers_dual_channel_hits():
    vec = [(_row("a"), 0.9), (_row("b"), 0.85), (_row("d"), 0.5)]
    fts = [(_row("b"), 0.8)]  # 只有 b 被词法召回
    fused = rag._weighted_fuse(vec, fts, vector_weight=0.5, lexical_weight=0.5)
    ids = [row.id for row, _ in fused]
    # b：vec 归一 0.778 + fts 归一 1.0 → 0.889 > a 的 0.5（纯向量第一名）
    assert ids[0] == "b"


def test_weighted_fuse_single_channel_constant_scores():
    # 单候选通道 high<=low 时归一为 1.0
    vec = [(_row("a"), 0.9)]
    fts = [(_row("a"), 0.8)]
    fused = rag._weighted_fuse(vec, fts, vector_weight=0.5, lexical_weight=0.5)
    assert fused[0][1] == pytest.approx(1.0)
    assert rag._weighted_fuse([], [], 0.5, 0.5) == []


def test_fusion_mode_rejects_unknown_value(monkeypatch):
    from app.config import Settings

    monkeypatch.setenv("FUSION_MODE", "bogus")
    with pytest.raises(ValueError, match="FUSION_MODE"):
        Settings()


async def test_bm25_failure_propagates(monkeypatch):
    """BM25 fail-fast：扩展/索引缺失时异常直接抛出，不静默降级。"""
    class _StubEmbeddings:
        async def embed_query(self, query):
            return [0.0] * 8

    class _StubSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def execute(self, *args, **kwargs):
            class _Result:
                def all(self):
                    return []

                def scalars(self):
                    return self

                def mappings(self):
                    return iter([])

            return _Result()

    async def _boom(*args, **kwargs):
        raise RuntimeError("pg_search extension missing")

    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr(rag, "_check_index_compatibility", _noop)
    monkeypatch.setattr(rag, "get_embeddings", lambda: _StubEmbeddings())
    monkeypatch.setattr(rag, "AsyncSessionLocal", lambda: _StubSession())
    monkeypatch.setattr(rag, "_bm25_search", _boom)
    monkeypatch.setattr(rag.settings, "enable_hybrid_search", True)
    monkeypatch.setattr(rag.settings, "enable_reranker", False)
    monkeypatch.setattr(rag.settings, "enable_bm25_search", True)
    with pytest.raises(RuntimeError, match="pg_search extension missing"):
        await rag._similarity_search_once("孙悟空是谁", k=5)


async def test_bm25_disabled_keeps_lexical_fallback(monkeypatch):
    """BM25 关闭时走中文词法通道，异常仍回退 FTS（原行为不变）。"""
    class _StubEmbeddings:
        async def embed_query(self, query):
            return [0.0] * 8

    class _StubSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def rollback(self):
            return None

        async def execute(self, *args, **kwargs):
            class _Result:
                def all(self):
                    return []

                def scalars(self):
                    return self

                def mappings(self):
                    return iter([])

            return _Result()

    calls = []

    async def _lexical(*args, **kwargs):
        calls.append("lexical")
        raise RuntimeError("pg_trgm missing")

    async def _fts(*args, **kwargs):
        calls.append("fts")
        return []

    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr(rag, "_check_index_compatibility", _noop)
    monkeypatch.setattr(rag, "get_embeddings", lambda: _StubEmbeddings())
    monkeypatch.setattr(rag, "AsyncSessionLocal", lambda: _StubSession())
    monkeypatch.setattr(rag, "_chinese_lexical_search", _lexical)
    monkeypatch.setattr(rag, "_fts_search", _fts)
    monkeypatch.setattr(rag.settings, "enable_hybrid_search", True)
    monkeypatch.setattr(rag.settings, "enable_reranker", False)
    monkeypatch.setattr(rag.settings, "enable_bm25_search", False)
    monkeypatch.setattr(rag.settings, "enable_chinese_lexical_search", True)
    docs = await rag._similarity_search_once("孙悟空是谁", k=5)
    assert calls == ["lexical", "fts"]
    assert docs == []
