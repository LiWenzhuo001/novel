"""基于 PostgreSQL/pgvector 的混合检索、重排和章节上下文扩展。「

门面模块：2026-08-30 起检索实现拆至 app/core/retrieval/ 各职责模块；
本文件保留主检索编排（_similarity_search_once 主管线（与高层门面函数，
并 re-export 全部旧符号，供 tools.py/kb_service.py/mcp_server.py/评测脚本/测试零改动导入。
"""

from __future__ import annotations

import asyncio
import time

from typing import Iterable, List

from langchain_core.documents import Document
from sqlalchemy import select

from app.config import settings
from app.core.context import get_current_user
from app.core.embed import get_embeddings
from app.core.logging_config import get_logger
from app.core.metrics import metrics
from app.db import AsyncSessionLocal
from app.db.models import KnowledgeFile

# ---- re-export（门面：保留全部旧符号（ ----  # noqa: E402
from app.core.retrieval.channels import (  # noqa: F401
    _CJK_RE,  # noqa: F401
    _FALLBACK_TOKEN_RE,  # noqa: F401
    _CHINESE_STOPWORDS,  # noqa: F401
    _BM25_SCORE_SQL,  # noqa: F401
    _parse_meta,  # noqa: F401
    _apply_filters,  # noqa: F401
    _vector_search,  # noqa: F401
    _fts_search,  # noqa: F401
    _tokenize_chinese_query,  # noqa: F401
    _chinese_lexical_search,  # noqa: F401
    _tokenize_for_bm25,  # noqa: F401
    _bm25_search,  # noqa: F401
)
from app.core.retrieval.fusion import _rrf_fuse, _minmax_normalize, _weighted_fuse  # noqa: F401
from app.core.retrieval.indexer import (  # noqa: F401
    _rows_for_documents,  # noqa: F401
    _embed_documents_batched,  # noqa: F401
    add_documents,  # noqa: F401
    replace_documents,  # noqa: F401
    delete_by_source,  # noqa: F401
    delete_by_file_id,  # noqa: F401
    count,  # noqa: F401
)
from app.core.retrieval.context import (  # noqa: F401
    _annotate_primary,  # noqa: F401
    chapter_local_refine,  # noqa: F401
    expand_novel_context,  # noqa: F401
)

log = get_logger("rag")


def _keep_candidate(scores: Iterable[float | None], threshold: float, *, hybrid: bool) -> bool:
    """判断候选是否进入后续阶段。

    混合检索只要求候选来自至少一个通道，不再用原始相似度阈值二次截断；
    纯向量检索仍沿用相似度阈值，避免改变既有单通道语义。
    """
    available = [score for score in scores if score is not None]
    if not available:
        return False
    return hybrid or max(available) >= threshold


async def _check_index_compatibility(file_id: str | None, owner: str) -> None:
    """检查当前 Embedding 配置与文件索引版本是否兼容。"""
    if not file_id:
        return
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(KnowledgeFile).where(KnowledgeFile.id == file_id, KnowledgeFile.user_id == owner)
        )
        record = result.scalars().first()
    if not record:
        return
    mismatches = []
    if record.embedding_model and record.embedding_model != settings.embedding_model:
        mismatches.append(f"embedding_model={record.embedding_model}")
    if record.embed_dim and record.embed_dim != settings.embed_dim:
        mismatches.append(f"embed_dim={record.embed_dim}")
    if record.chunk_size and record.chunk_size != settings.novel_chunk_size:
        mismatches.append(f"chunk_size={record.chunk_size}")
    if record.chunk_overlap is not None and record.chunk_overlap != settings.novel_chunk_overlap:
        mismatches.append(f"chunk_overlap={record.chunk_overlap}")
    if mismatches:
        raise RuntimeError(
            "索引版本与当前检索配置不兼容（" + ", ".join(mismatches) + "），请重新索引该小说。"
        )


def _trace_candidates(docs: list[Document] | list, stage: str) -> list[dict]:
    """把检索阶段候选压缩为可序列化诊断信息，不携带正文全文。"""
    result = []
    for rank, item in enumerate(docs, start=1):
        if isinstance(item, Document):
            meta = item.metadata
        else:
            meta = _parse_meta(item[0]) if isinstance(item, tuple) else _parse_meta(item)
            if isinstance(item, tuple) and len(item) > 1:
                meta["channel_score"] = float(item[1])
        result.append({
            "stage": stage,
            "id": meta.get("id"),
            "file_id": meta.get("file_id"),
            "chapter_no": meta.get("chapter_no"),
            "chunk_no": meta.get("chunk_no"),
            "rank": rank,
            "score": meta.get("score", meta.get("channel_score")),
            "vector_score": meta.get("vector_score"),
            "fts_score": meta.get("fts_score"),
            "rrf_score": meta.get("rrf_score"),
            "reranker_score": meta.get("reranker_score"),
            "reranker_rank": meta.get("reranker_rank"),
            "reranker_protected": bool(meta.get("reranker_protected", False)),
            "final_score": meta.get("final_score"),
            "matched_queries": meta.get("matched_queries"),
            "query_ranks": meta.get("query_ranks"),
            "query_match_count": meta.get("query_match_count"),
            "multi_query_score": meta.get("multi_query_score"),
        })
    return result


async def _similarity_search_once(
    query: str, k: int = None, filter_source: str = None,
    domain: str = "novel", file_id: str = None, trace: dict | None = None,
    rerank_query: str | None = None,
) -> List[Document]:
    """Hybrid vector/lexical retrieval with optional cross-encoder reranking.

    ``trace`` 为可选的内部诊断容器；不传时保持原有返回格式和开销。
    """
    k = k or settings.top_k
    metrics.incr("retrieval_calls")
    started = time.perf_counter()
    owner = get_current_user()
    if trace is not None:
        trace.clear()
        trace.update({
            "query": query,
            "file_id": file_id,
            "vector_candidates": [],
            "lexical_candidates": [],
            "rrf_candidates": [],
            "reranker_candidates": [],
            "reranker_ranked_candidates": [],
            "final_results": [],
            "filtered_counts": {},
            "reranker_enabled": bool(settings.enable_reranker),
            "reranker_failed": False,
            "fallback_reason": None,
            "phase_timings_ms": {},
        })
    await _check_index_compatibility(file_id, owner)
    embedding_started = time.perf_counter()
    qvec = await asyncio.to_thread(get_embeddings().embed_query, query)
    if trace is not None:
        trace["phase_timings_ms"]["embedding"] = round((time.perf_counter() - embedding_started) * 1000, 2)
    # 没有 Reranker 时也保留完整混合候选池，最后再截取 Top-K；这样后续可在
    # 已召回候选内做去重、章节聚合或边界处理，而无需再次执行主检索。
    pool_n = (
        max(k, settings.reranker_candidate_n)
        if settings.enable_reranker
        else max(k, settings.hybrid_candidate_k) if settings.enable_hybrid_search else k
    )
    candidate_threshold = (
        settings.reranker_candidate_threshold if settings.enable_reranker
        else settings.similarity_threshold
    )

    async with AsyncSessionLocal() as session:
        if settings.enable_hybrid_search:
            candidate_k = max(settings.hybrid_candidate_k, pool_n)
            vector_started = time.perf_counter()
            vector_results = await _vector_search(
                session, qvec, candidate_k, filter_source, owner, domain, file_id
            )
            if trace is not None:
                trace["vector_candidates"] = _trace_candidates(vector_results, "vector")
                trace["phase_timings_ms"]["vector_search"] = round((time.perf_counter() - vector_started) * 1000, 2)
            lexical_started = time.perf_counter()
            if settings.enable_bm25_search and _CJK_RE.search(query):
                # fail-fast：BM25 开启即视为该环境已具备 pg_search 扩展 + bm25 索引
                # （见迁移 20260828_0012）。异常直接抛出，不静默降级——缺扩展/索引是
                # 必须修复的配置错误，静默回退会让词法通道悄悄退化且无人察觉。
                lexical_results = await _bm25_search(
                    session, query, candidate_k, filter_source, owner, domain, file_id
                )
            else:
                try:
                    if settings.enable_chinese_lexical_search and _CJK_RE.search(query):
                        lexical_results = await _chinese_lexical_search(
                            session, query, candidate_k, filter_source, owner, domain, file_id
                        )
                    else:
                        lexical_results = await _fts_search(
                            session, query, candidate_k, filter_source, owner, domain, file_id
                        )
                except Exception as exc:  # pg_trgm/permissions unavailable
                    await session.rollback()
                    log.warning("lexical_search.fallback", error=str(exc))
                    lexical_results = await _fts_search(
                        session, query, candidate_k, filter_source, owner, domain, file_id
                    )

            if trace is not None:
                trace["lexical_candidates"] = _trace_candidates(lexical_results, "lexical")
                trace["phase_timings_ms"]["lexical_search"] = round((time.perf_counter() - lexical_started) * 1000, 2)
            # 评测追踪保留完整候选池，实际送入 Reranker 的数量仍由 pool_n 控制。
            rrf_started = time.perf_counter()
            # RRF 分数始终计算：作为 metadata 保留（rerank 保护逻辑依赖），
            # 并供 trace 的损失归因使用；是否用它决定池内顺序由 fusion_mode 控制。
            rrf_fused_all = _rrf_fuse(vector_results, lexical_results, candidate_k, settings.rrf_k)
            rrf_scores = {row.id: score for row, score in rrf_fused_all}
            if settings.fusion_mode == "weighted":
                fused_all = _weighted_fuse(
                    vector_results, lexical_results, settings.vector_weight, settings.lexical_weight
                )
            else:
                fused_all = rrf_fused_all
            fused = fused_all[:pool_n]
            if trace is not None:
                trace["rrf_candidates"] = _trace_candidates(fused_all, "rrf")
                trace["phase_timings_ms"]["rrf_fusion"] = round((time.perf_counter() - rrf_started) * 1000, 2)
            vector_scores = {row.id: score for row, score in vector_results}
            lexical_scores = {row.id: score for row, score in lexical_results}
            filter_started = time.perf_counter()
            pool: list[Document] = []
            for row, fused_score in fused:
                vector_score = vector_scores.get(row.id)
                lexical_score = lexical_scores.get(row.id)
                available = [score for score in (vector_score, lexical_score) if score is not None]
                best_score = max(available) if available else 0.0
                # 混合召回已由两路 Top-N 和融合控制噪声，避免再次按原始分数截断。
                if not _keep_candidate(available, candidate_threshold, hybrid=True):
                    if trace is not None:
                        trace["filtered_counts"]["missing_channel_score"] = trace["filtered_counts"].get("missing_channel_score", 0) + 1
                    continue
                meta = _parse_meta(row)
                meta.update({
                    # weighted 模式下 score 语义统一为最终融合分；rrf 模式保持各通道原始
                    # 最高分（历史行为不变）。rerank 保护与降级排序依赖该字段的一致性。
                    "score": round(fused_score, 4) if settings.fusion_mode == "weighted" else round(best_score, 4),
                    "score_type": "hybrid" if vector_score is not None and lexical_score is not None else (
                        "vector" if vector_score is not None else "fts"
                    ),
                    "vector_score": round(vector_score, 4) if vector_score is not None else None,
                    "fts_score": round(lexical_score, 4) if lexical_score is not None else None,
                    "rrf_score": round(rrf_scores.get(row.id, 0.0), 6),
                    "fusion_mode": settings.fusion_mode,
                    "reranked": False,
                    "neighbor": False,
                })
                pool.append(Document(page_content=row.content, metadata=meta))
            if trace is not None:
                trace["phase_timings_ms"]["candidate_filter"] = round((time.perf_counter() - filter_started) * 1000, 2)
        else:
            raw = await _vector_search(session, qvec, pool_n, filter_source, owner, domain, file_id)
            pool = []
            if trace is not None:
                trace["vector_candidates"] = _trace_candidates(raw, "vector")
            for row, score in raw:
                if not _keep_candidate([score], candidate_threshold, hybrid=False):
                    if trace is not None:
                        trace["filtered_counts"]["candidate_threshold"] = trace["filtered_counts"].get("candidate_threshold", 0) + 1
                    continue
                meta = _parse_meta(row)
                meta.update({
                    "score": round(score, 4), "score_type": "vector",
                    "vector_score": round(score, 4), "fts_score": None,
                    "rrf_score": None, "reranked": False, "neighbor": False,
                })
                pool.append(Document(page_content=row.content, metadata=meta))

    # 重排只处理召回候选；失败时回退到当前候选顺序，不能阻断问答。
    if trace is not None:
        trace["reranker_candidates"] = _trace_candidates(pool, "reranker")
    reranker_started = time.perf_counter()
    if settings.enable_reranker and pool:
        try:
            from app.core.rerank import rerank
            # 保留原候选列表的引用，用于记录完整重排顺序；rerank 返回的仅是最终 Top-K。
            reranker_pool = list(pool)
            pool = await asyncio.to_thread(rerank, rerank_query or query, reranker_pool, k)
            if trace is not None:
                trace["reranker_ranked_candidates"] = _trace_candidates(
                    sorted(reranker_pool, key=lambda doc: int(doc.metadata.get("reranker_rank") or 10**9)),
                    "reranker_ranked",
                )
        except Exception as exc:  # noqa: BLE001
            # 用 error 级别：重排失败意味着本轮检索静默降级，是需要人介入的异常，
            # 而不是可忽略的噪声。同时保留完整错误信息，避免只看到异常类名。
            log.error("reranker.failed", error_type=type(exc).__name__, error=str(exc))
            if trace is not None:
                trace["reranker_failed"] = True
                trace["fallback_reason"] = type(exc).__name__
                trace["reranker_error"] = str(exc)
            pool = pool[:k]
    else:
        pool = pool[:k]

    final = _annotate_primary(pool)
    if trace is not None:
        trace["final_results"] = _trace_candidates(final, "final")
        trace["phase_timings_ms"]["reranker"] = round((time.perf_counter() - reranker_started) * 1000, 2)
        trace["phase_timings_ms"]["total"] = round((time.perf_counter() - started) * 1000, 2)
        trace["candidate_counts"] = {
            "vector": len(trace.get("vector_candidates", [])),
            "lexical": len(trace.get("lexical_candidates", [])),
            "rrf": len(trace.get("rrf_candidates", [])),
            "reranker": len(trace.get("reranker_candidates", [])),
            "reranker_ranked": len(trace.get("reranker_ranked_candidates", [])),
            "final": len(trace.get("final_results", [])),
        }
    metrics.record_latency("retrieval", (time.perf_counter() - started) * 1000)
    return final


def build_merged_retrieval_query(standalone_query: str, retrieval_query: str | None = None) -> str:
    """把自然问题和 RAG 检索线索合并成一次检索使用的 Query。

    多轮对话的原始输入可能包含无法独立理解的指代，因此这里使用
    ``standalone_query`` 作为语义锚点，不直接把 raw original_query 放入检索文本。
    """
    base = (standalone_query or "").strip()
    hint = (retrieval_query or "").strip()
    if not base:
        return hint
    if not hint or hint == base:
        return base
    # 直接拼接两部分文本，避免“问题/检索线索”等模板词进入中文词法召回，
    # 同时保留 standalone_query 的完整语义和模型提取的原文检索线索。
    return f"{base} {hint}"


async def similarity_search(
    query: str, k: int = None, filter_source: str = None,
    domain: str = "novel", file_id: str = None, trace: dict | None = None,
    retrieval_query: str | None = None,
) -> List[Document]:
    """使用合并后的单一 Query 执行一次共享 RAG 检索。"""
    merged_query = build_merged_retrieval_query(query, retrieval_query)
    if not merged_query:
        return []

    docs = await _similarity_search_once(
        merged_query,
        k,
        filter_source,
        domain,
        file_id,
        trace,
        # Reranker 面向自然问题，不被关键词式检索线索带偏。
        rerank_query=query,
    )
    if trace is not None:
        trace.update({
            "base_query": query,
            "retrieval_query": retrieval_query if retrieval_query and retrieval_query.strip() != query.strip() else None,
            "merged_query": merged_query,
            "query_count": 1,
            "retrieval_mode": "merged_single_query",
        })
    return docs


async def similarity_search_with_trace(
    query: str, k: int = None, filter_source: str = None,
    domain: str = "novel", file_id: str = None,
    retrieval_query: str | None = None,
) -> tuple[List[Document], dict]:
    """执行检索并返回阶段候选追踪，供离线评测和问题诊断使用。"""
    trace: dict = {}
    docs = await similarity_search(
        query, k, filter_source, domain, file_id, trace=trace, retrieval_query=retrieval_query
    )
    return docs, trace


async def retrieve_novel_context(
    query: str, k: int = None, neighbor_window: int | None = None, file_id: str = None,
    retrieval_query: str | None = None, user_id: str | None = None,
) -> List[Document]:
    """执行一次主检索，按需做章节内二级精排，最后补充邻居片段。

    顺序是刻意的：二级精排必须在邻居扩展**之前**完成，否则邻居片段会被二次
    检索的顺序打乱，评测也无法区分“主检索命中”与“上下文扩展”。
    """
    k = k or settings.novel_context_k
    primary = await similarity_search(
        query, k=k, domain="novel", file_id=file_id, retrieval_query=retrieval_query
    )
    if settings.enable_chapter_local_retrieval and file_id:
        owner = user_id or get_current_user()
        primary = await chapter_local_refine(query, primary, k, file_id, owner)
    return await expand_novel_context(primary, neighbor_window, user_id)
