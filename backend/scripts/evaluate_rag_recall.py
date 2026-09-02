# -*- coding: utf-8 -*-
"""Evaluate the current novel RAG retrieval chain without invoking Agent synthesis."""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Allow running as `python backend/scripts/evaluate_rag_recall.py` from the repo root.
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings  # noqa: E402
from app.core.context import reset_current_user, set_current_user  # noqa: E402
from app.core.rag import (  # noqa: E402
    chapter_local_refine,
    expand_novel_context,
    similarity_search_with_trace,
)
from app.core.query_rewriter import rewrite_query  # noqa: E402
from app.db import AsyncSessionLocal  # noqa: E402
from app.db.models import KnowledgeFile  # noqa: E402
from sqlalchemy import select  # noqa: E402

DEFAULT_DATASET = Path(__file__).resolve().parents[2] / "evals" / "datasets" / "xiyouji_recall.jsonl"
DEFAULT_REPORT_DIR = Path(__file__).resolve().parents[2] / "evals" / "reports"


def load_cases(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 评测题，并拒绝缺少金标准定位的记录。"""
    cases: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not item.get("id") or not item.get("query") or not item.get("gold_chunks"):
            raise ValueError(f"评测题缺少 id/query/gold_chunks：第 {line_no} 行")
        cases.append(item)
    if not cases:
        raise ValueError(f"评测集为空：{path}")
    return cases


def load_preparation_cache(path: Path) -> dict[str, Any]:
    """读取可复用的 Query Preparation 快照，供参数对照实验冻结 LLM 输出。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        raise ValueError(f"Query Preparation 缓存格式错误：{path}")
    cases: dict[str, dict[str, Any]] = {}
    for item in payload["cases"]:
        if not isinstance(item, dict) or not item.get("id"):
            raise ValueError(f"Query Preparation 缓存缺少 id：{path}")
        case_id = str(item["id"])
        if case_id in cases:
            raise ValueError(f"Query Preparation 缓存包含重复 id：{case_id}")
        standalone = str(item.get("standalone_query") or "").strip()
        retrieval = str(item.get("retrieval_query") or "").strip()
        if not standalone or not retrieval:
            raise ValueError(f"Query Preparation 缓存缺少 Query：{case_id}")
        cases[case_id] = {
            "id": case_id,
            "query": str(item.get("query") or ""),
            "original": str(item.get("original") or item.get("query") or ""),
            "standalone_query": standalone,
            "retrieval_query": retrieval,
            "intent": item.get("intent", "other"),
            "entities": item.get("entities", []),
            "evidence_focus": item.get("evidence_focus", []),
            "confidence": item.get("confidence", 0.0),
            "applied": bool(item.get("applied", True)),
            "reason": item.get("reason", "cached"),
            "latency_ms": float(item.get("latency_ms", 0.0) or 0.0),
            "source": "cache",
        }
    payload["cases"] = cases
    return payload


def validate_preparation_cache(
    cache: dict[str, Any],
    cases: list[dict[str, Any]],
    *,
    file_id: str,
    index_metadata: dict[str, Any],
) -> None:
    """确保 Query 快照与本次数据集、文件和索引身份一致。"""
    cached_cases = cache.get("cases", {})
    expected_ids = {str(case["id"]) for case in cases}
    actual_ids = set(cached_cases)
    if expected_ids != actual_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise RuntimeError(f"Query Preparation 缓存题目不匹配：missing={missing}, extra={extra}")
    cached_file_id = cache.get("file_id")
    if cached_file_id and cached_file_id != file_id:
        raise RuntimeError(f"Query Preparation 缓存 file_id 不一致：cache={cached_file_id}, current={file_id}")
    cached_source_hash = cache.get("source_hash")
    if cached_source_hash and cached_source_hash != index_metadata.get("source_hash"):
        raise RuntimeError("Query Preparation 缓存 source_hash 与当前索引不一致")
    for case in cases:
        item = cached_cases[str(case["id"])]
        if item.get("query") and item["query"].strip() != str(case["query"]).strip():
            raise RuntimeError(f"Query Preparation 缓存原问题不一致：{case['id']}")


def write_preparation_cache(
    path: Path,
    cases: list[dict[str, Any]],
    rows: dict[str, dict[str, Any]],
    *,
    dataset: Path,
    file_id: str,
    index_metadata: dict[str, Any],
) -> None:
    """保存 Query Preparation 快照，后续实验只复用输出而不再次调用 LLM。"""
    payload = {
        "version": 1,
        "dataset": str(dataset.resolve()),
        "file_id": file_id,
        "source_hash": index_metadata.get("source_hash"),
        "index_version": index_metadata.get("index_version"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cases": [rows[str(case["id"])] for case in cases],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def source_key(source: dict[str, Any]) -> tuple[Any, Any, Any]:
    return source.get("file_id"), source.get("chapter_no"), source.get("chunk_no")


def gold_maps(case: dict[str, Any]) -> tuple[set[tuple[Any, Any, Any]], dict[tuple[Any, Any, Any], int]]:
    keys: set[tuple[Any, Any, Any]] = set()
    grades: dict[tuple[Any, Any, Any], int] = {}
    for item in case["gold_chunks"]:
        key = (case.get("file_id"), item.get("chapter_no"), item.get("chunk_no"))
        keys.add(key)
        grades[key] = max(0, int(item.get("relevance", 1)))
    return keys, grades


def is_gold(source: dict[str, Any], keys: set[tuple[Any, Any, Any]]) -> bool:
    return source_key(source) in keys and not bool(source.get("neighbor"))


def dcg(values: list[int]) -> float:
    return sum((2**value - 1) / math.log2(index + 2) for index, value in enumerate(values))


def ndcg(ranked: list[dict[str, Any]], grades: dict[tuple[Any, Any, Any], int], k: int) -> float:
    observed = [grades.get(source_key(item), 0) for item in ranked[:k]]
    ideal = sorted(grades.values(), reverse=True)[:k]
    ideal_score = dcg(ideal)
    return dcg(observed) / ideal_score if ideal_score else 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _bucket_query_length(query: str) -> str:
    length = len(query.strip())
    if length <= 10:
        return "short"
    if length <= 25:
        return "medium"
    return "long"


def case_features(case: dict[str, Any], index_metadata: dict[str, Any]) -> dict[str, Any]:
    """生成可审计的测试分层字段；缺失的人工标签明确记为 N/A。"""
    query = str(case.get("query", ""))
    chapters = case.get("gold_chapters") or []
    chapter_count = index_metadata.get("chapters")
    chapter_position = "N/A"
    if chapters and isinstance(chapter_count, int) and chapter_count > 0:
        chapter_no = min(int(item) for item in chapters)
        ratio = chapter_no / chapter_count
        chapter_position = "early" if ratio <= 1 / 3 else "middle" if ratio <= 2 / 3 else "late"
    entity_count = case.get("entity_count", "N/A")
    if isinstance(entity_count, int):
        entity_bucket = "none" if entity_count == 0 else "single" if entity_count == 1 else "multi"
    else:
        entity_bucket = "N/A"
    return {
        "query_length": len(query),
        "query_length_bucket": _bucket_query_length(query),
        "entity_count": entity_count,
        "entity_count_bucket": entity_bucket,
        "has_explicit_character": case.get("has_explicit_character", "N/A"),
        "has_alias_or_title": case.get("has_alias_or_title", "N/A"),
        "has_pronoun": case.get("has_pronoun", "N/A"),
        "has_chapter_hint": case.get("has_chapter_hint", "N/A"),
        "requires_cross_chapter": case.get("requires_cross_chapter", "N/A"),
        "requires_multiple_evidence": case.get("requires_multiple_evidence", "N/A"),
        "answer_evidence_span": case.get("answer_evidence_span", "N/A"),
        "chapter_position": chapter_position,
        "query_time_bucket": case.get("query_time_bucket", "N/A"),
    }


def _trace_timings(trace: dict[str, Any] | None) -> dict[str, float]:
    """汇总单查询或多 Query trace 的阶段耗时。"""
    if not trace:
        return {}
    traces = trace.get("variant_traces")
    sources = traces if isinstance(traces, list) else [trace]
    result: dict[str, float] = defaultdict(float)
    for source in sources:
        for name, value in (source.get("phase_timings_ms", {}) or {}).items():
            try:
                result[name] += float(value)
            except (TypeError, ValueError):
                continue
    return {key: round(value, 2) for key, value in result.items()}


def _stage_gold_info(trace: dict[str, Any] | None, gold: set[tuple[Any, Any, Any]]) -> dict[str, Any]:
    stages = ("vector_candidates", "lexical_candidates", "rrf_candidates", "reranker_candidates", "reranker_ranked_candidates", "final_results")
    info: dict[str, Any] = {}
    for stage in stages:
        candidates = trace_stage_candidates(trace or {}, stage)
        ranks: list[int] = []
        for index, item in enumerate(candidates, start=1):
            if (item.get("file_id"), item.get("chapter_no"), item.get("chunk_no")) in gold:
                ranks.append(index)
        info[stage] = {"hit": bool(ranks), "gold_ranks": ranks, "candidate_count": len(candidates)}
    if info["final_results"]["hit"]:
        loss_point = "none"
    elif info["reranker_ranked_candidates"]["hit"]:
        # 重排后的候选仍存在但最终结果没有命中，说明是最终 Top-K 截断。
        loss_point = "final_top_k_cut"
    elif info["reranker_candidates"]["hit"]:
        # 候选在重排前存在、重排后消失，说明排序阶段丢失。
        loss_point = "reranker_loss"
    elif info["rrf_candidates"]["hit"]:
        loss_point = "candidate_filter_loss"
    elif info["vector_candidates"]["hit"] or info["lexical_candidates"]["hit"]:
        loss_point = "rrf_fusion_loss"
    else:
        loss_point = "candidate_generation_missed"
    info["loss_point"] = loss_point
    return info


def _git_metadata() -> dict[str, Any]:
    """记录工作区版本信息；脏工作区不伪装成稳定提交。"""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=BACKEND_DIR.parent, capture_output=True, text=True, check=False).stdout.strip()
        status = subprocess.run(["git", "status", "--short"], cwd=BACKEND_DIR.parent, capture_output=True, text=True, check=False).stdout.strip()
        return {"head": head or "unknown", "dirty": bool(status), "status_lines": len(status.splitlines()) if status else 0}
    except OSError:
        return {"head": "unknown", "dirty": None, "status_lines": None}


async def validate_index(file_id: str, user_id: str, expected_source_hash: str | None = None) -> dict[str, Any]:
    """校验评测文件、租户、原文哈希和索引参数，避免误报召回率。"""
    async with AsyncSessionLocal() as session:
        record = (await session.execute(select(KnowledgeFile).where(KnowledgeFile.id == file_id, KnowledgeFile.user_id == user_id))).scalars().first()
    if record is None:
        raise RuntimeError(f"找不到评测索引或 user_id 不匹配：file_id={file_id}, user_id={user_id}")
    if record.status != "indexed":
        raise RuntimeError(f"评测文件尚未完成索引：status={record.status}")
    if expected_source_hash and record.source_hash != expected_source_hash:
        raise RuntimeError(
            f"原文哈希与索引不一致或索引缺少哈希：index={record.source_hash}, expected={expected_source_hash}"
        )
    mismatches = []
    for field, expected in (("embedding_model", settings.embedding_model), ("embed_dim", settings.embed_dim), ("chunk_size", settings.novel_chunk_size), ("chunk_overlap", settings.novel_chunk_overlap)):
        actual = getattr(record, field, None)
        if actual is not None and actual != expected:
            mismatches.append(f"{field}: index={actual}, runtime={expected}")
    if mismatches:
        raise RuntimeError("评测索引与当前运行配置不兼容：" + "; ".join(mismatches))
    return {"file_id": record.id, "user_id": record.user_id, "source_hash": record.source_hash, "embedding_model": record.embedding_model, "embed_dim": record.embed_dim, "chunk_size": record.chunk_size, "chunk_overlap": record.chunk_overlap, "index_version": record.index_version, "chunks": record.chunks, "chapters": record.chapter_count}


def trace_stage_candidates(trace: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    """兼容单查询与双查询 trace，返回某阶段按块去重后的候选并集。"""
    variant_traces = trace.get("variant_traces")
    sources: list[dict[str, Any]] = [trace]
    if isinstance(variant_traces, list):
        sources.extend(item for item in variant_traces if isinstance(item, dict))
    merged: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    for source in sources:
        candidates = source.get(stage, [])
        if not isinstance(candidates, list):
            continue
        for rank, item in enumerate(candidates, start=1):
            if not isinstance(item, dict):
                continue
            key = (item.get("file_id"), item.get("chapter_no"), item.get("chunk_no"))
            previous = merged.get(key)
            if previous is None or rank < previous[0]:
                merged[key] = (rank, item)
    return [item for _, item in sorted(merged.values(), key=lambda pair: pair[0])]


def metrics_for_case(
    case: dict[str, Any],
    primary: list[dict[str, Any]],
    expanded: list[dict[str, Any]],
    latency_ms: float,
    trace: dict[str, Any] | None = None,
    index_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """计算单题指标，并保留阶段级证据以便定位召回损失。"""
    gold, grades = gold_maps(case)
    ranked = [item for item in primary if not item.get("neighbor")]
    stage_diagnostics = _stage_gold_info(trace, gold) if trace is not None else {}
    result: dict[str, Any] = {
        "id": case["id"],
        "category": case.get("category", "unknown"),
        "query": case["query"],
        "gold_chunks": case["gold_chunks"],
        "gold_chapters": case.get("gold_chapters", []),
        "features": case_features(case, index_metadata or {}),
        "latency_ms": round(latency_ms, 2),
        "top_k": [],
        "context_top_k": [],
        "stage_diagnostics": stage_diagnostics,
    }
    for item in ranked[:10]:
        result["top_k"].append({
            "file_id": item.get("file_id"),
            "chapter": item.get("chapter"),
            "chapter_no": item.get("chapter_no"),
            "chunk_no": item.get("chunk_no"),
            "retrieval_rank": item.get("retrieval_rank"),
            "score_type": item.get("score_type"),
            "score": item.get("score"),
            "vector_score": item.get("vector_score"),
            "fts_score": item.get("fts_score"),
            "rrf_score": item.get("rrf_score"),
            "reranker_score": item.get("reranker_score"),
            "reranker_rank": item.get("reranker_rank"),
            "reranker_protected": bool(item.get("reranker_protected", False)),
            "final_score": item.get("final_score"),
            "reranked": item.get("reranked", False),
            "matched_queries": item.get("matched_queries"),
            "query_ranks": item.get("query_ranks"),
            "query_match_count": item.get("query_match_count"),
            "neighbor": item.get("neighbor", False),
            "snippet": item.get("snippet", ""),
            "is_gold": is_gold(item, gold),
            "relevance": grades.get(source_key(item), 0),
        })
    for item in expanded:
        result["context_top_k"].append({
            "chapter_no": item.get("chapter_no"),
            "chunk_no": item.get("chunk_no"),
            "neighbor": bool(item.get("neighbor")),
            "is_gold": is_gold(item, gold),
        })
    for k in (1, 5, 10):
        hits = sum(is_gold(item, gold) for item in ranked[:k])
        result[f"recall_at_{k}"] = round(hits / len(gold), 4) if gold else 0.0
        result[f"precision_at_{k}"] = round(hits / k, 4)
        result[f"hit_at_{k}"] = bool(hits)
    first = next((index for index, item in enumerate(ranked[:10], start=1) if is_gold(item, gold)), None)
    result["mrr_at_10"] = round(1 / first, 4) if first else 0.0
    result["ndcg_at_10"] = round(ndcg(ranked, grades, 10), 4)
    result["neighbor_gold_hits"] = sum(is_gold(item, gold) for item in expanded if item.get("neighbor"))
    result["context_gold_hits"] = sum(is_gold(item, gold) for item in expanded)
    result["chapter_hit"] = bool(set(case.get("gold_chapters", [])) & {item.get("chapter_no") for item in ranked[:10]})
    result["chunk_hit"] = bool(any(is_gold(item, gold) for item in ranked[:10]))
    result["failure_reason"] = failure_reason(case, ranked, gold, trace)
    if result["failure_reason"] and result["chapter_hit"] and result["context_gold_hits"] > 0:
        # gold 在邻居上下文而不在主 Top-K 时，优先标记为分块边界问题，
        # 不把它误判为“完全没有召回”。
        result["failure_reason"] = "gold_boundary_mismatch"
    if trace is not None:
        for stage, key in (
            ("vector_candidates", "candidate_recall_vector"),
            ("lexical_candidates", "candidate_recall_lexical"),
            ("rrf_candidates", "candidate_recall_rrf"),
            ("reranker_candidates", "pre_rerank_recall"),
        ):
            candidates = trace_stage_candidates(trace, stage)
            result[key] = round(
                sum((item.get("file_id"), item.get("chapter_no"), item.get("chunk_no")) in gold for item in candidates) / len(gold),
                4,
            ) if gold else 0.0
        result["phase_timings_ms"] = _trace_timings(trace)
        result["candidate_counts"] = trace.get("candidate_counts", {})
        result["trace"] = trace
    return result


def failure_reason(
    case: dict[str, Any],
    ranked: list[dict[str, Any]],
    gold: set[tuple[Any, Any, Any]],
    trace: dict[str, Any] | None = None,
) -> str | None:
    """依据阶段 trace 归因；无法确认时使用明确的未知分类。"""
    if any(is_gold(item, gold) for item in ranked[:10]):
        return None
    if not ranked:
        return "empty_retrieval"
    expected_file = case.get("file_id")
    if any(item.get("file_id") not in (None, expected_file) for item in ranked):
        return "file_filter_mismatch"
    stage = _stage_gold_info(trace, gold) if trace is not None else {}
    expected_chapters = set(case.get("gold_chapters", []))
    if expected_chapters and any(item.get("chapter_no") in expected_chapters for item in ranked[:10]):
        return "chapter_hit_but_gold_chunk_missed"
    if stage.get("reranker_ranked_candidates", {}).get("hit"):
        return "final_top_k_cut"
    if stage.get("reranker_candidates", {}).get("hit"):
        # 关闭重排时 "reranker_candidates" 只是截断前的候选池（rag.py 无条件写入该
        # trace），金标在其中却没进 Top-K 与重排无关，纯属 Top-K 截断。此前一律归因
        # 为 reranker_loss，会在关闭重排的评测里凭空造出一条误导性的失败类别。
        return "final_top_k_cut" if not settings.enable_reranker else "reranker_loss"
    if stage.get("rrf_candidates", {}).get("hit"):
        return "candidate_filter_loss"
    if stage.get("vector_candidates", {}).get("hit") or stage.get("lexical_candidates", {}).get("hit"):
        return "rrf_fusion_loss"
    if trace is not None:
        return "candidate_generation_missed"
    if any(item.get("score_type") == "reranker" for item in ranked):
        return "not_in_final_top_k_after_rerank"
    return "not_observed_in_final_top_k"


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(name: str, rows: list[dict[str, Any]] = results) -> float:
        return round(statistics.mean([float(row.get(name, 0.0)) for row in rows]), 4) if rows else 0.0

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "questions": len(rows),
            "recall_at_1": mean("recall_at_1", rows),
            "recall_at_5": mean("recall_at_5", rows),
            "recall_at_10": mean("recall_at_10", rows),
            "candidate_recall_vector": mean("candidate_recall_vector", rows),
            "candidate_recall_lexical": mean("candidate_recall_lexical", rows),
            "candidate_recall_rrf": mean("candidate_recall_rrf", rows),
            "pre_rerank_recall": mean("pre_rerank_recall", rows),
            "precision_at_5": mean("precision_at_5", rows),
            "mrr_at_10": mean("mrr_at_10", rows),
            "ndcg_at_10": mean("ndcg_at_10", rows),
            "chapter_hit_rate": round(sum(row["chapter_hit"] for row in rows) / len(rows), 4) if rows else 0.0,
            "chunk_hit_rate": round(sum(row["chunk_hit"] for row in rows) / len(rows), 4) if rows else 0.0,
            "neighbor_context_hit_rate": round(sum(bool(row["context_gold_hits"]) for row in rows) / len(rows), 4) if rows else 0.0,
            "empty_rate": round(sum(not row["top_k"] for row in rows) / len(rows), 4) if rows else 0.0,
            "avg_latency_ms": mean("latency_ms", rows),
            "p50_latency_ms": round(percentile([row["latency_ms"] for row in rows], 0.50), 2) if rows else 0.0,
            "p95_latency_ms": round(percentile([row["latency_ms"] for row in rows], 0.95), 2) if rows else 0.0,
        }

    def group_by(feature: str) -> dict[str, dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in results:
            value = row.get("features", {}).get(feature, "N/A")
            if isinstance(value, bool):
                value = str(value).lower()
            grouped[str(value)].append(row)
        return {key: summarize(rows) for key, rows in sorted(grouped.items())}

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[row["category"]].append(row)
    phase_names = ("embedding", "vector_search", "lexical_search", "rrf_fusion", "candidate_filter", "reranker", "total")
    phase_timings = {
        name: {
            "avg_ms": round(statistics.mean(float(row.get("phase_timings_ms", {}).get(name, 0.0)) for row in results), 2) if results else 0.0,
            "p95_ms": round(percentile([float(row.get("phase_timings_ms", {}).get(name, 0.0)) for row in results], 0.95), 2) if results else 0.0,
        }
        for name in phase_names
    }
    return {
        "overall": summarize(results),
        "phase_timings_ms": phase_timings,
        "by_category": {category: summarize(rows) for category, rows in sorted(grouped.items())},
        "by_feature": {
            feature: group_by(feature)
            for feature in (
                "chapter_position",
                "query_length_bucket",
                "entity_count_bucket",
                "has_explicit_character",
                "has_alias_or_title",
                "has_pronoun",
                "has_chapter_hint",
                "requires_cross_chapter",
                "requires_multiple_evidence",
                "answer_evidence_span",
                "query_time_bucket",
            )
        },
        "failure_reasons": {
            reason: sum(row.get("failure_reason") == reason for row in results)
            for reason in sorted({row.get("failure_reason") for row in results if row.get("failure_reason")})
        },
        "loss_points": {
            point: sum(row.get("stage_diagnostics", {}).get("loss_point") == point for row in results)
            for point in sorted({row.get("stage_diagnostics", {}).get("loss_point") for row in results if row.get("stage_diagnostics", {}).get("loss_point")})
        },
    }


def dataset_profile(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总评测集覆盖范围与缺失字段，不对缺失标签做推断。"""
    feature_fields = (
        "entity_count", "has_explicit_character", "has_alias_or_title",
        "has_pronoun", "has_chapter_hint", "requires_cross_chapter",
        "requires_multiple_evidence", "answer_evidence_span", "query_time_bucket",
    )
    return {
        "questions": len(cases),
        "categories": {
            str(category): sum(item.get("category") == category for item in cases)
            for category in sorted({item.get("category", "unknown") for item in cases})
        },
        "file_ids": {
            str(file_id): sum(item.get("file_id") == file_id for item in cases)
            for file_id in sorted({item.get("file_id") for item in cases})
        },
        "gold_chunk_count": sum(len(item.get("gold_chunks", [])) for item in cases),
        "single_gold_chunk_questions": sum(len(item.get("gold_chunks", [])) == 1 for item in cases),
        "missing_feature_fields": {
            field: sum(field not in item for item in cases)
            for field in feature_fields
        },
        "time_bucket_available": any(item.get("query_time_bucket") not in (None, "") for item in cases),
    }


def load_baseline_report(path: str | None) -> dict[str, Any] | None:
    """读取可选历史报告，用于同口径指标差异对比。"""
    if not path:
        return None
    report_path = Path(path)
    if not report_path.is_file():
        return {"path": str(report_path), "error": "not_found"}
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
        return {
            "path": str(report_path.resolve()),
            "generated_at": data.get("generated_at"),
            "config": data.get("config", {}),
            "metrics": data.get("metrics", {}).get("overall", {}),
        }
    except (OSError, json.JSONDecodeError) as exc:
        return {"path": str(report_path), "error": type(exc).__name__}


def compare_baseline(current: dict[str, Any], baseline: dict[str, Any] | None) -> dict[str, Any] | None:
    if not baseline or baseline.get("error"):
        return baseline
    previous = baseline.get("metrics", {})
    keys = (
        "recall_at_1", "recall_at_5", "recall_at_10", "candidate_recall_vector",
        "candidate_recall_lexical", "candidate_recall_rrf", "pre_rerank_recall",
        "mrr_at_10", "ndcg_at_10", "chapter_hit_rate", "chunk_hit_rate",
        "avg_latency_ms", "p95_latency_ms", "empty_rate",
    )
    delta = {}
    for key in keys:
        if key in current and key in previous:
            delta[key] = round(float(current[key]) - float(previous[key]), 4)
    return {**baseline, "delta": delta}


def config_snapshot() -> dict[str, Any]:
    return {
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "embed_dim": settings.embed_dim,
        "embedding_check_ctx_length": settings.embedding_check_ctx_length,
        "novel_chunk_size": settings.novel_chunk_size,
        "novel_chunk_overlap": settings.novel_chunk_overlap,
        "novel_context_k": settings.novel_context_k,
        "novel_neighbor_window": settings.novel_neighbor_window,
        "hybrid_candidate_k": settings.hybrid_candidate_k,
        "enable_hybrid_search": settings.enable_hybrid_search,
        "fusion_mode": settings.fusion_mode,
        "vector_weight": settings.vector_weight,
        "lexical_weight": settings.lexical_weight,
        "enable_chinese_lexical_search": settings.enable_chinese_lexical_search,
        "enable_bm25_search": settings.enable_bm25_search,
        "enable_reranker": settings.enable_reranker,
        "reranker_model": settings.reranker_model,
        "reranker_candidate_n": settings.reranker_candidate_n,
        "similarity_threshold": settings.similarity_threshold,
        "reranker_candidate_threshold": settings.reranker_candidate_threshold,
        "enable_reranker_blend": settings.enable_reranker_blend,
        "reranker_weight": settings.reranker_weight,
        "rrf_weight": settings.rrf_weight,
        "raw_score_weight": settings.raw_score_weight,
        "reranker_protect_top_n": settings.reranker_protect_top_n,
        "reranker_protect_slots": settings.reranker_protect_slots,
        "enable_chapter_local_retrieval": settings.enable_chapter_local_retrieval,
        "enable_query_rewrite": settings.enable_query_rewrite,
        "query_rewrite_prompt_version": settings.query_rewrite_prompt_version,
        "query_rewrite_max_chars": settings.query_rewrite_max_chars,
    }


def _dataset_label(payload: dict[str, Any]) -> str:
    """从数据集元数据获取作品名，避免报告标题固定成某一本小说。"""
    profile = payload.get("dataset_profile", {})
    dataset = str(payload.get("dataset", ""))
    if "hongloumeng" in dataset.lower():
        return "红楼梦"
    if "xiyouji" in dataset.lower():
        return "西游记"
    return str(profile.get("works") or "小说")


def _metric_explanation(payload: dict[str, Any]) -> list[str]:
    """生成与 aggregate/metrics_for_case 实现一致的可读计算说明。"""
    overall = payload["metrics"]["overall"]
    cases = payload.get("cases", [])
    questions = int(overall.get("questions") or len(cases))
    hit_counts = {
        "recall_at_1": sum(float(row.get("recall_at_1", 0)) > 0 for row in cases),
        "recall_at_5": sum(float(row.get("recall_at_5", 0)) > 0 for row in cases),
        "recall_at_10": sum(float(row.get("recall_at_10", 0)) > 0 for row in cases),
        "chapter": sum(bool(row.get("chapter_hit")) for row in cases),
        "chunk": sum(bool(row.get("chunk_hit")) for row in cases),
        "context": sum(bool(row.get("context_gold_hits")) for row in cases),
        "empty": sum(not row.get("top_k") for row in cases),
    }
    candidate_hits = {}
    for key in (
        "candidate_recall_vector",
        "candidate_recall_lexical",
        "candidate_recall_rrf",
        "pre_rerank_recall",
    ):
        candidate_hits[key] = sum(float(row.get(key, 0)) > 0 for row in cases)
    lines = [
        "## 指标如何计算",
        "",
        f"本报告评测 {questions} 道题；每题的金标片段数可能不同，脚本先逐题计算，再对题目取算术平均。主结果只统计 `neighbor=false` 的片段。",
        "",
        "| 指标 | 计算方法 | 本次代入结果 |",
        "|---|---|---:|",
        f"| Recall@1 | 每题前 1 名命中的金标数 ÷ 该题金标数，再对题目平均 | {hit_counts['recall_at_1']}/{questions} 题命中；总体 {overall['recall_at_1']} |",
        f"| Recall@5 | 每题前 5 名命中的金标数 ÷ 该题金标数，再对题目平均 | {hit_counts['recall_at_5']}/{questions} 题命中；总体 {overall['recall_at_5']} |",
        f"| Recall@10 | 每题前 10 名命中的金标数 ÷ 该题金标数，再对题目平均 | {hit_counts['recall_at_10']}/{questions} 题命中；总体 {overall['recall_at_10']} |",
        f"| Precision@5 | 每题前 5 名命中的金标数 ÷ 5，再对题目平均 | 总体 {overall['precision_at_5']} |",
        f"| MRR@10 | 每题取第一个金标排名的倒数，前 10 名未命中记 0，再平均 | 总体 {overall['mrr_at_10']} |",
        f"| nDCG@10 | 按 gold `relevance` 计算前 10 名 DCG，再除以理想 DCG，最后平均 | 总体 {overall['ndcg_at_10']} |",
        f"| 章节命中率 | 前 10 名出现目标章节的题数 ÷ {questions} | {hit_counts['chapter']}/{questions} = {overall['chapter_hit_rate']} |",
        f"| 片段命中率 | 前 10 名出现目标片段的题数 ÷ {questions} | {hit_counts['chunk']}/{questions} = {overall['chunk_hit_rate']} |",
        f"| 邻居上下文覆盖率 | 扩展上下文出现任一金标的题数 ÷ {questions}；不等于主召回 | {hit_counts['context']}/{questions} = {overall['neighbor_context_hit_rate']} |",
        f"| 无结果率 | 主结果为空的题数 ÷ {questions} | {hit_counts['empty']}/{questions} = {overall['empty_rate']} |",
        "",
        "### 候选阶段",
        "",
        "候选阶段按 `(file_id, chapter_no, chunk_no)` 去重；某阶段包含金标即算该题命中，用于定位损失发生在哪一步，不代表最终 Top-K 命中。",
        "",
        "| 阶段 | 含义 | 命中题数 / 总题数 | 总体值 |",
        "|---|---|---:|---:|",
        f"| 向量候选 | Embedding 向量检索候选池包含金标 | {candidate_hits['candidate_recall_vector']}/{questions} | {overall['candidate_recall_vector']} |",
        f"| 词法候选 | BM25/中文词法候选池包含金标 | {candidate_hits['candidate_recall_lexical']}/{questions} | {overall['candidate_recall_lexical']} |",
        f"| RRF 候选 | 向量与词法融合、过滤后候选池包含金标 | {candidate_hits['candidate_recall_rrf']}/{questions} | {overall['candidate_recall_rrf']} |",
        f"| 重排前候选 | 送入 Reranker 前候选池包含金标 | {candidate_hits['pre_rerank_recall']}/{questions} | {overall['pre_rerank_recall']} |",
        "",
        "### 延迟与失败归因",
        "",
        f"端到端平均/P50/P95 分别是 {overall['avg_latency_ms']} / {overall['p50_latency_ms']} / {overall['p95_latency_ms']} ms；P50/P95 对每题端到端耗时排序后按脚本的线性插值 percentile 计算。阶段耗时来自检索 trace，不包含 Query Rewrite 时会另行说明。",
        "",
        "`failure_reasons` 是面向题目的最终归因；`loss_points` 是基于各阶段 trace 的机器归因。两者可能不同，因为章节命中但金标片段未命中时，最终归因会覆盖为更易读的章节级失败。",
        "",
        "每题的原问题、金标、Top-K、分数、是否金标、阶段 gold rank、阶段耗时和失败原因保存在同名 `_cases.jsonl`，完整 `trace` 保存在同名 JSON 的 `cases` 数组中。",
        "",
    ]
    return lines


def markdown_report(payload: dict[str, Any]) -> str:
    overall = payload["metrics"]["overall"]
    work = _dataset_label(payload)
    title = f"# {work} RAG Query Rewriter 召回报告" if payload.get("use_rewrite") else f"# {work} RAG 召回基线报告"
    lines = [

        title,
        "",
        f"生成时间：{payload['generated_at']}",
        f"评测文件：`{payload['file_id']}`",
        f"评测题数：{overall['questions']}",
        f"Query Rewriter：{'开启' if payload.get('use_rewrite') else '关闭'}",
        "",
        "## 总体指标",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
    ]
    for key, label in (("recall_at_1", "Recall@1"), ("recall_at_5", "Recall@5"), ("recall_at_10", "Recall@10"), ("candidate_recall_vector", "向量候选 Recall"), ("candidate_recall_lexical", "词法候选 Recall"), ("candidate_recall_rrf", "RRF 候选 Recall"), ("pre_rerank_recall", "重排前 Recall"), ("precision_at_5", "Precision@5"), ("mrr_at_10", "MRR@10"), ("ndcg_at_10", "nDCG@10"), ("chapter_hit_rate", "章节命中率"), ("chunk_hit_rate", "片段命中率"), ("neighbor_context_hit_rate", "邻居上下文覆盖率"), ("avg_latency_ms", "平均延迟 ms"), ("p95_latency_ms", "P95 延迟 ms"), ("empty_rate", "无结果率")):
        lines.append(f"| {label} | {overall[key]} |")
    lines += ["", *_metric_explanation(payload)]
    baseline = payload.get("baseline_comparison")
    if baseline:
        lines += ["", "## 与历史基线对比", ""]
        if baseline.get("error"):
            lines.append(f"基线报告不可用：{baseline.get('path')} / {baseline.get('error')}")
        else:
            lines.append(f"基线报告：`{baseline.get('path')}`")
            lines.append("")
            lines.append("| 指标 | 当前 | 基线 | 差值 |")
            lines.append("|---|---:|---:|---:|")
            for key in ("recall_at_10", "candidate_recall_rrf", "mrr_at_10", "chapter_hit_rate", "chunk_hit_rate", "p95_latency_ms"):
                if key in overall and key in baseline.get("metrics", {}):
                    lines.append(f"| {key} | {overall[key]} | {baseline['metrics'][key]} | {baseline.get('delta', {}).get(key)} |")
    rewrite_summary = payload.get("rewrite_summary")
    if rewrite_summary:
        lines += [
            "",
            "## Query Rewriter",
            "",
            f"- 实际改写：{rewrite_summary['applied']}/{rewrite_summary['questions']}（{rewrite_summary['applied_rate']}）",
            f"- 平均改写延迟：{rewrite_summary['avg_latency_ms']} ms",
            f"- 结果原因：`{json.dumps(rewrite_summary['reasons'], ensure_ascii=False)}`",
        ]
    lines += ["", "## 分类型指标", "", "| 类型 | 题数 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | 片段命中率 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for category, values in payload["metrics"]["by_category"].items():
        lines.append(f"| {category} | {values['questions']} | {values['recall_at_5']} | {values['recall_at_10']} | {values['mrr_at_10']} | {values['ndcg_at_10']} | {values['chunk_hit_rate']} |")
    lines += ["", "## 特征/场景分层", ""]
    for feature, groups in payload["metrics"].get("by_feature", {}).items():
        lines.append(f"### {feature}")
        lines.append("")
        lines.append("| 分组 | 题数 | Recall@10 | 章节命中率 | 片段命中率 | P95 ms |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for group, values in groups.items():
            lines.append(f"| {group} | {values['questions']} | {values['recall_at_10']} | {values['chapter_hit_rate']} | {values['chunk_hit_rate']} | {values['p95_latency_ms']} |")
        lines.append("")
    lines += ["## 阶段耗时", "", "| 阶段 | 平均 ms | P95 ms |", "|---|---:|---:|"]
    for phase, values in payload["metrics"].get("phase_timings_ms", {}).items():
        lines.append(f"| {phase} | {values['avg_ms']} | {values['p95_ms']} |")
    lines += ["", "## 失败原因与损失环节", "", "```json", json.dumps({"failure_reasons": payload["metrics"].get("failure_reasons", {}), "loss_points": payload["metrics"].get("loss_points", {})}, ensure_ascii=False, indent=2), "```", "", "## 检索配置", "", "```json", json.dumps(payload["config"], ensure_ascii=False, indent=2), "```", "", "## 未命中题目", ""]
    misses = [row for row in payload["cases"] if row.get("failure_reason")]
    if not misses:
        lines.append("无未命中题目。")
    else:
        for row in misses:
            lines.append(f"- `{row['id']}` {row['query']}：{row['failure_reason']}")
            for item in row["top_k"][:3]:
                lines.append(f"  - Top-{item.get('retrieval_rank')}: 第{item.get('chapter_no')}回 / 片段{item.get('chunk_no')} / {item.get('score_type')} / {item.get('score')}")
    return "\n".join(lines) + "\n"


async def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_path = Path(args.dataset)
    cases = load_cases(dataset_path)
    meta_path = dataset_path.with_suffix(".meta.json")
    expected_source_hash = None
    if meta_path.is_file():
        expected_source_hash = json.loads(meta_path.read_text(encoding="utf-8")).get("source_hash")
    index_metadata = await validate_index(args.file_id, args.user_id, expected_source_hash)
    preparation_cache = None
    if args.preparation_cache:
        preparation_cache = load_preparation_cache(Path(args.preparation_cache))
        validate_preparation_cache(
            preparation_cache,
            cases,
            file_id=args.file_id,
            index_metadata=index_metadata,
        )
    # RAG 查询会按 ContextVar 注入 user_id；评测必须显式使用索引所属租户，
    # 否则多租户过滤会把正确索引误判为空结果。
    user_token = set_current_user(args.user_id)
    results: list[dict[str, Any]] = []
    preparation_rows: dict[str, dict[str, Any]] = {}
    try:
        for case in cases:
            started = time.perf_counter()
            file_id = case.get("file_id") or args.file_id
            standalone = case["query"]
            retrieval = case["query"]
            rewrite_info = None
            case_id = str(case["id"])
            if preparation_cache is not None:
                cached = preparation_cache["cases"][case_id]
                standalone = cached["standalone_query"]
                retrieval = cached["retrieval_query"]
                rewrite_info = dict(cached)
                rewrite_info["source"] = "cache"
                rewrite_info["latency_ms"] = 0.0
            elif args.rewrite:
                rewrite_started = time.perf_counter()
                rr = await rewrite_query(case["query"], [])
                standalone = rr.standalone_query or case["query"]
                retrieval = rr.retrieval_query or standalone
                rewrite_info = {
                    "id": case_id,
                    "query": case["query"],
                    "applied": bool(rr.applied),
                    "reason": rr.reason,
                    "original": rr.original,
                    "standalone_query": rr.standalone_query,
                    "retrieval_query": rr.retrieval_query,
                    "intent": rr.intent,
                    "entities": list(rr.entities),
                    "evidence_focus": list(rr.evidence_focus),
                    "confidence": rr.confidence,
                    "latency_ms": round((time.perf_counter() - rewrite_started) * 1000, 2),
                    "source": "llm",
                }
            if rewrite_info is not None:
                preparation_rows[case_id] = dict(rewrite_info)
            primary_docs, trace = await similarity_search_with_trace(
                standalone, k=args.k, domain="novel", file_id=file_id, retrieval_query=retrieval
            )
            if args.chapter_local:
                # 章节内二级精排必须在邻居扩展之前：否则邻居片段会被二次检索顺序打乱。
                primary_docs = await chapter_local_refine(
                    standalone, primary_docs, args.k, file_id, args.user_id
                )
            expanded_docs = primary_docs
            if args.with_context:
                # 上下文扩展只读取主命中的邻居片段，不重复执行 RAG 主检索。
                expanded_docs = await expand_novel_context(primary_docs, args.neighbor_window)
            primary = []
            for doc in primary_docs:
                item = dict(doc.metadata)
                item["snippet"] = doc.page_content[:240].replace("\n", " ").strip()
                primary.append(item)
            expanded = []
            for doc in expanded_docs:
                item = dict(doc.metadata)
                item["snippet"] = doc.page_content[:240].replace("\n", " ").strip()
                expanded.append(item)
            result = metrics_for_case(
                case, primary, expanded, (time.perf_counter() - started) * 1000, trace, index_metadata
            )
            if rewrite_info is not None:
                result["rewrite"] = rewrite_info
                if result.get("failure_reason") and rewrite_info.get("reason") not in {"rewritten", "cached"}:
                    result["failure_reason"] = "preparation_fallback"
            results.append(result)
            print(
                f"[{len(results):02d}/{len(cases):02d}] {case['id']} "
                f"{'hit' if result['chunk_hit'] else 'MISS'}"
                + (f" (rewrite {rewrite_info['reason']})" if rewrite_info else "")
            )

        rewrite_rows = [row["rewrite"] for row in results if "rewrite" in row]
        rewrite_summary = None
        if rewrite_rows:
            reasons: dict[str, int] = defaultdict(int)
            for item in rewrite_rows:
                reasons[str(item.get("reason", "unknown"))] += 1
            rewrite_summary = {
                "questions": len(rewrite_rows),
                "applied": sum(bool(item.get("applied")) for item in rewrite_rows),
                "applied_rate": round(sum(bool(item.get("applied")) for item in rewrite_rows) / len(rewrite_rows), 4),
                "avg_latency_ms": round(statistics.mean(float(item.get("latency_ms", 0.0)) for item in rewrite_rows), 2),
                "reasons": dict(sorted(reasons.items())),
            }

        if args.write_preparation_cache:
            missing = [str(case["id"]) for case in cases if str(case["id"]) not in preparation_rows]
            if missing:
                raise RuntimeError(f"无法写入 Query Preparation 缓存，缺少题目输出：{missing}")
            write_preparation_cache(
                Path(args.write_preparation_cache),
                cases,
                preparation_rows,
                dataset=dataset_path,
                file_id=args.file_id,
                index_metadata=index_metadata,
            )
        aggregate_metrics = aggregate(results)
        baseline_comparison = compare_baseline(aggregate_metrics["overall"], load_baseline_report(args.baseline_report))
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset": str(Path(args.dataset).resolve()),
            "dataset_profile": dataset_profile(cases),
            "file_id": args.file_id,
            "user_id": args.user_id,
            "index_metadata": index_metadata,
            "k": args.k,
            "with_context": args.with_context,
            "neighbor_window": args.neighbor_window,
            "use_rewrite": bool(args.rewrite or args.preparation_cache),
            "query_preparation_mode": "cache" if args.preparation_cache else "llm" if args.rewrite else "disabled",
            "query_preparation_cache": str(Path(args.preparation_cache).resolve()) if args.preparation_cache else None,
            "query_preparation_cache_written": str(Path(args.write_preparation_cache).resolve()) if args.write_preparation_cache else None,
            "rewrite_summary": rewrite_summary,
            "config": config_snapshot(),
            "code_version": _git_metadata(),
            "evaluation_definition": {
                "primary_only": True,
                "neighbor_excluded": True,
                "ranking_field": "retrieval_rank",
                "gold_policy": "gold_chunks with relevance grades",
                "missing_feature_value": "N/A",
            },
            "metrics": aggregate_metrics,
            "baseline_comparison": baseline_comparison,
            "cases": results,
        }
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = args.name
        (out_dir / f"{stem}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out_dir / f"{stem}.md").write_text(
            markdown_report(payload), encoding="utf-8"
        )
        with (out_dir / f"{stem}_cases.jsonl").open("w", encoding="utf-8") as handle:
            for row in results:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(payload["metrics"], ensure_ascii=False, indent=2))
        return payload
    finally:
        reset_current_user(user_token)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评测当前小说 RAG 的 Recall@K、MRR 和 nDCG")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--file-id", required=True)
    parser.add_argument("--user-id", default=settings.default_user, help="索引所属用户 ID，多租户场景必须与 knowledge_files.user_id 一致")
    parser.add_argument("--output-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--name", default="xiyouji_rag_baseline")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--with-context", action="store_true")
    parser.add_argument("--neighbor-window", type=int, default=1)
    parser.add_argument(
        "--chapter-local",
        action="store_true",
        help="启用章节内二级精排（在主检索命中的章节内重新排序片段），用于 A/B 验证",
    )
    parser.add_argument("--rewrite", action="store_true", help="对每题先执行一次 RAG Query Preparation")
    parser.add_argument("--preparation-cache", default=None, help="复用已保存的 Query Preparation 快照，跳过 LLM 调用")
    parser.add_argument("--write-preparation-cache", default=None, help="保存本次 Query Preparation 输出，供后续实验复用")
    parser.add_argument("--baseline-report", default=None, help="可选历史 JSON 报告，用于同口径指标对比")
    parser.add_argument("--with-diagnostics", action="store_true", help="保留每题阶段诊断与分层特征；默认也会保留关键诊断")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
