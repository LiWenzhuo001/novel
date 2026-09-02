"""Offline RAG evaluation for labeled chunk-level relevance judgments.

Example:
    python scripts/run_rag_eval.py --eval evals/rag_queries.json --output evals/results/current
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _key(item: dict) -> tuple:
    return item.get("file_id"), item.get("chapter_no"), item.get("chunk_no")


def _dcg(relevances: list[int]) -> float:
    return sum(value / math.log2(index + 2) for index, value in enumerate(relevances))


def _case_metrics(retrieved: list[dict], relevant: set[tuple], k: int) -> dict:
    top = retrieved[:k]
    flags = [1 if _key(item) in relevant else 0 for item in top]
    hits = sum(flags)
    recall = hits / len(relevant) if relevant else 0.0
    precision = hits / k if k else 0.0
    reciprocal_rank = next((1.0 / (index + 1) for index, value in enumerate(flags) if value), 0.0)
    ideal = [1] * min(len(relevant), k)
    ndcg = _dcg(flags) / _dcg(ideal) if ideal else 0.0
    return {"recall": recall, "precision": precision, "mrr": reciprocal_rank, "ndcg": ndcg, "hits": hits}


async def evaluate(eval_path: Path, output_prefix: Path, user_id: str, allow_small: bool) -> dict:
    from app.config import settings
    from app.core.context import reset_current_user, set_current_user
    from app.core.rag import similarity_search

    cases = json.loads(eval_path.read_text(encoding="utf-8"))
    validated = [case for case in cases if case.get("validated", True)]
    if not validated:
        raise SystemExit("评测集没有 validated=true 的人工标注用例。")
    if len(validated) < 40 and not allow_small:
        raise SystemExit(
            f"有效人工标注仅 {len(validated)} 条；验收要求至少 40 条。"
            "补齐后重试，或仅调试时使用 --allow-small。"
        )

    token = set_current_user(user_id)
    rows = []
    latencies = []
    try:
        for case in validated:
            started = time.perf_counter()
            docs = await similarity_search(case["query"], k=10, file_id=case.get("file_id"))
            latency_ms = (time.perf_counter() - started) * 1000
            latencies.append(latency_ms)
            retrieved = [doc.metadata for doc in docs]
            relevant = {_key(item) for item in case.get("relevant_chunks", [])}
            rows.append({
                "id": case.get("id"),
                "category": case.get("category", "unknown"),
                "query": case["query"],
                "latency_ms": round(latency_ms, 1),
                "metrics_at_5": _case_metrics(retrieved, relevant, 5),
                "metrics_at_10": _case_metrics(retrieved, relevant, 10),
                "retrieved": [
                    {
                        "file_id": meta.get("file_id"),
                        "chapter_no": meta.get("chapter_no"),
                        "chunk_no": meta.get("chunk_no"),
                        "score": meta.get("score"),
                        "score_type": meta.get("score_type"),
                        "retrieval_rank": meta.get("retrieval_rank"),
                    }
                    for meta in retrieved
                ],
            })
    finally:
        reset_current_user(token)

    def average(path: tuple[str, str]) -> float:
        return statistics.fmean(row[path[0]][path[1]] for row in rows) if rows else 0.0

    sorted_latency = sorted(latencies)
    p95 = sorted_latency[min(len(sorted_latency) - 1, int(len(sorted_latency) * 0.95))] if sorted_latency else 0.0
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "eval_file": str(eval_path),
        "cases": len(rows),
        "config": {
            "embedding_model": settings.embedding_model,
            "embed_dim": settings.embed_dim,
            "chunk_size": settings.novel_chunk_size,
            "chunk_overlap": settings.novel_chunk_overlap,
            "hybrid_candidate_k": settings.hybrid_candidate_k,
            "reranker_enabled": settings.enable_reranker,
            "reranker_model": settings.reranker_model,
        },
        "metrics": {
            "recall_at_5": round(average(("metrics_at_5", "recall")), 4),
            "recall_at_10": round(average(("metrics_at_10", "recall")), 4),
            "precision_at_5": round(average(("metrics_at_5", "precision")), 4),
            "mrr_at_10": round(average(("metrics_at_10", "mrr")), 4),
            "ndcg_at_10": round(average(("metrics_at_10", "ndcg")), 4),
            "no_result_rate": round(sum(not row["retrieved"] for row in rows) / len(rows), 4) if rows else 0.0,
            "avg_latency_ms": round(statistics.fmean(latencies), 1) if latencies else 0.0,
            "p95_latency_ms": round(p95, 1),
        },
        "results": rows,
    }

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_prefix.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metrics = summary["metrics"]
    markdown = [
        "# RAG 离线评测结果", "",
        f"- 时间：{summary['generated_at']}",
        f"- 用例：{summary['cases']}",
        f"- Recall@5：{metrics['recall_at_5']:.2%}",
        f"- Recall@10：{metrics['recall_at_10']:.2%}",
        f"- Precision@5：{metrics['precision_at_5']:.2%}",
        f"- MRR@10：{metrics['mrr_at_10']:.4f}",
        f"- nDCG@10：{metrics['ndcg_at_10']:.4f}",
        f"- 无结果率：{metrics['no_result_rate']:.2%}",
        f"- 平均延迟：{metrics['avg_latency_ms']:.1f} ms",
        f"- P95 延迟：{metrics['p95_latency_ms']:.1f} ms", "",
        "| ID | 分类 | Recall@10 | MRR@10 | 延迟(ms) |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            f"| {row['id']} | {row['category']} | {row['metrics_at_10']['recall']:.2%} | "
            f"{row['metrics_at_10']['mrr']:.3f} | {row['latency_ms']:.1f} |"
        )
    output_prefix.with_suffix(".md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", type=Path, default=ROOT / "evals" / "rag_queries.json")
    parser.add_argument("--output", type=Path, default=ROOT / "evals" / "results" / "current")
    parser.add_argument("--user", default="default")
    parser.add_argument("--allow-small", action="store_true")
    args = parser.parse_args()
    result = asyncio.run(evaluate(args.eval, args.output, args.user, args.allow_small))
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
