"""《西游记》v2 —— 生成 Query Preparation 缓存（冻结 100 题的改写输出）。

把 LLM 改写结果落盘，使池化与后续所有实验都可复现，不受模型抖动影响。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.core.query_rewriter import rewrite_query  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TOPIC_FILE = ROOT / "evals" / "datasets" / "xiyouji_v2_topics.jsonl"


async def main(out_path: Path) -> None:
    topics = [
        json.loads(line) for line in TOPIC_FILE.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    rows = []
    reasons: dict[str, int] = {}
    total_ms = 0.0
    for i, topic in enumerate(topics, start=1):
        started = time.perf_counter()
        rr = await rewrite_query(topic["query"], [])
        elapsed = (time.perf_counter() - started) * 1000
        total_ms += elapsed
        reasons[rr.reason] = reasons.get(rr.reason, 0) + 1
        rows.append({
            "id": topic["id"],
            "query": topic["query"],
            "original": rr.original,
            "standalone_query": rr.standalone_query or topic["query"],
            "retrieval_query": rr.retrieval_query,
            "intent": rr.intent,
            "entities": list(rr.entities),
            "evidence_focus": list(rr.evidence_focus),
            "confidence": rr.confidence,
            "applied": bool(rr.applied),
            "reason": rr.reason,
            "latency_ms": round(elapsed, 2),
        })
        if i % 10 == 0:
            print(f"  {i}/{len(topics)}  累计 {total_ms/1000:.1f}s", file=sys.stderr)

    payload = {
        "version": "v2-query-prep-1",
        "dataset": str(TOPIC_FILE),
        "file_id": topic.get("file_id"),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "prompt_version": settings.query_rewrite_prompt_version,
        "questions": len(rows),
        "applied": sum(bool(r["applied"]) for r in rows),
        "avg_latency_ms": round(total_ms / len(rows), 2),
        "reasons": reasons,
        "cases": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {len(rows)} 条 -> {out_path}")
    print(f"  应用改写 {payload['applied']}/{len(rows)}  平均 {payload['avg_latency_ms']}ms")
    print(f"  原因分布 {reasons}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "evals" / "datasets" / "xiyouji_v2_query_preparation.json"))
    args = parser.parse_args()
    asyncio.run(main(Path(args.out)))
