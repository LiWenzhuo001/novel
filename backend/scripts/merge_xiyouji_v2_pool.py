"""《西游记》v2 —— 第 3 步：合并池化结果，生成待判定文件。

合并规则：
- 各配置命中的片段取并集，按 chunk_no 去重
- 用 RRF 式融合分对候选排序（被多个配置命中、且名次靠前的排前面）
- 每题保留 Top-N 候选，附原文片段供判定

输出分块文件，便于分批人工判定。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlalchemy import text  # noqa: E402
from app.db import async_engine  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
POOL_DIR = ROOT / "evals" / "pooling"
TOPIC_FILE = ROOT / "evals" / "datasets" / "xiyouji_v2_topics.jsonl"
RRF_C = 60
SNIPPET = 150


async def main(configs: list[str], keep: int, snippet: int, batch_size: int) -> None:
    topics = [
        json.loads(line) for line in TOPIC_FILE.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    pools: dict[str, dict[str, list[dict]]] = {}
    for cfg in configs:
        path = POOL_DIR / f"pool_{cfg}.jsonl"
        pools[cfg] = {
            row["id"]: row["hits"]
            for row in (json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip())
        }

    file_ids = {t["file_id"] for t in topics}
    async with async_engine.connect() as conn:
        chunks: dict[tuple[str, int], tuple[int, str]] = {}
        for fid in file_ids:
            rows = (await conn.execute(
                text("SELECT chunk_no, chapter_no, content FROM embeddings WHERE file_id=:f"), {"f": fid}
            )).all()
            for chunk_no, chapter_no, content in rows:
                chunks[(fid, chunk_no)] = (chapter_no, content or "")

    out_dir = POOL_DIR / "judging"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("batch_*.jsonl"):
        old.unlink()

    records = []
    pool_sizes = []
    for topic in topics:
        fid = topic["file_id"]
        scored: dict[int, dict] = {}
        for cfg in configs:
            for hit in pools[cfg].get(topic["id"], []):
                cn = hit["chunk_no"]
                if cn is None:
                    continue
                entry = scored.setdefault(cn, {
                    "chunk_no": cn,
                    "chapter_no": hit.get("chapter_no"),
                    "rrf": 0.0,
                    "found_by": [],
                    "best_rank": 99,
                })
                entry["rrf"] += 1.0 / (RRF_C + hit["rank"])
                entry["found_by"].append(f"{cfg}#{hit['rank']}")
                entry["best_rank"] = min(entry["best_rank"], hit["rank"])
        ranked = sorted(scored.values(), key=lambda e: (-e["rrf"], e["best_rank"]))
        pool_sizes.append(len(ranked))
        kept = ranked[:keep]
        cands = []
        for e in kept:
            chapter_no, content = chunks.get((fid, e["chunk_no"]), (e["chapter_no"], ""))
            body = " ".join(content.split())
            cands.append({
                "chunk_no": e["chunk_no"],
                "chapter_no": chapter_no,
                "found_by": e["found_by"],
                "text": body[:snippet],
            })
        records.append({
            "id": topic["id"],
            "category": topic["category"],
            "query": topic["query"],
            "narrative": topic["narrative"],
            "file_id": fid,
            "pool_size": len(ranked),
            "candidates": cands,
        })

    n_batches = (len(records) + batch_size - 1) // batch_size
    for i in range(n_batches):
        part = records[i * batch_size:(i + 1) * batch_size]
        p = out_dir / f"batch_{i + 1:02d}.jsonl"
        p.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in part) + "\n", encoding="utf-8"
        )
        print(f"  {p.name}: {len(part)} 题")

    print(f"\n共 {len(records)} 题，{n_batches} 个批次")
    print(f"池大小：平均 {sum(pool_sizes)/len(pool_sizes):.1f}  最小 {min(pool_sizes)}  最大 {max(pool_sizes)}")
    print(f"每题保留判定候选：{keep}   片段长度：{snippet} 字")
    empty = [r["id"] for r in records if not r["candidates"]]
    print(f"无候选题：{len(empty)} {empty[:10]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default="hybrid_default,vector_only,hybrid_fts,hybrid_wide,hybrid_rewrite")
    parser.add_argument("--keep", type=int, default=14)
    parser.add_argument("--snippet", type=int, default=SNIPPET)
    parser.add_argument("--batch-size", type=int, default=9)
    args = parser.parse_args()
    asyncio.run(main(args.configs.split(","), args.keep, args.snippet, args.batch_size))
