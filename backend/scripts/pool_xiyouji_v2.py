"""《西游记》v2 —— 第 2 步：多配置池化（TREC pooling）。

为什么要池化
------------
若只用当前系统自己召回的结果去做相关性判定，那么"没被它召回的相关片段"永远不会被
发现，金标会系统性偏向该系统，之后所有对比实验都占它便宜。
标准做法是让**多个差异化系统**各自取 Top-N，合并去重后统一判定。

本脚本一次只跑一个配置（配置由环境变量注入，因为 app.config 在 import 时读取 env），
输出该配置下每题的 Top-N 命中。由调用方用不同 env 跑多次，再合并。

用法
----
    cd backend
    POSTGRES_HOST=127.0.0.1 \
    ENABLE_HYBRID_SEARCH=true ENABLE_CHINESE_LEXICAL_SEARCH=true \
    ./.venv/Scripts/python.exe scripts/pool_xiyouji_v2.py \
      --config hybrid_default --top 8
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.context import set_current_user  # noqa: E402
from app.core.rag import similarity_search  # noqa: E402
from app.config import settings  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TOPIC_FILE = ROOT / "evals" / "datasets" / "xiyouji_v2_topics.jsonl"
POOL_DIR = ROOT / "evals" / "pooling"


def _load_prep(path: Path) -> dict[str, dict]:
    """载入 Query Preparation 缓存，兼容 JSONL 与 JSON 对象两种格式。"""
    text = path.read_text(encoding="utf-8")
    raw_cases = None
    try:
        payload = json.loads(text)
        if isinstance(payload, dict) and "cases" in payload:
            raw_cases = payload["cases"]
    except json.JSONDecodeError:
        raw_cases = [json.loads(line) for line in text.splitlines() if line.strip()]
    if raw_cases is None:
        raw_cases = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(raw_cases, dict):
        return raw_cases
    return {c["id"]: c for c in raw_cases}


async def run(config: str, top: int, user_id: str, rewrite: bool) -> None:
    topics = [
        json.loads(line) for line in TOPIC_FILE.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    prep = None
    if rewrite:
        cache = ROOT / "evals" / "datasets" / "xiyouji_v2_query_preparation.json"
        if cache.exists():
            prep = _load_prep(cache)
        else:
            print(f"[warn] 未找到改写缓存 {cache}，回退为原始问题", file=sys.stderr)

    set_current_user(user_id)
    POOL_DIR.mkdir(parents=True, exist_ok=True)
    out = POOL_DIR / f"pool_{config}.jsonl"

    lines = []
    for i, topic in enumerate(topics, start=1):
        query = topic["query"]
        retrieval_query = None
        if prep and topic["id"] in prep:
            item = prep[topic["id"]]
            query = item.get("standalone_query") or query
            retrieval_query = item.get("retrieval_query")
        docs = await similarity_search(
            query, k=top, domain="novel", file_id=topic["file_id"],
            retrieval_query=retrieval_query,
        )
        hits = [
            {
                "chunk_no": d.metadata.get("chunk_no"),
                "chapter_no": d.metadata.get("chapter_no"),
                "rank": r,
                "score": d.metadata.get("score"),
                "score_type": d.metadata.get("score_type"),
            }
            for r, d in enumerate(docs, start=1)
        ]
        lines.append(json.dumps({
            "id": topic["id"],
            "query_used": query,
            "hits": hits,
        }, ensure_ascii=False))
        if i % 20 == 0:
            print(f"  {i}/{len(topics)}", file=sys.stderr)

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[{config}] 已写入 {len(lines)} 题 -> {out}")
    print(f"  生效配置: hybrid={settings.enable_hybrid_search} "
          f"chinese_lexical={settings.enable_chinese_lexical_search} "
          f"K={settings.hybrid_candidate_k} rrf_k={settings.rrf_k} "
          f"reranker={settings.enable_reranker} rewrite={'cache' if prep else 'off'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="单配置池化")
    parser.add_argument("--config", required=True, help="配置名，用于输出文件名")
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--user-id", default="c1e78368550f498787e3871ed9291b63")
    parser.add_argument("--rewrite", action="store_true", help="使用改写缓存中的查询")
    args = parser.parse_args()
    asyncio.run(run(args.config, args.top, args.user_id, args.rewrite))


if __name__ == "__main__":
    main()
