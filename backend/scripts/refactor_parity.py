"""RAG 重构前后奇偶校验：抓取确定性检索输出并在重构后重跑比对。



用法（在 backend 目录下，需要 POSTGRES_HOST=127.0.0.1 连通数据库）：

    1) 重构前先抓基线（默认写入 evals/baselines/refactor_parity_pre.json)：
       python scripts/refactor_parity.py capture
    2) 重构后重跑抓取（默认写入 evals/baselines/refactor_parity_post.json）：
       python scripts/refactor_parity.py capture --out evals/baselines/refactor_parity_post.json
    3) 逐查询比对（任一差异即报行为漂移）：
       python scripts/refactor_parity.py compare --base evals/baselines/refactor_parity_pre.json \
           --cur evals/baselines/refactor_parity_post.json

与仓库内 `evals/baselines/refactor_parity_before.json` 同构：每个查询记录
`final`（Top-8：file_id/chapter_no/chunk_no/rank）、`rrf_order`（RRF 候选前 20：
chapter_no/chunk_no）、`counts`（各阶段候选数）、`timings_keys`（阶段时间键）。
额外附 `settings` 快照，用于排除"条件不一致导致漂移"的误判。检索管线在
相同条件下是确定性的，pre/post 必须逐字节一致。


代表查询与书库 file_id 硬编码，与基线文件来源一致：《红楼梦》三条、《西游记》一条。

"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core import rag  # noqa: E402
from app.core.context import reset_current_user, set_current_user  # noqa: E402

# 与 evals/baselines/refactor_parity_before.json 同源的 4 条代表查询。

QUERIES: list[tuple[str, str]] = [
    ("香菱学诗是谁教的？", "93e25961b957"),        # 红楼梦
    ("孙悟空的金箍棒是从哪里来的？", "c959fb4c1a30"),    # 西游记
    ("贾宝玉佩戴的玉上刻着什么字？", "93e25961b957"),    # 红楼梦
    ("林黛玉为什么葬花？", "93e25961b957"),              # 红楼梦

]

# 评测索引归属租户（与 evaluate_rag_recall 常规跑法一致）。
USER_ID = "c1e78368550f498787e3871ed9291b63"
TOP_K = 8


def _settings_snapshot() -> dict:
    """记录与检索确定性相关的活动配置，供 pre/post 比对时排除环境漂移。"""
    s = rag.settings
    return {
        "enable_hybrid_search": s.enable_hybrid_search,
        "enable_bm25_search": s.enable_bm25_search,
        "enable_chinese_lexical_search": s.enable_chinese_lexical_search,
        "fusion_mode": s.fusion_mode,
        "enable_reranker": s.enable_reranker,
        "hybrid_candidate_k": s.hybrid_candidate_k,
        "reranker_candidate_n": s.reranker_candidate_n,
        "top_k": s.top_k,
    }


def _entry(query: str, docs, trace) -> dict:
    final = []
    for doc in docs:
        meta = doc.metadata
        final.append(
            [
                meta.get("file_id"),
                meta.get("chapter_no"),
                meta.get("chunk_no"),
                meta.get("retrieval_rank"),
            ]
        )
    rrf_order = None
    if "rrf_candidates" in trace:
        rrf_order = [
            [c.get("chapter_no"), c.get("chunk_no")]
            for c in trace["rrf_candidates"][:20]
        ]
    return {
        "query": query,
        "final": final,
        "rrf_order": rrf_order,
        "counts": trace.get("candidate_counts", {}),
        "timings_keys": sorted(trace.get("phase_timings_ms", {})),
        "settings": _settings_snapshot(),
    }


async def capture(out_path: Path) -> None:
    """对 4 条代表查询各执行一次主检索并抓取完整阶段输出。"""
    token = set_current_user(USER_ID)
    try:
        payload = []
        for query, file_id in QUERIES:
            docs, trace = await rag.similarity_search_with_trace(
                query,
                k=TOP_K,
                domain="novel",
                file_id=file_id,
                # 不接 Query Rewriter：改写是非确定入口，且基线文件聚焦检索管线本身。
                retrieval_query=None,
            )
            payload.append(_entry(query, docs, trace))
            print(f"  ok: {query}  final={len(docs)}", flush=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote {out_path}（{len(payload)} 条查询）")
    finally:
        reset_current_user(token)


def _norm_payload(data: list[dict]) -> list[dict]:
    """清洗设置快照等非比对字段，保持与基线 JSON 同构。"""
    out = []
    for entry in data:
        out.append(
            {
                "query": entry["query"],
                "final": entry.get("final") or [],
                "rrf_order": entry.get("rrf_order") or [],
                "counts": entry.get("counts") or {},
                "timings_keys": entry.get("timings_keys") or [],
            }
        )
    return out


def compare(base_path: Path, cur_path: Path) -> int:
    base = json.loads(base_path.read_text(encoding="utf-8"))
    cur_data = json.loads(cur_path.read_text(encoding="utf-8"))
    base_norm = {e["query"]: e for e in _norm_payload(base)}
    cur_norm = {e["query"]: e for e in _norm_payload(cur_data)}
    base_q = list(base_norm)
    cur_q = list(cur_norm)
    if base_q != cur_q:
        print(f"查询集合不一致：base={base_q} cur={cur_q}")
        return 2
    failures = 0
    for q in base_q:
        b = base_norm[q]
        c = cur_norm[q]
        diffs = []
        for key in ("final", "rrf_order", "counts", "timings_keys"):
            if b.get(key) != c.get(key):
                diffs.append(key)
        if diffs:
            failures += 1
            print(f"[FAIL] {q}：{diffs}")
            print(f"    base.final={json.dumps(b['final'], ensure_ascii=False)}")
            print(f"    cur.final ={json.dumps(c['final'], ensure_ascii=False)}")
        else:
            print(f"[ok]  {q}")
    # 配置快照仅做提示，不算漂移（供人工排查环境差异）。
    if base and "settings" in base[0] and cur_data and "settings" in cur_data[0]:
        if base[0]["settings"] != cur_data[0]["settings"]:
            print("note: 前后 settings 快照不一致（见文件内 settings，人工核对）", end=" ")
    print(f"--- {len(base_q) - failures}/{len(base_q)} 条查询一致")
    return 0 if failures == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG 重构奇偶校验")
    sub = parser.add_subparsers(dest="mode", required=True)
    cap = sub.add_parser("capture", help="抓取当前实现下的检索输出")
    cap.add_argument(
        "--out",
        default=BACKEND_DIR.parent / "evals" / "baselines" / "refactor_parity_pre.json",
    )
    cmp = sub.add_parser("compare", help="比对两份抓取输出")
    cmp.add_argument("--base", required=True)
    cmp.add_argument("--cur", required=True)
    args = parser.parse_args()
    if args.mode == "capture":
        asyncio.run(capture(Path(args.out)))
        return 0
    return compare(Path(args.base), Path(args.cur))


if __name__ == "__main__":
    raise SystemExit(main())
