"""检测 RAG 评测集的「查询泄漏」与金标正确性。

背景
----
金标题通常是**读过原文之后**写的，很容易不自觉把片段里的独特措辞抄进问题里
（例如直接引用回目原文、或照抄人物原话）。这类题目会被词法检索一步命中，
使 Recall 虚高，进而误导后续所有调参决策。

本脚本做两件事：
1. **金标校验**：每个 gold_chunk 必须真实存在，且 evidence_quote 必须出现在该
   片段的原文中（否则说明标注时张冠李戴）。
2. **泄漏度量**：计算「问题」与「金标片段原文」的最长公共子串长度。
   子串越长，说明问题里嵌入的原文越多，题目越"送分"。

经验阈值（以人工构造的 xiyouji_recall.jsonl 为参照）：
- 该数据集最长公共子串为 5 字（"大闹天宫的"），属自然提问的正常水平；
- 新建数据集应控制在 **<= 6 字**；超过则改写题目。

用法
----
    cd backend
    python scripts/check_dataset_leak.py ../evals/datasets/xiyouji_recall.jsonl
    python scripts/check_dataset_leak.py ../evals/datasets/hongloumeng_recall.jsonl --max-leak 6

退出码：0 = 通过；1 = 存在金标错误或泄漏超标。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text  # noqa: E402

from app.db import async_engine  # noqa: E402

# 归一化时丢弃这些字符，避免标点差异造成漏判
_PUNCT = re.compile(r"[\s,，。、；：:？?！!\"“”‘’'`()《》〈〉\[\]…—\-]")


def norm(s: str) -> str:
    return _PUNCT.sub("", s or "")


def longest_common_substring(a: str, b: str) -> tuple[int, str]:
    """返回最长公共子串的 (长度, 内容)。评测题很短，O(n*m) 足够。"""
    if not a or not b:
        return 0, ""
    prev = [0] * (len(b) + 1)
    best, end = 0, 0
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best, end = cur[j], i
        prev = cur
    return best, a[end - best:end]


async def check(dataset: Path, max_leak: int, verbose: bool) -> int:
    cases = [
        json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    file_ids = {case.get("file_id") for case in cases if case.get("file_id")}
    if not file_ids:
        print(f"[FAIL] 数据集缺少 file_id：{dataset}")
        return 1

    async with async_engine.connect() as conn:
        chunks: dict[tuple[str, int, int], str] = {}
        for fid in file_ids:
            rows = (
                await conn.execute(
                    text(
                        "SELECT chunk_no, chapter_no, content FROM embeddings "
                        "WHERE file_id = :f"
                    ),
                    {"f": fid},
                )
            ).all()
            for chunk_no, chapter_no, content in rows:
                chunks[(fid, chunk_no, chapter_no)] = content or ""

    gold_errors: list[str] = []
    leaks: list[tuple[str, int, str, str]] = []
    for case in cases:
        fid = case.get("file_id")
        q = norm(case["query"])
        best, frag = 0, ""
        for g in case.get("gold_chunks", []):
            key = (fid, g.get("chunk_no"), g.get("chapter_no"))
            if key not in chunks:
                gold_errors.append(f"{case['id']}: 金标片段不存在 {key}")
                continue
            content = chunks[key]
            quote = g.get("evidence_quote")
            if quote and norm(quote) not in norm(content):
                gold_errors.append(f"{case['id']}: evidence_quote 不在片段原文中 -> {quote[:24]}")
            n, f = longest_common_substring(q, norm(content))
            if n > best:
                best, frag = n, f
        leaks.append((case["id"], best, frag, case["query"]))

    over = [item for item in leaks if item[1] > max_leak]
    print(f"数据集：{dataset}")
    print(f"题目数：{len(cases)}   嵌入模型向量块：{len(chunks)}")
    print(f"金标校验：{'通过' if not gold_errors else f'{len(gold_errors)} 处错误'}")
    for err in gold_errors[:20]:
        print(f"   [x] {err}")
    print(f"泄漏阈值：最长公共子串 <= {max_leak} 字；超标 {len(over)}/{len(leaks)} 题")
    for cid, n, frag, q in sorted(over, key=lambda x: -x[1])[:20]:
        print(f"   [!] {cid}  {n}字  片段\"{frag}\"   题目：{q}")
    if verbose:
        print("全部题目泄漏长度（降序前 10）：")
        for cid, n, frag, q in sorted(leaks, key=lambda x: -x[1])[:10]:
            print(f"   {cid}  {n}字  \"{frag}\"")

    ok = not gold_errors and not over
    print("结论：", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="校验 RAG 评测集的金标正确性与查询泄漏")
    parser.add_argument("dataset", help="JSONL 评测集路径")
    parser.add_argument("--max-leak", type=int, default=6, help="允许的最长公共子串字数（默认 6）")
    parser.add_argument("--verbose", action="store_true", help="打印全部题目的泄漏长度")
    args = parser.parse_args()
    return asyncio.run(check(Path(args.dataset), args.max_leak, args.verbose))


if __name__ == "__main__":
    raise SystemExit(main())
