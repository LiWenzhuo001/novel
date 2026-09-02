# -*- coding: utf-8 -*-
"""从召回评测集生成答案侧评测金标（reference_answer + key_points）。

数据来源：evals/datasets/{xiyouji,hongloumeng}_recall.jsonl（各 40 题，含金标片段定位
与 evidence_quote）。本脚本按类别分层确定性抽样 N 题，从数据库回读金标片段正文，
让 LLM 严格依据片段起草参考答案与关键点（决策：LLM 起草、不做人工复核），并做
自动校验代替人工：key_points 中的数字/双字词必须能在片段原文中找到依据。

产物：evals/datasets/{stem}_answer.jsonl —— evaluate_agent_answer.py 的评测输入。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[2] / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from pydantic import BaseModel, Field  # noqa: E402

from app.config import settings  # noqa: E402
from app.core.llm import get_llm  # noqa: E402
from app.db import AsyncSessionLocal  # noqa: E402
from app.db.models import Embedding  # noqa: E402
from sqlalchemy import select  # noqa: E402

DEFAULT_DATASET_DIR = Path(__file__).resolve().parent


class DraftAnswer(BaseModel):
    """LLM 起草参考答案的严格 JSON 结构。"""

    reference_answer: str = Field(min_length=20, max_length=400)
    key_points: list[str] = Field(min_length=2, max_length=5)


def load_cases(path: Path) -> list[dict]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if item.get("id") and item.get("query") and item.get("gold_chunks"):
                cases.append(item)
    if not cases:
        raise ValueError(f"评测集为空：{path}")
    return cases


def stratified_subset(cases: list[dict], size: int) -> list[dict]:
    """按类别分层、组内按 id 排序、用最大余数法确定各组配额，保证确定性。"""
    by_category: dict[str, list[dict]] = defaultdict(list)
    for case in sorted(cases, key=lambda item: str(item["id"])):
        by_category[str(case.get("category", "unknown"))].append(case)
    categories = sorted(by_category)
    if size >= len(cases):
        return sorted(cases, key=lambda item: str(item["id"]))
    quotas = {c: size * len(by_category[c]) // len(cases) for c in categories}
    remainder = size - sum(quotas.values())
    # 余数按各组小数部分从大到小分配；并列时按类别名稳定排序。
    fractions = sorted(
        categories,
        key=lambda c: (-(size * len(by_category[c]) / len(cases) - quotas[c]), c),
    )
    for c in fractions[:remainder]:
        quotas[c] += 1
    picked: list[dict] = []
    for c in categories:
        picked.extend(by_category[c][: quotas[c]])
    return picked


async def fetch_gold_content(file_id: str, user_id: str, chapter_no: int, chunk_no: int) -> str | None:
    async with AsyncSessionLocal() as session:
        stmt = select(Embedding.content).where(
            Embedding.user_id == user_id,
            Embedding.file_id == file_id,
            Embedding.chapter_no == chapter_no,
            Embedding.chunk_no == chunk_no,
        )
        return (await session.execute(stmt)).scalar_one_or_none()


def _char_bigrams(text: str) -> set[str]:
    cleaned = re.sub(r"\s+", "", text)
    return {cleaned[i:i + 2] for i in range(len(cleaned) - 1)}


def key_point_supported(key_point: str, chunk: str) -> bool:
    """自动校验：key_point 应能在片段原文中找到可溯源依据。

    规则：数字（阿拉伯/常见中文数字单字）必须出现在片段中；且满足以下之一——
    存在长度 >=3 的连续中文串（或其子串）在片段中出现（锚点），或中文双字
    bigram 覆盖率 >= 0.45。key_point 允许轻度改写，但实体与数字必须可溯源。
    """
    chunk_bigrams = _char_bigrams(chunk)
    point_bigrams = _char_bigrams(key_point)
    coverage = len(point_bigrams & chunk_bigrams) / len(point_bigrams) if point_bigrams else 0.0
    for number in re.findall(r"[0-9]+", key_point):
        if number not in chunk:
            return False
    for char in re.findall(r"[一二两三四五六七八九十百千]", key_point):
        if char not in chunk:
            return False
    for run in re.findall(r"[\u3400-\u9fff]{2,}", key_point):
        if run in chunk:
            return True
        if any(run[i:i + 3] in chunk for i in range(len(run) - 2)):
            return True
    return coverage >= 0.45


def _extract_json(content: object) -> dict:
    text = content if isinstance(content, str) else str(content)
    fenced = re.search(r"\{[\s\S]*\}", text.strip())
    if not fenced:
        raise ValueError("json_object_not_found")
    return json.loads(fenced.group(0))


async def draft_one(llm, case: dict, chunk: str) -> DraftAnswer:
    """调用 LLM 起草参考答案；key_point 校验失败时带反馈重试一次。"""
    prompt = (
        f"问题：{case['query']}\n\n原文片段（唯一事实来源）：\n{chunk}\n\n"
        "请基于且仅基于该片段起草评测金标。输出 JSON：\n"
        '{"reference_answer": "100~200字的参考答案，严格依据片段表述，不引入片段之外的信息",\n'
        ' "key_points": ["关键事实点", ...]}\n'
        "key_points 给 3~5 条，每条不超过 40 字，必须是片段原文中可直接验证的陈述（含人物、事件或数字）；\n"
        "禁止使用片段中没有的章节号、回目或任何外部知识。\n只输出 JSON。"
    )
    feedback = ""
    payload = None
    for attempt in range(2):
        response = await llm.ainvoke([
            {"role": "system", "content": "你是小说问答评测的金标起草器，只依据给定原文回答，不使用外部知识。只输出 JSON。" + feedback},
            {"role": "user", "content": prompt},
        ])
        payload = DraftAnswer.model_validate(_extract_json(response.content))
        invalid = [kp for kp in payload.key_points if not key_point_supported(kp, chunk)]
        if not invalid:
            return payload
        feedback = (
            "上一稿中以下关键点无法从原文片段中找到足够依据，请更贴近原文措辞重写："
            + "；".join(invalid)
        )
        if attempt == 0:
            print(f"    校验未过，重试：{case['id']} invalid={invalid}")
    return payload


async def run(args: argparse.Namespace) -> None:
    dataset_path = Path(args.dataset)
    cases = load_cases(dataset_path)
    subset = stratified_subset(cases, args.subset)
    if args.limit:
        subset = subset[: args.limit]
    print(f"{dataset_path.name}: 共 {len(cases)} 题，分层抽样 {len(subset)} 题")
    print(f"  类别分布：{dict(Counter(str(c.get('category', 'unknown')) for c in subset))}")

    llm = get_llm(temperature=0, max_tokens=600)
    rows = []
    failed_validation = 0
    for index, case in enumerate(subset, start=1):
        gold = case["gold_chunks"][0]
        chunk = await fetch_gold_content(
            case.get("file_id") or args.file_id or "", args.user_id,
            gold.get("chapter_no"), gold.get("chunk_no"),
        )
        if not chunk:
            print(f"[{index:02d}/{len(subset):02d}] {case['id']} 跳过：金标片段不在库中")
            continue
        payload = await draft_one(llm, case, chunk)
        # 剔除仍无法溯源的 key_point（LLM 混入片段外信息的兜底）；>=2 条才算校验通过。
        supported = [kp for kp in payload.key_points if key_point_supported(kp, chunk)]
        validated = len(supported) >= 2
        failed_validation += 0 if validated else 1
        rows.append({
            "id": case["id"],
            "category": case.get("category", "unknown"),
            "query": case["query"],
            "file_id": case.get("file_id") or args.file_id,
            "gold_chunks": case["gold_chunks"],
            "gold_chapters": case.get("gold_chapters", []),
            "reference_answer": payload.reference_answer,
            "key_points": supported,
            "key_points_validated": validated,
            "answer_source": "llm_draft",
            "model": settings.llm_model,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        })
        status = "ok" if validated else "VALIDATION_WEAK"
        print(f"[{index:02d}/{len(subset):02d}] {case['id']} {status} key_points={len(supported)}/{len(payload.key_points)}")
        await asyncio.sleep(0.2)

    if not rows:
        raise RuntimeError("没有生成任何金标答案")
    output_path = Path(args.output) if args.output else dataset_path.with_name(
        dataset_path.stem.replace("_recall", "") + "_answer.jsonl"
    )
    output_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    meta = {
        "source_dataset": str(dataset_path.resolve()),
        "subset": len(rows),
        "stratified_from": len(cases),
        "validation_weak": failed_validation,
        "model": settings.llm_model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    output_path.with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"已写入 {len(rows)} 题 -> {output_path}（校验薄弱 {failed_validation} 题）")
    print("\n===== 抽样展示（供事后抽查）=====")
    for row in rows[:3]:
        print(f"\n[{row['id']}] {row['query']}")
        print(f"  参考答案：{row['reference_answer']}")
        print("  关键点：" + " / ".join(row["key_points"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成答案侧评测金标（LLM 起草 + 自动校验）")
    parser.add_argument("--dataset", required=True, help="召回评测集 JSONL")
    parser.add_argument("--user-id", required=True, help="索引所属用户 ID")
    parser.add_argument("--file-id", default=None, help="数据集行缺 file_id 时的兜底索引")
    parser.add_argument("--subset", type=int, default=20, help="分层抽样题数")
    parser.add_argument("--limit", type=int, default=0, help="调试：只处理前 N 题")
    parser.add_argument("--output", default=None, help="输出路径，默认 {stem}_answer.jsonl")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
