# -*- coding: utf-8 -*-
"""Cross-encoder 重排器健全性测试：验证“650 字中文块在 512 token 下被截断”假设。

背景：29 次受控实验中开启重排恒为负收益（evals/README.md §五），但评测数据显示
瓶颈在“池内排序”（RRF 池召回 0.575 → Top-10 仅 0.325），这正是重排器该解决的问题。
首要嫌疑：rerank.py 用 CrossEncoder(settings.reranker_model) 初始化，未设置 max_length，
bge-reranker-base 基于 XLM-RoBERTa，位置编码上限 514 token——650 字中文块必然被截断，
金标证据句若在块尾部则重排器根本看不到。

本脚本对每题构造 (query, 金标片段) 与若干 (query, 同章节干扰片段) 对，对比两种变体：
  - full：完整片段按模型原生 max_length 截断（生产行为）
  - window：把金标片段按 evidence_quote 居中裁一个 ~450 字窗口（上界诊断：
    回答“若重排器能看到证据，能否把金标排第一”——不是生产候选方案）

另支持 --models 追加长上下文重排模型（如 BAAI/bge-reranker-v2-m3，max_length 可 >512），
用于验证“换模型能否解决”。

判定规则（写入报告）：
  - window_top1 - full_top1 >= 0.15 → 截断实锤：修复方向是证据感知窗口/分块或换长上下文模型
  - 两者都低 → 模型对该任务本身偏弱，截断不是主因
  - 同时报告 pair token 长度分布与超 512 占比、模型原生上限、输出范围（双重 sigmoid 检测）
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings  # noqa: E402
from app.db import AsyncSessionLocal  # noqa: E402
from app.db.models import Embedding  # noqa: E402
from sqlalchemy import select  # noqa: E402

DEFAULT_DATASET = Path(__file__).resolve().parents[2] / "evals" / "datasets" / "xiyouji_recall.jsonl"
DEFAULT_REPORT_DIR = Path(__file__).resolve().parents[2] / "evals" / "reports"
_WINDOW_CHARS = 450  # 中文约 1 字 1 token，留出 query 与特殊 token 的余量


def load_cases(path: Path) -> list[dict]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if item.get("query") and item.get("gold_chunks"):
                cases.append(item)
    return cases


async def fetch_chunks(file_id: str, user_id: str, wanted: dict[int, list[int]]) -> dict[int, str]:
    """按 (chapter_no, chunk_no) 批量回读片段正文。"""
    result: dict[int, str] = {}
    async with AsyncSessionLocal() as session:
        for chapter_no, chunk_nos in wanted.items():
            stmt = select(Embedding.chunk_no, Embedding.content).where(
                Embedding.user_id == user_id,
                Embedding.file_id == file_id,
                Embedding.chapter_no == chapter_no,
                Embedding.chunk_no.in_(tuple(chunk_nos)),
            )
            for chunk_no, content in (await session.execute(stmt)).all():
                result[(chapter_no, chunk_no)] = content
    return result


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def make_window(chunk: str, quote: str | None, max_chars: int = _WINDOW_CHARS) -> str:
    """以证据句为重心裁窗口；找不到证据句时回退为片段开头截断。"""
    if len(chunk) <= max_chars:
        return chunk
    norm_quote = _squash(quote or "")
    center: int | None = None
    if norm_quote:
        probe = 0
        while probe < len(norm_quote):
            found = chunk.find(norm_quote[probe:probe + 12])
            if found >= 0:
                center = found + len(norm_quote) // 2
                break
            probe += 12
    if center is None:
        return chunk[:max_chars]
    half = max(0, (max_chars - len(norm_quote)) // 2)
    start = max(0, center - half)
    return chunk[start:start + max_chars]


async def build_pairs(cases: list[dict], user_id: str, distractors_per_case: int, seed: int = 42) -> list[dict]:
    """构造样本；干扰片段取自金标同章节其他片段（确定性随机）。"""
    rng = random.Random(seed)
    samples = []
    for case in cases:
        gold_item = case["gold_chunks"][0]
        file_id = case.get("file_id")
        chapter_no, chunk_no = gold_item.get("chapter_no"), gold_item.get("chunk_no")
        if not file_id or chapter_no is None or chunk_no is None:
            continue
        async with AsyncSessionLocal() as session:
            sibling_stmt = (
                select(Embedding.chunk_no)
                .where(
                    Embedding.user_id == user_id,
                    Embedding.file_id == file_id,
                    Embedding.chapter_no == chapter_no,
                    Embedding.chunk_no != chunk_no,
                )
                .order_by(Embedding.chunk_no)
                .limit(200)
            )
            siblings = [row[0] for row in (await session.execute(sibling_stmt)).all()]
        picked = rng.sample(siblings, min(distractors_per_case, len(siblings))) if siblings else []
        contents = await fetch_chunks(file_id, user_id, {chapter_no: [chunk_no, *picked]})
        gold_text = contents.get((chapter_no, chunk_no))
        if not gold_text:
            continue
        quote = gold_item.get("evidence_quote")
        norm_quote = _squash(quote or "")
        norm_gold = _squash(gold_text)
        quote_pos = None
        if norm_quote:
            probe = 0
            while probe < len(norm_quote):
                found = norm_gold.find(norm_quote[probe:probe + 12])
                if found >= 0:
                    quote_pos = round(found / max(1, len(norm_gold)), 3)
                    break
                probe += 12
        samples.append({
            "id": case.get("id", "?"),
            "query": case["query"],
            "gold": gold_text,
            "gold_window": make_window(gold_text, quote),
            "distractors": [contents[(chapter_no, c)] for c in picked if (chapter_no, c) in contents],
            "quote_position": quote_pos,
        })
    return samples


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    low = int(position // 1)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def model_native_limit(model_name: str) -> int:
    """模型可用的最大 pair 长度；XLM-RoBERTa 系为 514（有效 512）。"""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name)
    return int(getattr(config, "max_position_embeddings", 512))


def evaluate_variant(
    samples: list[dict], model_name: str, max_length: int, variant: str, device: str | None,
    tokenizer,
) -> dict:
    """在指定模型/max_length/变体下打分并汇总金标区分度指标。"""
    from sentence_transformers import CrossEncoder

    text_key = {"full": "gold", "window": "gold_window"}[variant]
    pair_lengths = []
    for sample in samples:
        pair_lengths.append(len(tokenizer(sample["query"], sample[text_key], truncation=False)["input_ids"]))
    kwargs = {"max_length": max_length}
    if device:
        kwargs["device"] = device
    model = CrossEncoder(model_name, **kwargs)
    gold_scores: list[float] = []
    distractor_scores: list[float] = []
    gold_top1 = gold_top3 = judged = 0
    for sample in samples:
        text = sample[text_key]
        candidates = [(sample["query"], text, True)] + [
            (sample["query"], d, False) for d in sample["distractors"]
        ]
        if len(candidates) < 2:
            continue
        judged += 1
        scored = [float(s) for s in model.predict([(q, t) for q, t, _ in candidates])]
        ranked = sorted(zip(candidates, scored), key=lambda item: item[1], reverse=True)
        gold_rank = next(index for index, (item, _score) in enumerate(ranked, start=1) if item[2])
        for item, score in zip(candidates, scored):
            (gold_scores if item[2] else distractor_scores).append(score)
        gold_top1 += int(gold_rank == 1)
        gold_top3 += int(gold_rank <= 3)
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return {
        "model": model_name,
        "max_length": max_length,
        "variant": variant,
        "judged": judged,
        "gold_top1_rate": round(gold_top1 / judged, 4) if judged else 0.0,
        "gold_top3_rate": round(gold_top3 / judged, 4) if judged else 0.0,
        "mean_gold_score": round(statistics.mean(gold_scores), 4) if gold_scores else 0.0,
        "mean_distractor_score": round(statistics.mean(distractor_scores), 4) if distractor_scores else 0.0,
        "gold_margin": round(statistics.mean(gold_scores) - statistics.mean(distractor_scores), 4)
        if gold_scores and distractor_scores else 0.0,
        "pair_token_p50": round(percentile([float(n) for n in pair_lengths], 0.5), 1),
        "pair_token_p95": round(percentile([float(n) for n in pair_lengths], 0.95), 1),
        "pair_over_512_ratio": round(sum(1 for n in pair_lengths if n > 512) / len(pair_lengths), 4)
        if pair_lengths else 0.0,
        "score_min": round(min(gold_scores + distractor_scores), 4) if gold_scores else 0.0,
        "score_max": round(max(gold_scores + distractor_scores), 4) if gold_scores else 0.0,
    }


def build_verdict(rows: list[dict]) -> dict:
    def find(model_name: str, variant: str) -> dict | None:
        matches = [r for r in rows if r["model"] == model_name and r["variant"] == variant]
        return matches[0] if matches else None

    findings: list[str] = []
    models = sorted({row["model"] for row in rows})
    for model_name in models:
        full = find(model_name, "full")
        window = find(model_name, "window")
        if not full:
            continue
        if window:
            delta = round(window["gold_top1_rate"] - full["gold_top1_rate"], 4)
            if delta >= 0.15:
                findings.append(
                    f"{model_name}：截断实锤——证据居中窗口 top-1 {window['gold_top1_rate']} vs "
                    f"完整截断 {full['gold_top1_rate']}（{delta:+.2%}）。修复方向：证据感知窗口/分块或换长上下文重排模型。"
                )
            elif full["gold_top1_rate"] < 0.5 and window["gold_top1_rate"] < 0.5:
                findings.append(
                    f"{model_name}：模型本身偏弱（full={full['gold_top1_rate']}, window={window['gold_top1_rate']} 均低），"
                    "截断不是主因，需换更强的重排模型或放弃重排路线。"
                )
            else:
                findings.append(
                    f"{model_name}：截断影响有限（window vs full top-1 变化 {delta:+.2%}），"
                    "“重排恒负”需另找原因（候选池构成/分数融合/数据集难度）。"
                )
        else:
            findings.append(f"{model_name}：full top-1={full['gold_top1_rate']}。")
    if rows and all(row["score_min"] >= 0.0 and row["score_max"] <= 1.0 for row in rows):
        findings.append(
            "模型输出已在 [0,1]：sentence-transformers 内置 sigmoid 生效，rerank.py:94 再套一层属于双重压缩"
            "（单调变换不改排序，但压扁 blend 权重的绝对量纲）。"
        )
    return {"findings": findings}


def markdown_report(payload: dict) -> str:
    lines = [
        "# Cross-encoder 重排器健全性测试",
        "",
        f"生成时间：{payload['generated_at']}",
        f"数据集：`{payload['dataset']}`｜题数：{payload['cases']}｜干扰片段/题：{payload['distractors_per_case']}",
        "",
        "| 模型 | max_length | 变体 | 金标 top-1 | 金标 top-3 | 金标均分 | 干扰均分 | 分差 | pair token p50/p95 | 超512占比 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['model']} | {row['max_length']} | {row['variant']} | {row['gold_top1_rate']} "
            f"| {row['gold_top3_rate']} | {row['mean_gold_score']} | {row['mean_distractor_score']} "
            f"| {row['gold_margin']} | {row['pair_token_p50']}/{row['pair_token_p95']} | {row['pair_over_512_ratio']} |"
        )
    if payload.get("model_limits"):
        lines += ["", "模型原生 pair 长度上限：`" + json.dumps(payload["model_limits"], ensure_ascii=False) + "`"]
    lines += ["", "## 结论", ""]
    lines += [f"- {item}" for item in payload["verdict"]["findings"]]
    return "\n".join(lines) + "\n"


async def run(args: argparse.Namespace) -> dict:
    from transformers import AutoTokenizer

    cases = load_cases(Path(args.dataset))
    if args.limit:
        cases = cases[: args.limit]
    print(f"加载数据集 {args.dataset}：{len(cases)} 题，构造 (query, 金标, 干扰) 样本…")
    samples = await build_pairs(cases, args.user_id, args.distractors)
    print(f"有效样本：{len(samples)}")
    quote_positions = [s["quote_position"] for s in samples if s["quote_position"] is not None]
    if quote_positions:
        print(
            f"证据句在片段内的相对位置：p50={percentile(quote_positions, 0.5):.2f} "
            f"p95={percentile(quote_positions, 0.95):.2f}"
        )
    model_limits = {}
    rows = []
    for model_name in args.models:
        limit = model_native_limit(model_name)
        model_limits[model_name] = limit
        effective = min(limit - 2, 512)  # 514 上限 → 有效 512
        if any(int(x) > effective for x in args.max_lengths.split(",")) and effective < 512:
            print(f"⚠️ {model_name} 基于 XLM-RoBERTa（上限 {limit} token），无法处理 >512 token 的 pair；")
            print("   650 字小说块在该模型下必然被截断——这本身就是一个结论。")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        for raw_length in [int(x) for x in args.max_lengths.split(",")]:
            max_length = min(raw_length, limit - 2)
            for variant in ("full", "window"):
                if max_length <= 512 and variant == "window" and max_length != min(raw_length, limit - 2):
                    continue  # 已跑过等效长度
                print(f"  {model_name} max_length={max_length} variant={variant} …")
                rows.append(evaluate_variant(samples, model_name, max_length, variant, args.device, tokenizer))
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(Path(args.dataset).resolve()),
        "cases": len(samples),
        "distractors_per_case": args.distractors,
        "model_limits": model_limits,
        "rows": rows,
        "verdict": build_verdict(rows),
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.name
    (out_dir / f"{stem}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / f"{stem}.md").write_text(markdown_report(payload), encoding="utf-8")
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))
    for item in payload["verdict"]["findings"]:
        print("结论：", item)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证重排器 512 token 截断假设")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--user-id", required=True, help="索引所属用户 ID（多租户行级隔离）")
    parser.add_argument("--limit", type=int, default=0, help="只取前 N 题（0=全部）")
    parser.add_argument("--distractors", type=int, default=4)
    parser.add_argument("--max-lengths", default="512", help="各模型测试的 pair 长度；超过模型上限会自动钳制")
    parser.add_argument(
        "--models", default=None,
        help="逗号分隔的重排模型列表，默认 [settings.reranker_model]；可追加 BAAI/bge-reranker-v2-m3 验证长上下文模型",
    )
    parser.add_argument("--device", default=None, help="cpu/cuda，默认跟随 torch auto")
    parser.add_argument("--output-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--name", default=None, help="报告文件名，默认 reranker_sanity_<日期>")
    return parser.parse_args()


if __name__ == "__main__":
    import asyncio

    args = parse_args()
    if not args.models:
        args.models = [settings.reranker_model]
    else:
        args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not args.name:
        args.name = f"reranker_sanity_{datetime.now(timezone.utc).strftime('%Y%m%d')}"
    asyncio.run(run(args))
