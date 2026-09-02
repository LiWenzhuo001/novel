# -*- coding: utf-8 -*-
"""端到端答案侧评测：multi_expert vs direct A/B + LLM-as-judge + 引用落地校验。

检索侧评测（evaluate_rag_recall.py）止步于召回；本脚本回答“agent 最终答案的质量”：

  run 阶段   冻结 Query（preparation cache）→ 无头运行 agent_graph（每个策略）
             → 收集最终答案、S# 引用、证据、meta → 原始输出落盘 *_raw.jsonl
  judge 阶段 LLM-as-judge 按 faithfulness / completeness / relevance / citation_support
             四维打分（judge 缓存冻结，可重跑不重跑 agent）→ 聚合 .json/.md 报告

引用落地校验不依赖 LLM：提取答案中的 [S#]，检测未知引用（幻觉引用）、
引用指向金标片段的比例。A/B 差值一律带配对 bootstrap 95% CI。

局限（写入报告）：judge 默认与被评 agent 同模型（自评偏差）；n=20/策略/数据集，
结论看 CI 而非点值。
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from pydantic import BaseModel, Field, field_validator  # noqa: E402

from app.config import settings  # noqa: E402
from app.core.context import reset_current_user, set_current_user  # noqa: E402
from app.core.llm import get_llm  # noqa: E402

DEFAULT_REPORT_DIR = Path(__file__).resolve().parents[2] / "evals" / "reports"
DEFAULT_DATASET_DIR = Path(__file__).resolve().parents[2] / "evals" / "datasets"
STRATEGY_ORDER = ("direct", "multi_expert")
_CITATION_RE = re.compile(r"\[S(\d+)\]")
_JUDGE_DIMS = ("faithfulness", "completeness", "relevance", "citation_support")


class JudgeVerdict(BaseModel):
    """LLM-as-judge 的严格 JSON 结构（1~5 分制）。"""

    faithfulness: int
    completeness: int
    relevance: int
    citation_support: int
    missing_key_points: list[str] = Field(default_factory=list)
    hallucination_flags: list[str] = Field(default_factory=list)
    rationale: str = ""

    @field_validator("faithfulness", "completeness", "relevance", "citation_support")
    @classmethod
    def _clamp(cls, value: int) -> int:
        return max(1, min(5, int(value)))


def load_answer_dataset(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            rows[str(item["id"])] = item
    if not rows:
        raise ValueError(f"答案数据集为空：{path}")
    return rows


def _load_preparation_cache(path: str) -> dict[str, dict[str, str]]:
    """复用 evaluate_rag_recall 的冻结 Query 快照（同口径校验）。"""
    from evaluate_rag_recall import load_preparation_cache

    payload = load_preparation_cache(Path(path))
    return {
        case_id: {
            "standalone_query": item["standalone_query"],
            "retrieval_query": item["retrieval_query"],
        }
        for case_id, item in payload["cases"].items()
    }


def extract_citations(answer: str) -> list[str]:
    """按出现顺序去重提取答案中的 [S#] 引用。"""
    seen: list[str] = []
    for match in _CITATION_RE.findall(answer or ""):
        token = f"S{match}"
        if token not in seen:
            seen.append(token)
    return seen


def citation_metrics(answer: str, sources: list[dict], gold_chunks: list[dict]) -> dict:
    """引用落地校验（纯代码，无 LLM）：
    unknown_citations = 引用了不存在的 S#（幻觉引用）；
    gold_citation_rate = 引用中指向金标片段的比例（relevance>=2）。
    """
    evidence_ids = {str(source.get("id")) for source in sources}
    gold_keys = {
        (item.get("chapter_no"), item.get("chunk_no"))
        for item in gold_chunks or []
        if int(item.get("relevance", 1) or 1) >= 2
    }
    citations = extract_citations(answer)
    unknown = [c for c in citations if c not in evidence_ids]
    by_id = {str(source.get("id")): source for source in sources}
    gold_hits = 0
    checked = 0
    for citation in citations:
        source = by_id.get(citation)
        if not source or source.get("neighbor"):
            continue
        checked += 1
        if (source.get("chapter_no"), source.get("chunk_no")) in gold_keys:
            gold_hits += 1
    return {
        "citations": citations,
        "citation_count": len(citations),
        "unknown_citations": unknown,
        "unknown_citation_rate": 1.0 if citations and len(unknown) == len(citations) else 0.0,
        "gold_citation_rate": round(gold_hits / checked, 4) if checked else 0.0,
    }


async def _fetch_evidence_content(file_id: str, user_id: str, sources: list[dict]) -> dict[str, str]:
    """observation 事件缺失时，按 (chapter_no, chunk_no) 从 DB 回读证据正文。"""
    from sqlalchemy import select

    from app.db import AsyncSessionLocal
    from app.db.models import Embedding

    wanted = {
        (source.get("chapter_no"), source.get("chunk_no"))
        for source in sources
        if source.get("chapter_no") is not None and source.get("chunk_no") is not None
    }
    if not wanted or not file_id:
        return {}
    content_map: dict[str, str] = {}
    async with AsyncSessionLocal() as session:
        for chapter_no, chunk_no in wanted:
            stmt = select(Embedding.content).where(
                Embedding.user_id == user_id,
                Embedding.file_id == file_id,
                Embedding.chapter_no == chapter_no,
                Embedding.chunk_no == chunk_no,
            )
            content = (await session.execute(stmt)).scalar_one_or_none()
            if content:
                content_map[f"{chapter_no}:{chunk_no}"] = content
    return content_map


async def run_case(case: dict, strategy: str, args: argparse.Namespace, preparation: dict[str, dict]) -> dict:
    """无头运行一次 agent，收集答案/引用/证据/meta。"""
    from app.agent.runtime import stream_agent_question

    case_id = str(case["id"])
    file_id = case.get("file_id") or args.file_id
    cached = preparation.get(case_id)
    standalone = cached["standalone_query"] if cached else case["query"]
    retrieval = cached["retrieval_query"] if cached else case["query"]

    answer = ""
    sources: list[dict] = []
    evidence: list[dict] = []
    meta: dict = {}
    error = None
    started = time.perf_counter()
    try:
        async with asyncio.timeout(args.case_timeout):
            async for event in stream_agent_question(
                query=standalone,
                strategy=strategy,
                file_id=file_id,
                original_query=case["query"],
                retrieval_query=retrieval,
                memory_context={},
            ):
                event_type, data = event.get("type"), event.get("data") or {}
                if event_type == "token":
                    answer = data if isinstance(data, str) else str(data)
                elif event_type == "sources":
                    sources = data if isinstance(data, list) else data.get("sources", [])
                elif event_type == "observation":
                    output = data.get("output") if isinstance(data.get("output"), dict) else {}
                    if output.get("evidence"):
                        evidence = output["evidence"]
                elif event_type == "meta":
                    meta = data
                elif event_type == "error":
                    error = data.get("message", "agent_error")
    except TimeoutError:
        error = f"case_timeout_{args.case_timeout}s"
    latency_ms = round((time.perf_counter() - started) * 1000, 2)

    if not evidence and sources:
        content_map = await _fetch_evidence_content(file_id, args.user_id, sources)
        evidence = [
            {"source": source, "content": content_map.get(f"{source.get('chapter_no')}:{source.get('chunk_no')}", "")}
            for source in sources
        ]
    reports = meta.get("reports") or []
    llm_calls = 1  # summary
    if strategy == "multi_expert":
        llm_calls += 1 + len(reports) + sum(1 for r in reports if r.get("corrected"))
    citations_info = citation_metrics(answer, sources, case.get("gold_chunks", []))
    return {
        "id": case_id,
        "category": case.get("category", "unknown"),
        "query": case["query"],
        "standalone_query": standalone,
        "retrieval_query": retrieval,
        "strategy": strategy,
        "answer": answer,
        "sources": [
            {
                "id": s.get("id"),
                "file_id": s.get("file_id"),
                "chapter_no": s.get("chapter_no"),
                "chunk_no": s.get("chunk_no"),
                "neighbor": bool(s.get("neighbor")),
                "retrieval_rank": s.get("retrieval_rank"),
            }
            for s in sources
        ],
        "evidence": [
            {
                "id": (item.get("source") or {}).get("id"),
                "chapter_no": (item.get("source") or {}).get("chapter_no"),
                "chunk_no": (item.get("source") or {}).get("chunk_no"),
                "content": item.get("content", ""),
            }
            for item in evidence
        ],
        **citations_info,
        "latency_ms": latency_ms,
        "llm_calls": llm_calls,
        "fallback_reason": meta.get("fallback_reason", ""),
        "answer_mode": meta.get("answer_mode", ""),
        "dispatch_mode": meta.get("dispatch_mode"),
        "expert_count": meta.get("expert_count", 0),
        "report_statuses": [r.get("status") for r in reports],
        "error": error,
    }


_JUDGE_SYSTEM = "你是严格的小说问答质量评审。只依据给出的材料评审，不引入外部知识。只输出 JSON。"


def _judge_prompt(case: dict, raw: dict) -> str:
    evidence_text = "\n\n".join(
        f"[{item.get('id') or 'S?'}] 第{item.get('chapter_no')}回/片段{item.get('chunk_no')}："
        f"{(item.get('content') or '')[:600]}"
        for item in raw.get("evidence", [])
    ) or "（无证据）"
    key_points = "\n".join(f"- {kp}" for kp in case.get("key_points", []) or []) or "（无）"
    return (
        f"问题：{case['query']}\n\n"
        f"参考答案（金标）：{case.get('reference_answer', '')}\n\n"
        f"关键点（候选答案应覆盖）：\n{key_points}\n\n"
        f"候选答案（待评）：\n{raw.get('answer', '')}\n\n"
        f"候选答案引用的证据（agent 检索所得）：\n{evidence_text}\n\n"
        "请逐维评分（1~5 整数）：\n"
        "- faithfulness 忠实性：候选答案的论断是否都被证据支撑，有无编造（5=全部有据，1=大量编造）\n"
        "- completeness 完整性：是否覆盖关键点与参考答案的要点（5=全覆盖，1=几乎未覆盖）\n"
        "- relevance 相关性：是否切题回答了问题（5=直接准确，1=答非所问）\n"
        "- citation_support 引用支撑：[S#] 引用是否真实支撑对应论断；无引用或引用与论断不符记 1 分\n"
        "另列出 missing_key_points（未覆盖的关键点原文）、hallucination_flags（编造论断原文）。"
        "输出 JSON：{\"faithfulness\": n, \"completeness\": n, \"relevance\": n, "
        "\"citation_support\": n, \"missing_key_points\": [...], \"hallucination_flags\": [...], "
        "\"rationale\": \"一句话总评\"}"
    )


def _judge_cache_key(raw: dict) -> str:
    digest = hashlib.sha1((raw.get("answer") or "").encode("utf-8")).hexdigest()[:16]
    return f"{raw['id']}|{raw['strategy']}|{digest}"


def _extract_json(content: object) -> dict:
    text = content if isinstance(content, str) else str(content)
    fenced = re.search(r"\{[\s\S]*\}", text.strip())
    if not fenced:
        raise ValueError("json_object_not_found")
    return json.loads(fenced.group(0))


async def judge_case(llm, case: dict, raw: dict, cache: dict) -> dict:
    key = _judge_cache_key(raw)
    if key in cache:
        return {**cache[key], "cached": True}
    verdict = None
    last_error = None
    for _attempt in range(3):
        try:
            response = await llm.ainvoke([
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": _judge_prompt(case, raw)},
            ])
            verdict = JudgeVerdict.model_validate(_extract_json(response.content))
            break
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"[:200]
            await asyncio.sleep(1.0)
    if verdict is None:
        return {
            "faithfulness": 0, "completeness": 0, "relevance": 0, "citation_support": 0,
            "missing_key_points": [], "hallucination_flags": [],
            "rationale": f"judge_failed: {last_error}", "cached": False,
        }
    data = verdict.model_dump()
    cache[key] = data
    return {**data, "cached": False}


def summarize_strategy(rows: list[dict]) -> dict:
    def mean(field: str) -> float:
        values = [float(r[field]) for r in rows if r.get(field) is not None]
        return round(statistics.mean(values), 4) if values else 0.0

    latencies = [float(r["latency_ms"]) for r in rows]
    return {
        "questions": len(rows),
        "faithfulness": mean("faithfulness"),
        "completeness": mean("completeness"),
        "relevance": mean("relevance"),
        "citation_support": mean("citation_support"),
        "gold_citation_rate": mean("gold_citation_rate"),
        "unknown_citation_case_rate": round(
            sum(1 for r in rows if r.get("unknown_citations")) / len(rows), 4
        ) if rows else 0.0,
        "empty_answer_rate": round(sum(1 for r in rows if not (r.get("answer") or "").strip()) / len(rows), 4) if rows else 0.0,
        "error_rate": round(sum(1 for r in rows if r.get("error")) / len(rows), 4) if rows else 0.0,
        "avg_latency_ms": round(statistics.mean(latencies), 1) if latencies else 0.0,
        "p95_latency_ms": round(percentile(latencies, 0.95), 1) if latencies else 0.0,
        "avg_llm_calls": mean("llm_calls"),
        "total_llm_calls": sum(int(r.get("llm_calls") or 0) for r in rows),
    }


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    low = int(position // 1)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def build_comparison(rows_by_strategy: dict[str, list[dict]]) -> dict:
    """direct 为 baseline 的配对对比，含 bootstrap 95% CI。"""
    from eval_stats import paired_bootstrap_ci, win_tie_loss

    if "direct" not in rows_by_strategy or len(rows_by_strategy) < 2:
        return {}
    baseline = {r["id"]: r for r in rows_by_strategy["direct"]}
    comparison: dict[str, Any] = {}
    for strategy, rows in rows_by_strategy.items():
        if strategy == "direct":
            continue
        matched = [
            (baseline[r["id"]], r)
            for r in rows
            if r["id"] in baseline and not (baseline[r["id"]].get("error") or r.get("error"))
        ]
        strategy_result: dict[str, Any] = {"matched_pairs": len(matched)}
        for dim in (*_JUDGE_DIMS, "gold_citation_rate", "latency_ms"):
            pairs = [(float(a.get(dim) or 0.0), float(b.get(dim) or 0.0)) for a, b in matched]
            bootstrap = paired_bootstrap_ci(pairs)
            bootstrap["win_tie_loss"] = win_tie_loss(pairs) if dim in _JUDGE_DIMS else None
            strategy_result[dim] = bootstrap
        comparison[strategy] = strategy_result
    return comparison


def markdown_report(payload: dict) -> str:
    lines = [
        f"# 答案侧评测：{' vs '.join(payload['strategies'])}",
        "",
        f"生成时间：{payload['generated_at']}｜数据集：`{payload['dataset']}`｜题数：{payload['questions']}",
        f"judge 模型：`{payload['judge_model']}`（与 agent 模型 {'相同，存在自评偏差' if payload['judge_model'] == payload['agent_model'] else '不同'}）",
        "",
        "## 各策略指标",
        "",
        "| 策略 | faithfulness | completeness | relevance | citation_support | 金标引用率 | 幻觉引用题率 | P95延迟ms | 平均LLM调用 | 总LLM调用 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy in payload["strategies"]:
        m = payload["strategy_metrics"][strategy]
        lines.append(
            f"| {strategy} | {m['faithfulness']} | {m['completeness']} | {m['relevance']} "
            f"| {m['citation_support']} | {m['gold_citation_rate']} | {m['unknown_citation_case_rate']} "
            f"| {m['p95_latency_ms']} | {m['avg_llm_calls']} | {m['total_llm_calls']} |"
        )
    comparison = payload.get("comparison") or {}
    if comparison:
        lines += ["", "## 策略对比（multi_expert - direct，配对 bootstrap 95% CI）", ""]
        for strategy, dims in comparison.items():
            lines.append(f"### {strategy} vs direct（配对 {dims['matched_pairs']} 题）")
            lines += ["", "| 维度 | direct | " + strategy + " | 差值 | 95% CI | 显著 | 胜/平/负 |", "|---|---:|---:|---:|---|---|---|"]
            for dim, result in dims.items():
                if not isinstance(result, dict) or "delta" not in result:
                    continue
                wtl = result.get("win_tie_loss")
                wtl_text = f"{wtl['wins']}/{wtl['ties']}/{wtl['losses']}" if wtl else "-"
                lines.append(
                    f"| {dim} | {result['mean_a']} | {result['mean_b']} | {result['delta']} "
                    f"| [{result['ci95'][0]}, {result['ci95'][1]}] | {'是' if result['significant'] else '否'} | {wtl_text} |"
                )
            lines.append("")
    lines += ["", "## 局限", ""]
    lines += [f"- {item}" for item in payload["limitations"]]
    return "\n".join(lines) + "\n"


async def run(args: argparse.Namespace) -> dict:
    answer_rows = load_answer_dataset(Path(args.dataset))
    cases = list(answer_rows.values())
    if args.limit:
        cases = cases[: args.limit]
    preparation = _load_preparation_cache(args.preparation_cache) if args.preparation_cache else {}
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    unknown = [s for s in strategies if s not in STRATEGY_ORDER]
    if unknown:
        raise ValueError(f"未知策略：{unknown}，支持 {STRATEGY_ORDER}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"{args.name}_raw.jsonl"
    user_token = set_current_user(args.user_id)
    try:
        if args.stage in {"run", "all"}:
            # 预热两件事，避免冷启动开销吃掉检索工具的 20s 超时预算：
            # 1) 首次 DB 连接：POSTGRES_HOST=localhost 时 Windows 先试 IPv6 ::1，
            #    被静默丢弃重试约 21s 后才回落 127.0.0.1（第二个连接仅 6ms）；
            # 2) 首次 embed_query：本地 bge-m3 现场加载约 10s。
            from sqlalchemy import text

            from app.db import AsyncSessionLocal
            from app.core.embed import get_embeddings

            async with AsyncSessionLocal() as session:
                await session.execute(text("select 1"))
            await asyncio.to_thread(get_embeddings().embed_query, "预热")
            raw_rows: dict[tuple[str, str], dict] = {}
            for strategy in strategies:
                for index, case in enumerate(cases, start=1):
                    raw = await run_case(case, strategy, args, preparation)
                    raw_rows[(raw["id"], strategy)] = raw
                    print(
                        f"[{strategy} {index:02d}/{len(cases):02d}] {raw['id']} "
                        f"answer={len(raw['answer'])}字 citations={raw['citation_count']} "
                        f"latency={raw['latency_ms']}ms calls={raw['llm_calls']}"
                        + (f" ERROR={raw['error']}" if raw["error"] else "")
                    )
            raw_path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in raw_rows.values()) + "\n",
                encoding="utf-8",
            )
            print(f"原始输出已写入 {raw_path}")
        if args.stage in {"judge", "all"}:
            if not raw_path.is_file():
                raise RuntimeError(f"找不到原始输出 {raw_path}，先跑 --stage run")
            raw_rows = [
                json.loads(line)
                for line in raw_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            cache_path = Path(args.judge_cache) if args.judge_cache else out_dir / f"{args.name}_judge_cache.json"
            cache: dict = {}
            if cache_path.is_file():
                cache = json.loads(cache_path.read_text(encoding="utf-8"))
            judge_llm = get_llm(model=args.judge_model or settings.judge_model, temperature=0, max_tokens=500)
            rows_by_strategy: dict[str, list[dict]] = defaultdict(list)
            for raw in raw_rows:
                case = answer_rows[raw["id"]]
                verdict = await judge_case(judge_llm, case, raw, cache)
                merged = {**raw, **verdict}
                rows_by_strategy[raw["strategy"]].append(merged)
            cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

            strategy_metrics = {s: summarize_strategy(rows) for s, rows in rows_by_strategy.items()}
            payload = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "dataset": str(Path(args.dataset).resolve()),
                "questions": len(cases),
                "strategies": [s for s in STRATEGY_ORDER if s in rows_by_strategy],
                "judge_model": args.judge_model or settings.judge_model,
                "agent_model": settings.llm_model,
                "strategy_metrics": strategy_metrics,
                "comparison": build_comparison(dict(rows_by_strategy)),
                "limitations": [
                    "judge 默认与被评 agent 同模型（DeepSeek），存在自评偏差；可用 --judge-model / JUDGE_MODEL 换模型重评（原始输出已落盘，无需重跑 agent）",
                    f"每策略仅 {len(cases)} 题，差值以 95% CI 为准，不显著时应视为打平",
                    "金标答案为 LLM 起草（依据金标片段，经自动溯源校验、剔除不可溯源关键点），未经人工复核",
                    "citation_support 维度：答案无 [S#] 引用或引用与论断不符记 1 分",
                ],
                "cases": [
                    {k: v for k, v in row.items() if k != "evidence"}
                    for rows in rows_by_strategy.values()
                    for row in rows
                ],
            }
            (out_dir / f"{args.name}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (out_dir / f"{args.name}.md").write_text(markdown_report(payload), encoding="utf-8")
            print(json.dumps(strategy_metrics, ensure_ascii=False, indent=2))
            return payload
        return {}
    finally:
        reset_current_user(user_token)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="答案侧端到端评测（run/judge 两阶段）")
    parser.add_argument("--dataset", required=True, help="答案评测集 JSONL（build_answer_dataset.py 产物）")
    parser.add_argument("--file-id", default=None, help="数据集行缺 file_id 时的兜底索引")
    parser.add_argument("--user-id", required=True, help="索引所属用户 ID")
    parser.add_argument("--strategies", default="direct,multi_expert")
    parser.add_argument("--stage", choices=["run", "judge", "all"], default="all")
    parser.add_argument("--preparation-cache", default=None, help="冻结 Query 快照（A/B 必须同口径）")
    parser.add_argument("--judge-cache", default=None)
    parser.add_argument("--judge-model", default=None, help="覆盖 settings.judge_model")
    parser.add_argument("--case-timeout", type=int, default=300, help="单题 agent 运行超时（秒）")
    parser.add_argument("--limit", type=int, default=0, help="调试：只取前 N 题")
    parser.add_argument("--output-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--name", required=True, help="报告/原始输出文件名前缀")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
