"""答案侧评测工具的单元测试：引用校验、judge 解析、分层抽样、bootstrap 统计。

这些脚本位于 backend/scripts 与 evals/datasets（非包），测试通过 sys.path 直连导入；
只测纯逻辑，不触 DB 与真实 LLM。
"""
import asyncio
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
EVAL_DATASETS_DIR = BACKEND_DIR.parent / "evals" / "datasets"
for path in (BACKEND_DIR, BACKEND_DIR / "scripts", EVAL_DATASETS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval_stats import paired_bootstrap_ci, win_tie_loss  # noqa: E402
from evaluate_agent_answer import (  # noqa: E402
    JudgeVerdict,
    _judge_cache_key,
    citation_metrics,
    extract_citations,
    judge_case,
)
from build_answer_dataset import key_point_supported, stratified_subset  # noqa: E402


# ---------- 引用提取与落地校验 ----------

def test_extract_citations_dedup_and_order():
    answer = "出生在第一回 [S2]，称王也在第一回 [S1]，见 [S2] 与 [S10]。"
    assert extract_citations(answer) == ["S2", "S1", "S10"]


def test_extract_citations_empty():
    assert extract_citations("") == []
    assert extract_citations("没有引用的答案 [S99x]") == []


def test_citation_metrics_unknown_and_gold():
    sources = [
        {"id": "S1", "chapter_no": 1, "chunk_no": 10, "neighbor": False},
        {"id": "S2", "chapter_no": 1, "chunk_no": 11, "neighbor": True},
    ]
    gold = [{"chapter_no": 1, "chunk_no": 10, "relevance": 2}]
    result = citation_metrics("甲 [S1] 乙 [S3] 丙 [S2]", sources, gold)
    assert result["citations"] == ["S1", "S3", "S2"]
    assert result["unknown_citations"] == ["S3"]
    # S2 是 neighbor，不计入金标引用率；S1 命中金标 → 1/1
    assert result["gold_citation_rate"] == 1.0
    assert result["unknown_citation_rate"] == 0.0


def test_citation_metrics_all_unknown():
    result = citation_metrics("凭空引用 [S9]", [{"id": "S1", "chapter_no": 2, "chunk_no": 3}], [])
    assert result["unknown_citation_rate"] == 1.0
    assert result["gold_citation_rate"] == 0.0


# ---------- judge 解析与缓存 ----------

def test_judge_verdict_clamps_to_1_5():
    verdict = JudgeVerdict(faithfulness=0, completeness=7, relevance=3, citation_support=5)
    assert verdict.faithfulness == 1
    assert verdict.completeness == 5
    assert verdict.relevance == 3


def test_judge_cache_key_depends_on_answer():
    base = {"id": "xyj-001", "strategy": "direct", "answer": "答案A"}
    other = {"id": "xyj-001", "strategy": "direct", "answer": "答案B"}
    assert _judge_cache_key(base) == _judge_cache_key(dict(base))
    assert _judge_cache_key(base) != _judge_cache_key(other)


def test_judge_case_uses_cache_without_llm():
    raw = {"id": "xyj-001", "strategy": "direct", "answer": "答案A"}
    key = _judge_cache_key(raw)
    cache = {key: {"faithfulness": 4, "completeness": 4, "relevance": 4, "citation_support": 4}}
    called = False

    class ExplodingLLM:
        async def ainvoke(self, messages):
            nonlocal called
            called = True
            raise AssertionError("cache 命中时不应调用 LLM")

    verdict = asyncio.run(judge_case(ExplodingLLM(), {"query": "q"}, raw, cache))
    assert called is False
    assert verdict["cached"] is True
    assert verdict["faithfulness"] == 4


# ---------- 分层抽样与 key_point 校验 ----------

def test_stratified_subset_deterministic_and_sized():
    cases = (
        [{"id": f"c-{cat}-{i:02d}", "category": cat} for i in range(10) for cat in ("a", "b", "c", "d")]
    )
    subset = stratified_subset(cases, 20)
    assert len(subset) == 20
    assert subset == stratified_subset(cases, 20)
    counts = {cat: sum(1 for c in subset if c["category"] == cat) for cat in ("a", "b", "c", "d")}
    assert set(counts.values()) == {5}
    # size >= 总数时全量返回
    assert len(stratified_subset(cases, 100)) == len(cases)


def test_key_point_supported_anchor_and_digit_rules():
    chunk = "贾母说软烟罗共有四种颜色，那银红的又叫作霞影纱。如今做窗屉用了 2 匹。"
    # 锚点：长中文连写在片段中
    assert key_point_supported("银红的又叫作霞影纱", chunk) is True
    # 数字必须出现
    assert key_point_supported("用了 3 匹", chunk) is False
    assert key_point_supported("用了 2 匹", chunk) is True
    # 无锚点且 bigram 覆盖率过低
    assert key_point_supported("完全无关的内容词串", chunk) is False


# ---------- bootstrap 统计 ----------

def test_paired_bootstrap_ci_constant_delta_is_significant():
    pairs = [(1.0, 2.0)] * 20
    result = paired_bootstrap_ci(pairs, n_boot=2000)
    assert result["delta"] == 1.0
    assert result["ci95"] == [1.0, 1.0]
    assert result["significant"] is True


def test_paired_bootstrap_ci_zero_delta_not_significant():
    pairs = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    result = paired_bootstrap_ci(pairs, n_boot=2000)
    assert result["significant"] is False


def test_paired_bootstrap_ci_empty():
    assert paired_bootstrap_ci([])["n"] == 0


def test_win_tie_loss():
    assert win_tie_loss([(1, 2), (2, 1), (3, 3), (1, 2)]) == {"wins": 2, "ties": 1, "losses": 1}
