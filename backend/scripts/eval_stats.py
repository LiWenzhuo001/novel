# -*- coding: utf-8 -*-
"""评测共享统计工具：配对 bootstrap 置信区间与 win/tie/loss。

n=20/40 的小样本评测中，0.05~0.10 的指标差可能只是抽样噪声；所有 A/B 结论
（策略对比、融合参数对比、检索通道对比）都必须带置信区间再下判断。
只依赖标准库，供 evaluate_rag_recall / evaluate_agent_answer 等脚本复用。
"""
from __future__ import annotations

import random
import statistics
from typing import Sequence


def paired_bootstrap_ci(
    pairs: Sequence[tuple[float, float]],
    *,
    n_boot: int = 10000,
    seed: int = 42,
) -> dict:
    """对逐题配对的 (baseline, treatment) 分差做配对 bootstrap。

    返回 baseline/treatment 均值、差值均值、差值的 95% CI 与是否显著（CI 不含 0）。
    pairs 为空时返回全 None 字段；单题时 CI 退化为该题差值。
    """
    if not pairs:
        return {"n": 0, "mean_a": None, "mean_b": None, "delta": None, "ci95": None, "significant": None}
    deltas = [b - a for a, b in pairs]
    n = len(deltas)
    rng = random.Random(seed)
    boot_means: list[float] = []
    for _ in range(n_boot):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        boot_means.append(statistics.fmean(sample))
    boot_means.sort()
    ci_low = boot_means[int(0.025 * (n_boot - 1))]
    ci_high = boot_means[int(0.975 * (n_boot - 1))]
    mean_a = statistics.fmean([a for a, _ in pairs])
    mean_b = statistics.fmean([b for _, b in pairs])
    return {
        "n": n,
        "mean_a": round(mean_a, 4),
        "mean_b": round(mean_b, 4),
        "delta": round(mean_b - mean_a, 4),
        "ci95": [round(ci_low, 4), round(ci_high, 4)],
        "significant": ci_low > 0 or ci_high < 0,
    }


def win_tie_loss(pairs: Sequence[tuple[float, float]], *, tol: float = 1e-9) -> dict:
    """逐题统计 treatment 相对 baseline 的胜/平/负。"""
    wins = ties = losses = 0
    for a, b in pairs:
        if b - a > tol:
            wins += 1
        elif a - b > tol:
            losses += 1
        else:
            ties += 1
    return {"wins": wins, "ties": ties, "losses": losses}
