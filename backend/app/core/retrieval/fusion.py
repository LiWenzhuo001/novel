"""融合层：RRF 融合、min-max 归一化与加权融合。「

!!! 业务代码请从 app.core.rag 导入。「"""

from __future__ import annotations

from typing import List

from app.db.models import Embedding


def _rrf_fuse(vector_results: list, lexical_results: list, k: int, rrf_c: int = 60) -> List[tuple]:
    """用 Reciprocal Rank Fusion 合并向量和词法候选，只把 RRF 用作排序信号。"""
    scores: dict[str, float] = {}
    rows_by_id: dict[str, Embedding] = {}
    for results in (vector_results, lexical_results):
        for rank_index, (row, _) in enumerate(results):
            scores[row.id] = scores.get(row.id, 0.0) + 1.0 / (rrf_c + rank_index + 1)
            rows_by_id[row.id] = row
    fused = [(rows_by_id[row_id], score) for row_id, score in scores.items()]
    fused.sort(key=lambda item: item[1], reverse=True)
    return fused[:k]


def _minmax_normalize(channel: dict[str, float]) -> dict[str, float]:
    """把单通道分数在合并候选集内做 min-max 归一；仅出现在该通道的候选参与边界。"""
    if not channel:
        return {}
    low, high = min(channel.values()), max(channel.values())
    if high <= low:
        return {row_id: 1.0 for row_id in channel}
    span = high - low
    return {row_id: (value - low) / span for row_id, value in channel.items()}


def _weighted_fuse(
    vector_results: list, lexical_results: list, vector_weight: float, lexical_weight: float,
) -> List[tuple]:
    """归一化加权融合：各通道原始分数池内 min-max 归一后加权求和。

    RRF 只用名次，会丢弃“两通道都把某候选排第 1”和“险排第 N”的差距；
    评测已定位瓶颈在池内排序（evals/README §四.3），此模式让通道相关度参与排序。
    单通道候选在缺失通道计 0 分——被双通道同时召回的候选天然占优。
    """
    vector_scores = {row.id: float(score) for row, score in vector_results}
    lexical_scores = {row.id: float(score) for row, score in lexical_results}
    rows_by_id: dict[str, Embedding] = {}
    for row, _ in [*vector_results, *lexical_results]:
        rows_by_id.setdefault(row.id, row)
    norm_vector = _minmax_normalize(vector_scores)
    norm_lexical = _minmax_normalize(lexical_scores)
    fused = [
        (row, vector_weight * norm_vector.get(row_id, 0.0) + lexical_weight * norm_lexical.get(row_id, 0.0))
        for row_id, row in rows_by_id.items()
    ]
    fused.sort(key=lambda item: item[1], reverse=True)
    return fused
