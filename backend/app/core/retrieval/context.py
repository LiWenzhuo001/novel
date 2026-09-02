"""上下文层：主命中文档补齐、章节局部精排与邻居扩展。「

!!! 业务代码请从 app.core.rag 导入。「"""

from __future__ import annotations

import asyncio

from typing import List

from langchain_core.documents import Document
from sqlalchemy import select

from app.config import settings
from app.core.context import get_current_user
from app.core.embed import get_embeddings
from app.core.logging_config import get_logger
from app.core.retrieval.channels import _parse_meta
from app.core.visibility import visible_user_filter
from app.db import AsyncSessionLocal
from app.db.models import Embedding

log = get_logger("rag")


def _annotate_primary(docs: list[Document]) -> list[Document]:
    """为主命中文档补齐 score_type、排名和各通道分数元数据。"""
    for rank, doc in enumerate(docs, start=1):
        doc.metadata["retrieval_rank"] = rank
        doc.metadata["neighbor"] = False
        doc.metadata.setdefault("score_type", "vector")
    return docs


async def chapter_local_refine(
    query: str,
    primary: list[Document],
    k: int,
    file_id: str | None,
    owner: str,
) -> list[Document]:
    """在主检索命中的章节内再次排序片段，改善“章节命中但片段偏移”的问题。

    针对本项目最大的失败模式：评测中 ``chapter_hit_but_gold_chunk_missed`` 占 30%
    （12/40），章节命中率 62.5% 远高于片段命中率 32.5%。主检索是全书范围的近似
    最近邻，容易落在正确章节的错误段落；这里把搜索空间收缩到已命中的若干章节，
    在小范围内重排，属于典型的 coarse-to-fine 二级检索。

    ⚠️ 实验性功能，A/B 验证结论为**负收益**，默认关闭（2026-08-28，40 题《西游记》
    单片段金标，其余条件一致）：

    ==========  ====================  ====================
    指标        关闭（基线）          开启
    ==========  ====================  ====================
    Recall@10   0.325                 **0.275**
    MRR@10      0.242                 **0.155**
    nDCG@10     0.260                 **0.184**
    P95 延迟    599 ms                652 ms
    ==========  ====================  ====================

    **根因**：候选里主检索文档带的是 RRF 融合序（向量 + 词法，候选召回 0.575），
    而本函数对所有候选按 ``score`` 排序——该字段在主检索文档上是**纯向量余弦分**
    （0.47~0.54），在章节内候选上也是向量分。于是 RRF 融合序被整体丢弃，退化成
    **纯向量检索**（候选召回仅 0.375），词法通道的贡献被抹掉，故反而更差。

    **真正的瓶颈不在这里**：候选池已经含金标的比例是 0.575，最终 Top-10 只留下
    0.325——损失发生在池内排序，而非池的召回能力。要提升应改 RRF 融合/池内排序
    （见 ``_similarity_search_once`` 的融合段），而不是加二级检索。

    如要重做本功能，必须先用统一量纲（例如对向量分与词法分各自做 min-max 归一）
    再融合，而不是直接按 ``score`` 排序。

    由 ``ENABLE_CHAPTER_LOCAL_RETRIEVAL`` 控制，默认关闭；关闭时零额外开销。
    """
    chapters = {doc.metadata.get("chapter_no") for doc in primary if doc.metadata.get("chapter_no") is not None}
    if not chapters or not file_id:
        return primary
    qvec = await asyncio.to_thread(get_embeddings().embed_query, query)
    async with AsyncSessionLocal() as session:
        distance = Embedding.embedding.cosine_distance(qvec)
        stmt = (
            select(Embedding, distance.label("distance"))
            .where(
                visible_user_filter(Embedding.user_id, owner),
                Embedding.domain == "novel",
                Embedding.file_id == file_id,
                Embedding.chapter_no.in_(list(chapters)),
            )
            .order_by(distance)
            .limit(max(k * 4, settings.chapter_local_candidate_k))
        )
        rows = (await session.execute(stmt)).all()
    candidates: dict[tuple[object, object], Document] = {}
    for doc in primary:
        candidates[(doc.metadata.get("file_id"), doc.metadata.get("chunk_no"))] = doc
    for row, distance_value in rows:
        meta = _parse_meta(row)
        score = max(0.0, min(1.0, 1.0 - float(distance_value)))
        meta.update({
            "score": round(score, 4),
            "score_type": "vector",
            "vector_score": round(score, 4),
            "fts_score": None,
            "rrf_score": None,
            "reranked": False,
            "neighbor": False,
            "local_refined": True,
        })
        key = (meta.get("file_id"), meta.get("chunk_no"))
        candidates[key] = Document(page_content=row.content, metadata=meta)
    refined = list(candidates.values())
    if settings.enable_reranker and refined:
        try:
            from app.core.rerank import rerank
            refined = await asyncio.to_thread(rerank, query, refined, k)
        except Exception as exc:  # noqa: BLE001
            log.error("chapter_local_rerank.failed", error_type=type(exc).__name__, error=str(exc))
            refined.sort(key=lambda doc: float(doc.metadata.get("score", 0.0) or 0.0), reverse=True)
            refined = refined[:k]
    else:
        refined.sort(key=lambda doc: float(doc.metadata.get("score", 0.0) or 0.0), reverse=True)
        refined = refined[:k]
    return _annotate_primary(refined)


async def expand_novel_context(
    primary: list[Document],
    neighbor_window: int | None = None,
    user_id: str | None = None,
) -> List[Document]:
    """为已完成的主检索结果补充同章节邻居片段。

    该函数不重新执行向量或词法检索，保证评测和生产请求都能明确区分
    “一次主检索”和后续的上下文扩展。
    """
    window = settings.novel_neighbor_window if neighbor_window is None else max(0, neighbor_window)
    if not primary or window <= 0:
        return primary

    owner = user_id or get_current_user()
    primary_keys = {(doc.metadata.get("file_id"), doc.metadata.get("chunk_no")) for doc in primary}
    primary_locations: dict[tuple[object, object], list[int]] = {}
    neighbor_rows: dict[tuple[object, object], Embedding] = {}
    async with AsyncSessionLocal() as session:
        for doc in primary:
            meta = doc.metadata
            chunk_no = meta.get("chunk_no")
            current_file = meta.get("file_id")
            chapter_no = meta.get("chapter_no")
            if not isinstance(chunk_no, int) or not current_file:
                continue
            primary_locations.setdefault((current_file, chapter_no), []).append(chunk_no)
            stmt = select(Embedding).where(
                visible_user_filter(Embedding.user_id, owner),
                Embedding.domain == "novel",
                Embedding.file_id == current_file,
                Embedding.chapter_no == chapter_no,
                Embedding.chunk_no.between(chunk_no - window, chunk_no + window),
            )
            result = await session.execute(stmt)
            for row in result.scalars().all():
                neighbor_rows[(row.file_id, row.chunk_no)] = row

    candidates: list[tuple[int, Embedding]] = []
    for key, row in neighbor_rows.items():
        if key in primary_keys:
            continue
        anchors = primary_locations.get((row.file_id, row.chapter_no), [])
        distance = min((abs((row.chunk_no or 0) - anchor) for anchor in anchors), default=window + 1)
        candidates.append((distance, row))
    candidates.sort(key=lambda item: (
        item[0], str(item[1].source or ""), int(item[1].chapter_no or 0), int(item[1].chunk_no or 0)
    ))

    neighbor_budget = max(2, window * 4)
    expanded = list(primary)
    for _, row in candidates[:neighbor_budget]:
        meta = _parse_meta(row)
        meta.update({
            "score": 0.0, "score_type": "neighbor", "neighbor": True,
            "retrieval_rank": None, "vector_score": None, "fts_score": None,
            "rrf_score": None, "reranked": False,
        })
        expanded.append(Document(page_content=row.content, metadata=meta))
    # 传给模型的上下文按原文顺序排列，但主命中排名已保存在 retrieval_rank。
    expanded.sort(key=lambda doc: (
        str(doc.metadata.get("source", "")),
        int(doc.metadata.get("chapter_no") or 0),
        int(doc.metadata.get("chunk_no") or 0),
    ))
    return expanded
