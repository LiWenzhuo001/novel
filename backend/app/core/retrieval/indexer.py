"""向量入库层：文档配对转换、分批嵌入与增删改查。「

!!! 业务代码请从 app.core.rag 导入。「"""

from __future__ import annotations

import asyncio
import json

from typing import Iterable, List

from langchain_core.documents import Document
from sqlalchemy import delete, func, select

from app.config import settings
from app.core.context import get_current_user
from app.core.embed import get_embeddings
from app.db import AsyncSessionLocal
from app.db.models import Embedding


def _rows_for_documents(chunks: Iterable[Document], vectors: Iterable[list[float]], user_id: str) -> list[Embedding]:
    """把 LangChain 文档和向量按位置配对转换为 Embedding ORM 行。"""
    chunk_list = list(chunks)
    vector_list = list(vectors)
    if len(chunk_list) != len(vector_list):
        raise RuntimeError(
            f"Embedding 返回数量异常：chunks={len(chunk_list)}, vectors={len(vector_list)}"
        )
    rows: list[Embedding] = []
    for doc, vec in zip(chunk_list, vector_list):
        meta = doc.metadata
        extra = {
            key: value
            for key, value in meta.items()
            if key not in ("source", "file_id", "domain", "chapter", "chapter_no", "chunk_no", "page")
        }
        rows.append(
            Embedding(
                content=doc.page_content,
                embedding=vec,
                source=meta.get("source", "未知"),
                file_id=meta.get("file_id"),
                user_id=user_id,
                domain=meta.get("domain", "novel"),
                chapter=meta.get("chapter"),
                chapter_no=meta.get("chapter_no"),
                chunk_no=meta.get("chunk_no"),
                page=meta.get("page"),
                meta_json=json.dumps(extra, ensure_ascii=False),
            )
        )
    return rows


async def _embed_documents_batched(chunks: list[Document]) -> list[list[float]]:
    """分批生成向量，降低 BGE-M3 处理长篇小说时的峰值内存。"""
    vectors: list[list[float]] = []
    batch_size = settings.embedding_batch_size
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        batch_vectors = await asyncio.to_thread(
            get_embeddings().embed_documents, [chunk.page_content for chunk in batch]
        )
        vectors.extend(batch_vectors)
    return vectors


async def add_documents(chunks: List[Document], user_id: str = None) -> None:
    """Embed and append documents."""
    if not chunks:
        return
    owner = user_id or get_current_user()
    vectors = await _embed_documents_batched(chunks)
    async with AsyncSessionLocal() as session:
        session.add_all(_rows_for_documents(chunks, vectors, owner))
        await session.commit()


async def replace_documents(file_id: str, chunks: List[Document], user_id: str = None) -> None:
    """Atomically replace one file after all new vectors have been produced successfully."""
    owner = user_id or get_current_user()
    vectors = await _embed_documents_batched(chunks) if chunks else []
    rows = _rows_for_documents(chunks, vectors, owner)
    async with AsyncSessionLocal() as session:
        async with session.begin():
            await session.execute(
                delete(Embedding).where(Embedding.file_id == file_id, Embedding.user_id == owner)
            )
            session.add_all(rows)


async def delete_by_source(source: str, user_id: str = None) -> None:
    owner = user_id or get_current_user()
    async with AsyncSessionLocal() as session:
        await session.execute(delete(Embedding).where(Embedding.source == source, Embedding.user_id == owner))
        await session.commit()


async def delete_by_file_id(file_id: str, user_id: str = None) -> None:
    """按文件和用户删除向量，避免跨租户误删。"""
    owner = user_id or get_current_user()
    async with AsyncSessionLocal() as session:
        await session.execute(delete(Embedding).where(Embedding.file_id == file_id, Embedding.user_id == owner))
        await session.commit()


async def count() -> int:
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(func.count()).select_from(Embedding))
        return int(result.scalar_one())
