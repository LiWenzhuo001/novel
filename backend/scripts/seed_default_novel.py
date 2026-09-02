# -*- coding: utf-8 -*-
"""把《红楼梦》种子为系统默认小说（system 租户），所有用户首次进入即有。

做法：从现有已验证的索引（默认 file_id 00df98487dd3）做**纯 SQL 复制**——
原文 txt、knowledge_files 元数据行、1982 条 embeddings 行各复制一份到
system 租户名下（新 file_id）。秒级完成，不重新跑 embedding、不调 LLM
章节解析，向量与现有金标完全一致。

幂等：system 租户下已存在 indexed 的目标书则直接跳过。
检索侧通过 user_id IN (当前用户, system) 读到它；删除/重建保持 owner
等值过滤，系统书对所有普通用户只读。
"""
from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
import uuid
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sqlalchemy import select  # noqa: E402

from app.config import settings  # noqa: E402
from app.db import AsyncSessionLocal  # noqa: E402
from app.db.models import Embedding, KnowledgeFile  # noqa: E402

DEFAULT_SOURCE_FILE_ID = "00df98487dd3"
COPY_BATCH = 500


async def system_book_exists(session) -> KnowledgeFile | None:
    stmt = select(KnowledgeFile).where(
        KnowledgeFile.user_id == settings.system_user,
        KnowledgeFile.status == "indexed",
    )
    return (await session.execute(stmt)).scalars().first()


def copyable_cols(model) -> list[str]:
    """生成列（如 search_vector）由数据库自动维护，复制时必须排除。"""
    return [c.name for c in model.__table__.columns if c.computed is None]


async def seed(source_file_id: str, system_user: str) -> None:
    async with AsyncSessionLocal() as session:
        if existing := await system_book_exists(session):
            print(f"已存在系统默认小说：file_id={existing.id}（{existing.filename}），跳过")
            return
        source = (await session.execute(
            select(KnowledgeFile).where(KnowledgeFile.id == source_file_id)
        )).scalars().first()
        if source is None or source.status != "indexed":
            raise RuntimeError(f"源文件不可用：{source_file_id}（不存在或未完成索引）")

        new_file_id = uuid.uuid4().hex[:12]
        row = {name: getattr(source, name) for name in copyable_cols(KnowledgeFile)}
        row.update({
            "id": new_file_id,
            "user_id": system_user,
            "filename": source.filename,
            "status": "indexed",
            "attempts": 0,
            "lease_id": None,
            "lease_until": None,
        })
        session.add(KnowledgeFile(**row))

        total = 0
        embedding_cols = copyable_cols(Embedding)
        embedding_rows = (await session.execute(
            select(Embedding.id).where(Embedding.file_id == source_file_id)
        )).scalars().all()
        for start in range(0, len(embedding_rows), COPY_BATCH):
            batch_ids = embedding_rows[start:start + COPY_BATCH]
            rows = (await session.execute(
                select(Embedding).where(Embedding.id.in_(tuple(batch_ids)))
            )).scalars().all()
            for emb in rows:
                cols = {name: getattr(emb, name) for name in embedding_cols}
                cols["id"] = uuid.uuid4().hex
                cols["file_id"] = new_file_id
                cols["user_id"] = system_user
                session.add(Embedding(**cols))
            total += len(rows)
        await session.commit()

    # 原文复制到新 file_id 名下，保证系统书可 reindex（仅管理员场景）。
    source_txt = next(Path(settings.raw_dir).glob(f"{source_file_id}.*"), None)
    if source_txt:
        target = Path(settings.raw_dir) / f"{new_file_id}{source_txt.suffix}"
        shutil.copyfile(source_txt, target)
        print(f"原文已复制：{target.name}")
    else:
        print(f"⚠️ 未找到原文 {source_file_id}.*，系统书无法 reindex（检索不受影响）")

    print(f"种子完成：{source.filename} {source_file_id} -> system 租户 {new_file_id}（{total} 条向量）")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="种子系统默认小说（幂等）")
    parser.add_argument("--source-file-id", default=DEFAULT_SOURCE_FILE_ID)
    parser.add_argument("--system-user", default=settings.system_user)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.system_user != settings.system_user:
        settings.system_user = args.system_user
    asyncio.run(seed(args.source_file_id, args.system_user))
