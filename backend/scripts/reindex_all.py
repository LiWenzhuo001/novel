"""Rebuild every knowledge file after changing the embedding dimension."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from sqlalchemy import select

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import settings
from app.db import AsyncSessionLocal
from app.db.models import KnowledgeFile
from app.services.kb_service import run_indexing


async def load_jobs(file_id: str | None = None) -> list[tuple[str, str, str, str, str]]:
    async with AsyncSessionLocal() as session:
        # 向量维度切换会清空统一 embeddings 表，因此需要重建所有知识库文件，
        # 不能只重建小说域，否则其他域会处于“已入库但无向量”的假可用状态。
        stmt = select(KnowledgeFile)
        if file_id:
            stmt = stmt.where(KnowledgeFile.id == file_id)
        rows = (await session.execute(stmt.order_by(KnowledgeFile.created_at))).scalars().all()
    return [(row.id, row.filename, "." + (row.filetype or ""), row.user_id, row.domain or "") for row in rows]


async def read_status(file_id: str, user_id: str) -> dict:
    async with AsyncSessionLocal() as session:
        row = (await session.execute(
            select(KnowledgeFile).where(KnowledgeFile.id == file_id, KnowledgeFile.user_id == user_id)
        )).scalars().first()
    if row is None:
        return {"file_id": file_id, "status": "missing"}
    return {
        "file_id": row.id,
        "filename": row.filename,
        "user_id": row.user_id,
        "status": row.status,
        "chunks": row.chunks,
        "embedding_model": row.embedding_model,
        "embed_dim": row.embed_dim,
        "index_stage": row.index_stage,
        "index_progress": row.index_progress,
        "error": row.error,
        "warning": row.index_warning,
    }


async def rebuild(file_id: str | None = None) -> int:
    jobs = await load_jobs(file_id)
    if not jobs:
        print("没有需要重建的知识库文件")
        return 1
    failures = 0
    print(
        f"开始全量重建，共 {len(jobs)} 个文件；"
        f"Embedding={settings.embedding_model}，维度={settings.embed_dim}；"
        "设备由 EMBEDDING_DEVICE 决定"
    )
    for index, (current_id, filename, ext, user_id, domain) in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] 开始：{filename} ({current_id})")
        try:
            # 只有小说域需要章节规则发现；其他知识库文件直接走原有解析流程。
            await run_indexing(current_id, filename, ext, user_id, domain == "novel")
            status = await read_status(current_id, user_id)
            print(json.dumps(status, ensure_ascii=False))
            if (
                status.get("status") != "indexed"
                or status.get("embedding_model") != settings.embedding_model
                or status.get("embed_dim") != settings.embed_dim
            ):
                failures += 1
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(json.dumps({"file_id": current_id, "status": "error", "error": str(exc)}, ensure_ascii=False))
    print(f"全量重建结束：成功 {len(jobs) - failures}，失败 {failures}")
    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="按当前 Embedding 配置重建全部知识库文件")
    parser.add_argument("--file-id", help="只重建一个文件；不传则重建全部知识库文件")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(rebuild(args.file_id)))


if __name__ == "__main__":
    main()
