"""系统默认小说（system 租户）的可见性与只读保护测试（mock 会话，无真实 DB）。"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.context import reset_current_user, set_current_user
from app.services import kb_service


def _fake_session(result_rows=None, first=None):
    """构造异步会话替身：execute 返回给定行集/单记录。"""
    session = MagicMock()
    scalars = MagicMock()
    scalars.all = MagicMock(return_value=result_rows or [])
    scalars.first = MagicMock(return_value=first)
    fake_result = MagicMock()
    fake_result.scalars = MagicMock(return_value=scalars)
    session.execute = AsyncMock(return_value=fake_result)
    session.delete = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return session


def _book_row(book_id: str, user_id: str, created_at: str):
    """list_files payload 会读取全部列表字段，桩对象需要带齐默认值。"""
    from datetime import datetime

    return SimpleNamespace(
        id=book_id, user_id=user_id,
        created_at=datetime.strptime(created_at, "%Y-%m-%d %H:%M"),
        filename="红楼梦.txt", filetype="txt", size=1, chunks=1,
        status="indexed", index_stage=None, index_progress=None,
        index_message=None, error=None, domain="novel",
        index_version="v1", embedding_model="BAAI/bge-m3", embed_dim=1024,
        chunk_size=650, chunk_overlap=120, indexed_at=None,
        chapter_count=1, unassigned_chunk_count=0,
        chapter_parse_status="ok", chapter_parser_mode="strict",
        chapter_parser_version="chapter-v3", detected_encoding="utf-8",
        index_warning=None, chapter_rule_confidence=None,
        chapter_rule_validated=None, chapter_detection_model=None,
        chapter_detection_error=None,
    )


@pytest.mark.asyncio
async def test_list_files_includes_system_book(monkeypatch):
    """list_files 应包含系统默认书并打 is_system 标记；本人书籍标记为 False。"""
    user_book = _book_row("mine", "alice", "2026-08-30 10:00")
    system_book = _book_row("sysbook", "system", "2026-08-29 09:00")
    monkeypatch.setattr(
        kb_service, "AsyncSessionLocal", lambda: _fake_session(result_rows=[system_book, user_book])
    )
    token = set_current_user("alice")
    try:
        files = await kb_service.list_files()
    finally:
        reset_current_user(token)
    by_id = {item["id"]: item for item in files}
    assert by_id["sysbook"]["is_system"] is True
    assert by_id["mine"]["is_system"] is False


@pytest.mark.asyncio
async def test_delete_file_cannot_touch_system_book(monkeypatch):
    """其他用户删除系统书：owner 等值过滤查不到记录，应返回 False（API 层 404）。"""
    session = _fake_session(first=None)  # owner 等值过滤下，alice 看不到 system 的书
    monkeypatch.setattr(kb_service, "AsyncSessionLocal", lambda: session)
    token = set_current_user("alice")
    try:
        result = await kb_service.delete_file("sysbook")
    finally:
        reset_current_user(token)
    assert result is False
    session.delete.assert_not_called()
