"""Memory inspection and deletion APIs."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from app.core.context import get_current_user
from app.db import AsyncSessionLocal
from app.db.models import ChatSession
from app.services import memory_service

router = APIRouter()


async def _verify_session(session_id: str) -> ChatSession:
    async with AsyncSessionLocal() as session:
        row = (await session.execute(
            select(ChatSession).where(
                ChatSession.id == session_id,
                ChatSession.user_id == get_current_user(),
            )
        )).scalars().first()
    if row is None:
        raise HTTPException(status_code=404, detail="session not found")
    return row


@router.get("/memories")
async def list_visible_memories(session_id: str | None = None, file_id: str | None = None):
    """列出当前用户在会话/小说/全局偏好三层作用域下可见的记忆。"""
    if session_id:
        session = await _verify_session(session_id)
        if file_id and session.file_id and file_id != session.file_id:
            raise HTTPException(status_code=409, detail="当前会话绑定了另一部小说")
        file_id = file_id or session.file_id
        rows = await memory_service.retrieve_memories(
            query="", session_id=session_id, file_id=file_id, limit=50
        )
    else:
        rows = await memory_service.list_memories(file_id=file_id, limit=50)
    return {"code": 0, "data": [memory_service.memory_to_dict(row) for row in rows]}


@router.delete("/memories/{memory_id}")
async def remove_memory(memory_id: str):
    """删除当前用户拥有的单条自动记忆。"""
    if not await memory_service.delete_memory(memory_id):
        raise HTTPException(status_code=404, detail="memory not found")
    return {"code": 0, "data": {"id": memory_id, "deleted": True}}


@router.get("/chat/sessions/{session_id}/memory-context")
async def get_memory_context(session_id: str, file_id: str | None = None):
    """读取当前会话下次回答可使用的摘要与记忆。"""
    session = await _verify_session(session_id)
    if file_id and session.file_id and file_id != session.file_id:
        raise HTTPException(status_code=409, detail="当前会话绑定了另一部小说")
    context = await memory_service.build_memory_context(
        session_id=session_id,
        file_id=file_id or session.file_id,
        query="",
    )
    return {"code": 0, "data": context}
