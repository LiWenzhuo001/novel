"""User authentication endpoints for user-level multi-tenancy."""

from __future__ import annotations

import re
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException
from sqlalchemy import func, or_, select

from app.core.context import get_current_user
from app.core.security import create_access_token, hash_password, verify_password
from app.db import AsyncSessionLocal
from app.db.models import User
from app.models.schemas import AuthResponse, LoginRequest, RegisterRequest, UserInfo

router = APIRouter(prefix="/auth", tags=["auth"])

_USERNAME_RE = re.compile(r"^[a-zA-Z0-9_\-]{3,64}$")


def _user_info(user: User) -> UserInfo:
    return UserInfo(
        id=user.id,
        username=user.username,
        email=user.email,
        display_name=user.display_name or user.username,
        is_admin=bool(user.is_admin),
    )


@router.post("/register")
async def register(req: RegisterRequest):
    """校验并创建用户；首个用户自动获得管理员标记。"""
    username = req.username.strip()
    if not _USERNAME_RE.match(username):
        raise HTTPException(status_code=400, detail="用户名只能包含字母、数字、下划线或中划线，长度 3-64")
    email = req.email.strip() if req.email else None
    async with AsyncSessionLocal() as session:
        stmt = select(User).where(User.username == username)
        if email:
            stmt = select(User).where(or_(User.username == username, User.email == email))
        exists = (await session.execute(stmt)).scalars().first()
        if exists:
            raise HTTPException(status_code=409, detail="用户名或邮箱已存在")
        user_count = await session.scalar(select(func.count()).select_from(User))
        user = User(
            id=uuid.uuid4().hex,
            username=username,
            email=email,
            display_name=req.display_name.strip() or username,
            password_hash=hash_password(req.password),
            is_active=True,
            is_admin=(user_count == 0),
        )
        session.add(user)
        await session.commit()
        token = create_access_token(user.id, user.username)
        return {"code": 0, "data": AuthResponse(access_token=token, user=_user_info(user)).model_dump(), "message": "ok"}


@router.post("/login")
async def login(req: LoginRequest):
    """验证用户密码和启用状态后签发访问令牌。"""
    username = req.username.strip()
    async with AsyncSessionLocal() as session:
        user = (await session.execute(select(User).where(User.username == username))).scalars().first()
        if not user or not verify_password(req.password, user.password_hash):
            raise HTTPException(status_code=401, detail="用户名或密码错误")
        if not user.is_active:
            raise HTTPException(status_code=403, detail="用户已被禁用")
        user.last_login_at = datetime.utcnow()
        await session.commit()
        token = create_access_token(user.id, user.username)
        return {"code": 0, "data": AuthResponse(access_token=token, user=_user_info(user)).model_dump(), "message": "ok"}


@router.get("/me")
async def me():
    """返回当前令牌对应的有效用户信息。"""
    user_id = get_current_user()
    async with AsyncSessionLocal() as session:
        user = await session.get(User, user_id)
        if not user or not user.is_active:
            raise HTTPException(status_code=401, detail="当前登录已失效")
        return {"code": 0, "data": _user_info(user).model_dump(), "message": "ok"}
