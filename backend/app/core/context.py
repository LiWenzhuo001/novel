"""请求级上下文——当前用户标识（多租户数据隔离）。

用 ContextVar 在异步请求生命周期内传递 user_id，避免把 user_id 当作参数
层层透传到 rag / kb / chat 每个函数签名里。鉴权中间件在每次请求开始时
set_current_user()，业务代码用 get_current_user() 读取并据此做行级过滤。

未开启鉴权时（默认），user_id 返回 settings.default_user（"default"），
所有行共享该用户，行为与旧版一致。
"""

from contextvars import ContextVar

from app.config import settings

_current_user: ContextVar[str] = ContextVar("current_user", default=settings.default_user)


def get_current_user() -> str:
    """返回当前请求的用户标识；未设置时使用默认用户。"""
    return _current_user.get()


def set_current_user(user_id: str):
    """设置当前请求的 user_id，返回 token 以便 finally 中重置。"""
    return _current_user.set(user_id)


def reset_current_user(token) -> None:
    """恢复 set_current_user 返回令牌对应的旧用户上下文。"""
    _current_user.reset(token)


# ===== 记忆工具的会话上下文 =====
# Agent 的记忆工具（search/add/update/delete）执行时需要知道当前会话与小说，
# 但这些参数不应暴露给模型（模型的工具 schema 只含业务参数）。
# 记忆决策节点执行前 set，工具函数内 get，模式与 current_user 一致。

_memory_session: ContextVar[tuple[str, str | None] | None] = ContextVar(
    "memory_session", default=None
)


def set_memory_session(session_id: str, file_id: str | None):
    """注入当前会话（session_id, file_id），返回 token 以便 finally 中重置。"""
    return _memory_session.set((session_id, file_id))


def get_memory_session() -> tuple[str, str | None] | None:
    """返回当前记忆工具可用的 (session_id, file_id)；未设置时返回 None。"""
    return _memory_session.get()


def reset_memory_session(token) -> None:
    """恢复 set_memory_session 返回令牌对应的旧上下文。"""
    _memory_session.reset(token)
