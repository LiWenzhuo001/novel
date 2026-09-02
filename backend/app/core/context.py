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
