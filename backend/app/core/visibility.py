"""多租户可见性过滤的唯一定义点。

候选数据可见租户 = 当前用户 + 系统默认书租户（system）。跨书混入由调用方的
file_id 过滤挡住；删除/重建路径刻意不使用本模块（保持 owner 等值过滤，
系统书对所有普通用户只读）。

修改可见性规则只需改这里——检索（含 BM25 裸 SQL）、邻居扩展、会话校验
全部经由此模块取租户列表。
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import ColumnElement

from app.config import settings


def visible_users(user_id: str) -> list[str]:
    """返回 user_id 可见的数据租户列表；system 租户本身不重复叠加。"""
    if user_id == settings.system_user:
        return [user_id]
    return [user_id, settings.system_user]


def visible_user_filter(column: Any, user_id: str) -> ColumnElement:
    """生成 `column IN 可见租户` 的 SQLAlchemy 过滤条件。"""
    return column.in_(visible_users(user_id))
