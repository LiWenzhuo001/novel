"""reconcile critical indexes missing from the migration chain

Revision ID: 20260912_0020
Revises: 20260912_0019

背景：check_schema 把 agent_memories.uq_agent_memories_preference_key（偏好
结构化 upsert 兜底）列为关键索引，但该索引在本地库是历史手工创建的，迁移链
里没有任何一环建过它——从旧库 stamp 对齐上链的环境（如服务器）永远缺失，
check_schema 必然失败。本迁移按守卫补齐关键索引（存在即跳过，幂等）：
- uq_agent_memories_preference_key：与本地库定义一致（三列唯一，PG 默认
  NULL 相异语义，普通记忆行不受影响）；
- memory_jobs 两个执行期索引（claim 复合 + maintain 幂等部分唯一）——
  0019 已建的环境守卫跳过，防止未来环境再漂移。
"""
import sqlalchemy as sa
from alembic import op

revision = "20260912_0020"
down_revision = "20260912_0019"
branch_labels = None
depends_on = None

_TARGET_INDEXES = {
    "agent_memories": (
        ("uq_agent_memories_preference_key",
         ["user_id", "memory_type", "preference_key"], {"unique": True}),
    ),
    "memory_jobs": (
        ("ix_memory_jobs_claim",
         ["status", "priority", "created_at"], {}),
        ("ux_memory_jobs_maintain_message",
         ["assistant_message_id"], {"unique": True}),
    ),
}


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())
    for table, indexes in _TARGET_INDEXES.items():
        if table not in tables:
            continue
        existing = {i["name"] for i in inspector.get_indexes(table)}
        for name, columns, kwargs in indexes:
            if name in existing:
                continue
            op.create_index(name, table, columns, **kwargs)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    for table, indexes in _TARGET_INDEXES.items():
        existing = {i["name"] for i in inspector.get_indexes(table)}
        for name, _columns, _kwargs in indexes:
            if name in existing:
                op.drop_index(name, table_name=table)
