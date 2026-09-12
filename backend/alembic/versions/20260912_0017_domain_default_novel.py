"""domain server_default: career (legacy) -> novel

Revision ID: 20260912_0017
Revises: 20260912_0016

背景：domain 列诞生于 career/novel 双领域时代（0002 迁移 server_default="career"），
系统收敛为纯小说域后，ORM 默认值是 "novel"，但数据库列默认仍是 "career"——
任何绕过 ORM 的裸 INSERT（脚本/手工修复）不带 domain 时会落进 career 域，
对 novel 查询"隐形"。本迁移把三表默认值对齐为 "novel"。

注意：schema_check（compare_metadata）默认不比对 server default，
此残留由 2026-09-12 残留清理专项人工盘点发现，非自动对账覆盖。
"""
from alembic import op

revision = "20260912_0017"
down_revision = "20260912_0016"
branch_labels = None
depends_on = None

_TABLES = ("chat_sessions", "embeddings", "knowledge_files")


def upgrade() -> None:
    for table in _TABLES:
        op.alter_column(table, "domain", server_default="novel")


def downgrade() -> None:
    for table in _TABLES:
        op.alter_column(table, "domain", server_default="career")
