"""knowledge_files.chapter_detection_requested NOT NULL alignment

Revision ID: 20260912_0018
Revises: 20260912_0017

背景：0007 以 NOT NULL + server_default=false 创建了该列（Alembic 路径的库），
但 create_all 路径（旧 init_db）建出的本地库是可空的——两环境对 ORM 元数据
（现补齐 nullable=False 注解）各自产生漂移方向不同的误报。本迁移用守卫统一：
可空则回填 false 后收紧；已 NOT NULL（服务器）幂等跳过。
"""
import sqlalchemy as sa
from alembic import op

revision = "20260912_0018"
down_revision = "20260912_0017"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    for column in sa.inspect(bind).get_columns("knowledge_files"):
        if column["name"] == "chapter_detection_requested":
            if column["nullable"]:
                op.execute(
                    sa.text(
                        "UPDATE knowledge_files SET chapter_detection_requested = :value "
                        "WHERE chapter_detection_requested IS NULL"
                    ).bindparams(value=False)
                )
                op.alter_column(
                    "knowledge_files",
                    "chapter_detection_requested",
                    nullable=False,
                    server_default=sa.text("false"),
                )
            return


def downgrade() -> None:
    op.alter_column(
        "knowledge_files",
        "chapter_detection_requested",
        nullable=True,
        existing_server_default=sa.text("false"),
    )
