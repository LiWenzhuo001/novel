"""novel world: chat_sessions roleplay columns + novel_characters cache table

Revision ID: 20260905_0015
Revises: 20260905_0014
"""
import sqlalchemy as sa
from alembic import op


revision = "20260905_0015"
down_revision = "20260905_0014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("chat_sessions", sa.Column("personas", sa.Text(), nullable=True))
    op.execute("UPDATE chat_sessions SET personas = '[]' WHERE personas IS NULL")
    op.add_column("chat_sessions", sa.Column("chapter_until", sa.Integer(), nullable=True))
    op.create_table(
        "novel_characters",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("user_id", sa.String(64), nullable=False, index=True),
        sa.Column("file_id", sa.String(32), nullable=False),
        sa.Column("name", sa.String(64), nullable=False),
        sa.Column("chapter_until", sa.Integer(), nullable=True),
        sa.Column("kind", sa.String(16), nullable=False, server_default="card"),
        sa.Column("content", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("source_hash", sa.String(64)),
        sa.Column("created_at", sa.DateTime()),
        sa.Column("updated_at", sa.DateTime()),
    )
    op.create_index("ix_novel_characters_file_id", "novel_characters", ["file_id"])
    op.create_index("ix_novel_characters_name", "novel_characters", ["name"])


def downgrade() -> None:
    op.drop_table("novel_characters")
    op.drop_index("ix_novel_characters_name", table_name="novel_characters")
    op.drop_index("ix_novel_characters_file_id", table_name="novel_characters")
    op.drop_column("chat_sessions", "chapter_until")
    op.drop_column("chat_sessions", "personas")
