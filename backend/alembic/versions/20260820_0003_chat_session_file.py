"""bind chat sessions to a selected novel file

Revision ID: 20260820_0003
Revises: 20260812_0002
"""
from alembic import op
import sqlalchemy as sa

revision = "20260820_0003"
down_revision = "20260812_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("chat_sessions", sa.Column("file_id", sa.String(32), nullable=True))
    op.create_index("ix_chat_sessions_file_id", "chat_sessions", ["file_id"])


def downgrade() -> None:
    op.drop_index("ix_chat_sessions_file_id", table_name="chat_sessions")
    op.drop_column("chat_sessions", "file_id")
