"""add TTL to agent memories and bounded-read index to chat messages

Revision ID: 20260903_0013
Revises: 20260828_0012
"""
from alembic import op
import sqlalchemy as sa

revision = "20260903_0013"
down_revision = "20260828_0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # B2: ttl_minutes 与 expires_at 同为 LangGraph twin-column TTL 模式；
    # 部分索引只覆盖真正带 TTL 的行，保证清扫任务走的索引极小。
    # A2: 重写读取 ORDER BY id DESC LIMIT n→ (session_id, id) 复合索引让该路径不再全表扫描。
    op.add_column("agent_memories", sa.Column("ttl_minutes", sa.Integer(), nullable=True))
    op.create_index(
        "ix_agent_memories_expires_at",
        "agent_memories",
        ["expires_at"],
        postgresql_where=sa.text("expires_at IS NOT NULL"),
    )
    op.create_index(
        "ix_chat_messages_session_id_id",
        "chat_messages",
        ["session_id", "id"],
    )


def downgrade() -> None:
    op.drop_index("ix_chat_messages_session_id_id", table_name="chat_messages")
    op.drop_index("ix_agent_memories_expires_at", table_name="agent_memories")
    op.drop_column("agent_memories", "ttl_minutes")