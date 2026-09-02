"""add agent summaries and long-term memory tables

Revision ID: 20260824_0004
Revises: 20260820_0003
"""
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision = "20260824_0004"
down_revision = "20260820_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "conversation_summaries",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("session_id", sa.String(32), sa.ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_id", sa.String(64), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False, server_default=""),
        sa.Column("covered_message_id", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("token_estimate", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime()),
        sa.Column("updated_at", sa.DateTime()),
    )
    op.create_index("ix_conversation_summaries_session_id", "conversation_summaries", ["session_id"])
    op.create_index("ix_conversation_summaries_user_id", "conversation_summaries", ["user_id"])

    op.create_table(
        "agent_memories",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("user_id", sa.String(64), nullable=False),
        sa.Column("session_id", sa.String(32), sa.ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=True),
        sa.Column("file_id", sa.String(32), nullable=True),
        sa.Column("memory_type", sa.String(32), nullable=False, server_default="session_fact"),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(1536), nullable=True),
        sa.Column("importance", sa.Float(), nullable=False, server_default="0.5"),
        sa.Column("source_message_id", sa.Integer(), nullable=True),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
        sa.Column("meta_json", sa.Text(), server_default="{}"),
        sa.Column("created_at", sa.DateTime()),
        sa.Column("updated_at", sa.DateTime()),
    )
    op.create_index("ix_agent_memories_user_id", "agent_memories", ["user_id"])
    op.create_index("ix_agent_memories_session_id", "agent_memories", ["session_id"])
    op.create_index("ix_agent_memories_file_id", "agent_memories", ["file_id"])
    op.create_index("ix_agent_memories_user_file", "agent_memories", ["user_id", "file_id"])


def downgrade() -> None:
    op.drop_index("ix_agent_memories_user_file", table_name="agent_memories")
    op.drop_index("ix_agent_memories_file_id", table_name="agent_memories")
    op.drop_index("ix_agent_memories_session_id", table_name="agent_memories")
    op.drop_index("ix_agent_memories_user_id", table_name="agent_memories")
    op.drop_table("agent_memories")
    op.drop_index("ix_conversation_summaries_user_id", table_name="conversation_summaries")
    op.drop_index("ix_conversation_summaries_session_id", table_name="conversation_summaries")
    op.drop_table("conversation_summaries")
