"""add structured preference fields to agent memories

Revision ID: 20260826_0011
Revises: 20260826_0010
"""
from alembic import op
import sqlalchemy as sa

revision = "20260826_0011"
down_revision = "20260826_0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("agent_memories", sa.Column("preference_key", sa.String(64), nullable=True))
    op.add_column("agent_memories", sa.Column("memory_version", sa.Integer(), nullable=False, server_default="1"))
    op.create_index("ix_agent_memories_preference_key", "agent_memories", ["preference_key"])
    op.create_index("ix_agent_memories_user_preference", "agent_memories", ["user_id", "memory_type", "preference_key"])


def downgrade() -> None:
    op.drop_index("ix_agent_memories_user_preference", table_name="agent_memories")
    op.drop_index("ix_agent_memories_preference_key", table_name="agent_memories")
    op.drop_column("agent_memories", "memory_version")
    op.drop_column("agent_memories", "preference_key")
