"""add vector and scope indexes for automatic memory retrieval

Revision ID: 20260826_0010
Revises: 20260825_0009
"""
from alembic import op

revision = "20260826_0010"
down_revision = "20260825_0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "CREATE INDEX IF NOT EXISTS agent_memories_embedding_idx "
        "ON agent_memories USING hnsw (embedding vector_cosine_ops)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_agent_memories_scope_active "
        "ON agent_memories (user_id, session_id, file_id, importance, updated_at)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_agent_memories_scope_active")
    op.execute("DROP INDEX IF EXISTS agent_memories_embedding_idx")
