"""align agent_memories.embedding dim with EMBED_DIM (512 -> 1024)

Revision ID: 20260905_0014
Revises: 20260903_0013
"""
from alembic import op


revision = "20260905_0014"
down_revision = "20260903_0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 列停留在旧 embedding 模型的 512 维，而 EMBED_DIM 已切到 Qwen3-Embedding-0.6B
    # (1024)，维度不一致让所有记忆 INSERT 报 "expected 512 dimensions, not 1024"。
    # 旧模型向量与新模型不兼容，先清空（记忆文本保留，读取端有按重要性排序的回退）。
    op.execute("UPDATE agent_memories SET embedding = NULL WHERE embedding IS NOT NULL")
    op.execute("ALTER TABLE agent_memories ALTER COLUMN embedding TYPE vector(1024)")


def downgrade() -> None:
    op.execute("UPDATE agent_memories SET embedding = NULL WHERE embedding IS NOT NULL")
    op.execute("ALTER TABLE agent_memories ALTER COLUMN embedding TYPE vector(512)")
