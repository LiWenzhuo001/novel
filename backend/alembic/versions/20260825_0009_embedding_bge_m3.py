"""rebuild embeddings with BGE-M3 1024 dimensions

Revision ID: 20260825_0009
Revises: 20260825_0008
Create Date: 2026-08-25

This migration is intentionally destructive for the vector payload: the caller
must rebuild all knowledge-file embeddings after upgrade.
"""
from alembic import op
import sqlalchemy as sa

revision = "20260825_0009"
down_revision = "20260825_0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 旧向量无法转换为 1024 维，先删除 HNSW 和旧 payload，再重建列。
    op.execute("DROP INDEX IF EXISTS embeddings_embedding_idx")
    op.execute("TRUNCATE TABLE embeddings")
    op.execute("ALTER TABLE embeddings DROP COLUMN IF EXISTS embedding")
    op.execute("ALTER TABLE embeddings ADD COLUMN embedding vector(1024) NOT NULL")
    op.execute(
        "CREATE INDEX IF NOT EXISTS embeddings_embedding_idx "
        "ON embeddings USING hnsw (embedding vector_cosine_ops)"
    )

    # 清空旧索引元数据，防止新维度重建失败时被误认为旧索引仍可用。
    op.execute("""
        UPDATE knowledge_files
        SET status = 'pending',
            chunks = 0,
            error = NULL,
            attempts = 0,
            lease_id = NULL,
            lease_until = NULL,
            index_version = NULL,
            embedding_model = NULL,
            embed_dim = NULL,
            indexed_at = NULL,
            index_stage = 'pending',
            index_progress = 0,
            index_message = '等待 BGE-M3 1024 维向量重建',
            index_warning = '已切换到 BGE-M3 1024 维，等待全量重建'
        WHERE TRUE
    """)


def downgrade() -> None:
    # 回滚只恢复列结构，不可能恢复已被清空的旧向量；需从数据库备份恢复数据。
    op.execute("DROP INDEX IF EXISTS embeddings_embedding_idx")
    op.execute("TRUNCATE TABLE embeddings")
    op.execute("ALTER TABLE embeddings DROP COLUMN IF EXISTS embedding")
    op.execute("ALTER TABLE embeddings ADD COLUMN embedding vector(512) NOT NULL")
    op.execute(
        "CREATE INDEX IF NOT EXISTS embeddings_embedding_idx "
        "ON embeddings USING hnsw (embedding vector_cosine_ops)"
    )
