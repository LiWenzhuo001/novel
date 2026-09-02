"""add rag index metadata and Chinese trigram search

Revision ID: 20260824_0005
Revises: 20260824_0004
Create Date: 2026-08-24
"""
from alembic import op
import sqlalchemy as sa

revision = "20260824_0005"
down_revision = "20260824_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("knowledge_files", sa.Column("index_version", sa.String(255), nullable=True))
    op.add_column("knowledge_files", sa.Column("embedding_model", sa.String(255), nullable=True))
    op.add_column("knowledge_files", sa.Column("embed_dim", sa.Integer(), nullable=True))
    op.add_column("knowledge_files", sa.Column("chunk_size", sa.Integer(), nullable=True))
    op.add_column("knowledge_files", sa.Column("chunk_overlap", sa.Integer(), nullable=True))
    op.add_column("knowledge_files", sa.Column("indexed_at", sa.DateTime(), nullable=True))
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.execute(
        "CREATE INDEX IF NOT EXISTS embeddings_content_trgm_idx "
        "ON embeddings USING gin (content gin_trgm_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS embeddings_content_trgm_idx")
    op.drop_column("knowledge_files", "indexed_at")
    op.drop_column("knowledge_files", "chunk_overlap")
    op.drop_column("knowledge_files", "chunk_size")
    op.drop_column("knowledge_files", "embed_dim")
    op.drop_column("knowledge_files", "embedding_model")
    op.drop_column("knowledge_files", "index_version")
