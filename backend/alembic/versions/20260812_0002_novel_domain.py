"""add domain fields for career and novel knowledge bases

Revision ID: 20260812_0002
Revises: 20260802_0001
"""
from alembic import op
import sqlalchemy as sa

revision = "20260812_0002"
down_revision = "20260802_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("embeddings", sa.Column("domain", sa.String(32), nullable=True, server_default="career"))
    op.add_column("embeddings", sa.Column("chapter", sa.String(255), nullable=True))
    op.add_column("embeddings", sa.Column("chapter_no", sa.Integer(), nullable=True))
    op.add_column("embeddings", sa.Column("chunk_no", sa.Integer(), nullable=True))
    op.add_column("embeddings", sa.Column("page", sa.Integer(), nullable=True))
    op.add_column("knowledge_files", sa.Column("domain", sa.String(32), nullable=True, server_default="career"))
    op.add_column("chat_sessions", sa.Column("domain", sa.String(32), nullable=True, server_default="career"))
    op.create_index("ix_embeddings_domain", "embeddings", ["domain"])
    op.create_index("ix_embeddings_chapter", "embeddings", ["chapter"])
    op.create_index("ix_embeddings_chapter_no", "embeddings", ["chapter_no"])
    op.create_index("ix_embeddings_chunk_no", "embeddings", ["chunk_no"])
    op.create_index(
        "ix_embeddings_novel_location",
        "embeddings",
        ["user_id", "domain", "file_id", "chapter_no", "chunk_no"],
    )
    op.create_index("ix_knowledge_files_domain", "knowledge_files", ["domain"])
    op.create_index("ix_chat_sessions_domain", "chat_sessions", ["domain"])
    op.execute("UPDATE embeddings SET domain = 'career' WHERE domain IS NULL")
    op.execute("UPDATE knowledge_files SET domain = 'career' WHERE domain IS NULL")
    op.execute("UPDATE chat_sessions SET domain = 'career' WHERE domain IS NULL")
    op.alter_column("embeddings", "domain", nullable=False, server_default="career")
    op.alter_column("knowledge_files", "domain", nullable=False, server_default="career")
    op.alter_column("chat_sessions", "domain", nullable=False, server_default="career")


def downgrade() -> None:
    op.drop_index("ix_chat_sessions_domain", table_name="chat_sessions")
    op.drop_index("ix_knowledge_files_domain", table_name="knowledge_files")
    op.drop_index("ix_embeddings_novel_location", table_name="embeddings")
    op.drop_index("ix_embeddings_chunk_no", table_name="embeddings")
    op.drop_index("ix_embeddings_chapter_no", table_name="embeddings")
    op.drop_index("ix_embeddings_chapter", table_name="embeddings")
    op.drop_index("ix_embeddings_domain", table_name="embeddings")
    op.drop_column("embeddings", "page")
    op.drop_column("embeddings", "chunk_no")
    op.drop_column("embeddings", "chapter_no")
    op.drop_column("embeddings", "chapter")
    op.drop_column("chat_sessions", "domain")
    op.drop_column("knowledge_files", "domain")
    op.drop_column("embeddings", "domain")
