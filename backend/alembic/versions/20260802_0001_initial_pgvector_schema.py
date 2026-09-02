"""initial pgvector schema and hardening fields

Revision ID: 20260802_0001
Revises:
Create Date: 2026-08-02
"""
from alembic import op
import sqlalchemy as sa
import os

from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision = "20260802_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "users",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("username", sa.String(64), nullable=False),
        sa.Column("email", sa.String(255)),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("display_name", sa.String(100), server_default=""),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("is_admin", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("created_at", sa.DateTime()),
        sa.Column("updated_at", sa.DateTime()),
        sa.Column("last_login_at", sa.DateTime()),
    )
    op.create_index("ix_users_username", "users", ["username"], unique=True)
    op.create_index("ix_users_email", "users", ["email"], unique=True, postgresql_where=sa.text("email IS NOT NULL"))

    op.create_table(
        "embeddings",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(int(os.getenv("EMBED_DIM", "1536"))), nullable=False),
        sa.Column("source", sa.String(255), nullable=False),
        sa.Column("file_id", sa.String(32)),
        sa.Column("user_id", sa.String(64), server_default="default"),
        sa.Column("meta_json", sa.Text()),
        sa.Column("search_vector", postgresql.TSVECTOR(), sa.Computed("to_tsvector('simple', content)", persisted=True)),
        sa.Column("created_at", sa.DateTime()),
    )
    op.create_index("ix_embeddings_source", "embeddings", ["source"])
    op.create_index("ix_embeddings_file_id", "embeddings", ["file_id"])
    op.create_index("ix_embeddings_user_id", "embeddings", ["user_id"])
    op.execute("CREATE INDEX IF NOT EXISTS embeddings_embedding_idx ON embeddings USING hnsw (embedding vector_cosine_ops)")
    op.execute("CREATE INDEX IF NOT EXISTS embeddings_search_vector_idx ON embeddings USING gin (search_vector)")

    op.create_table(
        "knowledge_files",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("filetype", sa.String(16), server_default=""),
        sa.Column("size", sa.Integer(), server_default="0"),
        sa.Column("chunks", sa.Integer(), server_default="0"),
        sa.Column("status", sa.String(16), server_default="pending"),
        sa.Column("error", sa.Text()),
        sa.Column("user_id", sa.String(64), server_default="default"),
        sa.Column("created_at", sa.DateTime()),
        sa.Column("attempts", sa.Integer(), server_default="0", nullable=False),
        sa.Column("lease_id", sa.String(64)),
        sa.Column("lease_until", sa.DateTime()),
        sa.Column("updated_at", sa.DateTime()),
    )
    op.create_index("ix_knowledge_files_user_id", "knowledge_files", ["user_id"])

    op.create_table(
        "chat_sessions",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("title", sa.String(255), server_default="新对话"),
        sa.Column("role", sa.String(32), server_default="student"),
        sa.Column("user_id", sa.String(64), server_default="default"),
        sa.Column("created_at", sa.DateTime()),
        sa.Column("updated_at", sa.DateTime()),
    )
    op.create_index("ix_chat_sessions_user_id", "chat_sessions", ["user_id"])

    op.create_table(
        "chat_messages",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("session_id", sa.String(32), sa.ForeignKey("chat_sessions.id", ondelete="CASCADE")),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("content", sa.Text(), server_default=""),
        sa.Column("sources", sa.Text(), server_default="[]"),
        sa.Column("created_at", sa.DateTime()),
    )
    op.create_index("ix_chat_messages_session_id", "chat_messages", ["session_id"])


def downgrade() -> None:
    op.drop_index("ix_chat_messages_session_id", table_name="chat_messages")
    op.drop_table("chat_messages")
    op.drop_index("ix_chat_sessions_user_id", table_name="chat_sessions")
    op.drop_table("chat_sessions")
    op.drop_index("ix_knowledge_files_user_id", table_name="knowledge_files")
    op.drop_table("knowledge_files")
    op.execute("DROP INDEX IF EXISTS embeddings_search_vector_idx")
    op.execute("DROP INDEX IF EXISTS embeddings_embedding_idx")
    op.drop_index("ix_embeddings_user_id", table_name="embeddings")
    op.drop_index("ix_embeddings_file_id", table_name="embeddings")
    op.drop_index("ix_embeddings_source", table_name="embeddings")
    op.drop_table("embeddings")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_index("ix_users_username", table_name="users")
    op.drop_table("users")
