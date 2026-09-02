"""add chapter parser diagnostics and index warnings

Revision ID: 20260824_0006
Revises: 20260824_0005
Create Date: 2026-08-24
"""
from alembic import op
import sqlalchemy as sa

revision = "20260824_0006"
down_revision = "20260824_0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("knowledge_files", sa.Column("chapter_count", sa.Integer(), nullable=True))
    op.add_column("knowledge_files", sa.Column("unassigned_chunk_count", sa.Integer(), nullable=True))
    op.add_column("knowledge_files", sa.Column("chapter_parse_status", sa.String(32), nullable=True))
    op.add_column("knowledge_files", sa.Column("chapter_parser_mode", sa.String(32), nullable=True))
    op.add_column("knowledge_files", sa.Column("chapter_parser_version", sa.String(64), nullable=True))
    op.add_column("knowledge_files", sa.Column("detected_encoding", sa.String(32), nullable=True))
    op.add_column("knowledge_files", sa.Column("index_warning", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("knowledge_files", "index_warning")
    op.drop_column("knowledge_files", "detected_encoding")
    op.drop_column("knowledge_files", "chapter_parser_version")
    op.drop_column("knowledge_files", "chapter_parser_mode")
    op.drop_column("knowledge_files", "chapter_parse_status")
    op.drop_column("knowledge_files", "unassigned_chunk_count")
    op.drop_column("knowledge_files", "chapter_count")
