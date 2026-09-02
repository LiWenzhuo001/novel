"""add index task progress fields

Revision ID: 20260825_0008
Revises: 20260824_0007
Create Date: 2026-08-25
"""
from alembic import op
import sqlalchemy as sa

revision = "20260825_0008"
down_revision = "20260824_0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("knowledge_files", sa.Column("index_stage", sa.String(32), nullable=True))
    op.add_column("knowledge_files", sa.Column("index_progress", sa.Integer(), nullable=True))
    op.add_column("knowledge_files", sa.Column("index_message", sa.String(255), nullable=True))


def downgrade() -> None:
    op.drop_column("knowledge_files", "index_message")
    op.drop_column("knowledge_files", "index_progress")
    op.drop_column("knowledge_files", "index_stage")
