"""add cached LLM chapter detection rules

Revision ID: 20260824_0007
Revises: 20260824_0006
Create Date: 2026-08-24
"""
from alembic import op
import sqlalchemy as sa

revision = "20260824_0007"
down_revision = "20260824_0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("knowledge_files", sa.Column("source_hash", sa.String(64), nullable=True))
    op.add_column("knowledge_files", sa.Column("chapter_rule_json", sa.Text(), nullable=True))
    op.add_column("knowledge_files", sa.Column("chapter_rule_confidence", sa.Float(), nullable=True))
    op.add_column("knowledge_files", sa.Column("chapter_rule_validated", sa.Boolean(), nullable=True, server_default=sa.text("false")))
    op.add_column("knowledge_files", sa.Column("chapter_detection_model", sa.String(255), nullable=True))
    op.add_column("knowledge_files", sa.Column("chapter_detection_prompt_version", sa.String(64), nullable=True))
    op.add_column("knowledge_files", sa.Column("chapter_detection_error", sa.Text(), nullable=True))
    op.add_column("knowledge_files", sa.Column("chapter_detection_requested", sa.Boolean(), nullable=False, server_default=sa.text("false")))


def downgrade() -> None:
    op.drop_column("knowledge_files", "chapter_detection_requested")
    op.drop_column("knowledge_files", "chapter_detection_error")
    op.drop_column("knowledge_files", "chapter_detection_prompt_version")
    op.drop_column("knowledge_files", "chapter_detection_model")
    op.drop_column("knowledge_files", "chapter_rule_validated")
    op.drop_column("knowledge_files", "chapter_rule_confidence")
    op.drop_column("knowledge_files", "chapter_rule_json")
    op.drop_column("knowledge_files", "source_hash")
