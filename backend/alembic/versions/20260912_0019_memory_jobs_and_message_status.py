"""memory_jobs task queue + chat_messages.status run-state

Revision ID: 20260912_0019
Revises: 20260912_0018

背景：Agent 单链路收敛后，回答过程不再直接写长期记忆——聊天后的记忆维护
（摘要+抽取）与模型发起的单条记忆操作统一入队 memory_jobs，由进程内 worker
原子领取执行。assistant_message_id 对 maintain 类任务是幂等键（部分唯一索引）。
chat_messages 增加 status 运行终态列：只有 completed 的回答参与后续 Query
改写历史；超时/异常的残缺输出标记 partial 保留但不进历史。
"""
import sqlalchemy as sa
from alembic import op

revision = "20260912_0019"
down_revision = "20260912_0018"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "memory_jobs",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("user_id", sa.String(64), nullable=False, index=True),
        sa.Column(
            "session_id",
            sa.String(32),
            sa.ForeignKey("chat_sessions.id", ondelete="CASCADE"),
            index=True,
        ),
        sa.Column("assistant_message_id", sa.Integer, index=True),
        sa.Column("kind", sa.String(16), nullable=False, server_default="maintain"),
        sa.Column("priority", sa.String(8), nullable=False, server_default="normal"),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending", index=True),
        sa.Column("attempts", sa.Integer, nullable=False, server_default="0"),
        sa.Column("max_attempts", sa.Integer, nullable=False, server_default="3"),
        sa.Column("lease_id", sa.String(64)),
        sa.Column("lease_until", sa.DateTime),
        sa.Column("payload", sa.Text, nullable=False, server_default="{}"),
        sa.Column("error_code", sa.String(64)),
        sa.Column("created_at", sa.DateTime),
        sa.Column("updated_at", sa.DateTime),
    )
    # maintain 任务幂等：同一条 assistant 消息只允许一个维护任务。
    op.create_index(
        "ux_memory_jobs_maintain_message",
        "memory_jobs",
        ["assistant_message_id"],
        unique=True,
        postgresql_where=sa.text("kind = 'maintain'"),
    )
    op.create_index(
        "ix_memory_jobs_claim",
        "memory_jobs",
        ["status", "priority", "created_at"],
    )

    bind = op.get_bind()
    columns = {column["name"] for column in sa.inspect(bind).get_columns("chat_messages")}
    if "status" not in columns:
        op.add_column(
            "chat_messages",
            sa.Column("status", sa.String(16), nullable=False, server_default="completed"),
        )


def downgrade() -> None:
    op.drop_index("ix_memory_jobs_claim", table_name="memory_jobs")
    op.drop_index("ux_memory_jobs_maintain_message", table_name="memory_jobs")
    op.drop_table("memory_jobs")
    op.drop_column("chat_messages", "status")
