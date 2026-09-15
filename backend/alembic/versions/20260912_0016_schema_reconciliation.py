"""schema reconciliation: converge init_db out-of-band structures into the migration chain

Revision ID: 20260912_0016
Revises: 20260905_0015

背景：应用启动曾长期通过 init_db() 的 `ADD COLUMN IF NOT EXISTS` / `CREATE INDEX`
带外修改 schema，导致部分结构与迁移链脱节。本 revision 把对账出的差异收敛进来：

1. 补齐 ORM 元数据声明但迁移链缺失的索引（7 个，含 0011 漏建的单列偏好索引）；
2. 多租户/领域字段与重试计数的 NOT NULL 收紧（先回填 NULL，再设约束）；
3. 清理 LangGraph checkpointer 实验遗留孤儿表（零代码引用）。

全部操作带存在性/可空性守卫，可重复执行；全新环境与漂移环境结果一致。
"""
import sys

import sqlalchemy as sa
from alembic import op

revision = "20260912_0016"
down_revision = "20260905_0015"
branch_labels = None
depends_on = None


def _trace(message: str) -> None:
    print(f"[0016] {message}", file=sys.stderr, flush=True)

# (表, 索引名, 列)——ORM 元数据声明且查询/约束依赖，但历史迁移未创建。
# 注意 ix_chat_messages_session_id_id（0013 迁移专属、不在 ORM 元数据里）也列入：
# 服务器上若为 create_all 塑形的存量库，升级时由此补齐；本地已存在则守卫跳过。
_MISSING_INDEXES = (
    ("agent_memories", "ix_agent_memories_preference_key", ["preference_key"]),
    ("chat_sessions", "ix_chat_sessions_user_id", ["user_id"]),
    ("embeddings", "ix_embeddings_user_id", ["user_id"]),
    ("embeddings", "ix_embeddings_chapter", ["chapter"]),
    ("embeddings", "ix_embeddings_chapter_no", ["chapter_no"]),
    ("embeddings", "ix_embeddings_chunk_no", ["chunk_no"]),
    ("knowledge_files", "ix_knowledge_files_user_id", ["user_id"]),
    ("chat_messages", "ix_chat_messages_session_id_id", ["session_id", "id"]),
)

# (表, 列, 回填默认值)——先回填 NULL 再 SET NOT NULL（Expand/Contract 的 Migrate 阶段）。
# 注意：asyncpg 强类型，回填值必须与列类型一致（attempts 是 INTEGER，用 int 0）。
_NULL_BACKFILLS = (
    ("chat_sessions", "domain", "novel"),
    ("embeddings", "domain", "novel"),
    ("knowledge_files", "domain", "novel"),
    ("knowledge_files", "attempts", 0),
)

_ORPHAN_TABLES = ("resume_checkpoint_writes", "resume_checkpoints")  # 依赖表在前删除


def _table_exists(bind, name: str) -> bool:
    return name in sa.inspect(bind).get_table_names()


def _index_exists(bind, table: str, name: str) -> bool:
    return any(idx["name"] == name for idx in sa.inspect(bind).get_indexes(table))


def _column_nullable(bind, table: str, column: str) -> bool | None:
    for column_info in sa.inspect(bind).get_columns(table):
        if column_info["name"] == column:
            return column_info["nullable"]
    return None


def upgrade() -> None:
    bind = op.get_bind()
    _trace("start: index 补齐")
    for table, name, columns in _MISSING_INDEXES:
        _trace(f"index {name}: exists={_index_exists(bind, table, name)}")
        if _table_exists(bind, table) and not _index_exists(bind, table, name):
            op.create_index(name, table, columns)
            _trace(f"index {name}: created")

    _trace("start: NULL 回填与 NOT NULL 收紧")
    for table, column, default in _NULL_BACKFILLS:
        if not _table_exists(bind, table):
            continue
        nullable = _column_nullable(bind, table, column)
        _trace(f"{table}.{column}: nullable={nullable}")
        if nullable is False:
            continue  # 已是 NOT NULL（全新环境由历史迁移创建）
        try:
            op.execute(
                sa.text(f"UPDATE {table} SET {column} = :value WHERE {column} IS NULL").bindparams(value=default)
            )
            _trace(f"{table}.{column}: backfilled")
            op.alter_column(table, column, nullable=False)
            _trace(f"{table}.{column}: NOT NULL 已设置")
        except Exception as exc:
            _trace(f"FAILED at {table}.{column}: {type(exc).__name__}: {str(exc)[:300]}")
            raise

    _trace("start: 孤儿表清理")
    for table in _ORPHAN_TABLES:
        _trace(f"drop {table}: exists={_table_exists(bind, table)}")
        if _table_exists(bind, table):
            op.drop_table(table)
            _trace(f"drop {table}: done")
    _trace("completed")


def downgrade() -> None:
    bind = op.get_bind()

    # 孤儿实验表不可忠实重建（数据已随 upgrade 删除），downgrade 仅移除本 revision
    # 新增的索引与约束；需要恢复请从备份取回。
    for table, name, columns in reversed(_MISSING_INDEXES):
        if _table_exists(bind, table) and _index_exists(bind, table, name):
            op.drop_index(name, table_name=table)

    for table, column, _default in reversed(_NULL_BACKFILLS):
        if _table_exists(bind, table):
            op.alter_column(table, column, nullable=True)
