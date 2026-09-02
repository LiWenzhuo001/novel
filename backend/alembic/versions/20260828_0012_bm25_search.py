"""BM25 词法通道（ParadeDB pg_search）：为 embeddings.content 建 bm25 索引。

spike 结论（evals/reports/bm25_spike_20260828.md，西游记 40 题、Top-60 候选）：
- ngram 2-3 分词候选召回 0.575 > 现 ILIKE 词法通道 0.525；平均延迟 2.5ms vs 224ms。
- 查询侧必须用 jieba 分词空格连接（中文整句默认 parse 会成一个超长 token，0 命中）。

迁移带扩展存在性 guard：pgvector/pgvector 镜像没有 pg_search 时跳过建索引，
检索运行时检测 ENABLE_BM25_SEARCH 且索引存在才走 BM25 通道，否则回退现词法通道。
"""

revision = "20260828_0012"
down_revision = "20260826_0011"
branch_labels = None
depends_on = None


BM25_INDEX = (
    "CREATE INDEX IF NOT EXISTS embeddings_bm25_idx ON embeddings USING bm25 (id, content) "
    "WITH (key_field='id', text_fields='{\"content\": {\"tokenizer\": "
    "{\"type\": \"ngram\", \"min_gram\": 2, \"max_gram\": 3, \"prefix_only\": false}}}')"
)


def _has_pg_search(conn) -> bool:
    from sqlalchemy import text

    row = conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'pg_search'")).fetchone()
    return row is not None


def upgrade() -> None:
    from sqlalchemy import text

    from alembic import op

    conn = op.get_bind()
    if not _has_pg_search(conn):
        # 环境没有 ParadeDB 扩展（如 CI 的 pgvector 镜像）：跳过，不阻断迁移链。
        print("pg_search 扩展不可用，跳过 BM25 索引创建（BM25 通道保持关闭）")
        return
    # 非并发 CREATE INDEX 允许在事务内执行，保持 alembic 默认事务行为。
    conn.execute(text(BM25_INDEX))


def downgrade() -> None:
    from sqlalchemy import text

    from alembic import op

    conn = op.get_bind()
    if not _has_pg_search(conn):
        print("pg_search 扩展不可用，跳过 BM25 索引删除")
        return
    conn.execute(text("DROP INDEX IF EXISTS embeddings_bm25_idx"))
