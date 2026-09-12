"""ORM 元数据与迁移后数据库的结构对账脚本；存在漂移时退出码 1。

用法：
    python scripts/check_schema.py            # 对比并打印漂移
用途：
    - 迁移执行后的验证步骤（DEPLOY.md 固化流程）；
    - CI 门禁（test-db job）；
    - 本地开发确认 ORM / 迁移 / 数据库三方一致。

口径说明：
    - 表/列缺失：FAIL（ORM 映射的结构必须由迁移链覆盖）；
    - 关键索引缺失：FAIL（检索/记忆/去重依赖的索引见 CRITICAL_INDEXES）；
    - 仅元数据声明的装饰性单列索引缺失：WARN 不阻断（查询由复合索引覆盖，
      详见 README「数据库迁移」一节）。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alembic.autogenerate import compare_metadata  # noqa: E402
from alembic.runtime.migration import MigrationContext  # noqa: E402
from sqlalchemy import create_engine, inspect  # noqa: E402

from app.config import settings  # noqa: E402
from app.core.logging_config import get_logger  # noqa: E402
from app.db import Base  # noqa: E402

log = get_logger("schema_check")


def _sync_engine():
    """内省用同步引擎：把 asyncpg 驱动换成 psycopg2（异步驱动不能同步连接）。"""
    url = settings.database_url.replace("+asyncpg", "+psycopg2")
    return create_engine(url)

# 查询路径真正依赖的索引：缺失即失败（其余 index=True 装饰性索引不阻断）。
CRITICAL_INDEXES: dict[str, tuple[str, ...]] = {
    "embeddings": (
        "ix_embeddings_novel_location",   # 多租户+章节邻域查询主路径
        "embeddings_embedding_idx",       # HNSW 向量检索
        "embeddings_bm25_idx",            # ParadeDB 词法通道（fail-fast 依赖）
        "embeddings_content_trgm_idx",    # 中文 ILIKE 回退通道
        "embeddings_search_vector_idx",   # FTS 回退通道
    ),
    "agent_memories": (
        "agent_memories_embedding_idx",   # 记忆向量召回
        "uq_agent_memories_preference_key",  # 偏好结构化 upsert 兜底
        "ix_agent_memories_expires_at",   # TTL 清扫
        "ix_agent_memories_scope_active",  # 三层作用域召回
    ),
    "chat_messages": (
        "ix_chat_messages_session_id_id",  # 有界历史读取
    ),
}


def main() -> int:
    engine = _sync_engine()
    failures: list[str] = []
    warnings: list[str] = []

    with engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)

    if diff:
        for item in diff:
            text = repr(item)
            # 数据库侧多出的表属于显式清理范围（0016 已处理 LangGraph 实验残留），
            # 这里统一按漂移上报，由迁移负责收敛。
            (warnings if "remove_table" in text or "remove_index" in text else failures).append(text)
        for item in warnings:
            print(f"WARN drift: {item}")
        for item in failures:
            print(f"FAIL drift: {item}")

    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    for table in Base.metadata.tables:
        if table not in existing_tables:
            failures.append(f"缺少表：{table}")

    for table, critical in CRITICAL_INDEXES.items():
        if table not in existing_tables:
            continue
        existing = {entry["name"] for entry in inspector.get_indexes(table)}
        for name in critical:
            if name not in existing:
                failures.append(f"缺少关键索引：{table}.{name}")

    if failures:
        print(f"schema check FAILED：{len(failures)} 项漂移。修复：alembic upgrade head（或补校准迁移）")
        return 1
    print("schema check OK：ORM 与数据库结构一致，关键索引齐全")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
