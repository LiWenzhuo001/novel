"""PostgreSQL + pgvector 异步数据层（SQLAlchemy 2.0 async + asyncpg）。

提供：
- async_engine / AsyncSessionLocal / Base
- get_db 依赖（FastAPI 用，AsyncSession）
- init_db：启动等待 Postgres 就绪、启用 vector 扩展并建表（幂等）

说明：
- 全部使用 SQLAlchemy 2.0 风格 select()/session.execute()，不再用 1.x 的 query()。
- pgvector 的 Vector 列、cosine_distance 比较器、TSVECTOR 生成列在异步下同样可用。
- 启动时会为历史数据库补充缺失列，保证渐进式升级不影响已有数据。
"""
import asyncio

from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base

from app.config import settings

# 异步连接串（postgresql+asyncpg://...），向量与业务数据共用同一库
ASYNC_DATABASE_URL = settings.async_database_url

async_engine = create_async_engine(
    ASYNC_DATABASE_URL,
    pool_pre_ping=True,  # 自动剔除失效连接
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,  # 提交后对象仍可用，避免懒加载触发额外 IO
    autoflush=False,
)

Base = declarative_base()

# 确保模型被注册到 Base.metadata（必须在 Base 定义后导入）
from app.db import models  # noqa: E402,F401


async def get_db():
    """FastAPI 依赖：每次请求一个异步会话，结束自动关闭。"""
    async with AsyncSessionLocal() as session:
        yield session


async def init_db(max_retries: int = 30, retry_interval: float = 2.0) -> None:
    """等待 Postgres 就绪并初始化（幂等）。

    1) 重试连接（docker-compose 中后端依赖 postgres 的 healthcheck，
       本地直接跑也可能遇到 Postgres 尚未就绪，这里做重试兜底）；
    2) 启用 pgvector 扩展（仅需一次；docker 环境下由 initdb 脚本预建）；
    3) 创建所有表；
    4) 为 embeddings.embedding 建 HNSW 索引、为 search_vector 建 GIN 索引。
    """
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            async with async_engine.connect() as conn:
                pass
            break
        except OperationalError as e:
            last_err = e
            if attempt < max_retries:
                await asyncio.sleep(retry_interval)
    else:
        raise last_err  # type: ignore[misc]

    # 启用 pgvector 扩展（非超级用户无权限时跳过，依赖 initdb 脚本预建）
    try:
        async with async_engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    except Exception as e:  # noqa: BLE001
        print(f"[INFO] 跳过 CREATE EXTENSION vector（可能已由初始化脚本创建或无权限）：{e}")

    # pg_trgm 用于中文关键词 ILIKE/相似度召回；无权限时运行期自动退回 simple FTS。
    try:
        async with async_engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
    except Exception as e:  # noqa: BLE001
        print(f"[INFO] 跳过 CREATE EXTENSION pg_trgm（中文词法检索将回退）：{e}")

    # 建表（同步 DDL，经 run_sync 在异步连接上执行）
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        dimension_result = await conn.execute(text("""
            SELECT a.atttypmod
            FROM pg_attribute AS a
            JOIN pg_class AS c ON c.oid = a.attrelid
            JOIN pg_namespace AS n ON n.oid = c.relnamespace
            WHERE n.nspname = 'public'
              AND c.relname = 'embeddings'
              AND a.attname = 'embedding'
              AND NOT a.attisdropped
        """))
        actual_dimension = dimension_result.scalar_one_or_none()
        if actual_dimension is not None and int(actual_dimension) != settings.embed_dim:
            raise RuntimeError(
                f"embeddings.embedding 当前为 {actual_dimension} 维，"
                f"运行配置要求 {settings.embed_dim} 维；请先执行向量维度迁移。"
            )
        # 轻量"迁移"：给已存在的表补新增列（create_all 不会为已有表加列，幂等）
        await conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS email VARCHAR(255)"))
        await conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS display_name VARCHAR(100) DEFAULT ''"))
        await conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT true"))
        await conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS is_admin BOOLEAN DEFAULT false"))
        await conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP"))
        await conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS last_login_at TIMESTAMP"))
        await conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ix_users_username ON users (username)"))
        await conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ix_users_email ON users (email) WHERE email IS NOT NULL"))
        await conn.execute(text("ALTER TABLE knowledge_files ADD COLUMN IF NOT EXISTS error TEXT"))
        await conn.execute(text("ALTER TABLE knowledge_files ADD COLUMN IF NOT EXISTS attempts INTEGER DEFAULT 0"))
        await conn.execute(text("ALTER TABLE knowledge_files ADD COLUMN IF NOT EXISTS lease_id VARCHAR(64)"))
        await conn.execute(text("ALTER TABLE knowledge_files ADD COLUMN IF NOT EXISTS lease_until TIMESTAMP"))
        await conn.execute(text("ALTER TABLE knowledge_files ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP"))
        await conn.execute(text("ALTER TABLE knowledge_files ADD COLUMN IF NOT EXISTS index_version VARCHAR(255)"))
        await conn.execute(text("ALTER TABLE knowledge_files ADD COLUMN IF NOT EXISTS embedding_model VARCHAR(255)"))
        await conn.execute(text("ALTER TABLE knowledge_files ADD COLUMN IF NOT EXISTS embed_dim INTEGER"))
        await conn.execute(text("ALTER TABLE knowledge_files ADD COLUMN IF NOT EXISTS chunk_size INTEGER"))
        await conn.execute(text("ALTER TABLE knowledge_files ADD COLUMN IF NOT EXISTS chunk_overlap INTEGER"))
        await conn.execute(text("ALTER TABLE knowledge_files ADD COLUMN IF NOT EXISTS indexed_at TIMESTAMP"))
        await conn.execute(text("ALTER TABLE knowledge_files ADD COLUMN IF NOT EXISTS chapter_count INTEGER"))
        await conn.execute(text("ALTER TABLE knowledge_files ADD COLUMN IF NOT EXISTS unassigned_chunk_count INTEGER"))
        await conn.execute(text("ALTER TABLE knowledge_files ADD COLUMN IF NOT EXISTS chapter_parse_status VARCHAR(32)"))
        await conn.execute(text("ALTER TABLE knowledge_files ADD COLUMN IF NOT EXISTS chapter_parser_mode VARCHAR(32)"))
        await conn.execute(text("ALTER TABLE knowledge_files ADD COLUMN IF NOT EXISTS chapter_parser_version VARCHAR(64)"))
        await conn.execute(text("ALTER TABLE knowledge_files ADD COLUMN IF NOT EXISTS detected_encoding VARCHAR(32)"))
        await conn.execute(text("ALTER TABLE knowledge_files ADD COLUMN IF NOT EXISTS index_warning TEXT"))
        await conn.execute(text("ALTER TABLE knowledge_files ADD COLUMN IF NOT EXISTS source_hash VARCHAR(64)"))
        await conn.execute(text("ALTER TABLE knowledge_files ADD COLUMN IF NOT EXISTS chapter_rule_json TEXT"))
        await conn.execute(text("ALTER TABLE knowledge_files ADD COLUMN IF NOT EXISTS chapter_rule_confidence DOUBLE PRECISION"))
        await conn.execute(text("ALTER TABLE knowledge_files ADD COLUMN IF NOT EXISTS chapter_rule_validated BOOLEAN DEFAULT false"))
        await conn.execute(text("ALTER TABLE knowledge_files ADD COLUMN IF NOT EXISTS chapter_detection_model VARCHAR(255)"))
        await conn.execute(text("ALTER TABLE knowledge_files ADD COLUMN IF NOT EXISTS chapter_detection_prompt_version VARCHAR(64)"))
        await conn.execute(text("ALTER TABLE knowledge_files ADD COLUMN IF NOT EXISTS chapter_detection_error TEXT"))
        await conn.execute(text("ALTER TABLE knowledge_files ADD COLUMN IF NOT EXISTS chapter_detection_requested BOOLEAN DEFAULT false"))
        await conn.execute(text("ALTER TABLE knowledge_files ADD COLUMN IF NOT EXISTS index_stage VARCHAR(32)"))
        await conn.execute(text("ALTER TABLE knowledge_files ADD COLUMN IF NOT EXISTS index_progress INTEGER"))
        await conn.execute(text("ALTER TABLE knowledge_files ADD COLUMN IF NOT EXISTS index_message VARCHAR(255)"))
        await conn.execute(text("ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS file_id VARCHAR(32)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_chat_sessions_file_id ON chat_sessions (file_id)"))
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS conversation_summaries (
                id VARCHAR(32) PRIMARY KEY,
                session_id VARCHAR(32) NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
                user_id VARCHAR(64) NOT NULL,
                summary TEXT NOT NULL DEFAULT '',
                covered_message_id INTEGER NOT NULL DEFAULT 0,
                token_estimate INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )
        """))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_conversation_summaries_session_id ON conversation_summaries (session_id)"))
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS agent_memories (
                id VARCHAR(32) PRIMARY KEY,
                user_id VARCHAR(64) NOT NULL,
                session_id VARCHAR(32) REFERENCES chat_sessions(id) ON DELETE CASCADE,
                file_id VARCHAR(32),
                memory_type VARCHAR(32) NOT NULL DEFAULT 'session_fact',
                preference_key VARCHAR(64),
                memory_version INTEGER NOT NULL DEFAULT 1,
                content TEXT NOT NULL,
                embedding vector,
                importance DOUBLE PRECISION NOT NULL DEFAULT 0.5,
                source_message_id INTEGER,
                expires_at TIMESTAMP,
                meta_json TEXT DEFAULT '{}',
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )
        """))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_agent_memories_user_file ON agent_memories (user_id, file_id)"))
        await conn.execute(text("ALTER TABLE agent_memories ADD COLUMN IF NOT EXISTS preference_key VARCHAR(64)"))
        await conn.execute(text("ALTER TABLE agent_memories ADD COLUMN IF NOT EXISTS memory_version INTEGER NOT NULL DEFAULT 1"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_agent_memories_user_preference ON agent_memories (user_id, memory_type, preference_key)"))
        # 多租户与知识领域隔离列：历史数据归入默认用户和 career 域，保持旧功能可见。
        for tbl in ("embeddings", "knowledge_files", "chat_sessions"):
            await conn.execute(text(f"ALTER TABLE {tbl} ADD COLUMN IF NOT EXISTS user_id VARCHAR(64)"))
            await conn.execute(
                text(f"UPDATE {tbl} SET user_id = :u WHERE user_id IS NULL").bindparams(u=settings.default_user)
            )
            await conn.execute(
                text(f"ALTER TABLE {tbl} ADD COLUMN IF NOT EXISTS domain VARCHAR(32) DEFAULT 'career'")
            )
            await conn.execute(text(f"UPDATE {tbl} SET domain = 'career' WHERE domain IS NULL"))
            await conn.execute(text(f"CREATE INDEX IF NOT EXISTS ix_{tbl}_domain ON {tbl} (domain)"))
        for column, sql_type in (
            ("chapter", "VARCHAR(255)"),
            ("chapter_no", "INTEGER"),
            ("chunk_no", "INTEGER"),
            ("page", "INTEGER"),
        ):
            await conn.execute(text(f"ALTER TABLE embeddings ADD COLUMN IF NOT EXISTS {column} {sql_type}"))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_embeddings_novel_location "
            "ON embeddings (user_id, domain, file_id, chapter_no, chunk_no)"
        ))

    # HNSW 索引（余弦检索加速）+ GIN 索引（全文检索加速）
    async with async_engine.begin() as conn:
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS embeddings_embedding_idx "
                "ON embeddings USING hnsw (embedding vector_cosine_ops)"
            )
        )
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS embeddings_search_vector_idx "
                "ON embeddings USING gin (search_vector)"
            )
        )
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS agent_memories_embedding_idx "
                "ON agent_memories USING hnsw (embedding vector_cosine_ops)"
            )
        )
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_agent_memories_scope_active "
                "ON agent_memories (user_id, session_id, file_id, importance, updated_at)"
            )
        )

    # 中文关键词 ILIKE 由 trigram GIN 加速；扩展不可用时不阻断启动。
    try:
        async with async_engine.begin() as conn:
            await conn.execute(text(
                "CREATE INDEX IF NOT EXISTS embeddings_content_trgm_idx "
                "ON embeddings USING gin (content gin_trgm_ops)"
            ))
    except Exception as e:  # noqa: BLE001
        print(f"[INFO] 跳过 embeddings_content_trgm_idx（中文词法检索将回退）：{e}")
