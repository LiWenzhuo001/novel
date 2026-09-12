"""PostgreSQL + pgvector 异步数据层（SQLAlchemy 2.0 async + asyncpg）。

提供：
- async_engine / AsyncSessionLocal / Base
- get_db 依赖（FastAPI 用，AsyncSession）
- 启动期就绪检查：连接可达、迁移版本与 Alembic head 一致、向量维度匹配

迁移权威是 Alembic（backend/alembic/versions/）：
- 全新数据库：由 initdb/01-extensions.sql 安装扩展，再 `alembic upgrade head`；
- 存量数据库：执行校准迁移 20260912_0016 对齐结构；
- 应用启动只做只读检查，**绝不执行任何 schema DDL**——检查失败即启动失败，
  不允许应用自动"修复" schema（避免双权威漂移，详见 20260912_0016 docstring）。

说明：
- 就绪检查零 SQL：连接探活由 pool_pre_ping 在租借时完成，迁移版本经
  MigrationContext 读取，向量维度走表反射——全部是 SQLAlchemy 原生接口。
- pgvector 的 Vector 列、cosine_distance 比较器、TSVECTOR 生成列在异步下同样可用。
"""
import asyncio
from pathlib import Path

from alembic.config import Config as AlembicConfig
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
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
    pool_pre_ping=True,  # 连接租借时自动探活，失效连接自动重建
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


def _alembic_head() -> str:
    """从迁移脚本目录读取当前 head（单一事实来源，不硬编码版本号）。"""
    alembic_ini = Path(__file__).resolve().parents[2] / "alembic.ini"
    script = ScriptDirectory.from_config(AlembicConfig(str(alembic_ini)))
    return script.get_current_head()


async def check_database_connection(max_retries: int = 5, retry_interval: float = 2.0) -> None:
    """等待 Postgres 可连接；失败抛出。

    探活由 pool_pre_ping 在连接租借时完成（租借成功即数据库可达），
    这里只做重试循环——compose 健康检查已门禁，重试是本地直跑的兜底。
    """
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            async with async_engine.connect():
                return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt < max_retries:
                await asyncio.sleep(retry_interval)
    raise RuntimeError(f"PostgreSQL 不可连接（重试 {max_retries} 次）：{last_err}") from last_err


async def check_schema_revision() -> str:
    """校验 alembic_version 与迁移链 head 严格一致；落后/超前/未初始化均抛出。

    落后的修复命令：alembic upgrade head；超前说明数据库比应用新，请升级应用。
    """
    head = _alembic_head()

    def _current_revision(sync_conn) -> str | None:
        return MigrationContext.configure(sync_conn).get_current_revision()

    async with async_engine.connect() as conn:
        current = await conn.run_sync(_current_revision)
    if current is None:
        raise RuntimeError("数据库尚未初始化迁移版本。请先执行：alembic upgrade head")
    if current != head:
        raise RuntimeError(
            f"迁移版本不一致：数据库在 {current}，应用期望 {head}。"
            "请执行：alembic upgrade head（版本超前时请先升级应用代码）"
        )
    return current


async def check_embedding_dimension() -> int:
    """只读校验 embeddings.embedding 维度与 EMBED_DIM 一致；不匹配抛出。

    维度变更的正确路径：改 .env 的 EMBED_DIM → 备份 → 维度专用迁移 →
    全量重索引（scripts/reindex_file.py）。应用不做自动迁移。
    """

    def _reflect_dim(sync_conn) -> int | None:
        from sqlalchemy import inspect

        for column in inspect(sync_conn).get_columns("embeddings"):
            if column["name"] == "embedding":
                return getattr(column["type"], "dim", None)
        return None

    async with async_engine.connect() as conn:
        dim = await conn.run_sync(_reflect_dim)
    if dim is None:
        raise RuntimeError("embeddings.embedding 列缺失。请执行：alembic upgrade head")
    if dim != settings.embed_dim:
        raise RuntimeError(
            f"向量维度不匹配：数据库为 {dim} 维，EMBED_DIM={settings.embed_dim}。"
            "请核对 .env 的 EMBED_DIM 与 embedding 模型，并按 README 处理维度变更。"
        )
    return settings.embed_dim


async def dispose_database() -> None:
    """应用关闭时释放连接池。"""
    await async_engine.dispose()
