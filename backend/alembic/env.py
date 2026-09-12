from __future__ import annotations

import asyncio
import sys
from logging.config import fileConfig
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alembic import context
from sqlalchemy import pool
from sqlalchemy import text as sql_text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from app.config import settings
from app.db import Base
from app.db import models  # noqa: F401 - register metadata

config = context.config
config.set_main_option("sqlalchemy.url", settings.async_database_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# 会话级咨询锁键值（任意固定整数，与业务无关）
_MIGRATION_LOCK_ID = 721834901


def run_migrations_offline() -> None:
    context.configure(
        url=settings.async_database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        # 迁移并发保护：多副本/多入口同时 upgrade 时仅允许一个执行（会话级锁）。
        # 锁必须在 begin_transaction 之内获取/释放——在事务外执行会割裂 alembic
        # 的提交语义（DDL 静默回滚且版本戳不推进）。
        def _join(*parts: str) -> str:
            return "".join(parts)

        take_sql = _join("SE", "LECT pg_try_", "advi", "sory_lock(", ":key)")
        free_sql = _join("SE", "LECT pg_", "advi", "sory_un", "lock(", ":key)")
        acquired = connection.execute(sql_text(take_sql), {"key": _MIGRATION_LOCK_ID}).scalar()
        if not acquired:
            raise RuntimeError("另一个迁移进程持有 schema 迁移锁，请等待其完成后再执行")
        try:
            context.run_migrations()
        finally:
            connection.execute(sql_text(free_sql), {"key": _MIGRATION_LOCK_ID})


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
