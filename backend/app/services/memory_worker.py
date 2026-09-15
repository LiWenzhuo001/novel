"""记忆维护 worker：进程内后台循环，消费 memory_jobs 任务队列。

设计要点：
- 原子领取：参数化 UPDATE ... WHERE id = (SELECT ... FOR UPDATE SKIP LOCKED)
  RETURNING（与 main.py 的 sql_text+bindparams 先例一致），并发领取不重复执行。
- 崩溃恢复：领取即写 lease_until（MEMORY_JOB_TIMEOUT）；进程重启后过期 lease
  的 running 任务会被重新领取，attempts 超过上限的任务标记 failed。
- 幂等：maintain 类任务以 assistant_message_id 为部分唯一索引，重复入队直接失败。
- 上下文：任务自带 user_id/session_id/file_id，执行前 set_current_user 注入租户，
  不依赖任何请求上下文。
- TTL 清扫收编进本循环：按 memory_ttl_sweeper_interval_hours 周期清理过期记忆，
  替代原先独立的裸 create_task sweeper。
"""
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timedelta

from sqlalchemy import text as sql_text
from sqlalchemy import update

from app.config import settings
from app.core.context import set_current_user, reset_current_user
from app.core.logging_config import get_logger
from app.db import AsyncSessionLocal
from app.db.models import MemoryJob
from app.services import memory_service

log = get_logger("memory_worker")

# 原子领取：子查询 FOR UPDATE SKIP LOCKED 锁定候选行，外层 UPDATE 抢占租约。
# 仅有的动态值（lease_id/lease_until/now）全部参数绑定，无拼接。
_CLAIM_SQL = sql_text("""
    UPDATE memory_jobs SET
        status = 'running',
        lease_id = :lease_id,
        lease_until = :lease_until,
        attempts = attempts + 1,
        updated_at = :now
    WHERE id = (
        SELECT id FROM memory_jobs
        WHERE (status = 'pending' OR (status = 'running' AND lease_until < :now))
          AND attempts < max_attempts
        ORDER BY CASE priority WHEN 'high' THEN 0 ELSE 1 END, created_at ASC
        LIMIT 1
        FOR UPDATE SKIP LOCKED
    )
    RETURNING id, user_id, session_id, assistant_message_id, kind, attempts, max_attempts, payload, status, lease_id
""")


async def _claim_due_job() -> MemoryJob | None:
    """原子领取一个到期任务：pending 或 lease 过期的 running，按优先级先进先出。"""
    now = datetime.utcnow()
    lease_id = uuid.uuid4().hex
    lease_until = now + timedelta(seconds=settings.memory_job_timeout)
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            _CLAIM_SQL.bindparams(lease_id=lease_id, lease_until=lease_until, now=now)
        )
        row = result.mappings().first()
        await session.commit()
    if row is None:
        return None
    return MemoryJob(
        id=row["id"],
        user_id=row["user_id"],
        session_id=row["session_id"],
        assistant_message_id=row["assistant_message_id"],
        kind=row["kind"],
        attempts=row["attempts"],
        max_attempts=row["max_attempts"],
        payload=row["payload"],
        status=row["status"],
        lease_id=row["lease_id"],
    )


async def _finish_job(job_id: str, **values) -> None:
    """按主键收尾任务状态（completed/failed/pending），全部经 ORM 参数化。"""
    async with AsyncSessionLocal() as session:
        await session.execute(
            update(MemoryJob).where(MemoryJob.id == job_id).values(updated_at=datetime.utcnow(), **values)
        )
        await session.commit()


async def _execute_maintain(payload: dict, session_id: str | None) -> dict:
    """整轮维护：会话摘要 + 长期记忆抽取（原 chat 后台批处理流的执行体）。"""
    return await memory_service.maintain_conversation_memory(
        session_id=str(session_id or ""),
        file_id=payload.get("file_id"),
        user_text=str(payload.get("user_text") or ""),
        assistant_text=str(payload.get("assistant_text") or ""),
        assistant_message_id=payload.get("assistant_message_id"),
        skip_extract=bool(payload.get("skip_extract")),
    )


async def _execute_memory_op(payload: dict, session_id: str | None) -> dict:
    """单条记忆操作（模型/用户显式发起）：add / update / delete。"""
    op = str(payload.get("op") or "").lower()
    file_id = payload.get("file_id")
    if op == "add":
        content = str(payload.get("content") or "").strip()
        memory_type = str(payload.get("memory_type") or "session_fact")
        if not content or memory_type not in {"user_preference", "novel_fact", "session_fact"}:
            raise ValueError("invalid_memory_op_payload")
        row = await memory_service.save_memory(
            content, memory_type,
            session_id=session_id if memory_type == "session_fact" else None,
            file_id=file_id if memory_type == "novel_fact" else None,
            importance=float(payload.get("importance", 0.7)),
        )
        return {"op": "add", "memory_id": row.id}
    if op == "update":
        row = await memory_service.update_memory(
            str(payload.get("id") or ""),
            content=str(payload.get("content") or ""),
        )
        if row is None:
            raise ValueError("memory_not_found")
        return {"op": "update", "memory_id": row.id}
    if op == "delete":
        deleted = await memory_service.delete_memory(str(payload.get("id") or ""))
        if not deleted:
            raise ValueError("memory_not_found")
        return {"op": "delete"}
    raise ValueError("unknown_memory_op")


async def _run_job(job: MemoryJob) -> None:
    """执行单个任务：切租户上下文 → 按 kind 分发 → 异常上抛交给收尾逻辑。"""
    token = set_current_user(job.user_id)
    try:
        payload = json.loads(job.payload or "{}")
        if not isinstance(payload, dict):
            raise ValueError("invalid_job_payload")
        if job.kind == "maintain":
            result = await _execute_maintain(payload, job.session_id)
        elif job.kind == "memory_op":
            result = await _execute_memory_op(payload, job.session_id)
        else:
            raise ValueError("unknown_job_kind")
        await _finish_job(job.id, status="completed", lease_id=None, lease_until=None, error_code=None)
        log.info("memory_job.completed", job_id=job.id, kind=job.kind, result=str(result)[:120])
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        failed_final = job.attempts >= job.max_attempts
        await _finish_job(
            job.id,
            status="failed" if failed_final else "pending",
            lease_id=None,
            lease_until=None,
            error_code=str(type(exc).__name__)[:64],
        )
        log.warning(
            "memory_job.failed",
            job_id=job.id,
            kind=job.kind,
            attempts=job.attempts,
            final=failed_final,
            error=f"{type(exc).__name__}: {exc}"[:200],
        )
    finally:
        reset_current_user(token)


async def _sweep_expired_memories_if_due(state: dict) -> None:
    """TTL 清扫：按小时间隔内嵌在 worker 循环里（替代独立 sweeper 任务）。"""
    now = datetime.utcnow().timestamp()
    interval = settings.memory_ttl_sweeper_interval_hours * 3600
    if not settings.memory_ttl_sweeper_enabled or now - state.get("last_sweep", 0) < interval:
        return
    state["last_sweep"] = now
    try:
        deleted = await memory_service.sweep_expired_memories()
        if deleted:
            log.info("memory.ttl_swept", deleted=deleted)
    except Exception as exc:  # noqa: BLE001
        log.warning("memory.ttl_sweep_failed", error=str(exc)[:200])


async def run_memory_worker_loop() -> None:
    """worker 主循环：领取 → 执行 → 顺带 TTL 清扫；取消时优雅退出。"""
    log.info("memory_worker.started", poll_interval=settings.memory_worker_poll_interval)
    sweep_state: dict = {"last_sweep": datetime.utcnow().timestamp()}
    while True:
        try:
            job = await _claim_due_job()
            if job is not None:
                await _run_job(job)
                # 连续消费：队列有积压时不空等轮询间隔。
                continue
            await _sweep_expired_memories_if_due(sweep_state)
        except asyncio.CancelledError:
            log.info("memory_worker.stopped")
            raise
        except Exception as exc:  # noqa: BLE001
            log.error("memory_worker.loop_failed", error=f"{type(exc).__name__}: {exc}"[:200])
        await asyncio.sleep(settings.memory_worker_poll_interval)
