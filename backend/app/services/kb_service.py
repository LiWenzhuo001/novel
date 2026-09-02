"""Knowledge-base ingestion, chapter-aware indexing and file lifecycle management."""
from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_core.documents import Document
from sqlalchemy import select

from app.config import settings
from app.core.context import get_current_user
from app.core.logging_config import get_logger
from app.core.metrics import metrics
from app.core.rag import delete_by_file_id, replace_documents
from app.db import AsyncSessionLocal
from app.db.models import KnowledgeFile
from app.services.chapter_detection import (
    ChapterDetectionError,
    ValidatedChapterRule,
    assess_deterministic_quality,
    deserialize_rule,
    discover_chapter_rule,
    serialize_rule,
    source_hash,
    validate_and_apply_rule,
)
from app.services.novel_service import CHAPTER_PARSER_VERSION, NovelSplitResult, split_novel_documents

log = get_logger("kb")
RAW_DIR = Path(settings.raw_dir).resolve()
LEASE_SECONDS = 15 * 60
MAX_INDEX_ATTEMPTS = 3


class ReindexConflict(RuntimeError):
    pass


class RawFileMissing(FileNotFoundError):
    pass


@dataclass(frozen=True)
class LoadedNovel:
    """已加载小说文档、检测编码和原文哈希。"""
    documents: list[Document]
    detected_encoding: str | None
    source_digest: str


def _decode_quality(text: str, encoding: str) -> float:
    if not text:
        return float("-inf")
    cjk = sum("\u3400" <= char <= "\u9fff" for char in text)
    common = sum(char in "的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主" for char in text)
    private_use = sum("\ue000" <= char <= "\uf8ff" for char in text)
    kana = sum("\u3040" <= char <= "\u30ff" for char in text)
    replacement = text.count("\ufffd")
    controls = sum(ord(char) < 32 and char not in "\n\r\t" for char in text)
    utf_bonus = 8 if encoding.startswith("utf-8") else 0
    return cjk + common * 2 + utf_bonus - private_use * 8 - kana * 4 - replacement * 20 - controls * 10


def _load_text_document(path: Path, raw: bytes | None = None) -> tuple[list[Document], str]:
    content = raw if raw is not None else path.read_bytes()
    if not content:
        raise ValueError(f"文本文件为空：{path.resolve()}")
    if content.startswith(b"\xef\xbb\xbf"):
        candidates = [("utf-8-sig", content.decode("utf-8-sig"))]
    else:
        candidates: list[tuple[str, str]] = []
        for encoding in ("utf-8", "gb18030", "gbk", "big5"):
            try:
                candidates.append((encoding, content.decode(encoding)))
            except UnicodeDecodeError:
                continue
    if not candidates:
        raise UnicodeError(
            f"无法解码文本文件 {path.resolve()}，已尝试：utf-8-sig, utf-8, gb18030, gbk, big5"
        )
    detected_encoding, text = max(candidates, key=lambda item: _decode_quality(item[1], item[0]))
    return [Document(
        page_content=text,
        metadata={
            "source": str(path.resolve()),
            "source_type": path.suffix.lstrip(".").lower(),
            "has_real_page": False,
        },
    )], detected_encoding


def _load_documents(path: Path, ext: str, raw: bytes | None = None) -> tuple[list[Document], str | None]:
    """按文件扩展名选择 TXT、PDF 或 DOCX 加载器。"""
    if not path.exists():
        raise RawFileMissing(f"原始文件不存在：{path.resolve()}")
    if not path.is_file():
        raise RawFileMissing(f"原始文件路径不是文件：{path.resolve()}")
    ext = ext.lower()
    if ext in (".txt", ".md"):
        return _load_text_document(path, raw)
    if ext == ".pdf":
        docs = PyPDFLoader(str(path)).load()
        for doc in docs:
            doc.metadata.update({"source_type": "pdf", "has_real_page": True})
        return docs, None
    if ext == ".docx":
        docs = Docx2txtLoader(str(path)).load()
        for doc in docs:
            doc.metadata.update({"source_type": "docx", "has_real_page": False})
        return docs, None
    raise ValueError(f"不支持的文件类型：{ext}")


def _load_novel(path: Path, ext: str) -> LoadedNovel:
    """读取原文并计算哈希，为索引和章节规则缓存提供版本标识。"""
    raw = path.read_bytes()
    docs, detected_encoding = _load_documents(path, ext, raw)
    return LoadedNovel(docs, detected_encoding, source_hash(raw))


def _load_and_split(save_path: Path, ext: str, filename: str, file_id: str) -> NovelSplitResult:
    loaded = _load_novel(save_path, ext)
    result = split_novel_documents(loaded.documents, filename, file_id)
    result.detected_encoding = loaded.detected_encoding
    return result


def _root_error(exc: Exception) -> Exception:
    root: Exception = exc
    seen: set[int] = set()
    while id(root) not in seen:
        seen.add(id(root))
        nested = root.__cause__ or root.__context__
        if not isinstance(nested, Exception):
            break
        root = nested
    return root


def _format_index_error(exc: Exception, path: Path) -> str:
    root = _root_error(exc)
    detail = str(root).strip() or str(exc).strip() or "未知错误"
    return f"{type(root).__name__}: {detail}（文件：{path.resolve()}）"[:500]


def _cache_matches(cache: dict[str, object], digest: str) -> bool:
    return bool(
        cache.get("chapter_rule_validated")
        and cache.get("chapter_rule_json")
        and cache.get("source_hash") == digest
        and cache.get("chapter_parser_version") == CHAPTER_PARSER_VERSION
        and cache.get("chapter_detection_prompt_version") == settings.chapter_detection_prompt_version
        and cache.get("chapter_detection_model") == settings.chapter_detection_model
    )


async def _resolve_assisted_rule(
    loaded: LoadedNovel,
    cached: dict[str, object],
    filename: str,
    file_id: str,
) -> ValidatedChapterRule:
    """优先复用同源已验证规则，否则调用模型发现并验证新规则。"""
    if _cache_matches(cached, loaded.source_digest):
        try:
            cached_rule = deserialize_rule(str(cached["chapter_rule_json"]))
            validated = await asyncio.to_thread(
                validate_and_apply_rule, loaded.documents, cached_rule, filename, file_id
            )
            log.info("chapter_detection.cache_hit", file_id=file_id)
            return validated
        except Exception as exc:  # noqa: BLE001
            log.warning("chapter_detection.cache_invalid", file_id=file_id, error=str(exc)[:300])
    discovery = await discover_chapter_rule(loaded.documents)
    validated = await asyncio.to_thread(
        validate_and_apply_rule, loaded.documents, discovery.rule, filename, file_id
    )
    log.info(
        "chapter_detection.llm_validated",
        file_id=file_id,
        confidence=discovery.rule.confidence,
        chapters=validated.result.chapter_count,
        validation=validated.validation,
    )
    return validated


def _set_index_metadata(
    record: KnowledgeFile,
    result: NovelSplitResult,
    digest: str,
    validated_rule: ValidatedChapterRule | None = None,
) -> None:
    """把分块、章节、模型和解析规则写回知识库文件记录。"""
    record.embedding_model = settings.embedding_model
    record.embed_dim = settings.embed_dim
    record.chunk_size = settings.novel_chunk_size
    record.chunk_overlap = settings.novel_chunk_overlap
    record.indexed_at = datetime.utcnow()
    record.source_hash = digest
    record.chapter_count = result.chapter_count
    record.unassigned_chunk_count = result.unassigned_chunk_count
    record.chapter_parse_status = "ok" if result.chapter_count > 0 else "unrecognized"
    record.chapter_parser_mode = result.parser_mode
    record.chapter_parser_version = result.parser_version
    record.detected_encoding = result.detected_encoding
    record.index_warning = "；".join(result.warnings) or None
    record.chapter_detection_requested = False
    record.chapter_detection_error = None
    if validated_rule is not None:
        record.chapter_rule_json = serialize_rule(validated_rule.rule)
        record.chapter_rule_confidence = validated_rule.rule.confidence
        record.chapter_rule_validated = True
        record.chapter_detection_model = settings.chapter_detection_model
        record.chapter_detection_prompt_version = settings.chapter_detection_prompt_version
    elif result.parser_mode != "llm_assisted":
        record.chapter_rule_json = None
        record.chapter_rule_confidence = None
        record.chapter_rule_validated = False
        record.chapter_detection_model = None
        record.chapter_detection_prompt_version = None
    record.index_version = (
        f"{settings.embedding_model}:{settings.embed_dim}:"
        f"{settings.novel_chunk_size}:{settings.novel_chunk_overlap}:"
        f"{result.parser_version}"
    )


_INDEX_PROGRESS_MESSAGES = {
    "pending": "等待索引任务启动",
    "loading": "正在读取文件",
    "parsing": "正在解析文本和章节",
    "analyzing_chapters": "正在分析章节格式",
    "building_embeddings": "正在建立向量索引",
    "switching": "正在切换索引",
    "completed": "索引完成",
    "failed": "索引失败",
}


async def _update_index_progress(
    file_id: str,
    owner: str,
    lease_id: str,
    stage: str,
    progress: int,
    message: str | None = None,
    *,
    reset: bool = False,
) -> bool:
    """Update persisted index progress only for the current leased task.

    Progress is deliberately stage-based rather than embedding-batch based.  A
    stale worker can still finish its local work, but it must not overwrite the
    progress of a newer worker because every write re-checks the lease.
    """
    stage = stage if stage in _INDEX_PROGRESS_MESSAGES else "building_embeddings"
    requested = max(0, min(100, int(progress)))
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(KnowledgeFile).where(
                KnowledgeFile.id == file_id,
                KnowledgeFile.user_id == owner,
            )
        )
        record = result.scalars().first()
        if not record or record.lease_id != lease_id:
            return False
        previous = getattr(record, "index_progress", None)
        if reset or previous is None:
            effective = requested
        else:
            effective = max(0, min(100, int(previous)), requested)
        record.index_stage = stage
        record.index_progress = effective
        record.index_message = message or _INDEX_PROGRESS_MESSAGES[stage]
        await session.commit()
    return True


async def _set_detection_progress(file_id: str, owner: str, lease_id: str, message: str) -> None:
    """Backward-compatible wrapper for chapter-detection progress updates."""
    await _update_index_progress(
        file_id,
        owner,
        lease_id,
        "analyzing_chapters",
        30,
        message,
    )


async def create_pending_file(filename: str, content: bytes) -> dict:
    """保存原始文件并创建待索引记录，立即返回任务信息。"""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in (".pdf", ".docx", ".txt", ".md"):
        raise ValueError(f"不支持的文件类型：{ext}")
    file_id = uuid.uuid4().hex[:12]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    save_path = RAW_DIR / f"{file_id}{ext}"
    save_path.write_bytes(content)
    async with AsyncSessionLocal() as session:
        session.add(KnowledgeFile(
            id=file_id,
            filename=filename,
            filetype=ext.lstrip("."),
            size=len(content),
            chunks=0,
            status="pending",
            index_stage="pending",
            index_progress=0,
            index_message=_INDEX_PROGRESS_MESSAGES["pending"],
            user_id=get_current_user(),
            domain="novel",
            source_hash=source_hash(content),
            chapter_detection_requested=False,
            created_at=datetime.utcnow(),
        ))
        await session.commit()
    return {
        "id": file_id,
        "filename": filename,
        "filetype": ext.lstrip("."),
        "size": len(content),
        "chunks": 0,
        "status": "pending",
        "index_stage": "pending",
        "index_progress": 0,
        "index_message": _INDEX_PROGRESS_MESSAGES["pending"],
        "domain": "novel",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


async def run_indexing(
    file_id: str,
    filename: str,
    ext: str,
    user_id: str | None = None,
    use_llm_chapter_detection: bool = False,
) -> None:
    """执行读取、章节解析、Embedding 和原子替换的后台索引状态机。"""
    save_path = RAW_DIR / f"{file_id}{ext}"
    lease_id = uuid.uuid4().hex
    owner = user_id or get_current_user()
    previous_chunks = 0
    cached: dict[str, object] = {}
    llm_requested = use_llm_chapter_detection
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(KnowledgeFile).where(KnowledgeFile.id == file_id, KnowledgeFile.user_id == owner)
        )
        record = result.scalars().first()
        if record is None or record.status == "indexed":
            return
        now = datetime.utcnow()
        if record.status == "indexing" and record.lease_until and record.lease_until > now:
            return
        previous_chunks = record.chunks or 0
        llm_requested = llm_requested or bool(record.chapter_detection_requested)
        cached = {
            "source_hash": record.source_hash,
            "chapter_rule_json": record.chapter_rule_json,
            "chapter_rule_validated": record.chapter_rule_validated,
            "chapter_parser_version": record.chapter_parser_version,
            "chapter_detection_prompt_version": record.chapter_detection_prompt_version,
            "chapter_detection_model": record.chapter_detection_model,
        }
        if record.attempts >= MAX_INDEX_ATTEMPTS:
            record.index_stage = "failed"
            record.index_message = (
                "索引重试次数超过上限，旧索引仍可使用"
                if previous_chunks else "索引重试次数超过上限"
            )
            if previous_chunks:
                record.status = "indexed"
                record.index_warning = "重新索引重试次数超过上限，旧索引仍可使用"
            else:
                record.status = "failed"
                record.error = record.error or "索引重试次数超过上限"
            await session.commit()
            return
        # 先取得租约再执行耗时步骤，后续进度写入会用 lease_id 防止旧任务覆盖新任务。
        record.status = "indexing"
        record.error = None
        record.index_warning = None
        record.chapter_detection_error = None
        record.index_stage = "loading"
        record.index_progress = 5
        record.index_message = _INDEX_PROGRESS_MESSAGES["loading"]
        record.attempts = (record.attempts or 0) + 1
        record.lease_id = lease_id
        record.lease_until = now + timedelta(seconds=LEASE_SECONDS)
        await session.commit()

    split_result: NovelSplitResult | None = None
    validated_rule: ValidatedChapterRule | None = None
    failure: str | None = None
    detection_failure = False
    try:
        loaded = await asyncio.to_thread(_load_novel, save_path, ext)
        await _update_index_progress(
            file_id, owner, lease_id, "parsing", 20,
        )
        split_result = await asyncio.to_thread(
            split_novel_documents, loaded.documents, filename, file_id
        )
        split_result.detected_encoding = loaded.detected_encoding
        quality_reasons = assess_deterministic_quality(loaded.documents, split_result)
        if llm_requested and quality_reasons:
            await _set_detection_progress(file_id, owner, lease_id, "正在分析章节格式")
            validated_rule = await _resolve_assisted_rule(
                loaded, cached, filename, file_id
            )
            split_result = validated_rule.result
            split_result.detected_encoding = loaded.detected_encoding
        if not split_result.documents:
            raise ValueError("文本清洗后没有可索引内容")
        await _update_index_progress(
            file_id, owner, lease_id, "building_embeddings", 60,
        )
        # Embedding 全部成功后才替换旧索引，重建失败时旧数据仍可检索。
        await replace_documents(file_id, split_result.documents, user_id=owner)
        await _update_index_progress(
            file_id, owner, lease_id, "switching", 95,
        )
        digest = loaded.source_digest
        log.info(
            "file.indexed",
            file_id=file_id,
            chunks=len(split_result.documents),
            chapters=split_result.chapter_count,
            parser_mode=split_result.parser_mode,
            encoding=split_result.detected_encoding,
        )
    except Exception as exc:  # noqa: BLE001
        detection_failure = isinstance(_root_error(exc), ChapterDetectionError) or isinstance(exc, ChapterDetectionError)
        failure = _format_index_error(exc, save_path)
        digest = str(cached.get("source_hash") or "")
        log.error(
            "file.index_failed",
            file_id=file_id,
            path=str(save_path.resolve()),
            error_type=type(_root_error(exc)).__name__,
            error=failure,
        )

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(KnowledgeFile).where(KnowledgeFile.id == file_id, KnowledgeFile.user_id == owner)
        )
        record = result.scalars().first()
        if not record or record.lease_id != lease_id:
            return
        if split_result is not None and failure is None:
            record.status = "indexed"
            record.error = None
            record.chunks = len(split_result.documents)
            _set_index_metadata(record, split_result, digest, validated_rule)
            record.index_stage = "completed"
            record.index_progress = 100
            record.index_message = _INDEX_PROGRESS_MESSAGES["completed"]
        elif previous_chunks:
            # 重新索引失败不影响已存在的旧向量，只把错误作为警告反馈给用户。
            record.status = "indexed"
            record.error = None
            record.index_stage = "failed"
            record.index_message = "索引失败，旧索引仍可使用"
            record.index_warning = (
                f"模型辅助章节识别失败，旧索引仍可使用：{failure}"
                if detection_failure else f"重新索引失败，旧索引仍可使用：{failure}"
            )
            record.chapter_detection_error = failure if detection_failure else None
            record.chapter_detection_requested = False
            if detection_failure:
                # 模型章节识别失败不计入普通索引重试次数。
                record.attempts = max(0, (record.attempts or 1) - 1)
        else:
            record.status = "failed"
            record.error = failure
            record.index_stage = "failed"
            record.index_message = "索引失败"
            record.chapter_detection_error = failure if detection_failure else None
            record.chapter_detection_requested = False
            if detection_failure:
                record.attempts = max(0, (record.attempts or 1) - 1)
        record.lease_id = None
        record.lease_until = None
        await session.commit()


async def prepare_reindex(file_id: str, user_id: str | None = None) -> tuple[str, str, str, str, bool]:
    """校验文件可重建后重置任务状态并返回后台索引参数。"""
    owner = user_id or get_current_user()
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(KnowledgeFile).where(KnowledgeFile.id == file_id, KnowledgeFile.user_id == owner)
        )
        record = result.scalars().first()
        if not record:
            raise ValueError("知识库文件不存在")
        if record.status in {"pending", "indexing"}:
            raise ReindexConflict("文件正在索引，请等待当前任务完成")
        ext = "." + (record.filetype or "")
        save_path = RAW_DIR / f"{file_id}{ext}"
        if not save_path.is_file():
            raise RawFileMissing(f"原始文件不存在，无法重新索引：{save_path.resolve()}")
        record.status = "pending"
        record.error = None
        record.index_stage = "pending"
        record.index_progress = 0
        record.index_message = _INDEX_PROGRESS_MESSAGES["pending"]
        record.index_warning = "等待重新索引"
        record.chapter_detection_requested = True
        record.attempts = 0
        record.lease_id = None
        record.lease_until = None
        await session.commit()
        return record.id, record.filename, ext, owner, True


async def reindex_file(file_id: str, user_id: str | None = None) -> dict:
    job = await prepare_reindex(file_id, user_id)
    await run_indexing(job[0], job[1], job[2], job[3], job[4])
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(KnowledgeFile).where(KnowledgeFile.id == file_id, KnowledgeFile.user_id == job[3])
        )
        record = result.scalars().first()
        if not record:
            raise RuntimeError("重新索引后文件记录丢失")
        return {
            "file_id": file_id,
            "status": record.status,
            "chunks": record.chunks,
            "chapters": record.chapter_count,
            "index_version": record.index_version,
            "warning": record.index_warning,
        }


async def recover_stale_indexing() -> list[tuple[str, str, str, str, bool]]:
    """启动时恢复过期租约任务，超过重试上限则保留旧索引或标记失败。"""
    now = datetime.utcnow()
    tasks: list[tuple[str, str, str, str, bool]] = []
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(KnowledgeFile).where(
                (KnowledgeFile.status == "pending")
                | ((KnowledgeFile.status == "indexing") & (KnowledgeFile.lease_until < now))
            )
        )
        for record in result.scalars().all():
            if record.attempts >= MAX_INDEX_ATTEMPTS:
                if record.chunks:
                    record.status = "indexed"
                    record.index_stage = "failed"
                    record.index_message = "索引恢复失败，旧索引仍可使用"
                    record.index_warning = "恢复索引任务失败次数超过上限，旧索引仍可使用"
                else:
                    record.status = "failed"
                    record.index_stage = "failed"
                    record.index_message = "索引恢复失败"
                    record.error = record.error or "索引重试次数超过上限"
                continue
            record.status = "pending"
            record.index_stage = "pending"
            record.index_progress = 0
            record.index_message = "等待恢复索引任务"
            record.lease_id = None
            record.lease_until = None
            tasks.append((
                record.id,
                record.filename,
                "." + (record.filetype or ""),
                record.user_id,
                bool(record.chapter_detection_requested),
            ))
        await session.commit()
    metrics.incr("index_backlog", len(tasks))
    return tasks


async def list_files() -> list:
    """返回当前用户可见的知识库文件（自己的 + 系统默认书）及索引、章节和进度信息。"""
    current_user = get_current_user()
    visible_users = [current_user]
    if settings.system_user != current_user:
        visible_users.append(settings.system_user)
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(KnowledgeFile)
            .where(KnowledgeFile.user_id.in_(visible_users))
            .order_by(KnowledgeFile.created_at.desc())
        )
        rows = result.scalars().all()
    return [
        {
            "id": record.id,
            "filename": record.filename,
            "filetype": record.filetype,
            "size": record.size,
            "chunks": record.chunks,
            "status": record.status,
            "index_stage": record.index_stage,
            "index_progress": record.index_progress,
            "index_message": record.index_message,
            "error": record.error,
            "domain": record.domain,
            "is_system": record.user_id == settings.system_user,
            "index_version": record.index_version,
            "embedding_model": record.embedding_model,
            "embed_dim": record.embed_dim,
            "chunk_size": record.chunk_size,
            "chunk_overlap": record.chunk_overlap,
            "indexed_at": record.indexed_at.strftime("%Y-%m-%d %H:%M") if record.indexed_at else None,
            "chapter_count": record.chapter_count,
            "unassigned_chunk_count": record.unassigned_chunk_count,
            "chapter_parse_status": record.chapter_parse_status,
            "chapter_parser_mode": record.chapter_parser_mode,
            "chapter_parser_version": record.chapter_parser_version,
            "chapter_index_stale": record.chapter_parser_version != CHAPTER_PARSER_VERSION,
            "detected_encoding": record.detected_encoding,
            "index_warning": record.index_warning,
            "chapter_rule_confidence": record.chapter_rule_confidence,
            "chapter_rule_validated": record.chapter_rule_validated,
            "chapter_detection_model": record.chapter_detection_model,
            "chapter_detection_error": record.chapter_detection_error,
            "created_at": record.created_at.strftime("%Y-%m-%d %H:%M") if record.created_at else "",
        }
        for record in rows
    ]


async def delete_file(file_id: str) -> bool:
    """删除原文、向量和文件记录，并严格限制在当前用户范围内。"""
    owner = get_current_user()
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(KnowledgeFile).where(KnowledgeFile.id == file_id, KnowledgeFile.user_id == owner)
        )
        record = result.scalars().first()
        if not record:
            return False
        await delete_by_file_id(file_id, user_id=owner)
        for path in RAW_DIR.glob(f"{file_id}.*"):
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                log.warning("file.raw_delete_failed", file_id=file_id, path=str(path), error=str(exc))
        await session.delete(record)
        await session.commit()
        return True
