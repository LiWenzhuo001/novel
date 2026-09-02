from pathlib import Path

import pytest

from app.services import kb_service


@pytest.mark.parametrize(
    ("encoding", "expected"),
    [
        ("utf-8", "utf-8"),
        ("utf-8-sig", "utf-8-sig"),
        ("gb18030", "gb18030"),
        ("big5", "big5"),
    ],
)
def test_load_text_document_encoding_fallback(tmp_path: Path, encoding: str, expected: str):
    path = tmp_path / "novel.txt"
    path.write_bytes("第一回 中文正文。".encode(encoding))
    docs, detected = kb_service._load_text_document(path)
    assert docs[0].page_content == "第一回 中文正文。"
    assert docs[0].metadata["has_real_page"] is False
    assert detected == expected


def test_load_documents_missing_file_has_absolute_path(tmp_path: Path):
    missing = tmp_path / "missing.txt"
    with pytest.raises(kb_service.RawFileMissing, match="原始文件不存在") as exc:
        kb_service._load_documents(missing, ".txt")
    assert str(missing.resolve()) in str(exc.value)


def test_raw_dir_is_absolute_and_rooted_in_backend():
    assert kb_service.RAW_DIR.is_absolute()
    assert kb_service.RAW_DIR.name == "raw"
    assert kb_service.RAW_DIR.parent.name == "data"
    assert kb_service.RAW_DIR.parent.parent.name == "backend"

@pytest.mark.asyncio
async def test_initial_index_never_enables_llm_chapter_detection(monkeypatch):
    from types import SimpleNamespace
    from app.services import kb_service

    class FakeScalars:
        def first(self):
            return record

    class FakeResult:
        def scalars(self):
            return FakeScalars()

    class FakeSession:
        async def execute(self, statement):
            return FakeResult()

        async def commit(self):
            return None

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

    record = SimpleNamespace(
        status="pending", lease_until=None, chunks=0, attempts=0,
        error=None, index_warning=None, chapter_detection_error=None,
        chapter_detection_requested=False, lease_id=None,
        source_hash=None, chapter_rule_json=None, chapter_rule_validated=False,
        chapter_parser_version=None, chapter_detection_prompt_version=None,
        chapter_detection_model=None,
    )
    monkeypatch.setattr(kb_service, "AsyncSessionLocal", lambda: FakeSession())
    monkeypatch.setattr(kb_service, "_load_novel", lambda *args: (_ for _ in ()).throw(RuntimeError("stop-after-mode-capture")))
    monkeypatch.setattr(kb_service, "_format_index_error", lambda *args: "stopped")
    await kb_service.run_indexing("f1", "book.txt", ".txt", "u1", False)
    assert record.chapter_detection_requested is False

@pytest.mark.asyncio
async def test_llm_detection_failure_keeps_old_index(monkeypatch):
    from types import SimpleNamespace
    from langchain_core.documents import Document
    from app.services import kb_service
    from app.services.chapter_detection import ChapterDetectionError
    from app.services.novel_service import NovelSplitResult

    class FakeScalars:
        def first(self):
            return record

    class FakeResult:
        def scalars(self):
            return FakeScalars()

    class FakeSession:
        async def execute(self, statement):
            return FakeResult()

        async def commit(self):
            return None

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

    record = SimpleNamespace(
        status="pending", lease_until=None, chunks=8, attempts=0,
        error=None, index_warning=None, chapter_detection_error=None,
        chapter_detection_requested=True, lease_id=None,
        source_hash="old", chapter_rule_json=None, chapter_rule_validated=False,
        chapter_parser_version=None, chapter_detection_prompt_version=None,
        chapter_detection_model=None,
    )
    low_result = NovelSplitResult(
        documents=[Document(page_content="旧索引之外的新普通分块", metadata={"chapter": "未分章"})],
        chapter_count=0,
        unassigned_chunk_count=1,
        parser_mode="none",
    )
    replaced = 0

    async def fail_detection(*args, **kwargs):
        raise ChapterDetectionError("invalid model rule")

    async def replace(*args, **kwargs):
        nonlocal replaced
        replaced += 1

    monkeypatch.setattr(kb_service, "AsyncSessionLocal", lambda: FakeSession())
    monkeypatch.setattr(kb_service, "_load_novel", lambda *args: kb_service.LoadedNovel([Document(page_content="无标题正文" * 200)], "utf-8", "new"))
    monkeypatch.setattr(kb_service, "split_novel_documents", lambda *args, **kwargs: low_result)
    monkeypatch.setattr(kb_service, "assess_deterministic_quality", lambda *args: ["chapter_count_zero"])
    monkeypatch.setattr(kb_service, "_resolve_assisted_rule", fail_detection)
    monkeypatch.setattr(kb_service, "replace_documents", replace)

    await kb_service.run_indexing("f1", "book.txt", ".txt", "u1", True)

    assert replaced == 0
    assert record.status == "indexed"
    assert record.chunks == 8
    assert record.attempts == 0
    assert "旧索引仍可使用" in record.index_warning
    assert "invalid model rule" in record.chapter_detection_error


@pytest.mark.asyncio
async def test_update_index_progress_requires_lease_and_is_monotonic(monkeypatch):
    from types import SimpleNamespace

    class FakeScalars:
        def first(self):
            return record

    class FakeResult:
        def scalars(self):
            return FakeScalars()

    class FakeSession:
        async def execute(self, statement):
            return FakeResult()

        async def commit(self):
            return None

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

    record = SimpleNamespace(
        lease_id="lease-1", index_progress=60, index_stage="building_embeddings",
        index_message=None,
    )
    monkeypatch.setattr(kb_service, "AsyncSessionLocal", lambda: FakeSession())

    assert await kb_service._update_index_progress(
        "f1", "u1", "lease-1", "parsing", 20,
    ) is True
    assert record.index_progress == 60
    assert record.index_stage == "parsing"

    assert await kb_service._update_index_progress(
        "f1", "u1", "wrong-lease", "switching", 95,
    ) is False
    assert record.index_progress == 60

    assert await kb_service._update_index_progress(
        "f1", "u1", "lease-1", "switching", 95,
    ) is True
    assert record.index_progress == 95


@pytest.mark.asyncio
async def test_update_index_progress_advances_with_current_lease(monkeypatch):
    from types import SimpleNamespace

    class FakeScalars:
        def first(self):
            return record

    class FakeResult:
        def scalars(self):
            return FakeScalars()

    class FakeSession:
        async def execute(self, statement):
            return FakeResult()

        async def commit(self):
            return None

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

    record = SimpleNamespace(
        lease_id="lease-1", index_progress=60, index_stage="building_embeddings",
        index_message=None,
    )
    monkeypatch.setattr(kb_service, "AsyncSessionLocal", lambda: FakeSession())

    assert await kb_service._update_index_progress(
        "f1", "u1", "lease-1", "switching", 95,
    ) is True
    assert record.index_progress == 95
    assert record.index_stage == "switching"
    assert record.index_message == "正在切换索引"
