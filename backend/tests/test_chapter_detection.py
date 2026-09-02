import json

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from app.services import chapter_detection as detection


def _body(label: str) -> str:
    return (label + "正文内容推动人物和情节发展。") * 35


def _rule(**overrides):
    payload = {
        "chapter_units": ["话"],
        "number_styles": ["financial_chinese"],
        "prefixes_to_ignore": ["※目录※"],
        "suffixes_to_ignore": [],
        "line_mode": "whole_line",
        "title_boundary": "line_end",
        "title_position": "after_number_unit",
        "volume_units": [],
        "numbering_scope": "global",
        "intro_markers": ["序章"],
        "tail_markers": ["尾声"],
        "expected_chapter_count": 3,
        "example_candidate_ids": ["D1L1", "D1L3", "D1L5"],
        "confidence": 0.96,
        "reason": "三个候选使用相同前缀和连续的财务中文话数",
    }
    payload.update(overrides)
    return detection.ChapterRulePayload.model_validate(payload)


def _unknown_format_doc() -> Document:
    return Document(page_content="\n".join([
        "※目录※ 第壹话 初遇",
        _body("一"),
        "※目录※ 第贰话 冲突",
        _body("二"),
        "※目录※ 第叁话 和解",
        _body("三"),
    ]))


def test_extract_candidates_keeps_context_and_limit():
    docs = [_unknown_format_doc()]
    candidates = detection.extract_chapter_candidates(docs, limit=2)
    assert len(candidates) == 2
    assert candidates[0].id == "D1L1"
    assert candidates[0].following.startswith("一正文")
    assert candidates[-1].id == "D1L5"


def test_safe_dsl_applies_unknown_chapter_format():
    validated = detection.validate_and_apply_rule(
        [_unknown_format_doc()], _rule(), "unknown.txt", "file-1"
    )
    assert validated.result.parser_mode == "llm_assisted"
    assert validated.result.chapter_count == 3
    assert validated.result.unassigned_chunk_count == 0
    chapters = list(dict.fromkeys(doc.metadata["chapter"] for doc in validated.result.documents))
    assert chapters == ["第壹话 初遇", "第贰话 冲突", "第叁话 和解"]


def test_raw_regex_field_is_rejected():
    payload = _rule().model_dump()
    payload["regex"] = ".*"
    with pytest.raises(ValidationError):
        detection.ChapterRulePayload.model_validate(payload)


def test_low_confidence_rule_is_rejected(monkeypatch):
    monkeypatch.setattr(detection.settings, "chapter_detection_confidence_threshold", 0.75)
    with pytest.raises(detection.ChapterDetectionError, match="置信度"):
        detection.validate_and_apply_rule(
            [_unknown_format_doc()], _rule(confidence=0.5), "unknown.txt", "file-1"
        )


def test_global_number_disorder_is_rejected():
    doc = Document(page_content="\n".join([
        "※目录※ 第肆话 四", _body("四"),
        "※目录※ 第壹话 一", _body("一"),
        "※目录※ 第叁话 三", _body("三"),
        "※目录※ 第贰话 二", _body("二"),
    ]))
    rule = _rule(
        expected_chapter_count=4,
        example_candidate_ids=["D1L1", "D1L3", "D1L7"],
    )
    with pytest.raises(detection.ChapterDetectionError, match="out_of_order"):
        detection.validate_and_apply_rule([doc], rule, "bad.txt", "bad-1")


def test_per_volume_number_reset_is_valid():
    doc = Document(page_content="\n".join([
        "第一卷", "第一章 开端", _body("一一"), "第二章 发展", _body("一二"),
        "第二卷", "第一章 转折", _body("二一"), "第二章 结局", _body("二二"),
    ]))
    rule = _rule(
        chapter_units=["章"],
        number_styles=["chinese"],
        prefixes_to_ignore=[],
        volume_units=["卷"],
        numbering_scope="per_volume",
        expected_chapter_count=4,
        example_candidate_ids=["D1L2", "D1L4", "D1L9"],
    )
    validated = detection.validate_and_apply_rule([doc], rule, "volumes.txt", "vol-1")
    assert validated.result.chapter_count == 4


def test_deterministic_quality_marks_long_unrecognized_text():
    from app.services.novel_service import split_novel_documents

    docs = [Document(page_content="没有章节标题。" * 20_000)]
    result = split_novel_documents(docs, "plain.txt", "plain-1")
    reasons = detection.assess_deterministic_quality(docs, result)
    assert "chapter_count_zero" in reasons
    assert "long_text_with_too_few_chapters" in reasons


@pytest.mark.asyncio
async def test_discovery_retries_invalid_json_once(monkeypatch):
    calls = 0
    valid = _rule().model_dump()

    class FakeModel:
        async def ainvoke(self, messages):
            nonlocal calls
            calls += 1
            if calls == 1:
                return AIMessage(content="not-json")
            return AIMessage(content=json.dumps(valid, ensure_ascii=False))

    monkeypatch.setattr(detection.settings, "chapter_detection_retries", 1)
    monkeypatch.setattr(detection.settings, "chapter_detection_confidence_threshold", 0.75)
    discovered = await detection.discover_chapter_rule([_unknown_format_doc()], llm=FakeModel())
    assert discovered.rule.expected_chapter_count == 3
    assert calls == 2


def test_rule_serialization_roundtrip():
    rule = _rule()
    assert detection.deserialize_rule(detection.serialize_rule(rule)) == rule


def test_cache_match_requires_hash_model_prompt_and_parser_version():
    from app.services import kb_service
    from app.services.novel_service import CHAPTER_PARSER_VERSION

    cache = {
        "chapter_rule_validated": True,
        "chapter_rule_json": detection.serialize_rule(_rule()),
        "source_hash": "abc",
        "chapter_parser_version": CHAPTER_PARSER_VERSION,
        "chapter_detection_prompt_version": detection.settings.chapter_detection_prompt_version,
        "chapter_detection_model": detection.settings.chapter_detection_model,
    }
    assert kb_service._cache_matches(cache, "abc") is True
    cache["source_hash"] = "changed"
    assert kb_service._cache_matches(cache, "abc") is False

_UNKNOWN_PREFIX_CASES = [
    "【正文目录】", "@@chapter@@", "--回目--", "[TABLE]", "章节导航:",
    "书籍目录 >>", "◇◇", "卷内索引-", "正文开始::", "#CHAPTER#",
    "【回目索引】", "===目录===", "小说正文/", "CONTENTS:", "篇章列表-",
    "章节索引>>>", "〔目录〕", "＊回目＊", "正文目录·", "BOOK-CONTENT-",
]


@pytest.mark.parametrize("prefix", _UNKNOWN_PREFIX_CASES)
def test_twenty_unknown_prefix_formats_are_safely_compiled(prefix):
    doc = Document(page_content="\n".join([
        f"{prefix}第1话 起点", _body("一"),
        f"{prefix}第2话 转折", _body("二"),
        f"{prefix}第3话 结局", _body("三"),
    ]))
    rule = _rule(
        number_styles=["arabic"],
        prefixes_to_ignore=[prefix],
        example_candidate_ids=["D1L1", "D1L3", "D1L5"],
    )
    validated = detection.validate_and_apply_rule([doc], rule, "formats.txt", "formats-1")
    assert validated.result.chapter_count == 3
    assert validated.result.parser_mode == "llm_assisted"

@pytest.mark.asyncio
async def test_valid_cached_rule_skips_llm(monkeypatch):
    from app.services import kb_service
    from app.services.novel_service import CHAPTER_PARSER_VERSION

    doc = _unknown_format_doc()
    loaded = kb_service.LoadedNovel([doc], "utf-8", "digest")
    cache = {
        "chapter_rule_validated": True,
        "chapter_rule_json": detection.serialize_rule(_rule()),
        "source_hash": "digest",
        "chapter_parser_version": CHAPTER_PARSER_VERSION,
        "chapter_detection_prompt_version": detection.settings.chapter_detection_prompt_version,
        "chapter_detection_model": detection.settings.chapter_detection_model,
    }

    async def must_not_call(*args, **kwargs):
        raise AssertionError("LLM should not be called for a valid cache")

    monkeypatch.setattr(kb_service, "discover_chapter_rule", must_not_call)
    validated = await kb_service._resolve_assisted_rule(loaded, cache, "cached.txt", "cached-1")
    assert validated.result.chapter_count == 3


@pytest.mark.asyncio
async def test_changed_hash_forces_new_discovery(monkeypatch):
    from app.services import kb_service
    from app.services.novel_service import CHAPTER_PARSER_VERSION

    doc = _unknown_format_doc()
    loaded = kb_service.LoadedNovel([doc], "utf-8", "new-digest")
    cache = {
        "chapter_rule_validated": True,
        "chapter_rule_json": detection.serialize_rule(_rule()),
        "source_hash": "old-digest",
        "chapter_parser_version": CHAPTER_PARSER_VERSION,
        "chapter_detection_prompt_version": detection.settings.chapter_detection_prompt_version,
        "chapter_detection_model": detection.settings.chapter_detection_model,
    }
    calls = 0

    async def discover(*args, **kwargs):
        nonlocal calls
        calls += 1
        return detection.ChapterRuleDiscovery(_rule(), "llm")

    monkeypatch.setattr(kb_service, "discover_chapter_rule", discover)
    validated = await kb_service._resolve_assisted_rule(loaded, cache, "changed.txt", "changed-1")
    assert validated.result.chapter_count == 3
    assert calls == 1
