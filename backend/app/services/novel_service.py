"""Deterministic novel cleaning, chapter detection and citation-aware chunking."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings

CHAPTER_PARSER_VERSION = "chapter-v3"
_NUMERALS = "0-9零〇○一二三四五六七八九十百千万两壹贰叁肆伍陆柒捌玖拾佰仟"
_DECORATION_RE = re.compile(r"^[=＊*#~\-—_【】\[\]（）()·•]+|[=＊*#~\-—_【】\[\]（）()·•]+$")
_CATALOG_PREFIX_RE = re.compile(
    r"^(?:(?:《[^》]*》)\s*)?(?:目录|目次|回目)\s*[:：·\-—]?\s*",
    re.IGNORECASE,
)
_MAIN_HEADING_RE = re.compile(
    rf"(?:(?:《[^》\n]{{1,40}}》\s*)?)(?:(?:卷\s*[{_NUMERALS}]+\s*)?)"
    rf"第\s*[{_NUMERALS}]+\s*[章节回卷部集](?:\s*[:：·\-—]?\s*.{{0,80}})?",
    re.IGNORECASE,
)
_ENGLISH_HEADING_RE = re.compile(r"(?:chapter|chap\.?|part)\s+\d+(?:\s*[:：.\-—]?\s*.{0,80})?", re.IGNORECASE)
_INTRO_HEADING_RE = re.compile(r"(?:序章|序言|前言|引子|楔子)(?:\s*[:：·\-—]?\s*.{0,80})?")
_TAIL_HEADING_RE = re.compile(r"(?:后记|尾声|终章|附录)(?:\s*[:：·\-—]?\s*.{0,80})?")
_INLINE_MARKER_RE = re.compile(
    rf"(?:^|(?<=[\n。！？]))\s*(?P<title>"
    rf"(?:(?:《[^》\n]{{1,40}}》\s*)?)(?:(?:卷\s*[{_NUMERALS}]+\s*)?)"
    rf"第\s*[{_NUMERALS}]+\s*[章节回卷部集][^\n。！？]{{0,80}})",
    re.IGNORECASE | re.MULTILINE,
)
_MULTI_SPACE_RE = re.compile(r"[ \t\u3000]+")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_INVISIBLE_RE = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")


@dataclass(frozen=True)
class ChapterHeading:
    """章节标题在清洗文本中的字符区间和标题类型。"""
    start: int
    end: int
    title: str
    kind: Literal["main", "intro", "tail"] = "main"


@dataclass
class NovelSplitResult:
    """章节解析和分块结果，包含章节统计、警告及解析模式。"""
    documents: list[Document]
    chapter_count: int
    unassigned_chunk_count: int
    parser_mode: Literal["strict", "inline_fallback", "llm_assisted", "none"]
    parser_version: str = CHAPTER_PARSER_VERSION
    warnings: list[str] = field(default_factory=list)
    detected_encoding: str | None = None


def clean_novel_text(text: str) -> str:
    """Normalize web/TXT artifacts while preserving paragraph boundaries."""
    text = unicodedata.normalize("NFKC", text or "")
    text = _INVISIBLE_RE.sub("", text).replace("\u00a0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_RE.sub("", text)
    lines = [_MULTI_SPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
    output: list[str] = []
    blank = False
    for line in lines:
        if not line:
            if not blank:
                output.append("")
            blank = True
            continue
        output.append(line)
        blank = False
    return "\n".join(output).strip()


def _strip_decorations(value: str) -> str:
    previous = value.strip()
    while True:
        current = _DECORATION_RE.sub("", previous).strip()
        if current == previous:
            return current[:100]
        previous = current


def _normalize_heading_candidate(candidate: str) -> str:
    """Remove known ebook catalog prefixes while preserving the actual heading."""
    return _CATALOG_PREFIX_RE.sub("", candidate).strip()[:100]


def _heading_kind(candidate: str) -> Literal["main", "intro", "tail"] | None:
    """识别候选标题是正文主章、引言还是尾声。"""
    if _MAIN_HEADING_RE.fullmatch(candidate) or _ENGLISH_HEADING_RE.fullmatch(candidate):
        return "main"
    # “前言。第一回……”属于正文内联标题，不能把整行吞成前言。
    if (_INTRO_HEADING_RE.fullmatch(candidate) or _TAIL_HEADING_RE.fullmatch(candidate)) and (
        _MAIN_HEADING_RE.search(candidate) or _ENGLISH_HEADING_RE.search(candidate)
    ):
        return None
    if _INTRO_HEADING_RE.fullmatch(candidate):
        return "intro"
    if _TAIL_HEADING_RE.fullmatch(candidate):
        return "tail"
    return None


def _find_strict_headings(text: str) -> list[ChapterHeading]:
    """按整行标题规则查找章节，优先作为确定性解析结果。"""
    headings: list[ChapterHeading] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        raw = line.rstrip("\r\n")
        leading = len(raw) - len(raw.lstrip())
        candidate = _normalize_heading_candidate(_strip_decorations(raw))
        kind = _heading_kind(candidate)
        if kind:
            start = offset + leading
            headings.append(ChapterHeading(start, offset + len(raw), candidate, kind))
        offset += len(line)
    if text and (not text.endswith("\n")) and not text.splitlines(keepends=True):
        candidate = _normalize_heading_candidate(_strip_decorations(text))
        kind = _heading_kind(candidate)
        if kind:
            headings.append(ChapterHeading(0, len(text), candidate, kind))
    return headings


def _find_inline_headings(text: str) -> list[ChapterHeading]:
    """查找正文段落开头的内联章节标题，作为兼容回退。"""
    candidates: list[ChapterHeading] = []
    last_start = -10_000
    for match in _INLINE_MARKER_RE.finditer(text):
        title = _strip_decorations(match.group("title"))
        marker_match = re.search(rf"第\s*[{_NUMERALS}]+\s*[章节回卷部集]", title)
        if not marker_match:
            continue
        # A conservative fallback: require wide separation so prose references do not fragment the book.
        if match.start("title") - last_start < 200:
            continue
        title = title[:100].rstrip(" :：·-—")
        candidates.append(ChapterHeading(match.start("title"), match.end("title"), title, "main"))
        last_start = match.start("title")
    return candidates if len(candidates) >= 2 else []


def _page_number(metadata: dict, fallback: int) -> int | None:
    if metadata.get("has_real_page") is False:
        return None
    return metadata.get("page", fallback)


def split_novel_documents(
    raw_docs: Iterable[Document],
    filename: str,
    file_id: str,
    *,
    headings_override: Sequence[Sequence[ChapterHeading]] | None = None,
    parser_mode_override: Literal["llm_assisted"] | None = None,
) -> NovelSplitResult:
    """Split a novel by chapters first and return parser quality statistics."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.novel_chunk_size,
        chunk_overlap=settings.novel_chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        add_start_index=True,
    )
    prepared: list[tuple[str, dict, int | None]] = []
    strict_by_doc: list[list[ChapterHeading]] = []
    for page_no, raw in enumerate(raw_docs):
        cleaned = clean_novel_text(raw.page_content)
        if not cleaned:
            continue
        metadata = dict(raw.metadata or {})
        prepared.append((cleaned, metadata, _page_number(metadata, page_no)))
        strict_by_doc.append(_find_strict_headings(cleaned))

    if headings_override is not None:
        if len(headings_override) != len(prepared):
            raise ValueError("headings_override_document_count_mismatch")
        parser_mode: Literal["strict", "inline_fallback", "llm_assisted", "none"] = (
            parser_mode_override or "llm_assisted"
        )
        headings_by_doc = [list(items) for items in headings_override]
    else:
        strict_count = sum(len(items) for items in strict_by_doc)
        # 整行规则优先；只有完全识别不到时才启用内联标题兼容模式。
        if strict_count:
            parser_mode = "strict"
            headings_by_doc = strict_by_doc
        else:
            fallback_by_doc = [_find_inline_headings(text) for text, _, _ in prepared]
            fallback_count = sum(len(items) for items in fallback_by_doc)
            if fallback_count >= 2:
                parser_mode = "inline_fallback"
                headings_by_doc = fallback_by_doc
            else:
                parser_mode = "none"
                headings_by_doc = [[] for _ in prepared]

    result: list[Document] = []
    chunk_no = 0
    current_chapter = "未分章"
    current_chapter_no = 0
    chapter_count = 0
    chapter_chunk_counts: dict[tuple[int, str], int] = {}

    for doc_index, (cleaned, base_meta, page) in enumerate(prepared):
        headings = headings_by_doc[doc_index]
        sections: list[tuple[int, int, str, int, str]] = []
        if not headings:
            sections.append((0, len(cleaned), current_chapter, current_chapter_no, "unassigned"))
        else:
            if headings[0].start > 0:
                prefix_title = current_chapter if result else "序章/前言"
                sections.append((0, headings[0].start, prefix_title, current_chapter_no, "intro"))
            for index, heading in enumerate(headings):
                if heading.kind == "intro" and current_chapter_no == 0:
                    next_no = 0
                else:
                    current_chapter_no += 1
                    next_no = current_chapter_no
                if heading.kind == "main":
                    chapter_count += 1
                current_chapter = heading.title
                end = headings[index + 1].start if index + 1 < len(headings) else len(cleaned)
                sections.append((heading.start, end, heading.title, next_no, heading.kind))

        # 每个分块都保留全书片段号和章节内片段号，供引用和邻居扩展使用。
        for start, end, chapter, chapter_no, chapter_kind in sections:
            section = cleaned[start:end].strip()
            if not section:
                continue
            for doc in splitter.create_documents([section], metadatas=[base_meta]):
                chunk_no += 1
                counter_key = (chapter_no, chapter)
                chapter_chunk_counts[counter_key] = chapter_chunk_counts.get(counter_key, 0) + 1
                local_start = int(doc.metadata.get("start_index", 0))
                char_start = start + local_start
                doc.metadata.update({
                    "domain": "novel",
                    "source": filename,
                    "file_id": file_id,
                    "chapter": chapter,
                    "chapter_no": chapter_no,
                    "chapter_kind": chapter_kind,
                    "page": page,
                    "chunk_no": chunk_no,
                    "chapter_chunk_no": chapter_chunk_counts[counter_key],
                    "char_start": char_start,
                    "char_end": char_start + len(doc.page_content),
                    "chapter_parser_version": CHAPTER_PARSER_VERSION,
                })
                result.append(doc)

    unassigned = sum(1 for doc in result if doc.metadata.get("chapter") == "未分章")
    warnings: list[str] = []
    if not result:
        warnings.append("文本为空或清洗后没有可索引内容")
    elif chapter_count == 0:
        warnings.append(f"已生成 {len(result)} 个片段，但未识别到章节标题")
    elif parser_mode == "inline_fallback":
        warnings.append("章节标题通过兼容模式识别，建议抽检章节边界")

    return NovelSplitResult(
        documents=result,
        chapter_count=chapter_count,
        unassigned_chunk_count=unassigned,
        parser_mode=parser_mode,
        warnings=warnings,
    )
