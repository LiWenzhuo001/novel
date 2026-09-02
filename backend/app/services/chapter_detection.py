"""LLM-assisted discovery of safe, validated chapter parsing rules."""
from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import unicodedata
from dataclasses import dataclass
from typing import Any, Literal, Sequence

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config import settings
from app.core.llm import get_llm
from app.core.logging_config import get_logger
from app.services.novel_service import (
    ChapterHeading,
    NovelSplitResult,
    clean_novel_text,
    split_novel_documents,
)

log = get_logger("chapter_detection")

_CHAPTER_UNITS = ("章", "回", "节", "幕", "话", "部", "集", "chapter", "chap.", "part")
_NUMBER_STYLES = ("arabic", "chinese", "financial_chinese", "roman")
_STRUCTURE_TOKEN_RE = re.compile(
    r"(?:第\s*[0-9零〇○一二三四五六七八九十百千万两壹贰叁肆伍陆柒捌玖拾佰仟IVXLCDMⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ]+\s*[章节回卷部集幕话]|"
    r"(?:chapter|chap\.?|part)\s+[0-9IVXLCDMⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ]+)",
    re.IGNORECASE,
)
_INTRO_TAIL_RE = re.compile(r"(?:序章|序言|前言|引子|楔子|后记|尾声|终章|附录)", re.IGNORECASE)
_SENTENCE_PUNCT_RE = re.compile(r"[。！？!?]")
_INVISIBLE_RE = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

_SIMPLE_DIGITS = {"零": 0, "〇": 0, "○": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_FINANCIAL_MAP = str.maketrans({"壹": "一", "贰": "二", "叁": "三", "肆": "四", "伍": "五", "陆": "六", "柒": "七", "捌": "八", "玖": "九", "拾": "十", "佰": "百", "仟": "千"})
_ROMAN_MAP = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
_ROMAN_TRANSLATE = str.maketrans({"Ⅰ": "I", "Ⅱ": "II", "Ⅲ": "III", "Ⅳ": "IV", "Ⅴ": "V", "Ⅵ": "VI", "Ⅶ": "VII", "Ⅷ": "VIII", "Ⅸ": "IX", "Ⅹ": "X", "Ⅺ": "XI", "Ⅻ": "XII"})


class ChapterDetectionError(RuntimeError):
    """A model-assisted rule could not be obtained or safely validated."""


class ChapterRulePayload(BaseModel):
    """模型输出的安全章节 DSL；禁止额外字段、正则和任意代码。"""
    model_config = ConfigDict(extra="forbid")

    chapter_units: list[Literal["章", "回", "节", "幕", "话", "部", "集", "chapter", "chap.", "part"]] = Field(min_length=1, max_length=8)
    number_styles: list[Literal["arabic", "chinese", "financial_chinese", "roman"]] = Field(min_length=1, max_length=4)
    prefixes_to_ignore: list[str] = Field(default_factory=list, max_length=8)
    suffixes_to_ignore: list[str] = Field(default_factory=list, max_length=8)
    line_mode: Literal["whole_line", "paragraph_start"]
    title_boundary: Literal["line_end", "double_space", "next_line"]
    title_position: Literal["after_number_unit", "next_line"]
    volume_units: list[Literal["卷", "部", "集", "volume", "part"]] = Field(default_factory=list, max_length=5)
    numbering_scope: Literal["global", "per_volume"]
    intro_markers: list[str] = Field(default_factory=list, max_length=8)
    tail_markers: list[str] = Field(default_factory=list, max_length=8)
    expected_chapter_count: int = Field(ge=2, le=5000)
    example_candidate_ids: list[str] = Field(min_length=2, max_length=12)
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=8, max_length=500)

    @field_validator("prefixes_to_ignore", "suffixes_to_ignore", "intro_markers", "tail_markers")
    @classmethod
    def validate_literals(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        for value in values:
            item = value.strip()
            if not item or len(item) > 40:
                raise ValueError("chapter_rule_literal_invalid")
            if item not in cleaned:
                cleaned.append(item)
        return cleaned

    @field_validator("example_candidate_ids")
    @classmethod
    def validate_candidate_ids(cls, values: list[str]) -> list[str]:
        if any(not re.fullmatch(r"D\d+L\d+", value) for value in values):
            raise ValueError("candidate_id_invalid")
        return list(dict.fromkeys(values))


@dataclass(frozen=True)
class ChapterCandidate:
    """发送给章节识别模型的短行候选及局部上下文。"""
    id: str
    document_index: int
    line_number: int
    nonblank_index: int
    char_start: int
    text: str
    previous: str
    following: str
    score: int

    def as_prompt_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "line": self.line_number,
            "text": self.text,
            "previous": self.previous,
            "next": self.following,
        }


@dataclass(frozen=True)
class MatchedHeading:
    """章节 DSL 在原文中确定性匹配后的标题记录。"""
    candidate_id: str
    document_index: int
    line_number: int
    title: str
    number: int | None
    unit: str
    volume_key: int
    sentence_like: bool


@dataclass(frozen=True)
class ValidatedChapterRule:
    """通过程序验证并可应用于分章的规则及验证结果。"""
    rule: ChapterRulePayload
    result: NovelSplitResult
    matched_headings: tuple[MatchedHeading, ...]
    validation: dict[str, Any]


@dataclass(frozen=True)
class ChapterRuleDiscovery:
    rule: ChapterRulePayload
    mode: Literal["llm", "cache"]
    raw_content: str = ""


def source_hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def serialize_rule(rule: ChapterRulePayload) -> str:
    return rule.model_dump_json()


def deserialize_rule(value: str) -> ChapterRulePayload:
    return ChapterRulePayload.model_validate_json(value)


def _structure_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", text or "")
    value = _INVISIBLE_RE.sub("", value).replace("\u00a0", " ")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    return _CONTROL_RE.sub("", value)


def _line_records(text: str) -> list[dict[str, Any]]:
    lines = text.splitlines(keepends=True)
    nonblank: list[tuple[int, str]] = [
        (index, line.rstrip("\r\n").strip())
        for index, line in enumerate(lines)
        if line.rstrip("\r\n").strip()
    ]
    neighbors: dict[int, tuple[str, str, int]] = {}
    for ordinal, (line_index, content) in enumerate(nonblank):
        previous = nonblank[ordinal - 1][1] if ordinal else ""
        following = nonblank[ordinal + 1][1] if ordinal + 1 < len(nonblank) else ""
        neighbors[line_index] = (previous, following, ordinal)
    records: list[dict[str, Any]] = []
    offset = 0
    for line_index, line in enumerate(lines):
        raw = line.rstrip("\r\n")
        stripped = raw.strip()
        if stripped:
            previous, following, ordinal = neighbors[line_index]
            records.append({
                "line_number": line_index + 1,
                "nonblank_index": ordinal,
                "char_start": offset + len(raw) - len(raw.lstrip()),
                "text": stripped,
                "previous": previous,
                "following": following,
            })
        offset += len(line)
    return records


def _candidate_shape(text: str) -> str:
    value = text.lower().strip()
    value = re.sub(r"[0-9零〇○一二三四五六七八九十百千万两壹贰叁肆伍陆柒捌玖拾佰仟IVXLCDMⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ]+", "#", value)
    value = re.sub(r"第\s*#\s*[章节回卷部集幕话].*", "第#章", value)
    value = re.sub(r"(?:chapter|chap\.?|part)\s*#.*", "chapter#", value)
    return value[:60]


def _sample_evenly(items: Sequence[ChapterCandidate], limit: int) -> list[ChapterCandidate]:
    if len(items) <= limit:
        return list(items)
    selected: dict[str, ChapterCandidate] = {}
    edge = min(max(10, limit // 6), len(items) // 2)
    for item in list(items[:edge]) + list(items[-edge:]):
        selected[item.id] = item
    remaining = max(0, limit - len(selected))
    if remaining:
        step = (len(items) - 1) / max(1, remaining - 1)
        for index in range(remaining):
            item = items[min(len(items) - 1, round(index * step))]
            selected[item.id] = item
    return sorted(selected.values(), key=lambda item: (item.document_index, item.line_number))[:limit]


def extract_chapter_candidates(raw_docs: Sequence[Document], limit: int | None = None) -> list[ChapterCandidate]:
    """提取短行标题候选，不把候选本身当作最终章节边界。"""
    maximum = limit or settings.chapter_detection_candidate_limit
    provisional: list[tuple[dict[str, Any], int, int]] = []
    shapes: dict[str, int] = {}
    for doc_index, doc in enumerate(raw_docs):
        for record in _line_records(_structure_text(doc.page_content)):
            text = record["text"]
            if len(text) > 160:
                continue
            structural = bool(_STRUCTURE_TOKEN_RE.search(text))
            intro_tail = bool(_INTRO_TAIL_RE.fullmatch(text))
            shape = _candidate_shape(text)
            shapes[shape] = shapes.get(shape, 0) + 1
            score = (12 if structural else 0) + (8 if intro_tail else 0)
            provisional.append((record, doc_index, score))

    candidates: list[ChapterCandidate] = []
    for record, doc_index, base_score in provisional:
        repeated = shapes.get(_candidate_shape(record["text"]), 0) >= 2
        if base_score == 0 and not repeated:
            continue
        score = base_score + (5 if repeated else 0) + (2 if len(record["text"]) <= 80 else 0)
        candidates.append(ChapterCandidate(
            id=f"D{doc_index + 1}L{record['line_number']}",
            document_index=doc_index,
            line_number=record["line_number"],
            nonblank_index=record["nonblank_index"],
            char_start=record["char_start"],
            text=record["text"][:160],
            previous=record["previous"][:80],
            following=record["following"][:80],
            score=score,
        ))
    candidates.sort(key=lambda item: (item.document_index, item.line_number))
    return _sample_evenly(candidates, maximum)


def _extract_json(content: object) -> dict[str, Any]:
    if isinstance(content, list):
        text = "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    else:
        text = content if isinstance(content, str) else str(content)
    match = re.search(r"\{[\s\S]*\}", text.strip().strip("`"))
    if not match:
        raise ValueError("json_object_not_found")
    return json.loads(match.group(0))


def _discovery_prompt(candidates: Sequence[ChapterCandidate]) -> str:
    contract = {
        "chapter_units": ["章/回/节/幕/话/部/集/chapter/chap./part"],
        "number_styles": list(_NUMBER_STYLES),
        "line_mode": ["whole_line", "paragraph_start"],
        "title_boundary": ["line_end", "double_space", "next_line"],
        "title_position": ["after_number_unit", "next_line"],
        "numbering_scope": ["global", "per_volume"],
    }
    return (
        "分析下面的小说短行候选，发现章节标题的结构规则。只输出一个 JSON 对象，不输出正则、代码、SQL或解释性正文。\n"
        "你只能从固定枚举中选择；prefixes_to_ignore/suffixes_to_ignore 必须是候选行中真实存在的字面前后缀。\n"
        "example_candidate_ids 必须引用真实候选 ID。expected_chapter_count 是整本书预计的主要章节数，不含序言/后记。\n"
        f"固定契约：{json.dumps(contract, ensure_ascii=False)}\n\n"
        f"候选：{json.dumps([item.as_prompt_dict() for item in candidates], ensure_ascii=False)}"
    )


async def discover_chapter_rule(raw_docs: Sequence[Document], *, llm=None) -> ChapterRuleDiscovery:
    """将有限候选交给模型生成章节 DSL，并执行一次 JSON 纠正回退。"""
    if not settings.enable_llm_chapter_detection:
        raise ChapterDetectionError("模型辅助章节识别已关闭")
    candidates = extract_chapter_candidates(raw_docs)
    if len(candidates) < 2:
        raise ChapterDetectionError("可供模型分析的章节候选不足")
    prompt = _discovery_prompt(candidates)
    model = llm or get_llm(
        model=settings.chapter_detection_model,
        temperature=0,
        max_tokens=settings.chapter_detection_max_tokens,
        timeout=settings.chapter_detection_timeout,
        max_retries=0,
    )
    previous = ""
    last_error: Exception | None = None
    for attempt in range(settings.chapter_detection_retries + 1):
        human = prompt
        if attempt:
            human += (
                "\n\n上一份输出未通过解析或校验。请只返回修正后的完整 JSON。"
                f"\n错误：{str(last_error)[:300]}\n上一份输出：{previous[:1500]}"
            )
        try:
            response = await model.ainvoke([
                SystemMessage(content="你是小说章节结构规则发现器，只输出安全 DSL JSON。"),
                HumanMessage(content=human),
            ])
            previous = response.content if isinstance(response.content, str) else str(response.content)
            rule = ChapterRulePayload.model_validate(_extract_json(response.content))
            if rule.confidence < settings.chapter_detection_confidence_threshold:
                raise ValueError("chapter_rule_confidence_too_low")
            candidate_ids = {item.id for item in candidates}
            if not set(rule.example_candidate_ids).issubset(candidate_ids):
                raise ValueError("chapter_rule_examples_not_in_candidates")
            return ChapterRuleDiscovery(rule=rule, mode="llm", raw_content=previous)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise ChapterDetectionError(f"模型章节方案生成失败：{str(last_error)[:300]}") from last_error


def _number_pattern(styles: Sequence[str]) -> str:
    parts: list[str] = []
    if "arabic" in styles:
        parts.append(r"[0-9]+")
    if "chinese" in styles:
        parts.append(r"[零〇○一二三四五六七八九十百千万两]+")
    if "financial_chinese" in styles:
        parts.append(r"[零〇○壹贰叁肆伍陆柒捌玖拾佰仟万]+")
    if "roman" in styles:
        parts.append(r"[IVXLCDMⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ]+")
    return "(?:" + "|".join(parts) + ")"


def _parse_chinese_number(value: str) -> int | None:
    value = value.translate(_FINANCIAL_MAP)
    if all(char in _SIMPLE_DIGITS for char in value):
        digits = "".join(str(_SIMPLE_DIGITS[char]) for char in value)
        return int(digits) if digits else None
    total = 0
    section = 0
    number = 0
    units = {"十": 10, "百": 100, "千": 1000, "万": 10000}
    for char in value:
        if char in _SIMPLE_DIGITS:
            number = _SIMPLE_DIGITS[char]
        elif char in units:
            unit = units[char]
            if unit == 10000:
                section = (section + number) * unit
                total += section
                section = 0
            else:
                section += (number or 1) * unit
            number = 0
        else:
            return None
    return total + section + number


def parse_number(value: str, styles: Sequence[str]) -> int | None:
    """把阿拉伯、中文、财务中文或罗马数字统一转换为整数。"""
    token = value.strip()
    if "arabic" in styles and token.isdigit():
        return int(token)
    if "roman" in styles:
        roman = token.translate(_ROMAN_TRANSLATE).upper()
        if roman and all(char in _ROMAN_MAP for char in roman):
            total = 0
            previous = 0
            for char in reversed(roman):
                current = _ROMAN_MAP[char]
                total += -current if current < previous else current
                previous = max(previous, current)
            return total or None
    if "chinese" in styles or "financial_chinese" in styles:
        return _parse_chinese_number(token)
    return None


def _strip_literals(value: str, prefixes: Sequence[str], suffixes: Sequence[str]) -> str:
    text = unicodedata.normalize("NFKC", value).strip()
    normalized_prefixes = [unicodedata.normalize("NFKC", item).strip() for item in prefixes]
    normalized_suffixes = [unicodedata.normalize("NFKC", item).strip() for item in suffixes]
    for prefix in sorted(normalized_prefixes, key=len, reverse=True):
        if prefix and text.startswith(prefix):
            text = text[len(prefix):].lstrip()
            break
    for suffix in sorted(normalized_suffixes, key=len, reverse=True):
        if suffix and text.endswith(suffix):
            text = text[:-len(suffix)].rstrip()
            break
    return text


def _compile_rule_patterns(rule: ChapterRulePayload) -> tuple[re.Pattern[str], re.Pattern[str] | None]:
    number = _number_pattern(rule.number_styles)
    unit_first = [unit for unit in rule.chapter_units if unit.lower() in {"chapter", "chap.", "part"}]
    unit_last = [unit for unit in rule.chapter_units if unit not in unit_first]
    alternatives: list[str] = []
    if unit_last:
        alternatives.append(rf"(?:第\s*)?(?P<number_last>{number})\s*(?P<unit_last>{'|'.join(re.escape(unit) for unit in unit_last)})")
    if unit_first:
        alternatives.append(rf"(?P<unit_first>{'|'.join(re.escape(unit) for unit in unit_first)})\s*(?P<number_first>{number})")
    chapter = re.compile(r"^(?:" + "|".join(alternatives) + r")", re.IGNORECASE)
    volume = None
    if rule.volume_units:
        volume_units = "|".join(re.escape(unit) for unit in sorted(rule.volume_units, key=len, reverse=True))
        volume = re.compile(rf"^(?:第\s*)?(?P<number>{number})\s*(?P<unit>{volume_units})", re.IGNORECASE)
    return chapter, volume


def _cleaned_line_map(text: str) -> dict[int, dict[str, Any]]:
    cleaned = clean_novel_text(text)
    records = _line_records(cleaned)
    return {record["nonblank_index"]: record for record in records}


def _apply_rule(raw_docs: Sequence[Document], rule: ChapterRulePayload) -> tuple[list[list[ChapterHeading]], list[MatchedHeading]]:
    """将固定 DSL 编译为内部匹配器并在原文中确定性生成标题。"""
    chapter_pattern, volume_pattern = _compile_rule_patterns(rule)
    headings_by_doc: list[list[ChapterHeading]] = []
    matches: list[MatchedHeading] = []
    volume_key = 0
    for doc_index, doc in enumerate(raw_docs):
        structure_records = _line_records(_structure_text(doc.page_content))
        clean_map = _cleaned_line_map(doc.page_content)
        doc_headings: list[ChapterHeading] = []
        for record_index, record in enumerate(structure_records):
            original = record["text"]
            candidate = _strip_literals(original, rule.prefixes_to_ignore, rule.suffixes_to_ignore)
            chapter_candidate = candidate
            if volume_pattern:
                volume_match = volume_pattern.match(candidate)
                if volume_match:
                    volume_key = parse_number(volume_match.group("number"), rule.number_styles) or (volume_key + 1)
                    chapter_candidate = candidate[volume_match.end():].lstrip(" ：:·-—")
                    if not chapter_candidate:
                        continue
            intro = next((item for item in rule.intro_markers if chapter_candidate.startswith(item)), None)
            tail = next((item for item in rule.tail_markers if chapter_candidate.startswith(item)), None)
            marker = chapter_pattern.match(chapter_candidate)
            if not marker and not intro and not tail:
                continue
            clean_record = clean_map.get(record["nonblank_index"])
            if not clean_record:
                continue
            if intro or tail:
                title = re.sub(r"\s+", " ", chapter_candidate)[:100]
                kind: Literal["main", "intro", "tail"] = "intro" if intro else "tail"
                doc_headings.append(ChapterHeading(clean_record["char_start"], clean_record["char_start"] + len(clean_record["text"]), title, kind))
                continue
            number_text = marker.groupdict().get("number_last") or marker.groupdict().get("number_first") or ""
            unit = marker.groupdict().get("unit_last") or marker.groupdict().get("unit_first") or ""
            chapter_number = parse_number(number_text, rule.number_styles)
            if chapter_number is None:
                continue
            title_source = chapter_candidate
            end_nonblank = record["nonblank_index"]
            if rule.title_position == "next_line" or rule.title_boundary == "next_line":
                if record_index + 1 >= len(structure_records):
                    continue
                next_record = structure_records[record_index + 1]
                next_clean = clean_map.get(next_record["nonblank_index"])
                if not next_clean:
                    continue
                title_source = chapter_candidate[:marker.end()].rstrip() + " " + next_record["text"].strip()
                end_nonblank = next_record["nonblank_index"]
            elif rule.title_boundary == "double_space":
                remainder = chapter_candidate
                boundary = re.search(r"\s{2,}", remainder[marker.end():])
                if boundary:
                    title_source = chapter_candidate[: marker.end() + boundary.start()].rstrip()
            title = re.sub(r"\s+", " ", _strip_literals(title_source, rule.prefixes_to_ignore, rule.suffixes_to_ignore)).strip()[:100]
            end_record = clean_map.get(end_nonblank, clean_record)
            sentence_like = len(original) > 100 or len(_SENTENCE_PUNCT_RE.findall(original)) >= 2
            candidate_id = f"D{doc_index + 1}L{record['line_number']}"
            doc_headings.append(ChapterHeading(
                clean_record["char_start"],
                end_record["char_start"] + len(end_record["text"]),
                title,
                "main",
            ))
            matches.append(MatchedHeading(
                candidate_id=candidate_id,
                document_index=doc_index,
                line_number=record["line_number"],
                title=title,
                number=chapter_number,
                unit=unit,
                volume_key=volume_key,
                sentence_like=sentence_like,
            ))
        doc_headings.sort(key=lambda item: item.start)
        headings_by_doc.append(doc_headings)
    return headings_by_doc, matches


def _chapter_lengths(raw_docs: Sequence[Document], headings_by_doc: Sequence[Sequence[ChapterHeading]]) -> list[int]:
    lengths: list[int] = []
    for doc, headings in zip(raw_docs, headings_by_doc):
        cleaned = clean_novel_text(doc.page_content)
        main = [heading for heading in headings if heading.kind == "main"]
        for index, heading in enumerate(main):
            end = main[index + 1].start if index + 1 < len(main) else len(cleaned)
            lengths.append(max(0, end - heading.start))
    return lengths


def _validate_numbering(matches: Sequence[MatchedHeading], scope: str) -> tuple[bool, str]:
    """检查章节编号的递增性、重复率和卷内重置规则。"""
    numbered = [match for match in matches if match.number is not None]
    if len(numbered) < 2:
        return False, "chapter_numbers_insufficient"
    groups: dict[int, list[int]] = {}
    for match in numbered:
        groups.setdefault(match.volume_key if scope == "per_volume" else 0, []).append(match.number or 0)
    for numbers in groups.values():
        inversions = sum(1 for left, right in zip(numbers, numbers[1:]) if right < left)
        duplicates = len(numbers) - len(set(numbers))
        if inversions > max(1, math.floor(len(numbers) * 0.05)):
            return False, "chapter_numbers_out_of_order"
        if duplicates / max(1, len(numbers)) > 0.10:
            return False, "chapter_numbers_too_many_duplicates"
    return True, "ok"


def validate_and_apply_rule(
    raw_docs: Sequence[Document],
    rule: ChapterRulePayload,
    filename: str,
    file_id: str,
) -> ValidatedChapterRule:
    """验证模型规则的数量、位置、正文长度和误切风险，成功后重新分章。"""
    if rule.confidence < settings.chapter_detection_confidence_threshold:
        raise ChapterDetectionError("模型章节方案置信度不足")
    headings_by_doc, matches = _apply_rule(raw_docs, rule)
    actual = len(matches)
    errors: list[str] = []
    # 只有通过数量、顺序、边界和正文长度检查，模型规则才允许替换确定性结果。
    if actual < 2:
        errors.append("matched_chapter_count_below_two")
    tolerance = max(2, math.ceil(rule.expected_chapter_count * 0.10))
    if abs(actual - rule.expected_chapter_count) > tolerance:
        errors.append("matched_count_differs_from_expected")
    if len({match.title for match in matches}) / max(1, actual) < 0.90:
        errors.append("duplicate_title_ratio_too_high")
    sentence_ratio = sum(match.sentence_like for match in matches) / max(1, actual)
    if sentence_ratio > 0.05:
        errors.append("sentence_like_match_ratio_too_high")
    numbering_ok, numbering_reason = _validate_numbering(matches, rule.numbering_scope)
    if not numbering_ok:
        errors.append(numbering_reason)
    matched_ids = {match.candidate_id for match in matches}
    if not set(rule.example_candidate_ids).issubset(matched_ids):
        errors.append("example_candidates_not_reproduced")
    elif actual >= 3:
        ordered_ids = [match.candidate_id for match in matches]
        example_positions = sorted(ordered_ids.index(item) / max(1, actual - 1) for item in rule.example_candidate_ids)
        if not (example_positions[0] <= 0.20 and example_positions[-1] >= 0.80 and any(0.25 <= value <= 0.75 for value in example_positions)):
            errors.append("example_candidates_do_not_cover_first_middle_last")
    lengths = _chapter_lengths(raw_docs, headings_by_doc)
    median_length = statistics.median(lengths) if lengths else 0
    if median_length < 200:
        errors.append("median_chapter_length_too_short")
    for headings in headings_by_doc:
        if any(right.start <= left.start for left, right in zip(headings, headings[1:])):
            errors.append("heading_positions_not_increasing")
            break
    if errors:
        raise ChapterDetectionError("模型章节方案验证失败：" + ", ".join(dict.fromkeys(errors)))
    result = split_novel_documents(
        raw_docs,
        filename,
        file_id,
        headings_override=headings_by_doc,
        parser_mode_override="llm_assisted",
    )
    validation = {
        "actual_chapter_count": actual,
        "expected_chapter_count": rule.expected_chapter_count,
        "median_chapter_length": median_length,
        "sentence_like_ratio": round(sentence_ratio, 4),
        "numbering": numbering_reason,
    }
    return ValidatedChapterRule(rule=rule, result=result, matched_headings=tuple(matches), validation=validation)


def assess_deterministic_quality(raw_docs: Sequence[Document], result: NovelSplitResult) -> list[str]:
    """评估确定性解析质量，决定是否需要 LLM 辅助章节识别。"""
    reasons: list[str] = []
    total_chars = sum(len(clean_novel_text(doc.page_content)) for doc in raw_docs)
    if result.chapter_count == 0:
        reasons.append("chapter_count_zero")
    if total_chars >= 100_000 and result.chapter_count < 3:
        reasons.append("long_text_with_too_few_chapters")
    unassigned_ratio = result.unassigned_chunk_count / max(1, len(result.documents))
    if unassigned_ratio > 0.5:
        reasons.append("unassigned_chunk_ratio_high")
    grouped: dict[tuple[int, str], int] = {}
    for doc in result.documents:
        chapter_no = int(doc.metadata.get("chapter_no") or 0)
        if chapter_no <= 0:
            continue
        key = (chapter_no, str(doc.metadata.get("chapter") or ""))
        grouped[key] = grouped.get(key, 0) + len(doc.page_content)
    if grouped and statistics.median(grouped.values()) < 200:
        reasons.append("median_chapter_length_too_short")
    titles = list(dict.fromkeys(
        str(doc.metadata.get("chapter") or "")
        for doc in result.documents
        if int(doc.metadata.get("chapter_no") or 0) > 0
    ))
    parsed = []
    for title in titles:
        match = _STRUCTURE_TOKEN_RE.search(title)
        if match:
            number_match = re.search(r"[0-9零〇○一二三四五六七八九十百千万两壹贰叁肆伍陆柒捌玖拾佰仟IVXLCDMⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ]+", match.group(0))
            if number_match:
                value = parse_number(number_match.group(0), _NUMBER_STYLES)
                if value is not None:
                    parsed.append(value)
    if len(parsed) >= 3:
        inversions = sum(1 for left, right in zip(parsed, parsed[1:]) if right < left)
        duplicates = len(parsed) - len(set(parsed))
        if inversions > max(1, math.floor(len(parsed) * 0.05)) or duplicates / len(parsed) > 0.10:
            reasons.append("chapter_numbering_disordered")
    return reasons
