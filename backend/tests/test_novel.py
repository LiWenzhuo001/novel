from langchain_core.documents import Document

from app.services.novel_service import (
    CHAPTER_PARSER_VERSION,
    clean_novel_text,
    split_novel_documents,
)


def test_clean_novel_text_preserves_paragraphs_and_removes_controls():
    raw = "\ufeff  第一章  开端\r\n\r\n  林舟\t醒来。\x00\r\n\r\n\r\n第二段。  "
    assert clean_novel_text(raw) == "第一章 开端\n\n林舟 醒来。\n\n第二段。"


def test_split_novel_documents_keeps_chapter_across_pages():
    docs = [
        Document(page_content="第一章 开端\n林舟在雨夜离开故乡。", metadata={"page": 0}),
        Document(page_content="第二天清晨，他抵达北城。", metadata={"page": 1}),
        Document(page_content="第二章 重逢\n林舟在车站见到旧友。", metadata={"page": 2}),
    ]
    result = split_novel_documents(docs, "sample.txt", "file-1")
    chunks = result.documents
    assert result.chapter_count == 2
    assert result.parser_mode == "strict"
    assert result.parser_version == CHAPTER_PARSER_VERSION
    assert [chunk.metadata["chunk_no"] for chunk in chunks] == list(range(1, len(chunks) + 1))
    assert chunks[0].metadata["chapter"] == "第一章 开端"
    assert chunks[1].metadata["chapter"] == "第一章 开端"
    assert chunks[-1].metadata["chapter"] == "第二章 重逢"


def test_split_novel_documents_uses_configured_chunk_size(monkeypatch):
    from app.services import novel_service

    monkeypatch.setattr(novel_service.settings, "novel_chunk_size", 30)
    monkeypatch.setattr(novel_service.settings, "novel_chunk_overlap", 5)
    docs = [Document(page_content="第一章 开端\n" + "楚姒看见谢煜璟。" * 12, metadata={"page": 0})]
    result = split_novel_documents(docs, "sample.txt", "file-1")
    assert len(result.documents) > 1
    assert all(len(chunk.page_content) <= 30 for chunk in result.documents)


def test_classic_chinese_chapter_formats_are_recognized():
    text = """《西游记》第一回 灵根育孕源流出 心性修持大道生
正文。
第〇回：测试标题
正文。
第壹回 古典数字标题
正文。
卷一 第一回 卷名前缀
正文。
尾声
结束。"""
    result = split_novel_documents([Document(page_content=text)], "西游记.txt", "xj-1")
    assert result.parser_mode == "strict"
    assert result.chapter_count == 4
    chapters = [doc.metadata["chapter"] for doc in result.documents]
    assert chapters[0].startswith("《西游记》第一回")
    assert any(chapter.startswith("第〇回") for chapter in chapters)
    assert any(chapter.startswith("第壹回") for chapter in chapters)
    assert any(chapter.startswith("卷一 第一回") for chapter in chapters)
    assert chapters[-1] == "尾声"


def test_prose_reference_does_not_create_chapter():
    text = "这一张建议和第八章连在一起看，么么哒！\n" + "普通正文。" * 30
    result = split_novel_documents([Document(page_content=text)], "sample.txt", "file-1")
    assert result.chapter_count == 0
    assert result.parser_mode == "none"
    assert result.chapter_parse_status if hasattr(result, "chapter_parse_status") else True
    assert result.unassigned_chunk_count == len(result.documents)


def test_inline_fallback_requires_two_well_separated_candidates():
    text = (
        "前言。第一回 灵根育孕源流出 心性修持大道生\n正文开始。"
        + "故事内容。" * 60
        + "。第二回 悟彻菩提真妙理 断魔归本合元神\n继续正文。"
        + "故事内容。" * 60
    )
    result = split_novel_documents([Document(page_content=text)], "西游记.txt", "xj-1")
    assert result.parser_mode == "inline_fallback"
    assert result.chapter_count == 2


def test_single_inline_candidate_is_not_used():
    text = "前言。第一回 灵根育孕源流出 心性修持大道生 " + "故事内容。" * 80
    result = split_novel_documents([Document(page_content=text)], "西游记.txt", "xj-1")
    assert result.parser_mode == "none"
    assert result.chapter_count == 0


def test_txt_without_real_pages_stores_none_page():
    result = split_novel_documents([
        Document(page_content="第一回 标题\n正文", metadata={"page": 0, "has_real_page": False})
    ], "西游记.txt", "xj-1")
    assert result.documents[0].metadata["page"] is None


def test_catalog_prefixed_journey_to_the_west_headings_are_recognized():
    text = """《西游记》

《》目录 第一回　灵根育孕源流出　心性修持大道生
正文一。
《》目录 第二回　悟彻菩提真妙理　断魔归本合元神
正文二。
《》目录 第一百回　径回东土　五圣成真
正文三。"""
    result = split_novel_documents([Document(page_content=text)], "西游记.txt", "xj-1")
    assert result.parser_mode == "strict"
    assert result.chapter_count == 3
    chapters = list(dict.fromkeys(doc.metadata["chapter"] for doc in result.documents))
    assert chapters == [
        "序章/前言",
        "第一回 灵根育孕源流出 心性修持大道生",
        "第二回 悟彻菩提真妙理 断魔归本合元神",
        "第一百回 径回东土 五圣成真",
    ]
