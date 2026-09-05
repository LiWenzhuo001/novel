"""「进入小说世界」：人物名册提取、角色卡生成与角色扮演流式编排。

设计要点：
- 检索只发生在**角色卡生成时**（k=12，范围 ≤ 剧情截至章节），人物的知识边界
  在生成这一刻固化进角色卡；对话阶段不再检索，直接以角色卡 + 时间线锚点生成。
- 缓存（novel_characters 表，按 file_id 共享）：
    roster 行     LLM 提取的主要人物名列表；
    card 行       单个人物的角色卡 JSON（persona/style/background/greeting）；
    scenario 行   一场对话的开场情景（与人物组合 + 截至章节绑定）。
  缓存以 (file_id, name, chapter_until) 定位，source_hash 变化时重新生成。
"""

from __future__ import annotations

import json
import random
from typing import Any, AsyncIterator

from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy import select

from app.config import settings
from app.core.context import get_current_user
from app.core.llm import get_llm
from app.core.logging_config import get_logger
from app.core.rag import retrieve_novel_context
from app.db import AsyncSessionLocal
from app.db.models import KnowledgeFile, NovelCharacter

log = get_logger("world_service")

ROSTER_NAME = "__roster__"
MAX_PERSONAS = 3
ROSTER_SIZE = 10
CARD_K = 12

_CARD_SPEC = (
    "只输出 JSON 对象，包含五个键：\n"
    '  "in_novel": 布尔值——依据片段判断该人物是否真实出现在这部小说中；\n'
    '  "persona": 人物性格与身份概述（80 字内）；\n'
    '  "style": 说话风格与语言习惯（50 字内，含口头禅/称谓示例）；\n'
    '  "background": 该人物截至当前章节的关键经历要点，分条分号分隔、含具体情节锚点（200 字内）；\n'
    '  "greeting": 人物对来访者的开场白（60 字内，第一人称，符合人物语气）。'
)


def _extract_json(text: str) -> Any:
    cleaned = text.strip()
    cleaned = cleaned.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        start, end = cleaned.find(open_ch), cleaned.rfind(close_ch)
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                continue
    return None


async def _llm_json(prompt: str, system: str, max_tokens: int) -> Any:
    model = get_llm(temperature=0, max_tokens=max_tokens, timeout=settings.llm_timeout)
    response = await model.ainvoke([
        SystemMessage(content=system),
        HumanMessage(content=prompt),
    ])
    return _extract_json(getattr(response, "content", ""))


async def _load_cache(file_id: str, name: str, chapter_until: int | None, kind: str, source_hash: str | None) -> Any:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(NovelCharacter).where(
                NovelCharacter.file_id == file_id,
                NovelCharacter.name == name,
                NovelCharacter.kind == kind,
                NovelCharacter.chapter_until == chapter_until,
            ).order_by(NovelCharacter.updated_at.desc()).limit(1)
        )
        row = result.scalars().first()
    if row is None:
        return None
    # 重索引后原文指纹变化即视为缓存失效，下次触发重新生成。
    if source_hash and row.source_hash and row.source_hash != source_hash:
        return None
    try:
        return json.loads(row.content or "{}")
    except json.JSONDecodeError:
        return None


async def _save_cache(
    file_id: str, name: str, chapter_until: int | None, kind: str,
    content: dict[str, Any], source_hash: str | None,
) -> None:
    user_id = get_current_user()
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(NovelCharacter).where(
                NovelCharacter.file_id == file_id,
                NovelCharacter.name == name,
                NovelCharacter.kind == kind,
                NovelCharacter.chapter_until == chapter_until,
            ).limit(1)
        )
        row = result.scalars().first()
        if row is None:
            row = NovelCharacter(
                file_id=file_id, name=name, chapter_until=chapter_until,
                kind=kind, user_id=user_id, source_hash=source_hash,
            )
            session.add(row)
        row.content = json.dumps(content, ensure_ascii=False)
        row.source_hash = source_hash
        await session.commit()


async def _file_source_hash(file_id: str) -> str | None:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(KnowledgeFile.source_hash).where(KnowledgeFile.id == file_id)
        )
        return result.scalar_one_or_none()


async def _visible_file(file_id: str) -> bool:
    from app.core.visibility import visible_user_filter

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(KnowledgeFile.id).where(
                KnowledgeFile.id == file_id,
                visible_user_filter(KnowledgeFile.user_id, get_current_user()),
                KnowledgeFile.status == "indexed",
            )
        )
        return result.scalar_one_or_none() is not None


async def list_characters(file_id: str, chapter_until: int | None) -> dict[str, Any]:
    """返回该书的主要人物推荐名册；无缓存时检索片段并调用 LLM 提取。"""
    if not await _visible_file(file_id):
        raise ValueError("目标小说尚未完成索引或不存在")
    source_hash = await _file_source_hash(file_id)
    cached = await _load_cache(file_id, ROSTER_NAME, chapter_until, "roster", source_hash)
    if cached and isinstance(cached.get("names"), list) and cached["names"]:
        return {"characters": cached["names"], "cached": True}

    docs = await retrieve_novel_context(
        "主要人物 出场 人物关系 性格", k=12, file_id=file_id, chapter_until=chapter_until,
    )
    context_text = "\n".join(doc.page_content[:400] for doc in docs)
    timeline = f"故事进行到第 {chapter_until} 章" if chapter_until else "全书范围"
    prompt = (
        f"以下是一部小说（{timeline}）的原文片段。请提取其中实际出场或被明确提及的"
        f"主要人物名，按重要性排序，最多 {ROSTER_SIZE} 个。只输出 JSON 数组（字符串列表），"
        "不要输出头衔、称呼或重叠的别称。\n\n"
        f"原文片段：\n{context_text}"
    )
    payload = await _llm_json(prompt, "你是小说人物名册提取器，只输出 JSON 数组。", 500)
    names: list[str] = []
    if isinstance(payload, list):
        seen: set[str] = set()
        for item in payload:
            name = str(item or "").strip().strip("《》「」")
            if name and len(name) <= 20 and name not in seen:
                seen.add(name)
                names.append(name)
    names = names[:ROSTER_SIZE]
    if not names:
        log.warning("world_roster.extract_failed", file_id=file_id)
        raise ValueError("人物提取失败，请重试或直接输入人物名")
    await _save_cache(
        file_id, ROSTER_NAME, chapter_until, "roster",
        {"names": names}, source_hash,
    )
    return {"characters": names, "cached": False}


async def _search_character_fragments(file_id: str, name: str, chapter_until: int | None):
    """按人物名检索相关片段；精确查询零命中时用宽松查询再试一次。"""
    docs = await retrieve_novel_context(
        f"{name} 言行 性格 情节", k=CARD_K, file_id=file_id, chapter_until=chapter_until,
    )
    if docs:
        return docs
    return await retrieve_novel_context(
        name, k=CARD_K, file_id=file_id, chapter_until=chapter_until,
    )


async def get_character_cards(
    file_id: str, names: list[str], chapter_until: int | None,
) -> dict[str, Any]:
    """返回每个选中人物的角色卡（无缓存则生成）与整场开场情景。

    人物的知识边界在此固化：检索范围 ≤ 剧情截至章节，之后的内容不会进入角色卡。
    """
    if not 1 <= len(names) <= MAX_PERSONAS:
        raise ValueError(f"人物数量需在 1 到 {MAX_PERSONAS} 之间")
    if not await _visible_file(file_id):
        raise ValueError("目标小说尚未完成索引或不存在")
    source_hash = await _file_source_hash(file_id)
    timeline = f"第 {chapter_until} 章" if chapter_until else "全书"

    cards: list[dict[str, Any]] = []
    for name in names:
        name = str(name or "").strip()[:20]
        if not name:
            continue
        cached = await _load_cache(file_id, name, chapter_until, "card", source_hash)
        if cached and cached.get("persona") and cached.get("in_novel", True):
            cards.append({"name": name, **{k: cached.get(k, "") for k in ("persona", "style", "background", "greeting")}})
            continue

        docs = await _search_character_fragments(file_id, name, chapter_until)
        if not docs:
            raise ValueError(f"小说中未找到「{name}」，请确认姓名（可用书中称呼）或换一个人物")

        context_text = "\n".join(doc.page_content[:400] for doc in docs)
        prompt = (
            f"以下是小说（截至 {timeline}）中与「{name}」相关的原文片段。"
            "请基于这些片段为该人物生成角色卡。" + _CARD_SPEC + "\n\n"
            f"人物名：{name}\n\n原文片段：\n{context_text}"
        )
        card = await _llm_json(
            prompt, "你是小说角色卡生成器，只输出符合要求的 JSON 对象。", 900,
        )
        if not isinstance(card, dict):
            log.warning("world_card.generate_failed", file_id=file_id, name=name)
            raise ValueError(f"「{name}」的角色卡生成失败，请重试或换一个人物")
        if card.get("in_novel") is False:
            raise ValueError(f"小说中未找到「{name}」，请确认姓名（可用书中称呼）或换一个人物")
        if not card.get("persona"):
            raise ValueError(f"「{name}」的角色卡生成失败，请重试或换一个人物")
        card = {key: str(card.get(key) or "").strip() for key in ("persona", "style", "background", "greeting")}
        await _save_cache(file_id, name, chapter_until, "card", card, source_hash)
        cards.append({"name": name, **card})

    if not cards:
        raise ValueError("请至少选择或输入一个人物")

    scenario = await _get_or_make_scenario(file_id, cards, chapter_until, source_hash)
    return {"cards": cards, "scenario": scenario}


async def _get_or_make_scenario(
    file_id: str, cards: list[dict[str, Any]], chapter_until: int | None, source_hash: str | None,
) -> str:
    """生成（或取缓存）本场对话的开场情景：访客登场的时间、地点与人物状态。"""
    scene_key = "__scenario__" + ",".join(sorted(card["name"] for card in cards))
    cached = await _load_cache(file_id, scene_key, chapter_until, "scenario", source_hash)
    if cached and cached.get("scenario"):
        return cached["scenario"]

    card_text = "\n".join(
        f"- {card['name']}：{card['persona']}；经历：{card['background']}"
        for card in cards
    )
    names = "、".join(card["name"] for card in cards)
    timeline = f"故事进行到第 {chapter_until} 章" if chapter_until else "全书完结后的世界"
    prompt = (
        f"一部小说（截至 {timeline}）中的角色：{names}，即将与一位「来自外界的访客」对话。\n"
        "请为这场对话写一段开场情景（150 字内），要求：\n"
        "- 用第二人称「你」描述访客所见：时间、地点、氛围；\n"
        "- 描写角色此刻正在做什么（严格符合其截至当前章节的处境，不剧透后续）；\n"
        "- 以角色即将注意到访客的动作收尾，自然引出开场白；\n"
        "- 文风贴合小说时代感。\n"
        "只输出 JSON：{\"scenario\": \"...\"}\n\n"
        f"角色卡：\n{card_text}"
    )
    payload = await _llm_json(
        prompt, "你是小说场景描写器，只输出符合要求的 JSON 对象。", 500,
    )
    scenario = str((payload or {}).get("scenario") or "").strip() if isinstance(payload, dict) else ""
    if not scenario:
        log.warning("world_scenario.generate_failed", file_id=file_id)
        scenario = (
            f"你推开一扇门，走进了这部小说的世界（{timeline}）。"
            f"{names}正等着与你交谈。"
        )
    await _save_cache(
        file_id, scene_key, chapter_until, "scenario", {"scenario": scenario}, source_hash,
    )
    return scenario


def _roleplay_system_prompt(cards: list[dict[str, Any]], chapter_until: int | None) -> str:
    timeline = (
        f"故事目前进行到第 {chapter_until} 章。第 {chapter_until} 章之后的所有情节尚未发生，"
        "你（们）不知道、不能提及、也不能预知任何后续内容。"
        if chapter_until else "故事涵盖全书内容。"
    )
    group_rule = ""
    if len(cards) > 1:
        names = "、".join(card["name"] for card in cards)
        group_rule = (
            f"\n【群聊规则】场上共有 {len(cards)} 个角色（{names}），由你同时扮演：\n"
            "- 每个角色的发言必须另起一段，并以完整人物名加粗开头，如「**林黛玉**：」；禁止使用简称、昵称或单字；\n"
            "- 随机决定本轮由谁回应：通常 1~2 个角色，最多全员；避免每次固定顺序、避免机械地人人发言；\n"
            "- 角色之间可以自然地接话、补充、认同或争执，符合各自立场；\n"
            "- 不要混淆各角色的立场和语气。"
        )
    card_text = "\n\n".join(
        f"【{card['name']}】\n性格与身份：{card['persona']}\n"
        f"说话风格：{card['style']}\n截至当前章节的经历：{card['background']}"
        for card in cards
    )
    return (
        "你正在和一位「来自外界的访客」交谈——对方不是小说中的人物，而是造访小说世界的客人。"
        "你必须始终扮演下面描述的小说角色，以第一人称与访客交谈。\n\n"
        f"【角色卡】\n{card_text}\n\n"
        f"【时间线】{timeline}\n"
        "【场景规则】\n"
        "- 角色卡中的「经历要点」就是你全部的亲身经历：谈及这些内容时以亲历者口吻自然讲述；\n"
        "- 超出经历要点的剧情细节，以人物的口吻表示不确定或不知道，绝不编造；\n"
        "- 始终保持在角色内：不要承认自己是 AI、程序或模型，不要跳出小说世界谈论现实；\n"
        "- 语言习惯贴合人物所处的时代与身份；发言使用完整人物名；不要输出 [S#] 标记或「根据检索」等内部过程描述。"
        f"{group_rule}"
    )


async def stream_roleplay(
    file_id: str,
    names: list[str],
    chapter_until: int | None,
    message: str,
    history: list[Any],
) -> AsyncIterator[dict[str, Any]]:
    """角色扮演的流式编排：角色卡 + 时间线锚点 + 对话历史 → 高温流式生成。

    对话阶段不检索：人物的知识边界已在角色卡生成时固化（含剧情截至章节）。
    """
    result = await get_character_cards(file_id, names, chapter_until)
    cards = result["cards"]
    # 随机打乱角色卡顺序：大模型对靠前内容有明显偏好，打乱可避免"总是同一个角色先发言"。
    cards = list(cards)
    random.shuffle(cards)
    yield {"type": "meta", "data": {
        "strategy": "roleplay",
        "personas": [card["name"] for card in cards],
        "chapter_until": chapter_until,
        "output_policy": {"allow_direct_quotes": True, "summary_only": False, "show_citations": False},
    }}

    history_text = _history_lines(history)
    system = _roleplay_system_prompt(cards, chapter_until)
    prompt = (
        (f"【此前对话】\n{history_text}\n\n" if history_text else "")
        + f"访客说：{message}\n\n请以角色身份继续对话。"
    )
    async for chunk in get_llm(streaming=True, temperature=0.8, max_tokens=settings.agent_synthesis_max_tokens).astream([
        SystemMessage(content=system),
        HumanMessage(content=prompt),
    ]):
        content = getattr(chunk, "content", "")
        if isinstance(content, str) and content:
            yield {"type": "token", "data": content}
        elif isinstance(content, list):
            text = "".join(
                block.get("text", "") for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
            if text:
                yield {"type": "token", "data": text}
    yield {"type": "meta", "data": {
        "strategy": "roleplay",
        "personas": [card["name"] for card in cards],
        "chapter_until": chapter_until,
        "steps": 1,
        "output_policy": {"allow_direct_quotes": True, "summary_only": False, "show_citations": False},
    }}


def _history_lines(history: list[Any] | None) -> str:
    """把对话历史拼成文本；兼容 dict（req.history）与 LangChain 消息对象两种形态。"""
    lines: list[str] = []
    for item in (history or [])[-8:]:
        if isinstance(item, dict):
            role = "访客" if item.get("role") == "user" else "角色"
            content = str(item.get("content", ""))
        else:
            mtype = str(getattr(item, "type", "") or type(item).__name__)
            role = "访客" if mtype in ("human", "user", "HumanMessage") else "角色"
            content = str(getattr(item, "content", ""))
        if content.strip():
            lines.append(f"{role}：{content}")
    return "\n".join(lines)
