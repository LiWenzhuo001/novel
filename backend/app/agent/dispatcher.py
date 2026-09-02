"""将独立问题分解为四个互补专家任务，并在模型失败时回退到模板。"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.agent.contracts import SPECIALIST_ORDER, build_expert_task, template_expert_tasks
from app.config import settings
from app.core.llm import get_llm
from app.core.logging_config import get_logger

log = get_logger("expert_dispatcher")


class DispatchPayload(BaseModel):
    """LLM 任务分解结果的严格 JSON 结构。"""
    character: str = Field(min_length=8, max_length=240)
    plot: str = Field(min_length=8, max_length=240)
    timeline: str = Field(min_length=8, max_length=240)
    locator: str = Field(min_length=8, max_length=240)


@dataclass(frozen=True)
class DispatchResult:
    """专家分派结果，记录任务、分派模式和回退原因。"""
    tasks: dict[str, dict]
    mode: str
    reason: str


def _extract_json(content: object) -> dict[str, Any]:
    """从模型文本或代码围栏中提取 JSON 对象。"""
    text = content if isinstance(content, str) else str(content)
    text = text.strip().strip("`").strip()
    fenced = re.search(r"\{[\s\S]*\}", text)
    if not fenced:
        raise ValueError("json_object_not_found")
    return json.loads(fenced.group(0))


def _normalized_task(text: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]", "", text.lower())


def _ngrams(text: str, size: int = 3) -> set[str]:
    normalized = _normalized_task(text)
    if len(normalized) < size:
        return {normalized} if normalized else set()
    return {normalized[index:index + size] for index in range(len(normalized) - size + 1)}


def _similarity(left: str, right: str) -> float:
    a, b = _ngrams(left), _ngrams(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _tasks_are_distinct(payload: DispatchPayload) -> bool:
    """用规范化文本和 3-gram 相似度检查四个子任务是否真正互补。"""
    values = [getattr(payload, name).strip() for name in SPECIALIST_ORDER]
    if len({_normalized_task(value) for value in values}) != len(values):
        return False
    return all(
        _similarity(values[left], values[right]) < 0.82
        for left in range(len(values))
        for right in range(left + 1, len(values))
    )


async def dispatch_expert_tasks(query: str, *, llm=None) -> DispatchResult:
    """调用一次 LLM 生成四个专属子任务；任何解析、校验或调用失败都回退到固定模板。"""
    if settings.agent_expert_dispatch_mode == "template":
        return DispatchResult(template_expert_tasks(query), "template", "configured_template")

    prompt = (
        "把下面的小说问题拆成四个互补、互斥的专家子任务。固定键只能是 character、plot、timeline、locator。\n"
        "character 只关注人物关系、立场、动机和变化；plot 只关注起因、冲突、转折、结果和影响；"
        "timeline 只关注时间线索与先后顺序；locator 只关注来源、章节、页码和片段。\n"
        "每项只写一条简洁任务，不回答问题，不引用原文，不改变用户意图。只输出 JSON。\n\n"
        f"问题：{query}"
    )
    try:
        model = llm or get_llm(temperature=0, max_tokens=settings.agent_dispatch_max_tokens)
        response = await model.ainvoke([
            SystemMessage(content="你是多专家任务分派器，只输出符合要求的 JSON。"),
            HumanMessage(content=prompt),
        ])
        payload = DispatchPayload.model_validate(_extract_json(response.content))
        if not _tasks_are_distinct(payload):
            raise ValueError("expert_tasks_not_distinct")
        tasks = {
            name: build_expert_task(name, query, getattr(payload, name))
            for name in SPECIALIST_ORDER
        }
        return DispatchResult(tasks, "hybrid", "llm_dispatched")
    except Exception as exc:  # noqa: BLE001
        log.warning("expert_dispatch.failed", error=str(exc)[:200])
        return DispatchResult(template_expert_tasks(query), "template", "dispatcher_fallback")
