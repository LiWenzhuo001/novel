"""定义四类专家的固定职责契约，并生成本轮专属任务。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpecialistContract:
    """单个专家的固定职责契约。契约限制专家关注范围、禁止越界内容和报告结构。"""
    name: str
    label: str
    focus: tuple[str, ...]
    forbidden: tuple[str, ...]
    output_format: str
    required_groups: tuple[tuple[str, ...], ...]


EXPERT_CONTRACTS: dict[str, SpecialistContract] = {
    "character": SpecialistContract(
        name="character",
        label="人物关系专家",
        focus=("人物身份与称谓", "关系边与双方立场", "人物动机", "关系变化阶段", "事实与推断的区分"),
        forbidden=("完整复述故事情节", "输出章节定位总表", "替代时间线专家排列全部事件"),
        output_format="Markdown 引用块；简明：只列与问题直接相关的最多 5 条关系边、每条一行；关系变化阶段最多 4 个；整体不超过 350 字；依次输出：关系边、变化阶段、事实与推断、证据不足。",
        required_groups=(("关系", "立场", "称谓"), ("动机", "变化", "阶段"), ("事实", "推断", "证据不足", "材料不足", "信息不足")),
    ),
    "plot": SpecialistContract(
        name="plot",
        label="情节发展专家",
        focus=("事件起因", "冲突与转折", "事件结果", "对人物关系或主线的影响", "伏笔与后续"),
        forbidden=("重新输出完整人物关系清单", "以章节定位表作为主体", "只做人物心理总论"),
        output_format="Markdown 引用块；按“起因 → 事件/冲突 → 转折 → 结果 → 影响”输出事件链。",
        required_groups=(("起因", "原因"), ("事件", "冲突", "转折"), ("结果", "影响", "后续")),
    ),
    "timeline": SpecialistContract(
        name="timeline",
        label="时间线专家",
        focus=("显式时间线索", "相对先后顺序", "事件节点", "关系或情节变化", "时间不确定性"),
        forbidden=("输出完整人物关系总论", "按人物分组复述全部情节", "伪造原文没有的日期"),
        output_format="Markdown 引用块；使用有序列表，逐项输出：时间/相对顺序、事件、变化、引用。",
        required_groups=(("首先", "最早", "之前", "随后", "之后", "后来", "最终", "时间"), ("事件", "发生", "变化", "顺序")),
    ),
    "locator": SpecialistContract(
        name="locator",
        label="章节定位专家",
        focus=("结论与来源 ID 对齐", "章节", "页码", "片段号", "定位信息缺失"),
        forbidden=("解释人物心理", "撰写人物关系总论", "完整复述情节"),
        output_format="Markdown 引用块；以表格为主体：结论 | 来源ID | 章节 | 页码 | 片段号；缺失项写“定位信息不足”。",
        required_groups=(("[S", "来源"), ("章节", "第", "页码", "片段", "定位信息不足")),
    ),
}

SPECIALIST_ORDER = ("character", "plot", "timeline", "locator")


def build_expert_task(name: str, query: str, dynamic_task: str) -> dict:
    """将固定契约和本轮动态重点组合为前端和专家节点都能使用的任务描述。"""
    contract = EXPERT_CONTRACTS[name]
    return {
        "label": contract.label,
        "task": dynamic_task.strip(),
        "focus": list(contract.focus),
        "forbidden": list(contract.forbidden),
        "output_format": contract.output_format,
        "query": query,
    }


def template_expert_tasks(query: str) -> dict[str, dict]:
    """生成不调用 LLM 的四专家默认任务，作为分派失败时的安全回退。"""
    dynamic = {
        "character": f"仅从人物关系、双方立场、动机和关系变化角度分析：{query}",
        "plot": f"仅提取与问题有关的事件因果、冲突、转折、结果和影响：{query}",
        "timeline": f"仅按原文中的显式时间或相对先后顺序整理相关事件：{query}",
        "locator": f"仅核对支持关键结论的来源 ID、章节、页码和片段位置：{query}",
    }
    return {name: build_expert_task(name, query, dynamic[name]) for name in SPECIALIST_ORDER}
