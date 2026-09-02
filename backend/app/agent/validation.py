"""校验专家报告是否符合职责契约，并检测报告之间的重复度。"""
from __future__ import annotations

import re
from typing import Any

from app.agent.contracts import EXPERT_CONTRACTS, SPECIALIST_ORDER


def _normalize(text: str) -> str:
    text = re.sub(r"(?m)^>\s?", "", text)
    return re.sub(r"[^\w\u4e00-\u9fff]", "", text.lower())


def char_ngrams(text: str, size: int = 3) -> set[str]:
    """将报告归一化后切成字符 n-gram，用于中文文本相似度比较。"""
    normalized = _normalize(text)
    if len(normalized) < size:
        return {normalized} if normalized else set()
    return {normalized[index:index + size] for index in range(len(normalized) - size + 1)}


def report_similarity(left: str, right: str) -> float:
    """计算两份专家报告的 Jaccard 相似度；结果只用于重复预警，不代表语义正确率。"""
    a, b = char_ngrams(left), char_ngrams(right)
    if not a or not b:
        return 0.0
    return round(len(a & b) / len(a | b), 4)


def validate_report(agent: str, report: str) -> dict[str, Any]:
    """按专家契约检查报告长度、引用、必需结构和越界内容。"""
    contract = EXPERT_CONTRACTS[agent]
    missing: list[str] = []
    forbidden_hits: list[str] = []
    normalized = _normalize(report)

    if len(normalized) < 40:
        missing.append("报告内容过短")
    if not re.search(r"\[S\d+\]", report):
        missing.append("缺少 [S#] 引用")

    matched_groups = 0
    for group in contract.required_groups:
        if any(keyword.lower() in report.lower() for keyword in group):
            matched_groups += 1
        else:
            missing.append("/".join(group))

    if agent == "timeline" and not re.search(r"(?:^|\n)\s*>?\s*(?:\d+[.、]|首先|最早|随后|之后|后来|最终)", report):
        missing.append("缺少有序时间节点")
    if agent == "locator" and "|" not in report and not re.search(r"章节|页码|片段", report):
        missing.append("缺少定位表或位置字段")
    if agent != "locator" and report.count("|---") >= 1 and "页码" in report and "片段" in report:
        forbidden_hits.append("越界输出章节定位总表")
    if agent == "locator" and any(keyword in report for keyword in ("内心", "性格分析", "真实动机")):
        forbidden_hits.append("越界解释人物心理")

    denominator = len(contract.required_groups) + 2
    score = (matched_groups + int(bool(re.search(r"\[S\d+\]", report))) + int(len(normalized) >= 40)) / denominator
    return {
        "contract_ok": not missing and not forbidden_hits,
        "score": round(score, 3),
        "missing_sections": missing,
        "forbidden_hits": forbidden_hits,
        "similarity_flags": [],
    }


def validate_reports(reports: list[dict[str, Any]], threshold: float) -> tuple[dict[str, dict], list[str]]:
    """批量校验报告并选择需要纠偏的专家；每对高度重复报告只保留契约得分较高者。"""
    validations: dict[str, dict] = {}
    by_agent = {report["agent"]: report for report in reports if report.get("status") == "ok"}
    for agent in SPECIALIST_ORDER:
        report = by_agent.get(agent)
        if report:
            validations[agent] = validate_report(agent, report.get("report", ""))
        else:
            validations[agent] = {
                "contract_ok": False,
                "score": 0.0,
                "missing_sections": ["专家未成功返回"],
                "forbidden_hits": [],
                "similarity_flags": [],
            }

    refine: set[str] = {
        agent for agent, result in validations.items()
        if agent in by_agent and not result["contract_ok"]
    }
    order_index = {name: index for index, name in enumerate(SPECIALIST_ORDER)}
    available = [name for name in SPECIALIST_ORDER if name in by_agent]
    for left_index, left in enumerate(available):
        for right in available[left_index + 1:]:
            similarity = report_similarity(by_agent[left]["report"], by_agent[right]["report"])
            if similarity < threshold:
                continue
            validations[left]["similarity_flags"].append({"agent": right, "score": similarity})
            validations[right]["similarity_flags"].append({"agent": left, "score": similarity})
            left_score, right_score = validations[left]["score"], validations[right]["score"]
            if left_score < right_score:
                loser = left
            elif right_score < left_score:
                loser = right
            else:
                loser = right if order_index[right] > order_index[left] else left
            refine.add(loser)

    return validations, [name for name in SPECIALIST_ORDER if name in refine]
