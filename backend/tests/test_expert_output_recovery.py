"""专家空输出补生成、相似度清洗与纠偏触发的专项测试（无 DB、无真实 LLM）。"""
from __future__ import annotations

from langchain_core.messages import AIMessage

from app.agent import runtime
from app.agent.contracts import build_expert_task
from app.agent.validation import report_similarity, validate_reports
from app.agent.types import AgentState
from app.core.llm import ModelTurnDelta


# ---------- 造数助手 ----------

def _fake_delta_stream(content="", reasoning=None):
    async def stream(messages, purpose, *, tools=None, max_tokens=None):
        if reasoning:
            yield ModelTurnDelta(kind="reasoning", text=reasoning)
        if content:
            yield ModelTurnDelta(kind="content", text=content)

    return stream


class _FakeLLM:
    def __init__(self, content):
        self._content = content
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        return AIMessage(content=self._content)


def _state():
    return {
        "standalone_query": "梳理人物关系",
        "original_query": "梳理人物关系",
        "expert_tasks": {"character": build_expert_task("character", "梳理人物关系", "动态任务")},
    }


_CONTRACT = None


def _contract():
    global _CONTRACT
    from app.agent.contracts import EXPERT_CONTRACTS
    _CONTRACT = _CONTRACT or EXPERT_CONTRACTS["character"]
    return _CONTRACT


async def _run_specialist(monkeypatch, *, content="", reasoning=None, fallback=""):
    """驱动 _run_specialist：专家流只产 reasoning/空 content，主模型补生成返回 fallback 文本。"""
    monkeypatch.setattr(runtime, "astream_model_turn", _fake_delta_stream(content=content, reasoning=reasoning))
    fake_llm = _FakeLLM(fallback)
    monkeypatch.setattr(runtime, "get_llm", lambda **kw: fake_llm)
    events = []

    class Queue:
        async def put(self, item):
            events.append(item)

    state: AgentState = {**_state(), "event_queue": Queue()}
    result = await runtime._run_specialist(_contract(), state, "expert-character")
    return result, events, fake_llm


# ---------- 空输出检测与补生成 ----------

async def test_reasoning_only_triggers_fallback_and_recovers(monkeypatch):
    """reasoning 有内容 + content 为空 → 主模型补生成一次 → recovered ok。"""
    report = "> 贾母怜黛玉 [S1] 关系变化：由疏到亲，动机是亲情，事实清楚。"
    result, events, fake_llm = await _run_specialist(monkeypatch, reasoning="思考过程", fallback=report)
    assert result["status"] == "ok" and result["recovered"] is True
    assert result["report"] == report
    assert fake_llm.calls == 1
    statuses = [e["data"].get("status") for e in events if e["type"] == "tool_end"]
    assert "fallback_generation" in statuses and statuses[-1] == "ok"
    # 补生成报告整段回灌 tool_token：前端「调用过程」可见正文，不再"只有思考没有回答"
    tokens = [e for e in events if e["type"] == "tool_token"]
    assert tokens and tokens[-1]["data"]["delta"] == report
    # summary 文案不得重复"完成"
    ends = [e["data"] for e in events if e["type"] == "tool_end"]
    assert "完成完成" not in ends[-1]["summary"]


async def test_silent_empty_output_goes_fallback_too(monkeypatch):
    """reasoning 与 content 都为空 → 同样走补生成而不是契约纠偏。"""
    result, events, fake_llm = await _run_specialist(monkeypatch, fallback="> 有效报告 [S1]")
    assert result["status"] == "ok" and result["recovered"] is True
    assert fake_llm.calls == 1


async def test_fallback_still_empty_marks_empty_output(monkeypatch):
    """补生成仍为空 → empty_output（不伪装成纠偏失败）。"""
    result, events, fake_llm = await _run_specialist(monkeypatch, fallback="   ")
    assert result["status"] == "empty_output"
    assert result["error_code"] == "expert_final_content_empty"
    assert fake_llm.calls == 1
    statuses = [e["data"].get("status") for e in events if e["type"] == "tool_end"]
    assert statuses[-1] == "empty_output"


async def test_empty_output_never_enters_refine():
    """empty_output 报告不参与契约校验集合，天然不进纠偏。"""
    reports = [{"agent": "character", "label": "人物关系专家", "status": "empty_output", "report": ""}]
    validations, refine = validate_reports(reports, 0.72)
    assert refine == []
    assert validations["character"]["missing_sections"] == ["专家未成功返回"]


# ---------- 相似度清洗 ----------

_SHELL = (
    "结论 来源ID 页码 片段 引用 第五回 第四回 第三十四回 第四十九回 一百回 "
    "[S1] [S2] [S3] [S4] [S5] [S6] [S7] [S8] [S9] [S10] [S11] [S12] "
    "关系边 变化阶段 时间线 起因 转折 影响 关系变化 事件结果 "
)


def _report(distinct: str) -> str:
    return f"> {_SHELL}{distinct} [S1]"


def test_similarity_ignores_shared_shell_material():
    """共享引用标记/章节名/结构词剔除后，实质不同的报告相似度低于阈值。"""
    left = _report("宝玉与黛玉互相试探，感情深厚")
    right = _report("凤姐协理宁国府，展现出管理才干与决断")
    assert report_similarity(left, right) < 0.72


def test_similarity_still_flags_identical_reports():
    left = _report("宝玉与黛玉互相试探，感情深厚")
    assert report_similarity(left, left) >= 0.72


def test_locator_excluded_from_pairwise_similarity():
    """定位专家表格与其他专家散文不参与正文相似度竞争。"""
    locator = "> | 结论 | 来源ID | 章节 | 页码 | 片段号 |\n> |---|---|---|---|---|\n> | 贾母怜黛玉 | [S4] | 第五回 | 定位信息不足 | 定位信息不足 |"
    character = _report("宝玉与黛玉互相试探，感情深厚")
    reports = [
        {"agent": "locator", "status": "ok", "report": locator},
        {"agent": "character", "status": "ok", "report": character},
    ]
    validations, refine = validate_reports(reports, 0.0)  # 阈值 0：任何成对比较都会 flag
    assert validations["locator"]["similarity_flags"] == []
    assert validations["character"]["similarity_flags"] == []
    assert "locator" not in refine or validations["locator"]["contract_ok"] is False  # 仅契约可触发


# ---------- 契约触发与合法报告 ----------

def test_contract_violations_still_trigger_refine():
    reports = [
        {"agent": "character", "status": "ok", "report": "太短"},
        {"agent": "plot", "status": "ok", "report": "> 没有引用也没有结构词的一段足够长的报告内容，但缺少起因事件结果等关键词。"},
        {"agent": "timeline", "status": "ok", "report": "> 时间与事件都提到了一些内容但没有有序节点标记，长度也足够长，避免过短判定干扰。"},
        {"agent": "locator", "status": "ok", "report": "> 只有散文没有表格也没有章节页码片段字样的定位报告，且不做心理分析。"},
    ]
    validations, refine = validate_reports(reports, 0.72)
    assert set(refine) == {"character", "plot", "timeline", "locator"}


def test_valid_reports_do_not_trigger_refine():
    reports = [
        {"agent": "character", "status": "ok", "report": "> 关系边：贾母怜黛玉 [S1]。变化阶段由疏到亲，动机是亲情；事实有据，推断部分证据不足。" * 2},
        {"agent": "plot", "status": "ok", "report": "> 起因：黛玉进府 [S1]。事件冲突：宝钗来后生隙。结果与影响：宝黛感情受挫。" * 2},
        {"agent": "timeline", "status": "ok", "report": "> 1. 首先黛玉进荣府 [S1]。\n> 2. 随后宝钗到达 [S2]。事件顺序有变化。" * 2},
        {"agent": "locator", "status": "ok", "report": "> | 贾母怜黛玉 | [S1] | 第五回 | 定位信息不足 | 定位信息不足 | [S1] 来源章节定位。" * 2},
    ]
    validations, refine = validate_reports(reports, 0.72)
    assert refine == []
    assert all(v["contract_ok"] for v in validations.values())


# ---------- metrics ----------

def test_metrics_incr_accumulates_value():
    from app.core.metrics import Metrics

    m = Metrics()
    m.incr("agent_expert_reasoning_chars", 1070)
    m.incr("agent_expert_reasoning_chars", 930)
    m.incr("agent_expert_empty_output_count")
    assert m.counters["agent_expert_reasoning_chars"] == 2000
    assert m.counters["agent_expert_empty_output_count"] == 1
