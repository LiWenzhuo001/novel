"""配置默认值守卫测试（无 DB / 无 LLM / 无网络依赖）。

存在理由：这几个检索参数有**三个真源**——config.py 的代码默认值、backend/.env 的
本地实际值、.env.example 的文档值。三者曾长期不一致（.env 跑 K=30 + 开重排 + 窗口0，
而已验证最优是 K=60 + 关重排 + 窗口1），且 .env 会静默覆盖代码默认值，导致"改了但没生效"。

这里断言的是**代码默认值**：CI 环境没有 backend/.env，load_dotenv() 静默跳过，
因此本测试在 CI 上校验的正是 config.py 里写的那个值。真源漂移会在 CI 直接暴露。

数值来源：evals/reports/ 下 29 份受控评测，当前最优配置
R@10=0.325 / P95 599ms（见 reports/xiyouji_baseline_20260828.json）。
"""

import pytest

from app.config import Settings
from app.core import rag


def test_user_auth_enabled_default_is_true(monkeypatch):
    """鉴权默认开启，避免"忘了开就裸奔"的配置型漏洞（2026-09-01 上线评估结论）。"""
    monkeypatch.setenv("JWT_SECRET", "test-secret")
    assert Settings().user_auth_enabled is True


def test_auth_enabled_without_jwt_secret_raises(monkeypatch):
    """开启 JWT 鉴权但缺 JWT_SECRET 时启动即失败，不静默随机兜底（显式配置是唯一真源）。"""
    monkeypatch.setenv("USER_AUTH_ENABLED", "true")
    monkeypatch.delenv("JWT_SECRET", raising=False)
    monkeypatch.delenv("API_TOKENS", raising=False)
    with pytest.raises(ValueError, match="JWT_SECRET"):
        Settings()


def test_hybrid_candidate_k_default_is_60():
    """候选池 60 优于 30：RRF 大池能容纳更多金标候选（池内召回 0.575）。"""
    assert Settings().hybrid_candidate_k == 60


def test_reranker_enabled_by_default():
    """重排默认开启、纯重排模式（blend 关）。"""
    settings = Settings()
    assert settings.enable_reranker is True
    assert settings.enable_reranker_blend is False
    assert settings.reranker_provider in {"local", "siliconflow"}
    assert settings.reranker_model.strip()


def test_api_embedding_sends_raw_text_by_default(monkeypatch):
    """第三方 API embedding 默认绕过 OpenAI tokenizer。"""
    monkeypatch.delenv("EMBEDDING_CHECK_CTX_LENGTH", raising=False)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "api")
    assert Settings().embedding_check_ctx_length is False


def test_novel_neighbor_window_default_is_1():
    """邻块窗口 1 能救回"命中相邻块"的样例；.env 曾为 0 与代码默认不一致。"""
    assert Settings().novel_neighbor_window == 1


def test_reranker_candidate_n_tracks_hybrid_candidate_k():
    """重排候选数必须跟随候选池，否则提高 K 会被下界静默截断回 20。

    这是一条曾经真实存在的耦合 bug：RERANKER_CANDIDATE_N 写死默认 20，
    HYBRID_CANDIDATE_K 提到 60 后重排池仍只有 20 条。
    """
    s = Settings()
    assert s.reranker_candidate_n >= s.hybrid_candidate_k


def test_chapter_local_retrieval_disabled_by_default():
    """章节内二级检索经 A/B 验证为负收益（R@10 0.325→0.275），默认必须关闭。"""
    assert Settings().enable_chapter_local_retrieval is False


def test_dead_query_expansion_config_removed():
    """ENABLE_RAG_QUERY_EXPANSION 实现已删除，配置残留会造成"改了没反应"的假象。"""
    assert not hasattr(Settings(), "enable_query_expansion")
    assert not hasattr(Settings(), "query_expansion_max_variants")


def test_chapter_local_refine_is_public_and_wired():
    """保证二级检索不再是死代码：函数公开，且确实被 retrieve_novel_context 调用。"""
    assert hasattr(rag, "chapter_local_refine")
    source = open(rag.__file__, encoding="utf-8").read()
    assert "chapter_local_refine(" in source.split("async def retrieve_novel_context")[-1], (
        "chapter_local_refine 未被 retrieve_novel_context 调用，ENABLE_CHAPTER_LOCAL_RETRIEVAL 将再次悬空"
    )
