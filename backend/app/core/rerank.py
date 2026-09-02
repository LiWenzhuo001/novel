"""本地 Cross-encoder 与 SiliconFlow 远程重排器（懒加载单例）。

在召回（向量 + 全文 RRF 融合）得到候选之后，对每个 (query, 文档) 重新打相关分并排序。
本地模式使用 sentence-transformers CrossEncoder，siliconflow 模式调用 classic 文本 rerank API。
"""
import math
import threading
import time

import httpx

from app.config import settings

_model = None
_model_lock = threading.Lock()
_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class RerankerUnavailable(RuntimeError):
    """重排器不可用：模型加载失败、远程调用失败或响应校验未通过。"""


class _SiliconFlowReranker:
    """SiliconFlow classic text rerank API 客户端。"""

    def __init__(self):
        settings.validate_reranker()
        self._client = httpx.Client(timeout=settings.reranker_timeout)
        self.last_trace_id = ""

    def predict(self, pairs):
        if not pairs:
            return []
        query = pairs[0][0]
        if not isinstance(query, str) or not query.strip():
            raise RerankerUnavailable("SiliconFlow reranker query 不能为空")

        documents = []
        for pair in pairs:
            if len(pair) != 2 or pair[0] != query or not isinstance(pair[1], str):
                raise RerankerUnavailable("SiliconFlow reranker 仅支持同一 query 的文本候选")
            documents.append(pair[1])

        payload = {
            "model": settings.reranker_model,
            "query": query,
            "documents": documents,
            "top_n": len(documents),
            "return_documents": False,
        }
        headers = {
            "Authorization": f"Bearer {settings.reranker_api_key}",
            "Content-Type": "application/json",
        }
        attempts = settings.reranker_max_retries + 1
        for attempt in range(attempts):
            try:
                response = self._client.post(
                    settings.reranker_url,
                    headers=headers,
                    json=payload,
                )
            except httpx.RequestError as exc:
                if attempt + 1 < attempts:
                    self._sleep_before_retry(attempt)
                    continue
                raise RerankerUnavailable(
                    f"SiliconFlow reranker 网络请求失败: {type(exc).__name__}"
                ) from exc

            trace_id = response.headers.get("x-siliconcloud-trace-id", "")
            self.last_trace_id = trace_id
            if response.status_code >= 400:
                if response.status_code in _RETRYABLE_STATUS_CODES and attempt + 1 < attempts:
                    self._sleep_before_retry(attempt)
                    continue
                detail = f"HTTP {response.status_code}"
                if trace_id:
                    detail += f", trace_id={trace_id}"
                raise RerankerUnavailable(f"SiliconFlow reranker 请求失败: {detail}")

            try:
                body = response.json()
            except ValueError as exc:
                raise RerankerUnavailable("SiliconFlow reranker 响应不是合法 JSON") from exc
            return self._parse_scores(body, len(documents), trace_id)

        raise RerankerUnavailable("SiliconFlow reranker 请求重试耗尽")

    @staticmethod
    def _sleep_before_retry(attempt: int) -> None:
        time.sleep(min(0.5 * (2**attempt), 4.0))

    @staticmethod
    def _parse_scores(body, document_count: int, trace_id: str = "") -> list[float]:
        if not isinstance(body, dict) or not isinstance(body.get("results"), list):
            raise RerankerUnavailable("SiliconFlow reranker 响应缺少 results")
        results = body["results"]
        if len(results) != document_count:
            raise RerankerUnavailable(
                "SiliconFlow reranker 返回数量异常: "
                f"docs={document_count}, results={len(results)}"
            )

        scores: list[float | None] = [None] * document_count
        for item in results:
            if not isinstance(item, dict) or isinstance(item.get("index"), bool):
                raise RerankerUnavailable("SiliconFlow reranker 返回缺少合法 index")
            index = item.get("index")
            score = item.get("relevance_score")
            if not isinstance(index, int) or index < 0 or index >= document_count:
                raise RerankerUnavailable("SiliconFlow reranker 返回 index 越界")
            if scores[index] is not None:
                raise RerankerUnavailable("SiliconFlow reranker 返回 index 重复")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise RerankerUnavailable("SiliconFlow reranker 返回非法 relevance_score")
            score = float(score)
            if not math.isfinite(score) or score < 0.0 or score > 1.0:
                raise RerankerUnavailable("SiliconFlow reranker relevance_score 不在 0~1")
            scores[index] = score

        if any(score is None for score in scores):
            detail = f", trace_id={trace_id}" if trace_id else ""
            raise RerankerUnavailable(f"SiliconFlow reranker 返回 index 不完整{detail}")
        return [float(score) for score in scores]


def get_reranker():
    """按 provider 懒加载进程级 reranker 单例。"""
    global _model
    if _model is not None:
        return _model

    with _model_lock:
        if _model is not None:
            return _model
        settings.validate_reranker()
        if settings.reranker_provider == "siliconflow":
            _model = _SiliconFlowReranker()
            return _model

        from sentence_transformers import CrossEncoder

        try:
            model = CrossEncoder(settings.reranker_model)
        except Exception as exc:  # noqa: BLE001
            raise RerankerUnavailable(f"重排模型加载失败: {type(exc).__name__}: {exc}") from exc

        # warmup：用最小输入触发一次真实前向，把缓存损坏暴露在加载阶段而非检索阶段。
        try:
            scores = model.predict([("健康检查", "健康检查")])
            if scores is None or len(list(scores)) != 1:
                raise RerankerUnavailable(f"重排模型健康检查返回异常: {scores!r}")
        except RerankerUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RerankerUnavailable(
                "重排模型健康检查失败（HF 缓存可能损坏，请检查 tokenizer.json 等文件是否为 0 字节）: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        _model = model
    return _model


def _sigmoid(x: float) -> float:
    # 数值稳定的 sigmoid
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _normalized(value: float | None, low: float, high: float) -> float:
    """把不同量纲的召回分数压到 0~1，避免直接相加造成权重失真。"""
    if value is None:
        return 0.0
    if high <= low:
        return 1.0
    return max(0.0, min(1.0, (float(value) - low) / (high - low)))


def rerank(query: str, docs, top_k: int):
    """按 reranker 重排候选，同时保留高 RRF/词法候选的保护名额。"""
    if not docs:
        return docs
    model = get_reranker()
    pairs = [[query, d.page_content] for d in docs]
    logits = list(model.predict(pairs))
    if len(logits) != len(docs):
        raise RuntimeError(f"Reranker 返回数量异常：docs={len(docs)}, scores={len(logits)}")
    # 本地旧版 CrossEncoder 可能返回 logit；SiliconFlow 返回 0~1 relevance_score。
    if logits and min(logits) >= 0.0 and max(logits) <= 1.0:
        scores = [float(value) for value in logits]
    else:
        scores = [_sigmoid(float(value)) for value in logits]

    rrf_values = [float(d.metadata.get("rrf_score") or 0.0) for d in docs]
    raw_values = [float(d.metadata.get("vector_score") or d.metadata.get("fts_score") or d.metadata.get("score") or 0.0) for d in docs]
    rrf_low, rrf_high = min(rrf_values or [0.0]), max(rrf_values or [1.0])
    raw_low, raw_high = min(raw_values or [0.0]), max(raw_values or [1.0])
    ranked: list[tuple[float, object]] = []
    for d, reranker_score in zip(docs, scores):
        original_score = float(d.metadata.get("score", 0.0) or 0.0)
        rrf_score = float(d.metadata.get("rrf_score") or 0.0)
        raw_score = float(d.metadata.get("vector_score") or d.metadata.get("fts_score") or original_score or 0.0)
        d.metadata["retrieval_score"] = original_score
        d.metadata["reranker_score"] = round(reranker_score, 4)
        d.metadata["score"] = round(reranker_score, 4)
        d.metadata["score_type"] = "reranker"
        d.metadata["reranked"] = True
        d.metadata["reranker_protected"] = False
        d.metadata["neighbor"] = False
        if settings.enable_reranker_blend:
            final_score = (
                settings.reranker_weight * reranker_score
                + settings.rrf_weight * _normalized(rrf_score, rrf_low, rrf_high)
                + settings.raw_score_weight * _normalized(raw_score, raw_low, raw_high)
            )
        else:
            final_score = reranker_score
        d.metadata["final_score"] = round(final_score, 6)
        ranked.append((final_score, d))

    ranked.sort(key=lambda item: item[0], reverse=True)
    # 将完整重排顺序写回候选元数据，评测可据此区分“重排丢失”和“最终 Top-K 截断”。
    for rank, (_, doc) in enumerate(ranked, start=1):
        doc.metadata["reranker_rank"] = rank
    selected = [doc for _, doc in ranked[:top_k]]
    if settings.enable_reranker_blend and settings.reranker_protect_slots and top_k > 0:
        # 从原始 RRF 前 N 个候选中补回少量被重排挤出的证据，防止单一模型排序失真。
        protected = sorted(
            docs,
            key=lambda d: float(d.metadata.get("rrf_score") or 0.0),
            reverse=True,
        )[:settings.reranker_protect_top_n]
        for candidate in protected:
            if candidate in selected:
                continue
            candidate.metadata["reranker_protected"] = True
            if len(selected) >= top_k:
                selected[-1] = candidate
            else:
                selected.append(candidate)
            if sum(item in selected for item in protected) >= settings.reranker_protect_slots:
                break
    selected = selected[:top_k]
    return selected
