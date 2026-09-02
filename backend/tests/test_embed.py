"""Embedding API 适配测试，不访问真实模型服务。"""

import pytest
from langchain_core.documents import Document

from app.core import embed
from app.core.retrieval import indexer


class _FakeEmbeddings:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        return [[float(index), 1.0] for index, _ in enumerate(texts)]

    def embed_query(self, text):
        self.calls.append([text])
        return [1.0, 1.0]


def test_api_embedding_disables_openai_tokenization(monkeypatch):
    captured = {}

    class FakeOpenAIEmbeddings:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("langchain_openai.OpenAIEmbeddings", FakeOpenAIEmbeddings)
    monkeypatch.setattr(embed, "_instance", None)
    monkeypatch.setattr(embed.settings, "embedding_provider", "api")
    monkeypatch.setattr(embed.settings, "embedding_model", "Qwen/Qwen3-Embedding-0.6B")
    monkeypatch.setattr(embed.settings, "embedding_base_url", "https://example.test/v1")
    monkeypatch.setattr(embed.settings, "embedding_timeout", 10.0)
    monkeypatch.setattr(embed.settings, "embedding_max_retries", 1)
    monkeypatch.setattr(embed.settings, "embedding_check_ctx_length", False)

    result = embed.get_embeddings()

    assert isinstance(result, FakeOpenAIEmbeddings)
    assert captured["model"] == "Qwen/Qwen3-Embedding-0.6B"
    assert captured["check_embedding_ctx_length"] is False
    assert captured["base_url"] == "https://example.test/v1"


def test_openai_embeddings_sends_raw_strings_when_length_check_disabled(monkeypatch):
    requests = []

    class FakeClient:
        def create(self, **kwargs):
            requests.append(kwargs)
            return {"data": [{"embedding": [1.0, 2.0]} for _ in kwargs["input"]]}

    from langchain_openai import OpenAIEmbeddings

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    embeddings = OpenAIEmbeddings(
        model="Qwen/Qwen3-Embedding-0.6B",
        base_url="https://example.test/v1",
        check_embedding_ctx_length=False,
        client=FakeClient(),
    )
    texts = ["贾宝玉初见林黛玉。", "换行文本\n含有 punctuation!"]

    assert embeddings.embed_documents(texts) == [[1.0, 2.0], [1.0, 2.0]]
    assert requests == [{"input": texts, "model": "Qwen/Qwen3-Embedding-0.6B"}]
    assert all(isinstance(item, str) for item in requests[0]["input"])


def test_openai_embeddings_query_sends_raw_string(monkeypatch):
    requests = []

    class FakeClient:
        def create(self, **kwargs):
            requests.append(kwargs)
            return {"data": [{"embedding": [1.0, 2.0]}]}

    from langchain_openai import OpenAIEmbeddings

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    embeddings = OpenAIEmbeddings(
        model="Qwen/Qwen3-Embedding-0.6B",
        base_url="https://example.test/v1",
        check_embedding_ctx_length=False,
        client=FakeClient(),
    )
    query = "秦可卿的兄弟是谁？"

    assert embeddings.embed_query(query) == [1.0, 2.0]
    assert requests == [{"input": [query], "model": "Qwen/Qwen3-Embedding-0.6B"}]


@pytest.mark.asyncio
async def test_indexer_preserves_embedding_batch_order(monkeypatch):
    fake = _FakeEmbeddings()
    monkeypatch.setattr(indexer, "get_embeddings", lambda: fake)
    monkeypatch.setattr(indexer.settings, "embedding_batch_size", 2)
    documents = [Document(page_content=f"片段-{index}") for index in range(5)]

    vectors = await indexer._embed_documents_batched(documents)

    assert fake.calls == [["片段-0", "片段-1"], ["片段-2", "片段-3"], ["片段-4"]]
    assert vectors == [[0.0, 1.0], [1.0, 1.0], [0.0, 1.0], [1.0, 1.0], [0.0, 1.0]]
