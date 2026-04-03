import numpy as np
import pytest

from src.core.types import Document
from src.retrieval.dense import DenseRetriever


def _make_corpus() -> list[Document]:
    return [
        Document(id="doc-1", title="Doc 1", text="alpha"),
        Document(id="doc-2", title="Doc 2", text="beta"),
    ]


@pytest.mark.asyncio
async def test_dense_cached_embeddings_match_fresh_normalization(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("EMBEDDING_CACHE_DIR", str(tmp_path))

    corpus = _make_corpus()
    doc_embeddings = np.array([[3.0, 0.0], [0.0, 4.0]])
    query_embedding = np.array([[1.0, 0.0]])

    async def embed_texts(self: DenseRetriever, texts: list[str]) -> np.ndarray:
        if texts == [doc.text for doc in corpus]:
            return doc_embeddings.copy()
        if texts == ["query"]:
            return query_embedding.copy()
        raise AssertionError(f"Unexpected texts: {texts}")

    monkeypatch.setattr(DenseRetriever, "_embed_texts", embed_texts)

    fresh_retriever = DenseRetriever(api_key="test-key")
    await fresh_retriever.index(corpus)

    cached_retriever = DenseRetriever(api_key="test-key")
    await cached_retriever.index(corpus)

    assert np.array_equal(cached_retriever.embeddings, fresh_retriever.embeddings)


@pytest.mark.asyncio
async def test_dense_retrieval_order_matches_between_fresh_and_cached_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("EMBEDDING_CACHE_DIR", str(tmp_path))

    corpus = _make_corpus()
    doc_embeddings = np.array([[100.0, 0.0], [1.0, 1.0]])
    query_embedding = np.array([[1.0, 1.0]])

    async def embed_texts(self: DenseRetriever, texts: list[str]) -> np.ndarray:
        if texts == [doc.text for doc in corpus]:
            return doc_embeddings.copy()
        if texts == ["query"]:
            return query_embedding.copy()
        raise AssertionError(f"Unexpected texts: {texts}")

    monkeypatch.setattr(DenseRetriever, "_embed_texts", embed_texts)

    fresh_retriever = DenseRetriever(api_key="test-key")
    fresh_result = await fresh_retriever.retrieve("query", corpus, top_k=2)

    cached_retriever = DenseRetriever(api_key="test-key")
    cached_result = await cached_retriever.retrieve("query", corpus, top_k=2)

    assert [doc.id for doc in cached_result.documents] == [doc.id for doc in fresh_result.documents]


@pytest.mark.asyncio
async def test_dense_retrieval_order_uses_stable_tie_breaker(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("EMBEDDING_CACHE_DIR", str(tmp_path))

    corpus = _make_corpus()
    doc_embeddings = np.array([[1.0, 0.0], [1.0, 0.0]])
    query_embedding = np.array([[1.0, 0.0]])

    async def embed_texts(self: DenseRetriever, texts: list[str]) -> np.ndarray:
        if texts == [doc.text for doc in corpus]:
            return doc_embeddings.copy()
        if texts == ["query"]:
            return query_embedding.copy()
        raise AssertionError(f"Unexpected texts: {texts}")

    monkeypatch.setattr(DenseRetriever, "_embed_texts", embed_texts)

    retriever = DenseRetriever(api_key="test-key")
    result = await retriever.retrieve("query", corpus, top_k=2)

    assert [doc.id for doc in result.documents] == ["doc-1", "doc-2"]
