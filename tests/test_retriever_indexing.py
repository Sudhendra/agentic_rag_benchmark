from src.core.retriever import BaseRetriever
from src.core.types import Document, RetrievalResult


class DummyRetriever(BaseRetriever):
    def __init__(self) -> None:
        super().__init__()
        self.index_calls = 0

    async def index(self, corpus: list[Document]) -> None:
        self.index_calls += 1
        self._corpus = corpus
        self.corpus_size = len(corpus)
        self._indexed_corpus_signature = self._get_corpus_signature(corpus)
        self.is_indexed = True

    async def retrieve(self, query: str, corpus: list[Document], top_k: int = 5) -> RetrievalResult:
        if self._ensure_indexed(corpus):
            await self.index(corpus)

        return RetrievalResult(
            documents=self._corpus[:top_k],
            scores=[1.0] * min(top_k, len(self._corpus)),
            query=query,
            retrieval_time_ms=0.0,
            method="bm25",
        )


async def test_reindexes_when_corpus_identity_changes_with_same_length() -> None:
    retriever = DummyRetriever()
    first_corpus = [
        Document(id="a", title="A", text="apple orchard"),
        Document(id="b", title="B", text="banana grove"),
    ]
    second_corpus = [
        Document(id="c", title="C", text="carrot patch"),
        Document(id="d", title="D", text="dragonfruit field"),
    ]

    await retriever.retrieve("first", first_corpus, top_k=1)
    second_result = await retriever.retrieve("second", second_corpus, top_k=1)

    assert retriever.index_calls == 2
    assert second_result.documents[0].id == "c"


async def test_reindexes_when_corpus_is_mutated_in_place() -> None:
    retriever = DummyRetriever()
    corpus = [
        Document(id="a", title="A", text="apple orchard"),
        Document(id="b", title="B", text="banana grove"),
    ]

    await retriever.retrieve("first", corpus, top_k=1)
    corpus[0] = Document(id="c", title="C", text="carrot patch")
    second_result = await retriever.retrieve("second", corpus, top_k=1)

    assert retriever.index_calls == 2
    assert second_result.documents[0].id == "c"
