from src.retrieval.hybrid import create_retriever


def test_create_retriever_filters_kwargs_for_bm25() -> None:
    retriever = create_retriever(
        method="bm25",
        bm25_weight=0.7,
        dense_weight=0.3,
        embedding_model="text-embedding-3-small",
    )

    assert retriever is not None
