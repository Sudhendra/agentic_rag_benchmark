from unittest.mock import AsyncMock

import pytest

from src.architectures.factory import create_architecture
from src.core.types import Document, RetrievalResult


@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    llm.model = "test-model"
    llm.generate.return_value = ("Test answer", 100, 0.001)
    return llm


@pytest.fixture
def mock_retriever():
    retriever = AsyncMock()
    retriever.retrieve.return_value = RetrievalResult(
        documents=[Document(id="1", title="Test Doc", text="Test content.")],
        scores=[0.9],
        query="test query",
        retrieval_time_ms=10.0,
        method="bm25",
    )
    return retriever


def test_create_architecture_react(mock_llm, mock_retriever):
    rag = create_architecture("react_rag", mock_llm, mock_retriever, {"react": {}})
    assert rag.get_name() == "react_rag"


def test_create_architecture_react_merges_top_level_config(mock_llm, mock_retriever):
    rag = create_architecture(
        "react_rag",
        mock_llm,
        mock_retriever,
        {"top_k": 7, "react": {"max_iterations": 3}},
    )

    assert rag.config["top_k"] == 7
    assert rag.config["max_iterations"] == 3
    assert "react" not in rag.config
