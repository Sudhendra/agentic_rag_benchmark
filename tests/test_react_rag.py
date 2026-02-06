"""Unit tests for ReAct RAG architecture."""

from unittest.mock import AsyncMock

import pytest

from src.architectures.agentic.react_rag import ReActRAG
from src.core.types import Document, Question, QuestionType, RetrievalResult


@pytest.fixture
def mock_llm():
    """Create a mock LLM client."""
    llm = AsyncMock()
    llm.model = "test-model"
    return llm


@pytest.fixture
def mock_retriever():
    """Create a mock retriever."""
    retriever = AsyncMock()
    retriever.retrieve.return_value = RetrievalResult(
        documents=[
            Document(id="1", title="Alpha Doc", text="Alpha is a test term."),
            Document(id="2", title="Beta Doc", text="Beta content here."),
        ],
        scores=[0.9, 0.8],
        query="alpha",
        retrieval_time_ms=10.0,
        method="bm25",
    )
    return retriever


@pytest.fixture
def sample_question():
    """Create a sample question for testing."""
    return Question(
        id="q1",
        text="What is alpha?",
        type=QuestionType.SINGLE_HOP,
        gold_answer="Final answer",
    )


@pytest.fixture
def sample_corpus():
    """Create a sample corpus for testing."""
    return [
        Document(id="d1", title="Doc 1", text="Alpha appears here."),
        Document(id="d2", title="Doc 2", text="Other text."),
    ]


@pytest.mark.asyncio
async def test_react_rag_search_and_finish(
    mock_llm, mock_retriever, sample_question, sample_corpus
):
    mock_llm.generate.side_effect = [
        ("Thought: search\nAction: search[alpha]", 10, 0.001),
        ("Thought: done\nAction: finish[Final answer]", 12, 0.001),
    ]
    rag = ReActRAG(mock_llm, mock_retriever, {"top_k": 3, "max_iterations": 2})
    response = await rag.answer(sample_question, sample_corpus)
    assert response.answer == "Final answer"
    assert response.num_retrieval_calls == 1
    assert response.num_llm_calls == 2


@pytest.mark.asyncio
async def test_react_rag_lookup_uses_retrieved_docs(
    mock_llm, mock_retriever, sample_question, sample_corpus
):
    mock_llm.generate.side_effect = [
        ("Thought: search\nAction: search[alpha]", 10, 0.001),
        ("Thought: lookup\nAction: lookup[alpha]", 9, 0.001),
    ]
    rag = ReActRAG(mock_llm, mock_retriever, {"top_k": 3, "max_iterations": 2})
    response = await rag.answer(sample_question, sample_corpus)
    assert "lookup" in response.reasoning_chain[1].action


@pytest.mark.asyncio
async def test_react_rag_max_iterations_fallback(
    mock_llm, mock_retriever, sample_question, sample_corpus
):
    mock_llm.generate.return_value = ("Thought: search\nAction: search[alpha]", 10, 0.001)
    rag = ReActRAG(mock_llm, mock_retriever, {"top_k": 3, "max_iterations": 1})
    response = await rag.answer(sample_question, sample_corpus)
    assert response.answer
