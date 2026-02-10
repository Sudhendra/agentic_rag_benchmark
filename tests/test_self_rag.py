"""Unit tests for Self-RAG architecture."""

from unittest.mock import AsyncMock

import pytest

from src.architectures.agentic.self_rag import SelfRAG
from src.core.types import (
    ArchitectureType,
    Document,
    Question,
    QuestionType,
    RetrievalResult,
)


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
            Document(id="1", title="Doc A", text="Paris is the capital of France."),
            Document(id="2", title="Doc B", text="France is in Europe."),
            Document(id="3", title="Doc C", text="Berlin is the capital of Germany."),
        ],
        scores=[0.9, 0.85, 0.7],
        query="capital of France",
        retrieval_time_ms=10.0,
        method="bm25",
    )
    return retriever


@pytest.fixture
def sample_question():
    """Create a sample question for testing."""
    return Question(
        id="q1",
        text="What is the capital of France?",
        type=QuestionType.BRIDGE,
        gold_answer="Paris",
    )


@pytest.fixture
def sample_corpus():
    """Create a sample corpus for testing."""
    return [
        Document(id="d1", title="Doc 1", text="Paris is the capital of France."),
        Document(id="d2", title="Doc 2", text="Berlin is the capital of Germany."),
    ]


# ── Tests: retrieval path ───────────────────────────────────────


@pytest.mark.asyncio
async def test_retrieval_needed_returns_best_candidate(
    mock_llm, mock_retriever, sample_question, sample_corpus
):
    """When retrieval is needed, Self-RAG retrieves, generates candidates, and picks best."""
    mock_llm.generate.side_effect = [
        # Phase 1: retrieval decision
        ("yes", 5, 0.001),
        # Phase 3a: candidate 1 - generate answer with relevance+support
        (
            "The capital of France is Paris.\n[IsRel] relevant\n[IsSup] fully supported",
            20,
            0.002,
        ),
        # Phase 3b: candidate 1 - utility
        ("5", 3, 0.001),
        # Phase 3a: candidate 2 - generate answer with relevance+support
        (
            "France is located in Europe.\n[IsRel] relevant\n[IsSup] partially supported",
            20,
            0.002,
        ),
        # Phase 3b: candidate 2 - utility
        ("3", 3, 0.001),
    ]

    rag = SelfRAG(
        mock_llm,
        mock_retriever,
        {"top_k": 3, "num_candidates": 2},
    )
    response = await rag.answer(sample_question, sample_corpus)

    assert response.answer == "The capital of France is Paris."
    assert response.num_retrieval_calls == 1
    # 1 retrieval decision + 2 candidates × 2 calls each = 5
    assert response.num_llm_calls == 5
    assert response.architecture == "self_rag"


@pytest.mark.asyncio
async def test_tracks_cost_and_tokens(mock_llm, mock_retriever, sample_question, sample_corpus):
    """Self-RAG should accumulate tokens and cost from all LLM calls."""
    mock_llm.generate.side_effect = [
        ("yes", 5, 0.001),
        ("Paris\n[IsRel] relevant\n[IsSup] fully supported", 20, 0.002),
        ("5", 3, 0.001),
    ]

    rag = SelfRAG(
        mock_llm,
        mock_retriever,
        {"top_k": 3, "num_candidates": 1},
    )
    response = await rag.answer(sample_question, sample_corpus)

    assert response.total_tokens == 5 + 20 + 3
    assert response.total_cost_usd == pytest.approx(0.001 + 0.002 + 0.001)


@pytest.mark.asyncio
async def test_reasoning_chain_records_all_phases(
    mock_llm, mock_retriever, sample_question, sample_corpus
):
    """Reasoning chain should have steps for decision, retrieve, candidates, and selection."""
    mock_llm.generate.side_effect = [
        ("yes", 5, 0.001),
        ("Paris\n[IsRel] relevant\n[IsSup] fully supported", 20, 0.002),
        ("5", 3, 0.001),
    ]

    rag = SelfRAG(
        mock_llm,
        mock_retriever,
        {"top_k": 3, "num_candidates": 1},
    )
    response = await rag.answer(sample_question, sample_corpus)

    # At least: retrieval decision step + candidate generation step
    assert len(response.reasoning_chain) >= 2
    # First step should be retrieval decision
    assert response.reasoning_chain[0].action == "retrieval_decision"


@pytest.mark.asyncio
async def test_retrieved_docs_populated(mock_llm, mock_retriever, sample_question, sample_corpus):
    """retrieved_docs should contain the retrieval result when retrieval occurs."""
    mock_llm.generate.side_effect = [
        ("yes", 5, 0.001),
        ("Paris\n[IsRel] relevant\n[IsSup] fully supported", 20, 0.002),
        ("5", 3, 0.001),
    ]

    rag = SelfRAG(
        mock_llm,
        mock_retriever,
        {"top_k": 3, "num_candidates": 1},
    )
    response = await rag.answer(sample_question, sample_corpus)

    assert len(response.retrieved_docs) == 1
    assert response.retrieved_docs[0].query == "capital of France"


# ── Tests: no-retrieval path ────────────────────────────────────


@pytest.mark.asyncio
async def test_no_retrieval_answers_directly(
    mock_llm, mock_retriever, sample_question, sample_corpus
):
    """When retrieval is not needed, Self-RAG answers directly with no retriever call."""
    mock_llm.generate.side_effect = [
        # Phase 1: no retrieval needed
        ("no", 5, 0.001),
        # Direct answer
        ("Paris", 10, 0.001),
    ]

    rag = SelfRAG(
        mock_llm,
        mock_retriever,
        {"top_k": 3, "num_candidates": 2},
    )
    response = await rag.answer(sample_question, sample_corpus)

    assert response.answer == "Paris"
    assert response.num_retrieval_calls == 0
    assert response.num_llm_calls == 2
    mock_retriever.retrieve.assert_not_called()


@pytest.mark.asyncio
async def test_no_retrieval_reasoning_chain(
    mock_llm, mock_retriever, sample_question, sample_corpus
):
    """Reasoning chain for no-retrieval path should have decision + direct answer."""
    mock_llm.generate.side_effect = [
        ("no", 5, 0.001),
        ("Paris", 10, 0.001),
    ]

    rag = SelfRAG(
        mock_llm,
        mock_retriever,
        {"top_k": 3, "num_candidates": 2},
    )
    response = await rag.answer(sample_question, sample_corpus)

    assert len(response.reasoning_chain) == 2
    assert response.reasoning_chain[0].action == "retrieval_decision"
    assert response.reasoning_chain[1].action == "direct_answer"


# ── Tests: edge cases ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_supported_candidates_falls_back(
    mock_llm, mock_retriever, sample_question, sample_corpus
):
    """If no candidates are well-supported, return the best available anyway."""
    mock_llm.generate.side_effect = [
        ("yes", 5, 0.001),
        ("Maybe Paris\n[IsRel] irrelevant\n[IsSup] no support", 20, 0.002),
        ("1", 3, 0.001),
    ]

    rag = SelfRAG(
        mock_llm,
        mock_retriever,
        {"top_k": 3, "num_candidates": 1},
    )
    response = await rag.answer(sample_question, sample_corpus)

    # Should still return an answer even if poorly supported
    assert response.answer == "Maybe Paris"


@pytest.mark.asyncio
async def test_malformed_llm_response_handles_gracefully(
    mock_llm, mock_retriever, sample_question, sample_corpus
):
    """If the LLM doesn't produce parseable reflection tokens, use defaults."""
    mock_llm.generate.side_effect = [
        ("yes", 5, 0.001),
        # No reflection tokens at all
        ("Paris is the capital", 20, 0.002),
        # Non-numeric utility
        ("high", 3, 0.001),
    ]

    rag = SelfRAG(
        mock_llm,
        mock_retriever,
        {"top_k": 3, "num_candidates": 1},
    )
    response = await rag.answer(sample_question, sample_corpus)

    # Should still return the generation as the answer
    assert response.answer == "Paris is the capital"


@pytest.mark.asyncio
async def test_ambiguous_retrieval_decision_defaults_to_yes(
    mock_llm, mock_retriever, sample_question, sample_corpus
):
    """If retrieval decision is ambiguous, default to retrieving."""
    mock_llm.generate.side_effect = [
        ("I think it might be helpful to look up some info", 10, 0.001),
        ("Paris\n[IsRel] relevant\n[IsSup] fully supported", 20, 0.002),
        ("5", 3, 0.001),
    ]

    rag = SelfRAG(
        mock_llm,
        mock_retriever,
        {"top_k": 3, "num_candidates": 1},
    )
    response = await rag.answer(sample_question, sample_corpus)

    # Should have retrieved since ambiguous defaults to yes
    assert response.num_retrieval_calls == 1


# ── Tests: metadata and config ──────────────────────────────────


def test_get_name(mock_llm, mock_retriever):
    rag = SelfRAG(mock_llm, mock_retriever, {})
    assert rag.get_name() == "self_rag"


def test_get_type(mock_llm, mock_retriever):
    rag = SelfRAG(mock_llm, mock_retriever, {})
    assert rag.get_type() == ArchitectureType.AGENTIC


def test_default_config_values(mock_llm, mock_retriever):
    rag = SelfRAG(mock_llm, mock_retriever, {})
    assert rag.config["top_k"] == 5
    assert rag.config["num_candidates"] == 3
    assert rag.config["max_context_tokens"] == 4000


def test_custom_config_values(mock_llm, mock_retriever):
    rag = SelfRAG(
        mock_llm,
        mock_retriever,
        {"top_k": 10, "num_candidates": 5},
    )
    assert rag.config["top_k"] == 10
    assert rag.config["num_candidates"] == 5
