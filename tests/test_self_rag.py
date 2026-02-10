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
        ("[Retrieval] yes", 5, 0.001),
        # Phase 2: relevance checks
        ("[IsRel] relevant", 3, 0.0005),
        ("[IsRel] relevant", 3, 0.0005),
        ("[IsRel] irrelevant", 3, 0.0005),
        # Phase 3: candidate 1 (generate, support, utility)
        ("The capital of France is Paris.", 20, 0.002),
        ("[IsSup] fully supported", 3, 0.0005),
        ("[IsUse] 5", 3, 0.001),
        # Phase 3: candidate 2 (generate, support, utility)
        ("France is located in Europe.", 20, 0.002),
        ("[IsSup] partially supported", 3, 0.0005),
        ("[IsUse] 3", 3, 0.001),
    ]

    rag = SelfRAG(
        mock_llm,
        mock_retriever,
        {"top_k": 3, "num_candidates": 2},
    )
    response = await rag.answer(sample_question, sample_corpus)

    assert response.answer == "The capital of France is Paris."
    assert response.num_retrieval_calls == 1
    # 1 decision + 3 relevance checks + 2 candidates × 3 calls each = 10
    assert response.num_llm_calls == 10
    assert response.architecture == "self_rag"


@pytest.mark.asyncio
async def test_tracks_cost_and_tokens(mock_llm, mock_retriever, sample_question, sample_corpus):
    """Self-RAG should accumulate tokens and cost from all LLM calls."""
    mock_llm.generate.side_effect = [
        ("[Retrieval] yes", 5, 0.001),
        ("[IsRel] relevant", 3, 0.0005),
        ("[IsRel] relevant", 3, 0.0005),
        ("[IsRel] irrelevant", 3, 0.0005),
        ("Paris", 20, 0.002),
        ("[IsSup] fully supported", 3, 0.0005),
        ("[IsUse] 5", 3, 0.001),
    ]

    rag = SelfRAG(
        mock_llm,
        mock_retriever,
        {"top_k": 3, "num_candidates": 1},
    )
    response = await rag.answer(sample_question, sample_corpus)

    assert response.total_tokens == 5 + 3 + 3 + 3 + 20 + 3 + 3
    assert response.total_cost_usd == pytest.approx(
        0.001 + 0.0005 + 0.0005 + 0.0005 + 0.002 + 0.0005 + 0.001
    )


@pytest.mark.asyncio
async def test_reasoning_chain_records_all_phases(
    mock_llm, mock_retriever, sample_question, sample_corpus
):
    """Reasoning chain should have steps for decision, retrieve, candidates, and selection."""
    mock_llm.generate.side_effect = [
        ("[Retrieval] yes", 5, 0.001),
        ("[IsRel] relevant", 3, 0.0005),
        ("[IsRel] relevant", 3, 0.0005),
        ("[IsRel] irrelevant", 3, 0.0005),
        ("Paris", 20, 0.002),
        ("[IsSup] fully supported", 3, 0.0005),
        ("[IsUse] 5", 3, 0.001),
    ]

    rag = SelfRAG(
        mock_llm,
        mock_retriever,
        {"top_k": 3, "num_candidates": 1},
    )
    response = await rag.answer(sample_question, sample_corpus)

    # At least: retrieval decision + retrieve + relevance + candidate steps
    assert len(response.reasoning_chain) >= 4
    # First step should be retrieval decision
    assert response.reasoning_chain[0].action == "retrieval_decision"


@pytest.mark.asyncio
async def test_retrieved_docs_populated(mock_llm, mock_retriever, sample_question, sample_corpus):
    """retrieved_docs should contain the retrieval result when retrieval occurs."""
    mock_llm.generate.side_effect = [
        ("[Retrieval] yes", 5, 0.001),
        ("[IsRel] relevant", 3, 0.0005),
        ("[IsRel] relevant", 3, 0.0005),
        ("[IsRel] irrelevant", 3, 0.0005),
        ("Paris", 20, 0.002),
        ("[IsSup] fully supported", 3, 0.0005),
        ("[IsUse] 5", 3, 0.001),
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
        ("[Retrieval] no", 5, 0.001),
        # Direct answer
        ("Paris", 10, 0.001),
        # Utility critique
        ("[IsUse] 5", 3, 0.001),
    ]

    rag = SelfRAG(
        mock_llm,
        mock_retriever,
        {"top_k": 3, "num_candidates": 2},
    )
    response = await rag.answer(sample_question, sample_corpus)

    assert response.answer == "Paris"
    assert response.num_retrieval_calls == 0
    assert response.num_llm_calls == 3
    mock_retriever.retrieve.assert_not_called()


@pytest.mark.asyncio
async def test_no_retrieval_reasoning_chain(
    mock_llm, mock_retriever, sample_question, sample_corpus
):
    """Reasoning chain for no-retrieval path should have decision + direct answer."""
    mock_llm.generate.side_effect = [
        ("[Retrieval] no", 5, 0.001),
        ("Paris", 10, 0.001),
        ("[IsUse] 5", 3, 0.001),
    ]

    rag = SelfRAG(
        mock_llm,
        mock_retriever,
        {"top_k": 3, "num_candidates": 2},
    )
    response = await rag.answer(sample_question, sample_corpus)

    assert len(response.reasoning_chain) == 3
    assert response.reasoning_chain[0].action == "retrieval_decision"
    assert response.reasoning_chain[1].action == "direct_answer"
    assert response.reasoning_chain[2].action == "rate_utility"


# ── Tests: edge cases ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_supported_candidates_falls_back(
    mock_llm, mock_retriever, sample_question, sample_corpus
):
    """If no candidates are well-supported, return the best available anyway."""
    mock_llm.generate.side_effect = [
        ("[Retrieval] yes", 5, 0.001),
        ("[IsRel] irrelevant", 3, 0.0005),
        ("[IsRel] irrelevant", 3, 0.0005),
        ("[IsRel] irrelevant", 3, 0.0005),
        ("Maybe Paris", 20, 0.002),
        ("[IsSup] no support", 3, 0.0005),
        ("[IsUse] 1", 3, 0.001),
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
        ("[Retrieval] yes", 5, 0.001),
        ("[IsRel] relevant", 3, 0.0005),
        ("[IsRel] relevant", 3, 0.0005),
        ("[IsRel] irrelevant", 3, 0.0005),
        # No reflection tokens at all
        ("Paris is the capital", 20, 0.002),
        # Support critique missing token
        ("unsure", 3, 0.001),
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
        ("[IsRel] relevant", 3, 0.0005),
        ("[IsRel] relevant", 3, 0.0005),
        ("[IsRel] irrelevant", 3, 0.0005),
        ("Paris", 20, 0.002),
        ("[IsSup] fully supported", 3, 0.0005),
        ("[IsUse] 5", 3, 0.001),
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


@pytest.mark.asyncio
async def test_retrieval_decision_parses_token(
    mock_llm, mock_retriever, sample_question, sample_corpus
):
    mock_llm.generate.side_effect = [
        ("Consider retrieval. [Retrieval] no", 5, 0.001),
        ("Paris", 10, 0.001),
        ("[IsUse] 5", 3, 0.001),
    ]

    rag = SelfRAG(mock_llm, mock_retriever, {})
    response = await rag.answer(sample_question, sample_corpus)

    assert response.num_retrieval_calls == 0
    assert response.answer == "Paris"


def test_parse_utility_prefers_isuse_token(mock_llm, mock_retriever):
    rag = SelfRAG(mock_llm, mock_retriever, {})
    response = "Rating: 2\n[IsUse] 5"
    assert rag._parse_utility(response) == 5


@pytest.mark.asyncio
async def test_relevance_filtering_limits_candidates(
    mock_llm, mock_retriever, sample_question, sample_corpus
):
    mock_llm.generate.side_effect = [
        ("[Retrieval] yes", 5, 0.001),
        ("[IsRel] relevant", 3, 0.0005),
        ("[IsRel] irrelevant", 3, 0.0005),
        ("Paris", 20, 0.002),
        ("[IsSup] fully supported", 3, 0.0005),
        ("[IsUse] 5", 3, 0.001),
    ]

    rag = SelfRAG(mock_llm, mock_retriever, {"top_k": 2, "num_candidates": 2})
    response = await rag.answer(sample_question, sample_corpus)

    assert response.answer == "Paris"
    assert response.num_llm_calls == 6


@pytest.mark.asyncio
async def test_support_critique_uses_generated_answer(
    mock_llm, mock_retriever, sample_question, sample_corpus
):
    mock_llm.generate.side_effect = [
        ("[Retrieval] yes", 5, 0.001),
        ("[IsRel] relevant", 3, 0.0005),
        ("Paris", 20, 0.002),
        ("[IsSup] fully supported", 3, 0.0005),
        ("[IsUse] 5", 3, 0.001),
    ]

    rag = SelfRAG(
        mock_llm,
        mock_retriever,
        {"top_k": 1, "num_candidates": 1},
    )
    response = await rag.answer(sample_question, sample_corpus)

    assert response.answer == "Paris"
    support_prompt = mock_llm.generate.call_args_list[3].args[0][0]["content"]
    assert "Paris" in support_prompt


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
