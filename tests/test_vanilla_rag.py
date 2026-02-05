"""Unit tests for Vanilla RAG architecture."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.architectures.vanilla_rag import VanillaRAG
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
    llm.generate.return_value = ("Test answer", 100, 0.001)
    return llm


@pytest.fixture
def mock_retriever():
    """Create a mock retriever."""
    retriever = AsyncMock()
    retriever.retrieve.return_value = RetrievalResult(
        documents=[Document(id="1", title="Test Doc", text="Test content.")],
        scores=[0.9],
        query="test query",
        retrieval_time_ms=10.0,
        method="bm25",
    )
    return retriever


@pytest.fixture
def sample_question():
    """Create a sample question for testing."""
    return Question(
        id="q1",
        text="What is the test?",
        type=QuestionType.SINGLE_HOP,
        gold_answer="Test answer",
    )


@pytest.fixture
def sample_corpus():
    """Create a sample corpus for testing."""
    return [
        Document(id="d1", title="Doc 1", text="This is document one."),
        Document(id="d2", title="Doc 2", text="This is document two."),
    ]


class TestVanillaRAGBasic:
    """Basic functionality tests for VanillaRAG."""

    @pytest.mark.asyncio
    async def test_returns_answer(self, mock_llm, mock_retriever, sample_question, sample_corpus):
        """Test that VanillaRAG returns an answer."""
        rag = VanillaRAG(mock_llm, mock_retriever, {"top_k": 5, "max_context_tokens": 4000})

        response = await rag.answer(sample_question, sample_corpus)

        assert response.answer == "Test answer"
        assert response.architecture == "vanilla_rag"

    @pytest.mark.asyncio
    async def test_tracks_retrieval_calls(
        self, mock_llm, mock_retriever, sample_question, sample_corpus
    ):
        """Test that VanillaRAG tracks the number of retrieval calls."""
        rag = VanillaRAG(mock_llm, mock_retriever, {"top_k": 5})

        response = await rag.answer(sample_question, sample_corpus)

        assert response.num_retrieval_calls == 1

    @pytest.mark.asyncio
    async def test_tracks_llm_calls(self, mock_llm, mock_retriever, sample_question, sample_corpus):
        """Test that VanillaRAG tracks the number of LLM calls."""
        rag = VanillaRAG(mock_llm, mock_retriever, {"top_k": 5})

        response = await rag.answer(sample_question, sample_corpus)

        assert response.num_llm_calls == 1


class TestVanillaRAGRetrieval:
    """Tests for VanillaRAG retrieval behavior."""

    @pytest.mark.asyncio
    async def test_calls_retriever_with_question(
        self, mock_llm, mock_retriever, sample_question, sample_corpus
    ):
        """Test that VanillaRAG passes the question to the retriever."""
        rag = VanillaRAG(mock_llm, mock_retriever, {"top_k": 3})

        await rag.answer(sample_question, sample_corpus)

        mock_retriever.retrieve.assert_called_once()
        call_kwargs = mock_retriever.retrieve.call_args.kwargs
        assert call_kwargs["query"] == sample_question.text

    @pytest.mark.asyncio
    async def test_uses_configured_top_k(
        self, mock_llm, mock_retriever, sample_question, sample_corpus
    ):
        """Test that VanillaRAG uses the configured top_k value."""
        rag = VanillaRAG(mock_llm, mock_retriever, {"top_k": 7})

        await rag.answer(sample_question, sample_corpus)

        call_kwargs = mock_retriever.retrieve.call_args.kwargs
        assert call_kwargs["top_k"] == 7

    @pytest.mark.asyncio
    async def test_passes_corpus_to_retriever(
        self, mock_llm, mock_retriever, sample_question, sample_corpus
    ):
        """Test that VanillaRAG passes the corpus to the retriever."""
        rag = VanillaRAG(mock_llm, mock_retriever, {"top_k": 5})

        await rag.answer(sample_question, sample_corpus)

        call_kwargs = mock_retriever.retrieve.call_args.kwargs
        assert call_kwargs["corpus"] == sample_corpus


class TestVanillaRAGCostTracking:
    """Tests for VanillaRAG cost tracking."""

    @pytest.mark.asyncio
    async def test_tracks_cost(self, mock_llm, mock_retriever, sample_question, sample_corpus):
        """Test that VanillaRAG tracks the cost."""
        rag = VanillaRAG(mock_llm, mock_retriever, {})

        response = await rag.answer(sample_question, sample_corpus)

        assert response.total_cost_usd == 0.001

    @pytest.mark.asyncio
    async def test_tracks_tokens(self, mock_llm, mock_retriever, sample_question, sample_corpus):
        """Test that VanillaRAG tracks the token count."""
        rag = VanillaRAG(mock_llm, mock_retriever, {})

        response = await rag.answer(sample_question, sample_corpus)

        assert response.total_tokens == 100

    @pytest.mark.asyncio
    async def test_tracks_latency(self, mock_llm, mock_retriever, sample_question, sample_corpus):
        """Test that VanillaRAG tracks latency."""
        rag = VanillaRAG(mock_llm, mock_retriever, {})

        response = await rag.answer(sample_question, sample_corpus)

        assert response.latency_ms > 0


class TestVanillaRAGMetadata:
    """Tests for VanillaRAG metadata methods."""

    def test_get_name(self, mock_llm, mock_retriever):
        """Test that get_name returns the correct architecture name."""
        rag = VanillaRAG(mock_llm, mock_retriever, {})
        assert rag.get_name() == "vanilla_rag"

    def test_get_type(self, mock_llm, mock_retriever):
        """Test that get_type returns VANILLA."""
        rag = VanillaRAG(mock_llm, mock_retriever, {})
        assert rag.get_type() == ArchitectureType.VANILLA

    def test_get_config_schema(self, mock_llm, mock_retriever):
        """Test that get_config_schema returns expected keys."""
        rag = VanillaRAG(mock_llm, mock_retriever, {})
        schema = rag.get_config_schema()

        assert "top_k" in schema
        assert "max_context_tokens" in schema
        assert "prompt_path" in schema


class TestVanillaRAGConfig:
    """Tests for VanillaRAG configuration handling."""

    def test_default_config_values(self, mock_llm, mock_retriever):
        """Test that default config values are applied."""
        rag = VanillaRAG(mock_llm, mock_retriever, {})

        assert rag.config["top_k"] == 5
        assert rag.config["max_context_tokens"] == 4000

    def test_custom_config_values(self, mock_llm, mock_retriever):
        """Test that custom config values are used."""
        rag = VanillaRAG(mock_llm, mock_retriever, {"top_k": 10, "max_context_tokens": 2000})

        assert rag.config["top_k"] == 10
        assert rag.config["max_context_tokens"] == 2000


class TestVanillaRAGReasoningChain:
    """Tests for VanillaRAG reasoning chain output."""

    @pytest.mark.asyncio
    async def test_returns_reasoning_chain(
        self, mock_llm, mock_retriever, sample_question, sample_corpus
    ):
        """Test that VanillaRAG returns a reasoning chain."""
        rag = VanillaRAG(mock_llm, mock_retriever, {})

        response = await rag.answer(sample_question, sample_corpus)

        assert len(response.reasoning_chain) == 1
        assert response.reasoning_chain[0].step_id == 1

    @pytest.mark.asyncio
    async def test_reasoning_chain_contains_action(
        self, mock_llm, mock_retriever, sample_question, sample_corpus
    ):
        """Test that reasoning chain contains the action."""
        rag = VanillaRAG(mock_llm, mock_retriever, {})

        response = await rag.answer(sample_question, sample_corpus)

        step = response.reasoning_chain[0]
        assert step.action == "retrieve_and_answer"


class TestVanillaRAGRetrievedDocs:
    """Tests for VanillaRAG retrieved documents output."""

    @pytest.mark.asyncio
    async def test_returns_retrieved_docs(
        self, mock_llm, mock_retriever, sample_question, sample_corpus
    ):
        """Test that VanillaRAG returns retrieved documents."""
        rag = VanillaRAG(mock_llm, mock_retriever, {})

        response = await rag.answer(sample_question, sample_corpus)

        assert len(response.retrieved_docs) == 1
        assert response.retrieved_docs[0].method == "bm25"
