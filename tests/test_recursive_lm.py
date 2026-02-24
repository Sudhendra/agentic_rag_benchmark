"""Unit tests for Recursive Language Model (RLM) architecture."""

from unittest.mock import AsyncMock

import pytest

from src.architectures.rlm.recursive_lm import RecursiveLM
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
            Document(id="2", title="Doc B", text="Berlin is the capital of Germany."),
        ],
        scores=[0.9, 0.8],
        query="capital",
        retrieval_time_ms=10.0,
        method="bm25",
    )
    return retriever


@pytest.fixture
def sample_question():
    """Create a sample multi-hop question."""
    return Question(
        id="q1",
        text="What is the capital of France?",
        type=QuestionType.SINGLE_HOP,
        gold_answer="Paris",
    )


@pytest.fixture
def bridge_question():
    """Create a sample bridge question requiring decomposition."""
    return Question(
        id="q2",
        text="Who is the mayor of the capital of France?",
        type=QuestionType.BRIDGE,
        gold_answer="Anne Hidalgo",
    )


@pytest.fixture
def sample_corpus():
    """Create a sample corpus."""
    return [
        Document(id="d1", title="France", text="Paris is the capital of France."),
        Document(id="d2", title="Germany", text="Berlin is the capital of Germany."),
    ]


def test_get_name():
    """Test architecture name."""
    llm = AsyncMock()
    llm.model = "test-model"
    retriever = AsyncMock()
    rag = RecursiveLM(llm, retriever, {})
    assert rag.get_name() == "recursive_lm"


def test_get_type():
    """Test architecture type."""
    llm = AsyncMock()
    llm.model = "test-model"
    retriever = AsyncMock()
    rag = RecursiveLM(llm, retriever, {})
    assert rag.get_type() == ArchitectureType.RLM


def test_config_defaults():
    """Test that default config is applied correctly."""
    llm = AsyncMock()
    llm.model = "test-model"
    retriever = AsyncMock()
    rag = RecursiveLM(llm, retriever, {})
    assert rag.config["max_depth"] == 3
    assert rag.config["top_k"] == 5
    assert rag.config["memoization"] is True
    assert rag.config["max_context_tokens"] == 4000


def test_config_override():
    """Test that config overrides are applied."""
    llm = AsyncMock()
    llm.model = "test-model"
    retriever = AsyncMock()
    rag = RecursiveLM(llm, retriever, {"max_depth": 2, "top_k": 3, "memoization": False})
    assert rag.config["max_depth"] == 2
    assert rag.config["top_k"] == 3
    assert rag.config["memoization"] is False


@pytest.mark.asyncio
async def test_direct_answer(mock_llm, mock_retriever, sample_question, sample_corpus):
    """Test that a simple question gets a direct answer without decomposition."""
    mock_llm.generate.return_value = ("DIRECT: Paris", 20, 0.002)

    rag = RecursiveLM(mock_llm, mock_retriever, {"max_depth": 3})
    response = await rag.answer(sample_question, sample_corpus)

    assert response.answer == "Paris"
    assert response.architecture == "recursive_lm"
    assert response.model == "test-model"
    # 1 retrieval (for the root question) + 1 LLM call (decompose decision)
    assert response.num_retrieval_calls == 1
    assert response.num_llm_calls == 1
    assert response.total_tokens == 20
    assert response.total_cost_usd == pytest.approx(0.002)
    assert len(response.reasoning_chain) == 1
    assert response.reasoning_chain[0].action == "finish"


@pytest.mark.asyncio
async def test_decompose_and_combine(mock_llm, mock_retriever, bridge_question, sample_corpus):
    """Test that a complex question gets decomposed, sub-questions answered, and combined."""
    mock_llm.generate.side_effect = [
        # Root question: decompose
        (
            "DECOMPOSE:\n"
            "- SUB: What is the capital of France?\n"
            "- SUB: Who is the mayor of Paris?\n"
            "COMBINE: Use the capital from Q1 and find its mayor from Q2.",
            30,
            0.003,
        ),
        # Sub-question 1: direct answer
        ("DIRECT: Paris", 15, 0.001),
        # Sub-question 2: direct answer
        ("DIRECT: Anne Hidalgo", 15, 0.001),
        # Combine call
        ("Anne Hidalgo", 10, 0.001),
    ]

    rag = RecursiveLM(mock_llm, mock_retriever, {"max_depth": 3})
    response = await rag.answer(bridge_question, sample_corpus)

    assert response.answer == "Anne Hidalgo"
    # 3 retrievals: root + 2 sub-questions
    assert response.num_retrieval_calls == 3
    # 4 LLM calls: root decompose + 2 sub-question decisions + 1 combine
    assert response.num_llm_calls == 4
    assert response.total_tokens == 30 + 15 + 15 + 10
    assert response.total_cost_usd == pytest.approx(0.006)

    # Check reasoning chain has decompose, two finish, and recurse steps
    actions = [step.action for step in response.reasoning_chain]
    assert "decompose" in actions
    assert actions.count("finish") == 2  # Two sub-questions answered directly
    assert "recurse" in actions  # Combination step


@pytest.mark.asyncio
async def test_max_depth_forces_direct(mock_llm, mock_retriever, bridge_question, sample_corpus):
    """Test that max_depth=0 forces a direct answer without decomposition."""
    # With max_depth=0, _recursive_answer should call _direct_answer immediately.
    mock_llm.generate.return_value = ("Paris", 10, 0.001)

    rag = RecursiveLM(mock_llm, mock_retriever, {"max_depth": 0})
    response = await rag.answer(bridge_question, sample_corpus)

    assert response.answer == "Paris"
    # 1 retrieval + 1 direct answer LLM call
    assert response.num_retrieval_calls == 1
    assert response.num_llm_calls == 1
    # Only a finish step (forced direct)
    assert len(response.reasoning_chain) == 1
    assert response.reasoning_chain[0].action == "finish"
    assert "Max depth" in response.reasoning_chain[0].thought


@pytest.mark.asyncio
async def test_memoization_cache_hit(mock_llm, mock_retriever, sample_corpus):
    """Test that memoization avoids re-answering the same sub-question."""
    # The decomposition produces two identical sub-questions.
    mock_llm.generate.side_effect = [
        # Root: decompose into two identical sub-questions
        (
            "DECOMPOSE:\n"
            "- SUB: What is the capital of France?\n"
            "- SUB: What is the capital of France?\n"
            "COMBINE: Return the answer from either sub-question.",
            30,
            0.003,
        ),
        # First sub-question: direct answer
        ("DIRECT: Paris", 15, 0.001),
        # Combine (second sub-question is a cache hit, no LLM call needed)
        ("Paris", 10, 0.001),
    ]

    question = Question(
        id="q3",
        text="What is the capital of France (twice)?",
        type=QuestionType.COMPOSITIONAL,
    )
    rag = RecursiveLM(mock_llm, mock_retriever, {"max_depth": 3, "memoization": True})
    response = await rag.answer(question, sample_corpus)

    assert response.answer == "Paris"
    # Only 3 LLM calls (root decompose + 1 sub-question + combine), not 4
    assert response.num_llm_calls == 3
    # Check we have a memo_hit step
    actions = [step.action for step in response.reasoning_chain]
    assert "memo_hit" in actions


@pytest.mark.asyncio
async def test_memoization_disabled(mock_llm, mock_retriever, sample_corpus):
    """Test that with memoization off, identical sub-questions are re-answered."""
    mock_llm.generate.side_effect = [
        # Root: decompose into two identical sub-questions
        (
            "DECOMPOSE:\n"
            "- SUB: What is the capital of France?\n"
            "- SUB: What is the capital of France?\n"
            "COMBINE: Return the answer.",
            30,
            0.003,
        ),
        # First sub-question
        ("DIRECT: Paris", 15, 0.001),
        # Second sub-question (no cache, answered again)
        ("DIRECT: Paris", 15, 0.001),
        # Combine
        ("Paris", 10, 0.001),
    ]

    question = Question(
        id="q4",
        text="Capital of France (twice)?",
        type=QuestionType.COMPOSITIONAL,
    )
    rag = RecursiveLM(mock_llm, mock_retriever, {"max_depth": 3, "memoization": False})
    response = await rag.answer(question, sample_corpus)

    assert response.answer == "Paris"
    # 4 LLM calls: root + both sub-questions + combine
    assert response.num_llm_calls == 4
    # No memo_hit steps
    actions = [step.action for step in response.reasoning_chain]
    assert "memo_hit" not in actions


@pytest.mark.asyncio
async def test_nested_decomposition(mock_llm, mock_retriever, sample_corpus):
    """Test two levels of decomposition (depth 0 decomposes, depth 1 decomposes again)."""
    mock_llm.generate.side_effect = [
        # Depth 0: decompose into 1 sub-question
        (
            "DECOMPOSE:\n- SUB: Who founded the capital of France?\nCOMBINE: State the founder.",
            25,
            0.002,
        ),
        # Depth 1: sub-question itself decomposes
        (
            "DECOMPOSE:\n"
            "- SUB: What is the capital of France?\n"
            "- SUB: Who founded Paris?\n"
            "COMBINE: Combine city and founder.",
            25,
            0.002,
        ),
        # Depth 2: direct answer for "What is the capital of France?"
        ("DIRECT: Paris", 10, 0.001),
        # Depth 2: direct answer for "Who founded Paris?"
        ("DIRECT: The Parisii tribe", 10, 0.001),
        # Depth 1 combine
        ("The Parisii tribe founded Paris.", 15, 0.001),
        # Depth 0 combine
        ("The Parisii tribe", 10, 0.001),
    ]

    question = Question(
        id="q5",
        text="Who founded the capital of France?",
        type=QuestionType.BRIDGE,
    )
    rag = RecursiveLM(mock_llm, mock_retriever, {"max_depth": 3, "memoization": False})
    response = await rag.answer(question, sample_corpus)

    assert response.answer == "The Parisii tribe"
    # 5 retrievals: root + 1 depth-1 + 2 depth-2 sub-questions + ... actually let's count:
    # depth 0 retrieves for root, depth 1 retrieves for "Who founded...",
    # depth 2 retrieves for "What is capital" and "Who founded Paris"
    assert response.num_retrieval_calls == 4
    # 6 LLM calls: 2 decompose decisions + 2 direct decisions + 2 combine
    assert response.num_llm_calls == 6


@pytest.mark.asyncio
async def test_fallback_parsing_no_format(mock_llm, mock_retriever, sample_question, sample_corpus):
    """Test that an unformatted LLM response is treated as a direct answer."""
    # The LLM just answers without using DIRECT: or DECOMPOSE: format.
    mock_llm.generate.return_value = ("The answer is Paris.", 15, 0.001)

    rag = RecursiveLM(mock_llm, mock_retriever, {"max_depth": 3})
    response = await rag.answer(sample_question, sample_corpus)

    # Should fall back to treating the entire response as a direct answer.
    assert response.answer == "The answer is Paris."
    assert response.num_llm_calls == 1


@pytest.mark.asyncio
async def test_response_metadata(mock_llm, mock_retriever, sample_question, sample_corpus):
    """Test that RAGResponse metadata fields are populated correctly."""
    mock_llm.generate.return_value = ("DIRECT: Paris", 20, 0.002)

    rag = RecursiveLM(mock_llm, mock_retriever, {})
    response = await rag.answer(sample_question, sample_corpus)

    assert response.model == "test-model"
    assert response.architecture == "recursive_lm"
    assert response.latency_ms > 0
    assert len(response.retrieved_docs) >= 1


@pytest.mark.asyncio
async def test_factory_creates_recursive_lm():
    """Test that the architecture factory can create RecursiveLM."""
    from src.architectures.factory import create_architecture

    llm = AsyncMock()
    llm.model = "test-model"
    retriever = AsyncMock()

    rag = create_architecture("recursive_lm", llm, retriever, {"rlm": {"max_depth": 2}})

    assert isinstance(rag, RecursiveLM)
    assert rag.get_name() == "recursive_lm"
    assert rag.config["max_depth"] == 2


def test_parse_decision_direct():
    """Test parsing a DIRECT response."""
    result = RecursiveLM._parse_decision("DIRECT: Paris")
    assert result["type"] == "direct"
    assert result["answer"] == "Paris"


def test_parse_decision_decompose():
    """Test parsing a DECOMPOSE response."""
    text = "DECOMPOSE:\n- SUB: What is X?\n- SUB: What is Y?\nCOMBINE: Merge X and Y."
    result = RecursiveLM._parse_decision(text)
    assert result["type"] == "decompose"
    assert len(result["sub_questions"]) == 2
    assert result["sub_questions"][0] == "What is X?"
    assert result["sub_questions"][1] == "What is Y?"
    assert result["combine_instruction"] == "Merge X and Y."


def test_parse_decision_fallback():
    """Test that unformatted text falls back to direct."""
    result = RecursiveLM._parse_decision("Just some random text answer.")
    assert result["type"] == "direct"
    assert result["answer"] == "Just some random text answer."
