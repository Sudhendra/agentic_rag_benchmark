"""Unit tests for IRCoT recursive architecture."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src.architectures.recursive.ircot import IRCoTRAG
from src.core.types import ArchitectureType, Document, Question, QuestionType, RetrievalResult


@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    llm.model = "test-model"
    return llm


@pytest.fixture
def sample_corpus():
    return [
        Document(id="d1", title="France", text="Paris is the capital of France."),
        Document(id="d2", title="Paris", text="Anne Hidalgo is the mayor of Paris."),
        Document(id="d3", title="Europe", text="France is in Europe."),
    ]


@pytest.fixture
def bridge_question():
    return Question(
        id="q1",
        text="Who is the mayor of the capital of France?",
        type=QuestionType.BRIDGE,
        gold_answer="Anne Hidalgo",
    )


@pytest.fixture
def sample_question():
    return Question(
        id="q2",
        text="What is the capital of France?",
        type=QuestionType.SINGLE_HOP,
        gold_answer="Paris",
    )


def make_retrieval_result(query: str, documents: list[Document]) -> RetrievalResult:
    return RetrievalResult(
        documents=documents,
        scores=[0.9 - (i * 0.1) for i in range(len(documents))],
        query=query,
        retrieval_time_ms=10.0,
        method="bm25",
    )


def test_get_name():
    llm = AsyncMock()
    llm.model = "test-model"
    retriever = AsyncMock()
    rag = IRCoTRAG(llm, retriever, {})
    assert rag.get_name() == "ircot_rag"


def test_get_type():
    llm = AsyncMock()
    llm.model = "test-model"
    retriever = AsyncMock()
    rag = IRCoTRAG(llm, retriever, {})
    assert rag.get_type() == ArchitectureType.RECURSIVE


def test_config_defaults():
    llm = AsyncMock()
    llm.model = "test-model"
    retriever = AsyncMock()
    rag = IRCoTRAG(llm, retriever, {})
    assert rag.config["top_k"] == 3
    assert rag.config["max_steps"] == 4
    assert rag.config["answer_trigger"] == "[ANSWER]"
    assert rag.config["max_docs"] == 8


@pytest.mark.asyncio
async def test_ircot_interleaves_retrieval_and_reasoning(mock_llm, bridge_question, sample_corpus):
    retriever = AsyncMock()
    retriever.retrieve.side_effect = [
        make_retrieval_result(bridge_question.text, sample_corpus[:2]),
        make_retrieval_result("France capital Paris", sample_corpus[1:3]),
    ]
    mock_llm.generate.side_effect = [
        ("France's capital is Paris.", 10, 0.001),
        ("[ANSWER] Anne Hidalgo", 12, 0.001),
        ("Anne Hidalgo", 8, 0.001),
    ]

    rag = IRCoTRAG(mock_llm, retriever, {"max_steps": 3, "top_k": 2})
    response = await rag.answer(bridge_question, sample_corpus)

    assert response.answer == "Anne Hidalgo"
    assert response.num_retrieval_calls == 2
    assert response.num_llm_calls == 3
    assert response.total_tokens == 30
    assert response.total_cost_usd == pytest.approx(0.003)
    assert [step.action for step in response.reasoning_chain] == ["reason", "finish", "synthesize"]
    assert retriever.retrieve.call_args_list[1].kwargs["query"] == "France capital Paris"


@pytest.mark.asyncio
async def test_ircot_forces_final_synthesis_after_max_steps(
    mock_llm, sample_question, sample_corpus
):
    retriever = AsyncMock()
    retriever.retrieve.side_effect = [
        make_retrieval_result(sample_question.text, sample_corpus[:2]),
        make_retrieval_result("Paris capital city", sample_corpus[:2]),
        make_retrieval_result("France capital city", sample_corpus[:2]),
    ]
    mock_llm.generate.side_effect = [
        ("Paris is the capital city.", 8, 0.001),
        ("France has Paris as its capital city.", 8, 0.001),
        ("Paris", 10, 0.001),
    ]

    rag = IRCoTRAG(mock_llm, retriever, {"max_steps": 2, "top_k": 2})
    response = await rag.answer(sample_question, sample_corpus)

    assert response.answer == "Paris"
    assert response.num_retrieval_calls == 3
    assert response.num_llm_calls == 3
    assert [step.action for step in response.reasoning_chain] == ["reason", "reason", "synthesize"]


@pytest.mark.asyncio
async def test_ircot_deduplicates_documents_across_rounds(mock_llm, sample_question, sample_corpus):
    retriever = AsyncMock()
    repeated_doc = sample_corpus[0]
    retriever.retrieve.side_effect = [
        make_retrieval_result(sample_question.text, [repeated_doc, sample_corpus[1]]),
        make_retrieval_result("Paris capital city", [repeated_doc, sample_corpus[2]]),
    ]
    mock_llm.generate.side_effect = [
        ("Paris is the capital city.", 8, 0.001),
        ("[ANSWER] Paris", 8, 0.001),
        ("Paris", 8, 0.001),
    ]

    rag = IRCoTRAG(mock_llm, retriever, {"max_steps": 2, "top_k": 2, "max_docs": 10})
    response = await rag.answer(sample_question, sample_corpus)

    synthesis_prompt = mock_llm.generate.call_args_list[-1].args[0][-1]["content"]
    assert synthesis_prompt.count("[1] France") == 1
    assert len(response.retrieved_docs) == 2


def test_extract_answer_from_trigger():
    assert IRCoTRAG._extract_answer("[ANSWER] Anne Hidalgo") == "Anne Hidalgo"


def test_extract_answer_from_fallback_phrase():
    assert IRCoTRAG._extract_answer("Therefore, the answer is: Paris") == "Paris"


def test_extract_query_from_reasoning_sentence():
    query = IRCoTRAG._extract_query("Therefore, France's capital is Paris.")
    assert query == "France capital Paris"


def test_prompt_file_mentions_answer_trigger():
    prompt = Path("prompts/ircot.txt").read_text()
    assert "[ANSWER]" in prompt
    assert "one next reasoning sentence" in prompt
