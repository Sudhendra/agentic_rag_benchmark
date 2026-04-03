import pytest

from src.core.types import ArchitectureType, Document, Question, QuestionType, RAGResponse
from src.evaluation.evaluator import Evaluator


class DummyRAG:
    def __init__(self) -> None:
        self.calls = 0

    async def answer(self, question: Question, corpus: list[Document]) -> RAGResponse:
        self.calls += 1
        return RAGResponse(
            answer=question.gold_answer or "",
            reasoning_chain=[],
            retrieved_docs=[],
            total_tokens=10,
            total_cost_usd=0.01,
            latency_ms=5.0,
            num_retrieval_calls=1,
            num_llm_calls=1,
            model="test-model",
            architecture="vanilla_rag",
        )

    def get_name(self) -> str:
        return "vanilla_rag"

    def get_type(self) -> ArchitectureType:
        return ArchitectureType.VANILLA


@pytest.fixture
def mock_rag() -> DummyRAG:
    return DummyRAG()


@pytest.fixture
def questions() -> list[Question]:
    return [
        Question(id="q1", text="Question 1?", type=QuestionType.SINGLE_HOP, gold_answer="A"),
        Question(id="q2", text="Question 2?", type=QuestionType.BRIDGE, gold_answer="B"),
    ]


@pytest.fixture
def corpus() -> list[Document]:
    return [Document(id="d1", title="Doc", text="Doc text.")]


@pytest.mark.asyncio
async def test_evaluator_aggregates_metrics(
    mock_rag: DummyRAG, questions: list[Question], corpus: list[Document]
) -> None:
    evaluator = Evaluator(mock_rag, max_concurrency=2, dataset_name="test")

    result = await evaluator.evaluate(questions, corpus)

    assert result.num_questions == len(questions)
    assert 0.0 <= result.avg_exact_match <= 1.0
    assert 0.0 <= result.avg_f1 <= 1.0
    assert result.total_cost_usd >= 0.0


class RecordingRAG(DummyRAG):
    def __init__(self) -> None:
        super().__init__()
        self.seen_corpora: dict[str, list[str]] = {}

    async def answer(self, question: Question, corpus: list[Document]) -> RAGResponse:
        self.seen_corpora[question.id] = [doc.id for doc in corpus]
        return await super().answer(question, corpus)


@pytest.mark.asyncio
async def test_evaluator_uses_question_scoped_corpus_when_present() -> None:
    rag = RecordingRAG()
    question_a_corpus = [Document(id="q1_doc", title="Shared", text="Alpha evidence")]
    question_b_corpus = [Document(id="q2_doc", title="Shared", text="Beta distractor")]
    questions = [
        Question(
            id="q1",
            text="Question A?",
            type=QuestionType.BRIDGE,
            gold_answer="A",
            candidate_corpus=question_a_corpus,
        ),
        Question(
            id="q2",
            text="Question B?",
            type=QuestionType.BRIDGE,
            gold_answer="B",
            candidate_corpus=question_b_corpus,
        ),
    ]
    merged_corpus = question_a_corpus + question_b_corpus

    evaluator = Evaluator(rag, max_concurrency=2, dataset_name="test")
    await evaluator.evaluate(questions, merged_corpus)

    assert rag.seen_corpora == {
        "q1": ["q1_doc"],
        "q2": ["q2_doc"],
    }
