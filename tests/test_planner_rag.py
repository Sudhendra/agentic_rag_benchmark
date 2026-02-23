"""Unit tests for Planner RAG architecture."""

from unittest.mock import AsyncMock

import pytest

from src.architectures.agentic.planner_rag import PlannerRAG
from src.core.types import ArchitectureType, Document, Question, QuestionType, RetrievalResult


def make_prompt_router(
    routes: list[tuple[str, list[tuple[str, int, float]]]],
    default_response: tuple[str, int, float],
):
    counters = {pattern: 0 for pattern, _ in routes}

    async def _generate(messages, *args, **kwargs):
        prompt = messages[-1]["content"]
        for pattern, responses in routes:
            if pattern not in prompt:
                continue
            index = counters[pattern]
            if index < len(responses):
                counters[pattern] = index + 1
                return responses[index]
            return responses[-1]
        return default_response

    return _generate


@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    llm.model = "test-model"
    return llm


@pytest.fixture
def mock_retriever():
    retriever = AsyncMock()
    retriever.retrieve.return_value = RetrievalResult(
        documents=[
            Document(id="d1", title="Doc 1", text="Paris is the capital of France."),
            Document(id="d2", title="Doc 2", text="France is in Europe."),
        ],
        scores=[0.9, 0.8],
        query="capital of france",
        retrieval_time_ms=10.0,
        method="bm25",
    )
    return retriever


@pytest.fixture
def sample_question():
    return Question(
        id="q1",
        text="What is the capital of France?",
        type=QuestionType.BRIDGE,
        gold_answer="Paris",
    )


@pytest.fixture
def sample_corpus():
    return [
        Document(id="1", title="A", text="Paris is the capital of France."),
        Document(id="2", title="B", text="Berlin is the capital of Germany."),
    ]


@pytest.mark.asyncio
async def test_happy_path_tree_planning(mock_llm, mock_retriever, sample_question, sample_corpus):
    mock_llm.generate.side_effect = make_prompt_router(
        routes=[
            (
                "Decide if the question needs recursive retrieval planning.",
                [('{"direct_answer": false, "reason": "multi-hop"}', 5, 0.001)],
            ),
            (
                "Choose exactly one next action from:",
                [
                    ('{"action": "ROLLOUT", "node_id": "root"}', 8, 0.001),
                    ('{"action": "SELECT", "node_id": "root.1"}', 7, 0.001),
                    ('{"action": "STOP"}', 6, 0.001),
                ],
            ),
            (
                "Expand the current node into useful sub-questions.",
                [
                    (
                        '{"sub_questions": ["Which city is the capital of France?", '
                        '"Where is France located?"]}',
                        12,
                        0.001,
                    )
                ],
            ),
            ("Answer the node question using only the provided context.", [("Paris", 15, 0.002)]),
            (
                "Estimate confidence that the node answer is correct using the provided context.",
                [("0.91", 4, 0.001), ("0.88", 4, 0.001)],
            ),
            (
                "Synthesize the final answer to the root question from solved node summaries.",
                [("Paris", 10, 0.001)],
            ),
        ],
        default_response=("Paris", 5, 0.001),
    )

    rag = PlannerRAG(mock_llm, mock_retriever, {"max_iterations": 3})
    response = await rag.answer(sample_question, sample_corpus)

    assert response.answer == "Paris"
    assert response.num_retrieval_calls >= 1
    actions = [step.action for step in response.reasoning_chain]
    assert "ROLLOUT" in actions
    assert "SELECT" in actions
    assert "synthesize" in actions


@pytest.mark.asyncio
async def test_direct_answer_path(mock_llm, mock_retriever, sample_question, sample_corpus):
    question = Question(
        id="q-direct",
        text="What is the capital of France?",
        type=QuestionType.SINGLE_HOP,
        gold_answer="Paris",
    )
    mock_llm.generate.side_effect = [
        ('{"direct_answer": true, "reason": "simple"}', 5, 0.001),
        ("Paris", 8, 0.001),
    ]

    rag = PlannerRAG(mock_llm, mock_retriever, {})
    response = await rag.answer(question, sample_corpus)

    assert response.answer == "Paris"
    assert response.num_retrieval_calls == 0
    assert response.num_llm_calls == 2


@pytest.mark.asyncio
async def test_bridge_questions_force_recursive_planning(mock_llm, mock_retriever, sample_corpus):
    question = Question(
        id="q-bridge-gate",
        text="Where was the Chief of Protocol born?",
        type=QuestionType.BRIDGE,
        gold_answer="Greenwich Village, New York City",
    )

    mock_llm.generate.side_effect = [
        ('{"action": "SELECT", "node_id": "root"}', 4, 0.001),
        ('{"answer": "Greenwich Village, New York City", "confidence": 0.95}', 8, 0.001),
        ("Greenwich Village, New York City", 6, 0.001),
    ]

    rag = PlannerRAG(mock_llm, mock_retriever, {"max_iterations": 1})
    response = await rag.answer(question, sample_corpus)

    assert response.num_retrieval_calls == 1
    assert response.reasoning_chain[0].action == "gate"
    assert "forced_recursive_for_question_type" in response.reasoning_chain[0].observation


@pytest.mark.asyncio
async def test_rollout_respects_branching_bound(
    mock_llm, mock_retriever, sample_question, sample_corpus
):
    mock_llm.generate.side_effect = [
        ('{"action": "ROLLOUT", "node_id": "root"}', 4, 0.001),
        (
            '{"sub_questions": ["q1", "q2", "q3"]}',
            7,
            0.001,
        ),
        ('{"action": "STOP"}', 4, 0.001),
        ("fallback", 5, 0.001),
        ("0.4", 3, 0.001),
        ("fallback", 5, 0.001),
    ]

    rag = PlannerRAG(
        mock_llm,
        mock_retriever,
        {"max_branching_factor": 2, "max_iterations": 2},
    )
    response = await rag.answer(sample_question, sample_corpus)

    rollout_steps = [step for step in response.reasoning_chain if step.action == "ROLLOUT"]
    assert rollout_steps
    assert "created_children=2" in rollout_steps[0].observation


@pytest.mark.asyncio
async def test_parser_robustness_with_malformed_action(
    mock_llm, mock_retriever, sample_question, sample_corpus
):
    mock_llm.generate.side_effect = make_prompt_router(
        routes=[
            (
                "Decide if the question needs recursive retrieval planning.",
                [('{"direct_answer": false}', 4, 0.001)],
            ),
            ("Choose exactly one next action from:", [("not valid json action output", 6, 0.001)]),
            ("Answer the node question using only the provided context.", [("unknown", 6, 0.001)]),
            (
                "Estimate confidence that the node answer is correct using the provided context.",
                [("0.3", 3, 0.001)],
            ),
            (
                "Synthesize the final answer to the root question from solved node summaries.",
                [("final", 5, 0.001)],
            ),
        ],
        default_response=("final", 5, 0.001),
    )

    rag = PlannerRAG(mock_llm, mock_retriever, {"max_iterations": 1})
    response = await rag.answer(sample_question, sample_corpus)

    assert response.answer
    assert response.num_llm_calls >= 3


@pytest.mark.asyncio
async def test_backtrack_behavior(mock_llm, mock_retriever, sample_question, sample_corpus):
    mock_llm.generate.side_effect = [
        ('{"action": "ROLLOUT", "node_id": "root"}', 4, 0.001),
        ('{"sub_questions": ["sq1"]}', 5, 0.001),
        ('{"action": "BACKTRACK", "node_id": "root.1"}', 4, 0.001),
        ('{"action": "STOP"}', 4, 0.001),
        ("fallback", 5, 0.001),
        ("0.4", 3, 0.001),
        ("fallback", 5, 0.001),
    ]

    rag = PlannerRAG(mock_llm, mock_retriever, {"max_iterations": 2})
    response = await rag.answer(sample_question, sample_corpus)

    backtrack_steps = [step for step in response.reasoning_chain if step.action == "BACKTRACK"]
    assert backtrack_steps
    assert "pruned=root.1" in backtrack_steps[0].observation


@pytest.mark.asyncio
async def test_confidence_based_early_stop(
    mock_llm, mock_retriever, sample_question, sample_corpus
):
    mock_llm.generate.side_effect = [
        ('{"action": "SELECT", "node_id": "root"}', 4, 0.001),
        ("Paris", 8, 0.001),
        ("0.95", 3, 0.001),
        ("Paris", 6, 0.001),
    ]

    rag = PlannerRAG(mock_llm, mock_retriever, {"min_stop_confidence": 0.8})
    response = await rag.answer(sample_question, sample_corpus)

    assert response.answer == "Paris"
    assert response.num_retrieval_calls == 1
    assert any(step.action == "STOP" for step in response.reasoning_chain)


@pytest.mark.asyncio
async def test_comparison_questions_force_recursive_planning(
    mock_llm, mock_retriever, sample_corpus
):
    question = Question(
        id="q2",
        text="Are Paris and Berlin in the same country?",
        type=QuestionType.COMPARISON,
        gold_answer="no",
    )
    mock_llm.generate.side_effect = [
        ('{"action": "SELECT", "node_id": "root"}', 4, 0.001),
        ("No.", 8, 0.001),
        ("0.95", 3, 0.001),
        ("No.", 6, 0.001),
    ]

    rag = PlannerRAG(mock_llm, mock_retriever, {"max_iterations": 1})
    response = await rag.answer(question, sample_corpus)

    assert response.num_retrieval_calls == 1
    assert response.answer == "no"
    assert response.reasoning_chain[0].action == "gate"
    assert "forced_recursive_for_question_type" in response.reasoning_chain[0].observation
    assert not any(step.action == "bridge_refine" for step in response.reasoning_chain)


@pytest.mark.asyncio
async def test_comparison_root_stop_is_not_gated_by_confidence(
    mock_llm, mock_retriever, sample_corpus
):
    question = Question(
        id="q2b",
        text="Are Paris and Berlin in the same country?",
        type=QuestionType.COMPARISON,
        gold_answer="no",
    )
    mock_llm.generate.side_effect = [
        ('{"action": "SELECT", "node_id": "root"}', 4, 0.001),
        ("No.", 8, 0.001),
        ("0.05", 3, 0.001),
        ("No.", 6, 0.001),
    ]

    rag = PlannerRAG(mock_llm, mock_retriever, {"max_iterations": 2, "min_stop_confidence": 0.99})
    response = await rag.answer(question, sample_corpus)

    assert response.answer == "no"
    assert response.num_retrieval_calls == 1
    assert any(
        step.action == "STOP" and "comparison_root_solved_stop" in step.observation
        for step in response.reasoning_chain
    )


@pytest.mark.asyncio
async def test_stop_is_overridden_when_tree_not_resolved(
    mock_llm, mock_retriever, sample_question, sample_corpus
):
    mock_llm.generate.side_effect = make_prompt_router(
        routes=[
            (
                "Decide if the question needs recursive retrieval planning.",
                [('{"direct_answer": false}', 4, 0.001)],
            ),
            ("Choose exactly one next action from:", [('{"action": "STOP"}', 4, 0.001)]),
            ("Answer the node question using only the provided context.", [("Paris", 8, 0.001)]),
            (
                "Estimate confidence that the node answer is correct using the provided context.",
                [("0.4", 3, 0.001)],
            ),
            (
                "Synthesize the final answer to the root question from solved node summaries.",
                [("Paris", 6, 0.001)],
            ),
        ],
        default_response=("Paris", 6, 0.001),
    )

    rag = PlannerRAG(mock_llm, mock_retriever, {"max_iterations": 1})
    response = await rag.answer(sample_question, sample_corpus)

    observations = [step.observation for step in response.reasoning_chain]
    assert any(
        "stop_overridden_to_select_due_to_unresolved_nodes" in obs
        or "bridge_guardrail_forced_rollout_root" in obs
        for obs in observations
    )


@pytest.mark.asyncio
async def test_forces_one_retrieval_before_synthesis(
    mock_llm, mock_retriever, sample_question, sample_corpus
):
    mock_llm.generate.side_effect = make_prompt_router(
        routes=[
            (
                "Decide if the question needs recursive retrieval planning.",
                [('{"direct_answer": false}', 4, 0.001)],
            ),
            (
                "Choose exactly one next action from:",
                [('{"action": "TRAVERSE", "node_id": "root"}', 4, 0.001)],
            ),
            (
                "Expand the current node into useful sub-questions.",
                [('{"sub_questions": ["sq1"]}', 5, 0.001)],
            ),
            (
                "Answer the node question using only the provided context.",
                [("root answer", 7, 0.001)],
            ),
            (
                "Estimate confidence that the node answer is correct using the provided context.",
                [("0.81", 3, 0.001)],
            ),
            (
                "Synthesize the final answer to the root question from solved node summaries.",
                [("root answer", 6, 0.001)],
            ),
        ],
        default_response=("root answer", 6, 0.001),
    )

    rag = PlannerRAG(mock_llm, mock_retriever, {"max_iterations": 1})
    response = await rag.answer(sample_question, sample_corpus)

    assert response.num_retrieval_calls >= 1
    assert any("forced_select_node=" in step.observation for step in response.reasoning_chain)


@pytest.mark.asyncio
async def test_traverse_to_open_child_triggers_retrieval_and_solve(
    mock_llm, mock_retriever, sample_corpus
):
    question = Question(
        id="q-traverse-child",
        text="What city is the capital of France?",
        type=QuestionType.SINGLE_HOP,
        gold_answer="Paris",
    )
    mock_llm.generate.side_effect = [
        ('{"action": "ROLLOUT", "node_id": "root"}', 4, 0.001),
        ('{"sub_questions": ["Which city is the capital of France?"]}', 6, 0.001),
        ('{"action": "TRAVERSE", "node_id": "root.1"}', 4, 0.001),
        ("Paris", 8, 0.001),
        ("0.93", 3, 0.001),
        ('{"action": "STOP"}', 4, 0.001),
        ("Paris", 6, 0.001),
    ]

    rag = PlannerRAG(
        mock_llm,
        mock_retriever,
        {"max_iterations": 3, "allow_direct_answer": False},
    )
    response = await rag.answer(question, sample_corpus)

    assert response.num_retrieval_calls == 1
    assert mock_retriever.retrieve.await_count == 1
    retrieval_query = mock_retriever.retrieve.await_args_list[0].kwargs["query"]
    assert "Which city is the capital of France?" in retrieval_query
    assert any(step.action == "TRAVERSE" for step in response.reasoning_chain)


@pytest.mark.asyncio
async def test_traverse_noop_is_still_overridden_to_select(mock_llm, mock_retriever, sample_corpus):
    question = Question(
        id="q-traverse-noop",
        text="What is the capital of France?",
        type=QuestionType.SINGLE_HOP,
        gold_answer="Paris",
    )
    mock_llm.generate.side_effect = [
        ('{"action": "TRAVERSE", "node_id": "root"}', 4, 0.001),
        ("Paris", 8, 0.001),
        ("0.95", 3, 0.001),
        ("Paris", 6, 0.001),
    ]

    rag = PlannerRAG(
        mock_llm,
        mock_retriever,
        {"max_iterations": 1, "allow_direct_answer": False},
    )
    response = await rag.answer(question, sample_corpus)

    assert response.num_retrieval_calls == 1
    assert mock_retriever.retrieve.await_count == 1
    observation = " ".join(step.observation for step in response.reasoning_chain)
    assert "traverse_noop_overridden_to_select" in observation


@pytest.mark.asyncio
async def test_bridge_generic_output_recovery_with_refine(
    mock_llm, mock_retriever, sample_question, sample_corpus
):
    mock_llm.generate.side_effect = make_prompt_router(
        routes=[
            (
                "Decide if the question needs recursive retrieval planning.",
                [('{"direct_answer": false}', 4, 0.001)],
            ),
            (
                "Choose exactly one next action from:",
                [
                    ('{"action": "ROLLOUT", "node_id": "root"}', 4, 0.001),
                    ('{"action": "SELECT", "node_id": "root.1"}', 4, 0.001),
                    ('{"action": "STOP"}', 4, 0.001),
                ],
            ),
            (
                "Expand the current node into useful sub-questions.",
                [('{"sub_questions": ["sq1"]}', 5, 0.001)],
            ),
            (
                "Answer the node question using only the provided context.",
                [("Chief of Protocol", 7, 0.001)],
            ),
            (
                "Estimate confidence that the node answer is correct using the provided context.",
                [("0.95", 3, 0.001)],
            ),
            (
                "Synthesize the final answer to the root question from solved node summaries.",
                [("no", 6, 0.001)],
            ),
            ("Refine the bridge/compositional final answer", [("Chief of Protocol", 6, 0.001)]),
        ],
        default_response=("Chief of Protocol", 6, 0.001),
    )

    rag = PlannerRAG(mock_llm, mock_retriever, {"max_iterations": 3})
    response = await rag.answer(sample_question, sample_corpus)

    assert response.answer == "Chief of Protocol"
    assert response.num_retrieval_calls == 1
    assert any(step.action == "bridge_refine" for step in response.reasoning_chain)


@pytest.mark.asyncio
async def test_bridge_invalid_refine_then_deterministic_fallback(
    mock_llm, mock_retriever, sample_question, sample_corpus
):
    mock_llm.generate.side_effect = make_prompt_router(
        routes=[
            (
                "Decide if the question needs recursive retrieval planning.",
                [('{"direct_answer": false}', 4, 0.001)],
            ),
            (
                "Choose exactly one next action from:",
                [
                    ('{"action": "ROLLOUT", "node_id": "root"}', 4, 0.001),
                    ('{"action": "SELECT", "node_id": "root.1"}', 4, 0.001),
                    ('{"action": "STOP"}', 4, 0.001),
                ],
            ),
            (
                "Expand the current node into useful sub-questions.",
                [('{"sub_questions": ["sq1"]}', 5, 0.001)],
            ),
            (
                "Answer the node question using only the provided context.",
                [("Chief of Protocol", 7, 0.001)],
            ),
            (
                "Estimate confidence that the node answer is correct using the provided context.",
                [("0.95", 3, 0.001)],
            ),
            (
                "Synthesize the final answer to the root question from solved node summaries.",
                [("no", 6, 0.001)],
            ),
            ("Refine the bridge/compositional final answer", [("unknown", 6, 0.001)]),
        ],
        default_response=("no", 6, 0.001),
    )

    rag = PlannerRAG(mock_llm, mock_retriever, {"max_iterations": 3})
    response = await rag.answer(sample_question, sample_corpus)

    assert response.answer == "Chief of Protocol"
    assert any(
        "fallback_answer=Chief of Protocol" in step.observation for step in response.reasoning_chain
    )


@pytest.mark.asyncio
async def test_bridge_partial_span_expansion_fallback(
    mock_llm, mock_retriever, sample_question, sample_corpus
):
    mock_llm.generate.side_effect = make_prompt_router(
        routes=[
            (
                "Decide if the question needs recursive retrieval planning.",
                [('{"direct_answer": false}', 4, 0.001)],
            ),
            (
                "Choose exactly one next action from:",
                [
                    ('{"action": "ROLLOUT", "node_id": "root"}', 4, 0.001),
                    ('{"action": "SELECT", "node_id": "root.1"}', 4, 0.001),
                    ('{"action": "STOP"}', 4, 0.001),
                ],
            ),
            (
                "Expand the current node into useful sub-questions.",
                [('{"sub_questions": ["sq1"]}', 5, 0.001)],
            ),
            (
                "Answer the node question using only the provided context.",
                [("Greenwich Village, New York City", 7, 0.001)],
            ),
            (
                "Estimate confidence that the node answer is correct using the provided context.",
                [("0.95", 3, 0.001)],
            ),
            (
                "Synthesize the final answer to the root question from solved node summaries.",
                [("New York City", 6, 0.001)],
            ),
            ("Refine the bridge/compositional final answer", [("New York City", 6, 0.001)]),
        ],
        default_response=("New York City", 6, 0.001),
    )

    rag = PlannerRAG(mock_llm, mock_retriever, {"max_iterations": 3})
    response = await rag.answer(sample_question, sample_corpus)

    assert response.answer == "Greenwich Village, New York City"


@pytest.mark.asyncio
async def test_bridge_refine_budget_bound_one_extra_call(
    mock_llm, mock_retriever, sample_question, sample_corpus
):
    mock_llm.generate.side_effect = make_prompt_router(
        routes=[
            (
                "Decide if the question needs recursive retrieval planning.",
                [('{"direct_answer": false}', 4, 0.001)],
            ),
            (
                "Choose exactly one next action from:",
                [
                    ('{"action": "ROLLOUT", "node_id": "root"}', 4, 0.001),
                    ('{"action": "SELECT", "node_id": "root.1"}', 4, 0.001),
                    ('{"action": "STOP"}', 4, 0.001),
                ],
            ),
            (
                "Expand the current node into useful sub-questions.",
                [('{"sub_questions": ["sq1"]}', 5, 0.001)],
            ),
            (
                "Answer the node question using only the provided context.",
                [("Chief of Protocol", 7, 0.001)],
            ),
            (
                "Estimate confidence that the node answer is correct using the provided context.",
                [("0.95", 3, 0.001)],
            ),
            (
                "Synthesize the final answer to the root question from solved node summaries.",
                [("no", 6, 0.001)],
            ),
            ("Refine the bridge/compositional final answer", [("Chief of Protocol", 6, 0.001)]),
        ],
        default_response=("Chief of Protocol", 6, 0.001),
    )

    rag = PlannerRAG(
        mock_llm, mock_retriever, {"max_iterations": 3, "bridge_refine_max_attempts": 1}
    )
    response = await rag.answer(sample_question, sample_corpus)

    prompts = [
        await_call.args[0][-1]["content"] for await_call in mock_llm.generate.await_args_list
    ]
    refine_calls = [
        prompt for prompt in prompts if "Refine the bridge/compositional final answer" in prompt
    ]

    assert len(refine_calls) == 1
    assert response.num_retrieval_calls == 1


def test_get_name_and_type(mock_llm, mock_retriever):
    rag = PlannerRAG(mock_llm, mock_retriever, {})
    assert rag.get_name() == "planner_rag"
    assert rag.get_type() == ArchitectureType.AGENTIC
