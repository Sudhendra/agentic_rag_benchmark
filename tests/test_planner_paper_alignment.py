"""Paper-alignment tests for Planner RAG behavior gaps."""

from unittest.mock import AsyncMock

import pytest

from src.architectures.agentic.planner_rag import PlannerRAG
from src.core.types import Document, Question, QuestionType, RetrievalResult


def make_prompt_router(routes: list[tuple[str, list[tuple[str, int, float]]]]):
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
        known_patterns = ", ".join(pattern for pattern, _ in routes)
        raise AssertionError(
            "Unmatched LLM prompt in test router. "
            f"Known patterns: [{known_patterns}] | Prompt: {prompt[:200]}"
        )

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
            Document(id="d1", title="Doc 1", text="Chief of Protocol was from Greenwich Village."),
            Document(id="d2", title="Doc 2", text="Greenwich Village is in New York City."),
        ],
        scores=[0.9, 0.8],
        query="test query",
        retrieval_time_ms=10.0,
        method="bm25",
    )
    return retriever


@pytest.fixture
def bridge_question():
    return Question(
        id="q-bridge",
        text="Where was the Chief of Protocol born?",
        type=QuestionType.BRIDGE,
        gold_answer="Greenwich Village, New York City",
    )


@pytest.fixture
def single_hop_question():
    return Question(
        id="q-single",
        text="What is the capital of France?",
        type=QuestionType.SINGLE_HOP,
        gold_answer="Paris",
    )


@pytest.fixture
def sample_corpus():
    return [
        Document(id="1", title="A", text="Paris is the capital of France."),
        Document(id="2", title="B", text="Greenwich Village is in New York City."),
    ]


@pytest.mark.asyncio
async def test_bridge_refine_respects_max_attempts_calls_twice_when_needed(
    mock_llm, mock_retriever, bridge_question, sample_corpus
):
    mock_llm.generate.side_effect = make_prompt_router(
        routes=[
            (
                "Decide if the question needs recursive retrieval planning.",
                [('{"direct_answer": false}', 4, 0.001)],
            ),
            (
                "Choose exactly one next action from:",
                [('{"action": "SELECT", "node_id": "root"}', 4, 0.001)],
            ),
            (
                "Answer the node question using only the provided context.",
                [("Chief of Protocol", 8, 0.001)],
            ),
            (
                "Estimate confidence that the node answer is correct using the provided context.",
                [("0.30", 3, 0.001)],
            ),
            (
                "Synthesize the final answer to the root question from solved node summaries.",
                [("no", 6, 0.001)],
            ),
            (
                "Refine the bridge/compositional final answer",
                [("unknown", 6, 0.001), ("Greenwich Village, New York City", 6, 0.001)],
            ),
        ]
    )

    rag = PlannerRAG(
        mock_llm,
        mock_retriever,
        {"max_iterations": 1, "bridge_refine_max_attempts": 2},
    )
    response = await rag.answer(bridge_question, sample_corpus)

    prompts = [call.args[0][-1]["content"] for call in mock_llm.generate.await_args_list]
    refine_calls = [
        prompt for prompt in prompts if "Refine the bridge/compositional final answer" in prompt
    ]

    actions = [step.action for step in response.reasoning_chain]

    assert response.answer == "Greenwich Village, New York City"
    assert actions.count("bridge_refine") >= 1
    assert len(refine_calls) == 2


@pytest.mark.asyncio
async def test_bridge_refine_disabled_skips_refine_call(
    mock_llm, mock_retriever, bridge_question, sample_corpus
):
    mock_llm.generate.side_effect = make_prompt_router(
        routes=[
            (
                "Decide if the question needs recursive retrieval planning.",
                [('{"direct_answer": false}', 4, 0.001)],
            ),
            (
                "Choose exactly one next action from:",
                [('{"action": "SELECT", "node_id": "root"}', 4, 0.001)],
            ),
            (
                "Answer the node question using only the provided context.",
                [("Chief of Protocol", 8, 0.001)],
            ),
            (
                "Estimate confidence that the node answer is correct using the provided context.",
                [("0.30", 3, 0.001)],
            ),
            (
                "Synthesize the final answer to the root question from solved node summaries.",
                [("Chief of Protocol", 6, 0.001)],
            ),
        ]
    )

    rag = PlannerRAG(
        mock_llm,
        mock_retriever,
        {"max_iterations": 1, "bridge_refine_enabled": False, "bridge_refine_max_attempts": 2},
    )
    response = await rag.answer(bridge_question, sample_corpus)

    prompts = [call.args[0][-1]["content"] for call in mock_llm.generate.await_args_list]
    refine_calls = [
        prompt for prompt in prompts if "Refine the bridge/compositional final answer" in prompt
    ]

    actions = [step.action for step in response.reasoning_chain]

    assert response.answer == "Chief of Protocol"
    assert len(refine_calls) == 0
    assert "bridge_refine" not in actions


@pytest.mark.asyncio
async def test_single_call_solver_json_path_reduces_llm_calls(
    mock_llm, mock_retriever, single_hop_question, sample_corpus
):
    mock_llm.generate.side_effect = make_prompt_router(
        routes=[
            (
                "Choose exactly one next action from:",
                [('{"action": "SELECT", "node_id": "root"}', 4, 0.001)],
            ),
            (
                "Answer the node question using only the provided context.",
                [('{"answer": "Paris", "confidence": 0.93}', 9, 0.001)],
            ),
            (
                "Synthesize the final answer to the root question from solved node summaries.",
                [("Paris", 6, 0.001)],
            ),
        ]
    )

    rag = PlannerRAG(
        mock_llm,
        mock_retriever,
        {"allow_direct_answer": False, "max_iterations": 1},
    )
    response = await rag.answer(single_hop_question, sample_corpus)

    prompts = [call.args[0][-1]["content"] for call in mock_llm.generate.await_args_list]
    confidence_calls = [
        prompt
        for prompt in prompts
        if "Estimate confidence that the node answer is correct using the provided context."
        in prompt
    ]

    assert response.answer == "Paris"
    assert response.num_llm_calls == 3
    assert len(confidence_calls) == 0


@pytest.mark.asyncio
async def test_solver_combined_json_fallbacks_when_confidence_value_out_of_range(
    mock_llm, mock_retriever, single_hop_question, sample_corpus
):
    mock_llm.generate.side_effect = make_prompt_router(
        routes=[
            (
                "Choose exactly one next action from:",
                [('{"action": "SELECT", "node_id": "root"}', 4, 0.001)],
            ),
            (
                "Answer the node question using only the provided context.",
                [('{"answer": "Paris", "confidence": 1.7}', 9, 0.001)],
            ),
            (
                "Estimate confidence that the node answer is correct using the provided context.",
                [("0.88", 3, 0.001)],
            ),
            (
                "Synthesize the final answer to the root question from solved node summaries.",
                [("Paris", 6, 0.001)],
            ),
        ]
    )

    rag = PlannerRAG(
        mock_llm,
        mock_retriever,
        {"allow_direct_answer": False, "max_iterations": 1},
    )
    response = await rag.answer(single_hop_question, sample_corpus)

    prompts = [call.args[0][-1]["content"] for call in mock_llm.generate.await_args_list]
    confidence_calls = [
        prompt
        for prompt in prompts
        if "Estimate confidence that the node answer is correct using the provided context."
        in prompt
    ]

    assert response.answer == "Paris"
    assert response.num_llm_calls == 4
    assert len(confidence_calls) == 1


@pytest.mark.asyncio
async def test_solver_combined_json_fallbacks_to_confidence_prompt_when_invalid(
    mock_llm, mock_retriever, single_hop_question, sample_corpus
):
    mock_llm.generate.side_effect = make_prompt_router(
        routes=[
            (
                "Choose exactly one next action from:",
                [('{"action": "SELECT", "node_id": "root"}', 4, 0.001)],
            ),
            (
                "Answer the node question using only the provided context.",
                [('{"answer": "Paris"}', 9, 0.001)],
            ),
            (
                "Estimate confidence that the node answer is correct using the provided context.",
                [("0.72", 3, 0.001)],
            ),
            (
                "Synthesize the final answer to the root question from solved node summaries.",
                [("Paris", 6, 0.001)],
            ),
        ]
    )

    rag = PlannerRAG(
        mock_llm,
        mock_retriever,
        {"allow_direct_answer": False, "max_iterations": 1},
    )
    response = await rag.answer(single_hop_question, sample_corpus)

    prompts = [call.args[0][-1]["content"] for call in mock_llm.generate.await_args_list]
    confidence_calls = [
        prompt
        for prompt in prompts
        if "Estimate confidence that the node answer is correct using the provided context."
        in prompt
    ]

    assert response.answer == "Paris"
    assert response.num_llm_calls == 4
    assert len(confidence_calls) == 1


@pytest.mark.asyncio
async def test_solver_combined_json_fallbacks_when_confidence_value_not_numeric(
    mock_llm, mock_retriever, single_hop_question, sample_corpus
):
    mock_llm.generate.side_effect = make_prompt_router(
        routes=[
            (
                "Choose exactly one next action from:",
                [('{"action": "SELECT", "node_id": "root"}', 4, 0.001)],
            ),
            (
                "Answer the node question using only the provided context.",
                [('{"answer": "Paris", "confidence": "high"}', 9, 0.001)],
            ),
            (
                "Estimate confidence that the node answer is correct using the provided context.",
                [("0.72", 3, 0.001)],
            ),
            (
                "Synthesize the final answer to the root question from solved node summaries.",
                [("Paris", 6, 0.001)],
            ),
        ]
    )

    rag = PlannerRAG(
        mock_llm,
        mock_retriever,
        {"allow_direct_answer": False, "max_iterations": 1},
    )
    response = await rag.answer(single_hop_question, sample_corpus)

    prompts = [call.args[0][-1]["content"] for call in mock_llm.generate.await_args_list]
    confidence_calls = [
        prompt
        for prompt in prompts
        if "Estimate confidence that the node answer is correct using the provided context."
        in prompt
    ]

    assert response.answer == "Paris"
    assert response.num_llm_calls == 4
    assert len(confidence_calls) == 1


@pytest.mark.asyncio
async def test_repeated_select_same_node_reuses_cached_solve(
    mock_llm, mock_retriever, single_hop_question, sample_corpus
):
    mock_llm.generate.side_effect = make_prompt_router(
        routes=[
            (
                "Choose exactly one next action from:",
                [
                    ('{"action": "SELECT", "node_id": "root"}', 4, 0.001),
                    ('{"action": "SELECT", "node_id": "root"}', 4, 0.001),
                ],
            ),
            (
                "Answer the node question using only the provided context.",
                [("Paris", 8, 0.001), ("Paris", 8, 0.001)],
            ),
            (
                "Estimate confidence that the node answer is correct using the provided context.",
                [("0.20", 3, 0.001), ("0.20", 3, 0.001)],
            ),
            (
                "Synthesize the final answer to the root question from solved node summaries.",
                [("Paris", 6, 0.001)],
            ),
        ]
    )

    rag = PlannerRAG(
        mock_llm,
        mock_retriever,
        {"allow_direct_answer": False, "max_iterations": 2, "min_stop_confidence": 0.9},
    )
    response = await rag.answer(single_hop_question, sample_corpus)

    select_steps = [step for step in response.reasoning_chain if step.action == "SELECT"]
    select_root_steps = [step for step in select_steps if '"node_id": "root"' in step.action_input]

    assert len(select_root_steps) == 2
    assert mock_retriever.retrieve.await_count == 1
    assert response.num_retrieval_calls == 1


@pytest.mark.asyncio
async def test_identical_query_uses_retrieval_cache(
    mock_llm, mock_retriever, single_hop_question, sample_corpus
):
    mock_llm.generate.side_effect = make_prompt_router(
        routes=[
            (
                "Choose exactly one next action from:",
                [
                    ('{"action": "ROLLOUT", "node_id": "root"}', 4, 0.001),
                    ('{"action": "SELECT", "node_id": "root.1"}', 4, 0.001),
                    ('{"action": "SELECT", "node_id": "root.2"}', 4, 0.001),
                ],
            ),
            (
                "Expand the current node into useful sub-questions.",
                [('{"sub_questions": ["same lookup", "same lookup"]}', 6, 0.001)],
            ),
            (
                "Answer the node question using only the provided context.",
                [('{"answer": "Paris", "confidence": 0.2}', 8, 0.001)],
            ),
            (
                "Synthesize the final answer to the root question from solved node summaries.",
                [("Paris", 6, 0.001)],
            ),
        ]
    )

    rag = PlannerRAG(
        mock_llm,
        mock_retriever,
        {
            "allow_direct_answer": False,
            "max_iterations": 3,
            "min_stop_confidence": 0.95,
        },
    )
    response = await rag.answer(single_hop_question, sample_corpus)

    assert mock_retriever.retrieve.await_count == 1
    assert response.num_retrieval_calls == 1


@pytest.mark.asyncio
async def test_reselect_solved_node_reuses_node_solution(
    mock_llm, mock_retriever, single_hop_question, sample_corpus
):
    mock_llm.generate.side_effect = make_prompt_router(
        routes=[
            (
                "Choose exactly one next action from:",
                [
                    ('{"action": "ROLLOUT", "node_id": "root"}', 4, 0.001),
                    ('{"action": "SELECT", "node_id": "root.1"}', 4, 0.001),
                    ('{"action": "SELECT", "node_id": "root.1"}', 4, 0.001),
                ],
            ),
            (
                "Expand the current node into useful sub-questions.",
                [('{"sub_questions": ["capital lookup"]}', 6, 0.001)],
            ),
            (
                "Answer the node question using only the provided context.",
                [('{"answer": "Paris", "confidence": 0.95}', 8, 0.001)],
            ),
            (
                "Synthesize the final answer to the root question from solved node summaries.",
                [("Paris", 6, 0.001)],
            ),
        ]
    )

    rag = PlannerRAG(
        mock_llm,
        mock_retriever,
        {
            "allow_direct_answer": False,
            "max_iterations": 3,
            "min_stop_confidence": 0.9,
        },
    )
    response = await rag.answer(single_hop_question, sample_corpus)

    prompts = [call.args[0][-1]["content"] for call in mock_llm.generate.await_args_list]
    solve_calls = [
        prompt
        for prompt in prompts
        if "Answer the node question using only the provided context." in prompt
    ]

    assert response.answer == "Paris"
    assert len(solve_calls) == 1
    assert mock_retriever.retrieve.await_count == 1


@pytest.mark.asyncio
async def test_child_solved_counter_counts_only_solved_children(
    mock_llm, mock_retriever, bridge_question, sample_corpus
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
                    ('{"action": "SELECT", "node_id": "root.2"}', 4, 0.001),
                ],
            ),
            (
                "Expand the current node into useful sub-questions.",
                [('{"sub_questions": ["bridge sq1", "bridge sq2"]}', 6, 0.001)],
            ),
            (
                "Answer the node question using only the provided context.",
                [("child-answer-1", 8, 0.001), ("child-answer-2", 8, 0.001)],
            ),
            (
                "Estimate confidence that the node answer is correct using the provided context.",
                [("0.20", 3, 0.001), ("0.20", 3, 0.001)],
            ),
            (
                "Synthesize the final answer to the root question from solved node summaries.",
                [("child-answer-2", 6, 0.001)],
            ),
        ]
    )

    rag = PlannerRAG(
        mock_llm,
        mock_retriever,
        {"allow_direct_answer": False, "max_iterations": 3, "min_stop_confidence": 0.9},
    )
    response = await rag.answer(bridge_question, sample_corpus)

    select_actions = [
        step.action_input
        for step in response.reasoning_chain
        if step.action == "SELECT" and "node_id" in step.action_input
    ]
    assert any('"node_id": "root.2"' in action for action in select_actions)

    queries = [call.kwargs["query"] for call in mock_retriever.retrieve.await_args_list]
    assert any("bridge sq1" in query for query in queries)
    assert any("bridge sq2" in query for query in queries)


def test_rollout_prunes_duplicate_subquestions(mock_llm, mock_retriever):
    rag = PlannerRAG(mock_llm, mock_retriever, {})
    candidates = [
        "Who founded Acme Corp?",
        "who founded acme corp",
        "Who founded Acme Corporation?",
        "When was Acme founded?",
    ]

    pruned = rag._prune_similar_sub_questions(candidates, threshold=0.85)

    assert pruned == ["Who founded Acme Corp?", "When was Acme founded?"]
