"""Unit tests for REAP recursive architecture."""

from unittest.mock import AsyncMock

import pytest

from src.architectures.recursive.reap import REAPRAG
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


def make_retrieval_result(query: str, documents: list[Document]) -> RetrievalResult:
    return RetrievalResult(
        documents=documents,
        scores=[0.9 - (i * 0.1) for i in range(len(documents))],
        query=query,
        retrieval_time_ms=10.0,
        method="bm25",
    )


def test_get_name():
    rag = REAPRAG(AsyncMock(model="test-model"), AsyncMock(), {})
    assert rag.get_name() == "reap_rag"


def test_get_type():
    rag = REAPRAG(AsyncMock(model="test-model"), AsyncMock(), {})
    assert rag.get_type() == ArchitectureType.RECURSIVE


def test_config_defaults():
    rag = REAPRAG(AsyncMock(model="test-model"), AsyncMock(), {})
    assert rag.config["max_iterations"] == 5
    assert rag.config["top_k"] == 3
    assert rag.config["max_active_requirements"] == 2


def test_extract_json_block_parses_llm_output():
    payload = REAPRAG._extract_json_block('analysis... {"next_step": "SYNTHESIZE_ANSWER"}')
    assert payload["next_step"] == "SYNTHESIZE_ANSWER"


def test_extract_json_block_prefers_first_valid_object():
    payload = REAPRAG._extract_json_block(
        'preface {"next_step": "EXECUTE"} trailing {"ignore": true}'
    )
    assert payload["next_step"] == "EXECUTE"


def test_extract_json_block_raises_when_missing_json():
    with pytest.raises(ValueError, match="No JSON object found"):
        REAPRAG._extract_json_block("no structured payload here")


def test_mark_requirements_resolved_only_for_direct_answers():
    plan = [{"requirement_id": "r1", "question": "Who...", "status": "pending"}]
    facts = [{"fulfills_requirement_id": "r1", "fulfillment_level": "DIRECT_ANSWER"}]
    resolved = REAPRAG._mark_resolved_requirements(plan, facts)
    assert resolved[0]["status"] == "resolved"


def test_mark_requirements_resolved_keeps_original_plan_unchanged():
    plan = [{"requirement_id": "r1", "question": "Who...", "status": "pending"}]
    facts = [{"fulfills_requirement_id": "r1", "fulfillment_level": "DIRECT_ANSWER"}]
    resolved = REAPRAG._mark_resolved_requirements(plan, facts)
    assert plan[0]["status"] == "pending"
    assert resolved[0]["status"] == "resolved"


def test_mark_requirements_resolved_ignores_partial_clues():
    plan = [{"requirement_id": "r1", "question": "Who...", "status": "pending"}]
    facts = [{"fulfills_requirement_id": "r1", "fulfillment_level": "PARTIAL_CLUE"}]
    resolved = REAPRAG._mark_resolved_requirements(plan, facts)
    assert resolved[0]["status"] == "pending"


def test_parse_reasoned_facts_requires_fulfillment_levels():
    facts = REAPRAG._parse_reasoned_facts(
        {
            "reasoned_facts": [
                {
                    "reasoning": "Paris is directly stated.",
                    "direct_evidence": "Paris is the capital of France.",
                    "statement": "The capital of France is Paris.",
                    "fulfills_requirement_id": "r1",
                    "fulfillment_level": "PARTIAL_CLUE",
                }
            ]
        }
    )
    assert facts[0]["fulfillment_level"] == "PARTIAL_CLUE"


def test_normalize_plan_treats_none_dependencies_as_empty_list():
    normalized = REAPRAG._normalize_plan(
        [
            {
                "requirement_id": "r1",
                "question": "What is the capital of France?",
                "depends_on": None,
                "status": "pending",
            }
        ]
    )
    assert normalized[0]["depends_on"] == []


def test_can_synthesize_requires_all_terminal_requirements_to_have_direct_answers():
    plan = [
        {"requirement_id": "r1", "question": "What is the capital of France?", "depends_on": []},
        {"requirement_id": "r2", "question": "Who is the mayor of Paris?", "depends_on": []},
    ]
    facts = [
        {
            "fulfills_requirement_id": "r1",
            "statement": "The capital of France is Paris.",
            "fulfillment_level": "DIRECT_ANSWER",
        }
    ]
    assert REAPRAG._can_synthesize(plan, facts) is False


@pytest.mark.asyncio
async def test_reap_recovers_from_malformed_planner_output(
    mock_llm, bridge_question, sample_corpus
):
    retriever = AsyncMock()
    retriever.retrieve.side_effect = [
        make_retrieval_result("What is the capital of France?", sample_corpus[:2]),
        make_retrieval_result("Who is the mayor of Paris?", sample_corpus[1:3]),
    ]
    mock_llm.generate.side_effect = [
        (
            '{"user_goal": "Find the mayor of the capital of France.", '
            '"requirements": ['
            '{"requirement_id": "r1", "question": "What is the capital of France?", "depends_on": [], "status": "pending"}, '
            '{"requirement_id": "r2", "question": "Who is the mayor of Paris?", "depends_on": ["r1"], "status": "pending"}'
            "]}",
            20,
            0.001,
        ),
        ("not valid json", 20, 0.001),
        (
            '{"reasoned_facts": ['
            '{"reasoning": "Paris is directly stated.", '
            '"direct_evidence": "Paris is the capital of France.", '
            '"statement": "The capital of France is Paris.", '
            '"fulfills_requirement_id": "r1", '
            '"fulfillment_level": "DIRECT_ANSWER"}'
            "]}",
            20,
            0.001,
        ),
        (
            '{"next_step": "EXECUTE", '
            '"updated_plan": ['
            '{"requirement_id": "r1", "question": "What is the capital of France?", "depends_on": [], "status": "resolved"}, '
            '{"requirement_id": "r2", "question": "Who is the mayor of Paris?", "depends_on": ["r1"], "status": "pending"}'
            "], "
            '"next_actions": [{"requirement_id": "r2", "question": "Who is the mayor of Paris?"}]}',
            20,
            0.001,
        ),
        (
            '{"reasoned_facts": ['
            '{"reasoning": "Anne Hidalgo is directly stated.", '
            '"direct_evidence": "Anne Hidalgo is the mayor of Paris.", '
            '"statement": "The mayor of Paris is Anne Hidalgo.", '
            '"fulfills_requirement_id": "r2", '
            '"fulfillment_level": "DIRECT_ANSWER"}'
            "]}",
            20,
            0.001,
        ),
        ("Anne Hidalgo", 10, 0.001),
    ]

    rag = REAPRAG(mock_llm, retriever, {"max_iterations": 4, "top_k": 2})
    response = await rag.answer(bridge_question, sample_corpus)

    assert response.answer == "Anne Hidalgo"
    assert response.reasoning_chain[1].action == "plan"


def test_resolve_prompt_path_supports_repo_relative_paths():
    rag = REAPRAG(AsyncMock(model="test-model"), AsyncMock(), {})
    prompt_path = rag._resolve_prompt_path("prompts/reap_decompose.txt")
    assert prompt_path.exists()


def test_select_final_answer_normalizes_comparison_answers_to_yes_no():
    answer = REAPRAG._select_final_answer(
        question_text="Were Scott Derrickson and Ed Wood of the same nationality?",
        plan=[{"requirement_id": "r1", "status": "resolved", "depends_on": []}],
        facts=[
            {
                "statement": "Scott Derrickson and Ed Wood are of the same nationality.",
                "fulfills_requirement_id": "r1",
                "fulfillment_level": "DIRECT_ANSWER",
            }
        ],
        llm_answer="Scott Derrickson and Ed Wood are of the same nationality.",
    )
    assert answer == "yes"


def test_select_final_answer_prefers_exact_title_for_position_question():
    answer = REAPRAG._select_final_answer(
        question_text=(
            "What government position was held by the woman who portrayed Corliss Archer "
            "in the film Kiss and Tell?"
        ),
        plan=[{"requirement_id": "r1", "status": "resolved", "depends_on": []}],
        facts=[
            {
                "statement": "Shirley Temple served as U.S. Ambassador to Ghana.",
                "fulfills_requirement_id": "r1",
                "fulfillment_level": "DIRECT_ANSWER",
            },
            {
                "statement": "Shirley Temple also served as Chief of Protocol of the United States.",
                "fulfills_requirement_id": "r1",
                "fulfillment_level": "DIRECT_ANSWER",
            },
        ],
        llm_answer="U.S. Ambassador to Ghana",
    )
    assert answer == "Chief of Protocol of the United States"


@pytest.mark.asyncio
async def test_reap_handles_direct_fact_extraction_flow(mock_llm, bridge_question, sample_corpus):
    retriever = AsyncMock()
    retriever.retrieve.side_effect = [
        make_retrieval_result("What is the capital of France?", sample_corpus[:2]),
        make_retrieval_result("Who is the mayor of Paris?", sample_corpus[1:3]),
    ]
    mock_llm.generate.side_effect = [
        (
            '{"user_goal": "Find the mayor of the capital of France.", '
            '"requirements": ['
            '{"requirement_id": "r1", "question": "What is the capital of France?", "depends_on": [], "status": "pending"}, '
            '{"requirement_id": "r2", "question": "Who is the mayor of Paris?", "depends_on": ["r1"], "status": "pending"}'
            "]}",
            20,
            0.001,
        ),
        (
            '{"next_step": "EXECUTE", '
            '"updated_plan": ['
            '{"requirement_id": "r1", "question": "What is the capital of France?", "depends_on": [], "status": "pending"}, '
            '{"requirement_id": "r2", "question": "Who is the mayor of Paris?", "depends_on": ["r1"], "status": "pending"}'
            "], "
            '"next_actions": [{"requirement_id": "r1", "question": "What is the capital of France?"}]}',
            20,
            0.001,
        ),
        (
            '{"reasoned_facts": ['
            '{"reasoning": "Paris is directly stated.", '
            '"direct_evidence": "Paris is the capital of France.", '
            '"statement": "The capital of France is Paris.", '
            '"fulfills_requirement_id": "r1", '
            '"fulfillment_level": "DIRECT_ANSWER"}'
            "]}",
            20,
            0.001,
        ),
        (
            '{"next_step": "EXECUTE", '
            '"updated_plan": ['
            '{"requirement_id": "r1", "question": "What is the capital of France?", "depends_on": [], "status": "resolved"}, '
            '{"requirement_id": "r2", "question": "Who is the mayor of Paris?", "depends_on": ["r1"], "status": "pending"}'
            "], "
            '"next_actions": [{"requirement_id": "r2", "question": "Who is the mayor of Paris?"}]}',
            20,
            0.001,
        ),
        (
            '{"reasoned_facts": ['
            '{"reasoning": "Anne Hidalgo is directly stated.", '
            '"direct_evidence": "Anne Hidalgo is the mayor of Paris.", '
            '"statement": "The mayor of Paris is Anne Hidalgo.", '
            '"fulfills_requirement_id": "r2", '
            '"fulfillment_level": "DIRECT_ANSWER"}'
            "]}",
            20,
            0.001,
        ),
        ("Anne Hidalgo", 10, 0.001),
    ]

    rag = REAPRAG(mock_llm, retriever, {"max_iterations": 4, "top_k": 2})
    response = await rag.answer(bridge_question, sample_corpus)

    assert response.answer == "Anne Hidalgo"
    assert response.num_retrieval_calls == 2
    assert response.num_llm_calls == 6
    assert [step.action for step in response.reasoning_chain] == [
        "decompose",
        "plan",
        "extract",
        "plan",
        "extract",
        "synthesize",
    ]


@pytest.mark.asyncio
async def test_reap_uses_replanner_after_partial_clue(mock_llm, bridge_question, sample_corpus):
    retriever = AsyncMock()
    retriever.retrieve.side_effect = [
        make_retrieval_result("What is the capital of France?", sample_corpus[:1]),
        make_retrieval_result("Who is the mayor of Paris?", sample_corpus[1:2]),
        make_retrieval_result("What is the capital of France?", sample_corpus[:1]),
    ]
    mock_llm.generate.side_effect = [
        (
            '{"user_goal": "Find the mayor of the capital of France.", '
            '"requirements": ['
            '{"requirement_id": "r1", "question": "What is the capital of France?", "depends_on": [], "status": "pending"}, '
            '{"requirement_id": "r2", "question": "Who is the mayor of Paris?", "depends_on": ["r1"], "status": "pending"}'
            "]}",
            20,
            0.001,
        ),
        (
            '{"next_step": "EXECUTE", '
            '"updated_plan": ['
            '{"requirement_id": "r1", "question": "What is the capital of France?", "depends_on": [], "status": "pending"}, '
            '{"requirement_id": "r2", "question": "Who is the mayor of Paris?", "depends_on": ["r1"], "status": "pending"}'
            "], "
            '"next_actions": [{"requirement_id": "r1", "question": "What is the capital of France?"}]}',
            20,
            0.001,
        ),
        (
            '{"reasoned_facts": ['
            '{"reasoning": "Paris is mentioned but the final bridge target is still unresolved.", '
            '"direct_evidence": "Paris is the capital of France.", '
            '"statement": "Paris is the capital of France.", '
            '"fulfills_requirement_id": "r1", '
            '"fulfillment_level": "PARTIAL_CLUE"}'
            "]}",
            20,
            0.001,
        ),
        (
            '{"next_step": "EXECUTE", '
            '"updated_plan": ['
            '{"requirement_id": "r1", "question": "What is the capital of France?", "depends_on": [], "status": "pending"}, '
            '{"requirement_id": "r2", "question": "Who is the mayor of Paris?", "depends_on": [], "status": "pending"}'
            "], "
            '"next_actions": [{"requirement_id": "r2", "question": "Who is the mayor of Paris?"}]}',
            20,
            0.001,
        ),
        (
            '{"reasoned_facts": ['
            '{"reasoning": "Anne Hidalgo is directly stated.", '
            '"direct_evidence": "Anne Hidalgo is the mayor of Paris.", '
            '"statement": "The mayor of Paris is Anne Hidalgo.", '
            '"fulfills_requirement_id": "r2", '
            '"fulfillment_level": "DIRECT_ANSWER"}'
            "]}",
            20,
            0.001,
        ),
        (
            '{"next_step": "EXECUTE", '
            '"updated_plan": ['
            '{"requirement_id": "r1", "question": "What is the capital of France?", "depends_on": [], "status": "pending"}, '
            '{"requirement_id": "r2", "question": "Who is the mayor of Paris?", "depends_on": [], "status": "resolved"}'
            "], "
            '"next_actions": [{"requirement_id": "r1", "question": "What is the capital of France?"}]}',
            20,
            0.001,
        ),
        (
            '{"reasoned_facts": ['
            '{"reasoning": "Paris is directly stated.", '
            '"direct_evidence": "Paris is the capital of France.", '
            '"statement": "The capital of France is Paris.", '
            '"fulfills_requirement_id": "r1", '
            '"fulfillment_level": "DIRECT_ANSWER"}'
            "]}",
            20,
            0.001,
        ),
        ("Anne Hidalgo", 10, 0.001),
    ]

    rag = REAPRAG(mock_llm, retriever, {"max_iterations": 4, "top_k": 2})
    response = await rag.answer(bridge_question, sample_corpus)

    assert response.answer == "Anne Hidalgo"
    assert "replan" in [step.action for step in response.reasoning_chain]


def test_prompt_files_exist():
    rag = REAPRAG(AsyncMock(model="test-model"), AsyncMock(), {})
    for prompt_name in [
        "reap_decompose.txt",
        "reap_extract.txt",
        "reap_plan.txt",
        "reap_replan.txt",
        "reap_synthesize.txt",
    ]:
        assert rag._resolve_prompt_path(f"prompts/{prompt_name}").exists()
