import json

from src.core.types import ArchitectureType, BenchmarkResult, EvaluationResult, QuestionType
from src.utils.results import save_results


def test_save_results_includes_supporting_fact_fields_consistently(tmp_path) -> None:
    output_dir = tmp_path / "run"
    per_question_result = EvaluationResult(
        question_id="q1",
        question_type=QuestionType.BRIDGE,
        exact_match=1.0,
        f1=1.0,
        predicted_answer="A",
        gold_answer="A",
        supporting_fact_em=1.0,
        supporting_fact_f1=1.0,
        joint_em=1.0,
        joint_f1=1.0,
        latency_ms=1.0,
        tokens_used=10,
        cost_usd=0.01,
        num_retrieval_calls=1,
        num_llm_calls=1,
    )
    per_question_result.supporting_fact_status = "computed"
    per_question_result.predicted_supporting_facts = [("Doc1", 0)]
    per_question_result.gold_supporting_facts = [("Doc1", 0)]

    result = BenchmarkResult(
        architecture="vanilla_rag",
        architecture_type=ArchitectureType.VANILLA,
        model="test-model",
        dataset="test",
        num_questions=1,
        avg_exact_match=1.0,
        avg_f1=1.0,
        avg_supporting_fact_em=1.0,
        avg_supporting_fact_f1=1.0,
        metrics_by_type={QuestionType.BRIDGE: {"exact_match": 1.0, "f1": 1.0}},
        avg_latency_ms=1.0,
        avg_tokens_per_question=10.0,
        avg_retrieval_calls=1.0,
        avg_llm_calls=1.0,
        total_cost_usd=0.01,
        total_tokens=10,
        per_question_results=[per_question_result],
    )
    result.avg_joint_em = 1.0
    result.avg_joint_f1 = 1.0

    save_results(
        result, output_dir, resolved_config={"evaluation": {"compute_supporting_facts": True}}
    )

    summary = json.loads((output_dir / "summary.json").read_text())
    prediction = json.loads((output_dir / "predictions.jsonl").read_text().splitlines()[0])

    assert summary["avg_joint_em"] == 1.0
    assert summary["avg_joint_f1"] == 1.0
    assert prediction["supporting_fact_status"] == "computed"
    assert prediction["predicted_supporting_facts"] == [["Doc1", 0]]
    assert prediction["gold_supporting_facts"] == [["Doc1", 0]]
