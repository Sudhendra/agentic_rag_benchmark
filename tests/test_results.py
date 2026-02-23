import json
from pathlib import Path

from src.core.types import ArchitectureType, BenchmarkResult, EvaluationResult, QuestionType
from src.utils.results import save_results


def test_save_results_writes_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    result = BenchmarkResult(
        architecture="vanilla_rag",
        architecture_type=ArchitectureType.VANILLA,
        model="test-model",
        dataset="test",
        num_questions=1,
        avg_exact_match=1.0,
        avg_f1=1.0,
        avg_supporting_fact_em=None,
        avg_supporting_fact_f1=None,
        metrics_by_type={QuestionType.SINGLE_HOP: {"exact_match": 1.0, "f1": 1.0}},
        avg_latency_ms=1.0,
        avg_tokens_per_question=10.0,
        avg_retrieval_calls=1.0,
        avg_llm_calls=1.0,
        total_cost_usd=0.01,
        total_tokens=10,
        per_question_results=[
            EvaluationResult(
                question_id="q1",
                question_type=QuestionType.SINGLE_HOP,
                exact_match=1.0,
                f1=1.0,
                predicted_answer="A",
                gold_answer="A",
                latency_ms=1.0,
                tokens_used=10,
                cost_usd=0.01,
                num_retrieval_calls=1,
                num_llm_calls=1,
            )
        ],
    )

    save_results(result, output_dir, resolved_config={"x": 1})

    assert (output_dir / "summary.json").exists()
    assert (output_dir / "predictions.jsonl").exists()
    assert (output_dir / "resolved_config.yaml").exists()

    summary = json.loads((output_dir / "summary.json").read_text())
    assert "metrics_by_type" in summary
    assert summary["metrics_by_type"]["single_hop"]["f1"] == 1.0
