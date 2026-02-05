from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from ..core.types import BenchmarkResult


def save_results(
    result: BenchmarkResult,
    output_dir: Path,
    resolved_config: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "architecture": result.architecture,
        "architecture_type": result.architecture_type.value,
        "model": result.model,
        "dataset": result.dataset,
        "num_questions": result.num_questions,
        "avg_exact_match": result.avg_exact_match,
        "avg_f1": result.avg_f1,
        "avg_supporting_fact_em": result.avg_supporting_fact_em,
        "avg_supporting_fact_f1": result.avg_supporting_fact_f1,
        "avg_latency_ms": result.avg_latency_ms,
        "avg_tokens_per_question": result.avg_tokens_per_question,
        "avg_retrieval_calls": result.avg_retrieval_calls,
        "avg_llm_calls": result.avg_llm_calls,
        "total_cost_usd": result.total_cost_usd,
        "total_tokens": result.total_tokens,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for item in result.per_question_results:
            record = {
                "question_id": item.question_id,
                "question_type": item.question_type.value,
                "predicted_answer": item.predicted_answer,
                "gold_answer": item.gold_answer,
                "exact_match": item.exact_match,
                "f1": item.f1,
                "supporting_fact_em": item.supporting_fact_em,
                "supporting_fact_f1": item.supporting_fact_f1,
                "joint_em": item.joint_em,
                "joint_f1": item.joint_f1,
                "latency_ms": item.latency_ms,
                "tokens_used": item.tokens_used,
                "cost_usd": item.cost_usd,
                "num_retrieval_calls": item.num_retrieval_calls,
                "num_llm_calls": item.num_llm_calls,
            }
            handle.write(json.dumps(record) + "\n")

    (output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved_config))
