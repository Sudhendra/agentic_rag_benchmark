#!/usr/bin/env python3
"""Classify prediction errors into a taxonomy of failure modes.

Usage:
    python scripts/error_taxonomy.py                              # Analyze all best runs
    python scripts/error_taxonomy.py --run <run_id>               # Analyze a single run
    python scripts/error_taxonomy.py --export results/taxonomy    # Export results
    python scripts/error_taxonomy.py --samples 20                 # Show sample errors per category
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

OUTPUT_DIR = ROOT_DIR / "results" / "error_taxonomy"

# Best runs per architecture (full HotpotQA, 7405 questions)
# Keyed by architecture name -> (run_dir_name, description)
BEST_RUNS: dict[str, tuple[str, str]] = {
    "vanilla_rag": ("bfc8f29304c541108d57bd5fb9b93c09", "BM25, best of 3 Vanilla runs"),
    "react_rag": ("25cc3f6b90df45799c62922a88c480c5", "BM25, best of 5 ReAct runs"),
    "self_rag": ("7272b4eb8c81467fb5b87561b7ccd8ae", "BM25, best of 2 Self-RAG runs"),
    "planner_rag": ("b4284f7fb0294f05b0ec3ac315aafd8e", "BM25"),
    "ircot_rag": ("4d923d09821d4670a12d056cc06ac635", "BM25, best of 3 IRCoT runs"),
    "reap_rag": ("c4da1615351b47bf8d5f8e0617188940", "BM25"),
    "recursive_lm": ("09581743885c4de7930cd11d6f43b20d", "BM25, default prompt"),
}

CATEGORIES = [
    "complete_miss",
    "low_overlap",
    "partial",
    "near_miss",
    "verbose",
    "loop",
    "yesno_flip",
    "correct",
]


def categorize_prediction(
    pred: dict,
    arch_p95_calls: float,
) -> list[str]:
    """Classify a single prediction into error categories.

    Returns list of category labels (a prediction can match multiple).
    """
    labels = []
    em = pred.get("exact_match", 0)
    f1 = pred.get("f1", 0.0)
    gold = pred.get("gold_answer", "")
    predicted = pred.get("predicted_answer", "")
    retrieval_calls = pred.get("num_retrieval_calls", 0)
    llm_calls = pred.get("num_llm_calls", 0)
    gold_len = len(gold.strip()) if gold else 1
    pred_len = len(predicted.strip()) if predicted else 0

    # Correct
    if em == 1.0:
        labels.append("correct")
        return labels

    # Yes/No flip
    gold_lower = gold.strip().lower()
    pred_lower = predicted.strip().lower()
    if gold_lower in ("yes", "no") and pred_lower in ("yes", "no") and gold_lower != pred_lower:
        labels.append("yesno_flip")

    # Verbose: predicted answer more than 5x gold length
    if pred_len > 5 * max(gold_len, 1) and f1 > 0:
        labels.append("verbose")

    # Loop: retrieval calls above architecture P95
    if retrieval_calls > arch_p95_calls:
        labels.append("loop")

    # Error severity (based on F1)
    if f1 == 0.0:
        labels.append("complete_miss")
    elif f1 < 0.3:
        labels.append("low_overlap")
    elif f1 < 0.7:
        labels.append("partial")
    else:
        labels.append("near_miss")

    return labels


def compute_p95(values: list[float]) -> float:
    """Compute the 95th percentile."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * 0.95)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]


def load_predictions(run_dir: Path) -> tuple[dict[str, Any], list[dict]]:
    """Load summary and predictions from a run directory."""
    summary_path = run_dir / "summary.json"
    predictions_path = run_dir / "predictions.jsonl"
    if not summary_path.exists() or not predictions_path.exists():
        return {}, []
    summary = json.loads(summary_path.read_text())
    predictions = []
    with open(predictions_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))
    return summary, predictions


def analyze_run(run_dir_name: str, description: str) -> dict[str, Any]:
    """Analyze a single run's predictions and produce taxonomy breakdown."""
    run_dir = ROOT_DIR / "results" / run_dir_name
    summary, predictions = load_predictions(run_dir)
    if not predictions:
        return {"error": f"No predictions found in {run_dir}"}

    arch = summary.get("architecture", "unknown")
    num_q = len(predictions)

    # Compute P95 retrieval calls for this architecture
    ret_calls = [p.get("num_retrieval_calls", 0) for p in predictions]
    p95_calls = compute_p95(ret_calls)

    # Classify each prediction
    category_counts: dict[str, int] = Counter()
    category_by_type: dict[str, dict[str, int]] = defaultdict(lambda: Counter())
    samples: dict[str, list[dict]] = defaultdict(list)
    architecture_errors: dict[str, Counter] = defaultdict(Counter)

    for pred in predictions:
        labels = categorize_prediction(pred, p95_calls)
        qtype = pred.get("question_type", "unknown")

        for label in labels:
            category_counts[label] += 1
            category_by_type[label][qtype] += 1

            # Collect up to 10 samples per category
            if len(samples[label]) < 10:
                samples[label].append(
                    {
                        "question_id": pred.get("question_id", ""),
                        "question_type": qtype,
                        "predicted": pred.get("predicted_answer", ""),
                        "gold": pred.get("gold_answer", ""),
                        "f1": pred.get("f1", 0),
                        "num_retrieval_calls": pred.get("num_retrieval_calls", 0),
                        "num_llm_calls": pred.get("num_llm_calls", 0),
                        "latency_ms": pred.get("latency_ms", 0),
                        "tokens_used": pred.get("tokens_used", 0),
                    }
                )

        # Track errors per architecture (non-correct labels)
        error_labels = [l for l in labels if l != "correct"]
        if error_labels:
            for el in error_labels:
                architecture_errors[el][arch] += 1

    # Compute percentages (non-correct percentages are of total questions)
    correct_count = category_counts.get("correct", 0)
    error_total = num_q - correct_count

    percentages = {}
    for cat in CATEGORIES:
        count = category_counts.get(cat, 0)
        if cat == "correct":
            percentages[cat] = count / num_q * 100 if num_q > 0 else 0
        else:
            percentages[cat] = count / num_q * 100 if num_q > 0 else 0

    # Error rate per question type
    qtype_totals: dict[str, int] = Counter()
    qtype_errors: dict[str, int] = Counter()
    for pred in predictions:
        qtype = pred.get("question_type", "unknown")
        qtype_totals[qtype] += 1
        em = pred.get("exact_match", 0)
        if em == 0:
            qtype_errors[qtype] += 1

    qtype_error_rates = {}
    for qt, total in qtype_totals.items():
        errs = qtype_errors.get(qt, 0)
        qtype_error_rates[qt] = {
            "total": total,
            "errors": errs,
            "error_rate": errs / total if total > 0 else 0,
        }

    # Cost of errors
    error_costs = sum(p.get("cost_usd", 0) for p in predictions if p.get("exact_match", 0) == 0)
    correct_costs = sum(p.get("cost_usd", 0) for p in predictions if p.get("exact_match", 0) == 1)
    total_cost = summary.get("total_cost_usd", 0)

    # Distribution of retrieval calls for errors vs correct
    error_ret_calls = [
        p.get("num_retrieval_calls", 0) for p in predictions if p.get("exact_match", 0) == 0
    ]
    correct_ret_calls = [
        p.get("num_retrieval_calls", 0) for p in predictions if p.get("exact_match", 0) == 1
    ]

    return {
        "architecture": arch,
        "description": description,
        "num_questions": num_q,
        "overall_em": summary.get("avg_exact_match", 0),
        "overall_f1": summary.get("avg_f1", 0),
        "total_cost": total_cost,
        "p95_retrieval_calls": p95_calls,
        "category_counts": dict(category_counts),
        "category_percentages": percentages,
        "category_by_type": {k: dict(v) for k, v in category_by_type.items()},
        "qtype_error_rates": qtype_error_rates,
        "error_cost": error_costs,
        "correct_cost": correct_costs,
        "error_cost_percent": (error_costs / total_cost * 100) if total_cost > 0 else 0,
        "error_avg_retrieval_calls": sum(error_ret_calls) / len(error_ret_calls)
        if error_ret_calls
        else 0,
        "correct_avg_retrieval_calls": sum(correct_ret_calls) / len(correct_ret_calls)
        if correct_ret_calls
        else 0,
        "samples": dict(samples),
    }


def print_architecture_report(result: dict) -> None:
    """Print a formatted report for one architecture."""
    arch = result.get("architecture", "unknown")
    desc = result.get("description", "")
    em = result.get("overall_em", 0)
    f1 = result.get("overall_f1", 0)
    num_q = result.get("num_questions", 0)
    pct = result.get("category_percentages", {})

    print(f"\n{'=' * 70}")
    print(f"  {arch}  ({desc})")
    print(f"  EM: {em:.1%}  |  F1: {f1:.1%}  |  n={num_q}")
    print(f"{'=' * 70}")

    # Category table
    print(f"\n  {'Category':<25} {'Count':>8} {'% of Total':>12}")
    print(f"  {'-' * 45}")
    for cat in CATEGORIES:
        count = result.get("category_counts", {}).get(cat, 0)
        pct_val = pct.get(cat, 0)
        label = cat.replace("_", " ").title()
        print(f"  {label:<25} {count:>8} {pct_val:>10.1f}%")
    print(f"  {'-' * 45}")

    # Error rate by question type
    qtype_rates = result.get("qtype_error_rates", {})
    if qtype_rates:
        print(f"\n  Error Rate by Question Type:")
        for qt, data in sorted(qtype_rates.items()):
            print(f"    {qt:<15}: {data['error_rate']:.1%} ({data['errors']}/{data['total']})")

    # Cost breakdown
    error_cost = result.get("error_cost", 0)
    correct_cost = result.get("correct_cost", 0)
    total_cost = result.get("total_cost", 0)
    error_cost_pct = result.get("error_cost_percent", 0)
    print(f"\n  Cost Breakdown:")
    print(f"    Total:     ${total_cost:.4f}")
    print(f"    Errors:    ${error_cost:.4f} ({error_cost_pct:.1f}%)")
    print(f"    Correct:   ${correct_cost:.4f} ({100 - error_cost_pct:.1f}%)")

    # Retrieval call comparison
    err_avg = result.get("error_avg_retrieval_calls", 0)
    cor_avg = result.get("correct_avg_retrieval_calls", 0)
    print(f"\n  Avg Retrieval Calls:")
    print(f"    Errors:   {err_avg:.1f}")
    print(f"    Correct:  {cor_avg:.1f}")
    ratio = err_avg / cor_avg if cor_avg > 0 else 0
    print(f"    Ratio:    {ratio:.1f}x")

    # Top samples for interesting categories
    samples = result.get("samples", {})
    interesting_cats = ["complete_miss", "near_miss", "loop", "yesno_flip", "verbose"]
    for cat in interesting_cats:
        cat_samples = samples.get(cat, [])
        if cat_samples:
            label = cat.replace("_", " ").title()
            print(f"\n  Sample {label} Errors:")
            for s in cat_samples[:5]:
                print(
                    f"    Q: {s['question_id'][:20]:<20} | Gold: {s['gold'][:40]:<40} | Pred: {s['predicted'][:40]:<40} | F1: {s['f1']:.3f}"
                )


def print_cross_architecture_summary(results: list[dict]) -> None:
    """Print a cross-architecture comparison table."""
    print(f"\n{'=' * 70}")
    print(f"  CROSS-ARCHITECTURE ERROR TAXONOMY")
    print(f"{'=' * 70}")

    # Header
    header = f"{'Architecture':<18}"
    for cat in CATEGORIES:
        label = cat.replace("_", "\n").title()
        header += f" {label:>10}"
    header += f" {'EM':>6} {'F1':>6}"
    print(f"\n{header}")
    print(f"{'-' * (18 + len(CATEGORIES) * 11 + 15)}")

    # Rows
    for r in results:
        arch = r.get("architecture", "?")[:16]
        pct = r.get("category_percentages", {})
        em = r.get("overall_em", 0)
        f1 = r.get("overall_f1", 0)
        row = f"{arch:<18}"
        for cat in CATEGORIES:
            row += f" {pct.get(cat, 0):>9.1f}%"
        row += f" {em:>5.1%} {f1:>5.1%}"
        print(row)


def export_results(results: list[dict], output_dir: Path) -> None:
    """Export taxonomy results as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Full results
    (output_dir / "taxonomy_results.json").write_text(json.dumps(results, indent=2, default=str))

    # Summary table
    summary = []
    for r in results:
        row = {
            "architecture": r["architecture"],
            "num_questions": r["num_questions"],
            "overall_em": r["overall_em"],
            "overall_f1": r["overall_f1"],
            **{f"pct_{cat}": r["category_percentages"].get(cat, 0) for cat in CATEGORIES},
            "error_cost_percent": r["error_cost_percent"],
            "error_avg_retrieval_calls": r["error_avg_retrieval_calls"],
            "correct_avg_retrieval_calls": r["correct_avg_retrieval_calls"],
            "total_cost": r["total_cost"],
        }
        summary.append(row)

    with open(output_dir / "taxonomy_summary.csv", "w", encoding="utf-8") as f:
        import csv

        if summary:
            writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)

    # Sample errors (one file per architecture)
    for r in results:
        arch = r["architecture"]
        samples = r.get("samples", {})
        if not samples:
            continue
        arch_dir = output_dir / "samples" / arch
        arch_dir.mkdir(parents=True, exist_ok=True)
        for cat, cat_samples in samples.items():
            path = arch_dir / f"{cat}.json"
            path.write_text(json.dumps(cat_samples, indent=2))

    print(f"\n  Results exported to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Error taxonomy for RAG predictions")
    parser.add_argument("--run", type=str, default=None, help="Single run directory to analyze")
    parser.add_argument("--export", type=str, default=str(OUTPUT_DIR), help="Export directory")
    parser.add_argument("--samples", type=int, default=5, help="Sample errors to show per category")
    args = parser.parse_args()

    output_dir = Path(args.export)

    # Determine which runs to analyze
    runs_to_analyze: list[tuple[str, str]] = []
    if args.run:
        runs_to_analyze.append((args.run, "user-specified"))
    else:
        runs_to_analyze = list(BEST_RUNS.values())

    results = []
    for run_dir_name, description in runs_to_analyze:
        result = analyze_run(run_dir_name, description)
        if "error" in result:
            print(f"  ERROR: {result['error']}")
            continue
        results.append(result)
        print_architecture_report(result)

    if len(results) > 1:
        print_cross_architecture_summary(results)

    if results:
        export_results(results, output_dir)


if __name__ == "__main__":
    main()
