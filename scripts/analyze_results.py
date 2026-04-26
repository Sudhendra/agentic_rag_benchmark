#!/usr/bin/env python3
"""Analyze and compare RAG benchmark results.

Usage:
    # Analyze a single run
    python scripts/analyze_results.py --results results/<run_id> --breakdown --errors

    # Compare multiple runs
    python scripts/analyze_results.py --results results/ --compare

    # Export comparison to CSV
    python scripts/analyze_results.py --results results/ --compare --export comparison.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
import random
from pathlib import Path
from typing import Any

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def load_results(results_dir: Path) -> dict[str, Any]:
    """Load summary and predictions from a results directory.

    Args:
        results_dir: Path to the results directory containing summary.json and predictions.jsonl

    Returns:
        Dictionary with 'summary' and 'predictions' keys
    """
    summary_path = results_dir / "summary.json"
    predictions_path = results_dir / "predictions.jsonl"

    if not summary_path.exists():
        raise FileNotFoundError(f"Summary not found: {summary_path}")

    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)

    predictions = []
    if predictions_path.exists():
        with open(predictions_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    predictions.append(json.loads(line))

    return {"summary": summary, "predictions": predictions}


def compute_bootstrap_ci(values: list[float], n_resamples: int = 1000, confidence_level: float = 0.95) -> tuple[float, float]:
    """Calculate bootstrap confidence interval for the mean.

    Args:
        values: List of float values (e.g., F1 scores)
        n_resamples: Number of bootstrap iterations
        confidence_level: Desired confidence level (e.g., 0.95)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if not values:
        return 0.0, 0.0
    n = len(values)
    means = []
    for _ in range(n_resamples):
        sample = [random.choice(values) for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lower_idx = int((1 - confidence_level) / 2 * n_resamples)
    upper_idx = int((1 + confidence_level) / 2 * n_resamples)
    return means[lower_idx], means[min(upper_idx, n_resamples - 1)]


def breakdown_by_question_type(predictions: list[dict]) -> dict[str, dict[str, float]]:
    """Compute metrics breakdown by question type.

    Args:
        predictions: List of prediction records

    Returns:
        Dictionary mapping question type to metrics
    """
    by_type: dict[str, dict[str, list]] = defaultdict(
        lambda: {"em": [], "f1": [], "latency": [], "tokens": []}
    )

    for pred in predictions:
        qtype = pred.get("question_type", "unknown")
        by_type[qtype]["em"].append(pred.get("exact_match", 0))
        by_type[qtype]["f1"].append(pred.get("f1", 0))
        by_type[qtype]["latency"].append(pred.get("latency_ms", 0))
        by_type[qtype]["tokens"].append(pred.get("tokens_used", 0))

    result = {}
    for qtype, metrics in by_type.items():
        count = len(metrics["em"])
        result[qtype] = {
            "count": count,
            "avg_em": sum(metrics["em"]) / count if count > 0 else 0,
            "avg_f1": sum(metrics["f1"]) / count if count > 0 else 0,
            "avg_latency_ms": sum(metrics["latency"]) / count if count > 0 else 0,
            "avg_tokens": sum(metrics["tokens"]) / count if count > 0 else 0,
        }

    return result


def breakdown_by_hop_count(predictions: list[dict]) -> dict[str, dict[str, float]]:
    """Compute metrics breakdown by hop count for MuSiQue.

    Extracts hop count from question_id (e.g., '2hop__460946_294723' -> 2 hops).

    Args:
        predictions: List of prediction records

    Returns:
        Dictionary mapping hop count to metrics
    """
    by_hop: dict[str, dict[str, list]] = defaultdict(
        lambda: {"em": [], "f1": [], "latency": [], "tokens": []}
    )

    for pred in predictions:
        qid = pred.get("question_id", "")

        # Extract hop count from question_id prefix
        if qid.startswith("2hop"):
            hops = "2-hop"
        elif qid.startswith("3hop"):
            hops = "3-hop"
        elif qid.startswith("4hop"):
            hops = "4-hop"
        elif qid.startswith("5hop"):
            hops = "5-hop"
        else:
            continue  # Skip unknown hop counts

        by_hop[hops]["em"].append(pred.get("exact_match", 0))
        by_hop[hops]["f1"].append(pred.get("f1", 0))
        by_hop[hops]["latency"].append(pred.get("latency_ms", 0))
        by_hop[hops]["tokens"].append(pred.get("tokens_used", 0))

    result = {}
    for hops, metrics in by_hop.items():
        count = len(metrics["em"])
        result[hops] = {
            "count": count,
            "avg_em": sum(metrics["em"]) / count if count > 0 else 0,
            "avg_f1": sum(metrics["f1"]) / count if count > 0 else 0,
            "avg_latency_ms": sum(metrics["latency"]) / count if count > 0 else 0,
            "avg_tokens": sum(metrics["tokens"]) / count if count > 0 else 0,
        }

    return result


def extract_errors(
    predictions: list[dict],
    threshold: float = 0.5,
    max_errors: int = 20,
) -> list[dict]:
    """Extract predictions with F1 below threshold for error analysis.

    Args:
        predictions: List of prediction records
        threshold: F1 threshold below which to consider as error
        max_errors: Maximum number of errors to return

    Returns:
        List of error records sorted by F1 score (ascending)
    """
    errors = []
    for pred in predictions:
        f1 = pred.get("f1", 0)
        if f1 < threshold:
            # Extract hop count from question_id
            qid = pred.get("question_id", "")
            if qid.startswith("2hop"):
                hops = "2-hop"
            elif qid.startswith("3hop"):
                hops = "3-hop"
            elif qid.startswith("4hop"):
                hops = "4-hop"
            elif qid.startswith("5hop"):
                hops = "5-hop"
            else:
                hops = "unknown"

            errors.append(
                {
                    "question_id": qid,
                    "question_type": pred.get("question_type", "unknown"),
                    "hops": hops,
                    "predicted": pred.get("predicted_answer", ""),
                    "gold": pred.get("gold_answer", ""),
                    "exact_match": pred.get("exact_match", 0),
                    "f1": f1,
                    "num_retrieval_calls": pred.get("num_retrieval_calls", 0),
                    "num_llm_calls": pred.get("num_llm_calls", 0),
                    "latency_ms": pred.get("latency_ms", 0),
                }
            )

    # Sort by F1 (worst first)
    errors.sort(key=lambda x: x["f1"])
    return errors[:max_errors]


def analyze_errors_by_hop(
    predictions: list[dict],
    threshold: float = 0.5,
) -> dict[str, dict]:
    """Analyze error distribution by hop count and question type.

    Args:
        predictions: List of prediction records
        threshold: F1 threshold for counting as error

    Returns:
        Dictionary with error statistics by hop count and question type
    """
    # Initialize counters
    stats = {}
    for hops in ["2-hop", "3-hop", "4-hop", "unknown"]:
        stats[hops] = {
            "total": 0,
            "errors": 0,
            "by_type": {},
        }

    for pred in predictions:
        # Extract hop count
        qid = pred.get("question_id", "")
        if qid.startswith("2hop"):
            hops = "2-hop"
        elif qid.startswith("3hop"):
            hops = "3-hop"
        elif qid.startswith("4hop"):
            hops = "4-hop"
        else:
            hops = "unknown"

        qtype = pred.get("question_type", "unknown")
        f1 = pred.get("f1", 0)
        is_error = f1 < threshold

        # Update total count
        stats[hops]["total"] += 1

        # Initialize type if needed
        if qtype not in stats[hops]["by_type"]:
            stats[hops]["by_type"][qtype] = {"total": 0, "errors": 0, "em_sum": 0, "f1_sum": 0}

        stats[hops]["by_type"][qtype]["total"] += 1
        stats[hops]["by_type"][qtype]["em_sum"] += pred.get("exact_match", 0)
        stats[hops]["by_type"][qtype]["f1_sum"] += f1

        if is_error:
            stats[hops]["errors"] += 1
            stats[hops]["by_type"][qtype]["errors"] += 1

    # Calculate percentages
    result = {}
    for hops, data in stats.items():
        if data["total"] > 0:
            result[hops] = {
                "total": data["total"],
                "errors": data["errors"],
                "error_rate": data["errors"] / data["total"],
                "by_type": {},
            }
            for qtype, type_data in data["by_type"].items():
                if type_data["total"] > 0:
                    result[hops]["by_type"][qtype] = {
                        "total": type_data["total"],
                        "errors": type_data["errors"],
                        "error_rate": type_data["errors"] / type_data["total"],
                        "avg_em": type_data["em_sum"] / type_data["total"],
                        "avg_f1": type_data["f1_sum"] / type_data["total"],
                    }

    return result


def analyze_errors_by_question_type(
    predictions: list[dict],
    threshold: float = 0.5,
) -> dict[str, dict]:
    """Analyze error distribution by question type (Bridge/Comparison).

    Args:
        predictions: List of prediction records
        threshold: F1 threshold for counting as error

    Returns:
        Dictionary with error statistics by question type
    """
    # Initialize counters
    stats = {}

    for pred in predictions:
        qtype = pred.get("question_type", "unknown")
        f1 = pred.get("f1", 0)
        is_error = f1 < threshold

        if qtype not in stats:
            stats[qtype] = {"total": 0, "errors": 0, "em_sum": 0, "f1_sum": 0}

        stats[qtype]["total"] += 1
        stats[qtype]["em_sum"] += pred.get("exact_match", 0)
        stats[qtype]["f1_sum"] += f1

        if is_error:
            stats[qtype]["errors"] += 1

    # Calculate percentages
    result = {}
    for qtype, data in stats.items():
        if data["total"] > 0:
            result[qtype] = {
                "total": data["total"],
                "errors": data["errors"],
                "error_rate": data["errors"] / data["total"],
                "avg_em": data["em_sum"] / data["total"],
                "avg_f1": data["f1_sum"] / data["total"],
            }

    return result


def find_run_directories(results_path: Path) -> list[Path]:
    """Find all valid run directories in a results path.

    Args:
        results_path: Path to results directory or parent of multiple runs

    Returns:
        List of paths to directories containing summary.json
    """
    if (results_path / "summary.json").exists():
        return [results_path]

    run_dirs = []
    for d in results_path.iterdir():
        if d.is_dir() and (d / "summary.json").exists():
            run_dirs.append(d)

    return sorted(run_dirs, key=lambda x: x.name)


def compare_runs(run_dirs: list[Path], compute_stats: bool = False) -> list[dict[str, Any]]:
    """Compare metrics across multiple runs.

    Args:
        run_dirs: List of run directories
        compute_stats: Whether to compute bootstrap confidence intervals

    Returns:
        List of summary records for each run
    """
    rows = []
    for run_dir in run_dirs:
        try:
            results = load_results(run_dir)
            summary = results["summary"]
            predictions = results["predictions"]
            row = {
                "run_id": run_dir.name[:12],  # Truncate long run IDs
                "architecture": summary.get("architecture", "unknown"),
                "model": summary.get("model", "unknown"),
                "num_questions": summary.get("num_questions", 0),
                "exact_match": summary.get("avg_exact_match", 0),
                "f1": summary.get("avg_f1", 0),
                "latency_ms": summary.get("avg_latency_ms", 0),
                "tokens_per_q": summary.get("avg_tokens_per_question", 0),
                "cost_usd": summary.get("total_cost_usd", 0),
            }
            if compute_stats and predictions:
                f1_scores = [p.get("f1", 0) for p in predictions]
                if f1_scores:
                    ci_lower, ci_upper = compute_bootstrap_ci(f1_scores)
                    row["f1_ci_lower"] = ci_lower
                    row["f1_ci_upper"] = ci_upper
            rows.append(row)
        except Exception as e:
            print(f"Warning: Failed to load {run_dir}: {e}", file=sys.stderr)

    return rows


def format_table(rows: list[dict], columns: list[str] | None = None) -> str:
    """Format rows as a text table.

    Args:
        rows: List of dictionaries with the same keys
        columns: Optional list of columns to include (in order)

    Returns:
        Formatted table string
    """
    if not rows:
        return "No data"

    if columns is None:
        columns = list(rows[0].keys())

    # Calculate column widths
    widths = {col: len(col) for col in columns}
    for row in rows:
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                val_str = f"{val:.4f}"
            else:
                val_str = str(val)
            widths[col] = max(widths[col], len(val_str))

    # Build header
    header = " | ".join(col.ljust(widths[col]) for col in columns)
    separator = "-+-".join("-" * widths[col] for col in columns)

    # Build rows
    lines = [header, separator]
    for row in rows:
        cells = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                val_str = f"{val:.4f}"
            else:
                val_str = str(val)
            cells.append(val_str.ljust(widths[col]))
        lines.append(" | ".join(cells))

    return "\n".join(lines)


def export_csv(rows: list[dict], output_path: Path, columns: list[str] | None = None) -> None:
    """Export rows to CSV file.

    Args:
        rows: List of dictionaries
        output_path: Path to output CSV file
        columns: Optional list of columns to include
    """
    if not rows:
        return

    if columns is None:
        columns = list(rows[0].keys())

    with open(output_path, "w", encoding="utf-8") as f:
        # Header
        f.write(",".join(columns) + "\n")
        # Rows
        for row in rows:
            cells = []
            for col in columns:
                val = row.get(col, "")
                if isinstance(val, str) and ("," in val or '"' in val):
                    val = f'"{val.replace(chr(34), chr(34) + chr(34))}"'
                else:
                    val = str(val)
                cells.append(val)
            f.write(",".join(cells) + "\n")

    print(f"Exported to {output_path}")


def print_summary(summary: dict) -> None:
    """Print a formatted summary."""
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    print(f"\nArchitecture: {summary.get('architecture', 'unknown')}")
    print(f"Model:        {summary.get('model', 'unknown')}")
    print(f"Dataset:      {summary.get('dataset', 'unknown')}")
    print(f"Questions:    {summary.get('num_questions', 0)}")

    print("\n--- Accuracy Metrics ---")
    print(
        f"Exact Match:  {summary.get('avg_exact_match', 0):.4f} ({summary.get('avg_exact_match', 0) * 100:.1f}%)"
    )
    print(f"F1 Score:     {summary.get('avg_f1', 0):.4f} ({summary.get('avg_f1', 0) * 100:.1f}%)")

    if summary.get("avg_supporting_fact_em") is not None:
        print(f"SP EM:        {summary.get('avg_supporting_fact_em', 0):.4f}")
        print(f"SP F1:        {summary.get('avg_supporting_fact_f1', 0):.4f}")

    print("\n--- Efficiency Metrics ---")
    print(f"Avg Latency:  {summary.get('avg_latency_ms', 0):.1f} ms")
    print(f"Avg Tokens:   {summary.get('avg_tokens_per_question', 0):.0f}")
    print(f"Total Cost:   ${summary.get('total_cost_usd', 0):.4f}")
    print(
        f"Cost/Question: ${summary.get('total_cost_usd', 0) / max(summary.get('num_questions', 1), 1):.5f}"
    )


def print_breakdown(breakdown: dict[str, dict]) -> None:
    """Print breakdown by question type."""
    print("\n" + "=" * 60)
    print("BREAKDOWN BY QUESTION TYPE")
    print("=" * 60 + "\n")

    rows = []
    for qtype, metrics in sorted(breakdown.items()):
        rows.append(
            {
                "type": qtype,
                "count": metrics["count"],
                "exact_match": metrics["avg_em"],
                "f1": metrics["avg_f1"],
                "latency_ms": metrics["avg_latency_ms"],
            }
        )

    print(format_table(rows))


def print_hop_breakdown(breakdown: dict[str, dict]) -> None:
    """Print breakdown by hop count."""
    print("\n" + "=" * 60)
    print("BREAKDOWN BY HOP COUNT (MuSiQue)")
    print("=" * 60 + "\n")

    rows = []
    # Sort by hop number
    hop_order = ["2-hop", "3-hop", "4-hop", "5-hop"]
    for hop in hop_order:
        if hop in breakdown:
            metrics = breakdown[hop]
            rows.append(
                {
                    "hops": hop,
                    "count": metrics["count"],
                    "exact_match": metrics["avg_em"],
                    "f1": metrics["avg_f1"],
                    "latency_ms": metrics["avg_latency_ms"],
                }
            )

    if rows:
        print(format_table(rows))
    else:
        print("No hop count data found (dataset may not be MuSiQue)")


def print_errors(errors: list[dict]) -> None:
    """Print error analysis."""
    print("\n" + "=" * 60)
    print(f"ERROR ANALYSIS (F1 < 0.5) - Showing {len(errors)} errors")
    print("=" * 60)

    for i, err in enumerate(errors, 1):
        hops = err.get("hops", "unknown")
        retrievals = err.get("num_retrieval_calls", "?")
        llm_calls = err.get("num_llm_calls", "?")

        print(f"\n[{i}] {err['question_id']} ({err['question_type']}, {hops})")
        print(
            f"    Predicted: {err['predicted'][:80]}{'...' if len(err['predicted']) > 80 else ''}"
        )
        print(f"    Gold:      {err['gold'][:80]}{'...' if len(err['gold']) > 80 else ''}")
        print(
            f"    EM: {err['exact_match']:.0%}  F1: {err['f1']:.4f} | Ret: {retrievals} | LLM: {llm_calls}"
        )


def print_error_analysis_by_hop(hop_stats: dict, threshold: float = 0.5) -> None:
    """Print error analysis by hop count and question type."""
    print("\n" + "=" * 60)
    print(f"ERROR ANALYSIS BY HOP (F1 < {threshold})")
    print("=" * 60)

    # Print overall by hop
    print("\n--- Error Rate by Hop Count ---")
    print(
        f"{'Hop':<10} {'Total':>8} {'Errors':>8} {'Error Rate':>12} {'Avg EM':>10} {'Avg F1':>10}"
    )
    print("-" * 60)

    for hops in ["2-hop", "3-hop", "4-hop"]:
        if hops in hop_stats:
            data = hop_stats[hops]
            print(
                f"{hops:<10} {data['total']:>8} {data['errors']:>8} {data['error_rate']:>11.1%} {data.get('avg_em', 0):>9.1%} {data.get('avg_f1', 0):>9.1%}"
            )

    # Print by question type
    print("\n--- Error Rate by Question Type ---")
    for hops in ["2-hop", "3-hop", "4-hop"]:
        if hops in hop_stats and hop_stats[hops].get("by_type"):
            print(f"\n{hops}:")
            for qtype, data in hop_stats[hops]["by_type"].items():
                print(
                    f"  {qtype:<15}: {data['total']:>4} questions, {data['errors']:>4} errors ({data['error_rate']:>5.1%}), EM: {data['avg_em']:.1%}, F1: {data['avg_f1']:.1%}"
                )


def print_error_analysis_by_qtype(qtype_stats: dict, threshold: float = 0.5) -> None:
    """Print error analysis by question type (Bridge/Comparison)."""
    print("\n" + "=" * 60)
    print(f"ERROR ANALYSIS BY QUESTION TYPE (F1 < {threshold})")
    print("=" * 60)

    print("\n--- Error Rate by Question Type ---")
    print(
        f"{'Type':<15} {'Total':>8} {'Errors':>8} {'Error Rate':>12} {'Avg EM':>10} {'Avg F1':>10}"
    )
    print("-" * 65)

    for qtype in sorted(qtype_stats.keys()):
        data = qtype_stats[qtype]
        print(
            f"{qtype:<15} {data['total']:>8} {data['errors']:>8} {data['error_rate']:>11.1%} {data['avg_em']:>9.1%} {data['avg_f1']:>9.1%}"
        )


def print_comparison(rows: list[dict], show_stats: bool = False) -> None:
    """Print comparison table."""
    print("\n" + "=" * 80)
    print("RUN COMPARISON")
    print("=" * 80 + "\n")

    if show_stats:
        columns = [
            "run_id",
            "architecture",
            "model",
            "num_questions",
            "exact_match",
            "f1",
            "f1_ci_lower",
            "f1_ci_upper",
            "latency_ms",
            "cost_usd",
        ]
    else:
        columns = [
            "run_id",
            "architecture",
            "model",
            "num_questions",
            "exact_match",
            "f1",
            "latency_ms",
            "cost_usd",
        ]
    print(format_table(rows, columns))

    # Find best run
    if rows:
        best_f1 = max(rows, key=lambda x: x.get("f1", 0))
        best_em = max(rows, key=lambda x: x.get("exact_match", 0))
        cheapest = min(rows, key=lambda x: x.get("cost_usd", float("inf")))
        fastest = min(rows, key=lambda x: x.get("latency_ms", float("inf")))

        print("\n--- Best Performers ---")
        print(f"Best F1:      {best_f1['run_id']} ({best_f1['f1']:.4f})")
        print(f"Best EM:      {best_em['run_id']} ({best_em['exact_match']:.4f})")
        print(f"Cheapest:     {cheapest['run_id']} (${cheapest['cost_usd']:.4f})")
        print(f"Fastest:      {fastest['run_id']} ({fastest['latency_ms']:.1f} ms)")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze RAG benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results",
        required=True,
        type=Path,
        help="Path to results directory or parent of multiple runs",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare multiple runs (finds all run dirs in --results)",
    )
    parser.add_argument(
        "--breakdown",
        action="store_true",
        help="Show breakdown by question type",
    )
    parser.add_argument(
        "--hops",
        action="store_true",
        help="Show breakdown by hop count (for MuSiQue)",
    )
    parser.add_argument(
        "--errors",
        action="store_true",
        help="Show error analysis",
    )
    parser.add_argument(
        "--error-threshold",
        type=float,
        default=0.5,
        help="F1 threshold for error analysis (default: 0.5)",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=20,
        help="Maximum errors to show (default: 20)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Compute bootstrap confidence intervals for F1 scores",
    )
    parser.add_argument(
        "--export",
        type=Path,
        help="Export comparison to CSV file",
    )
    parser.add_argument(
        "--pareto",
        type=Path,
        help="Export pareto frontier data (cost/latency vs f1) to CSV file",
    )

    args = parser.parse_args()

    if not args.results.exists():
        print(f"Error: Results path not found: {args.results}", file=sys.stderr)
        sys.exit(1)

    if args.compare:
        # Compare mode
        run_dirs = find_run_directories(args.results)
        if not run_dirs:
            print("No run directories found with summary.json", file=sys.stderr)
            sys.exit(1)

        print(f"Found {len(run_dirs)} run(s) to compare")
        rows = compare_runs(run_dirs, compute_stats=args.stats)
        print_comparison(rows, show_stats=args.stats)

        if args.export:
            columns = list(rows[0].keys()) if rows else None
            export_csv(rows, args.export, columns=columns)
            
        if args.pareto:
            pareto_cols = ["run_id", "architecture", "model", "f1", "latency_ms", "cost_usd"]
            export_csv(rows, args.pareto, columns=pareto_cols)
    else:
        # Single run mode
        if (args.results / "summary.json").exists():
            results_dir = args.results
        else:
            # Try to find the most recent run
            run_dirs = find_run_directories(args.results)
            if not run_dirs:
                print("No run directories found", file=sys.stderr)
                sys.exit(1)
            results_dir = run_dirs[-1]  # Most recent
            print(f"Using most recent run: {results_dir.name}")

        results = load_results(results_dir)

        # Print summary
        print_summary(results["summary"])

        if args.breakdown and results["predictions"]:
            breakdown = breakdown_by_question_type(results["predictions"])
            print_breakdown(breakdown)

        if args.hops and results["predictions"]:
            hop_breakdown = breakdown_by_hop_count(results["predictions"])
            print_hop_breakdown(hop_breakdown)

        if args.errors and results["predictions"]:
            errors = extract_errors(
                results["predictions"],
                threshold=args.error_threshold,
                max_errors=args.max_errors,
            )
            if errors:
                print_errors(errors)
            else:
                print(f"\nNo errors found (all predictions have F1 >= {args.error_threshold})")

            # Also print error analysis by hop
            hop_stats = analyze_errors_by_hop(
                results["predictions"],
                threshold=args.error_threshold,
            )
            print_error_analysis_by_hop(hop_stats, threshold=args.error_threshold)

            # Also print error analysis by question type
            qtype_stats = analyze_errors_by_question_type(
                results["predictions"],
                threshold=args.error_threshold,
            )
            print_error_analysis_by_qtype(qtype_stats, threshold=args.error_threshold)


if __name__ == "__main__":
    main()
