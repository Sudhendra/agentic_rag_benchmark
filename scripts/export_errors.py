#!/usr/bin/env python3
"""Export errors for manual categorization.

This script extracts all errors from predictions and exports them to a CSV
file with columns ready for manual categorization.

Usage:
    python scripts/export_errors.py --results results/<run_id> --output errors.csv

    # Then manually fill in the 'category' column
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


CATEGORIES = [
    "RETRIEVAL_FAIL",  # Didn't retrieve relevant passages (0-1 retrieval calls)
    "REASONING_FAIL",  # Retrieved right info but reasoned incorrectly
    "ENTITY_CONFUSION",  # Mixed up similar entities
    "PREMATURE_TERMINATION",  # Gave up early ("no information", "unknown")
    "HALLUCINATION",  # Made up facts not in context
    "PARTIAL_ANSWER",  # Answer is close but incomplete
    "GROUND_TRUTH_ERROR",  # Gold answer appears wrong
    "OTHER",  # Something else
]


def load_predictions(results_dir: Path):
    """Load predictions from a results directory."""
    predictions_path = results_dir / "predictions.jsonl"

    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions not found: {predictions_path}")

    predictions = []
    with open(predictions_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))

    return predictions


def extract_hop(qid: str) -> str:
    """Extract hop count from question ID."""
    if qid.startswith("2hop"):
        return "2-hop"
    elif qid.startswith("3hop"):
        return "3-hop"
    elif qid.startswith("4hop"):
        return "4-hop"
    elif qid.startswith("5hop"):
        return "5-hop"
    return "unknown"


def classify_retrieval_pattern(pred: dict) -> str:
    """Classify retrieval pattern for hints."""
    num_retrievals = pred.get("num_retrieval_calls", 0)
    if num_retrievals <= 1:
        return "LOW_RETRIEVAL"
    elif num_retrievals >= 5:
        return "HIGH_RETRIEVAL"
    return "MED_RETRIEVAL"


def export_errors(
    predictions: list[dict],
    output_path: Path,
    threshold: float = 0.5,
    include_correct: bool = False,
):
    """Export errors to CSV for manual categorization.

    Args:
        predictions: List of prediction records
        output_path: Path to output CSV file
        threshold: F1 threshold for counting as error
        include_correct: Whether to include correctly answered questions
    """
    errors = []

    for pred in predictions:
        f1 = pred.get("f1", 0)
        is_error = f1 < threshold

        if not is_error and not include_correct:
            continue

        qid = pred.get("question_id", "")

        # Truncate long answers for readability
        predicted = pred.get("predicted_answer", "")[:200]
        gold = pred.get("gold_answer", "")[:200]

        errors.append(
            {
                "question_id": qid,
                "question_type": pred.get("question_type", "unknown"),
                "hops": extract_hop(qid),
                "predicted_answer": predicted,
                "gold_answer": gold,
                "exact_match": pred.get("exact_match", 0),
                "f1": f1,
                "num_retrieval_calls": pred.get("num_retrieval_calls", 0),
                "num_llm_calls": pred.get("num_llm_calls", 0),
                "retrieval_pattern": classify_retrieval_pattern(pred),
                "category": "",  # To be filled manually
                "notes": "",  # To be filled manually
            }
        )

    # Sort by F1 (worst first)
    errors.sort(key=lambda x: x["f1"])

    # Write to CSV
    fieldnames = [
        "question_id",
        "question_type",
        "hops",
        "predicted_answer",
        "gold_answer",
        "exact_match",
        "f1",
        "num_retrieval_calls",
        "num_llm_calls",
        "retrieval_pattern",
        "category",
        "notes",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(errors)

    return len(errors)


def print_summary(predictions: list[dict], threshold: float = 0.5):
    """Print summary statistics."""
    total = len(predictions)
    errors = sum(1 for p in predictions if p.get("f1", 0) < threshold)
    correct = total - errors

    print(f"\nTotal questions: {total}")
    print(f"Errors (F1 < {threshold}): {errors} ({100 * errors / total:.1f}%)")
    print(f"Correct: {correct} ({100 * correct / total:.1f}%)")

    # By question type
    print("\n--- By Question Type ---")
    by_type = {}
    for p in predictions:
        qt = p.get("question_type", "unknown")
        if qt not in by_type:
            by_type[qt] = {"total": 0, "errors": 0}
        by_type[qt]["total"] += 1
        if p.get("f1", 0) < threshold:
            by_type[qt]["errors"] += 1

    for qt, data in sorted(by_type.items()):
        print(
            f"  {qt}: {data['errors']}/{data['total']} errors ({100 * data['errors'] / data['total']:.1f}%)"
        )

    # By hop (for MuSiQue)
    by_hop = {}
    for p in predictions:
        qid = p.get("question_id", "")
        hops = extract_hop(qid)
        if hops not in by_hop:
            by_hop[hops] = {"total": 0, "errors": 0}
        by_hop[hops]["total"] += 1
        if p.get("f1", 0) < threshold:
            by_hop[hops]["errors"] += 1

    if by_hop:
        print("\n--- By Hop Count ---")
        for hops, data in sorted(by_hop.items()):
            print(
                f"  {hops}: {data['errors']}/{data['total']} errors ({100 * data['errors'] / data['total']:.1f}%)"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Export errors for manual categorization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results",
        required=True,
        type=Path,
        help="Path to results directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV file path (default: errors_<run_id>.csv)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="F1 threshold for counting as error (default: 0.5)",
    )
    parser.add_argument(
        "--include-correct",
        action="store_true",
        help="Also include correctly answered questions",
    )

    args = parser.parse_args()

    # Find results directory
    if (args.results / "summary.json").exists():
        results_dir = args.results
    else:
        # Try to find run directory
        run_dirs = [
            d for d in args.results.iterdir() if d.is_dir() and (d / "summary.json").exists()
        ]
        if not run_dirs:
            print("Error: No results directory found", file=sys.stderr)
            sys.exit(1)
        results_dir = sorted(run_dirs, key=lambda x: x.name)[-1]
        print(f"Using most recent run: {results_dir.name}")

    # Load predictions
    print(f"Loading predictions from {results_dir}...")
    predictions = load_predictions(results_dir)
    print(f"Loaded {len(predictions)} predictions")

    # Print summary
    print_summary(predictions, args.threshold)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = Path(f"errors_{results_dir.name}.csv")

    # Export errors
    num_errors = export_errors(
        predictions,
        output_path,
        threshold=args.threshold,
        include_correct=args.include_correct,
    )

    print(f"\nExported {num_errors} records to {output_path}")
    print("\nNow manually fill in the 'category' column using these codes:")
    for cat in CATEGORIES:
        print(f"  - {cat}")
    print("\nCategory descriptions:")
    print("  RETRIEVAL_FAIL: Didn't retrieve relevant passages (0-1 retrieval calls)")
    print("  REASONING_FAIL: Retrieved right info but reasoned incorrectly")
    print("  ENTITY_CONFUSION: Mixed up similar entities")
    print("  PREMATURE_TERMINATION: Gave up early ('no information', 'unknown')")
    print("  HALLUCINATION: Made up facts not in context")
    print("  PARTIAL_ANSWER: Answer is close but incomplete")
    print("  GROUND_TRUTH_ERROR: Gold answer appears wrong")
    print("  OTHER: Something else")


if __name__ == "__main__":
    main()
