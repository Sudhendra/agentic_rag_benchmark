#!/usr/bin/env python3
"""Equal-compute Pareto analysis.

Replots the cost-performance frontier using token budget instead of dollar cost.
This answers: "if every architecture had the same compute budget, who wins?"

Two analyses:
  1. Token-vs-accuracy Pareto frontier (model-independent)
  2. Accuracy at fixed token budgets (capped compute comparison)

Usage:
    python scripts/equal_compute_analysis.py
    python scripts/equal_compute_analysis.py --dataset hotpotqa
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

FIGURES_DIR = ROOT_DIR / "results" / "figures"
OUTPUT_DIR = ROOT_DIR / "results" / "equal_compute"

BEST_RUNS = {
    "hotpotqa": {
        "Vanilla RAG": ("bfc8f29304c5", "Dense"),
        "ReAct RAG": ("25cc3f6b90df", "Hybrid"),
        "Self-RAG": ("7272b4eb8c81", "Hybrid"),
        "Planner RAG": ("b4284f7fb029", "Dense"),
        "IRCoT": ("4d923d09821d", "Hybrid"),
        "REAP": ("c4da1615351b", "Dense"),
        "Recursive LM": ("09581743885c", "Hybrid"),
    },
    "musique": {
        "Vanilla RAG": ("e1c01e603e1b", "Dense"),
        "ReAct RAG": ("8d4736d42905", "Dense"),
        "Self-RAG": ("1c0a526fc03e", "Dense"),
        "Planner RAG": ("66516cec05b6", "Dense"),
        "IRCoT": ("51077083f3ca", "Dense"),
        "REAP": ("37b58fa1b536", "Hybrid"),
        "Recursive LM": ("99900bcd153f", "Dense"),
    },
}

ARCH_ORDER = [
    "Vanilla RAG",
    "ReAct RAG",
    "Self-RAG",
    "Planner RAG",
    "IRCoT",
    "REAP",
    "Recursive LM",
]

ARCH_COLORS = {
    "Vanilla RAG": "#2ca02c",
    "ReAct RAG": "#d62728",
    "Self-RAG": "#ff7f0e",
    "Planner RAG": "#9467bd",
    "IRCoT": "#1f77b4",
    "REAP": "#8c564b",
    "Recursive LM": "#e377c2",
}

ARCH_MARKERS = {
    "Vanilla RAG": "o",
    "ReAct RAG": "s",
    "Self-RAG": "D",
    "Planner RAG": "v",
    "IRCoT": "^",
    "REAP": "X",
    "Recursive LM": "P",
}


def find_run_dir(run_id_prefix: str) -> Path | None:
    results_root = ROOT_DIR / "results"
    for d in results_root.iterdir():
        if d.is_dir() and d.name.startswith(run_id_prefix):
            return d
    return None


def load_per_question_data(run_id_prefix: str) -> list[dict]:
    """Load per-question token usage and correctness."""
    run_dir = find_run_dir(run_id_prefix)
    if run_dir is None:
        raise FileNotFoundError(f"Run directory not found for {run_id_prefix}")

    pred_path = run_dir / "predictions.jsonl"
    if not pred_path.exists():
        raise FileNotFoundError(f"predictions.jsonl not found in {run_dir}")

    predictions = []
    with open(pred_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))
    return predictions


def load_all_data(dataset: str) -> dict[str, list[dict]]:
    """Load per-question data for all architectures."""
    all_data = {}
    for arch in ARCH_ORDER:
        run_info = BEST_RUNS[dataset].get(arch)
        if run_info is None:
            continue
        run_id, retriever = run_info
        try:
            data = load_per_question_data(run_id)
            all_data[arch] = data
            print(f"  Loaded {arch}: {len(data)} predictions ({run_id}, {retriever})")
        except FileNotFoundError as e:
            print(f"  [WARN] {e}")
    return all_data


# ---------------------------------------------------------------------------
# Analysis 1: Token-vs-accuracy Pareto frontier
# ---------------------------------------------------------------------------


def compute_token_pareto(all_data: dict[str, list[dict]]) -> dict:
    """Compute (avg_tokens, EM) for each architecture."""
    results = {}
    for arch, preds in all_data.items():
        avg_tokens = np.mean([p["tokens_used"] for p in preds])
        em = np.mean([p["exact_match"] for p in preds])
        f1 = np.mean([p["f1"] for p in preds])
        results[arch] = {
            "avg_tokens": float(avg_tokens),
            "em": float(em),
            "f1": float(f1),
        }
    return results


def plot_token_pareto(pareto_data: dict, dataset: str, output_dir: Path) -> Path:
    """Plot accuracy vs token budget (model-independent Pareto)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for arch in ARCH_ORDER:
        if arch not in pareto_data:
            continue
        d = pareto_data[arch]
        ax1.scatter(
            d["avg_tokens"],
            d["em"] * 100,
            c=ARCH_COLORS[arch],
            marker=ARCH_MARKERS[arch],
            s=150,
            zorder=5,
            edgecolors="black",
            linewidth=0.5,
            label=arch,
        )
        ax2.scatter(
            d["avg_tokens"],
            d["f1"] * 100,
            c=ARCH_COLORS[arch],
            marker=ARCH_MARKERS[arch],
            s=150,
            zorder=5,
            edgecolors="black",
            linewidth=0.5,
            label=arch,
        )

    for ax, metric in [(ax1, "EM"), (ax2, "F1")]:
        ax.set_xlabel("Avg Tokens per Question", fontsize=12)
        ax.set_ylabel(f"{metric} (%)", fontsize=12)
        ax.set_title(f"{metric} vs Token Budget -- {dataset.upper()}", fontsize=13)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"fig10_equal_compute_pareto_{dataset}.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Analysis 2: Accuracy at fixed token budgets
# ---------------------------------------------------------------------------


def compute_accuracy_at_budgets(
    all_data: dict[str, list[dict]],
    budgets: list[int] | None = None,
) -> dict:
    """For each architecture, compute accuracy at various token budgets.

    For budget B: EM = fraction of questions where tokens_used <= B AND correct.
    This represents "what accuracy would you get if you capped compute at B tokens?"
    """
    if budgets is None:
        # Use Vanilla's avg tokens as reference point
        vanilla_tokens = np.mean([p["tokens_used"] for p in all_data["Vanilla RAG"]])
        budgets = sorted(
            set(
                [
                    500,
                    750,
                    1000,
                    int(vanilla_tokens),
                    2000,
                    3000,
                    5000,
                    8000,
                    12000,
                ]
            )
        )

    results = {"budgets": budgets, "architectures": {}}

    for arch, preds in all_data.items():
        arch_data = {"em_at_budget": [], "f1_at_budget": [], "coverage": []}
        for budget in budgets:
            # Questions that fit within budget
            fitting = [p for p in preds if p["tokens_used"] <= budget]
            coverage = len(fitting) / len(preds) if preds else 0
            if fitting:
                em = np.mean([p["exact_match"] for p in fitting])
                f1 = np.mean([p["f1"] for p in fitting])
            else:
                em = 0.0
                f1 = 0.0
            arch_data["em_at_budget"].append(float(em))
            arch_data["f1_at_budget"].append(float(f1))
            arch_data["coverage"].append(float(coverage))
        results["architectures"][arch] = arch_data

    return results


def plot_accuracy_at_budgets(budget_data: dict, dataset: str, output_dir: Path) -> Path:
    """Plot accuracy vs token budget curves for all architectures."""
    fig, ax = plt.subplots(figsize=(12, 7))

    budgets = budget_data["budgets"]

    for arch in ARCH_ORDER:
        if arch not in budget_data["architectures"]:
            continue
        arch_data = budget_data["architectures"][arch]
        ax.plot(
            budgets,
            [em * 100 for em in arch_data["em_at_budget"]],
            c=ARCH_COLORS[arch],
            marker=ARCH_MARKERS[arch],
            markersize=7,
            linewidth=2,
            label=arch,
        )

    # Mark Vanilla's token budget with a vertical line
    vanilla_data = budget_data["architectures"].get("Vanilla RAG", {})
    if vanilla_data:
        # Find the budget closest to Vanilla's avg tokens
        vanilla_idx = max(
            range(len(budgets)),
            key=lambda i: budgets[i] if vanilla_data["coverage"][i] > 0.9 else 0,
        )
        ax.axvline(
            budgets[vanilla_idx],
            color="green",
            linestyle="--",
            alpha=0.5,
            label=f"Vanilla budget ({budgets[vanilla_idx]} tokens)",
        )

    ax.set_xlabel("Token Budget per Question", fontsize=12)
    ax.set_ylabel("EM (%) on Questions Within Budget", fontsize=12)
    ax.set_title(
        f"Accuracy at Equal Compute -- {dataset.upper()}\n"
        f'"If every architecture had the same token budget, who wins?"',
        fontsize=13,
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"fig10b_accuracy_at_budget_{dataset}.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Analysis 3: Who wins at Vanilla's compute?
# ---------------------------------------------------------------------------


def compute_equal_compute_winner(all_data: dict[str, list[dict]]) -> dict:
    """Compute accuracy when all architectures are restricted to Vanilla's budget."""
    vanilla_tokens = np.mean([p["tokens_used"] for p in all_data["Vanilla RAG"]])
    budget = int(vanilla_tokens)

    results = {"vanilla_budget": budget, "architectures": {}}

    for arch, preds in all_data.items():
        fitting = [p for p in preds if p["tokens_used"] <= budget]
        coverage = len(fitting) / len(preds)
        if fitting:
            em = np.mean([p["exact_match"] for p in fitting])
            f1 = np.mean([p["f1"] for p in fitting])
        else:
            em = 0.0
            f1 = 0.0
        results["architectures"][arch] = {
            "em": float(em),
            "f1": float(f1),
            "coverage": float(coverage),
            "n_fitting": len(fitting),
        }

    return results


def print_equal_compute_summary(winner_data: dict, dataset: str):
    """Print summary of equal-compute comparison."""
    budget = winner_data["vanilla_budget"]
    print(f"\n{'=' * 70}")
    print(f"EQUAL-COMPUTE ANALYSIS -- {dataset.upper()}")
    print(f"{'=' * 70}")
    print(f"  Vanilla's budget: {budget} tokens/question")
    print()
    print(f"  {'Architecture':<15s} {'EM':>8s} {'F1':>8s} {'Coverage':>10s} {'n_fit':>6s}")
    print(f"  {'-' * 15} {'-' * 8} {'-' * 8} {'-' * 10} {'-' * 6}")

    sorted_archs = sorted(
        winner_data["architectures"].items(),
        key=lambda x: x[1]["em"],
        reverse=True,
    )
    for arch, data in sorted_archs:
        print(
            f"  {arch:<15s} {data['em']:>7.1%} {data['f1']:>7.1%} "
            f"{data['coverage']:>9.1%} {data['n_fitting']:>6d}"
        )

    winner = sorted_archs[0]
    print(f"\n  Winner at equal compute: {winner[0]} ({winner[1]['em']:.1%} EM)")
    print(f"  (Only {winner[1]['coverage']:.0%} of questions fit within budget)")

    # Key insight
    vanilla_em = winner_data["architectures"]["Vanilla RAG"]["em"]
    react_em = winner_data["architectures"].get("ReAct RAG", {}).get("em", 0)
    print(f"\n  Key insight: At {budget} tokens/question:")
    print(f"    Vanilla RAG: {vanilla_em:.1%} EM (100% coverage)")
    if react_em is not None:
        print(f"    ReAct RAG:   {react_em:.1%} EM (restricted to questions that fit)")
        if vanilla_em > react_em:
            print(f"    -> Vanilla BEATS ReAct at equal compute!")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_json(
    pareto_data: dict, budget_data: dict, winner_data: dict, dataset: str, output_dir: Path
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "dataset": dataset,
        "pareto": pareto_data,
        "accuracy_at_budgets": budget_data,
        "equal_compute_winner": winner_data,
    }
    output_path = output_dir / f"equal_compute_{dataset}.json"
    output_path.write_text(json.dumps(output, indent=2))
    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Equal-compute Pareto analysis",
    )
    parser.add_argument(
        "--dataset",
        choices=["hotpotqa", "musique", "both"],
        default="both",
    )
    args = parser.parse_args()

    datasets = ["hotpotqa", "musique"] if args.dataset == "both" else [args.dataset]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        print(f"\n{'-' * 60}")
        print(f"Loading data for {dataset.upper()}...")
        print(f"{'-' * 60}")
        all_data = load_all_data(dataset)
        if not all_data:
            continue

        # Analysis 1: Token Pareto
        pareto_data = compute_token_pareto(all_data)
        pareto_path = plot_token_pareto(pareto_data, dataset, FIGURES_DIR)
        print(f"\n  Pareto plot: {pareto_path}")

        # Analysis 2: Accuracy at budgets
        budget_data = compute_accuracy_at_budgets(all_data)
        budget_path = plot_accuracy_at_budgets(budget_data, dataset, FIGURES_DIR)
        print(f"  Budget plot: {budget_path}")

        # Analysis 3: Equal-compute winner
        winner_data = compute_equal_compute_winner(all_data)
        print_equal_compute_summary(winner_data, dataset)

        # Export
        json_path = export_json(pareto_data, budget_data, winner_data, dataset, OUTPUT_DIR)
        print(f"  JSON: {json_path}")

    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"  JSON:   {OUTPUT_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
