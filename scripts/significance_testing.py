#!/usr/bin/env python3
"""Statistical significance testing across architecture pairs.

Computes:
  1. McNemar's test for EM (binary correct/incorrect per question)
  2. Paired bootstrap for F1 differences
  3. Bonferroni correction across all pairwise comparisons

Generates:
  - 7x7 significance heatmap (HotpotQA + MuSiQue)
  - CSV with all p-values and significance flags
  - Summary table for the paper

Usage:
    python scripts/significance_testing.py
    python scripts/significance_testing.py --dataset hotpotqa
    python scripts/significance_testing.py --n-bootstrap 50000
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

FIGURES_DIR = ROOT_DIR / "results" / "figures"
OUTPUT_DIR = ROOT_DIR / "results" / "significance"

# ---------------------------------------------------------------------------
# Best run per architecture (matching paper Table 1)
# ---------------------------------------------------------------------------

BEST_RUNS = {
    "hotpotqa": {
        "Vanilla RAG": "bfc8f29304c5",
        "ReAct RAG": "25cc3f6b90df",
        "Self-RAG": "7272b4eb8c81",
        "Planner RAG": "b4284f7fb029",
        "IRCoT": "4d923d09821d",
        "REAP": "c4da1615351b",
        "Recursive LM": "09581743885c",
    },
    "musique": {
        "Vanilla RAG": "e1c01e603e1b",
        "ReAct RAG": "8d4736d42905",
        "Self-RAG": "1c0a526fc03e",
        "Planner RAG": "66516cec05b6",
        "IRCoT": "51077083f3ca",
        "REAP": "37b58fa1b536",
        "Recursive LM": "99900bcd153f",
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def find_run_dir(run_id_prefix: str) -> Path | None:
    results_root = ROOT_DIR / "results"
    for d in results_root.iterdir():
        if d.is_dir() and d.name.startswith(run_id_prefix):
            return d
    return None


def load_predictions(run_id_prefix: str) -> dict[str, dict]:
    """Load predictions.jsonl, return dict keyed by question_id."""
    run_dir = find_run_dir(run_id_prefix)
    if run_dir is None:
        raise FileNotFoundError(f"Run directory not found for {run_id_prefix}")

    pred_path = run_dir / "predictions.jsonl"
    if not pred_path.exists():
        raise FileNotFoundError(f"predictions.jsonl not found in {run_dir}")

    predictions = {}
    with open(pred_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                p = json.loads(line)
                predictions[p["question_id"]] = p
    return predictions


def load_all_predictions(dataset: str) -> dict[str, dict[str, dict]]:
    """Load predictions for all architectures on a dataset.
    Returns {arch_name: {question_id: prediction_dict}}.
    """
    all_preds = {}
    for arch in ARCH_ORDER:
        run_id = BEST_RUNS[dataset].get(arch)
        if run_id is None:
            print(f"  [WARN] No run ID for {arch} on {dataset}")
            continue
        try:
            preds = load_predictions(run_id)
            all_preds[arch] = preds
            print(f"  Loaded {arch}: {len(preds)} predictions ({run_id})")
        except FileNotFoundError as e:
            print(f"  [WARN] {e}")
    return all_preds


# ---------------------------------------------------------------------------
# McNemar's test
# ---------------------------------------------------------------------------


def mcnemar_test(preds_a: dict, preds_b: dict) -> tuple[float, float]:
    """Compute McNemar's test for two architectures.

    Contingency table:
      b = A correct, B wrong
      c = A wrong, B correct

    Returns (chi2_statistic, p_value).
    """
    common_ids = set(preds_a.keys()) & set(preds_b.keys())
    b = 0  # A correct, B wrong
    c = 0  # A wrong, B correct
    for qid in common_ids:
        a_correct = preds_a[qid]["exact_match"] >= 0.5
        b_correct = preds_b[qid]["exact_match"] >= 0.5
        if a_correct and not b_correct:
            b += 1
        elif not a_correct and b_correct:
            c += 1

    # McNemar's test with continuity correction
    if b + c == 0:
        return 0.0, 1.0
    chi2_val: float = float((abs(b - c) - 1) ** 2 / (b + c))
    p_raw = stats.chi2.sf(chi2_val, df=1)
    p_value: float = float(p_raw)
    return chi2_val, p_value


# ---------------------------------------------------------------------------
# Paired bootstrap for F1
# ---------------------------------------------------------------------------


def paired_bootstrap_f1(
    preds_a: dict,
    preds_b: dict,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> tuple[float, float]:
    """Paired bootstrap test for F1 difference.

    Returns (observed_diff, p_value) where p_value is the proportion of
    bootstrap samples where the difference is <= 0 (for one-sided test
    that A > B).
    """
    common_ids = sorted(set(preds_a.keys()) & set(preds_b.keys()))
    n = len(common_ids)
    if n == 0:
        return 0.0, 1.0

    f1_a = np.array([preds_a[qid]["f1"] for qid in common_ids])
    f1_b = np.array([preds_b[qid]["f1"] for qid in common_ids])

    observed_diff = np.mean(f1_a) - np.mean(f1_b)

    rng = np.random.default_rng(seed)
    count_extreme = 0
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        boot_diff = np.mean(f1_a[indices]) - np.mean(f1_b[indices])
        # Two-sided: count how often |boot_diff| >= |observed_diff| under null
        # Under null, the diff should be centered at 0
        if abs(boot_diff - 0) >= abs(observed_diff):
            count_extreme += 1

    p_value = (count_extreme + 1) / (n_bootstrap + 1)
    return float(observed_diff), float(p_value)


# ---------------------------------------------------------------------------
# Compute full significance matrix
# ---------------------------------------------------------------------------


def compute_significance_matrix(
    all_preds: dict[str, dict[str, dict]],
    n_bootstrap: int = 10000,
) -> dict:
    """Compute pairwise significance for all architecture pairs."""
    n_archs = len(ARCH_ORDER)
    archs = [a for a in ARCH_ORDER if a in all_preds]

    mcnemar_matrix = np.ones((len(archs), len(archs)))
    mcnemar_diff = np.zeros((len(archs), len(archs)))
    bootstrap_matrix = np.ones((len(archs), len(archs)))
    bootstrap_diff = np.zeros((len(archs), len(archs)))

    n_comparisons = len(archs) * (len(archs) - 1) // 2
    bonferroni_alpha = 0.05 / n_comparisons

    for i, arch_a in enumerate(archs):
        for j, arch_b in enumerate(archs):
            if i == j:
                continue
            if i > j:
                # Use symmetric results
                mcnemar_matrix[i, j] = mcnemar_matrix[j, i]
                mcnemar_diff[i, j] = -mcnemar_diff[j, i]
                bootstrap_matrix[i, j] = bootstrap_matrix[j, i]
                bootstrap_diff[i, j] = -bootstrap_diff[j, i]
                continue

            preds_a = all_preds[arch_a]
            preds_b = all_preds[arch_b]

            # McNemar
            chi2, p_mcnemar = mcnemar_test(preds_a, preds_b)
            mcnemar_matrix[i, j] = p_mcnemar
            mcnemar_matrix[j, i] = p_mcnemar

            # Compute EM difference for annotation
            common = set(preds_a.keys()) & set(preds_b.keys())
            em_a = np.mean([preds_a[qid]["exact_match"] for qid in common])
            em_b = np.mean([preds_b[qid]["exact_match"] for qid in common])
            mcnemar_diff[i, j] = em_a - em_b
            mcnemar_diff[j, i] = em_b - em_a

            # Paired bootstrap
            f1_diff, p_boot = paired_bootstrap_f1(preds_a, preds_b, n_bootstrap=n_bootstrap)
            bootstrap_matrix[i, j] = p_boot
            bootstrap_matrix[j, i] = p_boot
            bootstrap_diff[i, j] = f1_diff
            bootstrap_diff[j, i] = -f1_diff

            sig_mcn = (
                "***"
                if p_mcnemar < bonferroni_alpha
                else ("**" if p_mcnemar < 0.01 else ("*" if p_mcnemar < 0.05 else "ns"))
            )
            sig_boot = (
                "***"
                if p_boot < bonferroni_alpha
                else ("**" if p_boot < 0.01 else ("*" if p_boot < 0.05 else "ns"))
            )

            print(
                f"  {arch_a:15s} vs {arch_b:15s}  "
                f"McNemar p={p_mcnemar:.4e} {sig_mcn}  "
                f"Bootstrap p={p_boot:.4e} {sig_boot}"
            )

    return {
        "archs": archs,
        "mcnemar_pvalues": mcnemar_matrix,
        "mcnemar_diff": mcnemar_diff,
        "bootstrap_pvalues": bootstrap_matrix,
        "bootstrap_diff": bootstrap_diff,
        "n_comparisons": n_comparisons,
        "bonferroni_alpha": bonferroni_alpha,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_significance_heatmap(
    result: dict,
    dataset: str,
    output_dir: Path,
) -> Path:
    """Generate significance heatmap figure."""
    archs = result["archs"]
    n = len(archs)
    pvals = result["mcnemar_pvalues"]
    diffs = result["mcnemar_diff"]
    bonf_alpha = result["bonferroni_alpha"]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Use -log10(p-value) for color scale
    log_pvals = -np.log10(np.clip(pvals, 1e-300, 1.0))

    im = ax.imshow(log_pvals, cmap="YlOrRd", aspect="auto", vmin=0, vmax=10)

    # Labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(archs, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(archs, fontsize=10)

    # Annotate each cell with p-value and significance
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "--", ha="center", va="center", fontsize=9, color="gray")
                continue
            p = pvals[i, j]
            diff = diffs[i, j]
            if p < bonf_alpha:
                sig = "***"
                color = "white"
            elif p < 0.01:
                sig = "**"
                color = "white"
            elif p < 0.05:
                sig = "*"
                color = "black"
            else:
                sig = "ns"
                color = "black"

            if p < 0.001:
                p_str = f"{p:.1e}"
            else:
                p_str = f"{p:.3f}"

            ax.text(
                j,
                i,
                f"{p_str}\n{sig}",
                ha="center",
                va="center",
                fontsize=7,
                color=color,
            )

    ax.set_title(
        f"Pairwise Significance (McNemar's Test) -- {dataset.upper()}\n"
        f"*** = significant after Bonferroni (alpha={bonf_alpha:.4f}),  "
        f"** = p<0.01,  * = p<0.05",
        fontsize=11,
    )

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("-log₁₀(p-value)", fontsize=10)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"fig9_significance_matrix_{dataset}.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


def export_csv(result: dict, dataset: str, output_dir: Path) -> Path:
    """Export significance results to CSV."""
    archs = result["archs"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"significance_{dataset}.csv"

    rows = []
    for i, arch_a in enumerate(archs):
        for j, arch_b in enumerate(archs):
            if i >= j:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "architecture_a": arch_a,
                    "architecture_b": arch_b,
                    "mcnemar_pvalue": result["mcnemar_pvalues"][i, j],
                    "em_diff_a_minus_b": result["mcnemar_diff"][i, j],
                    "bootstrap_pvalue": result["bootstrap_pvalues"][i, j],
                    "f1_diff_a_minus_b": result["bootstrap_diff"][i, j],
                    "bonferroni_alpha": result["bonferroni_alpha"],
                    "significant_bonferroni": result["mcnemar_pvalues"][i, j]
                    < result["bonferroni_alpha"],
                    "significant_0.05": result["mcnemar_pvalues"][i, j] < 0.05,
                }
            )

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    return output_path


# ---------------------------------------------------------------------------
# Paper-ready summary
# ---------------------------------------------------------------------------


def print_paper_summary(result: dict, dataset: str):
    """Print a summary suitable for the paper."""
    archs = result["archs"]
    bonf_alpha = result["bonferroni_alpha"]
    n = len(archs)

    print(f"\n{'=' * 70}")
    print(f"SIGNIFICANCE SUMMARY -- {dataset.upper()}")
    print(f"{'=' * 70}")
    print(f"  Comparisons: {result['n_comparisons']}")
    print(f"  Bonferroni alpha: {bonf_alpha:.4f}")
    print()

    sig_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            p = result["mcnemar_pvalues"][i, j]
            diff = result["mcnemar_diff"][i, j]
            if p < bonf_alpha:
                sig_count += 1
                direction = ">" if diff > 0 else "<"
                print(
                    f"  {archs[i]:15s} {direction} {archs[j]:15s}  Delta={abs(diff):.1%}  p={p:.2e} ***"
                )

    nonsig = result["n_comparisons"] - sig_count
    print(f"\n  Significant after Bonferroni: {sig_count}/{result['n_comparisons']}")
    print(f"  Not significant: {nonsig}/{result['n_comparisons']}")
    print()

    # Key comparisons for the paper
    print("  Key comparisons for paper:")
    key_pairs = [
        ("Vanilla RAG", "ReAct RAG"),
        ("Vanilla RAG", "Recursive LM"),
        ("Vanilla RAG", "IRCoT"),
        ("IRCoT", "ReAct RAG"),
        ("IRCoT", "Recursive LM"),
        ("Recursive LM", "ReAct RAG"),
    ]
    for a, b in key_pairs:
        if a in archs and b in archs:
            i, j = archs.index(a), archs.index(b)
            p = result["mcnemar_pvalues"][i, j]
            diff = result["mcnemar_diff"][i, j]
            sig = "significant" if p < bonf_alpha else "NOT significant"
            print(f"    {a} vs {b}: Delta={diff:+.1%} EM, p={p:.2e} -> {sig}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Statistical significance testing across architecture pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=["hotpotqa", "musique", "both"],
        default="both",
        help="Dataset to analyze",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10000,
        help="Number of bootstrap iterations (default: 10000)",
    )
    args = parser.parse_args()

    datasets = ["hotpotqa", "musique"] if args.dataset == "both" else [args.dataset]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        print(f"\n{'-' * 60}")
        print(f"Loading predictions for {dataset.upper()}...")
        print(f"{'-' * 60}")
        all_preds = load_all_predictions(dataset)
        if len(all_preds) < 2:
            print(f"[ERROR] Need at least 2 architectures, got {len(all_preds)}")
            continue

        print(f"\nComputing significance ({len(all_preds)} architectures)...")
        result = compute_significance_matrix(all_preds, args.n_bootstrap)

        # Visualize
        fig_path = plot_significance_heatmap(result, dataset, FIGURES_DIR)
        print(f"\n  Heatmap: {fig_path}")

        # Export CSV
        csv_path = export_csv(result, dataset, OUTPUT_DIR)
        print(f"  CSV:     {csv_path}")

        # Paper summary
        print_paper_summary(result, dataset)

    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"  CSVs:    {OUTPUT_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
