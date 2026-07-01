#!/usr/bin/env python3
"""Deeper analysis of existing data: error correlation, complementarity, per-retriever.

Three analyses:
  1. Error correlation matrix - which architectures fail on the same questions?
  2. Complementarity / ensemble potential - oracle ensemble upper bound
  3. Per-retriever deep dive - when does BM25 beat Dense?

Usage:
    python scripts/deeper_analysis.py
    python scripts/deeper_analysis.py --dataset hotpotqa
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

FIGURES_DIR = ROOT_DIR / "results" / "figures"
OUTPUT_DIR = ROOT_DIR / "results" / "deeper_analysis"

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

ALL_RUNS = {
    "hotpotqa": {
        "Vanilla RAG": {"Dense": "bfc8f29304c5", "Hybrid": "b054e24bdec5", "BM25": "74d7b162f218"},
        "ReAct RAG": {"Dense": "47103104139a", "Hybrid": "25cc3f6b90df", "BM25": "0e7932b02fd5"},
        "Self-RAG": {"Dense": "72dc70f2c3e5", "Hybrid": "7272b4eb8c81", "BM25": "e8d5733032b2"},
        "Planner RAG": {"Dense": "b4284f7fb029", "Hybrid": "dedaa9b2bfb5", "BM25": "19114c8be979"},
        "IRCoT": {"Dense": "3e4b5fc82f99", "Hybrid": "4d923d09821d", "BM25": "1c4afb9455fd"},
        "REAP": {"Dense": "c4da1615351b", "Hybrid": "f667025aa8ab", "BM25": "0eb193182083"},
        "Recursive LM": {"Dense": "a381842dbea6", "Hybrid": "09581743885c", "BM25": "9b4f758783b7"},
    },
    "musique": {
        "Vanilla RAG": {"Dense": "e1c01e603e1b", "Hybrid": "db4e27283f1c", "BM25": "eecfa8d0669c"},
        "ReAct RAG": {"Dense": "8d4736d42905", "Hybrid": "f34150a94ab1", "BM25": "85e4a6b94230"},
        "Self-RAG": {"Dense": "1c0a526fc03e", "Hybrid": "b6cd2de95a51", "BM25": "b666a9d4f08f"},
        "Planner RAG": {"Dense": "66516cec05b6", "Hybrid": "dc3a1e2ecbfb", "BM25": "6cc7eba11f34"},
        "IRCoT": {"Dense": "51077083f3ca", "Hybrid": "202cc61ab6bb", "BM25": "d9578a21a73b"},
        "REAP": {"Dense": "842f7d8bb682", "Hybrid": "37b58fa1b536", "BM25": "26863c8883c1"},
        "Recursive LM": {"Dense": "99900bcd153f", "Hybrid": "f01337dd1fe8", "BM25": "32f5bafa00b7"},
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

ARCH_SHORT = {
    "Vanilla RAG": "Vanilla",
    "ReAct RAG": "ReAct",
    "Self-RAG": "Self-RAG",
    "Planner RAG": "Planner",
    "IRCoT": "IRCoT",
    "REAP": "REAP",
    "Recursive LM": "RLM",
}


def find_run_dir(run_id_prefix: str) -> Path | None:
    results_root = ROOT_DIR / "results"
    for d in results_root.iterdir():
        if d.is_dir() and d.name.startswith(run_id_prefix):
            return d
    return None


def load_predictions(run_id_prefix: str) -> dict[str, dict]:
    run_dir = find_run_dir(run_id_prefix)
    if run_dir is None:
        raise FileNotFoundError(f"Run not found: {run_id_prefix}")
    pp = run_dir / "predictions.jsonl"
    if not pp.exists():
        raise FileNotFoundError(f"predictions.jsonl not found in {run_dir}")
    predictions = {}
    with open(pp, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                p = json.loads(line)
                predictions[p["question_id"]] = p
    return predictions


def load_all_predictions(dataset: str) -> dict[str, dict[str, dict]]:
    all_preds = {}
    for arch in ARCH_ORDER:
        run_id = BEST_RUNS[dataset].get(arch)
        if not run_id:
            continue
        try:
            preds = load_predictions(run_id)
            all_preds[arch] = preds
            print(f"  Loaded {arch}: {len(preds)} predictions")
        except FileNotFoundError as e:
            print(f"  [WARN] {e}")
    return all_preds


# ---------------------------------------------------------------------------
# Analysis 1: Error correlation matrix
# ---------------------------------------------------------------------------


def compute_error_correlation(all_preds: dict[str, dict[str, dict]]) -> dict:
    """Compute pairwise error correlation: do architectures fail on same questions?"""
    archs = [a for a in ARCH_ORDER if a in all_preds]
    n = len(archs)

    # Get common question IDs across all architectures
    common_ids = set(all_preds[archs[0]].keys())
    for a in archs[1:]:
        common_ids &= set(all_preds[a].keys())
    common_ids = sorted(common_ids)
    print(f"  Common questions: {len(common_ids)}")

    # Build correctness matrix (1 = correct, 0 = wrong)
    correctness = {}
    for a in archs:
        em_vals = [1.0 if all_preds[a][qid]["exact_match"] >= 0.5 else 0.0 for qid in common_ids]
        correctness[a] = np.array(em_vals)

    # Compute phi correlation (binary correlation coefficient)
    corr_matrix = np.ones((n, n))
    for i, a in enumerate(archs):
        for j, b in enumerate(archs):
            if i == j:
                continue
            # Phi coefficient = Pearson correlation for binary data
            x = correctness[a]
            y = correctness[b]
            if np.std(x) > 0 and np.std(y) > 0:
                corr = np.corrcoef(x, y)[0, 1]
            else:
                corr = 0.0
            corr_matrix[i, j] = corr

    return {
        "archs": archs,
        "corr_matrix": corr_matrix,
        "correctness": correctness,
        "common_ids": common_ids,
    }


def plot_error_correlation(result: dict, dataset: str, output_dir: Path) -> Path:
    """Plot error correlation heatmap."""
    archs = result["archs"]
    n = len(archs)
    corr = result["corr_matrix"]

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap="RdYlBu_r", vmin=-0.2, vmax=1.0, aspect="auto")

    short_names = [ARCH_SHORT.get(a, a) for a in archs]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(short_names, fontsize=11)

    for i in range(n):
        for j in range(n):
            color = "white" if abs(corr[i, j]) > 0.7 else "black"
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=9, color=color)

    ax.set_title(
        f"Error Correlation Matrix ({dataset.upper()})\n"
        f"High = architectures fail on same questions (redundant)\n"
        f"Low = complementary failures (ensemble potential)",
        fontsize=12,
    )
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Phi Correlation", fontsize=10)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"fig12_error_correlation_{dataset}.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Analysis 2: Complementarity / ensemble potential
# ---------------------------------------------------------------------------


def compute_complementarity(result: dict) -> dict:
    """Compute oracle ensemble upper bound and pairwise complementarity."""
    archs = result["archs"]
    correctness = result["correctness"]
    common_ids = result["common_ids"]
    n = len(archs)
    n_q = len(common_ids)

    # Pairwise: both correct, A only, B only, both wrong
    pairwise = {}
    for i, a in enumerate(archs):
        for j, b in enumerate(archs):
            if i >= j:
                continue
            ca = correctness[a]
            cb = correctness[b]
            both_correct = int(np.sum((ca == 1) & (cb == 1)))
            a_only = int(np.sum((ca == 1) & (cb == 0)))
            b_only = int(np.sum((ca == 0) & (cb == 1)))
            both_wrong = int(np.sum((ca == 0) & (cb == 0)))
            # Oracle ensemble: correct if either is correct
            oracle = both_correct + a_only + b_only
            oracle_em = oracle / n_q
            # Complementarity score: fraction of questions where they disagree
            disagreement = (a_only + b_only) / n_q
            pairwise[f"{a} vs {b}"] = {
                "both_correct": both_correct,
                "a_only": a_only,
                "b_only": b_only,
                "both_wrong": both_wrong,
                "oracle_em": oracle_em,
                "complementarity": disagreement,
            }

    # Full oracle: correct if ANY architecture is correct
    any_correct = np.zeros(n_q)
    for a in archs:
        any_correct = np.maximum(any_correct, correctness[a])
    full_oracle_em = float(np.mean(any_correct))

    # Best single architecture
    best_arch = max(archs, key=lambda a: np.mean(correctness[a]))
    best_em = float(np.mean(correctness[best_arch]))

    # Headroom: how much could ensemble improve?
    headroom = full_oracle_em - best_em

    # Top-3 oracle (best 3 architectures)
    arch_ems = [(a, np.mean(correctness[a])) for a in archs]
    arch_ems.sort(key=lambda x: x[1], reverse=True)
    top3 = [a for a, _ in arch_ems[:3]]
    any_correct_top3 = np.zeros(n_q)
    for a in top3:
        any_correct_top3 = np.maximum(any_correct_top3, correctness[a])
    top3_oracle_em = float(np.mean(any_correct_top3))

    return {
        "pairwise": pairwise,
        "full_oracle_em": full_oracle_em,
        "best_single": best_arch,
        "best_single_em": best_em,
        "headroom": headroom,
        "top3_archs": top3,
        "top3_oracle_em": top3_oracle_em,
        "n_questions": n_q,
    }


def print_complementarity_summary(comp: dict, dataset: str):
    print(f"\n{'=' * 70}")
    print(f"COMPLEMENTARITY / ENSEMBLE POTENTIAL ({dataset.upper()})")
    print(f"{'=' * 70}")
    print(f"  Best single architecture: {comp['best_single']} ({comp['best_single_em']:.1%} EM)")
    print(f"  Top-3 oracle ensemble:    {comp['top3_oracle_em']:.1%} EM")
    print(f"  Full oracle (all 7):      {comp['full_oracle_em']:.1%} EM")
    print(f"  Ensemble headroom:        +{comp['headroom']:.1%} (best single -> oracle)")
    print()

    # Most complementary pairs
    print("  Most complementary pairs (highest disagreement):")
    pairs = sorted(comp["pairwise"].items(), key=lambda x: x[1]["complementarity"], reverse=True)
    for name, data in pairs[:5]:
        print(
            f"    {name:40s}  disagreement={data['complementarity']:.1%}  "
            f"oracle={data['oracle_em']:.1%}"
        )

    print()
    print("  Least complementary pairs (most redundant):")
    for name, data in pairs[-3:]:
        print(
            f"    {name:40s}  disagreement={data['complementarity']:.1%}  "
            f"oracle={data['oracle_em']:.1%}"
        )


# ---------------------------------------------------------------------------
# Analysis 3: Per-retriever deep dive
# ---------------------------------------------------------------------------


def compute_per_retriever_analysis(dataset: str) -> dict:
    """Analyze when BM25 beats Dense, when Hybrid wins, etc."""
    results = {}
    for arch in ARCH_ORDER:
        runs = ALL_RUNS.get(dataset, {}).get(arch, {})
        arch_data = {}
        for retriever, run_id in runs.items():
            try:
                preds = load_predictions(run_id)
                em = np.mean([p["exact_match"] for p in preds.values()])
                f1 = np.mean([p["f1"] for p in preds.values()])
                arch_data[retriever] = {"em": float(em), "f1": float(f1), "n": len(preds)}
            except FileNotFoundError:
                continue
        if arch_data:
            results[arch] = arch_data

    return results


def plot_per_retriever(retriever_data: dict, dataset: str, output_dir: Path) -> Path:
    """Plot per-retriever accuracy for each architecture."""
    archs = [a for a in ARCH_ORDER if a in retriever_data]
    n_archs = len(archs)
    retrievers = ["Dense", "Hybrid", "BM25"]

    x = np.arange(n_archs)
    width = 0.25
    colors = {"Dense": "#1f77b4", "Hybrid": "#2ca02c", "BM25": "#ff7f0e"}

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, ret in enumerate(retrievers):
        vals = []
        for a in archs:
            if ret in retriever_data[a]:
                vals.append(retriever_data[a][ret]["em"] * 100)
            else:
                vals.append(0)
        offset = (i - 1) * width
        ax.bar(
            x + offset, vals, width, color=colors[ret], edgecolor="black", linewidth=0.3, label=ret
        )

    ax.set_xticks(x)
    ax.set_xticklabels([ARCH_SHORT.get(a, a) for a in archs], fontsize=11)
    ax.set_ylabel("EM (%)", fontsize=12)
    ax.set_title(f"Per-Retriever Accuracy ({dataset.upper()})", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"fig13_per_retriever_{dataset}.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return output_path


def print_retriever_summary(retriever_data: dict, dataset: str):
    print(f"\n{'=' * 70}")
    print(f"PER-RETRIEVER DEEP DIVE ({dataset.upper()})")
    print(f"{'=' * 70}")
    print(
        f"  {'Architecture':<15s} {'Dense':>8s} {'Hybrid':>8s} {'BM25':>8s} {'Best':>8s} {'Gap':>8s}"
    )
    print(f"  {'-' * 15} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}")

    for arch in ARCH_ORDER:
        if arch not in retriever_data:
            continue
        data = retriever_data[arch]
        ems = {r: data[r]["em"] for r in ["Dense", "Hybrid", "BM25"] if r in data}
        if not ems:
            continue
        best_ret = max(ems, key=ems.get)
        worst_ret = min(ems, key=ems.get)
        gap = ems[best_ret] - ems[worst_ret]
        dense_str = f"{ems.get('Dense', 0):.1%}" if "Dense" in ems else "--"
        hybrid_str = f"{ems.get('Hybrid', 0):.1%}" if "Hybrid" in ems else "--"
        bm25_str = f"{ems.get('BM25', 0):.1%}" if "BM25" in ems else "--"
        print(
            f"  {arch:<15s} {dense_str:>8s} {hybrid_str:>8s} {bm25_str:>8s} "
            f"{best_ret:>8s} {gap:>7.1%}"
        )

    print()
    print("  Key findings:")
    # When does BM25 beat Dense?
    for arch in ARCH_ORDER:
        if arch not in retriever_data:
            continue
        data = retriever_data[arch]
        if "BM25" in data and "Dense" in data:
            if data["BM25"]["em"] > data["Dense"]["em"]:
                print(
                    f"    BM25 beats Dense for {arch}: "
                    f"{data['BM25']['em']:.1%} vs {data['Dense']['em']:.1%}"
                )


# ---------------------------------------------------------------------------
# Analysis 4: Per-question-type breakdown with retrievers
# ---------------------------------------------------------------------------


def compute_question_type_analysis(all_preds: dict, dataset: str) -> dict:
    """Break down accuracy by question type across architectures."""
    archs = [a for a in ARCH_ORDER if a in all_preds]
    results = {}

    for arch in archs:
        preds = all_preds[arch]
        types = {}
        for qid, p in preds.items():
            qt = p.get("question_type", "unknown")
            if qt not in types:
                types[qt] = {"correct": 0, "total": 0}
            types[qt]["total"] += 1
            if p["exact_match"] >= 0.5:
                types[qt]["correct"] += 1
        for qt in types:
            types[qt]["em"] = types[qt]["correct"] / types[qt]["total"]
        results[arch] = types

    return results


def print_question_type_summary(qt_data: dict, dataset: str):
    print(f"\n{'=' * 70}")
    print(f"QUESTION TYPE BREAKDOWN ({dataset.upper()})")
    print(f"{'=' * 70}")

    # Get all question types
    all_types = set()
    for arch_data in qt_data.values():
        all_types.update(arch_data.keys())
    all_types = sorted(all_types)

    header = f"  {'Architecture':<15s}"
    for qt in all_types:
        header += f" {qt:>12s}"
    print(header)
    print(f"  {'-' * 15}" + "".join([f" {'-' * 12}"] * len(all_types)))

    for arch in ARCH_ORDER:
        if arch not in qt_data:
            continue
        row = f"  {arch:<15s}"
        for qt in all_types:
            if qt in qt_data[arch]:
                row += f" {qt_data[arch][qt]['em']:>11.1%}"
            else:
                row += f" {'--':>12s}"
        print(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Deeper analysis of existing data")
    parser.add_argument("--dataset", choices=["hotpotqa", "musique", "both"], default="both")
    args = parser.parse_args()

    datasets = ["hotpotqa", "musique"] if args.dataset == "both" else [args.dataset]
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for dataset in datasets:
        print(f"\n{'=' * 70}")
        print(f"DEEPER ANALYSIS: {dataset.upper()}")
        print(f"{'=' * 70}")

        print(f"\nLoading predictions for {dataset}...")
        all_preds = load_all_predictions(dataset)
        if not all_preds:
            continue

        # Analysis 1: Error correlation
        print(f"\n[1/4] Error correlation matrix...")
        corr_result = compute_error_correlation(all_preds)
        corr_path = plot_error_correlation(corr_result, dataset, FIGURES_DIR)
        print(f"  Figure: {corr_path}")

        # Analysis 2: Complementarity
        print(f"\n[2/4] Complementarity / ensemble potential...")
        comp_result = compute_complementarity(corr_result)
        print_complementarity_summary(comp_result, dataset)

        # Analysis 3: Per-retriever
        print(f"\n[3/4] Per-retriever deep dive...")
        retriever_result = compute_per_retriever_analysis(dataset)
        ret_path = plot_per_retriever(retriever_result, dataset, FIGURES_DIR)
        print(f"  Figure: {ret_path}")
        print_retriever_summary(retriever_result, dataset)

        # Analysis 4: Question type
        print(f"\n[4/4] Question type breakdown...")
        qt_result = compute_question_type_analysis(all_preds, dataset)
        print_question_type_summary(qt_result, dataset)

        # Save results
        # Convert numpy arrays to lists for JSON
        save_result = {
            "dataset": dataset,
            "correlation": {
                "archs": corr_result["archs"],
                "matrix": corr_result["corr_matrix"].tolist(),
            },
            "complementarity": {
                "full_oracle_em": comp_result["full_oracle_em"],
                "best_single": comp_result["best_single"],
                "best_single_em": comp_result["best_single_em"],
                "headroom": comp_result["headroom"],
                "top3_archs": comp_result["top3_archs"],
                "top3_oracle_em": comp_result["top3_oracle_em"],
            },
            "per_retriever": retriever_result,
            "question_type": qt_result,
        }
        all_results[dataset] = save_result

    # Save JSON
    json_path = OUTPUT_DIR / "deeper_analysis_results.json"
    json_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\n  All results: {json_path}")

    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"  JSON:   {OUTPUT_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
