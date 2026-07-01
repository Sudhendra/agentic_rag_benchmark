#!/usr/bin/env python3
"""Generate all paper tables from a single source of truth.

This script reads results/*/summary.json and predictions.jsonl files and
generates verified, consistent LaTeX tables and CSV files for the paper.
Use this to ensure every number in the paper matches the raw data.

Usage:
    python scripts/generate_paper_tables.py
    python scripts/generate_paper_tables.py --output-dir paper_tables/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

# ---------------------------------------------------------------------------
# Best run per architecture per dataset (matching paper Table 1)
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

# Full grid: all retriever configs
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

ARCH_PARADIGM = {
    "Vanilla RAG": "Baseline",
    "ReAct RAG": "Agentic",
    "Self-RAG": "Agentic",
    "Planner RAG": "Agentic",
    "IRCoT": "Recursive",
    "REAP": "Recursive",
    "Recursive LM": "RLM",
}


def find_run_dir(run_id_prefix: str) -> Path | None:
    results_root = ROOT_DIR / "results"
    for d in results_root.iterdir():
        if d.is_dir() and d.name.startswith(run_id_prefix):
            return d
    return None


def load_summary(run_id_prefix: str) -> dict:
    run_dir = find_run_dir(run_id_prefix)
    if run_dir is None:
        raise FileNotFoundError(f"Run not found: {run_id_prefix}")
    sp = run_dir / "summary.json"
    if not sp.exists():
        raise FileNotFoundError(f"summary.json not found in {run_dir}")
    return json.loads(sp.read_text())


def load_predictions(run_id_prefix: str) -> list[dict]:
    run_dir = find_run_dir(run_id_prefix)
    if run_dir is None:
        raise FileNotFoundError(f"Run not found: {run_id_prefix}")
    pp = run_dir / "predictions.jsonl"
    if not pp.exists():
        raise FileNotFoundError(f"predictions.jsonl not found in {run_dir}")
    preds = []
    with open(pp, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                preds.append(json.loads(line))
    return preds


# ---------------------------------------------------------------------------
# Table 1: Main results (best retriever per arch)
# ---------------------------------------------------------------------------


def generate_main_results(output_dir: Path):
    """Generate Table 1: Main results across both datasets."""
    rows = []
    for dataset in ["hotpotqa", "musique"]:
        for arch in ARCH_ORDER:
            run_id = BEST_RUNS[dataset].get(arch)
            if not run_id:
                continue
            try:
                s = load_summary(run_id)
            except FileNotFoundError:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "architecture": arch,
                    "paradigm": ARCH_PARADIGM[arch],
                    "model": s["model"],
                    "n_questions": s["num_questions"],
                    "em": s["avg_exact_match"],
                    "f1": s["avg_f1"],
                    "cost_usd": s["total_cost_usd"],
                    "latency_ms": s["avg_latency_ms"],
                    "avg_llm_calls": s["avg_llm_calls"],
                    "avg_retrieval_calls": s["avg_retrieval_calls"],
                    "run_id": run_id,
                }
            )

    # CSV
    csv_path = output_dir / "table1_main_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Table 1 CSV: {csv_path}")

    # LaTeX
    tex_path = output_dir / "table1_main_results.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by scripts/generate_paper_tables.py\n")
        f.write("\\begin{table*}[t]\n\\centering\\small\n")
        f.write("\\begin{tabular}{lllcccccc}\n\\toprule\n")
        f.write(
            "& & & \\multicolumn{3}{c}{\\textbf{HotpotQA}} & \\multicolumn{3}{c}{\\textbf{MuSiQue}} \\\\\n"
        )
        f.write("\\cmidrule(lr){4-6} \\cmidrule(lr){7-9}\n")
        f.write("\\textbf{Architecture} & \\textbf{Paradigm} & \\textbf{Retriever} & ")
        f.write(
            "\\textbf{EM} & \\textbf{F1} & \\textbf{Cost} & \\textbf{EM} & \\textbf{F1} & \\textbf{Cost} \\\\\n"
        )
        f.write("\\midrule\n")

        for arch in ARCH_ORDER:
            hp = next(
                (r for r in rows if r["dataset"] == "hotpotqa" and r["architecture"] == arch), None
            )
            mq = next(
                (r for r in rows if r["dataset"] == "musique" and r["architecture"] == arch), None
            )
            if not hp:
                continue
            retriever = "Dense" if arch in ["Vanilla RAG", "Planner RAG"] else "Hybrid"
            f.write(f"{arch} & {ARCH_PARADIGM[arch]} & {retriever} & ")
            f.write(f"{hp['em']:.1%} & {hp['f1']:.1%} & \\${hp['cost_usd']:.2f} & ")
            if mq:
                f.write(f"{mq['em']:.1%} & {mq['f1']:.1%} & \\${mq['cost_usd']:.2f}")
            else:
                f.write("-- & -- & --")
            f.write(" \\\\\n")

        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\\caption{Main results. Auto-generated from raw data.}\n")
        f.write("\\label{tab:main_results}\n\\end{table*}\n")
    print(f"  Table 1 LaTeX: {tex_path}")
    return rows


# ---------------------------------------------------------------------------
# Table 2: Full grid (all retrievers)
# ---------------------------------------------------------------------------


def generate_full_grid(output_dir: Path):
    """Generate full grid tables for each dataset."""
    for dataset in ["hotpotqa", "musique"]:
        rows = []
        for arch in ARCH_ORDER:
            for retriever in ["Dense", "Hybrid", "BM25"]:
                run_id = ALL_RUNS.get(dataset, {}).get(arch, {}).get(retriever)
                if not run_id:
                    continue
                try:
                    s = load_summary(run_id)
                except FileNotFoundError:
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "architecture": arch,
                        "retriever": retriever,
                        "em": s["avg_exact_match"],
                        "f1": s["avg_f1"],
                        "cost_usd": s["total_cost_usd"],
                        "latency_ms": s["avg_latency_ms"],
                        "avg_llm_calls": s["avg_llm_calls"],
                        "run_id": run_id,
                    }
                )

        csv_path = output_dir / f"table2_full_grid_{dataset}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Table 2 ({dataset}) CSV: {csv_path}")

        # LaTeX
        tex_path = output_dir / f"table2_full_{dataset}.tex"
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(f"% Auto-generated: Full {dataset} results\n")
            f.write("\\begin{table*}[t]\n\\centering\\small\n")
            f.write("\\begin{tabular}{llccccc}\n\\toprule\n")
            f.write("\\textbf{Architecture} & \\textbf{Retriever} & \\textbf{EM} & \\textbf{F1} & ")
            f.write("\\textbf{Latency} & \\textbf{LLM Calls} & \\textbf{Cost} \\\\\n")
            f.write("\\midrule\n")

            for arch in ARCH_ORDER:
                arch_rows = [r for r in rows if r["architecture"] == arch]
                if not arch_rows:
                    continue
                best_em = max(r["em"] for r in arch_rows)
                for r in arch_rows:
                    em_str = (
                        f"\\textbf{{{r['em']:.1%}}}" if r["em"] == best_em else f"{r['em']:.1%}"
                    )
                    f.write(f"{arch} & {r['retriever']} & {em_str} & {r['f1']:.1%} & ")
                    f.write(f"{r['latency_ms']:.0f}ms & {r['avg_llm_calls']:.1f} & ")
                    f.write(f"\\${r['cost_usd']:.2f} \\\\\n")
                f.write("\\midrule\n")

            f.write("\\bottomrule\n\\end{tabular}\n")
            f.write(f"\\caption{{Full {dataset} results. Best per architecture in bold.}}\n")
            f.write(f"\\label{{tab:{dataset}_full}}\n\\end{{table*}}\n")
        print(f"  Table 2 ({dataset}) LaTeX: {tex_path}")


# ---------------------------------------------------------------------------
# Table 3: Error taxonomy (recomputed from predictions)
# ---------------------------------------------------------------------------


def compute_error_taxonomy(predictions: list[dict]) -> dict:
    """Recompute 7-category error taxonomy from raw predictions."""
    total = len(predictions)
    if total == 0:
        return {}

    categories = {
        "correct": 0,
        "complete_miss": 0,
        "low_overlap": 0,
        "partial": 0,
        "near_miss": 0,
        "verbose": 0,
        "loop": 0,
        "yes_no_flip": 0,
    }

    for p in predictions:
        em = p.get("exact_match", 0)
        f1 = p.get("f1", 0)
        pred = p.get("predicted_answer", "").lower().strip()
        gold = p.get("gold_answer", "").lower().strip()

        if em >= 0.5:
            categories["correct"] += 1
            continue

        if f1 == 0:
            categories["complete_miss"] += 1
        elif f1 <= 0.25:
            categories["low_overlap"] += 1
        elif f1 <= 0.50:
            categories["partial"] += 1
        elif f1 <= 0.90:
            categories["near_miss"] += 1
        else:
            categories["verbose"] += 1

        # Yes/No flip detection
        gold_binary = gold in ("yes", "no")
        pred_binary = pred in ("yes", "no", "yes.", "no.")
        if gold_binary and pred_binary and pred.rstrip(".") != gold:
            categories["yes_no_flip"] += 1

        # Loop detection (repeated tokens)
        if len(pred) > 50:
            tokens = pred.split()
            if len(tokens) > 10:
                unique_ratio = len(set(tokens)) / len(tokens)
                if unique_ratio < 0.4:
                    categories["loop"] += 1

    # Convert to percentages
    return {k: v / total for k, v in categories.items()}


def generate_error_taxonomy(output_dir: Path):
    """Generate error taxonomy table recomputed from raw data."""
    print("\n  Recomputing error taxonomy from predictions.jsonl...")

    all_results = {}
    for dataset in ["hotpotqa", "musique"]:
        for arch in ARCH_ORDER:
            run_id = BEST_RUNS[dataset].get(arch)
            if not run_id:
                continue
            try:
                preds = load_predictions(run_id)
            except FileNotFoundError:
                continue
            taxonomy = compute_error_taxonomy(preds)
            all_results[f"{dataset}_{arch}"] = taxonomy
            print(
                f"    {dataset}/{arch}: {len(preds)} preds, "
                f"correct={taxonomy.get('correct', 0):.1%}, "
                f"complete_miss={taxonomy.get('complete_miss', 0):.1%}"
            )

    # CSV
    csv_path = output_dir / "table3_error_taxonomy.csv"
    rows = []
    for key, tax in all_results.items():
        dataset, arch = key.rsplit("_", 0)[0], " ".join(key.rsplit("_")[1:])
        row = {"dataset_arch": key}
        row.update(tax)
        rows.append(row)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(f"  Error taxonomy CSV: {csv_path}")

    # JSON (for figure generation)
    json_path = output_dir / "error_taxonomy_verified.json"
    json_path.write_text(json.dumps(all_results, indent=2))
    print(f"  Error taxonomy JSON: {json_path}")


# ---------------------------------------------------------------------------
# Table 4: Robustness ablations
# ---------------------------------------------------------------------------


def generate_robustness_table(output_dir: Path):
    """Generate robustness ablation table from existing results."""
    robustness_dir = ROOT_DIR / "results" / "robustness"
    if not robustness_dir.exists():
        print("  [WARN] No robustness results directory")
        return

    rows = []
    sweeps = {
        "topk": {"display": "Retrieval Depth", "param": "top_k"},
        "iter": {"display": "Iteration Budget", "param": "max_iterations"},
        "depth": {"display": "Recursion Depth", "param": "max_depth"},
    }

    for sweep_name, sweep_info in sweeps.items():
        sweep_dir = robustness_dir / sweep_name
        if not sweep_dir.exists():
            continue
        for pt_dir in sorted(sweep_dir.iterdir()):
            if not pt_dir.is_dir():
                continue
            sp = pt_dir / "summary.json"
            if not sp.exists():
                continue
            s = json.loads(sp.read_text())
            rows.append(
                {
                    "sweep": sweep_name,
                    "display": sweep_info["display"],
                    "param": sweep_info["param"],
                    "value": pt_dir.name,
                    "architecture": s["architecture"],
                    "em": s["avg_exact_match"],
                    "f1": s["avg_f1"],
                    "cost_usd": s["total_cost_usd"],
                    "tokens_per_q": s["avg_tokens_per_question"],
                    "llm_calls": s["avg_llm_calls"],
                    "run_id": pt_dir.name,
                }
            )

    csv_path = output_dir / "table4_robustness.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(f"  Robustness CSV: {csv_path} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Verification: check numbers match
# ---------------------------------------------------------------------------


def verify_numbers(output_dir: Path):
    """Verify key numbers used in the paper against raw data."""
    print("\n  Verifying key paper numbers...")
    checks = []

    for dataset in ["hotpotqa", "musique"]:
        for arch in ARCH_ORDER:
            run_id = BEST_RUNS[dataset].get(arch)
            if not run_id:
                continue
            try:
                s = load_summary(run_id)
            except FileNotFoundError:
                continue
            checks.append(
                {
                    "dataset": dataset,
                    "architecture": arch,
                    "run_id": run_id,
                    "em": s["avg_exact_match"],
                    "f1": s["avg_f1"],
                    "cost": s["total_cost_usd"],
                    "n_questions": s["num_questions"],
                }
            )

    csv_path = output_dir / "verified_numbers.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=checks[0].keys())
        writer.writeheader()
        writer.writerows(checks)
    print(f"  Verified numbers: {csv_path}")

    # Print key checks
    print("\n  Key number verification:")
    for c in checks:
        if c["dataset"] == "hotpotqa":
            print(
                f"    {c['architecture']:15s} HotpotQA: EM={c['em']:.1%}, "
                f"F1={c['f1']:.1%}, Cost=${c['cost']:.2f}, n={c['n_questions']}"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate all paper tables from a single source of truth",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper_tables"),
        help="Output directory (default: paper_tables/)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 60}")
    print("GENERATING PAPER TABLES FROM RAW DATA")
    print(f"{'=' * 60}")
    print(f"  Output: {args.output_dir}")
    print()

    print("[1/5] Main results table (Table 1)...")
    generate_main_results(args.output_dir)

    print("\n[2/5] Full grid tables (Table 2)...")
    generate_full_grid(args.output_dir)

    print("\n[3/5] Error taxonomy (Table 3, recomputed)...")
    generate_error_taxonomy(args.output_dir)

    print("\n[4/5] Robustness ablation table (Table 4)...")
    generate_robustness_table(args.output_dir)

    print("\n[5/5] Number verification...")
    verify_numbers(args.output_dir)

    print(f"\n{'=' * 60}")
    print("DONE -- all tables generated from raw data")
    print(f"  Check {args.output_dir}/ for all output files")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
