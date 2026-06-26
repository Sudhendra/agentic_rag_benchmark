#!/usr/bin/env python3
"""Extract representative failure cases for qualitative analysis.

Generates a CSV of failure cases suitable for paper case studies, showing:
  - Questions where all architectures fail (hardest questions)
  - Questions where only one architecture succeeds (signature strengths)
  - Near-miss cases (F1 > 0.5 but EM = 0)
  - Yes/No flip cases
  - Questions where ReAct succeeds but IRCoT fails (and vice versa)

Usage:
    python scripts/failure_case_studies.py
    python scripts/failure_case_studies.py --dataset hotpotqa --n-cases 20
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

OUTPUT_DIR = ROOT_DIR / "results" / "failure_cases"

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
        except FileNotFoundError:
            pass
    return all_preds


def build_question_table(all_preds: dict) -> list[dict]:
    """Build a table with one row per question, columns per architecture."""
    # Get common question IDs
    archs = list(all_preds.keys())
    common_ids = set(all_preds[archs[0]].keys())
    for a in archs[1:]:
        common_ids &= set(all_preds[a].keys())

    rows = []
    for qid in sorted(common_ids):
        row = {"question_id": qid}
        for arch in archs:
            p = all_preds[arch][qid]
            row[f"{arch}_em"] = p["exact_match"]
            row[f"{arch}_f1"] = p["f1"]
            row[f"{arch}_pred"] = p["predicted_answer"]
            row[f"{arch}_tokens"] = p["tokens_used"]
        # Use first arch for question metadata
        first_arch = archs[0]
        row["gold_answer"] = all_preds[first_arch][qid]["gold_answer"]
        row["question_type"] = all_preds[first_arch][qid].get("question_type", "unknown")
        rows.append(row)

    return rows


def extract_failure_cases(rows: list[dict], n_cases: int = 20) -> dict:
    """Extract different categories of failure cases."""
    archs = [a for a in ARCH_ORDER if f"{a}_em" in rows[0]]
    cases = {}

    # Category 1: All architectures fail (hardest questions)
    all_fail = [r for r in rows if all(r[f"{a}_em"] < 0.5 for a in archs)]
    all_fail.sort(key=lambda r: max(r[f"{a}_f1"] for a in archs), reverse=True)
    cases["all_fail"] = all_fail[:n_cases]

    # Category 2: Only one architecture succeeds (signature strengths)
    cases["solo_success"] = {}
    for arch in archs:
        others = [a for a in archs if a != arch]
        solo = [
            r for r in rows if r[f"{arch}_em"] >= 0.5 and all(r[f"{a}_em"] < 0.5 for a in others)
        ]
        cases["solo_success"][arch] = solo[:n_cases]

    # Category 3: Near-miss cases (F1 > 0.5 but EM = 0) - for IRCoT
    near_miss = []
    for r in rows:
        for arch in archs:
            if r[f"{arch}_em"] < 0.5 and r[f"{arch}_f1"] > 0.5:
                near_miss.append({**r, "near_miss_arch": arch})
                break
    near_miss.sort(key=lambda r: r[f"{r['near_miss_arch']}_f1"], reverse=True)
    cases["near_miss"] = near_miss[:n_cases]

    # Category 4: Yes/No flip cases
    yn_flip = []
    for r in rows:
        gold = r["gold_answer"].lower().strip()
        if gold in ("yes", "no"):
            for arch in archs:
                pred = r[f"{arch}_pred"].lower().strip().rstrip(".")
                if pred in ("yes", "no") and pred != gold:
                    yn_flip.append({**r, "flip_arch": arch})
                    break
    cases["yes_no_flip"] = yn_flip[:n_cases]

    # Category 5: ReAct succeeds, IRCoT fails (and vice versa)
    if "ReAct RAG" in archs and "IRCoT" in archs:
        react_solo = [r for r in rows if r["ReAct RAG_em"] >= 0.5 and r["IRCoT_em"] < 0.5]
        ircot_solo = [r for r in rows if r["IRCoT_em"] >= 0.5 and r["ReAct RAG_em"] < 0.5]
        cases["react_beats_ircot"] = react_solo[:n_cases]
        cases["ircot_beats_react"] = ircot_solo[:n_cases]

    # Category 6: Vanilla beats ReAct (complexity backfire)
    if "Vanilla RAG" in archs and "ReAct RAG" in archs:
        vanilla_beats = [r for r in rows if r["Vanilla RAG_em"] >= 0.5 and r["ReAct RAG_em"] < 0.5]
        cases["vanilla_beats_react"] = vanilla_beats[:n_cases]

    return cases


def write_csv(cases: dict, output_dir: Path, dataset: str, archs: list):
    """Write failure cases to CSV for manual review."""
    output_dir.mkdir(parents=True, exist_ok=True)

    archs = [a for a in archs if a in ARCH_ORDER]

    for category, case_list in cases.items():
        if not case_list:
            continue

        if isinstance(case_list, dict):
            # solo_success is a dict of arch -> cases
            for arch, arch_cases in case_list.items():
                if not arch_cases:
                    continue
                filename = f"{dataset}_{category}_{arch.replace(' ', '_').lower()}.csv"
                _write_case_csv(arch_cases, output_dir / filename, archs, extra_cols=["solo_arch"])
                print(f"  {category}/{arch}: {len(arch_cases)} cases -> {filename}")
        else:
            filename = f"{dataset}_{category}.csv"
            extra = []
            if category == "near_miss":
                extra = ["near_miss_arch"]
            elif category == "yes_no_flip":
                extra = ["flip_arch"]
            _write_case_csv(case_list, output_dir / filename, archs, extra_cols=extra)
            print(f"  {category}: {len(case_list)} cases -> {filename}")


def _write_case_csv(cases: list, path: Path, archs: list, extra_cols: list = None):
    """Write a single category to CSV."""
    if not cases:
        return

    # Build column list
    base_cols = ["question_id", "question_type", "gold_answer"]
    arch_cols = []
    for arch in archs:
        arch_cols.extend([f"{arch}_em", f"{arch}_f1", f"{arch}_pred", f"{arch}_tokens"])
    if extra_cols:
        base_cols.extend(extra_cols)

    all_cols = base_cols + arch_cols

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
        writer.writeheader()
        for case in cases:
            # Truncate predictions for readability
            row = dict(case)
            for arch in archs:
                pred_key = f"{arch}_pred"
                if pred_key in row and len(str(row[pred_key])) > 100:
                    row[pred_key] = str(row[pred_key])[:100] + "..."
            writer.writerow(row)


def print_summary(cases: dict, dataset: str):
    """Print summary of failure case categories."""
    print(f"\n{'=' * 70}")
    print(f"FAILURE CASE STUDIES ({dataset.upper()})")
    print(f"{'=' * 70}")

    archs = ARCH_ORDER

    # All fail
    n_all_fail = len(cases.get("all_fail", []))
    print(f"\n  All architectures fail: {n_all_fail} cases extracted")
    if cases.get("all_fail"):
        for r in cases["all_fail"][:3]:
            print(
                f"    Q: {r['question_id'][:12]}  gold='{r['gold_answer'][:30]}'  "
                f"type={r['question_type']}"
            )

    # Solo success
    print(f"\n  Solo success cases (only one arch correct):")
    for arch in archs:
        solo = cases.get("solo_success", {}).get(arch, [])
        if solo:
            print(f"    {arch:15s}: {len(solo)} questions where only this arch succeeds")

    # Near miss
    n_near = len(cases.get("near_miss", []))
    print(f"\n  Near-miss cases (F1>0.5, EM=0): {n_near}")
    if cases.get("near_miss"):
        for r in cases["near_miss"][:3]:
            arch = r["near_miss_arch"]
            print(
                f"    Q: {r['question_id'][:12]}  gold='{r['gold_answer'][:30]}'  "
                f"{arch} pred='{r[f'{arch}_pred'][:30]}'  F1={r[f'{arch}_f1']:.2f}"
            )

    # Yes/No flip
    n_flip = len(cases.get("yes_no_flip", []))
    print(f"\n  Yes/No flip cases: {n_flip}")

    # ReAct vs IRCoT
    n_rb = len(cases.get("react_beats_ircot", []))
    n_ib = len(cases.get("ircot_beats_react", []))
    print(f"\n  ReAct succeeds, IRCoT fails: {n_rb}")
    print(f"  IRCoT succeeds, ReAct fails: {n_ib}")

    # Vanilla beats ReAct
    n_vb = len(cases.get("vanilla_beats_react", []))
    print(f"\n  Vanilla beats ReAct (complexity backfire): {n_vb}")


def main():
    parser = argparse.ArgumentParser(description="Extract failure case studies")
    parser.add_argument("--dataset", choices=["hotpotqa", "musique", "both"], default="both")
    parser.add_argument("--n-cases", type=int, default=20, help="Max cases per category")
    args = parser.parse_args()

    datasets = ["hotpotqa", "musique"] if args.dataset == "both" else [args.dataset]

    for dataset in datasets:
        print(f"\n{'=' * 70}")
        print(f"Loading predictions for {dataset.upper()}...")
        print(f"{'=' * 70}")

        all_preds = load_all_predictions(dataset)
        if not all_preds:
            continue

        print(f"  Building question table...")
        rows = build_question_table(all_preds)
        print(f"  Total questions: {len(rows)}")

        print(f"  Extracting failure cases...")
        cases = extract_failure_cases(rows, n_cases=args.n_cases)

        print_summary(cases, dataset)
        write_csv(cases, OUTPUT_DIR, dataset, list(all_preds.keys()))

    print(f"\n{'=' * 70}")
    print(f"DONE")
    print(f"  CSVs: {OUTPUT_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
