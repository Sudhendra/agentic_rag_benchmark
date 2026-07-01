#!/usr/bin/env python3
"""Run ablation sweeps to measure hyperparameter sensitivity.

Three sweeps are run, each varying a single architectural parameter:

  top_k sweep   — Vanilla RAG, BM25, 100q: how retrieval depth affects accuracy
  iter sweep    — ReAct, BM25, 50q:        how iteration budget affects accuracy
  depth sweep   — RLM, BM25, 50q:          how recursion depth affects accuracy

Each sweep finds the performance knee — the parameter value where accuracy
stops improving and cost starts rising — and exports results for figure generation.

Usage:
    python scripts/robustness_evaluator.py                  # All 3 sweeps
    python scripts/robustness_evaluator.py --dry-run        # Show plan, no API calls
    python scripts/robustness_evaluator.py --sweep topk     # One sweep only
    python scripts/robustness_evaluator.py --sweep iter depth  # Two sweeps
    python scripts/robustness_evaluator.py --skip-existing  # Resume interrupted run
    python scripts/robustness_evaluator.py --analyze-only   # Report from saved data
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv

load_dotenv()

from scripts.analyze_results import load_results

# ---------------------------------------------------------------------------
# Sweep registry
# Each entry in a sweep: (param_value, config_path, is_default)
# is_default=True marks the configuration used in the main benchmark runs.
# ---------------------------------------------------------------------------

SWEEPS: dict[str, dict] = {
    "topk": {
        "display": "Retrieval Depth (top_k) — Vanilla RAG",
        "param_name": "top_k",
        "param_unit": "docs",
        "architecture": "vanilla_rag",
        "dataset": "hotpotqa",
        "n_questions": 100,
        "description": (
            "How does the number of retrieved documents affect accuracy and cost? "
            "At what point does more context stop helping?"
        ),
        "points": [
            {"value": 3, "config": "configs/robustness_topk3.yaml", "is_default": False},
            {"value": 5, "config": "configs/robustness_topk5.yaml", "is_default": True},
            {"value": 10, "config": "configs/robustness_topk10.yaml", "is_default": False},
            {"value": 20, "config": "configs/robustness_topk20.yaml", "is_default": False},
        ],
        "est_cost_usd": 0.06,  # 4 × 100q × ~$0.00015/q (Vanilla is cheap)
    },
    "iter": {
        "display": "Iteration Budget (max_iterations) — ReAct",
        "param_name": "max_iterations",
        "param_unit": "iterations",
        "architecture": "react_rag",
        "dataset": "hotpotqa",
        "n_questions": 50,
        "description": (
            "How many search-reason cycles does ReAct need? "
            "Does allowing more iterations improve accuracy or just cost?"
        ),
        "points": [
            {"value": 3, "config": "configs/robustness_react_iter3.yaml", "is_default": False},
            {"value": 7, "config": "configs/robustness_react_iter7.yaml", "is_default": True},
            {"value": 10, "config": "configs/robustness_react_iter10.yaml", "is_default": False},
        ],
        "est_cost_usd": 0.18,  # 3 × 50q × ~$0.0012/q (ReAct is expensive)
    },
    "depth": {
        "display": "Recursion Depth (max_depth) — RLM",
        "param_name": "max_depth",
        "param_unit": "levels",
        "architecture": "recursive_lm",
        "dataset": "hotpotqa",
        "n_questions": 50,
        "description": (
            "How deep should RLM recurse? "
            "Does deeper decomposition improve multi-hop performance or just increase loop rate?"
        ),
        "points": [
            {"value": 2, "config": "configs/robustness_rlm_depth2.yaml", "is_default": False},
            {"value": 3, "config": "configs/robustness_rlm_depth3.yaml", "is_default": True},
            {"value": 5, "config": "configs/robustness_rlm_depth5.yaml", "is_default": False},
        ],
        "est_cost_usd": 0.12,  # 3 × 50q × ~$0.0008/q
    },
}

RESULTS_DIR = ROOT_DIR / "results" / "robustness"
PROGRESS_FILE = RESULTS_DIR / "progress.json"


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------


def _load_progress() -> dict[str, str]:
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_progress(progress: dict[str, str]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def _find_latest_run_dir() -> Path | None:
    results_root = ROOT_DIR / "results"
    if not results_root.exists():
        return None
    run_dirs = [
        d
        for d in results_root.iterdir()
        if d.is_dir()
        and d.name not in ("robustness", "sensitivity", "figures", "error_taxonomy")
        and (d / "summary.json").exists()
    ]
    return max(run_dirs, key=lambda d: d.stat().st_ctime) if run_dirs else None


def _get_run_id(run_dir: Path | None) -> str | None:
    if run_dir is None:
        return None
    try:
        data = json.loads((run_dir / "summary.json").read_text())
        return data.get("run_id") or run_dir.name[:12]
    except Exception:
        return run_dir.name[:12]


def _point_key(sweep_name: str, value: int) -> str:
    return f"{sweep_name}_{value}"


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def dry_run(selected_sweeps: list[str], skip_existing: bool, progress: dict) -> None:
    print("=" * 70)
    print("ROBUSTNESS ABLATION SWEEPS — DRY RUN")
    print("=" * 70)

    total_est = 0.0
    for sw_name in selected_sweeps:
        sw = SWEEPS[sw_name]
        print(f"\n  [{sw_name}] {sw['display']}")
        print(f"  {sw['description']}")
        print(f"\n  {'Value':<10} {'Default':<10} {'Config':<45} {'Status'}")
        print(f"  {'-' * 10} {'-' * 10} {'-' * 45} {'-' * 10}")
        for pt in sw["points"]:
            key = _point_key(sw_name, pt["value"])
            config_path = ROOT_DIR / pt["config"]
            status = (
                "SKIP"
                if (skip_existing and key in progress)
                else ("DONE" if key in progress else ("OK" if config_path.exists() else "MISSING"))
            )
            default_mark = "← default" if pt["is_default"] else ""
            print(f"  {pt['value']:<10} {default_mark:<10} {pt['config']:<45} [{status}]")
        total_est += sw["est_cost_usd"]

    print(f"\n  Total estimated cost: ~${total_est:.2f}")
    print(f"  Results dir:          {RESULTS_DIR}")
    print(f"  OPENAI_API_KEY:       {'SET' if os.environ.get('OPENAI_API_KEY') else 'NOT SET'}")
    print()


# ---------------------------------------------------------------------------
# Single-run executor
# ---------------------------------------------------------------------------


def run_point(sweep_name: str, point: dict, n: int, total: int) -> str | None:
    sw = SWEEPS[sweep_name]
    value = point["value"]
    config_path = ROOT_DIR / point["config"]
    label = f"[{n}/{total}] {sw['display']} | {sw['param_name']}={value}"

    print(f"\n{'─' * 60}")
    print(f"  {label}")
    if point["is_default"]:
        print(f"  (this is the default value used in the main benchmark)")

    if not config_path.exists():
        print(f"  [ERROR] Config not found: {config_path}")
        return None

    cmd = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "run_experiment.py"),
        "--config",
        str(config_path),
    ]

    start = time.time()
    process: subprocess.Popen[str] | None = None
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(ROOT_DIR),
        )
        stdout = process.stdout
        assert stdout is not None
        for line in iter(stdout.readline, ""):
            if not line:
                break
            sys.stdout.write(f"    [{sweep_name}/{value}] {line}")
            sys.stdout.flush()
        process.wait()
    except KeyboardInterrupt:
        if process is not None:
            process.kill()
        raise

    elapsed = time.time() - start
    mins, secs = divmod(int(elapsed), 60)
    returncode = process.returncode if process is not None else 1
    print(f"\n  Exit code: {returncode}  |  Wall time: {mins}m {secs}s")

    if returncode != 0:
        print(f"  [FAILED] {label}")
        return None

    run_dir = _find_latest_run_dir()
    run_id = _get_run_id(run_dir)
    print(f"  [DONE] run_id: {run_id}")
    return run_id


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _load_sweep_results(sweep_name: str, progress: dict[str, str]) -> list[dict]:
    """Load summary rows for all completed points in a sweep."""
    sw = SWEEPS[sweep_name]
    rows = []
    for pt in sw["points"]:
        key = _point_key(sweep_name, pt["value"])
        run_id_prefix = progress.get(key)
        if not run_id_prefix:
            continue

        # Search for the run directory
        results_root = ROOT_DIR / "results"
        matched: list[Path] = []
        for d in results_root.iterdir():
            if d.is_dir() and (d / "summary.json").exists():
                if d.name.startswith(run_id_prefix) or run_id_prefix in d.name:
                    matched.append(d)
        # Also check robustness subdirectory
        rob_dir = RESULTS_DIR / sweep_name / str(pt["value"])
        if not matched and rob_dir.exists() and (rob_dir / "summary.json").exists():
            matched = [rob_dir]
        if not matched:
            continue

        run_dir = matched[0]
        try:
            results = load_results(run_dir)
            summary = results["summary"]
            rows.append(
                {
                    "param_value": pt["value"],
                    "is_default": pt["is_default"],
                    "run_id": run_dir.name[:12],
                    "num_questions": summary.get("num_questions", 0),
                    "exact_match": summary.get("avg_exact_match", 0.0),
                    "f1": summary.get("avg_f1", 0.0),
                    "cost_usd": summary.get("total_cost_usd", 0.0),
                    "latency_ms": summary.get("avg_latency_ms", 0.0),
                    "avg_llm_calls": summary.get("avg_llm_calls", 0.0),
                    "avg_retrieval_calls": summary.get("avg_retrieval_calls", 0.0),
                }
            )
        except Exception as e:
            print(f"  [WARN] Could not load {run_dir}: {e}", file=sys.stderr)
    return rows


def _find_knee(rows: list[dict]) -> dict | None:
    """Find the value with best EM/cost ratio (knee of the curve)."""
    if not rows:
        return None
    # Simple: best EM per dollar
    scored = []
    for r in rows:
        cost = r["cost_usd"]
        em = r["exact_match"]
        score = em / cost if cost > 0 else em * 1000
        scored.append((score, r))
    return max(scored, key=lambda x: x[0])[1]


def print_sweep_table(sweep_name: str, rows: list[dict]) -> None:
    sw = SWEEPS[sweep_name]
    if not rows:
        print(f"\n  No results yet for [{sweep_name}] sweep.")
        return

    rows_sorted = sorted(rows, key=lambda r: r["param_value"])

    print(f"\n{'─' * 72}")
    print(f"  [{sweep_name}] {sw['display']}")
    print(f"{'─' * 72}")
    print(
        f"  {sw['param_name']:<8} {'Default':<9} {'EM':>7} {'F1':>7} "
        f"{'Cost':>9} {'LLM calls':>10} {'Ret calls':>10}"
    )
    print(f"  {'-' * 8} {'-' * 9} {'-' * 7} {'-' * 7} {'-' * 9} {'-' * 10} {'-' * 10}")

    default_em = None
    for r in rows_sorted:
        marker = "← def" if r["is_default"] else ""
        if r["is_default"]:
            default_em = r["exact_match"]
        delta = ""
        if default_em is not None and not r["is_default"]:
            diff = r["exact_match"] - default_em
            delta = f" ({'+' if diff >= 0 else ''}{diff:.1%})"
        print(
            f"  {r['param_value']:<8} {marker:<9} "
            f"{r['exact_match']:>6.1%}{delta:<9} {r['f1']:>6.1%}  "
            f"${r['cost_usd']:>7.4f} {r['avg_llm_calls']:>9.1f} {r['avg_retrieval_calls']:>9.1f}"
        )

    # Insights
    if len(rows_sorted) >= 2:
        best_em = max(rows_sorted, key=lambda r: r["exact_match"])
        cheapest = min(rows_sorted, key=lambda r: r["cost_usd"])
        knee = _find_knee(rows_sorted)
        em_range = max(r["exact_match"] for r in rows_sorted) - min(
            r["exact_match"] for r in rows_sorted
        )

        print(
            f"\n  Best EM:       {sw['param_name']}={best_em['param_value']}  ({best_em['exact_match']:.1%})"
        )
        print(
            f"  Cheapest:      {sw['param_name']}={cheapest['param_value']}  (${cheapest['cost_usd']:.4f})"
        )
        if knee:
            print(
                f"  Recommended:   {sw['param_name']}={knee['param_value']}  (best EM/cost ratio)"
            )
        print(f"  EM range:      {em_range:.1%}  across all settings")


def print_cross_sweep_summary(all_results: dict[str, list[dict]]) -> None:
    print("\n" + "=" * 72)
    print("ABLATION SUMMARY — HYPERPARAMETER SENSITIVITY")
    print("=" * 72)
    print(
        f"\n  {'Sweep':<12} {'Param':<18} {'Points':<8} "
        f"{'EM Range':<12} {'Best value':<12} {'vs Default'}"
    )
    print(f"  {'-' * 12} {'-' * 18} {'-' * 8} {'-' * 12} {'-' * 12} {'-' * 12}")

    for sw_name, rows in all_results.items():
        if not rows:
            continue
        sw = SWEEPS[sw_name]
        rows_s = sorted(rows, key=lambda r: r["param_value"])
        best = max(rows_s, key=lambda r: r["exact_match"])
        default_row = next((r for r in rows_s if r["is_default"]), None)
        em_range = max(r["exact_match"] for r in rows_s) - min(r["exact_match"] for r in rows_s)
        vs_default = ""
        if default_row:
            diff = best["exact_match"] - default_row["exact_match"]
            vs_default = f"{'+' if diff >= 0 else ''}{diff:.1%}"
        print(
            f"  {sw_name:<12} {sw['param_name']:<18} {len(rows):<8} "
            f"{em_range:<11.1%} {best['param_value']!s:<12} {vs_default}"
        )

    print()
    print("  Interpretation:")
    print("  EM range shows how sensitive each architecture is to the hyperparameter.")
    print("  'vs Default' shows how much the best value improves on the benchmark setting.")


def export_json(all_results: dict[str, list[dict]]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "robustness_results.json"

    export: dict[str, object] = {}
    for sw_name, rows in all_results.items():
        if not rows:
            continue
        sw = SWEEPS[sw_name]
        rows_s = sorted(rows, key=lambda r: r["param_value"])
        export[sw_name] = {
            "display": sw["display"],
            "param_name": sw["param_name"],
            "param_unit": sw["param_unit"],
            "architecture": sw["architecture"],
            "points": {
                str(r["param_value"]): {
                    "param_value": r["param_value"],
                    "is_default": r["is_default"],
                    "run_id": r["run_id"],
                    "exact_match": r["exact_match"],
                    "f1": r["f1"],
                    "cost_usd": r["cost_usd"],
                    "latency_ms": r["latency_ms"],
                    "avg_llm_calls": r["avg_llm_calls"],
                    "avg_retrieval_calls": r["avg_retrieval_calls"],
                }
                for r in rows_s
            },
        }

    output_path.write_text(json.dumps(export, indent=2))
    print(f"\n  Exported: {output_path}")
    return output_path


def analyze_existing(selected_sweeps: list[str], progress: dict[str, str]) -> dict[str, list[dict]]:
    all_results: dict[str, list[dict]] = {}
    print("\nAnalyzing existing results...")
    for sw_name in selected_sweeps:
        rows = _load_sweep_results(sw_name, progress)
        all_results[sw_name] = rows
        print_sweep_table(sw_name, rows)

    if any(all_results.values()):
        print_cross_sweep_summary(all_results)
        export_json(all_results)
    else:
        print("\n  No results found. Run the sweeps first.")

    return all_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ablation sweeps for hyperparameter sensitivity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--sweep",
        nargs="+",
        choices=list(SWEEPS.keys()),
        default=list(SWEEPS.keys()),
        help="Sweeps to run (default: all). Choices: topk, iter, depth",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show execution plan without making API calls",
    )
    parser.add_argument(
        "--skip-existing",
        "--resume",
        action="store_true",
        dest="skip_existing",
        help="Skip configs that already have results in the progress file",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Print report from existing results without running anything",
    )
    args = parser.parse_args()

    selected_sweeps: list[str] = list(dict.fromkeys(args.sweep))

    progress = _load_progress()

    if args.dry_run:
        dry_run(selected_sweeps, args.skip_existing, progress)
        return

    if args.analyze_only:
        analyze_existing(selected_sweeps, progress)
        return

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    # ── Build run list ────────────────────────────────────────────────────────
    to_run: list[tuple[str, dict]] = []
    for sw_name in selected_sweeps:
        for pt in SWEEPS[sw_name]["points"]:
            key = _point_key(sw_name, pt["value"])
            if args.skip_existing and key in progress:
                continue
            to_run.append((sw_name, pt))

    total = len(to_run)
    if total == 0:
        print("Nothing to run — all sweep points already complete.")
        analyze_existing(selected_sweeps, progress)
        return

    # ── Summary header ────────────────────────────────────────────────────────
    total_est = sum(SWEEPS[sw]["est_cost_usd"] for sw in selected_sweeps)
    print("=" * 70)
    print("ROBUSTNESS ABLATION SWEEPS")
    print(f"  Sweeps:    {', '.join(selected_sweeps)}")
    print(f"  Points:    {total}")
    print(f"  Est. cost: ~${total_est:.2f}")
    print(f"  OPENAI_API_KEY: SET")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for i, (sw_name, pt) in enumerate(to_run, 1):
        key = _point_key(sw_name, pt["value"])
        try:
            run_id = run_point(sw_name, pt, i, total)
        except KeyboardInterrupt:
            print("\n\n[ABORTED] Saving progress...")
            _save_progress(progress)
            print(f"  Resume with: python scripts/robustness_evaluator.py --skip-existing")
            sys.exit(1)

        if run_id:
            progress[key] = run_id
            _save_progress(progress)

            # Copy results into organised subdirectory
            run_dir = _find_latest_run_dir()
            if run_dir:
                import shutil

                target = RESULTS_DIR / sw_name / str(pt["value"])
                if not target.exists():
                    try:
                        shutil.copytree(str(run_dir), str(target))
                    except Exception:
                        pass

    # ── Final report ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ALL SWEEP POINTS COMPLETE — GENERATING REPORT")
    print("=" * 70)
    analyze_existing(selected_sweeps, progress)

    total_cost = sum(
        r["cost_usd"] for sw in selected_sweeps for r in _load_sweep_results(sw, progress)
    )
    print(f"\nTotal experiment cost: ${total_cost:.4f}")
    print(f"Results JSON:          {RESULTS_DIR / 'robustness_results.json'}")
    print(f"To re-run report:      python scripts/robustness_evaluator.py --analyze-only")


if __name__ == "__main__":
    main()
