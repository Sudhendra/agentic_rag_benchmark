#!/usr/bin/env python3
"""Run all 2WikiMultiHopQA experiments sequentially.

Executes all 21 architecture × retriever combinations on 2WikiMultiHopQA
(12,576 questions). Vanilla BM25 is already complete; the remaining 20 runs
are queued cheapest-first to minimise cost if interrupted.

Usage:
    python scripts/run_2wiki_experiments.py                 # Run all pending
    python scripts/run_2wiki_experiments.py --dry-run       # Show plan, no API calls
    python scripts/run_2wiki_experiments.py --skip-existing # Skip already-completed runs
    python scripts/run_2wiki_experiments.py --arch ircot    # Run one architecture only
    python scripts/run_2wiki_experiments.py --resume        # Alias for --skip-existing
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

# ---------------------------------------------------------------------------
# Run plan — ordered cheapest/fastest first to minimise cost if interrupted.
# Approximate per-run cost on gpt-4o-mini at 12,576 questions:
#   vanilla  ~$1.20  |  self_rag  ~$2.50  |  ircot  ~$2.50  |  rlm  ~$4.00
#   planner  ~$5.00  |  reap      ~$8.00  |  react  ~$14.00
#
# Within each architecture: BM25 → Dense → Hybrid
# (BM25 skips embedding calls, so it's the cheapest retriever)
# ---------------------------------------------------------------------------

RUNS: list[dict] = [
    # ── Vanilla (dense + hybrid; bm25 already done: fb0e11a5) ──────────────
    {
        "arch": "vanilla_rag",
        "retriever": "dense",
        "config": "configs/vanilla_2wiki_dense_full.yaml",
        "est_cost_usd": 1.20,
        "already_done": False,
    },
    {
        "arch": "vanilla_rag",
        "retriever": "hybrid",
        "config": "configs/vanilla_2wiki_hybrid_full.yaml",
        "est_cost_usd": 1.20,
        "already_done": False,
    },
    # ── Vanilla BM25 — already complete ─────────────────────────────────────
    {
        "arch": "vanilla_rag",
        "retriever": "bm25",
        "config": "configs/vanilla_2wiki_bm25_full.yaml",
        "est_cost_usd": 1.16,
        "already_done": True,  # run fb0e11a5 completed
    },
    # ── Self-RAG ─────────────────────────────────────────────────────────────
    {
        "arch": "self_rag",
        "retriever": "bm25",
        "config": "configs/self_rag_2wiki_bm25_full.yaml",
        "est_cost_usd": 2.50,
        "already_done": False,
    },
    {
        "arch": "self_rag",
        "retriever": "dense",
        "config": "configs/self_rag_2wiki_dense_full.yaml",
        "est_cost_usd": 2.50,
        "already_done": False,
    },
    {
        "arch": "self_rag",
        "retriever": "hybrid",
        "config": "configs/self_rag_2wiki_hybrid_full.yaml",
        "est_cost_usd": 2.50,
        "already_done": False,
    },
    # ── IRCoT ────────────────────────────────────────────────────────────────
    {
        "arch": "ircot_rag",
        "retriever": "bm25",
        "config": "configs/ircot_2wiki_bm25_full.yaml",
        "est_cost_usd": 2.50,
        "already_done": False,
    },
    {
        "arch": "ircot_rag",
        "retriever": "dense",
        "config": "configs/ircot_2wiki_dense_full.yaml",
        "est_cost_usd": 2.50,
        "already_done": False,
    },
    {
        "arch": "ircot_rag",
        "retriever": "hybrid",
        "config": "configs/ircot_2wiki_hybrid_full.yaml",
        "est_cost_usd": 2.50,
        "already_done": False,
    },
    # ── Recursive LM ─────────────────────────────────────────────────────────
    {
        "arch": "recursive_lm",
        "retriever": "bm25",
        "config": "configs/rlm_2wiki_bm25_full.yaml",
        "est_cost_usd": 4.00,
        "already_done": False,
    },
    {
        "arch": "recursive_lm",
        "retriever": "dense",
        "config": "configs/rlm_2wiki_dense_full.yaml",
        "est_cost_usd": 4.00,
        "already_done": False,
    },
    {
        "arch": "recursive_lm",
        "retriever": "hybrid",
        "config": "configs/rlm_2wiki_hybrid_full.yaml",
        "est_cost_usd": 4.00,
        "already_done": False,
    },
    # ── Planner RAG ──────────────────────────────────────────────────────────
    {
        "arch": "planner_rag",
        "retriever": "bm25",
        "config": "configs/planner_2wiki_bm25_full.yaml",
        "est_cost_usd": 5.00,
        "already_done": False,
    },
    {
        "arch": "planner_rag",
        "retriever": "dense",
        "config": "configs/planner_2wiki_dense_full.yaml",
        "est_cost_usd": 5.00,
        "already_done": False,
    },
    {
        "arch": "planner_rag",
        "retriever": "hybrid",
        "config": "configs/planner_2wiki_hybrid_full.yaml",
        "est_cost_usd": 5.00,
        "already_done": False,
    },
    # ── REAP ─────────────────────────────────────────────────────────────────
    {
        "arch": "reap_rag",
        "retriever": "bm25",
        "config": "configs/reap_2wiki_bm25_full.yaml",
        "est_cost_usd": 8.00,
        "already_done": False,
    },
    {
        "arch": "reap_rag",
        "retriever": "dense",
        "config": "configs/reap_2wiki_dense_full.yaml",
        "est_cost_usd": 8.00,
        "already_done": False,
    },
    {
        "arch": "reap_rag",
        "retriever": "hybrid",
        "config": "configs/reap_2wiki_hybrid_full.yaml",
        "est_cost_usd": 8.00,
        "already_done": False,
    },
    # ── ReAct (most expensive — runs last) ───────────────────────────────────
    {
        "arch": "react_rag",
        "retriever": "bm25",
        "config": "configs/react_2wiki_bm25_full.yaml",
        "est_cost_usd": 14.00,
        "already_done": False,
    },
    {
        "arch": "react_rag",
        "retriever": "dense",
        "config": "configs/react_2wiki_dense_full.yaml",
        "est_cost_usd": 14.00,
        "already_done": False,
    },
    {
        "arch": "react_rag",
        "retriever": "hybrid",
        "config": "configs/react_2wiki_hybrid_full.yaml",
        "est_cost_usd": 14.00,
        "already_done": False,
    },
]

RESULTS_DIR = ROOT_DIR / "results"
PROGRESS_FILE = ROOT_DIR / "results" / "2wiki_progress.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_progress() -> dict[str, str]:
    """Return {config_name: run_id} for completed runs saved in progress file."""
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
    """Return the most recently created results directory with a summary.json."""
    if not RESULTS_DIR.exists():
        return None
    run_dirs = [d for d in RESULTS_DIR.iterdir() if d.is_dir() and (d / "summary.json").exists()]
    return max(run_dirs, key=lambda d: d.stat().st_ctime) if run_dirs else None


def _get_run_id_from_dir(run_dir: Path | None) -> str | None:
    if run_dir is None:
        return None
    summary = run_dir / "summary.json"
    if not summary.exists():
        return None
    try:
        data = json.loads(summary.read_text())
        return data.get("run_id") or run_dir.name[:12]
    except Exception:
        return run_dir.name[:12]


def _config_key(run: dict) -> str:
    return Path(run["config"]).stem


def _select_runs(
    arch_filter: str | None,
    skip_existing: bool,
    progress: dict[str, str],
) -> list[dict]:
    """Return the subset of RUNS to actually execute."""
    selected = []
    for run in RUNS:
        if run["already_done"]:
            continue
        if arch_filter and run["arch"] != arch_filter:
            continue
        key = _config_key(run)
        if skip_existing and key in progress:
            continue
        selected.append(run)
    return selected


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _print_plan(runs_to_run: list[dict], progress: dict[str, str]) -> None:
    total_pending = sum(1 for r in RUNS if not r["already_done"])
    already_done = [r for r in RUNS if r["already_done"]]
    skipped = [r for r in RUNS if not r["already_done"] and _config_key(r) in progress]

    print("=" * 70)
    print("2WikiMultiHopQA — Full Experiment Suite")
    print(f"Total runs in plan:  {len(RUNS)}")
    print(f"Already completed:   {len(already_done) + len(skipped)}")
    print(f"To run now:          {len(runs_to_run)}")
    est_total = sum(r["est_cost_usd"] for r in runs_to_run)
    print(f"Estimated cost:      ~${est_total:.2f}")
    print("=" * 70)

    if not runs_to_run:
        print("\nNothing to run — all experiments are complete.")
        return

    print(f"\n{'#':<4} {'Architecture':<16} {'Retriever':<10} {'Est. Cost':>10}  Config")
    print("-" * 70)
    for i, run in enumerate(runs_to_run, 1):
        print(
            f"{i:<4} {run['arch']:<16} {run['retriever']:<10}"
            f" ${run['est_cost_usd']:>8.2f}  {run['config']}"
        )

    # Already-done note
    if already_done:
        print(f"\n  Already done (skipped):")
        for r in already_done:
            print(f"    {r['arch']:<16} {r['retriever']:<10}  [pre-existing]")
    if skipped:
        print(f"\n  Previously completed (--skip-existing):")
        for r in skipped:
            run_id = progress.get(_config_key(r), "?")
            print(f"    {r['arch']:<16} {r['retriever']:<10}  [{run_id}]")

    print()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_single(run: dict, run_number: int, total: int) -> str | None:
    """Run one experiment config. Returns run_id on success, None on failure."""
    arch = run["arch"]
    retriever = run["retriever"]
    config_path = ROOT_DIR / run["config"]

    label = f"[{run_number}/{total}] {arch} / {retriever}"
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"  Config: {config_path.name}  |  Est. cost: ~${run['est_cost_usd']:.2f}")
    print(f"{'=' * 70}")

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
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(ROOT_DIR),
    )

    try:
        for line in iter(process.stdout.readline, ""):  # type: ignore[union-attr]
            if not line:
                break
            sys.stdout.write(f"    {line}")
            sys.stdout.flush()
        process.wait()
    except KeyboardInterrupt:
        process.kill()
        print(f"\n  [INTERRUPTED] {label}")
        raise

    elapsed = time.time() - start
    mins, secs = divmod(int(elapsed), 60)
    print(f"\n  Exit code: {process.returncode}  |  Wall time: {mins}m {secs}s")

    if process.returncode != 0:
        print(f"  [FAILED] {label}")
        return None

    run_dir = _find_latest_run_dir()
    run_id = _get_run_id_from_dir(run_dir)
    print(f"  [DONE] run_id: {run_id}")
    return run_id


def run_all(runs: list[dict], progress: dict[str, str]) -> dict[str, str]:
    """Execute all runs sequentially, saving progress after each."""
    total = len(runs)
    succeeded = 0
    failed: list[str] = []

    for i, run in enumerate(runs, 1):
        key = _config_key(run)
        try:
            run_id = run_single(run, i, total)
        except KeyboardInterrupt:
            print("\n\n[ABORTED] Saving progress before exit...")
            _save_progress(progress)
            print(f"  Progress saved to {PROGRESS_FILE}")
            print(f"  Resume with: python scripts/run_2wiki_experiments.py --skip-existing")
            sys.exit(1)

        if run_id:
            progress[key] = run_id
            _save_progress(progress)
            succeeded += 1
        else:
            failed.append(f"{run['arch']}/{run['retriever']}")

    # ── Final summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("2WikiMultiHopQA Suite — COMPLETE")
    print(f"  Succeeded: {succeeded}/{total}")
    if failed:
        print(f"  Failed:    {len(failed)}")
        for f in failed:
            print(f"    - {f}")
    print(f"  Progress file: {PROGRESS_FILE}")
    print("=" * 70)
    return progress


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all 2WikiMultiHopQA experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the execution plan without making any API calls",
    )
    parser.add_argument(
        "--skip-existing",
        "--resume",
        action="store_true",
        dest="skip_existing",
        help="Skip configs that appear in the progress file (safe to resume)",
    )
    parser.add_argument(
        "--arch",
        choices=[
            "vanilla_rag",
            "self_rag",
            "ircot_rag",
            "recursive_lm",
            "planner_rag",
            "reap_rag",
            "react_rag",
        ],
        help="Run only this architecture (all retrievers)",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY") and not args.dry_run:
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        print("  Set it with:  set OPENAI_API_KEY=sk-...", file=sys.stderr)
        sys.exit(1)

    progress = _load_progress()
    runs_to_run = _select_runs(args.arch, args.skip_existing, progress)

    _print_plan(runs_to_run, progress)

    if args.dry_run:
        print("Dry run — no experiments executed.")
        return

    if not runs_to_run:
        return

    print(f"OPENAI_API_KEY: SET")
    print("Starting in 3 seconds  (Ctrl-C to abort safely)...")
    time.sleep(3)

    run_all(runs_to_run, progress)


if __name__ == "__main__":
    main()
