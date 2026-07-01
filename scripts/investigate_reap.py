#!/usr/bin/env python3
"""Investigate the REAP implementation gap (28.1% vs paper's 59.2% EM).

This script:
  1.  Exports error cases from the best existing REAP/gpt-4o-mini run
      (run_id c4da1615351b) for qualitative analysis.
  2.  Runs REAP with gpt-4o on 200 questions to directly test the
      model-strength hypothesis.
  3.  Prints a side-by-side comparison and interprets the result.

Usage:
    # Export errors from existing mini run (no API calls)
    python scripts/investigate_reap.py --export-only

    # Run gpt-4o experiment + export (costs ~$5)
    python scripts/investigate_reap.py

    # Just compare if gpt-4o run already done
    python scripts/investigate_reap.py --compare-only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Best existing REAP run (gpt-4o-mini, 7405q, 28.1% EM, BM25)
BEST_MINI_RUN_ID = "c4da1615351b47bf8d5f8e0617188940"

# Config for gpt-4o experiment
GPT4O_CONFIG = ROOT_DIR / "configs" / "reap_gpt4o_200q.yaml"

# Where to save outputs
OUTPUT_DIR = ROOT_DIR / "results" / "reap_investigation"

# Paper's reported result
PAPER_EM = 0.592
PAPER_MODEL = "unknown (likely GPT-4 or fine-tuned)"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def find_run_dir(run_id_prefix: str) -> Path | None:
    """Locate the results directory for a given run ID prefix."""
    results_root = ROOT_DIR / "results"
    for d in results_root.iterdir():
        if d.is_dir() and d.name.startswith(run_id_prefix):
            return d
    return None


def load_summary(run_dir: Path) -> dict:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found in {run_dir}")
    return json.loads(summary_path.read_text())


def find_latest_run_dir() -> Path | None:
    """Find the most recently modified results directory."""
    results_root = ROOT_DIR / "results"
    dirs = [d for d in results_root.iterdir() if d.is_dir() and (d / "summary.json").exists()]
    if not dirs:
        return None
    return max(dirs, key=lambda d: d.stat().st_mtime)


# ---------------------------------------------------------------------------
# Export errors
# ---------------------------------------------------------------------------


def export_errors_from_mini(output_dir: Path) -> Path:
    """Export errors from the existing gpt-4o-mini REAP run."""
    run_dir = find_run_dir(BEST_MINI_RUN_ID)
    if run_dir is None:
        print(f"[WARN] Could not find run directory for {BEST_MINI_RUN_ID[:12]}")
        print("       Skipping error export.")
        return output_dir / "errors_mini_not_found.csv"

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "reap_mini_errors.csv"

    print(f"\nExporting errors from REAP/gpt-4o-mini run ({BEST_MINI_RUN_ID[:12]})...")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT_DIR / "scripts" / "export_errors.py"),
            "--results",
            str(run_dir),
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT_DIR),
    )

    if result.returncode != 0:
        print(f"[WARN] export_errors.py failed:\n{result.stderr}")
    else:
        print(result.stdout.strip())
        print(f"  Saved: {output_path}")

    return output_path


# ---------------------------------------------------------------------------
# Run gpt-4o experiment
# ---------------------------------------------------------------------------


def run_gpt4o_experiment() -> str | None:
    """Run REAP with gpt-4o on 200 questions. Returns run_id or None."""
    if not GPT4O_CONFIG.exists():
        print(f"[ERROR] Config not found: {GPT4O_CONFIG}")
        return None

    print(f"\n{'─' * 60}")
    print("  Running REAP / gpt-4o / 200q (est. cost ~$5)")
    print(f"  Config: {GPT4O_CONFIG.name}")
    print(f"{'─' * 60}")

    cmd = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "run_experiment.py"),
        "--config",
        str(GPT4O_CONFIG),
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
            sys.stdout.write(f"    [gpt-4o] {line}")
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
        print("[FAILED] gpt-4o run failed.")
        return None

    run_dir = find_latest_run_dir()
    if run_dir is None:
        return None
    run_id = run_dir.name
    print(f"  [DONE] run_id: {run_id}")
    return run_id


# ---------------------------------------------------------------------------
# Find gpt-4o run (for --compare-only)
# ---------------------------------------------------------------------------


def find_gpt4o_run() -> Path | None:
    """Find a completed REAP/gpt-4o run in results."""
    results_root = ROOT_DIR / "results"
    candidates = []
    for d in results_root.iterdir():
        if not d.is_dir():
            continue
        summary_path = d / "summary.json"
        if not summary_path.exists():
            continue
        try:
            s = json.loads(summary_path.read_text())
        except Exception:
            continue
        if s.get("architecture") == "reap_rag" and s.get("model") == "gpt-4o":
            candidates.append((d.stat().st_mtime, d))
    if not candidates:
        return None
    return max(candidates)[1]


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------


def print_comparison(mini_em: float, gpt4o_em: float | None) -> None:
    print("\n" + "=" * 70)
    print("REAP MODEL-STRENGTH INVESTIGATION — RESULTS")
    print("=" * 70)
    print()
    print(f"  {'Source':<40} {'EM':>8}  {'vs mini'}")
    print(f"  {'-' * 40} {'-' * 8}  {'-' * 12}")
    print(f"  {'REAP paper (original)':<40} {PAPER_EM:>7.1%}  --")
    print(f"  {'Our REAP (gpt-4o-mini, 7405q)':<40} {mini_em:>7.1%}  baseline")

    if gpt4o_em is not None:
        delta = gpt4o_em - mini_em
        print(
            f"  {'Our REAP (gpt-4o, 200q)':<40} {gpt4o_em:>7.1%}  "
            f"{'+' if delta >= 0 else ''}{delta:.1%}"
        )

    print()

    # Interpretation
    paper_gap = PAPER_EM - mini_em
    print(f"  Gap to paper: {paper_gap:.1%}")

    if gpt4o_em is not None:
        model_contribution = gpt4o_em - mini_em
        remaining_gap = PAPER_EM - gpt4o_em
        pct_explained = model_contribution / paper_gap * 100 if paper_gap > 0 else 0
        print()
        print("  Hypothesis test (model-strength as primary cause):")
        print(f"    Model upgrade contribution:  {model_contribution:+.1%}")
        print(f"    Remaining gap to paper:       {remaining_gap:.1%}")
        print(f"    % of gap explained by model:  {pct_explained:.0f}%")
        print()

        if gpt4o_em >= 0.45:
            print(
                "  VERDICT: Model strength is the PRIMARY driver. gpt-4o reaches "
                f"{gpt4o_em:.1%} EM,\n"
                "  confirming that the implementation is sound and the gap is "
                "largely due\n"
                "  to using gpt-4o-mini vs the original paper's stronger model.\n"
                "  This validates the paper's framing as a controlled comparison."
            )
        elif gpt4o_em >= 0.35:
            print(
                "  VERDICT: Model strength is a PARTIAL driver. gpt-4o improves "
                f"to {gpt4o_em:.1%} EM\n"
                "  but significant gap remains. Some implementation or prompt "
                "differences\n"
                "  likely also contribute. Recommend reviewing decomposition prompts."
            )
        else:
            print(
                "  VERDICT: Model strength alone does NOT explain the gap. "
                f"gpt-4o only reaches {gpt4o_em:.1%} EM.\n"
                "  Implementation differences are likely significant. Review "
                "REAP decomposition\n"
                "  strategy and plan-execution logic before finalising paper claims."
            )
    else:
        print(
            "\n  gpt-4o experiment not yet run.\n"
            "  Run without --export-only to test the model-strength hypothesis."
        )

    print()
    print("  Paper framing guidance:")
    print(
        "  > 'We use gpt-4o-mini throughout for controlled cost comparison. "
        "REAP's 28.1% EM\n"
        "  > reflects this model constraint. A targeted gpt-4o probe (200q) "
        "yielded {gpt4o_em:.1%} EM,\n"
        "  > suggesting model capability accounts for ~{pct_explained:.0f}% of "
        "the gap to the original paper.'".format(
            gpt4o_em=gpt4o_em if gpt4o_em is not None else 0.0,
            pct_explained=(
                (gpt4o_em - mini_em) / (PAPER_EM - mini_em) * 100
                if gpt4o_em is not None and (PAPER_EM - mini_em) > 0
                else 0
            ),
        )
    )
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Investigate REAP implementation gap",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Only export errors from existing mini run; do not run gpt-4o",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Compare existing runs without running new experiments",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load mini results ─────────────────────────────────────────────────
    mini_run_dir = find_run_dir(BEST_MINI_RUN_ID)
    if mini_run_dir is None:
        print(f"[ERROR] Cannot find mini run {BEST_MINI_RUN_ID[:12]}")
        sys.exit(1)

    mini_summary = load_summary(mini_run_dir)
    mini_em: float = mini_summary["avg_exact_match"]

    print("=" * 70)
    print("REAP INVESTIGATION")
    print("=" * 70)
    print(f"  Baseline:  REAP/gpt-4o-mini  EM={mini_em:.1%}  (run {BEST_MINI_RUN_ID[:12]})")
    print(f"  Paper:     REAP (original)    EM={PAPER_EM:.1%}  ({PAPER_MODEL})")
    print(f"  Gap:       {PAPER_EM - mini_em:.1%}")
    print()

    # ── Export errors from mini ───────────────────────────────────────────
    error_csv = export_errors_from_mini(OUTPUT_DIR)

    if args.export_only:
        print_comparison(mini_em, None)
        print(f"  Error CSV: {error_csv}")
        return

    # ── Compare-only: find existing gpt-4o run ────────────────────────────
    if args.compare_only:
        gpt4o_run_dir = find_gpt4o_run()
        if gpt4o_run_dir is None:
            print(
                "[INFO] No completed REAP/gpt-4o run found. "
                "Run without --compare-only to create one."
            )
            print_comparison(mini_em, None)
        else:
            gpt4o_summary = load_summary(gpt4o_run_dir)
            gpt4o_em: float = gpt4o_summary["avg_exact_match"]
            print(
                f"  Found gpt-4o run: {gpt4o_run_dir.name[:12]}  "
                f"(n={gpt4o_summary['num_questions']})"
            )
            print_comparison(mini_em, gpt4o_em)
        return

    # ── Full run ──────────────────────────────────────────────────────────
    import os

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    run_id = run_gpt4o_experiment()
    if run_id is None:
        print("[FAILED] Could not complete gpt-4o run.")
        print_comparison(mini_em, None)
        return

    # Load gpt-4o results
    gpt4o_run_dir = find_run_dir(run_id[:12])
    if gpt4o_run_dir is None:
        print(f"[WARN] Could not locate gpt-4o run directory for {run_id[:12]}")
        print_comparison(mini_em, None)
        return

    gpt4o_summary = load_summary(gpt4o_run_dir)
    gpt4o_em = gpt4o_summary["avg_exact_match"]

    # Save run_id for future reference
    record_path = OUTPUT_DIR / "gpt4o_run_id.txt"
    record_path.write_text(run_id)
    print(f"  Run ID saved: {record_path}")

    print_comparison(mini_em, gpt4o_em)

    # Export errors from gpt-4o run too
    gpt4o_error_path = OUTPUT_DIR / "reap_gpt4o_errors.csv"
    subprocess.run(
        [
            sys.executable,
            str(ROOT_DIR / "scripts" / "export_errors.py"),
            "--results",
            str(gpt4o_run_dir),
            "--output",
            str(gpt4o_error_path),
        ],
        cwd=str(ROOT_DIR),
    )
    print(f"\n  gpt-4o errors exported: {gpt4o_error_path}")
    print(f"  mini errors exported:   {error_csv}")
    print("\nNext: open both CSVs and compare failure modes between models.")


if __name__ == "__main__":
    main()
