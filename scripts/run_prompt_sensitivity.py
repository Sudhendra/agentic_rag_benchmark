#!/usr/bin/env python3
"""Run prompt sensitivity study across all three prompt-sensitive architectures.

Executes multiple prompt variants for RLM (5 variants), ReAct (3 variants),
and IRCoT (3 variants) on 50 HotpotQA questions each, then produces a
cross-architecture "prompt tax" comparison report.

The prompt tax — the performance swing caused purely by wording changes —
is one of the paper's key novel findings. This script generates the data
behind that result.

Usage:
    python scripts/run_prompt_sensitivity.py                    # All 3 architectures
    python scripts/run_prompt_sensitivity.py --dry-run          # Show plan, no API calls
    python scripts/run_prompt_sensitivity.py --arch rlm         # Single architecture
    python scripts/run_prompt_sensitivity.py --arch react ircot # Two architectures
    python scripts/run_prompt_sensitivity.py --skip-existing    # Skip completed runs
    python scripts/run_prompt_sensitivity.py --analyze-only     # Report from saved data
    python scripts/run_prompt_sensitivity.py --subset 200       # Larger run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv

load_dotenv()

from scripts.analyze_results import compare_runs, find_run_directories, load_results

# ---------------------------------------------------------------------------
# Variant registry
# Each entry: key (short id), name (display), description, config path.
# ---------------------------------------------------------------------------

RLM_VARIANTS: list[dict] = [
    {
        "key": "v0",
        "name": "Baseline",
        "description": "Standard instructions with format guidelines",
        "config": "configs/sensitivity/rlm_v0_baseline.yaml",
        "arch": "rlm",
    },
    {
        "key": "v1",
        "name": "Minimalist",
        "description": "Stripped-down, minimal instructions only",
        "config": "configs/sensitivity/rlm_v1_minimalist.yaml",
        "arch": "rlm",
    },
    {
        "key": "v2",
        "name": "Structured",
        "description": "Explicit step-by-step reasoning scaffolding",
        "config": "configs/sensitivity/rlm_v2_structured.yaml",
        "arch": "rlm",
    },
    {
        "key": "v3",
        "name": "Persona",
        "description": "Expert PhD researcher persona framing",
        "config": "configs/sensitivity/rlm_v3_persona.yaml",
        "arch": "rlm",
    },
    {
        "key": "v4",
        "name": "Strict",
        "description": "Heavy output-format constraints with negative examples",
        "config": "configs/sensitivity/rlm_v4_strict.yaml",
        "arch": "rlm",
    },
]

REACT_VARIANTS: list[dict] = [
    {
        "key": "v0",
        "name": "Baseline",
        "description": "Standard ReAct with few-shot examples",
        "config": "configs/sensitivity/react_v0_baseline.yaml",
        "arch": "react",
    },
    {
        "key": "v1",
        "name": "Minimalist",
        "description": "Minimal instructions, no examples",
        "config": "configs/sensitivity/react_v1_minimalist.yaml",
        "arch": "react",
    },
    {
        "key": "v2",
        "name": "Strict",
        "description": "Explicit format rules with step limit and negative examples",
        "config": "configs/sensitivity/react_v2_strict.yaml",
        "arch": "react",
    },
]

IRCOT_VARIANTS: list[dict] = [
    {
        "key": "v0",
        "name": "Baseline",
        "description": "Standard IRCoT with grounding constraints",
        "config": "configs/sensitivity/ircot_v0_baseline.yaml",
        "arch": "ircot",
    },
    {
        "key": "v1",
        "name": "Minimalist",
        "description": "Minimal one-liner instruction",
        "config": "configs/sensitivity/ircot_v1_minimalist.yaml",
        "arch": "ircot",
    },
    {
        "key": "v2",
        "name": "Strict",
        "description": "Explicit format constraints with negative examples",
        "config": "configs/sensitivity/ircot_v2_strict.yaml",
        "arch": "ircot",
    },
]

ARCH_VARIANTS: dict[str, list[dict]] = {
    "rlm": RLM_VARIANTS,
    "react": REACT_VARIANTS,
    "ircot": IRCOT_VARIANTS,
}

ARCH_DISPLAY: dict[str, str] = {
    "rlm": "Recursive LM",
    "react": "ReAct",
    "ircot": "IRCoT",
}

RESULTS_DIR = ROOT_DIR / "results" / "sensitivity"
PROGRESS_FILE = RESULTS_DIR / "progress.json"


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------


def _load_progress() -> dict[str, str]:
    """Return {config_stem: run_id} for completed runs."""
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
    if not (ROOT_DIR / "results").exists():
        return None
    run_dirs = [
        d
        for d in (ROOT_DIR / "results").iterdir()
        if d.is_dir() and d.name != "sensitivity" and (d / "summary.json").exists()
    ]
    return max(run_dirs, key=lambda d: d.stat().st_ctime) if run_dirs else None


def _get_run_id(run_dir: Path | None) -> str | None:
    if run_dir is None:
        return None
    summary = run_dir / "summary.json"
    try:
        data = json.loads(summary.read_text())
        return data.get("run_id") or run_dir.name[:12]
    except Exception:
        return run_dir.name[:12]


# ---------------------------------------------------------------------------
# Dry run output
# ---------------------------------------------------------------------------


def dry_run(selected_archs: list[str], subset: int | None) -> None:
    print("=" * 70)
    print("PROMPT SENSITIVITY STUDY — DRY RUN")
    print("=" * 70)

    total_variants = sum(len(ARCH_VARIANTS[a]) for a in selected_archs)
    total_questions = total_variants * (subset or 50)
    est_cost = 0.0

    for arch in selected_archs:
        variants = ARCH_VARIANTS[arch]
        print(f"\n  [{ARCH_DISPLAY[arch]}]  {len(variants)} variants")
        for v in variants:
            config_path = ROOT_DIR / v["config"]
            prompt_path = _resolve_prompt_path(v["config"], arch)
            config_ok = "OK" if config_path.exists() else "MISSING"
            prompt_ok = "OK" if (prompt_path and prompt_path.exists()) else "MISSING"
            cost = _estimate_variant_cost(arch, subset or 50)
            est_cost += cost
            print(
                f"    [{v['key']}] {v['name']:<14}  "
                f"config:[{config_ok}]  prompt:[{prompt_ok}]  ~${cost:.3f}"
            )

    print(f"\n  Total variants:   {total_variants}")
    print(f"  Total questions:  {total_questions}")
    print(f"  Estimated cost:   ~${est_cost:.2f}")
    print(f"  Results dir:      {RESULTS_DIR}")
    print(f"  OPENAI_API_KEY:   {'SET' if os.environ.get('OPENAI_API_KEY') else 'NOT SET'}")
    print()


def _resolve_prompt_path(config_path_str: str, arch: str) -> Path | None:
    config_path = ROOT_DIR / config_path_str
    if not config_path.exists():
        return None
    import yaml

    try:
        data = yaml.safe_load(config_path.read_text()) or {}
    except Exception:
        return None
    # Each arch stores prompt under its own key
    arch_key_map = {"rlm": "rlm", "react": "react", "ircot": "ircot"}
    section_key = arch_key_map.get(arch, arch)
    section = data.get(section_key, {}) or {}
    prompt_path_str = section.get("prompt_path")
    if prompt_path_str:
        return ROOT_DIR / prompt_path_str
    return None


def _estimate_variant_cost(arch: str, n_questions: int) -> float:
    """Rough per-variant cost on gpt-4o-mini."""
    cost_per_q = {"rlm": 0.003, "react": 0.006, "ircot": 0.002}
    return n_questions * cost_per_q.get(arch, 0.003)


# ---------------------------------------------------------------------------
# Single-variant runner
# ---------------------------------------------------------------------------


def run_variant(variant: dict, n: int, total: int, subset_override: int | None) -> str | None:
    """Run one variant config. Returns run_id on success, None on failure."""
    key = variant["key"]
    name = variant["name"]
    arch = variant["arch"]
    config_path = ROOT_DIR / variant["config"]

    label = f"[{n}/{total}] {ARCH_DISPLAY[arch]} / {name}"
    print(f"\n{'─' * 60}")
    print(f"  {label}  ({variant['description']})")

    if not config_path.exists():
        print(f"  [SKIP] Config not found: {config_path}")
        return None

    cmd = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "run_experiment.py"),
        "--config",
        str(config_path),
    ]
    if subset_override is not None:
        cmd += ["--subset", str(subset_override)]

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
            sys.stdout.write(f"    [{key}] {line}")
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


def _collect_arch_results(arch: str, progress: dict[str, str]) -> list[dict]:
    """Load summary rows for all completed variants of an architecture."""
    variants = ARCH_VARIANTS[arch]
    rows = []
    for v in variants:
        key = Path(v["config"]).stem
        run_id_prefix = progress.get(key)
        if not run_id_prefix:
            continue
        # Find matching results directory
        results_root = ROOT_DIR / "results"
        matched: list[Path] = []
        for d in results_root.iterdir():
            if d.is_dir() and (d / "summary.json").exists():
                if d.name.startswith(run_id_prefix) or run_id_prefix in d.name:
                    matched.append(d)
        if not matched:
            # Fall back: try sensitivity subdirectory copy
            sens_dir = RESULTS_DIR / arch / key
            if sens_dir.exists() and (sens_dir / "summary.json").exists():
                matched = [sens_dir]
        if not matched:
            continue
        run_dir = matched[0]
        try:
            results = load_results(run_dir)
            summary = results["summary"]
            rows.append(
                {
                    "variant_key": v["key"],
                    "variant_name": v["name"],
                    "arch": arch,
                    "run_id": run_dir.name[:12],
                    "num_questions": summary.get("num_questions", 0),
                    "exact_match": summary.get("avg_exact_match", 0.0),
                    "f1": summary.get("avg_f1", 0.0),
                    "cost_usd": summary.get("total_cost_usd", 0.0),
                    "latency_ms": summary.get("avg_latency_ms", 0.0),
                    "tokens_per_q": summary.get("avg_tokens_per_question", 0.0),
                }
            )
        except Exception as e:
            print(f"  [WARN] Could not load {run_dir}: {e}", file=sys.stderr)
    return rows


def _variance_stats(values: list[float]) -> dict[str, float]:
    if len(values) < 2:
        return {"mean": values[0] if values else 0.0, "std": 0.0, "range": 0.0, "cv": 0.0}
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    std = math.sqrt(variance)
    rng = max(values) - min(values)
    cv = std / mean if mean > 0 else 0.0
    return {"mean": mean, "std": std, "range": rng, "cv": cv}


def print_arch_table(arch: str, rows: list[dict]) -> None:
    """Print sensitivity table for one architecture."""
    if not rows:
        print(f"\n  No results yet for {ARCH_DISPLAY[arch]}.")
        return

    print(f"\n{'─' * 70}")
    print(f"  {ARCH_DISPLAY[arch]}  ({len(rows)} variants)")
    print(f"{'─' * 70}")
    print(f"  {'Variant':<14} {'EM':>7} {'F1':>7} {'Cost':>9} {'Latency':>10} {'Tok/Q':>7}")
    print(f"  {'-' * 14} {'-' * 7} {'-' * 7} {'-' * 9} {'-' * 10} {'-' * 7}")

    ems = []
    f1s = []
    for r in rows:
        em = r["exact_match"]
        f1 = r["f1"]
        ems.append(em)
        f1s.append(f1)
        print(
            f"  {r['variant_name']:<14} {em:>6.1%} {f1:>6.1%}"
            f"  ${r['cost_usd']:>7.3f} {r['latency_ms']:>9.0f}ms {r['tokens_per_q']:>6.0f}"
        )

    if len(ems) >= 2:
        em_stats = _variance_stats(ems)
        f1_stats = _variance_stats(f1s)
        best_name = rows[ems.index(max(ems))]["variant_name"]
        worst_name = rows[ems.index(min(ems))]["variant_name"]
        print(
            f"\n  Prompt Tax (EM swing):  {em_stats['range']:.1%}  "
            f"(best: {best_name} {max(ems):.1%}  worst: {worst_name} {min(ems):.1%})"
        )
        print(f"  Prompt Tax (F1 swing):  {f1_stats['range']:.1%}")
        print(f"  EM Std Dev:             {em_stats['std']:.1%}")
        print(f"  EM Coef of Variation:   {em_stats['cv']:.2f}")


def print_cross_arch_summary(all_results: dict[str, list[dict]]) -> None:
    """Print the headline cross-architecture prompt-tax comparison table."""
    print("\n" + "=" * 70)
    print("PROMPT TAX — CROSS-ARCHITECTURE SUMMARY")
    print("=" * 70)
    print(
        f"\n  {'Architecture':<16} {'Variants':>8} {'EM Swing':>10} {'F1 Swing':>10}"
        f"  {'Best Variant':<14} {'Worst Variant':<14}"
    )
    print(f"  {'-' * 16} {'-' * 8} {'-' * 10} {'-' * 10}  {'-' * 14} {'-' * 14}")

    for arch in ["rlm", "react", "ircot"]:
        rows = all_results.get(arch, [])
        if len(rows) < 2:
            print(f"  {ARCH_DISPLAY[arch]:<16} {'—':>8}  (insufficient data)")
            continue
        ems = [r["exact_match"] for r in rows]
        f1s = [r["f1"] for r in rows]
        em_swing = max(ems) - min(ems)
        f1_swing = max(f1s) - min(f1s)
        best = rows[ems.index(max(ems))]["variant_name"]
        worst = rows[ems.index(min(ems))]["variant_name"]
        print(
            f"  {ARCH_DISPLAY[arch]:<16} {len(rows):>8} {em_swing:>9.1%} {f1_swing:>9.1%}"
            f"  {best:<14} {worst:<14}"
        )

    print()
    print("  Interpretation:")
    print("  A large EM swing means the architecture is brittle to prompt wording.")
    print("  A small swing means the architecture is robust (low 'prompt tax').")


def export_json(all_results: dict[str, list[dict]], progress: dict[str, str]) -> Path:
    """Export all sensitivity results to JSON for figure generation."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "sensitivity_results.json"

    export: dict[str, object] = {}
    for arch, rows in all_results.items():
        if not rows:
            continue
        ems = [r["exact_match"] for r in rows]
        f1s = [r["f1"] for r in rows]
        em_stats = _variance_stats(ems) if ems else {}
        f1_stats = _variance_stats(f1s) if f1s else {}
        export[arch] = {
            "display_name": ARCH_DISPLAY[arch],
            "num_variants": len(rows),
            "variants": {
                r["variant_key"]: {
                    "name": r["variant_name"],
                    "exact_match": r["exact_match"],
                    "f1": r["f1"],
                    "cost_usd": r["cost_usd"],
                    "latency_ms": r["latency_ms"],
                    "tokens_per_q": r["tokens_per_q"],
                    "num_questions": r["num_questions"],
                }
                for r in rows
            },
            "summary": {
                "em_mean": em_stats.get("mean", 0),
                "em_std": em_stats.get("std", 0),
                "em_range": em_stats.get("range", 0),
                "em_cv": em_stats.get("cv", 0),
                "f1_mean": f1_stats.get("mean", 0),
                "f1_std": f1_stats.get("std", 0),
                "f1_range": f1_stats.get("range", 0),
                "f1_cv": f1_stats.get("cv", 0),
            },
        }

    output_path.write_text(json.dumps(export, indent=2))
    print(f"\n  Exported: {output_path}")
    return output_path


def analyze_existing(selected_archs: list[str], progress: dict[str, str]) -> dict[str, list[dict]]:
    """Load and report on all completed variant runs."""
    all_results: dict[str, list[dict]] = {}

    print("\nAnalyzing existing results...")
    for arch in selected_archs:
        rows = _collect_arch_results(arch, progress)
        all_results[arch] = rows
        print_arch_table(arch, rows)

    if any(all_results.values()):
        print_cross_arch_summary(all_results)
        export_json(all_results, progress)
    else:
        print("\n  No results found. Run experiments first.")

    return all_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prompt sensitivity study for RLM, ReAct, and IRCoT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--arch",
        nargs="+",
        choices=["rlm", "react", "ircot"],
        default=["rlm", "react", "ircot"],
        help="Architectures to study (default: all three)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would run without making API calls",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip variants that are already in the progress file",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Print report from existing results without running anything",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Override number of questions per variant (default: from config, usually 50)",
    )
    args = parser.parse_args()

    # Deduplicate while preserving order
    selected_archs: list[str] = list(dict.fromkeys(args.arch))

    if args.dry_run:
        dry_run(selected_archs, args.subset)
        return

    progress = _load_progress()

    if args.analyze_only:
        analyze_existing(selected_archs, progress)
        return

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        print("  Set it with:  set OPENAI_API_KEY=sk-...", file=sys.stderr)
        sys.exit(1)

    # ── Build flat list of variants to run ───────────────────────────────────
    to_run: list[dict] = []
    for arch in selected_archs:
        for v in ARCH_VARIANTS[arch]:
            key = Path(v["config"]).stem
            if args.skip_existing and key in progress:
                continue
            to_run.append(v)

    total = len(to_run)
    if total == 0:
        print("Nothing to run — all variants already complete.")
        print("  Use --analyze-only to see results.")
        analyze_existing(selected_archs, progress)
        return

    # ── Summary header ────────────────────────────────────────────────────────
    print("=" * 70)
    print("PROMPT SENSITIVITY STUDY")
    print(f"  Architectures:  {', '.join(ARCH_DISPLAY[a] for a in selected_archs)}")
    print(f"  Variants:       {total}")
    est = sum(_estimate_variant_cost(v["arch"], args.subset or 50) for v in to_run)
    print(f"  Est. cost:      ~${est:.2f}")
    print(f"  OPENAI_API_KEY: SET")
    print("=" * 70)

    # ── Run variants ──────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for i, variant in enumerate(to_run, 1):
        key = Path(variant["config"]).stem
        try:
            run_id = run_variant(variant, i, total, args.subset)
        except KeyboardInterrupt:
            print("\n\n[ABORTED] Saving progress...")
            _save_progress(progress)
            print(f"  Resume with: python scripts/run_prompt_sensitivity.py --skip-existing")
            sys.exit(1)

        if run_id:
            progress[key] = run_id
            _save_progress(progress)

            # Copy results to named sensitivity dir for easy retrieval
            run_dir = _find_latest_run_dir()
            if run_dir:
                import shutil

                arch_dir = RESULTS_DIR / variant["arch"] / key
                if not arch_dir.exists():
                    try:
                        shutil.copytree(str(run_dir), str(arch_dir))
                    except Exception:
                        pass  # non-fatal

    # ── Final report ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ALL VARIANTS COMPLETE — GENERATING REPORT")
    print("=" * 70)
    analyze_existing(selected_archs, progress)

    total_cost = sum(
        r["cost_usd"] for arch in selected_archs for r in _collect_arch_results(arch, progress)
    )
    print(f"\nTotal experiment cost: ${total_cost:.4f}")
    print(f"Results JSON:          {RESULTS_DIR / 'sensitivity_results.json'}")
    print(f"To re-run report:      python scripts/run_prompt_sensitivity.py --analyze-only")


if __name__ == "__main__":
    main()
