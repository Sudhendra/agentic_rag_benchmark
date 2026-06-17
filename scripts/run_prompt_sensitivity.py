#!/usr/bin/env python3
"""Run RLM prompt sensitivity study across 5 prompt variants.

Usage:
    python scripts/run_prompt_sensitivity.py                     # Run all 5 variants
    python scripts/run_prompt_sensitivity.py --dry-run           # Show what would run
    python scripts/run_prompt_sensitivity.py --variant v0        # Run single variant
    python scripts/run_prompt_sensitivity.py --skip-existing     # Skip completed runs
    python scripts/run_prompt_sensitivity.py --analyze-only      # Just analyze existing results
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

load_dotenv()

from scripts.analyze_results import (
    breakdown_by_question_type,
    compare_runs,
    find_run_directories,
    load_results,
)

VARIANTS = [
    {
        "key": "v0",
        "name": "Baseline",
        "description": "Current rlm.txt as-is",
        "config": "configs/sensitivity/rlm_v0_baseline.yaml",
    },
    {
        "key": "v1",
        "name": "Minimalist",
        "description": "Stripped down, minimal instructions",
        "config": "configs/sensitivity/rlm_v1_minimalist.yaml",
    },
    {
        "key": "v2",
        "name": "Structured",
        "description": "Step-by-step reasoning scaffolding",
        "config": "configs/sensitivity/rlm_v2_structured.yaml",
    },
    {
        "key": "v3",
        "name": "Persona",
        "description": "Expert PhD persona framing",
        "config": "configs/sensitivity/rlm_v3_persona.yaml",
    },
    {
        "key": "v4",
        "name": "Strict",
        "description": "Heavy output format constraints, negative examples",
        "config": "configs/sensitivity/rlm_v4_strict.yaml",
    },
]

RESULTS_DIR = ROOT_DIR / "results" / "sensitivity"


def dry_run() -> None:
    print("=" * 60)
    print("PROMPT SENSITIVITY STUDY — DRY RUN")
    print("=" * 60)
    for v in VARIANTS:
        config_path = ROOT_DIR / v["config"]
        config_status = "EXISTS" if config_path.exists() else "MISSING"
        prompt_path = _get_prompt_path(v["config"])
        prompt_status = "EXISTS" if prompt_path and prompt_path.exists() else "MISSING"
        print(f"\n  [{v['key']}] {v['name']}: {v['description']}")
        print(f"         Config:  {config_path}  [{config_status}]")
        print(f"         Prompt:  {prompt_path}  [{prompt_status}]")
    print()
    expected_cost = len(VARIANTS) * _estimate_run_cost()
    print(f"Estimated cost: ~${expected_cost:.2f} (all 5 variants x 50 questions)")
    print(f"Results dir:    {RESULTS_DIR}")
    print(f"OPENAI_API_KEY: {'SET' if os.environ.get('OPENAI_API_KEY') else 'NOT SET'}")
    print()


def _get_prompt_path(config_path_str: str) -> Path | None:
    config_path = ROOT_DIR / config_path_str
    if not config_path.exists():
        return None
    import yaml

    try:
        data = yaml.safe_load(config_path.read_text())
    except Exception:
        return None
    # Resolve prompt_path from config
    prompt_path = None
    if data:
        rlm = data.get("rlm", {})
        prompt_path_str = rlm.get("prompt_path")
        if prompt_path_str:
            prompt_path = ROOT_DIR / prompt_path_str
    return prompt_path


def _estimate_run_cost() -> float:
    """Estimate cost for one variant run (50 questions, RLM, gpt-4o-mini).
    RLM makes ~5-15 LLM calls per question at ~$0.002/question for gpt-4o-mini.
    """
    return 50 * 0.003  # $0.003 per question for RLM on gpt-4o-mini


def run_variant(variant: dict) -> Path | None:
    """Run a single prompt variant experiment. Returns run directory path or None."""
    key = variant["key"]
    name = variant["name"]
    config_path = ROOT_DIR / variant["config"]

    if not config_path.exists():
        print(f"  [SKIP] Config not found: {config_path}")
        return None

    print(f"\n  [{key}] Running {name}...")
    print(f"         Config: {config_path}\n")

    cmd = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "run_experiment.py"),
        "--config",
        str(config_path),
    ]

    exit_code = 0
    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        for line in iter(process.stdout.readline, ""):
            if not line:
                break
            sys.stdout.write(f"  [{key}] {line}")
            sys.stdout.flush()
        process.wait()
        exit_code = process.returncode
        print(f"\n  [{key}] Exit code: {exit_code}")
    except subprocess.TimeoutExpired:
        print(f"  [{key}] [TIMEOUT] Exceeded 600s")
        process.kill()
        return None
    except Exception as e:
        print(f"  [{key}] [ERROR] {e}")
        return None

    if exit_code != 0:
        return None

    return _find_latest_run_dir()


def _find_latest_run_dir() -> Path | None:
    """Find the most recently created results directory."""
    results_root = ROOT_DIR / "results"
    if not results_root.exists():
        return None
    run_dirs = [d for d in results_root.iterdir() if d.is_dir() and (d / "summary.json").exists()]
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda d: d.stat().st_ctime)


def collect_results() -> list[dict]:
    """Collect results from all completed variant runs."""
    run_dirs = find_run_directories(RESULTS_DIR) if RESULTS_DIR.exists() else []
    if not run_dirs:
        # Also check for any sensitivity runs in the main results directory
        print("  No sensitivity results found in results/sensitivity/")
        print("  Looking for recently created run directories...")
        main_results = ROOT_DIR / "results"
        if main_results.exists():
            all_dirs = find_run_directories(main_results)
            # Filter by runs containing sensitivity names
            run_dirs = [
                d
                for d in all_dirs
                if any(v["key"] in d.name or v["name"].lower() in d.name for v in VARIANTS)
            ]
            if run_dirs:
                print(f"  Found {len(run_dirs)} in main results directory")

    if not run_dirs:
        # Fall back to loading via config name
        return _load_by_config_names()

    rows = compare_runs(run_dirs, compute_stats=True)
    return rows


def _load_by_config_names() -> list[dict]:
    """Fallback: scan all results directories and match by config name."""
    results_root = ROOT_DIR / "results"
    if not results_root.exists():
        return []
    rows = []
    for d in results_root.iterdir():
        if not d.is_dir():
            continue
        summary_path = d / "summary.json"
        if not summary_path.exists():
            continue
        # Check if the resolved_config has sensitivity prompt
        config_path = d / "resolved_config.yaml"
        if config_path.exists():
            try:
                import yaml

                config = yaml.safe_load(config_path.read_text())
                prompt = (config or {}).get("rlm", {}).get("prompt_path", "")
                if "sensitivity" in str(prompt):
                    results = load_results(d)
                    rows.append(
                        {
                            "run_id": d.name[:12],
                            "architecture": results["summary"].get("architecture", "unknown"),
                            "model": results["summary"].get("model", "unknown"),
                            "num_questions": results["summary"].get("num_questions", 0),
                            "exact_match": results["summary"].get("avg_exact_match", 0),
                            "f1": results["summary"].get("avg_f1", 0),
                            "latency_ms": results["summary"].get("avg_latency_ms", 0),
                            "tokens_per_q": results["summary"].get("avg_tokens_per_question", 0),
                            "cost_usd": results["summary"].get("total_cost_usd", 0),
                        }
                    )
            except Exception:
                pass
    return rows


def print_sensitivity_table(rows: list[dict]) -> None:
    """Print formatted sensitivity comparison table with variance stats."""
    if not rows:
        print("\n  No results found. Run the variants first.")
        return

    print("\n" + "=" * 80)
    print("PROMPT SENSITIVITY STUDY — RESULTS")
    print("=" * 80)

    # Sort by variant key for consistent ordering
    variant_keys = [v["key"] for v in VARIANTS]
    key_to_name = {v["key"]: v["name"] for v in VARIANTS}
    rows.sort(key=lambda r: _extract_variant_key(r.get("run_id", ""), variant_keys))

    print(
        f"\n{'Variant':<15} {'EM':>8} {'F1':>8} {'F1 CI':>16} {'Cost':>10} {'Latency':>10} {'Tok/Q':>8}"
    )
    print("-" * 75)

    ems = []
    f1s = []
    costs = []
    for r in rows:
        variant_key = _extract_variant_key(r.get("run_id", ""), variant_keys)
        variant_name = key_to_name.get(variant_key, r.get("run_id", "?"))
        em = r.get("exact_match", 0)
        f1 = r.get("f1", 0)
        ci = f"({r.get('f1_ci_lower', 0):.3f}-{r.get('f1_ci_upper', 0):.3f})"
        cost = r.get("cost_usd", 0)
        latency = r.get("latency_ms", 0)
        tokens = r.get("tokens_per_q", 0)
        print(
            f"{variant_name:<15} {em:>7.1%} {f1:>7.1%} {ci:>16} ${cost:<7.3f} {latency:>8.0f}ms {tokens:>6.0f}"
        )
        ems.append(em)
        f1s.append(f1)
        costs.append(cost)

    # Compute variance statistics
    print(f"\n{'-' * 75}")
    print("VARIANCE ANALYSIS")
    print(f"{'-' * 75}")

    if len(ems) >= 2:
        em_range = max(ems) - min(ems)
        f1_range = max(f1s) - min(f1s)
        em_mean = sum(ems) / len(ems)
        f1_mean = sum(f1s) / len(f1s)
        em_var = sum((x - em_mean) ** 2 for x in ems) / len(ems)
        f1_var = sum((x - f1_mean) ** 2 for x in f1s) / len(f1s)
        em_std = math.sqrt(em_var)
        f1_std = math.sqrt(f1_var)
        em_cv = em_std / em_mean if em_mean > 0 else 0
        f1_cv = f1_std / f1_mean if f1_mean > 0 else 0

        print(f"{'EM Range:':<20} {em_range:>7.1%}")
        print(f"{'F1 Range:':<20} {f1_range:>7.1%}")
        print(f"{'EM Std Dev:':<20} {em_std:>7.1%}")
        print(f"{'F1 Std Dev:':<20} {f1_std:>7.1%}")
        print(f"{'EM Coef of Variation:':<20} {em_cv:>7.2f}")
        print(f"{'F1 Coef of Variation:':<20} {f1_cv:>7.2f}")

        # Identify best and worst
        best_idx = ems.index(max(ems))
        worst_idx = ems.index(min(ems))
        best_name = key_to_name.get(
            _extract_variant_key(rows[best_idx].get("run_id", ""), variant_keys), "?"
        )
        worst_name = key_to_name.get(
            _extract_variant_key(rows[worst_idx].get("run_id", ""), variant_keys), "?"
        )
        swing = max(ems) - min(ems)
        print(f"\n{'Best variant:':<20} {best_name} ({max(ems):.1%} EM)")
        print(f"{'Worst variant:':<20} {worst_name} ({min(ems):.1%} EM)")
        print(f"{'Swing (Max-Min):':<20} {swing:.1%} EM")

    # Cost analysis
    if len(costs) >= 2:
        max_cost = max(costs)
        min_cost = min(costs)
        print(f"\n{'Cost range:':<20} ${min_cost:.3f} — ${max_cost:.3f}")
        print(f"{'Cost ratio:':<20} {max_cost / min_cost:.1f}x" if min_cost > 0 else "")


def _extract_variant_key(run_id: str, known_keys: list[str]) -> str:
    """Try to extract variant key from run_id."""
    run_lower = run_id.lower()
    for key in known_keys:
        if key in run_lower:
            return key
    # Check for variant descriptions
    name_to_key = {v["name"].lower(): v["key"] for v in VARIANTS}
    for name, key in name_to_key.items():
        if name in run_lower:
            return key
    return run_id


def export_sensitivity_json(rows: list[dict]) -> Path:
    """Export sensitivity results as JSON for notebook consumption."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "sensitivity_results.json"

    variant_keys = [v["key"] for v in VARIANTS]
    variants_data = {}
    for r in rows:
        key = _extract_variant_key(r.get("run_id", ""), variant_keys)
        name = next((v["name"] for v in VARIANTS if v["key"] == key), key)
        variants_data[key] = {
            "name": name,
            "exact_match": r.get("exact_match", 0),
            "f1": r.get("f1", 0),
            "f1_ci_lower": r.get("f1_ci_lower", 0),
            "f1_ci_upper": r.get("f1_ci_upper", 0),
            "cost_usd": r.get("cost_usd", 0),
            "latency_ms": r.get("latency_ms", 0),
            "tokens_per_q": r.get("tokens_per_q", 0),
            "num_questions": r.get("num_questions", 0),
        }

    # Compute summary stats
    ems = [v["exact_match"] for v in variants_data.values() if v["exact_match"] > 0]
    f1s = [v["f1"] for v in variants_data.values() if v["f1"] > 0]
    summary = {}
    if ems:
        em_mean = sum(ems) / len(ems)
        em_std = math.sqrt(sum((x - em_mean) ** 2 for x in ems) / len(ems))
        summary["em_mean"] = em_mean
        summary["em_std"] = em_std
        summary["em_range"] = max(ems) - min(ems)
        summary["em_cv"] = em_std / em_mean if em_mean > 0 else 0
    if f1s:
        f1_mean = sum(f1s) / len(f1s)
        f1_std = math.sqrt(sum((x - f1_mean) ** 2 for x in f1s) / len(f1s))
        summary["f1_mean"] = f1_mean
        summary["f1_std"] = f1_std
        summary["f1_range"] = max(f1s) - min(f1s)
        summary["f1_cv"] = f1_std / f1_mean if f1_mean > 0 else 0

    output = {
        "variants": variants_data,
        "summary": summary,
        "num_variants": sum(1 for v in variants_data.values() if v["exact_match"] > 0),
    }

    output_path.write_text(json.dumps(output, indent=2))
    print(f"\n  Exported: {output_path}")
    return output_path


def analyze_existing() -> list[dict]:
    """Analyze existing sensitivity run results without running new experiments."""
    print("\nAnalyzing existing results...")
    rows = collect_results()

    if rows:
        print_sensitivity_table(rows)
        export_sensitivity_json(rows)
    else:
        print("  No existing sensitivity results found.")
        print("  Run `python scripts/run_prompt_sensitivity.py` to generate them.")

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RLM prompt sensitivity study across 5 prompt variants"
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would run")
    parser.add_argument(
        "--variant", choices=[v["key"] for v in VARIANTS], help="Run single variant"
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip completed runs")
    parser.add_argument("--analyze-only", action="store_true", help="Just analyze existing results")
    parser.add_argument(
        "--subset", type=int, default=None, help="Override subset size for all variants"
    )
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
        return

    if args.analyze_only:
        analyze_existing()
        return

    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. Set it first or use --dry-run.")
        print("  export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    # Determine which variants to run
    selected_variants = VARIANTS
    if args.variant:
        selected_variants = [v for v in VARIANTS if v["key"] == args.variant]

    # Override subset size if specified
    if args.subset is not None:
        for v in selected_variants:
            config_path = ROOT_DIR / v["config"]
            if config_path.exists():
                import yaml

                config = yaml.safe_load(config_path.read_text())
                data = config.get("data", {})
                orig_subset = data.get("subset_size", "?")
                data["subset_size"] = args.subset
                config["data"] = data
                config_path.write_text(yaml.safe_dump(config))
                print(f"  [{v['key']}] Overrode subset_size: {orig_subset} -> {args.subset}")

    print("=" * 60)
    print("PROMPT SENSITIVITY STUDY")
    print(f"Variants: {len(selected_variants)}")
    print(f"API Key:  {'SET' if os.environ.get('OPENAI_API_KEY') else 'NOT SET'}")
    print("=" * 60)

    # Run each variant sequentially
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for variant in selected_variants:
        run_dir = run_variant(variant)
        if run_dir:
            # Move or symlink results to sensitivity directory
            target_dir = RESULTS_DIR / variant["key"]
            try:
                import shutil

                if not target_dir.exists():
                    shutil.copytree(run_dir, target_dir)
                    print(f"         Results copied to: {target_dir}")
            except Exception as e:
                print(f"         [WARN] Could not copy results: {e}")

    # Collect and display results
    rows = analyze_existing()
    if rows:
        export_sensitivity_json(rows)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

    if len(selected_variants) == len(VARIANTS):
        total_cost = sum(r.get("cost_usd", 0) for r in rows)
        print(f"\nTotal experiment cost: ${total_cost:.4f}")
        print(f"To run more questions: python scripts/run_prompt_sensitivity.py --subset 200")
        print(f"To analyze only:       python scripts/run_prompt_sensitivity.py --analyze-only")
        print(f"To run single variant: python scripts/run_prompt_sensitivity.py --variant v1")


if __name__ == "__main__":
    main()
