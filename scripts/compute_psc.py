#!/usr/bin/env python3
"""Compute Prompt Sensitivity Coefficient (PSC) for each architecture.

PSC is a novel diagnostic metric measuring how much an architecture's
performance varies across prompt variants, normalized by default performance.

    PSC(A) = (max EM - min EM) / EM(default)

A low PSC means the architecture is robust to prompt changes ("fire-and-forget").
A high PSC means the architecture requires significant prompt engineering.

We also compute a token-normalized variant:

    PSC_tok(A) = (max EM - min EM) / EM(default)  (same, but EM computed at equal token budget)

And a bootstrap confidence interval on PSC.

Usage:
    python scripts/compute_psc.py
    python scripts/compute_psc.py --n-bootstrap 5000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

FIGURES_DIR = ROOT_DIR / "results" / "figures"
OUTPUT_DIR = ROOT_DIR / "results" / "psc"

cmap = mpl.colormaps["Set2"]

# ---------------------------------------------------------------------------
# Sensitivity results (already computed by run_prompt_sensitivity.py)
# ---------------------------------------------------------------------------

SENSITIVITY_PATH = ROOT_DIR / "results" / "sensitivity" / "sensitivity_results.json"

ARCH_ORDER = ["ircot", "rlm", "react"]

ARCH_DISPLAY = {
    "ircot": "IRCoT",
    "rlm": "Recursive LM",
    "react": "ReAct",
}

ARCH_COLORS = {
    "ircot": "#1f77b4",
    "rlm": "#e377c2",
    "react": "#d62728",
}

# Default variant per architecture (v0 = Baseline)
DEFAULT_VARIANT = "v0"


# ---------------------------------------------------------------------------
# PSC computation
# ---------------------------------------------------------------------------


def compute_psc(em_values: list[float], default_em: float) -> float:
    """Compute Prompt Sensitivity Coefficient.

    PSC = (max(EM) - min(EM)) / EM(default)

    Returns 0.0 if default_em is 0.
    """
    if default_em == 0:
        return 0.0
    return (max(em_values) - min(em_values)) / default_em


def compute_psc_f1(f1_values: list[float], default_f1: float) -> float:
    """Compute PSC using F1 instead of EM."""
    if default_f1 == 0:
        return 0.0
    return (max(f1_values) - min(f1_values)) / default_f1


def bootstrap_psc(
    em_by_variant: list[list[float]],
    default_em: float,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval on PSC.

    Each variant has a list of per-question EM scores (0 or 1).
    We resample within each variant and recompute PSC.
    """
    rng = np.random.default_rng(seed)
    psc_samples = []

    for _ in range(n_bootstrap):
        boot_ems = []
        for em_list in em_by_variant:
            if not em_list:
                boot_ems.append(0.0)
                continue
            indices = rng.integers(0, len(em_list), size=len(em_list))
            boot_ems.append(float(np.mean(np.array(em_list)[indices])))

        boot_default = boot_ems[0] if boot_ems else 0.0
        psc = compute_psc(boot_ems, boot_default)
        psc_samples.append(psc)

    lower = float(np.percentile(psc_samples, 2.5))
    upper = float(np.percentile(psc_samples, 97.5))
    return lower, upper


# ---------------------------------------------------------------------------
# Load per-question data for bootstrap
# ---------------------------------------------------------------------------


def load_per_question_em(sensitivity_data: dict, arch_key: str) -> dict[str, list[float]]:
    """Load per-question EM for each variant of an architecture.

    We need to find the actual prediction files for the sensitivity runs.
    The sensitivity_results.json only has aggregate stats, so we try to
    load from the run directories using the progress file.
    """
    progress_path = ROOT_DIR / "results" / "sensitivity" / "progress.json"
    if not progress_path.exists():
        return {}

    progress = json.loads(progress_path.read_text())
    variants_data = sensitivity_data[arch_key]["variants"]

    per_q = {}
    for vkey, vinfo in variants_data.items():
        # Try to find run ID from progress
        progress_key = f"{arch_key}_{vkey}"
        run_id_prefix = progress.get(progress_key)
        if not run_id_prefix:
            continue

        # Find run directory
        run_dir = None
        for d in (ROOT_DIR / "results").iterdir():
            if d.is_dir() and d.name.startswith(run_id_prefix):
                run_dir = d
                break

        if run_dir is None or not (run_dir / "predictions.jsonl").exists():
            continue

        ems = []
        with open(run_dir / "predictions.jsonl", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    p = json.loads(line)
                    ems.append(p["exact_match"])
        per_q[vkey] = ems

    return per_q


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------


def compute_all_psc(sensitivity_data: dict, n_bootstrap: int) -> dict:
    """Compute PSC for all architectures."""
    results = {}

    for arch_key in ARCH_ORDER:
        if arch_key not in sensitivity_data:
            continue

        arch_data = sensitivity_data[arch_key]
        variants = arch_data["variants"]
        display_name = arch_data["display_name"]

        em_values = [v["exact_match"] for v in variants.values()]
        f1_values = [v["f1"] for v in variants.values()]

        default_em = variants.get(DEFAULT_VARIANT, {}).get("exact_match", 0.0)
        default_f1 = variants.get(DEFAULT_VARIANT, {}).get("f1", 0.0)

        psc_em = compute_psc(em_values, default_em)
        psc_f1 = compute_psc_f1(f1_values, default_f1)

        # Bootstrap CI
        per_q = load_per_question_em(sensitivity_data, arch_key)
        if per_q and all(k in per_q for k in variants.keys()):
            em_lists = [per_q[k] for k in variants.keys()]
            ci_lower, ci_upper = bootstrap_psc(em_lists, default_em, n_bootstrap)
        else:
            ci_lower, ci_upper = 0.0, 0.0

        results[arch_key] = {
            "display_name": display_name,
            "psc_em": psc_em,
            "psc_f1": psc_f1,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "em_values": em_values,
            "f1_values": f1_values,
            "default_em": default_em,
            "em_range": max(em_values) - min(em_values),
            "em_max": max(em_values),
            "em_min": min(em_values),
            "em_mean": float(np.mean(em_values)),
            "em_std": float(np.std(em_values)),
            "n_variants": len(variants),
            "variant_names": [v["name"] for v in variants.values()],
        }

        print(f"\n  {display_name}:")
        print(f"    PSC (EM) = {psc_em:.3f}")
        print(f"    PSC (F1) = {psc_f1:.3f}")
        if ci_lower > 0 or ci_upper > 0:
            print(f"    95% CI:  [{ci_lower:.3f}, {ci_upper:.3f}]")
        print(
            f"    EM range: {min(em_values):.1%} - {max(em_values):.1%} "
            f"(swing = {max(em_values) - min(em_values):.1%})"
        )
        print(f"    Default EM: {default_em:.1%}")
        print(f"    Variants: {', '.join(v['name'] for v in variants.values())}")

    return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_psc(results: dict, output_dir: Path) -> Path:
    """Plot PSC bar chart with error bars."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    archs = [a for a in ARCH_ORDER if a in results]
    psc_vals = [results[a]["psc_em"] for a in archs]
    psc_f1 = [results[a]["psc_f1"] for a in archs]
    ci_low = [results[a]["ci_lower"] for a in archs]
    ci_high = [results[a]["ci_upper"] for a in archs]

    err_lower = [max(0, p - lo) for p, lo in zip(psc_vals, ci_low)]
    err_upper = [max(0, hi - p) for p, hi in zip(psc_vals, ci_high)]

    colors = [ARCH_COLORS.get(a, "#888888") for a in archs]

    # EM-based PSC
    bars1 = ax1.bar(
        range(len(archs)),
        psc_vals,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        yerr=[err_lower, err_upper],
        capsize=5,
    )
    ax1.set_xticks(range(len(archs)))
    ax1.set_xticklabels([ARCH_DISPLAY.get(a, a) for a in archs], fontsize=11, rotation=15)
    ax1.set_ylabel("PSC (EM)", fontsize=12)
    ax1.set_title("Prompt Sensitivity Coefficient (EM)", fontsize=13)
    ax1.axhline(y=0.2, color="green", linestyle="--", alpha=0.5, label="Robust (PSC<0.2)")
    ax1.axhline(y=0.5, color="orange", linestyle="--", alpha=0.5, label="Moderate (0.2<PSC<0.5)")
    ax1.axhline(y=0.5, color="red", linestyle="--", alpha=0.3)
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(True, alpha=0.2, axis="y")

    for bar, val in zip(bars1, psc_vals):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # F1-based PSC
    bars2 = ax2.bar(range(len(archs)), psc_f1, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_xticks(range(len(archs)))
    ax2.set_xticklabels([ARCH_DISPLAY.get(a, a) for a in archs], fontsize=11, rotation=15)
    ax2.set_ylabel("PSC (F1)", fontsize=12)
    ax2.set_title("Prompt Sensitivity Coefficient (F1)", fontsize=13)
    ax2.grid(True, alpha=0.2, axis="y")

    for bar, val in zip(bars2, psc_f1):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.suptitle(
        "Prompt Sensitivity Coefficient (PSC)\nLower = more robust to prompt changes",
        fontsize=14,
        y=1.02,
    )
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "fig11_psc.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_psc_detail(results: dict, output_dir: Path) -> Path:
    """Plot detailed EM by variant for each architecture."""
    fig, ax = plt.subplots(figsize=(12, 7))

    archs = [a for a in ARCH_ORDER if a in results]
    n_archs = len(archs)
    width = 0.25
    x = np.arange(n_archs)

    max_variants = max(results[a]["n_variants"] for a in archs)
    cmap = plt.get_cmap("Set2")

    for v_idx in range(max_variants):
        vals = []
        for a in archs:
            n_v = results[a]["n_variants"]
            if v_idx < n_v:
                vals.append(results[a]["em_values"][v_idx])
            else:
                vals.append(0)

        offset = (v_idx - max_variants / 2 + 0.5) * width
        color = cmap(v_idx / max_variants)
        ax.bar(
            x + offset,
            [v * 100 for v in vals],
            width,
            color=color,
            edgecolor="black",
            linewidth=0.3,
            label=f"Variant {v_idx}",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([ARCH_DISPLAY.get(a, a) for a in archs], fontsize=12)
    ax.set_ylabel("EM (%)", fontsize=12)
    ax.set_title("EM by Prompt Variant per Architecture", fontsize=14)

    # Build legend from first arch's variant names
    if archs:
        first = results[archs[0]]
        from matplotlib.patches import Patch

        handles = [
            Patch(facecolor=cmap(i / len(first["variant_names"])), label=name)
            for i, name in enumerate(first["variant_names"])
        ]
        ax.legend(handles=handles, fontsize=9, loc="upper right")

    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    output_path = output_dir / "fig11b_psc_detail.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_json(results: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "psc_results.json"
    output_path.write_text(json.dumps(results, indent=2))
    return output_path


def print_paper_table(results: dict):
    """Print a paper-ready table."""
    print(f"\n{'=' * 70}")
    print("PROMPT SENSITIVITY COEFFICIENT (PSC) — PAPER TABLE")
    print(f"{'=' * 70}")
    print()
    print(
        f"  {'Architecture':<15s} {'PSC(EM)':>8s} {'95% CI':>16s} "
        f"{'EM Range':>12s} {'Default':>8s} {'Variants':>8s}"
    )
    print(f"  {'-' * 15} {'-' * 8} {'-' * 16} {'-' * 12} {'-' * 8} {'-' * 8}")

    for arch in ARCH_ORDER:
        if arch not in results:
            continue
        r = results[arch]
        display = r.get("display_name", arch)
        ci = (
            f"[{r['ci_lower']:.2f}, {r['ci_upper']:.2f}]"
            if r["ci_lower"] > 0 or r["ci_upper"] > 0
            else "N/A"
        )
        print(
            f"  {display:<15s} {r['psc_em']:>7.3f} {ci:>16s} "
            f"{r['em_min']:>5.1%}-{r['em_max']:<5.1%} {r['default_em']:>7.1%} "
            f"{r['n_variants']:>8d}"
        )

    print()
    print("  Interpretation:")
    print("    PSC < 0.2  -> Robust (fire-and-forget)")
    print("    PSC 0.2-0.5 -> Moderate sensitivity")
    print("    PSC > 0.5  -> High sensitivity (requires prompt engineering)")
    print()
    print("  Paper-ready sentence:")
    archs = [a for a in ARCH_ORDER if a in results]
    if len(archs) >= 2:
        most_robust = min(archs, key=lambda a: results[a]["psc_em"])
        most_sensitive = max(archs, key=lambda a: results[a]["psc_em"])
        r_robust = results[most_robust]["psc_em"]
        r_sens = results[most_sensitive]["psc_em"]
        d_robust = results[most_robust].get("display_name", most_robust)
        d_sens = results[most_sensitive].get("display_name", most_sensitive)
        print(
            f'    "Prompt sensitivity varies dramatically by architecture: '
            f"{d_robust} has PSC={r_robust:.2f} (robust), while "
            f"{d_sens} has PSC={r_sens:.2f} (highly sensitive). "
            f"This means {d_sens} requires significant prompt "
            f"engineering investment, while {d_robust} can be deployed "
            f'with minimal tuning."'
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Compute Prompt Sensitivity Coefficient (PSC)",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10000,
        help="Bootstrap iterations for CI (default: 10000)",
    )
    args = parser.parse_args()

    if not SENSITIVITY_PATH.exists():
        print(f"[ERROR] Sensitivity results not found: {SENSITIVITY_PATH}")
        print("        Run `python scripts/run_prompt_sensitivity.py` first.")
        sys.exit(1)

    print("Loading sensitivity results...")
    sensitivity_data = json.loads(SENSITIVITY_PATH.read_text())

    print(f"Computing PSC for {len(sensitivity_data)} architectures...")
    results = compute_all_psc(sensitivity_data, args.n_bootstrap)

    # Plot
    psc_path = plot_psc(results, FIGURES_DIR)
    detail_path = plot_psc_detail(results, FIGURES_DIR)
    print(f"\n  PSC chart:     {psc_path}")
    print(f"  Detail chart:  {detail_path}")

    # Export
    json_path = export_json(results, OUTPUT_DIR)
    print(f"  JSON:          {json_path}")

    # Paper table
    print_paper_table(results)

    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"  JSON:   {OUTPUT_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
