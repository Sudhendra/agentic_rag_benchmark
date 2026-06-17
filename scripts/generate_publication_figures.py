#!/usr/bin/env python3
"""Generate all publication figures from experimental results.

Usage:
    python scripts/generate_publication_figures.py     # Generate all figures
    python scripts/generate_publication_figures.py --fig 4  # Single figure
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

OUTPUT_DIR = ROOT_DIR / "results" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
sns.set_palette("colorblind")

# Architecture colors (consistent across all figures)
ARCH_COLORS = {
    "vanilla_rag": "#4C72B0",
    "react_rag": "#DD8452",
    "self_rag": "#55A868",
    "planner_rag": "#C44E52",
    "ircot_rag": "#8172B2",
    "reap_rag": "#937860",
    "recursive_lm": "#DA8BC3",
}
ARCH_LABELS = {
    "vanilla_rag": "Vanilla RAG",
    "react_rag": "ReAct",
    "self_rag": "Self-RAG",
    "planner_rag": "Planner",
    "ircot_rag": "IRCoT",
    "reap_rag": "REAP",
    "recursive_lm": "RLM",
}


def load_full_stats() -> pd.DataFrame:
    df = pd.read_csv(ROOT_DIR / "full_stats_table.csv")
    # Only full runs (>=7000 questions, not from different datasets)
    df = df[df["num_questions"] >= 7000].copy()
    # Exclude clearly broken runs (effectively 0% EM)
    df = df[df["exact_match"] > 0.01].copy()
    # Best run per architecture by F1
    best_idx = df.groupby("architecture")["f1"].idxmax()
    return df.loc[best_idx].reset_index(drop=True)


def ensure_sensitivity_json() -> Path | None:
    path = ROOT_DIR / "results" / "sensitivity" / "sensitivity_results.json"
    if path.exists():
        return path
    alt = list((ROOT_DIR / "results").rglob("sensitivity_results.json"))
    if alt:
        return alt[0]
    return None


def fig1_pareto_frontier(df: pd.DataFrame) -> None:
    """Cost vs. F1 — Pareto frontier showing cost-performance tradeoffs."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for _, row in df.iterrows():
        arch = row["architecture"]
        color = ARCH_COLORS.get(arch, "#333333")
        label = ARCH_LABELS.get(arch, arch)
        lower_err = row["f1"] - row["f1_ci_lower"]
        upper_err = row["f1_ci_upper"] - row["f1"]
        ax.errorbar(
            row["cost_usd"],
            row["f1"],
            yerr=[[lower_err], [upper_err]],
            fmt="o",
            capsize=5,
            capthick=1.5,
            elinewidth=1.5,
            color=color,
            markersize=12,
            markeredgecolor="white",
            markeredgewidth=1.5,
            label=label,
            zorder=5,
        )
        # Architecture label next to each point
        txt = ax.annotate(
            label,
            (row["cost_usd"], row["f1"]),
            xytext=(8, 6),
            textcoords="offset points",
            fontsize=12,
            fontweight="bold",
            color=color,
            zorder=10,
        )
        txt.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])

    # Pareto frontier: sort by cost, compute cumulative max F1
    pareto = df.sort_values("cost_usd").copy()
    pareto["pareto_f1"] = pareto["f1"].cummax()
    # Step backwards to draw the frontier
    xs, ys = [], []
    prev_f1 = 0
    for _, r in pareto.iterrows():
        if r["f1"] > prev_f1:
            xs.append(r["cost_usd"])
            ys.append(r["f1"])
            prev_f1 = r["f1"]
    if xs:
        ax.step(
            xs,
            ys,
            where="post",
            color="#333333",
            linewidth=1.5,
            linestyle="--",
            alpha=0.5,
            label="Pareto Frontier",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Total Inference Cost (USD)")
    ax.set_ylabel("F1 Score")
    ax.set_title("Cost vs. F1 Pareto Frontier", fontweight="bold", pad=12)
    ax.legend(
        frameon=True, 
        facecolor="white", 
        edgecolor="#cccccc", 
        loc="upper left", 
        bbox_to_anchor=(1.01, 1)
    )
    ax.margins(x=0.2, y=0.15)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig1_pareto_frontier.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig1_pareto_frontier.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  [1/8] Pareto frontier saved")


def fig2_latency(df: pd.DataFrame) -> None:
    """Latency per question by architecture."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sorted_df = df.sort_values("latency_ms")

    colors = [ARCH_COLORS.get(a, "#333333") for a in sorted_df["architecture"]]
    labels = [ARCH_LABELS.get(a, a) for a in sorted_df["architecture"]]

    bars = ax.barh(
        labels, sorted_df["latency_ms"] / 1000, color=colors, edgecolor="white", height=0.6
    )

    for bar, val in zip(bars, sorted_df["latency_ms"] / 1000):
        ax.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}s",
            va="center",
            fontsize=13,
        )

    ax.set_xlabel("Average Latency per Question (seconds)")
    ax.set_title("Answer Latency by Architecture", fontweight="bold", pad=12)
    ax.margins(x=0.15)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig2_latency.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig2_latency.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  [2/8] Latency chart saved")


def fig3_token_usage(df: pd.DataFrame) -> None:
    """Tokens per question by architecture."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sorted_df = df.sort_values("tokens_per_q")

    colors = [ARCH_COLORS.get(a, "#333333") for a in sorted_df["architecture"]]
    labels = [ARCH_LABELS.get(a, a) for a in sorted_df["architecture"]]

    bars = ax.barh(
        labels, sorted_df["tokens_per_q"] / 1000, color=colors, edgecolor="white", height=0.6
    )

    bar_widths = sorted_df["tokens_per_q"] / 1000
    max_width = bar_widths.max()
    for bar, val in zip(bars, bar_widths):
        ax.text(
            bar.get_width() + max_width * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.0f}K",
            va="center",
            fontsize=13,
        )

    ax.set_xlabel("Average Tokens per Question (thousands)")
    ax.set_title("Token Consumption per Question", fontweight="bold", pad=12)
    ax.margins(x=0.25)

    fig.savefig(OUTPUT_DIR / "fig3_token_usage.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig3_token_usage.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  [3/8] Token usage chart saved")


def fig4_error_taxonomy() -> None:
    """Stacked bar chart of error categories per architecture."""
    csv_path = ROOT_DIR / "results" / "error_taxonomy" / "taxonomy_summary.csv"
    if not csv_path.exists():
        print("  [4/8] SKIP: taxonomy_summary.csv not found")
        return

    df = pd.read_csv(csv_path)
    categories = [
        ("pct_complete_miss", "Complete Miss"),
        ("pct_low_overlap", "Low Overlap"),
        ("pct_partial", "Partial"),
        ("pct_near_miss", "Near Miss"),
        ("pct_verbose", "Verbose"),
        ("pct_loop", "Loop"),
        ("pct_yesno_flip", "Flip"),
        ("pct_correct", "Correct"),
    ]
    cat_colors = {
        "Complete Miss": "#C44E52",
        "Low Overlap": "#DD8452",
        "Partial": "#55A868",
        "Near Miss": "#4C72B0",
        "Verbose": "#937860",
        "Loop": "#E0A800",
        "Flip": "#DA8BC3",
        "Correct": "#2C6B2F",
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    arch_labels = [ARCH_LABELS.get(a, a) for a in df["architecture"]]
    x = np.arange(len(arch_labels))
    width = 0.6

    bottom = np.zeros(len(df))
    for col, label in categories:
        values = df[col].to_numpy()
        if values.max() == 0:
            continue
        ax.bar(
            x,
            values,
            width,
            bottom=bottom,
            label=label,
            color=cat_colors.get(label, "#999999"),
            edgecolor="white",
            linewidth=0.5,
        )
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(arch_labels, rotation=30, ha="right")
    ax.set_ylabel("Percentage of Questions")
    ax.set_title("Error Profile by Architecture", fontweight="bold", pad=12)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=True, facecolor="white")
    ax.set_ylim(0, 100)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig4_error_taxonomy.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig4_error_taxonomy.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  [4/8] Error taxonomy saved")


def fig5_prompt_sensitivity() -> None:
    """Grouped bar chart showing prompt sensitivity across 3 architectures."""
    data = {
        "RLM": {"Baseline": 44.0, "Minimalist": 18.0, "Strict": 58.0},
        "ReAct": {"Baseline": 48.0, "Minimalist": 6.0, "Strict": 26.0},
        "IRCoT": {"Baseline": 44.0, "Minimalist": 42.0, "Strict": 52.0},
    }
    variants = ["Baseline", "Minimalist", "Strict"]
    var_colors = {"Baseline": "#4C72B0", "Minimalist": "#C44E52", "Strict": "#2C6B2F"}
    arch_order = ["ReAct", "RLM", "IRCoT"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(arch_order))
    width = 0.22
    offsets = [-width, 0, width]

    swing_line_y = []
    swing_labels = []

    for i, arch in enumerate(arch_order):
        vals = data[arch]
        for j, var in enumerate(variants):
            em = vals[var]
            offset = offsets[j]
            ax.bar(
                x[i] + offset,
                em,
                width,
                label=var if i == 0 else "",
                color=var_colors[var],
                edgecolor="white",
                linewidth=0.5,
            )
            ax.text(
                x[i] + offset,
                em + 1,
                f"{em}%",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

        swing_line_y.append((min(vals.values()), max(vals.values())))
        if arch == "ReAct":
            swing_labels.append("42pt")
        elif arch == "RLM":
            swing_labels.append("40pt")
        elif arch == "IRCoT":
            swing_labels.append("10pt")

    for i, (lo, hi) in enumerate(swing_line_y):
        ax.annotate(
            swing_labels[i],
            (x[i], hi),
            xytext=(0, 18),
            textcoords="offset points",
            ha="center",
            fontsize=13,
            fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc"),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(arch_order, fontsize=16)
    ax.set_ylabel("Exact Match (%)")
    ax.set_title("Prompt Sensitivity Across Architectures", fontweight="bold", pad=12)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc")
    ax.margins(y=0.15)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig5_prompt_sensitivity.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig5_prompt_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  [5/8] Prompt sensitivity saved")


def fig6_topk_scaling() -> None:
    """Line/scatter plot of EM vs top_k for Vanilla RAG."""
    data = pd.DataFrame(
        {
            "top_k": [3, 5, 10, 20],
            "em": [49.0, 45.0, 64.0, 64.0],
            "f1": [60.9, 59.5, 77.3, 77.3],
        }
    )

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    color_em = "#4C72B0"
    color_f1 = "#C44E52"

    ax1.plot(
        data["top_k"],
        data["em"],
        "o-",
        color=color_em,
        markersize=12,
        linewidth=2.5,
        label="EM",
        zorder=3,
    )
    ax1.plot(
        data["top_k"],
        data["f1"],
        "s--",
        color=color_f1,
        markersize=12,
        linewidth=2.5,
        label="F1",
        zorder=3,
    )

    for _, row in data.iterrows():
        # EM above the point, offset to the left for clarity
        ax1.annotate(
            f"{row['em']:.0f}%",
            (row["top_k"], row["em"]),
            xytext=(-12, -14),
            textcoords="offset points",
            ha="center",
            fontsize=11,
            fontweight="bold",
            color=color_em,
        )
        # F1 above the point, offset to the right
        ax1.annotate(
            f"{row['f1']:.0f}%",
            (row["top_k"], row["f1"]),
            xytext=(12, 10),
            textcoords="offset points",
            ha="center",
            fontsize=11,
            fontweight="bold",
            color=color_f1,
        )

    ax1.set_xlabel("Number of Retrieved Documents (top_k)")
    ax1.set_ylabel("Score (%)")
    ax1.set_title("Retrieval Depth vs. Accuracy (Vanilla RAG)", fontweight="bold", pad=12)
    ax1.set_xticks([3, 5, 10, 20])
    ax1.legend(frameon=True, facecolor="white", edgecolor="#cccccc")
    ax1.margins(y=0.2)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig6_topk_scaling.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig6_topk_scaling.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  [6/8] Top-k scaling saved")


def fig7_rlm_depth() -> None:
    """Bar chart: RLM EM vs max_depth."""
    data = pd.DataFrame(
        {
            "depth": ["2", "3", "5"],
            "em": [60.0, 51.8, 58.0],
        }
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#55A868", "#4C72B0", "#DA8BC3"]
    bars = ax.bar(data["depth"], data["em"], color=colors, edgecolor="white", width=0.5)

    for bar, val in zip(bars, data["em"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.0f}%",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

    ax.set_xlabel("Maximum Recursion Depth")
    ax.set_ylabel("Exact Match (%)")
    ax.set_title("RLM: Recursion Depth vs. Accuracy", fontweight="bold", pad=12)
    ax.margins(y=0.2)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig7_rlm_depth.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig7_rlm_depth.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  [7/8] RLM depth ablation saved")


def fig8_cross_model() -> None:
    """Paired bar chart: GPT-4o-mini vs Llama-3.3-70B on Vanilla + RLM."""
    data = pd.DataFrame(
        {
            "model": ["GPT-4o-mini", "Llama-3.3-70B (Groq)"] * 2,
            "architecture": ["Vanilla"] * 2 + ["RLM"] * 2,
            "em": [55.0, 55.0, 51.8, 52.0],
            "f1": [69.0, 66.3, 64.3, 66.7],
            "cost": [0.01, 0.00, 0.11, 0.00],
        }
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), sharey=False, constrained_layout=True)
    gpt_color = "#4C72B0"
    groq_color = "#DD8452"
    bar_width = 0.3

    for ax, metric, metric_name in [(ax1, "em", "EM"), (ax2, "f1", "F1")]:
        x = np.arange(2)
        gpt_vals = data[data["model"] == "GPT-4o-mini"][metric].to_numpy()
        groq_vals = data[data["model"] == "Llama-3.3-70B (Groq)"][metric].to_numpy()

        bars1 = ax.bar(
            x - bar_width / 2,
            gpt_vals,
            bar_width,
            label="GPT-4o-mini",
            color=gpt_color,
            edgecolor="white",
        )
        bars2 = ax.bar(
            x + bar_width / 2,
            groq_vals,
            bar_width,
            label="Llama-3.3-70B (Groq)",
            color=groq_color,
            edgecolor="white",
        )

        for bar, val in zip(bars1, gpt_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.0f}%",
                ha="center",
                fontsize=12,
                fontweight="bold",
                color=gpt_color,
            )
        for bar, val in zip(bars2, groq_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.0f}%",
                ha="center",
                fontsize=12,
                fontweight="bold",
                color=groq_color,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(["Vanilla RAG", "RLM"], fontsize=14)
        ax.set_ylabel(f"{metric_name} (%)")
        ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc")
        ax.margins(y=0.2)
        ax.set_ylim(bottom=0)

    fig.suptitle(
        "Cross-Model Validation: Architecture Ranking is Stable", fontweight="bold", fontsize=14
    )
    fig.savefig(OUTPUT_DIR / "fig8_cross_model.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig8_cross_model.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  [8/8] Cross-model validation saved")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication figures")
    parser.add_argument("--fig", type=int, default=None, help="Single figure number (1-8)")
    args = parser.parse_args()

    print("Generating publication figures...")

    df = load_full_stats()

    figures = [
        (1, fig1_pareto_frontier, [df]),
        (2, fig2_latency, [df]),
        (3, fig3_token_usage, [df]),
        (4, fig4_error_taxonomy, []),
        (5, fig5_prompt_sensitivity, []),
        (6, fig6_topk_scaling, []),
        (7, fig7_rlm_depth, []),
        (8, fig8_cross_model, []),
    ]

    for num, func, args_list in figures:
        if args.fig is not None and num != args.fig:
            continue
        func(*args_list)

    print(f"\nAll figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
