#!/usr/bin/env python3
"""Generate validation plots for the three theoretical results.

Theorem 1: Agentic token complexity T(K) = O(K^2 * t_bar)
  - Plot: predicted O(K^2) curve vs actual token counts from ReAct iter ablation

Theorem 2: RLM termination (DAG vs cycle)
  - Plot: decomposition graph illustration + loop rate measurement

Theorem 3: Retrieval sufficiency P_succ(n,k) = (1-(1-p)^n)^k
  - Plot: predicted curve (fitted p) vs actual EM for top_k=3/5/10/20

Usage:
    python scripts/plot_theory_validation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

FIGURES_DIR = ROOT_DIR / "results" / "figures"
THEORY_DIR = ROOT_DIR / "results" / "theory"
THEORY_DIR.mkdir(parents=True, exist_ok=True)

# Robustness data (from our ablation runs)
REACT_ITER_DATA = {
    3: (0.32, 4405),
    7: (0.48, 8172),
    10: (0.48, 10790),
}

TOPK_DATA = {
    3: (0.49, 100),
    5: (0.52, 100),
    10: (0.64, 100),
    20: (0.64, 100),
}

RLM_LOOP_RATE = 0.034


# ---------------------------------------------------------------------------
# Theorem 1: Quadratic token complexity validation
# ---------------------------------------------------------------------------


def quadratic_model(K, a, b):
    return a * K**2 + b * K


def linear_model(K, a, b):
    return a * K + b


def plot_theorem1_validation(output_dir):
    Ks = np.array(list(REACT_ITER_DATA.keys()), dtype=float)
    tokens = np.array([REACT_ITER_DATA[k][1] for k in REACT_ITER_DATA.keys()], dtype=float)

    popt_quad, _ = curve_fit(quadratic_model, Ks, tokens, p0=[100, 100])
    popt_lin, _ = curve_fit(linear_model, Ks, tokens, p0=[500, 0])

    K_smooth = np.linspace(1, 12, 100)
    quad_fit = quadratic_model(K_smooth, *popt_quad)
    lin_fit = linear_model(K_smooth, *popt_lin)

    quad_pred = quadratic_model(Ks, *popt_quad)
    lin_pred = linear_model(Ks, *popt_lin)
    ss_res_quad = np.sum((tokens - quad_pred) ** 2)
    ss_res_lin = np.sum((tokens - lin_pred) ** 2)
    ss_tot = np.sum((tokens - np.mean(tokens)) ** 2)
    r2_quad = 1 - ss_res_quad / ss_tot if ss_tot > 0 else 0
    r2_lin = 1 - ss_res_lin / ss_tot if ss_tot > 0 else 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.scatter(
        Ks,
        tokens,
        c="#d62728",
        s=150,
        zorder=5,
        edgecolors="black",
        linewidth=0.5,
        label="Measured (ReAct RAG)",
    )
    ax1.plot(
        K_smooth,
        quad_fit,
        "r-",
        linewidth=2,
        alpha=0.8,
        label="Quadratic: T=%.0f*K^2 + %.0f*K (R2=%.4f)" % (popt_quad[0], popt_quad[1], r2_quad),
    )
    ax1.plot(
        K_smooth,
        lin_fit,
        "b--",
        linewidth=2,
        alpha=0.5,
        label="Linear: T=%.0f*K + %.0f (R2=%.4f)" % (popt_lin[0], popt_lin[1], r2_lin),
    )

    for k, t in REACT_ITER_DATA.items():
        ax1.annotate(
            "K=%d: %d tokens" % (k, t[1]),
            (k, t[1]),
            textcoords="offset points",
            xytext=(10, 10),
            fontsize=9,
        )

    ax1.set_xlabel("Max Iterations (K)", fontsize=12)
    ax1.set_ylabel("Avg Tokens per Question", fontsize=12)
    ax1.set_title("Theorem 1: Agentic Token Complexity\nT(K) = O(K^2 * t_bar)", fontsize=13)
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 13)

    ems = [REACT_ITER_DATA[k][0] for k in REACT_ITER_DATA.keys()]
    for i in range(1, len(Ks)):
        delta_tokens = tokens[i] - tokens[i - 1]
        delta_em = ems[i] - ems[i - 1]
        marginal = delta_tokens / delta_em if delta_em > 0 else float("inf")
        mid_k = (Ks[i] + Ks[i - 1]) / 2
        mid_t = (tokens[i] + tokens[i - 1]) / 2
        label = "d_tokens/d_EM = inf" if delta_em == 0 else "d_tokens/d_EM = %.0f" % marginal
        ax2.annotate(
            label,
            (mid_k, mid_t),
            textcoords="offset points",
            xytext=(10, 0),
            fontsize=9,
            color="red" if delta_em == 0 else "black",
        )
        ax2.scatter([mid_k], [mid_t], c="orange", s=80, zorder=5)

    ax2.set_xlabel("Iteration Range Midpoint", fontsize=12)
    ax2.set_ylabel("Tokens per Question", fontsize=12)
    ax2.set_title("Marginal Cost: Past K*=7, Zero Accuracy Gain\n(Corollary 1.2)", fontsize=13)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "fig_theory1_agentic_complexity.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print("  Theorem 1 validation: %s" % output_path)
    print("    Quadratic R2 = %.4f" % r2_quad)
    print("    Linear R2    = %.4f" % r2_lin)
    print("    Fit: T = %.0f*K^2 + %.0f*K" % (popt_quad[0], popt_quad[1]))

    result = {
        "quadratic_r2": float(r2_quad),
        "linear_r2": float(r2_lin),
        "fit_a": float(popt_quad[0]),
        "fit_b": float(popt_quad[1]),
        "data": {str(k): {"em": v[0], "tokens": v[1]} for k, v in REACT_ITER_DATA.items()},
    }
    return output_path, result


# ---------------------------------------------------------------------------
# Theorem 2: RLM termination illustration
# ---------------------------------------------------------------------------


def plot_theorem2_validation(output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.set_xlim(-0.5, 3.5)
    ax1.set_ylim(-0.5, 2.5)
    ax1.set_title("Terminating Decomposition (DAG)\nRLM terminates: graph is acyclic", fontsize=12)

    nodes_dag = {
        "Q": (1.5, 2.0),
        "Q1": (0.5, 1.0),
        "Q2": (2.5, 1.0),
        "A1": (0.5, 0.0),
        "A2": (2.5, 0.0),
    }
    for name, (x, y) in nodes_dag.items():
        color = "#2ca02c" if name.startswith("A") else "#1f77b4"
        ax1.scatter(x, y, c=color, s=800, zorder=5, edgecolors="black", linewidth=1)
        ax1.text(
            x, y, name, ha="center", va="center", fontsize=14, fontweight="bold", color="white"
        )

    for src, dst in [("Q", "Q1"), ("Q", "Q2"), ("Q1", "A1"), ("Q2", "A2")]:
        sx, sy = nodes_dag[src]
        dx, dy = nodes_dag[dst]
        ax1.annotate(
            "",
            xy=(dx, dy + 0.15),
            xytext=(sx, sy - 0.15),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
        )

    ax1.text(
        1.5,
        -0.4,
        "Theorem 2: RLM terminates iff G_d(q) is a DAG",
        ha="center",
        fontsize=10,
        style="italic",
        color="green",
    )
    ax1.axis("off")

    ax2.set_xlim(-0.5, 3.5)
    ax2.set_ylim(-0.5, 2.5)
    ax2.set_title(
        "Non-terminating (Cycle)\nRLM loops: %.1f%% of runs" % (RLM_LOOP_RATE * 100), fontsize=12
    )

    nodes_cycle = {"Q": (1.5, 2.0), "Q1": (0.5, 1.0), "Q2": (2.5, 1.0), "Q1'": (0.5, 0.0)}
    for name, (x, y) in nodes_cycle.items():
        color = "#d62728" if name == "Q1'" else "#1f77b4"
        ax2.scatter(x, y, c=color, s=800, zorder=5, edgecolors="black", linewidth=1)
        ax2.text(
            x, y, name, ha="center", va="center", fontsize=14, fontweight="bold", color="white"
        )

    for src, dst in [("Q", "Q1"), ("Q", "Q2"), ("Q1", "Q1'")]:
        sx, sy = nodes_cycle[src]
        dx, dy = nodes_cycle[dst]
        ax2.annotate(
            "",
            xy=(dx, dy + 0.15),
            xytext=(sx, sy - 0.15),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
        )

    ax2.annotate(
        "",
        xy=(0.3, 1.1),
        xytext=(0.3, -0.1),
        arrowprops=dict(arrowstyle="->", color="red", lw=2, connectionstyle="arc3,rad=-0.5"),
    )
    ax2.text(
        -0.2, 0.5, "CYCLE", fontsize=11, color="red", fontweight="bold", rotation=90, va="center"
    )
    ax2.text(
        1.5,
        -0.4,
        "Corollary 2.2: With depth limit D, RLM terminates in O(b^D) calls",
        ha="center",
        fontsize=10,
        style="italic",
        color="green",
    )
    ax2.axis("off")

    plt.suptitle("Theorem 2: RLM Termination Analysis", fontsize=14, y=1.02)
    plt.tight_layout()
    output_path = output_dir / "fig_theory2_rlm_termination.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print("  Theorem 2 validation: %s" % output_path)
    print("    RLM loop rate: %.1f%% (cycle probability)" % (RLM_LOOP_RATE * 100))

    result = {"loop_rate": RLM_LOOP_RATE}
    return output_path, result


# ---------------------------------------------------------------------------
# Theorem 3: Retrieval sufficiency bound validation
# ---------------------------------------------------------------------------


def retrieval_model(n, p, k=2):
    return (1 - (1 - p) ** n) ** k


def plot_theorem3_validation(output_dir):
    ns = np.array(list(TOPK_DATA.keys()), dtype=float)
    ems = np.array([TOPK_DATA[n][0] for n in TOPK_DATA.keys()], dtype=float)

    # Fit p using first data point (top_k=3)
    # P_succ(3, 2) = (1-(1-p)^3)^2 = ems[0]
    # Solve: 1-(1-p)^3 = sqrt(ems[0])
    # (1-p)^3 = 1 - sqrt(ems[0])
    # p = 1 - (1 - sqrt(ems[0]))^(1/3)
    p_fitted = 1 - (1 - np.sqrt(ems[0])) ** (1.0 / 3.0)

    # Predict for all n
    n_smooth = np.linspace(1, 25, 100)
    predicted_2hop = retrieval_model(n_smooth, p_fitted, k=2)
    predicted_3hop = retrieval_model(n_smooth, p_fitted, k=3)
    predicted_4hop = retrieval_model(n_smooth, p_fitted, k=4)

    # R^2 for 2-hop fit
    predicted_actual = retrieval_model(ns, p_fitted, k=2)
    ss_res = np.sum((ems - predicted_actual) ** 2)
    ss_tot = np.sum((ems - np.mean(ems)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Fitted curve vs actual
    ax1.scatter(
        ns,
        ems * 100,
        c="#2ca02c",
        s=150,
        zorder=5,
        edgecolors="black",
        linewidth=0.5,
        label="Measured (Vanilla RAG, 2-hop)",
    )
    ax1.plot(
        n_smooth,
        predicted_2hop * 100,
        "g-",
        linewidth=2,
        label="Theorem 3 (2-hop, p=%.3f, R2=%.4f)" % (p_fitted, r2),
    )
    ax1.plot(
        n_smooth, predicted_3hop * 100, "b--", linewidth=1.5, alpha=0.7, label="Predicted 3-hop"
    )
    ax1.plot(
        n_smooth, predicted_4hop * 100, "r:", linewidth=1.5, alpha=0.7, label="Predicted 4-hop"
    )

    for n_val, em_val in TOPK_DATA.items():
        ax1.annotate(
            "k=%d: %.0f%%" % (n_val, em_val[0] * 100),
            (n_val, em_val[0] * 100),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9,
        )

    ax1.set_xlabel("Retrieval Depth (top-k)", fontsize=12)
    ax1.set_ylabel("EM (%)", fontsize=12)
    ax1.set_title(
        "Theorem 3: Retrieval Sufficiency Bound\nP_succ(n,k) = (1-(1-p)^n)^k", fontsize=13
    )
    ax1.legend(fontsize=9, loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 25)

    # Plot 2: Accuracy gain delta (predicted vs actual)
    actual_deltas = np.diff(ems * 100)
    pred_deltas = np.diff(retrieval_model(ns, p_fitted, k=2) * 100)
    bar_width = 0.35
    x_pos = np.arange(len(actual_deltas))
    ax2.bar(
        x_pos - bar_width / 2,
        actual_deltas,
        bar_width,
        color="#2ca02c",
        alpha=0.7,
        label="Measured EM gain",
    )
    ax2.bar(
        x_pos + bar_width / 2,
        pred_deltas,
        bar_width,
        color="#1f77b4",
        alpha=0.7,
        label="Predicted by Theorem 3",
    )
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(["%d->%d" % (ns[i], ns[i + 1]) for i in range(len(actual_deltas))])
    ax2.set_xlabel("top-k Transition", fontsize=12)
    ax2.set_ylabel("EM Gain (pp)", fontsize=12)
    ax2.set_title("Predicted vs Actual Accuracy Gains", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    output_path = output_dir / "fig_theory3_retrieval_sufficiency.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print("  Theorem 3 validation: %s" % output_path)
    print("    Fitted p = %.4f" % p_fitted)
    print("    R2 = %.4f" % r2)
    print("    Predictions:")
    for n_val in TOPK_DATA:
        actual = TOPK_DATA[n_val][0]
        predicted = retrieval_model(n_val, p_fitted, k=2)
        print(
            "      top_k=%d: actual=%.1f%%, predicted=%.1f%%"
            % (n_val, actual * 100, predicted * 100)
        )

    result = {
        "fitted_p": float(p_fitted),
        "r2": float(r2),
        "data": {
            str(n): {"actual_em": v[0], "predicted_em": float(retrieval_model(n, p_fitted, k=2))}
            for n, v in TOPK_DATA.items()
        },
    }
    return output_path, result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("THEORY VALIDATION PLOTS")
    print("=" * 60)

    all_results = {}

    print("\n[1/3] Theorem 1: Agentic token complexity...")
    path1, res1 = plot_theorem1_validation(FIGURES_DIR)
    all_results["theorem1"] = res1

    print("\n[2/3] Theorem 2: RLM termination...")
    path2, res2 = plot_theorem2_validation(FIGURES_DIR)
    all_results["theorem2"] = res2

    print("\n[3/3] Theorem 3: Retrieval sufficiency...")
    path3, res3 = plot_theorem3_validation(FIGURES_DIR)
    all_results["theorem3"] = res3

    # Save all results
    results_path = THEORY_DIR / "theory_validation_results.json"
    results_path.write_text(json.dumps(all_results, indent=2))
    print("\n  All results: %s" % results_path)

    print("\n" + "=" * 60)
    print("DONE")
    print("  Figures: %s" % FIGURES_DIR)
    print("  Results: %s" % THEORY_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
