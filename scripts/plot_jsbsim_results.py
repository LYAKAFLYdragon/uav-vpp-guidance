"""Generate JSBSim validation figures for the paper."""
import json
import os
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

OUT_DIR = "outputs/jsbsim_figures"
os.makedirs(OUT_DIR, exist_ok=True)


def load_aggregate(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_heatmap(agg: List[dict], out_path: str):
    """Success-rate heatmap: rows = (method, target), cols = scenario."""
    methods = ["VPP", "No-VPP"]
    targets = ["mild", "weave"]
    scenarios = ["favorable", "neutral", "disadvantage", "challenging"]

    # Build matrix: one row per (method, target)
    matrix = []
    row_labels = []
    for target in targets:
        for method in methods:
            row = []
            for scen in scenarios:
                val = next(
                    (
                        r
                        for r in agg
                        if r["method"] == method
                        and r["target"] == target
                        and r["scenario"] == scen
                    ),
                    None,
                )
                row.append(val["mean_success_rate"] * 100 if val else np.nan)
            matrix.append(row)
            row_labels.append(f"{method}\n({target})")

    matrix = np.array(matrix)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    im = ax.imshow(matrix, vmin=0, vmax=100, cmap="RdYlGn")
    ax.set_xticks(np.arange(len(scenarios)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels([s.capitalize() for s in scenarios])
    ax.set_yticklabels(row_labels)
    ax.set_title("JSBSim F-16 success rate (%) — VPP vs No-VPP")
    fig.colorbar(im, ax=ax)
    for i in range(len(row_labels)):
        for j in range(len(scenarios)):
            text = f"{matrix[i, j]:.0f}"
            if np.isnan(matrix[i, j]):
                text = "—"
            ax.text(j, i, text, ha="center", va="center", color="black", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved heatmap: {out_path}")


def plot_before_after(out_path: str):
    """Bar chart: disadvantage success rate before/after the crash fix."""
    # Pre-fix values from prior report (crash = 0% success)
    before = {"VPP mild": 0.0, "VPP weave": 0.0, "No-VPP mild": 0.0, "No-VPP weave": 0.0}
    after = {"VPP mild": 100.0, "VPP weave": 100.0, "No-VPP mild": 0.0, "No-VPP weave": 100.0}
    labels = list(before.keys())
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    bars1 = ax.bar(x - width / 2, [before[k] for k in labels], width, label="Before fix", color="salmon")
    bars2 = ax.bar(x + width / 2, [after[k] for k in labels], width, label="After fix", color="seagreen")
    ax.set_ylabel("Success rate (%) on Disadvantage")
    ax.set_title("Disadvantage scenario: crash fix before/after")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim([0, 110])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for i, k in enumerate(labels):
        b1 = before[k]
        b2 = after[k]
        label1 = "0\n(crash)" if b1 == 0 else f"{b1:.0f}"
        label2 = "0\n(timeout)" if b2 == 0 else f"{b2:.0f}"
        ax.text(i - width / 2, b1 + 2, label1, ha="center", va="bottom", fontsize=8, color="salmon")
        ax.text(i + width / 2, b2 + 2, label2, ha="center", va="bottom", fontsize=8, color="seagreen")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved before/after: {out_path}")


def plot_training_curves(out_path: str):
    """Training-curve success rate: VPP vs No-VPP on JSBSim weave."""
    paths = {
        "VPP": "outputs/experiments/no_prediction_vpp_ppo_jsbsim_weave/logs/episode_train_log.csv",
        "No-VPP": "outputs/experiments/no_vpp_ppo_jsbsim_weave/logs/episode_train_log.csv",
    }
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {"VPP": "#1f77b4", "No-VPP": "#ff7f0e"}
    for name, path in paths.items():
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        # success is boolean-ish; compute moving average over episodes
        df["success_ma"] = df["success"].rolling(window=50, min_periods=1).mean()
        ax.plot(df["step"], df["success_ma"] * 100, label=name, color=colors[name], linewidth=1.5)
    ax.set_xlabel("Environment step")
    ax.set_ylabel("Success rate moving average (%)")
    ax.set_title("JSBSim weave training: VPP vs No-VPP")
    ax.set_ylim([0, 105])
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved training curves: {out_path}")


def main():
    agg_path = "outputs/jsbsim_matrix_eval/aggregate.json"
    if os.path.exists(agg_path):
        agg = load_aggregate(agg_path)
        plot_heatmap(agg, os.path.join(OUT_DIR, "jsbsim_success_heatmap.png"))
    else:
        print(f"Warning: {agg_path} not found; skipping heatmap.")

    plot_before_after(os.path.join(OUT_DIR, "disadvantage_before_after.png"))
    plot_training_curves(os.path.join(OUT_DIR, "training_curve_vpp_vs_novpp.png"))


if __name__ == "__main__":
    main()
