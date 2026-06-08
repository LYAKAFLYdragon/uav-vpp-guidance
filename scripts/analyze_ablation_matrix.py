#!/usr/bin/env python3
"""
Analyze ablation matrix results and generate summary tables/figures.

Usage:
    python scripts/analyze_ablation_matrix.py \
        --manifest outputs/ablation_matrix/manifest.json \
        --output-dir outputs/ablation_matrix/analysis
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_all_episodes(manifest):
    """Load all per-episode CSVs from manifest."""
    all_data = []
    for entry in manifest:
        if entry["status"] != "success":
            continue

        eval_dir = Path(entry["checkpoint"]).parent.parent / "evaluation"
        csv_path = eval_dir / "raw_episodes.csv"

        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        df["method_key"] = entry["method"]
        df["target_mode"] = entry["target_mode"]
        df["training_seed"] = entry["seed"]
        all_data.append(df)

    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


def summarize_by_method_target(df):
    """Summary: method × target_mode → mean SR ± std across seeds."""
    summary = df.groupby(["method_key", "target_mode"])["is_success"].agg(
        ["mean", "std", "count"]
    ).reset_index()
    summary.columns = ["method", "target_mode", "success_rate_mean", "success_rate_std", "n_episodes"]
    return summary


def plot_heatmap(df, output_dir):
    """Plot success rate heatmap: methods × target_modes."""
    pivot = df.groupby(["method_key", "target_mode"])["is_success"].mean().unstack()

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                       color="white" if val < 0.3 or val > 0.7 else "black",
                       fontsize=10, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Success Rate")
    ax.set_title("Ablation Matrix: Success Rate by Method and Target Mode", fontsize=12)
    ax.set_xlabel("Target Motion Mode")
    ax.set_ylabel("Method")

    plt.tight_layout()
    plt.savefig(output_dir / "ablation_heatmap.png", dpi=300)
    plt.close()


def plot_convergence_curve(manifest, output_dir):
    """Plot success rate vs training steps for best method."""
    # TODO: requires training log parsing
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/ablation_matrix/analysis")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all data
    df = load_all_episodes(manifest)
    if df.empty:
        print("No evaluation data found.")
        return

    # Summary
    summary = summarize_by_method_target(df)
    summary.to_csv(output_dir / "summary_by_method_target.csv", index=False)
    print("Summary saved to:", output_dir / "summary_by_method_target.csv")

    # Heatmap
    plot_heatmap(df, output_dir)
    print("Heatmap saved to:", output_dir / "ablation_heatmap.png")

    # Per-scenario breakdown
    scenario_summary = df.groupby(["method_key", "target_mode", "scenario"])["is_success"].mean().unstack()
    scenario_summary.to_csv(output_dir / "summary_by_scenario.csv")
    print("Scenario summary saved to:", output_dir / "summary_by_scenario.csv")


if __name__ == "__main__":
    main()
