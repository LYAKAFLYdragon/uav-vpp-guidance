#!/usr/bin/env python3
"""Generate Stage 6B paper figures: training curves and comparison boxplots."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


def plot_training_curves(output_dir: Path):
    """Plot training success rate curves for 3 methods x 3 seeds."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    methods = {
        "No-Prediction": "outputs/experiments/no_prediction_vpp_ppo_seed",
        "CV Prediction": "outputs/experiments/vpp_ppo_cv_prediction_seed",
        "CA Prediction": "outputs/experiments/vpp_ppo_ca_prediction_seed",
    }
    colors = {"0": "#1f77b4", "1": "#ff7f0e", "2": "#2ca02c"}

    for idx, (method_name, path_prefix) in enumerate(methods.items()):
        ax = axes[idx]
        for seed in ["0", "1", "2"]:
            log_path = Path(f"{path_prefix}{seed}/logs/eval_log.csv")
            if not log_path.exists():
                continue
            df = pd.read_csv(log_path)
            ax.plot(df["step"], df["success_rate"], label=f"Seed {seed}", color=colors[seed], alpha=0.8)
            ax.fill_between(
                df["step"],
                np.clip(df["success_rate"] - 0.05, 0, 1),
                np.clip(df["success_rate"] + 0.05, 0, 1),
                alpha=0.1, color=colors[seed],
            )
        ax.set_xlabel("Training Steps", fontsize=11)
        ax.set_ylabel("Success Rate", fontsize=11)
        ax.set_title(method_name, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "training_curves.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_dir / 'training_curves.png'}")


def plot_comparison_boxplot(benchmark_csv: Path, output_dir: Path):
    """Plot per-scenario success rate boxplot for 3 methods."""
    df = pd.read_csv(benchmark_csv)
    methods = ["no_prediction", "cv_prediction", "ca_prediction"]
    df = df[df["method"].isin(methods)]

    fig, ax = plt.subplots(figsize=(10, 6))

    scenarios = sorted(df["scenario"].unique())
    x = np.arange(len(scenarios))
    width = 0.25

    for i, method in enumerate(methods):
        means = []
        stds = []
        for scen in scenarios:
            data = df[(df["method"] == method) & (df["scenario"] == scen)]["is_success"].astype(float)
            means.append(data.mean() if len(data) > 0 else 0)
            stds.append(data.std() if len(data) > 1 else 0)
        ax.bar(x + i * width, means, width, yerr=stds, label=method.replace("_", " ").title(), alpha=0.85, capsize=3)

    ax.set_xlabel("Scenario", fontsize=11)
    ax.set_ylabel("Success Rate", fontsize=11)
    ax.set_title("Stage 6B: Success Rate by Method and Scenario", fontsize=12, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(scenarios, rotation=15, ha="right")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "comparison_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_dir / 'comparison_boxplot.png'}")


def plot_success_rate_overall(benchmark_csv: Path, output_dir: Path):
    """Plot overall success rate comparison with error bars."""
    df = pd.read_csv(benchmark_csv)
    methods = ["no_prediction", "cv_prediction", "ca_prediction"]
    df = df[df["method"].isin(methods)]

    fig, ax = plt.subplots(figsize=(7, 5))

    means = []
    stds = []
    labels = []
    for method in methods:
        data = df[df["method"] == method]["is_success"].astype(float)
        means.append(data.mean())
        stds.append(data.std() / np.sqrt(len(data)))  # SEM
        labels.append(method.replace("_", " ").title())

    colors = ["#2ca02c", "#1f77b4", "#ff7f0e"]
    bars = ax.bar(labels, means, yerr=stds, color=colors, alpha=0.85, capsize=5, edgecolor="black", linewidth=0.5)

    # Add value labels on bars
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{mean:.1%}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Success Rate", fontsize=11)
    ax.set_title("Stage 6B: Overall Success Rate Comparison", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "overall_success_rate.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_dir / 'overall_success_rate.png'}")


def main():
    output_dir = Path("docs/results/stage6b")
    benchmark_csv = Path("outputs/stage6b/benchmark/raw_episodes.csv")

    plot_training_curves(output_dir)
    if benchmark_csv.exists():
        plot_comparison_boxplot(benchmark_csv, output_dir)
        plot_success_rate_overall(benchmark_csv, output_dir)
    else:
        print(f"Warning: {benchmark_csv} not found, skipping boxplots")

    print("\nStage 6B figures generated successfully.")


if __name__ == "__main__":
    main()
