#!/usr/bin/env python3
"""Aggregate potential-based reward shaping (PBS) ablation results."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_condition_dfs(root: Path, condition: str):
    """Load all eval_log.csv files for a condition (single-seed or multi-seed)."""
    single = root / condition / "logs" / "eval_log.csv"
    if single.exists():
        return [pd.read_csv(single)]

    dfs = []
    for seed_dir in sorted((root / condition).glob("seed*")):
        path = seed_dir / "logs" / "eval_log.csv"
        if path.exists():
            dfs.append(pd.read_csv(path))
    return dfs


def aggregate_on_steps(dfs, column):
    """Return mean and std of a column aligned on environment step."""
    merged = None
    for i, df in enumerate(dfs):
        col = df[["step", column]].rename(columns={column: f"run_{i}"})
        merged = col if merged is None else pd.merge(merged, col, on="step", how="outer")
    merged = merged.sort_values("step").set_index("step")
    mean = merged.mean(axis=1, skipna=True)
    std = merged.std(axis=1, skipna=True)
    return mean, std


def first_step_reaching(df, column, threshold):
    reached = df[df[column] >= threshold]
    return int(reached.iloc[0]["step"]) if not reached.empty else None


def main():
    parser = argparse.ArgumentParser(description="Aggregate PBS ablation results")
    parser.add_argument("--root", type=Path, default=Path("outputs/pbs_ablation_multi"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/results/pbs_ablation_multi"))
    args = parser.parse_args()

    conditions = {
        "with_pbs": "With PBS",
        "without_pbs": "Without PBS",
    }
    colors = {"with_pbs": "#2ca02c", "without_pbs": "#d62728"}

    data = {k: load_condition_dfs(args.root, k) for k in conditions}

    args.output_dir.mkdir(parents=True, exist_ok=True)

    lines = ["# Potential-Based Reward Shaping (PBS) Ablation", ""]
    lines.append(f"*Aggregated over the available seeds in `{args.root}`.*")
    lines.append("")
    lines.append("| Condition | Seeds | Final SR | Final Return | First step ≥50% SR | Control Effort | Smoothness |")
    lines.append("|-----------|-------|----------|--------------|--------------------|----------------|------------|")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for key, label in conditions.items():
        dfs = data[key]
        if not dfs:
            continue

        final_srs = [df["success_rate"].iloc[-1] for df in dfs]
        final_returns = [df["mean_return"].iloc[-1] for df in dfs]
        first_50s = [first_step_reaching(df, "success_rate", 0.5) for df in dfs]
        final_efforts = [df["mean_control_effort"].iloc[-1] for df in dfs]
        final_smooths = [df["mean_command_smoothness"].iloc[-1] for df in dfs]

        sr_mean, sr_std = np.mean(final_srs), np.std(final_srs, ddof=1 if len(final_srs) > 1 else 0)
        ret_mean, ret_std = np.mean(final_returns), np.std(final_returns, ddof=1 if len(final_returns) > 1 else 0)
        effort_mean, effort_std = np.mean(final_efforts), np.std(final_efforts, ddof=1 if len(final_efforts) > 1 else 0)
        smooth_mean, smooth_std = np.mean(final_smooths), np.std(final_smooths, ddof=1 if len(final_smooths) > 1 else 0)
        first50_str = f"{int(np.mean([x for x in first_50s if x is not None]))}" if any(x is not None for x in first_50s) else "—"

        lines.append(
            f"| {label} | {len(dfs)} | {sr_mean:.2%} ± {sr_std:.2%} | "
            f"{ret_mean:.1f} ± {ret_std:.1f} | {first50_str} | "
            f"{effort_mean:.0f} ± {effort_std:.0f} | {smooth_mean:.1f} ± {smooth_std:.1f} |"
        )

        for col, ax in zip(["success_rate", "mean_return", "mean_min_range_m"], axes):
            mean, std = aggregate_on_steps(dfs, col)
            ax.plot(mean.index, mean.values, label=label, color=colors[key])
            if len(dfs) > 1:
                ax.fill_between(mean.index, mean.values - std.values, mean.values + std.values,
                                color=colors[key], alpha=0.2)

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("This ablation tests Proposition~22: potential-based reward shaping should accelerate early learning without changing the MDP optimal policy.")
    lines.append("If the multi-seed means for the two conditions converge to similar final success rates, the claim is supported.")
    lines.append("Differences in the first step reaching 50% success rate indicate the early-acceleration effect of PBS.")

    summary_path = args.output_dir / "summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved summary to {summary_path}")

    axes[0].set_xlabel("Environment steps")
    axes[0].set_ylabel("Success rate")
    axes[0].set_title("Success rate")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Environment steps")
    axes[1].set_ylabel("Mean return")
    axes[1].set_title("Mean return")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel("Environment steps")
    axes[2].set_ylabel("Min range (m)")
    axes[2].set_title("Minimum range")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = args.output_dir / "learning_curves.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
