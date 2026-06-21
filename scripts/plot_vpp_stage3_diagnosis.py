"""Plot diagnostic curves for VPP Stage 3 vs No-VPP Stage 3."""
import os
import sys
import csv
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) if k != "step" else int(v) for k, v in row.items()})
    return rows


def plot_metric(ax, series_dict, key, title, ylabel, ylim=None, log=False):
    for label, rows in series_dict.items():
        steps = [r["step"] for r in rows]
        vals = [r.get(key, 0.0) for r in rows]
        # replace inf/nan for plotting
        safe = []
        for v in vals:
            if np.isfinite(v) and abs(v) < 1e12:
                safe.append(v)
            else:
                safe.append(np.nan)
        ax.plot(steps, safe, label=label, linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(ylim)
    if log:
        ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)


def main():
    base = Path("outputs/adversarial_curriculum_pilot")
    conditions = {
        "VPP Stage 2": base / "vpp_full_gate25_s0/stage2/pursuer/logs/adversarial_training_log.csv",
        "VPP Stage 3": base / "vpp_full_gate25_s0/stage3/pursuer/logs/adversarial_training_log.csv",
        "No-VPP Stage 2": base / "no_vpp_full_gate10_s0/stage2/pursuer/logs/adversarial_training_log.csv",
        "No-VPP Stage 3": base / "no_vpp_full_gate10_s0/stage3/pursuer/logs/adversarial_training_log.csv",
    }

    data = {}
    for name, path in conditions.items():
        if path.exists():
            data[name] = load_csv(path)

    out_dir = Path("outputs/adversarial_pilot/diagnosis_figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: capture_rate
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_metric(ax, data, "capture_rate", "Pursuer Capture Rate", "capture_rate", ylim=(-0.05, 0.7))
    fig.tight_layout()
    fig.savefig(out_dir / "pursuer_capture_rate.png", dpi=150)
    plt.close(fig)

    # Plot 2: value_loss (log scale, clipped)
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_metric(ax, data, "value_loss", "Pursuer Value Loss", "value_loss", log=True)
    fig.tight_layout()
    fig.savefig(out_dir / "pursuer_value_loss.png", dpi=150)
    plt.close(fig)

    # Plot 3: entropy
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_metric(ax, data, "entropy", "Pursuer Entropy", "entropy")
    fig.tight_layout()
    fig.savefig(out_dir / "pursuer_entropy.png", dpi=150)
    plt.close(fig)

    # Plot 4: approx_kl
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_metric(ax, data, "approx_kl", "Pursuer Approx KL", "approx_kl")
    fig.tight_layout()
    fig.savefig(out_dir / "pursuer_approx_kl.png", dpi=150)
    plt.close(fig)

    # Plot 5: policy_loss
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_metric(ax, data, "policy_loss", "Pursuer Policy Loss", "policy_loss")
    fig.tight_layout()
    fig.savefig(out_dir / "pursuer_policy_loss.png", dpi=150)
    plt.close(fig)

    # Plot 6: mean_return (clipped)
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_metric(ax, data, "mean_return", "Pursuer Mean Return", "mean_return")
    fig.tight_layout()
    fig.savefig(out_dir / "pursuer_mean_return.png", dpi=150)
    plt.close(fig)

    # Target plots
    target_conditions = {
        "VPP Stage 2": base / "vpp_full_gate25_s0/stage2/target/logs/target_training_log.csv",
        "VPP Stage 3": base / "vpp_full_gate25_s0/stage3/target/logs/target_training_log.csv",
        "No-VPP Stage 2": base / "no_vpp_full_gate10_s0/stage2/target/logs/target_training_log.csv",
        "No-VPP Stage 3": base / "no_vpp_full_gate10_s0/stage3/target/logs/target_training_log.csv",
    }
    target_data = {}
    for name, path in target_conditions.items():
        if path.exists():
            target_data[name] = load_csv(path)

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_metric(ax, target_data, "eval_survival_rate", "Target Eval Survival Rate", "survival_rate", ylim=(-0.05, 0.7))
    fig.tight_layout()
    fig.savefig(out_dir / "target_survival_rate.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_metric(ax, target_data, "eval_capture_rate", "Target Eval Capture Rate", "capture_rate", ylim=(-0.05, 0.7))
    fig.tight_layout()
    fig.savefig(out_dir / "target_capture_rate.png", dpi=150)
    plt.close(fig)

    print(f"Figures saved to {out_dir}")


if __name__ == "__main__":
    main()
