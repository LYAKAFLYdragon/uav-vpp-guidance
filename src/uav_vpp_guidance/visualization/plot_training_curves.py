"""
Training curve plotting script.

Reads train_log.csv and eval_log.csv from a PPO experiment and generates
paper-ready figures.

Usage:
    python -m uav_vpp_guidance.visualization.plot_training_curves \
        --log-dir outputs/experiments/no_prediction_vpp_ppo/logs \
        --output outputs/experiments/no_prediction_vpp_ppo/figures
"""

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv_dict(path):
    """Read a CSV file into a list of dictionaries."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def to_float(value, default=0.0):
    """Safely convert string to float."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def smooth_curve(values, weight=0.6):
    """Exponential moving average smoothing."""
    if not values:
        return []
    smoothed = []
    last = values[0]
    for v in values:
        last = last * weight + v * (1 - weight)
        smoothed.append(last)
    return smoothed


def plot_episode_return(train_data, output_path):
    """Plot episode return over training steps."""
    steps = [to_float(r["step"]) for r in train_data if r.get("step")]
    returns = [to_float(r["episode_return"]) for r in train_data if r.get("episode_return")]

    if not steps or not returns:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, returns, alpha=0.3, color="steelblue", label="Raw")
    if len(returns) > 5:
        ax.plot(steps, smooth_curve(returns), color="darkblue", linewidth=1.5, label="Smoothed")
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Episode Return", fontsize=12)
    ax.set_title("PPO Training: Episode Return", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_success_rate(train_data, eval_data, output_path):
    """Plot success rate from training and evaluation."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Training success (per episode)
    train_steps = [to_float(r["step"]) for r in train_data if r.get("success") != ""]
    train_success = [to_float(r["success"]) for r in train_data if r.get("success") != ""]
    if train_steps and train_success:
        ax.scatter(train_steps, train_success, alpha=0.3, s=10, color="steelblue", label="Train Episode")
        if len(train_success) > 10:
            window = min(20, len(train_success) // 5)
            if window > 1:
                smoothed = []
                for i in range(len(train_success)):
                    start = max(0, i - window // 2)
                    end = min(len(train_success), i + window // 2 + 1)
                    smoothed.append(np.mean(train_success[start:end]))
                ax.plot(train_steps, smoothed, color="darkblue", linewidth=1.5, label="Train Smoothed")

    # Evaluation success
    eval_steps = [to_float(r["step"]) for r in eval_data if r.get("success_rate")]
    eval_success = [to_float(r["success_rate"]) for r in eval_data if r.get("success_rate")]
    if eval_steps and eval_success:
        ax.plot(eval_steps, eval_success, "o-", color="darkorange", linewidth=2, markersize=6, label="Eval")

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_title("PPO Training: Success Rate", fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_score_win_rate(eval_data, output_path):
    """Plot score win rate from evaluation."""
    eval_steps = [to_float(r["step"]) for r in eval_data if r.get("success_rate")]
    if not eval_steps:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    # We don't have explicit score_win in eval_data, but we can show success rate
    # as a proxy for score advantage. In future versions, add score_win_rate to eval.
    success_rates = [to_float(r["success_rate"]) for r in eval_data if r.get("success_rate")]
    ax.plot(eval_steps, success_rates, "o-", color="seagreen", linewidth=2, markersize=6)
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_title("PPO Evaluation: Success Rate", fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_losses(train_data, output_path):
    """Plot policy loss, value loss, and entropy."""
    steps = []
    policy_losses = []
    value_losses = []
    entropies = []

    for r in train_data:
        if r.get("policy_loss") not in ("", None):
            steps.append(to_float(r["step"]))
            policy_losses.append(to_float(r["policy_loss"]))
            value_losses.append(to_float(r["value_loss"]))
            entropies.append(to_float(r["entropy"]))

    if not steps:
        return

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    axes[0].plot(steps, policy_losses, color="darkblue", linewidth=1.2)
    axes[0].set_ylabel("Policy Loss", fontsize=11)
    axes[0].set_title("PPO Training Losses", fontsize=14)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, value_losses, color="darkred", linewidth=1.2)
    axes[1].set_ylabel("Value Loss", fontsize=11)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, entropies, color="darkgreen", linewidth=1.2)
    axes[2].set_ylabel("Entropy", fontsize=11)
    axes[2].set_xlabel("Training Step", fontsize=12)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_final_range_ata(eval_data, output_path):
    """Plot mean final range and ATA from evaluation."""
    eval_steps = [to_float(r["step"]) for r in eval_data if r.get("mean_final_range_m")]
    if not eval_steps:
        return

    ranges = [to_float(r["mean_final_range_m"]) for r in eval_data if r.get("mean_final_range_m")]
    atas = [to_float(r["mean_final_ata_deg"]) for r in eval_data if r.get("mean_final_ata_deg")]

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    axes[0].plot(eval_steps, ranges, "o-", color="purple", linewidth=2, markersize=6)
    axes[0].set_ylabel("Mean Final Range (m)", fontsize=11)
    axes[0].set_title("PPO Evaluation: Final Geometry", fontsize=14)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(eval_steps, atas, "o-", color="brown", linewidth=2, markersize=6)
    axes[1].set_ylabel("Mean Final ATA (deg)", fontsize=11)
    axes[1].set_xlabel("Training Step", fontsize=12)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_kl_clip(train_data, output_path):
    """Plot approx KL divergence and clip fraction."""
    steps = []
    kls = []
    clips = []

    for r in train_data:
        if r.get("approx_kl") not in ("", None):
            steps.append(to_float(r["step"]))
            kls.append(to_float(r["approx_kl"]))
            clips.append(to_float(r["clip_fraction"]))

    if not steps:
        return

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    axes[0].plot(steps, kls, color="teal", linewidth=1.2)
    axes[0].set_ylabel("Approx KL", fontsize=11)
    axes[0].set_title("PPO Training: KL and Clip Fraction", fontsize=14)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, clips, color="coral", linewidth=1.2)
    axes[1].set_ylabel("Clip Fraction", fontsize=11)
    axes[1].set_xlabel("Training Step", fontsize=12)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot PPO training curves")
    parser.add_argument("--log-dir", type=str, required=True, help="Directory containing train_log.csv and eval_log.csv")
    parser.add_argument("--output", type=str, required=True, help="Output directory for figures")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    train_data = read_csv_dict(os.path.join(args.log_dir, "train_log.csv"))
    eval_data = read_csv_dict(os.path.join(args.log_dir, "eval_log.csv"))

    print(f"Loaded {len(train_data)} training records, {len(eval_data)} evaluation records")

    plot_episode_return(train_data, os.path.join(args.output, "training_return.png"))
    plot_success_rate(train_data, eval_data, os.path.join(args.output, "training_success_rate.png"))
    plot_score_win_rate(eval_data, os.path.join(args.output, "training_score_win_rate.png"))
    plot_losses(train_data, os.path.join(args.output, "training_loss.png"))
    plot_final_range_ata(eval_data, os.path.join(args.output, "eval_range_ata.png"))
    plot_kl_clip(train_data, os.path.join(args.output, "training_kl_clip.png"))

    print(f"\nAll figures saved to: {args.output}")


if __name__ == "__main__":
    main()
