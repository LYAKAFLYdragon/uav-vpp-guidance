#!/usr/bin/env python3
"""
Aggregate 3-seed smoke comparison across Baseline PPO, CR-PPO, and Intentional PPO.

Reads:
  outputs/smoke_compare/{baseline,cr_ppo,intentional}/seed{0,1,2}/logs/eval_log.csv

Writes:
  outputs/smoke_compare/summary.csv
  outputs/smoke_compare/summary.md
"""

import csv
import os
from pathlib import Path

import numpy as np


ROOT = Path("outputs/smoke_compare")
ALGORITHMS = {
    "baseline": "Baseline PPO",
    "cr_ppo": "CR-PPO",
    "intentional": "Intentional PPO",
}
SEEDS = [0, 1, 2]
METRICS = [
    "mean_return",
    "success_rate",
    "crash_rate",
    "out_of_bounds_rate",
    "timeout_rate",
]
UPDATE_METRICS_COMMON = [
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
    "explained_variance",
]
UPDATE_METRICS_EXTRA = {
    "baseline": [],
    "cr_ppo": ["complexity"],
    "intentional": ["scale_actor", "scale_critic", "ema_abs_adv"],
}


def read_last_eval(algorithm, seed):
    path = ROOT / algorithm / f"seed{seed}" / "logs" / "eval_log.csv"
    if not path.exists():
        return None
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        return None
    last = rows[-1]
    return {m: float(last.get(m, np.nan)) for m in METRICS}


def read_last_update(algorithm, seed):
    path = ROOT / algorithm / f"seed{seed}" / "logs" / "update_train_log.csv"
    if not path.exists():
        return None
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        return None
    last = rows[-1]
    keys = UPDATE_METRICS_COMMON + UPDATE_METRICS_EXTRA.get(algorithm, [])
    return {k: float(last.get(k, np.nan)) for k in keys}


def aggregate_algorithm(algorithm):
    per_seed_eval = []
    per_seed_update = []
    for seed in SEEDS:
        eval_m = read_last_eval(algorithm, seed)
        update_m = read_last_update(algorithm, seed)
        if eval_m is not None:
            per_seed_eval.append(eval_m)
        if update_m is not None:
            per_seed_update.append(update_m)
    if not per_seed_eval:
        return None

    result = {"n_seeds": len(per_seed_eval)}
    for m in METRICS:
        vals = [s[m] for s in per_seed_eval]
        result[f"{m}_mean"] = float(np.mean(vals))
        result[f"{m}_std"] = float(np.std(vals))

    update_keys = UPDATE_METRICS_COMMON + UPDATE_METRICS_EXTRA.get(algorithm, [])
    for m in update_keys:
        vals = [s[m] for s in per_seed_update if m in s]
        if vals:
            result[f"{m}_mean"] = float(np.mean(vals))
            result[f"{m}_std"] = float(np.std(vals))

    return result


def main():
    summary = {}
    for key, name in ALGORITHMS.items():
        agg = aggregate_algorithm(key)
        if agg is not None:
            summary[key] = agg
            summary[key]["name"] = name

    if not summary:
        print(f"No evaluation logs found under {ROOT}")
        return

    # Print markdown table
    md_lines = [
        "# 3-Seed Smoke Comparison Summary",
        "",
        "## Final Evaluation Metrics",
        "",
        "| Algorithm | Return (mean ± std) | Success Rate | Crash Rate | OOB Rate | Timeout Rate |",
        "|-----------|--------------------|--------------|-----------|---------|--------------|",
    ]
    for key in ALGORITHMS:
        if key not in summary:
            continue
        s = summary[key]
        md_lines.append(
            f"| {s['name']} | "
            f"{s['mean_return_mean']:.2f} ± {s['mean_return_std']:.2f} | "
            f"{s['success_rate_mean']:.2%} ± {s['success_rate_std']:.2%} | "
            f"{s['crash_rate_mean']:.2%} ± {s['crash_rate_std']:.2%} | "
            f"{s['out_of_bounds_rate_mean']:.2%} ± {s['out_of_bounds_rate_std']:.2%} | "
            f"{s['timeout_rate_mean']:.2%} ± {s['timeout_rate_std']:.2%} |"
        )

    md_lines.append("")
    md_lines.append("## Last Training Update Metrics")
    md_lines.append("")
    md_lines.append("| Algorithm | Policy Loss | Value Loss | Entropy | Approx KL | Clip Fraction | Explained Var |")
    md_lines.append("|-----------|-------------|------------|---------|-----------|---------------|---------------|")
    for key in ALGORITHMS:
        if key not in summary:
            continue
        s = summary[key]
        md_lines.append(
            f"| {s['name']} | "
            f"{s.get('policy_loss_mean', np.nan):.4f} ± {s.get('policy_loss_std', 0):.4f} | "
            f"{s.get('value_loss_mean', np.nan):.2f} ± {s.get('value_loss_std', 0):.2f} | "
            f"{s.get('entropy_mean', np.nan):.4f} ± {s.get('entropy_std', 0):.4f} | "
            f"{s.get('approx_kl_mean', np.nan):.4f} ± {s.get('approx_kl_std', 0):.4f} | "
            f"{s.get('clip_fraction_mean', np.nan):.2%} ± {s.get('clip_fraction_std', 0):.2%} | "
            f"{s.get('explained_variance_mean', np.nan):.4f} ± {s.get('explained_variance_std', 0):.4f} |"
        )

    # Algorithm-specific extras
    md_lines.append("")
    md_lines.append("## Algorithm-Specific Diagnostics")
    md_lines.append("")
    md_lines.append("| Algorithm | Extra Metrics |")
    md_lines.append("|-----------|---------------|")
    for key in ALGORITHMS:
        if key not in summary:
            continue
        s = summary[key]
        extras = UPDATE_METRICS_EXTRA.get(key, [])
        if not extras:
            md_lines.append(f"| {s['name']} | — |")
            continue
        parts = []
        for m in extras:
            mean = s.get(f"{m}_mean", np.nan)
            std = s.get(f"{m}_std", 0)
            parts.append(f"{m}={mean:.4f}±{std:.4f}")
        md_lines.append(f"| {s['name']} | {', '.join(parts)} |")

    md_lines.append("")
    md_lines.append("*Smoke configuration: `--smoke`, `backend=simple`, 3 random seeds each. "
                    "Smoke mode uses only 512 env steps, so final success rates are expected to be near zero. "
                    "Focus on training-stability diagnostics rather than final performance.*")

    md_text = "\n".join(md_lines)
    print(md_text)

    # Save files
    ROOT.mkdir(parents=True, exist_ok=True)
    md_path = ROOT / "summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"\nSaved markdown summary to {md_path}")

    csv_path = ROOT / "summary.csv"
    all_update_keys = UPDATE_METRICS_COMMON.copy()
    for extras in UPDATE_METRICS_EXTRA.values():
        all_update_keys.extend(extras)
    all_update_keys = list(dict.fromkeys(all_update_keys))  # preserve order, dedup
    fieldnames = ["algorithm", "n_seeds"] + [
        f"{m}_{stat}" for m in METRICS for stat in ("mean", "std")
    ] + [
        f"{m}_{stat}" for m in all_update_keys for stat in ("mean", "std")
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in ALGORITHMS:
            if key not in summary:
                continue
            row = {"algorithm": summary[key]["name"], "n_seeds": summary[key]["n_seeds"]}
            for m in METRICS:
                row[f"{m}_mean"] = summary[key][f"{m}_mean"]
                row[f"{m}_std"] = summary[key][f"{m}_std"]
            for m in all_update_keys:
                row[f"{m}_mean"] = summary[key].get(f"{m}_mean", np.nan)
                row[f"{m}_std"] = summary[key].get(f"{m}_std", np.nan)
            writer.writerow(row)
    print(f"Saved CSV summary to {csv_path}")


if __name__ == "__main__":
    main()
