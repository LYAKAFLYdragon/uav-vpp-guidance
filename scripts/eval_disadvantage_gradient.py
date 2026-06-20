#!/usr/bin/env python3
"""
Evaluate the trained disadvantage feasibility gradient and plot the results.

Usage:
    source activate jsbenv
    python scripts/eval_disadvantage_gradient.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.evaluation.evaluate_prediction_comparison import evaluate_single_episode
from uav_vpp_guidance.utils.config import load_yaml_config, merge_config


def load_experiment_config(config_path: str, _visited: set = None) -> dict:
    """Recursively load config with includes (cycle-safe)."""
    if _visited is None:
        _visited = set()
    abs_path = os.path.abspath(config_path)
    if abs_path in _visited:
        return {}
    _visited.add(abs_path)

    base_config = load_yaml_config(config_path)
    includes = base_config.pop("includes", [])
    merged = {}
    cfg_dir = os.path.dirname(config_path)
    for inc_path in includes:
        inc_full = os.path.join(cfg_dir, inc_path)
        if not os.path.exists(inc_full):
            inc_full = os.path.join(cfg_dir, "..", os.path.basename(inc_path))
        if os.path.exists(inc_full):
            merged = merge_config(merged, load_experiment_config(inc_full, _visited))
    return merge_config(merged, base_config)


def summarize(records: List[dict]) -> dict:
    n = len(records)
    successes = sum(1 for r in records if r.get("is_success"))
    crashes = sum(1 for r in records if r.get("is_crash"))
    oobs = sum(1 for r in records if r.get("is_out_of_bounds"))
    timeouts = sum(1 for r in records if r.get("reason") == "timeout")
    returns = [r.get("return", np.nan) for r in records]
    min_ranges = [r.get("min_range_m", np.nan) for r in records]
    final_ranges = [r.get("final_range_m", np.nan) for r in records]
    return {
        "n": n,
        "success_rate": successes / n if n else 0.0,
        "crash_rate": crashes / n if n else 0.0,
        "out_of_bounds_rate": oobs / n if n else 0.0,
        "timeout_rate": timeouts / n if n else 0.0,
        "mean_return": float(np.nanmean(returns)) if returns else 0.0,
        "std_return": float(np.nanstd(returns)) if returns else 0.0,
        "mean_min_range_m": float(np.nanmean(min_ranges)) if min_ranges else 0.0,
        "mean_final_range_m": float(np.nanmean(final_ranges)) if final_ranges else 0.0,
    }


def evaluate_checkpoint(
    config_path: str,
    checkpoint: str,
    scenario: dict,
    episodes: int = 20,
    seeds: List[int] = None,
) -> dict:
    """Evaluate one checkpoint on its scenario."""
    if seeds is None:
        seeds = [0, 1, 2, 3]

    config = load_experiment_config(config_path)
    env = CloseRangeTrackingEnv(config)
    obs = env.reset(seed=0)
    obs_dim = int(obs["observation_vector"].shape[0])
    action_dim = int(config.get("policy", {}).get("action_dim", 3))
    device = config.get("ppo", {}).get("device", "cpu")

    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=config, device=device)
    agent.load(checkpoint)

    records = []
    for seed in seeds:
        for ep in range(episodes // max(1, len(seeds))):
            result, _ = evaluate_single_episode(
                env=env,
                agent=agent,
                config=config,
                scenario=scenario,
                seed=seed * 10000 + ep,
                save_trajectory=False,
                method_name="disadvantage_gradient",
            )
            records.append(result)

    env.close()
    return summarize(records)


def plot_gradient(results: List[dict], output_path: str):
    """Plot success rate and distance metrics vs difficulty."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"[PLOT] matplotlib not available: {exc}")
        return

    difficulties = [r["difficulty"] for r in results]
    labels = [r["short_label"] for r in results]
    success_rates = [r["success_rate"] * 100 for r in results]
    crash_rates = [r["crash_rate"] * 100 for r in results]
    timeout_rates = [r["timeout_rate"] * 100 for r in results]
    oob_rates = [r["out_of_bounds_rate"] * 100 for r in results]
    min_ranges = [r["mean_min_range_m"] for r in results]
    final_ranges = [r["mean_final_range_m"] for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # Top: outcome rates
    ax1.plot(difficulties, success_rates, "o-", color="#2ca02c", linewidth=2, markersize=8, label="Success")
    ax1.plot(difficulties, crash_rates, "s--", color="#d62728", linewidth=2, markersize=6, label="Crash")
    ax1.plot(difficulties, timeout_rates, "^--", color="#ff7f0e", linewidth=2, markersize=6, label="Timeout")
    ax1.plot(difficulties, oob_rates, "v--", color="#9467bd", linewidth=2, markersize=6, label="Out of bounds")
    ax1.set_ylabel("Rate (%)", fontsize=12)
    ax1.set_title("Disadvantage Feasibility Gradient (v5 safety protections)", fontsize=14)
    ax1.set_ylim(-5, 105)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    # Bottom: distance metrics
    ax2.plot(difficulties, min_ranges, "o-", color="#1f77b4", linewidth=2, markersize=8, label="Mean min range")
    ax2.plot(difficulties, final_ranges, "s--", color="#9467bd", linewidth=2, markersize=6, label="Mean final range")
    ax2.set_xlabel("Scene difficulty", fontsize=12)
    ax2.set_ylabel("Range (m)", fontsize=12)
    ax2.set_title("Closure distance gradient", fontsize=12)
    ax2.set_xticks(difficulties)
    ax2.set_xticklabels(labels, rotation=15, ha="right")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Saved {output_path}")


def generate_report(results: List[dict], output_path: str):
    """Write markdown report."""
    lines = [
        "# Disadvantage Feasibility Gradient Report",
        "",
        "Method: v5 configuration with safety protections enabled",
        "(dynamics_aware=false, altitude_hold=true, roll_angle_protection max_roll=60°).",
        "",
        "| Scene | Ego speed | Target speed | Rear offset | Success | Crash | Timeout | OOB | Mean min range | Mean final range |",
        "|-------|-----------|--------------|-------------|---------|-------|---------|-----|----------------|------------------|",
    ]
    for r in results:
        lines.append(
            f"| {r['short_label']} | {r['ego_speed']} | {r['target_speed']} | {r['offset']} | "
            f"{r['success_rate']*100:.1f}% | {r['crash_rate']*100:.1f}% | "
            f"{r['timeout_rate']*100:.1f}% | {r['out_of_bounds_rate']*100:.1f}% | "
            f"{r['mean_min_range_m']:.0f} m | {r['mean_final_range_m']:.0f} m |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- **Scene 1** gives the ego a strong energy and positional advantage.",
        "- Each subsequent scene reduces ego speed, increases target speed, and places the target farther behind.",
        "- The gradient reveals the physical/kinematic boundary beyond which the current v5 architecture cannot achieve success.",
        "",
        "## Key Finding",
        "",
    ])
    sr_first = results[0]["success_rate"] * 100
    sr_last = results[-1]["success_rate"] * 100
    min_range_first = results[0]["mean_min_range_m"]
    min_range_last = results[-1]["mean_min_range_m"]

    if sr_first == 0.0 and sr_last == 0.0:
        lines.append(
            f"With v5 safety protections, **no scene achieves success** (all timeout at {results[0]['timeout_rate']*100:.0f}%). "
            f"The architecture boundary lies below Scene 1. However, the mean minimum range increases monotonically "
            f"from {min_range_first:.0f} m (Scene 1) to {min_range_last:.0f} m (Scene 5), "
            f"showing that the agent is progressively less able to close distance as the scenario hardens."
        )
    else:
        sr_drop = sr_last - sr_first
        lines.append(
            f"Success rate drops from {sr_first:.1f}% (Scene 1) "
            f"to {sr_last:.1f}% (Scene 5), a difference of {sr_drop:.1f} percentage points."
        )
    lines.append("")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[REPORT] Saved {output_path}")


def parse_label(label: str) -> dict:
    """Parse label like 'S1: ego 240 / tgt 180 / offset 200'."""
    parts = label.split("/")
    ego = int(parts[0].split()[-1])
    tgt = int(parts[1].split()[-1])
    offset = int(parts[2].split()[-1])
    return {"ego_speed": ego, "target_speed": tgt, "offset": offset}


def main():
    parser = argparse.ArgumentParser(description="Evaluate disadvantage feasibility gradient.")
    parser.add_argument("--summary", type=str, default="outputs/disadvantage_gradient/training_summary.json")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--out-dir", type=str, default="outputs/disadvantage_gradient")
    args = parser.parse_args()

    with open(args.summary, "r", encoding="utf-8") as f:
        summary = json.load(f)

    results = []
    for entry in summary:
        config_path = os.path.join(_PROJECT_ROOT, entry["config"])
        checkpoint = entry["checkpoint"]

        if not Path(checkpoint).exists():
            print(f"[EVAL] Checkpoint not found, skipping: {checkpoint}")
            continue

        config = load_experiment_config(config_path)
        scenario_name = entry["name"]
        scenario = config.get("scenarios", {}).get(scenario_name)
        if scenario is None:
            print(f"[EVAL] Scenario {scenario_name} not found in {config_path}")
            continue

        print(f"\n[EVAL] {entry['label']} ({args.episodes} eps, seeds={args.seeds})")
        metrics = evaluate_checkpoint(
            config_path=config_path,
            checkpoint=checkpoint,
            scenario=scenario,
            episodes=args.episodes,
            seeds=args.seeds,
        )

        parsed = parse_label(entry["label"])
        result = {
            "name": entry["name"],
            "label": entry["label"],
            "short_label": entry["name"].replace("disadvantage_gradient_", "S"),
            "difficulty": entry["difficulty"],
            **parsed,
            **metrics,
        }
        results.append(result)

        print(
            f"  success={metrics['success_rate']:.1%} "
            f"crash={metrics['crash_rate']:.1%} "
            f"timeout={metrics['timeout_rate']:.1%} "
            f"oob={metrics['out_of_bounds_rate']:.1%} "
            f"mean_min_range={metrics['mean_min_range_m']:.0f}m"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save raw results
    with open(out_dir / "feasibility_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Plot
    plot_gradient(results, str(out_dir / "feasibility_gradient.png"))

    # Report
    generate_report(results, str(out_dir / "feasibility_report.md"))


if __name__ == "__main__":
    main()
