#!/usr/bin/env python3
"""
Deep analysis of disadvantage failure trajectories:
Dense baseline vs Sparse-Gaussian policy (JSBSim).

Saves per-episode CSVs and a summary JSON, plus comparative plots.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.evaluation.evaluate_prediction_comparison import (
    evaluate_single_episode,
)
from uav_vpp_guidance.utils.config import load_yaml_config, merge_config


def load_experiment_config(config_path: str) -> dict:
    base_config = load_yaml_config(config_path)
    includes = base_config.pop("includes", [])
    merged = {}
    for inc_path in includes:
        inc_full = Path(config_path).parent / inc_path
        if inc_full.exists():
            merged = merge_config(merged, load_yaml_config(str(inc_full)))
    return merge_config(merged, base_config)


def make_sparse_config(base_config: dict) -> dict:
    """Convert a dense config into the sparse-gaussian variant used for validation."""
    cfg = merge_config({}, base_config)
    cfg.setdefault("reward", {})["use_sparse"] = True
    cfg["sparse_reward"] = {
        "enabled": True,
        "terminal_success": 10.0,
        "terminal_crash": -10.0,
        "terminal_failure": -5.0,
        "event_range_m": 1200.0,
        "event_ata_deg": 45.0,
        "event_reward": 1.0,
        "lock_reward": 0.0,
        "gaussian_window": 50,
        "gaussian_sigma_ratio": 0.5,
        "relabelling": {
            "enabled": True,
            "terminal_kernel": "linear",
            "event_kernel": "gaussian",
        },
    }
    return cfg


def analyze_policy(name: str, config: dict, checkpoint: str, scenario: dict,
                   episodes: int, seed_base: int, out_dir: Path):
    env = CloseRangeTrackingEnv(config)
    obs = env.reset(seed=0)
    obs_dim = int(obs["observation_vector"].shape[0])
    action_dim = int(config.get("policy", {}).get("action_dim", 3))
    device = config.get("ppo", {}).get("device", "cpu")

    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=config, device=device)
    agent.load(checkpoint)

    records = []
    trajectories_dir = out_dir / "trajectories" / name
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(episodes):
        seed = seed_base + ep
        result, trajectory = evaluate_single_episode(
            env=env,
            agent=agent,
            config=config,
            scenario=scenario,
            seed=seed,
            save_trajectory=True,
            method_name=name,
        )
        records.append(result)
        traj_path = trajectories_dir / f"ep{ep:03d}_seed{seed}.csv"
        if trajectory:
            with open(traj_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=trajectory[0].keys())
                writer.writeheader()
                writer.writerows(trajectory)
        print(
            f"[{name}] ep={ep:03d} success={result['is_success']} "
            f"reason={result['reason']:12s} len={result['length']:3d} "
            f"min_range={result['min_range_m']:.0f} final_range={result['final_range_m']:.0f} "
            f"final_ata={result['final_ata_deg']:.1f}"
        )

    env.close()

    def safe_mean(vals):
        clean = [v for v in vals if np.isfinite(v)]
        return float(np.mean(clean)) if clean else float("nan")

    def safe_std(vals):
        clean = [v for v in vals if np.isfinite(v)]
        return float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0

    summary = {
        "name": name,
        "episodes": len(records),
        "success_rate": sum(r["is_success"] for r in records) / len(records),
        "crash_rate": sum(r["is_crash"] for r in records) / len(records),
        "timeout_rate": sum(r["is_timeout"] for r in records) / len(records),
        "out_of_bounds_rate": sum(r["is_out_of_bounds"] for r in records) / len(records),
        "mean_return": safe_mean([r["return"] for r in records]),
        "std_return": safe_std([r["return"] for r in records]),
        "mean_length": safe_mean([r["length"] for r in records]),
        "mean_min_range_m": safe_mean([r["min_range_m"] for r in records]),
        "std_min_range_m": safe_std([r["min_range_m"] for r in records]),
        "mean_final_range_m": safe_mean([r["final_range_m"] for r in records]),
        "std_final_range_m": safe_std([r["final_range_m"] for r in records]),
        "mean_final_ata_deg": safe_mean([r["final_ata_deg"] for r in records]),
        "std_final_ata_deg": safe_std([r["final_ata_deg"] for r in records]),
        "mean_nz_cmd_max": safe_mean([r["nz_cmd_max"] for r in records]),
        "mean_nz_cmd_saturation_rate": safe_mean([r["nz_cmd_saturation_rate"] for r in records]),
        "mean_roll_rate_cmd_max": safe_mean([r["roll_rate_cmd_max"] for r in records]),
        "mean_roll_rate_cmd_saturation_rate": safe_mean([r["roll_rate_cmd_saturation_rate"] for r in records]),
        "mean_throttle_cmd_mean": safe_mean([r["throttle_cmd_mean"] for r in records]),
        "mean_min_altitude_m": safe_mean([r.get("min_altitude_m", float("nan")) for r in records]),
        "mean_altitude_loss_rate": safe_mean([r.get("altitude_loss_rate", float("nan")) for r in records]),
        "mean_energy_proxy": safe_mean([r["energy_proxy"] for r in records]),
        "mean_time_to_first_advantage_s": safe_mean([r["time_to_first_advantage_s"] for r in records]),
        "mean_advantage_hold_time_s": safe_mean([r["advantage_hold_time_s"] for r in records]),
    }
    return records, summary


def plot_comparison(records_dense, records_sparse, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping plots")
        return

    def _extract_series(records, key):
        series = []
        for r in records:
            traj_path = out_dir / "trajectories" / r["name"] / f"ep{r['episode']:03d}_seed{r['seed']}.csv"
            # episode index not stored in result; fallback to matching by seed is fragile.
            # Instead the caller attaches trajectory data separately. Here we skip per-episode file IO.
        return series

    # Compute terminal-geometry histograms from result dicts.
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    def _hist(ax, vals_dense, vals_sparse, title, xlabel):
        bins = np.linspace(min(min(vals_dense), min(vals_sparse)),
                           max(max(vals_dense), max(vals_sparse)), 21)
        ax.hist(vals_dense, bins=bins, alpha=0.6, label="dense", color="black")
        ax.hist(vals_sparse, bins=bins, alpha=0.6, label="sparse_gaussian", color="green")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _finite(vals):
        return [v for v in vals if np.isfinite(v)]

    _hist(axes[0, 0],
          _finite([r["final_range_m"] for r in records_dense]),
          _finite([r["final_range_m"] for r in records_sparse]),
          "Final range distribution (disadvantage)", "range_m")
    _hist(axes[0, 1],
          _finite([abs(r["final_ata_deg"]) for r in records_dense]),
          _finite([abs(r["final_ata_deg"]) for r in records_sparse]),
          "Final |ATA| distribution (disadvantage)", "deg")
    _hist(axes[1, 0],
          _finite([r["min_range_m"] for r in records_dense]),
          _finite([r["min_range_m"] for r in records_sparse]),
          "Minimum range distribution (disadvantage)", "range_m")
    _hist(axes[1, 1],
          _finite([r["length"] for r in records_dense]),
          _finite([r["length"] for r in records_sparse]),
          "Episode length distribution (disadvantage)", "steps")

    fig.tight_layout()
    fig.savefig(out_dir / "disadvantage_failure_distributions.png", dpi=150)
    plt.close(fig)
    print(f"Saved distribution plot to {out_dir / 'disadvantage_failure_distributions.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/experiment/maneuver_target_vpp_pilot.yaml")
    parser.add_argument("--scenario", default="disadvantage")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--dense-checkpoint",
                        default="outputs/sparse_reward_comparison_jsbsim/dense/checkpoints/best.pt")
    parser.add_argument("--sparse-checkpoint",
                        default="outputs/sparse_reward_comparison_jsbsim/sparse_gaussian/checkpoints/best.pt")
    parser.add_argument("--output-dir", default="outputs/disadvantage_dense_vs_sparse_analysis")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_config = load_experiment_config(args.config)
    base_config.setdefault("experiment", {})["seed"] = 0
    scenario = base_config["scenarios"][args.scenario]

    print(f"Analyzing {args.episodes} '{args.scenario}' episodes for Dense and Sparse-Gaussian ...")

    records_dense, summary_dense = analyze_policy(
        "dense", base_config, args.dense_checkpoint, scenario,
        args.episodes, args.seed_base, out_dir,
    )
    records_sparse, summary_sparse = analyze_policy(
        "sparse_gaussian", make_sparse_config(base_config), args.sparse_checkpoint, scenario,
        args.episodes, args.seed_base + 10000, out_dir,
    )

    # Tag records for downstream plotting.
    for r in records_dense:
        r["name"] = "dense"
    for r in records_sparse:
        r["name"] = "sparse_gaussian"

    plot_comparison(records_dense, records_sparse, out_dir)

    report = {
        "scenario": args.scenario,
        "episodes": args.episodes,
        "dense": summary_dense,
        "sparse_gaussian": summary_sparse,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nSummary written to {out_dir / 'summary.json'}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
