#!/usr/bin/env python3
"""Analyze VPP-Disadvantage-Focused-v2 behavior on disadvantage scenarios."""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.evaluation.evaluate_prediction_comparison import evaluate_single_episode
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


def main():
    cfg_path = "config/experiment/maneuver_target_vpp_pilot_disadvantage_focused_v2.yaml"
    ckpt_path = "outputs/experiments/maneuver_target_focused_v2_vpp_30k_s0/checkpoints/best.pt"
    out_dir = Path("outputs/maneuver_target_pilot/disadvantage_v2_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_experiment_config(cfg_path)
    env = CloseRangeTrackingEnv(config)
    obs = env.reset(seed=0)
    obs_dim = int(obs["observation_vector"].shape[0])
    action_dim = int(config.get("policy", {}).get("action_dim", 3))
    device = config.get("ppo", {}).get("device", "cpu")

    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=config, device=device)
    agent.load(ckpt_path)

    scenarios = ["disadvantage", "disadvantage_easy"]
    summary = {}
    for scen_name in scenarios:
        scenario = config["scenarios"][scen_name]
        records = []
        for seed in range(3):
            result, trajectory = evaluate_single_episode(
                env=env,
                agent=agent,
                config=config,
                scenario=scenario,
                seed=seed,
                save_trajectory=True,
                method_name="focused_v2",
            )
            records.append(result)
            traj_path = out_dir / f"{scen_name}_seed{seed}.csv"
            if trajectory:
                with open(traj_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=trajectory[0].keys())
                    writer.writeheader()
                    writer.writerows(trajectory)
                print(f"Saved {traj_path}")
            print(
                f"{scen_name} seed={seed}: success={result['is_success']} "
                f"reason={result['reason']} min_range={result['min_range_m']:.0f} "
                f"final_range={result['final_range_m']:.0f} final_ata={result.get('final_ata_deg', np.nan):.1f}"
            )
        summary[scen_name] = {
            "success_rate": sum(r["is_success"] for r in records) / len(records),
            "mean_min_range_m": float(np.mean([r["min_range_m"] for r in records])),
            "mean_final_range_m": float(np.mean([r["final_range_m"] for r in records])),
        }

    env.close()
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
