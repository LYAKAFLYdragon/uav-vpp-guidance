#!/usr/bin/env python3
"""Evaluate focused v5 checkpoint on the missing disadvantage_easy scenario."""
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
    cfg_path = "config/experiment/maneuver_target_vpp_pilot_disadvantage_focused_v5.yaml"
    checkpoint = "outputs/experiments/maneuver_target_vpp_pilot_disadvantage_focused_v5_s0/checkpoints/best.pt"
    config = load_experiment_config(cfg_path)

    scenario = {
        "name": "disadvantage_easy",
        "description": "Easier variant: ego has speed advantage, target is closer and slower.",
        "own_init": {
            "position_m": [0.0, 0.0, 5000.0],
            "velocity_mps": 220.0,
            "heading_deg": 0.0,
        },
        "target_init": {
            "position_m": [-400.0, 300.0, 5000.0],
            "velocity_mps": 200.0,
            "heading_deg": 20.0,
        },
    }

    env = CloseRangeTrackingEnv(config)
    obs = env.reset(seed=0)
    obs_dim = int(obs["observation_vector"].shape[0])
    action_dim = int(config.get("policy", {}).get("action_dim", 3))
    device = config.get("ppo", {}).get("device", "cpu")
    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=config, device=device)
    agent.load(checkpoint)

    records = []
    for seed in [0, 1, 2]:
        for ep in range(20):
            result, _ = evaluate_single_episode(
                env=env, agent=agent, config=config,
                scenario=scenario, seed=seed * 10000 + ep,
                save_trajectory=False, method_name="focused_v5",
            )
            records.append(result)

    env.close()
    n = len(records)
    successes = sum(1 for r in records if r.get("is_success"))
    crashes = sum(1 for r in records if r.get("is_crash"))
    timeouts = sum(1 for r in records if r.get("reason") == "timeout")
    returns = [r.get("return", np.nan) for r in records]
    min_ranges = [r.get("min_range_m", np.nan) for r in records]
    final_ranges = [r.get("final_range_m", np.nan) for r in records]
    print(f"disadvantage_easy (v5 checkpoint, not trained on it): n={n}")
    print(f"  success={successes/n:.1%} crash={crashes/n:.1%} timeout={timeouts/n:.1%}")
    print(f"  mean_return={np.nanmean(returns):.1f}  mean_min_range={np.nanmean(min_ranges):.0f}  mean_final_range={np.nanmean(final_ranges):.0f}")


if __name__ == "__main__":
    main()
