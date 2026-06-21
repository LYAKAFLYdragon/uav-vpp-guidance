#!/usr/bin/env python3
"""Diagnose a single feasibility-gradient scenario by running one deterministic episode."""
from __future__ import annotations

import json
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


def load_experiment_config(config_path: Path) -> dict:
    base_config = load_yaml_config(str(config_path))
    includes = base_config.pop("includes", [])
    merged = {}
    for inc_path in includes:
        inc_full = Path(config_path).parent / inc_path
        if inc_full.exists():
            merged = merge_config(merged, load_yaml_config(str(inc_full)))
    return merge_config(merged, base_config)


def main():
    scenario_name = sys.argv[1] if len(sys.argv) > 1 else "disadvantage_easy_speed_advantage"
    checkpoint = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    cfg_path = Path("outputs/feasibility_gradient/configs/train_disadvantage_easy_speed_advantage.yaml")
    if not cfg_path.exists():
        base_abs = (Path.cwd() / "config/experiment/maneuver_target_pilot_base.yaml").as_posix()
        bridge_abs = (Path.cwd() / "config/experiment/feasibility_gradient_scenarios.yaml").as_posix()
        cfg_path = Path("outputs/feasibility_gradient/configs/diagnose.yaml")
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(f"""includes:
  - {base_abs}
  - {bridge_abs}

curriculum:
  stage_gate_sr: 0.50
  stages:
    - progress_end: 1.0
      scenario_names:
        - {scenario_name}
""", encoding="utf-8")

    config = load_experiment_config(cfg_path)
    env = CloseRangeTrackingEnv(config)
    obs = env.reset(seed=0)
    obs_dim = int(obs["observation_vector"].shape[0])
    action_dim = int(config.get("policy", {}).get("action_dim", 3))
    device = config.get("ppo", {}).get("device", "cuda")

    if checkpoint is not None and checkpoint.exists():
        agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=config, device=device)
        agent.load(checkpoint)
        use_agent = True
    else:
        use_agent = False
        agent = None

    scenario = config["scenarios"][scenario_name]
    obs = env.reset(scenario=scenario, seed=0)

    print(f"Scenario: {scenario_name}")
    print(f"Initial obs keys: {obs.keys()}")
    print(f"Initial obs vector: {obs['observation_vector'][:10]}...")

    for step in range(env.max_steps):
        obs_vec = obs["observation_vector"]
        if use_agent:
            action = agent.get_deterministic_action(obs_vec)
        else:
            action = np.zeros(action_dim)
        obs, reward, terminated, truncated, info = env.step(action)
        rel = obs.get("relative_state", {})
        print(
            f"step={step:3d} alt={info.get('altitude_m', np.nan):8.1f} "
            f"range={rel.get('range_m', np.nan):10.1f} "
            f"ata={np.rad2deg(rel.get('ata_rad', np.nan)):7.2f} "
            f"reward={reward if reward is not None else np.nan:8.2f} reason={(info.get('reason', '-') or '-')[:10]:10s}"
        )
        if terminated or truncated:
            print(f"Terminated at step {step}: reason={info.get('reason')}")
            break

    env.close()


if __name__ == "__main__":
    main()
