#!/usr/bin/env python3
"""Evaluate a single trained PPO checkpoint per scenario and write a JSON summary."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.utils.config import load_yaml_config, merge_config


def load_config(config_path: str) -> dict:
    """Recursively load included YAML configs."""
    base = load_yaml_config(config_path)
    includes = base.pop("includes", [])
    merged = {}
    for inc in includes:
        inc_full = Path(config_path).parent / inc
        if inc_full.exists():
            merged = merge_config(merged, load_config(str(inc_full)))
    return merge_config(merged, base)


def evaluate(config: dict, checkpoint: str, episodes: int, seed_base: int):
    env = CloseRangeTrackingEnv(config)
    env.set_domain_rand_scale(0.0)
    obs = env.reset(seed=0)
    obs_dim = int(obs["observation_vector"].shape[0])
    action_dim = int(config.get("policy", {}).get("action_dim", 3))
    device = config.get("ppo", {}).get("device", "cpu")

    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=config, device=device)
    agent.load(checkpoint)

    scenarios = config.get("scenarios", {})
    per_scenario = {}
    overall = {"success": [], "crash": [], "timeout": [], "out_of_bounds": [], "returns": []}

    for scenario_name, scenario in scenarios.items():
        metrics = {
            "success": [], "crash": [], "timeout": [], "out_of_bounds": [],
            "returns": [], "lengths": [], "final_range": [], "final_ata": [],
            "min_range": [],
        }
        for ep in range(episodes):
            seed = ep + seed_base + 1000 * (list(scenarios.keys()).index(scenario_name) + 1)
            obs = env.reset(scenario=scenario, seed=seed)
            ep_return = 0.0
            ep_length = 0
            min_range = float("inf")
            final_range = 0.0
            final_ata = 0.0
            reason = "timeout"

            while True:
                obs_vec = obs["observation_vector"]
                action = agent.get_deterministic_action(obs_vec)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_return += reward
                ep_length += 1
                range_m = info.get("range_m", float("nan"))
                ata_deg = info.get("ata_deg", float("nan"))
                if np.isfinite(range_m):
                    min_range = min(min_range, float(range_m))
                    final_range = float(range_m)
                if np.isfinite(ata_deg):
                    final_ata = float(ata_deg)
                if terminated or truncated:
                    reason = info.get("reason", "unknown")
                    break

            metrics["returns"].append(ep_return)
            metrics["lengths"].append(ep_length)
            metrics["success"].append(1.0 if reason == "success" else 0.0)
            metrics["crash"].append(1.0 if reason == "crash" else 0.0)
            metrics["timeout"].append(1.0 if reason == "timeout" else 0.0)
            metrics["out_of_bounds"].append(1.0 if reason == "out_of_bounds" else 0.0)
            metrics["final_range"].append(final_range)
            metrics["final_ata"].append(final_ata)
            metrics["min_range"].append(min_range)

        per_scenario[scenario_name] = {
            "success_rate": float(np.mean(metrics["success"])),
            "crash_rate": float(np.mean(metrics["crash"])),
            "timeout_rate": float(np.mean(metrics["timeout"])),
            "out_of_bounds_rate": float(np.mean(metrics["out_of_bounds"])),
            "mean_return": float(np.mean(metrics["returns"])),
            "std_return": float(np.std(metrics["returns"])),
            "mean_length": float(np.mean(metrics["lengths"])),
            "mean_final_range_m": float(np.nanmean(metrics["final_range"])),
            "mean_final_ata_deg": float(np.nanmean(np.abs(metrics["final_ata"]))),
            "mean_min_range_m": float(np.nanmean(metrics["min_range"])),
        }
        for k in ["success", "crash", "timeout", "out_of_bounds", "returns"]:
            overall[k].extend(metrics[k])

    env.close()
    overall_metrics = {
        "success_rate": float(np.mean(overall["success"])),
        "crash_rate": float(np.mean(overall["crash"])),
        "timeout_rate": float(np.mean(overall["timeout"])),
        "out_of_bounds_rate": float(np.mean(overall["out_of_bounds"])),
        "mean_return": float(np.mean(overall["returns"])),
        "std_return": float(np.std(overall["returns"])),
    }
    return {"overall": overall_metrics, "per_scenario": per_scenario}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--output", default="eval_summary.json")
    args = parser.parse_args()

    config = load_config(args.config)
    config.setdefault("experiment", {})["seed"] = 0
    results = evaluate(config, args.checkpoint, args.episodes, args.seed_base)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
