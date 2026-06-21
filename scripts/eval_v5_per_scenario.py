#!/usr/bin/env python3
"""Per-scenario evaluation of the focused v5 checkpoint."""
from __future__ import annotations

import json
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


def summarize(records: list) -> dict:
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
        "success_rate": successes / n,
        "crash_rate": crashes / n,
        "out_of_bounds_rate": oobs / n,
        "timeout_rate": timeouts / n,
        "mean_return": float(np.nanmean(returns)),
        "std_return": float(np.nanstd(returns)),
        "mean_min_range_m": float(np.nanmean(min_ranges)),
        "mean_final_range_m": float(np.nanmean(final_ranges)),
    }


def main():
    cfg_path = "config/experiment/maneuver_target_vpp_pilot_disadvantage_focused_v5.yaml"
    checkpoint = "outputs/experiments/maneuver_target_vpp_pilot_disadvantage_focused_v5_s0/checkpoints/best.pt"
    episodes = 20
    seeds = [0, 1, 2]

    config = load_experiment_config(cfg_path)
    env = CloseRangeTrackingEnv(config)
    obs = env.reset(seed=0)
    obs_dim = int(obs["observation_vector"].shape[0])
    action_dim = int(config.get("policy", {}).get("action_dim", 3))
    device = config.get("ppo", {}).get("device", "cpu")

    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=config, device=device)
    agent.load(checkpoint)

    scenarios = config.get("scenarios", {})
    per_scenario = {}
    all_records = []

    for scen_name, scenario in scenarios.items():
        records = []
        for seed in seeds:
            for ep in range(episodes):
                result, _ = evaluate_single_episode(
                    env=env,
                    agent=agent,
                    config=config,
                    scenario=scenario,
                    seed=seed * 10000 + ep,
                    save_trajectory=False,
                    method_name="focused_v5",
                )
                records.append(result)
        per_scenario[scen_name] = summarize(records)
        all_records.extend(records)

    env.close()
    overall = summarize(all_records)

    print(f"\nFocused v5 per-scenario ({episodes} eps x {len(seeds)} seeds):")
    print(f"{'Scenario':<20} {'N':>4} {'Success%':>10} {'Crash%':>8} {'OOB%':>8} {'Timeout%':>10} {'MeanReturn':>12} {'MeanMinRng':>12} {'MeanFinRng':>12}")
    print("-" * 110)
    for scen, s in per_scenario.items():
        print(
            f"{scen:<20} {s['n']:>4} "
            f"{s['success_rate']*100:>9.1f}% {s['crash_rate']*100:>7.1f}% "
            f"{s['out_of_bounds_rate']*100:>7.1f}% {s['timeout_rate']*100:>9.1f}% "
            f"{s['mean_return']:>11.1f} {s['mean_min_range_m']:>11.0f} {s['mean_final_range_m']:>11.0f}"
        )
    print("-" * 110)
    print(
        f"{'OVERALL':<20} {overall['n']:>4} "
        f"{overall['success_rate']*100:>9.1f}% {overall['crash_rate']*100:>7.1f}% "
        f"{overall['out_of_bounds_rate']*100:>7.1f}% {overall['timeout_rate']*100:>9.1f}% "
        f"{overall['mean_return']:>11.1f} {overall['mean_min_range_m']:>11.0f} {overall['mean_final_range_m']:>11.0f}"
    )

    out = {
        "method": "VPP-Disadvantage-Focused-v5",
        "per_scenario": per_scenario,
        "overall": overall,
    }
    out_file = Path("outputs/maneuver_target_pilot/disadvantage_focused_v5_eval.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
