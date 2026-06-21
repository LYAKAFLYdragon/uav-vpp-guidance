#!/usr/bin/env python3
"""Test fixed target-frame VPP actions under v6 config on disadvantage scenarios."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

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


class FixedAgent:
    def __init__(self, action):
        self.action = np.asarray(action, dtype=np.float64)

    def get_deterministic_action(self, obs):
        return self.action.copy()


def main():
    cfg_path = "config/experiment/maneuver_target_vpp_pilot_disadvantage_focused_v6.yaml"
    config = load_experiment_config(cfg_path)

    scenarios = {
        "disadvantage": config["scenarios"]["disadvantage"],
        "disadvantage_easy": config["scenarios"]["disadvantage_easy"],
    }

    for action_name, action in [
        ("lead_left", [1.0, -1.0, 0.0]),
        ("lead", [1.0, 0.0, 0.0]),
        ("left", [0.0, -1.0, 0.0]),
        ("lead_right", [1.0, 1.0, 0.0]),
    ]:
        print(f"\n=== Action: {action_name} {action} ===")
        for scen_name, scenario in scenarios.items():
            env = CloseRangeTrackingEnv(config)
            agent = FixedAgent(action)
            records = []
            for seed in range(20):
                result, _ = evaluate_single_episode(
                    env=env, agent=agent, config=config,
                    scenario=scenario, seed=seed,
                    save_trajectory=False, method_name="fixed_tv",
                )
                records.append(result)
            env.close()
            n = len(records)
            successes = sum(1 for r in records if r.get("is_success"))
            crashes = sum(1 for r in records if r.get("is_crash"))
            timeouts = sum(1 for r in records if r.get("reason") == "timeout")
            final_ranges = [r.get("final_range_m", np.nan) for r in records]
            min_ranges = [r.get("min_range_m", np.nan) for r in records]
            print(f"  {scen_name}: success={successes/n:.1%} crash={crashes/n:.1%} timeout={timeouts/n:.1%} min_range={np.nanmean(min_ranges):.0f} final_range={np.nanmean(final_ranges):.0f}")


if __name__ == "__main__":
    main()
