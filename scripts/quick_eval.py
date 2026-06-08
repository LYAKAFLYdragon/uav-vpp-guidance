#!/usr/bin/env python3
"""Quick evaluation: single method, print summary only."""

import argparse
import sys
from pathlib import Path

import yaml
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uav_vpp_guidance.envs.scenario_registry import initialize_canonical_scenarios
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.evaluation.evaluate_prediction_comparison import evaluate_single_episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--gains", type=str, default=None, help="JSON gains file (for bilevel)")
    args = parser.parse_args()

    initialize_canonical_scenarios()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    includes = config.pop("includes", [])
    merged = {}
    for inc_path in includes:
        inc_full = Path(args.config).parent / inc_path
        if inc_full.exists():
            with open(inc_full, "r", encoding="utf-8") as fi:
                merged = {**merged, **yaml.safe_load(fi)}
    config = {**merged, **config}

    if args.gains:
        import json
        with open(args.gains, "r") as f:
            gains_data = json.load(f)
        best_gains = gains_data.get("best_gains", gains_data.get("gains", {}))
        if "guidance" not in config:
            config["guidance"] = {}
        if "gains" not in config["guidance"]:
            config["guidance"]["gains"] = {}
        config["guidance"]["gains"].update(best_gains)

    env = CloseRangeTrackingEnv(config)
    obs = env.reset(seed=0)
    obs_dim = int(obs["observation_vector"].shape[0])

    agent = PPOAgent(obs_dim=obs_dim, action_dim=3, config=config, device="cpu")
    agent.load(args.checkpoint)

    scenarios = list(config.get("scenarios", {}).values())
    if not scenarios:
        from uav_vpp_guidance.envs.scenario_registry import ScenarioRegistry
        scenarios = ScenarioRegistry.get_regression_suite()[:4]

    results = []
    for scen in scenarios:
        for seed in args.seeds:
            result, _ = evaluate_single_episode(
                env=env, agent=agent, config=config,
                scenario=scen, seed=seed,
                save_trajectory=False, method_name="eval",
            )
            results.append(result)

    env.close()

    returns = [r["return"] for r in results]
    successes = [r["is_success"] for r in results]
    crashes = [r["is_crash"] for r in results]

    print(f"N={len(results)}")
    print(f"Success rate: {sum(successes)/len(successes):.1%}")
    print(f"Crash rate: {sum(crashes)/len(crashes):.1%}")
    print(f"Mean return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")


if __name__ == "__main__":
    main()
