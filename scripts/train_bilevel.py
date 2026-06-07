#!/usr/bin/env python3
"""Stage 7B: Bilevel strategy-gain optimization."""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.envs.scenario_registry import initialize_canonical_scenarios
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.gain_optimizer.bilevel_trainer import BilevelTrainer
from uav_vpp_guidance.gain_optimizer.cem import CEMGainOptimizer
from uav_vpp_guidance.gain_optimizer.gain_space import GainSpace


def load_config(config_path: str) -> dict:
    full_config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    method_override = full_config.get("methods", {}).get("no_prediction", {})
    base_config = copy.deepcopy(full_config)
    for k, v in method_override.items():
        if isinstance(v, dict) and k in base_config and isinstance(base_config[k], dict):
            base_config[k].update(copy.deepcopy(v))
        else:
            base_config[k] = copy.deepcopy(v)
    return base_config


def build_gain_space(config: dict) -> GainSpace:
    """Build GainSpace from config bounds."""
    gain_bounds = config.get("guidance", {}).get("gain_space", {
        "k_los": [0.1, 3.0],
        "k_pos": [0.1, 2.0],
        "k_roll": [0.1, 3.0],
        "k_speed": [0.0, 1.0],
    })
    return GainSpace(gain_bounds)


def main():
    parser = argparse.ArgumentParser(description="Bilevel strategy-gain optimization")
    parser.add_argument(
        "--config",
        type=str,
        default="config/experiment/stage6f5_feasible_geometry.yaml",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Initial PPO checkpoint"
    )
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--outer-every", type=int, default=10)
    parser.add_argument("--inner-iter", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="outputs/bilevel_training")
    args = parser.parse_args()

    config = load_config(args.config)
    config["backend"] = "simple"
    config["env"]["backend"] = "simple"
    config["env"]["use_jsbsim"] = False
    if "mode_switch" not in config.get("guidance", {}):
        config["guidance"]["mode_switch"] = {}
    config["guidance"]["mode_switch"]["enabled"] = False

    initialize_canonical_scenarios()

    def env_factory():
        return CloseRangeTrackingEnv(copy.deepcopy(config))

    env = env_factory()
    obs = env.reset(seed=0)
    obs_dim = int(obs["observation_vector"].shape[0])
    policy = PPOAgent(obs_dim=obs_dim, action_dim=3, config=config, device="cpu")
    policy.load(args.checkpoint)
    env.close()

    gain_space = build_gain_space(config)
    cem_config = {
        "candidates": 12,
        "elite_ratio": 0.25,
        "noise_floor": 0.05,
        "convergence_tol": 0.001,
        "random_seed": 42,
    }
    cem = CEMGainOptimizer(gain_space, cem_config)

    from uav_vpp_guidance.envs.scenario_registry import ScenarioRegistry

    bilevel_config = {
        "outer_every": args.outer_every,
        "inner_iter": args.inner_iter,
        "n_episodes": args.n_episodes,
        "eval_seeds": [0, 1, 2],
        "eval_scenarios": ScenarioRegistry.get_regression_suite()[:2],
    }

    trainer = BilevelTrainer(env_factory, policy, cem, bilevel_config)
    results = trainer.train()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "bilevel_results.json"

    def serialize(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [serialize(v) for v in obj]
        return obj

    result_path.write_text(
        json.dumps(serialize(results), indent=2), encoding="utf-8"
    )
    print(f"\nSaved results: {result_path}")
    print(f"Best eval SR: {results['best_success_rate']:.2%}")
    print(f"Best episode: {results['best_policy_episode']}")
    print(f"Best gains: {results['best_gains']}")


if __name__ == "__main__":
    main()
