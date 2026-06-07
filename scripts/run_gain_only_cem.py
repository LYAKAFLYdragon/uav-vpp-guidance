#!/usr/bin/env python3
"""Stage 7A: Gain-only CEM optimization.

Freeze PPO policy, use CEM to optimize guidance gains.
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.envs.scenario_registry import ScenarioRegistry, initialize_canonical_scenarios
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.evaluation.evaluate_prediction_comparison import evaluate_single_episode
from uav_vpp_guidance.gain_optimizer.cem import CEMGainOptimizer
from uav_vpp_guidance.gain_optimizer.gain_space import GainSpace
from uav_vpp_guidance.guidance.gain_config import GuidanceGains


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


def _build_guidance_gains(gains_dict: dict) -> GuidanceGains:
    """Filter gains_dict to only include valid GuidanceGains fields."""
    valid_fields = set(GuidanceGains.__dataclass_fields__.keys())
    filtered = {k: v for k, v in gains_dict.items() if k in valid_fields}
    return GuidanceGains(**filtered)


def make_evaluator(env: CloseRangeTrackingEnv, agent: PPOAgent, scenarios: list, seeds: tuple):
    """Create an evaluator function for CEM."""
    def evaluator(gains_dict: dict) -> float:
        # Set gains on env (filtered to valid dataclass fields)
        gains = _build_guidance_gains(gains_dict)
        env.current_gains = gains

        successes = 0
        total = 0
        for scen in scenarios:
            for seed in seeds:
                result, _ = evaluate_single_episode(
                    env=env,
                    agent=agent,
                    config=env.config,
                    scenario=scen,
                    seed=seed,
                    save_trajectory=False,
                    method_name="gain_only_cem",
                )
                if result.get("is_success", False):
                    successes += 1
                total += 1
        return successes / total if total > 0 else 0.0

    return evaluator


def main():
    parser = argparse.ArgumentParser(description="Gain-only CEM optimization")
    parser.add_argument("--config", type=str, default="config/experiment/stage6f5_feasible_geometry.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to frozen PPO checkpoint")
    parser.add_argument("--n-iter", type=int, default=50)
    parser.add_argument("--candidates", type=int, default=12)
    parser.add_argument("--elite-ratio", type=float, default=0.25)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--scenarios", type=str, nargs="+", default=None,
                        choices=["regression", "candidate", "negative"])
    parser.add_argument("--output-dir", type=str, default="outputs/gain_only_cem")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config = load_config(args.config)
    config["backend"] = "simple"
    config["env"]["backend"] = "simple"
    config["env"]["use_jsbsim"] = False
    # Disable mode-switch for pure gain eval
    if "mode_switch" not in config.get("guidance", {}):
        config["guidance"]["mode_switch"] = {}
    config["guidance"]["mode_switch"]["enabled"] = False

    # Initialize scenarios
    initialize_canonical_scenarios()
    if args.scenarios is None or args.scenarios == ["regression"]:
        scenarios = ScenarioRegistry.get_regression_suite()
    elif args.scenarios == ["candidate"]:
        scenarios = ScenarioRegistry.get_candidate_suite()
    elif args.scenarios == ["negative"]:
        scenarios = ScenarioRegistry.get_negative_suite()
    else:
        scenarios = ScenarioRegistry.get_regression_suite()

    # Create env and load frozen policy
    env = CloseRangeTrackingEnv(config)
    obs = env.reset(seed=0)
    obs_dim = int(obs["observation_vector"].shape[0])
    agent = PPOAgent(obs_dim=obs_dim, action_dim=3, config=config, device="cpu")
    agent.load(args.checkpoint)

    # Build gain space and CEM
    gain_space = build_gain_space(config)
    cem_config = {
        "candidates": args.candidates,
        "elite_ratio": args.elite_ratio,
        "noise_floor": 0.05,
        "convergence_tol": 0.001,
    }
    cem = CEMGainOptimizer(gain_space, cem_config)

    # Create evaluator
    evaluator = make_evaluator(env, agent, scenarios, tuple(args.seeds))

    # Run optimization
    print(f"Starting CEM optimization: {args.n_iter} iterations")
    print(f"Gain space: {gain_space.names}")
    print(f"Bounds low: {gain_space.low}")
    print(f"Bounds high: {gain_space.high}")
    print(f"Scenarios: {len(scenarios)}, Seeds: {args.seeds}")
    print()

    best_gains, history = cem.optimize(evaluator, n_iter=args.n_iter)

    # Save results
    results = {
        "best_gains": best_gains,
        "best_score": history[-1]["best_score"] if history else None,
        "n_iterations": len(history),
        "final_mean": history[-1]["mean"].tolist() if history else None,
        "final_std": history[-1]["std"].tolist() if history else None,
        "history": [
            {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in h.items()}
            for h in history
        ],
    }
    result_path = output_dir / "cem_results.json"
    result_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved results: {result_path}")

    # Print summary
    print("\n" + "=" * 50)
    print("CEM Optimization Complete")
    print("=" * 50)
    print(f"Best gains: {best_gains}")
    print(f"Best score: {results['best_score']:.4f}")
    print(f"Iterations: {results['n_iterations']}")
    print("Score progression: ", end="")
    for h in history:
        print(f"{h['best_score']:.3f}", end=" ")
    print()

    env.close()


if __name__ == "__main__":
    main()
