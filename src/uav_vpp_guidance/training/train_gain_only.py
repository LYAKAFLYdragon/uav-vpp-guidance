"""Training script: Freeze a trained policy and optimize guidance gains only.

Usage:
    python -m uav_vpp_guidance.training.train_gain_only \
        --config config/experiment/gain_only_cem.yaml

Supports --dry-run for smoke testing configuration and initialization.
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.envs.scenario_registry import (
    ScenarioRegistry,
    initialize_canonical_scenarios,
)
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.evaluation.evaluate_prediction_comparison import (
    evaluate_single_episode,
)
from uav_vpp_guidance.gain_optimizer.cem import CEMGainOptimizer
from uav_vpp_guidance.gain_optimizer.gain_space import GainSpace
from uav_vpp_guidance.guidance.gain_config import GuidanceGains


def _resolve_config(config_path: str, allow_missing_includes: bool = False) -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    for section in ("env", "guidance", "gain_optimizer", "ppo"):
        if section not in config:
            continue
        include_key = f"{section}_config_file" if section == "env" else "config_file"
        include_file = config[section].get(include_key) if isinstance(config[section], dict) else None
        if include_file is None:
            include_file = config.get(f"{section}_config_file")

        if include_file:
            include_path = config_path.parent.parent.parent / include_file
            if not include_path.exists():
                msg = f"Include file not found: {include_path} (referenced from {config_path})"
                if allow_missing_includes:
                    print(f"WARNING: {msg}")
                    continue
                raise FileNotFoundError(msg)
            included = yaml.safe_load(include_path.read_text(encoding="utf-8"))
            if isinstance(config.get(section), dict) and isinstance(included, dict):
                merged = copy.deepcopy(included)
                merged.update(config[section])
                config[section] = merged
            elif isinstance(included, dict):
                config[section] = copy.deepcopy(included)

    return config


def _build_gain_space(config: dict) -> GainSpace:
    """Build GainSpace from config bounds.

    Prefers ``guidance.gain_space``, falls back to top-level ``gain_space``,
    and finally to the canonical 5-D search space documented in
    config/canonical/gain_space.yaml.
    """
    gain_bounds = config.get("guidance", {}).get("gain_space")
    if gain_bounds is None:
        gain_bounds = config.get("gain_space")
    if gain_bounds is None:
        gain_bounds = {
            "k_los": [0.5, 4.0],
            "k_pos": [0.1, 2.0],
            "k_damp": [0.2, 3.0],
            "k_roll": [0.5, 2.0],
            "k_speed": [0.1, 1.0],
        }
    return GainSpace(gain_bounds)


def _build_guidance_gains(gains_dict: dict) -> GuidanceGains:
    valid_fields = set(GuidanceGains.__dataclass_fields__.keys())
    filtered = {k: v for k, v in gains_dict.items() if k in valid_fields}
    return GuidanceGains(**filtered)


def _serialize(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    return obj


def main():
    parser = argparse.ArgumentParser(
        description="Optimize guidance gains with frozen policy."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment configuration YAML file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to frozen PPO checkpoint.",
    )
    parser.add_argument("--n-iter", type=int, default=50)
    parser.add_argument("--candidates", type=int, default=12)
    parser.add_argument("--elite-ratio", type=float, default=0.25)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="+",
        default=None,
        choices=["regression", "candidate", "negative"],
    )
    parser.add_argument("--output-dir", type=str, default="outputs/gain_only_cem")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config, env, policy, and optimizer initialization only.",
    )
    parser.add_argument(
        "--allow-missing-includes",
        action="store_true",
        help="Allow missing config include files (warn instead of fail).",
    )
    parser.add_argument(
        "--allow-random-init",
        action="store_true",
        help="Allow optimization from random initialization when checkpoint is missing. "
             "Results will be marked invalid_for_paper.",
    )
    args = parser.parse_args()

    print(f"[GainOnly] Loading config: {args.config}")
    config = _resolve_config(args.config, allow_missing_includes=args.allow_missing_includes)

    config["backend"] = "simple"
    if "env" not in config:
        config["env"] = {}
    config["env"]["backend"] = "simple"
    config["env"]["use_jsbsim"] = False
    if "guidance" not in config:
        config["guidance"] = {}
    if "mode_switch" not in config["guidance"]:
        config["guidance"]["mode_switch"] = {}
    config["guidance"]["mode_switch"]["enabled"] = False

    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = config.get("policy_checkpoint")
    if not checkpoint and not args.dry_run:
        print("ERROR: No checkpoint specified. Use --checkpoint or set policy_checkpoint in config.")
        sys.exit(1)

    print(f"[GainOnly] Checkpoint: {checkpoint}")
    checkpoint_exists = Path(checkpoint).exists() if checkpoint else False
    if not checkpoint_exists and not args.dry_run and not args.allow_random_init:
        print(f"ERROR: Checkpoint not found: {checkpoint}")
        print("Use --allow-random-init to proceed with random initialization.")
        sys.exit(1)
    if not checkpoint_exists:
        print(f"WARNING: Checkpoint not found: {checkpoint}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    initialize_canonical_scenarios()
    if args.scenarios is None or args.scenarios == ["regression"]:
        scenarios = ScenarioRegistry.get_regression_suite()
    elif args.scenarios == ["candidate"]:
        scenarios = ScenarioRegistry.get_candidate_suite()
    elif args.scenarios == ["negative"]:
        scenarios = ScenarioRegistry.get_negative_suite()
    else:
        scenarios = ScenarioRegistry.get_regression_suite()

    print("[GainOnly] Initializing environment...")
    env = CloseRangeTrackingEnv(config)
    obs = env.reset(seed=0)
    obs_dim = int(obs["observation_vector"].shape[0])

    print(f"[GainOnly] Observation dim: {obs_dim}")
    print("[GainOnly] Initializing policy...")
    agent = PPOAgent(obs_dim=obs_dim, action_dim=3, config=config, device="cpu")

    if checkpoint_exists:
        agent.load(checkpoint)
        print(f"[GainOnly] Loaded checkpoint: {checkpoint}")
    else:
        print("[GainOnly] Using randomly initialized policy (checkpoint missing)")

    gain_space = _build_gain_space(config)
    cem_config = {
        "candidates": args.candidates,
        "elite_ratio": args.elite_ratio,
        "noise_floor": 0.05,
        "convergence_tol": 0.001,
        "random_seed": config.get("seed", 42),
    }
    cem = CEMGainOptimizer(gain_space, cem_config)
    print(f"[GainOnly] Gain space: {gain_space.names}")

    if args.dry_run:
        print("\n" + "=" * 50)
        print("DRY RUN COMPLETE")
        print("=" * 50)
        print("Config loaded: OK")
        print("Environment initialized: OK")
        print(f"Policy initialized: OK (checkpoint_exists={checkpoint_exists})")
        print("Gain optimizer initialized: OK")
        print(f"Output dir: {args.output_dir}")
        env.close()
        return

    def evaluator(gains_dict: dict) -> float:
        gains = _build_guidance_gains(gains_dict)
        env.current_gains = gains
        successes = 0
        total = 0
        for scen in scenarios:
            for seed in args.seeds:
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

    print(f"\n[GainOnly] Starting CEM optimization: {args.n_iter} iterations")
    print(f"Scenarios: {len(scenarios)}, Seeds: {args.seeds}")
    best_gains, history = cem.optimize(evaluator, n_iter=args.n_iter)

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
    serialized = _serialize(results)
    serialized["invalid_for_paper"] = not checkpoint_exists
    serialized["checkpoint_exists"] = checkpoint_exists
    serialized["checkpoint_path"] = checkpoint
    result_path = output_dir / "cem_results.json"
    result_path.write_text(json.dumps(serialized, indent=2), encoding="utf-8")
    print(f"\nSaved results: {result_path}")

    print("\n" + "=" * 50)
    print("CEM Optimization Complete")
    print("=" * 50)
    print(f"Best gains: {best_gains}")
    print(f"Best score: {results['best_score']:.4f}")
    print(f"Iterations: {results['n_iterations']}")
    if not checkpoint_exists:
        print("⚠️ INVALID FOR PAPER: checkpoint was missing, random init used")

    env.close()


if __name__ == "__main__":
    main()
