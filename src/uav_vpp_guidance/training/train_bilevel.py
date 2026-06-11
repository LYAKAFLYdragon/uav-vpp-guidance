"""Training script: Strategy-gain bilevel training.

Usage:
    python -m uav_vpp_guidance.training.train_bilevel \
        --config config/experiment/proposed_bilevel.yaml

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
from uav_vpp_guidance.gain_optimizer.bilevel_trainer import BilevelTrainer
from uav_vpp_guidance.gain_optimizer.cem import CEMGainOptimizer
from uav_vpp_guidance.gain_optimizer.gain_space import GainSpace


def _resolve_config(config_path: str, allow_missing_includes: bool = False) -> dict:
    """Load config and resolve includes."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    # Resolve includes
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
    gain_bounds = config.get("guidance", {}).get("gain_space", {
        "k_los": [0.1, 3.0],
        "k_pos": [0.1, 2.0],
        "k_roll": [0.1, 3.0],
        "k_speed": [0.0, 1.0],
    })
    return GainSpace(gain_bounds)


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
        description="Bilevel strategy-gain optimization training."
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
        help="Initial PPO checkpoint path.",
    )
    parser.add_argument(
        "--n-episodes", type=int, default=100, help="Total training episodes"
    )
    parser.add_argument(
        "--outer-every", type=int, default=10, help="Policy update frequency"
    )
    parser.add_argument(
        "--inner-iter", type=int, default=20, help="CEM iterations per inner loop"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/bilevel_training"
    )
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
        help="Allow training from random initialization when checkpoint is missing. "
             "Results will be marked invalid_for_paper.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed override (sets config seed and CEM random seed).",
    )
    args = parser.parse_args()

    print(f"[Bilevel] Loading config: {args.config}")
    config = _resolve_config(args.config, allow_missing_includes=args.allow_missing_includes)

    # Apply seed override
    if args.seed is not None:
        config["seed"] = args.seed
        print(f"[Bilevel] Seed override: {args.seed}")

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

    # Resolve checkpoint
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = config.get("policy_checkpoint")
    if not checkpoint and not args.dry_run:
        print("ERROR: No checkpoint specified. Use --checkpoint or set policy_checkpoint in config.")
        sys.exit(1)

    print(f"[Bilevel] Checkpoint: {checkpoint}")
    checkpoint_exists = Path(checkpoint).exists() if checkpoint else False
    if not checkpoint_exists and not args.dry_run and not args.allow_random_init:
        print(f"ERROR: Checkpoint not found: {checkpoint}")
        print("Use --allow-random-init to proceed with random initialization.")
        sys.exit(1)
    if not checkpoint_exists:
        print(f"WARNING: Checkpoint not found: {checkpoint}")

    initialize_canonical_scenarios()

    def env_factory():
        return CloseRangeTrackingEnv(copy.deepcopy(config))

    print("[Bilevel] Initializing environment...")
    env = env_factory()
    obs = env.reset(seed=0)
    obs_dim = int(obs["observation_vector"].shape[0])
    env.close()

    print(f"[Bilevel] Observation dim: {obs_dim}")
    print("[Bilevel] Initializing policy...")
    policy = PPOAgent(obs_dim=obs_dim, action_dim=3, config=config, device="cpu")

    if checkpoint_exists:
        policy.load(checkpoint)
        print(f"[Bilevel] Loaded checkpoint: {checkpoint}")
    else:
        print("[Bilevel] Using randomly initialized policy (checkpoint missing)")

    gain_space = _build_gain_space(config)
    cem_config = {
        "candidates": config.get("bilevel", {}).get("candidates", 12),
        "elite_ratio": config.get("bilevel", {}).get("elite_ratio", 0.25),
        "noise_floor": 0.05,
        "convergence_tol": 0.001,
        "random_seed": config.get("seed", 42),
    }
    cem = CEMGainOptimizer(gain_space, cem_config)
    print(f"[Bilevel] Gain space: {gain_space.names}")

    if args.dry_run:
        print("\n" + "=" * 50)
        print("DRY RUN COMPLETE")
        print("=" * 50)
        print("Config loaded: OK")
        print("Environment initialized: OK")
        print(f"Policy initialized: OK (checkpoint_exists={checkpoint_exists})")
        print("Gain optimizer initialized: OK")
        print(f"Output dir: {args.output_dir}")
        return

    bilevel_config = {
        "outer_every": args.outer_every,
        "inner_iter": args.inner_iter,
        "n_episodes": args.n_episodes,
        "eval_seeds": [0, 1, 2],
        "eval_scenarios": ScenarioRegistry.get_regression_suite(),
        "checkpoint_dir": args.output_dir,
    }

    trainer = BilevelTrainer(env_factory, policy, cem, bilevel_config)
    results = trainer.train()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "bilevel_results.json"
    serialized = _serialize(results)
    serialized["invalid_for_paper"] = not checkpoint_exists
    serialized["checkpoint_exists"] = checkpoint_exists
    serialized["checkpoint_path"] = checkpoint
    result_path.write_text(
        json.dumps(serialized, indent=2), encoding="utf-8"
    )
    print(f"\nSaved results: {result_path}")
    print(f"Best eval SR: {results['best_success_rate']:.2%}")
    print(f"Best episode: {results['best_policy_episode']}")
    print(f"Best gains: {results['best_gains']}")
    if not checkpoint_exists:
        print("⚠️ INVALID FOR PAPER: checkpoint was missing, random init used")


if __name__ == "__main__":
    main()
