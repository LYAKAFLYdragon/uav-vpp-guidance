"""
Training script: Virtual Pursuit Point policy under fixed guidance gains.

Usage:
    # Dry-run (verify config)
    python -m uav_vpp_guidance.training.train_fixed_gain \
        --config config/experiment/fixed_gain_vpp.yaml --dry-run

    # Smoke test (fast)
    python -m uav_vpp_guidance.training.train_fixed_gain \
        --config config/experiment/fixed_gain_vpp.yaml --smoke

    # Full training
    python -m uav_vpp_guidance.training.train_fixed_gain \
        --config config/experiment/fixed_gain_vpp.yaml
"""

import argparse
import os
import sys

# Allow importing train_ppo from the sibling module
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from train_no_prediction_vpp_ppo import load_experiment_config, train_ppo
from uav_vpp_guidance.utils.seed import set_seed


def main():
    """Train VPP policy with fixed guidance gains (no optimization)."""
    parser = argparse.ArgumentParser(
        description="Train VPP policy with fixed guidance gains."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment configuration YAML file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed override.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory override.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a minimal smoke test (reduced timesteps).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse config, print resolved settings, and exit.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["simple", "jsbsim"],
        help="Override simulation backend (default: from config).",
    )
    parser.add_argument(
        "--use-jsbsim",
        action="store_true",
        help="Force use_jsbsim=True (equivalent to --backend jsbsim).",
    )
    args = parser.parse_args()

    # Load and merge configuration
    config = load_experiment_config(args.config)

    # Backend override
    backend = args.backend
    if args.use_jsbsim:
        backend = "jsbsim"
    if backend is not None:
        config["backend"] = backend
        if "env" not in config:
            config["env"] = {}
        config["env"]["backend"] = backend
        config["env"]["use_jsbsim"] = (backend == "jsbsim")
        print(f"Backend override: {backend}")

    seed = args.seed if args.seed is not None else config.get("experiment", {}).get("seed", 0)
    set_seed(seed)

    exp_name = config.get("experiment", {}).get("name", "fixed_gain_vpp")
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(
            config.get("experiment", {}).get("output_root", "outputs"),
            "experiments",
            exp_name,
        )
    os.makedirs(output_dir, exist_ok=True)

    # Resolve fixed gains from config
    guidance_cfg = config.get("guidance", {})
    gains = guidance_cfg.get("gains", {})
    if not gains:
        print("WARNING: No fixed gains found in config['guidance']['gains']. Using defaults.")

    # Ensure trajectory prediction is disabled for this baseline
    tp_enabled = config.get("trajectory_prediction", {}).get("enabled", False)
    if tp_enabled:
        print("WARNING: trajectory_prediction.enabled is True! Forcing to False for fixed-gain baseline.")
        config["trajectory_prediction"]["enabled"] = False

    print(f"Experiment: {exp_name}")
    print(f"Output dir: {output_dir}")
    print(f"Seed: {seed}")
    print(f"Fixed gains: {gains}")
    print(f"Trajectory prediction: {'enabled' if tp_enabled else 'disabled'}")

    if args.dry_run:
        print("\n[DRY-RUN] Configuration resolved successfully.")
        print(f"  total_timesteps: {config.get('ppo', {}).get('total_timesteps', 200000)}")
        print(f"  rollout_steps: {config.get('ppo', {}).get('rollout_steps', 2048)}")
        print(f"  action_dim: {config.get('policy', {}).get('action_dim', 3)}")
        print(f"  gains: {gains}")
        print("Exiting without training.")
        return

    train_ppo(config, output_dir, smoke=args.smoke)


if __name__ == "__main__":
    main()
