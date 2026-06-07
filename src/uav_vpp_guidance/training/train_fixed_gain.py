"""
Training script: Virtual Pursuit Point policy under fixed guidance gains.

Usage:
    python -m uav_vpp_guidance.training.train_fixed_gain \
        --config config/experiment/fixed_gain_vpp.yaml

TODO: Implement training loop by migrating from legacy runner:
  <JSBSIM_ROOT>/runner/jsbsim_runner.py
"""

import argparse


def main():
    """
    Train virtual pursuit point policy under fixed guidance gains.
    """
    parser = argparse.ArgumentParser(
        description="Train VPP policy with fixed guidance gains."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment configuration YAML file.",
    )
    args = parser.parse_args()

    # TODO:
    # 1. Load and merge YAML configs.
    # 2. Set random seeds.
    # 3. Initialize JSBSimEnv, CloseRangeTrackingEnv, PPOAgent.
    # 4. Fix guidance gains from config (no optimization).
    # 5. Run PPO training loop.
    # 6. Save checkpoints and config snapshot.
    raise NotImplementedError


if __name__ == "__main__":
    main()
