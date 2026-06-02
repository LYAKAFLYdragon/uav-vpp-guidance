"""
Training script: Freeze a trained policy and optimize guidance gains only.

Usage:
    python -m uav_vpp_guidance.training.train_gain_only \
        --config config/experiment/gain_only_cem.yaml

TODO: Implement gain-only optimization loop.
"""

import argparse


def main():
    """
    Freeze a trained policy and optimize guidance gains only.
    """
    parser = argparse.ArgumentParser(
        description="Optimize guidance gains with frozen policy."
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
    # 2. Load frozen policy checkpoint.
    # 3. Initialize CEMGainOptimizer.
    # 4. Evaluate candidate gains using frozen policy.
    # 5. Update gain distribution via CEM.
    # 6. Save best gains and evaluation logs.
    raise NotImplementedError


if __name__ == "__main__":
    main()
