"""
Training script: Strategy-gain bilevel training.

Usage:
    python -m uav_vpp_guidance.training.train_bilevel \
        --config config/experiment/proposed_bilevel.yaml

TODO: Implement bilevel training loop.
"""

import argparse


def main():
    """
    Run strategy-gain bilevel training.
    """
    parser = argparse.ArgumentParser(
        description="Bilevel strategy-gain optimization training."
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
    # 2. Initialize environment, policy, and gain optimizer.
    # 3. Run BilevelTrainer.train() alternating loop.
    # 4. Save intermediate and final results.
    raise NotImplementedError


if __name__ == "__main__":
    main()
