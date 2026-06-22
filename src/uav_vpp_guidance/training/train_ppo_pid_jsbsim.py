#!/usr/bin/env python3
"""
PPO+PID Hybrid Training for JSBSim backend.

Thin wrapper around uav_vpp_guidance.training.train_ppo_pid so that the CLI
entry point matches the flight-control comparison paper workflow:

    python -m uav_vpp_guidance.training.train_ppo_pid_jsbsim \
        --config config/experiment/train_ppo_pid_jsbsim.yaml

The policy outputs a 4D action: [Δx, Δy, Δz, aggressiveness].
"""

from .train_ppo_pid import main

if __name__ == "__main__":
    main()
