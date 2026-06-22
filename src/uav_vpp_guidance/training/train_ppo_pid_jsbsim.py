"""
PPO+PID JSBSim training entry point.

Thin wrapper around :mod:`uav_vpp_guidance.training.train_ppo_pid` so the
standard JSBSim smoke-test command works:

    python -m uav_vpp_guidance.training.train_ppo_pid_jsbsim \
        --config config/experiment/train_ppo_pid_jsbsim.yaml --smoke
"""

from .train_ppo_pid import main

if __name__ == "__main__":
    main()
