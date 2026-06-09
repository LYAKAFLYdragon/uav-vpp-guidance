"""
End-to-End PPO Agent.

Directly outputs normalized control commands in [-1, 1] for each actuator
channel [nz_cmd, roll_rate_cmd, throttle_cmd], without virtual pursuit point
or guidance law.  The environment maps these normalized commands to physical
limits via linear scaling.

This agent is a thin wrapper over :class:`PPOAgent` that fixes
``action_dim = 3`` and provides a convenience clip method aligned with the
policy's configured action bounds.
"""

import numpy as np

from .ppo_agent import PPOAgent


class EndToEndPPOAgent(PPOAgent):
    """
    PPO agent for end-to-end control.

    Inherits all training logic from :class:`PPOAgent`. The key difference is
    that ``action_dim`` is fixed to 3 (nz, roll_rate, throttle).  The network
    outputs normalized commands in [-1, 1] (controlled by
    ``config['policy']['action_low/high']``), which the environment then maps
    to physical actuator limits.

    Args:
        obs_dim (int): Observation dimension.
        action_dim (int): Action dimension (should be 3 for end-to-end control).
        config (dict): Full experiment configuration.
        device (str): Torch device (``cpu`` or ``cuda``).
    """

    def __init__(self, obs_dim, action_dim, config, device="cpu"):
        super().__init__(obs_dim, action_dim, config, device)

        policy_cfg = config.get("policy", {})
        self.action_low_np = np.array(
            policy_cfg.get("action_low", [-1.0] * self.action_dim), dtype=np.float32
        )
        self.action_high_np = np.array(
            policy_cfg.get("action_high", [1.0] * self.action_dim), dtype=np.float32
        )

    def clip_action(self, action: np.ndarray) -> np.ndarray:
        """
        Hard-clip action to the configured policy bounds.

        For the end-to-end baseline this is typically [-1, 1]; the environment
        then scales to physical limits.  This guard catches numerical issues
        or external perturbations.

        Args:
            action (np.ndarray): Action vector, shape ``(action_dim,)``.

        Returns:
            np.ndarray: Clipped action vector.
        """
        return np.clip(action, self.action_low_np, self.action_high_np)
