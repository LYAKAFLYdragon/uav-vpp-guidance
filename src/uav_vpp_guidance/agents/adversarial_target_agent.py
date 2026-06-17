"""
Adversarial Target Agent for UAV pursuit-evasion.

Wraps PPOAgent to provide a target (evader) RL agent with:
- Flipped-perspective observation (from target looking at pursuer)
- Direct flight-control action space [nz, roll_rate, throttle]
- Escape-oriented reward function (computed externally by the environment)

The target agent outputs normalized actions in [-1, 1]^3 that the
adversarial environment maps to physical commands via LowLevelController.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .ppo_agent import PPOAgent
from ..envs.observation import compute_relative_geometry, build_observation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Normalization constants (must match observation.py for consistency)
# ---------------------------------------------------------------------------
REF_RANGE: float = 3000.0
REF_RANGE_RATE: float = 200.0
REF_ALT: float = 10000.0
REF_SPEED: float = 400.0

# ---------------------------------------------------------------------------
# Physical action limits for target aircraft (same as pursuer limits)
# ---------------------------------------------------------------------------
DEFAULT_ACTION_LIMITS: Dict[str, Tuple[float, float]] = {
    "nz": (-2.0, 7.0),
    "roll_rate": (-1.5, 1.5),
    "throttle": (0.0, 1.0),
}


class AdversarialTargetAgent:
    """
    Target (evader) PPO agent for adversarial pursuit-evasion.

    Wraps a PPOAgent and adds:
    - Target-perspective observation building (own=target, target=pursuer)
    - Physical action mapping from normalized [-1,1]^3 to [nz, roll_rate, throttle]
    - Delegation of all RL operations to the wrapped PPOAgent

    The observation space is 16-dimensional (same base features as the pursuer
    but computed with swapped perspectives).

    The action space is 3-dimensional continuous:
        [nz_cmd_norm, roll_rate_cmd_norm, throttle_cmd_norm] in [-1, 1]^3
    """

    def __init__(
        self,
        config: Dict[str, Any],
        device: str = "cpu",
    ) -> None:
        """
        Args:
            config: Full configuration dictionary. Must contain at least
                ``ppo`` / ``policy`` sections for PPOAgent initialization.
            device: Torch device string (``"cpu"`` or ``"cuda"``).
        """
        self.config = config
        obs_dim = 16  # base observation (no temporal, no gains, no scenario)
        action_dim = 3  # [nz, roll_rate, throttle]

        # Build the underlying PPO agent
        self._ppo = PPOAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            config=config,
            device=device,
        )

        # Action limits for physical command mapping
        limits = config.get("limits", {})
        self._nz_min = float(limits.get("nz_min", DEFAULT_ACTION_LIMITS["nz"][0]))
        self._nz_max = float(limits.get("nz_max", DEFAULT_ACTION_LIMITS["nz"][1]))
        self._rr_min = float(limits.get("roll_rate_min", DEFAULT_ACTION_LIMITS["roll_rate"][0]))
        self._rr_max = float(limits.get("roll_rate_max", DEFAULT_ACTION_LIMITS["roll_rate"][1]))
        self._th_min = float(limits.get("throttle_min", DEFAULT_ACTION_LIMITS["throttle"][0]))
        self._th_max = float(limits.get("throttle_max", DEFAULT_ACTION_LIMITS["throttle"][1]))

        logger.info(
            "AdversarialTargetAgent initialized: obs_dim=%d action_dim=%d device=%s",
            obs_dim,
            action_dim,
            device,
        )

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def build_observation(
        self,
        target_state: Dict[str, Any],
        pursuer_state: Dict[str, Any],
    ) -> np.ndarray:
        """
        Build observation vector from the target's perspective.

        The target is treated as "own" aircraft and the pursuer as "target"
        so that relative geometry (range, range_rate, LOS, ATA, AA) is
        computed from the evader's point of view.

        Args:
            target_state: Target (own) aircraft state dict.
            pursuer_state: Pursuer (adversary) aircraft state dict.

        Returns:
            np.ndarray: Observation vector, shape (16,).
        """
        return build_observation(target_state, pursuer_state)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
        store: bool = True,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Select a normalized action from the observation.

        Args:
            obs: Observation vector, shape (16,).
            deterministic: If True, use the mean action (eval mode).
            store: If True, store the transition in the rollout buffer.

        Returns:
            Tuple of (action_norm, log_prob, value):
                - action_norm: Normalized action in [-1, 1]^3.
                - log_prob: Log probability of the action.
                - value: Value estimate.
        """
        return self._ppo.select_action(obs, deterministic=deterministic, store=store)

    def get_deterministic_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Get the deterministic (mean) action for evaluation.

        Args:
            obs: Observation vector, shape (16,).

        Returns:
            np.ndarray: Normalized action in [-1, 1]^3.
        """
        action, _, _ = self._ppo.select_action(obs, deterministic=True, store=False)
        return action

    # ------------------------------------------------------------------
    # Physical action mapping
    # ------------------------------------------------------------------

    def map_action_to_physical(self, action_norm: np.ndarray) -> Dict[str, float]:
        """
        Map normalized action [-1, 1]^3 to physical flight commands.

        Uses the same linear mapping as CloseRangeTrackingEnv for the
        end-to-end / direct-command mode:
            nz       = act[0] * (nz_max - nz_min) / 2 + (nz_max + nz_min) / 2
            roll_rate = act[1] * (rr_max - rr_min) / 2 + (rr_max + rr_min) / 2
            throttle  = act[2] * (th_max - th_min) / 2 + (th_max + th_min) / 2

        Args:
            action_norm: Normalized action array, shape (3,), values in [-1, 1].

        Returns:
            Dict with keys ``nz_cmd``, ``roll_rate_cmd``, ``throttle_cmd``.
        """
        nz = float(action_norm[0]) * (self._nz_max - self._nz_min) / 2.0 + (self._nz_max + self._nz_min) / 2.0
        roll_rate = float(action_norm[1]) * (self._rr_max - self._rr_min) / 2.0 + (self._rr_max + self._rr_min) / 2.0
        throttle = float(action_norm[2]) * (self._th_max - self._th_min) / 2.0 + (self._th_max + self._th_min) / 2.0
        return {
            "nz_cmd": nz,
            "roll_rate_cmd": roll_rate,
            "throttle_cmd": throttle,
        }

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def store_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store a transition in the rollout buffer (delegates to PPOAgent)."""
        self._ppo.store_transition(obs, action, log_prob, reward, done, value, info)

    def update(self, next_obs: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Perform a PPO update on collected rollout data.

        Args:
            next_obs: Observation after the last transition (for GAE bootstrap).

        Returns:
            Dict with training statistics (policy_loss, value_loss, entropy, etc.).
        """
        return self._ppo.update(next_obs)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save agent checkpoint to disk."""
        self._ppo.save(path)
        logger.info("AdversarialTargetAgent saved to %s", path)

    def load(self, path: str) -> None:
        """Load agent checkpoint from disk."""
        self._ppo.load(path)
        logger.info("AdversarialTargetAgent loaded from %s", path)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def obs_dim(self) -> int:
        return self._ppo.obs_dim

    @property
    def action_dim(self) -> int:
        return self._ppo.action_dim

    @property
    def total_timesteps(self) -> int:
        return self._ppo.total_timesteps

    @property
    def total_updates(self) -> int:
        return self._ppo.total_updates

    @property
    def network(self):
        """Access the underlying policy network (for checkpoint inspection)."""
        return self._ppo.network

    def train(self) -> None:
        """Set agent to training mode."""
        self._ppo.network.train()

    def eval(self) -> None:
        """Set agent to evaluation mode."""
        self._ppo.network.eval()
