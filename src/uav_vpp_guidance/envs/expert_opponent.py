"""Rule-based expert opponent wrapper."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ..expert_system.expert_vpp_policy import ExpertVPPPolicy
from .observation import compute_relative_geometry
from .opponent_policy import OpponentPolicy


class ExpertOpponent(OpponentPolicy):
    """Opponent adapter around ``ExpertVPPPolicy``."""

    action_mode = "vpp"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.policy = ExpertVPPPolicy(self.config)
        self._last_action = np.zeros(3, dtype=np.float64)

    def reset(self) -> None:
        if hasattr(self.policy, "reset_history"):
            self.policy.reset_history()

    def act(self, opponent_obs: Dict[str, Any]) -> np.ndarray:
        own_state = opponent_obs["own_state"]
        target_state = opponent_obs["target_state"]
        rel_state = opponent_obs.get("physical_relative_state")
        if rel_state is None:
            rel_state = compute_relative_geometry(own_state, target_state)
        action = self.policy.get_action(own_state, target_state, rel_state)
        self._last_action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        return self._last_action

    def get_diagnostics(self) -> Dict[str, Any]:
        diagnostics = {}
        if hasattr(self.policy, "get_last_diagnostics"):
            diagnostics.update(self.policy.get_last_diagnostics())
        diagnostics.update(
            {
                "opponent_type": "expert",
                "opponent_action_mode": self.action_mode,
                "opponent_action": self._last_action.tolist(),
            }
        )
        return diagnostics
