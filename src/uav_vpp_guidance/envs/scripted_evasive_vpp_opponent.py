"""Deterministic current-state-only evasive 3-D VPP opponent."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .observation import compute_relative_geometry
from .opponent_policy import OpponentPolicy


class ScriptedEvasiveVPPOpponent(OpponentPolicy):
    """A checkpoint-free opponent that breaks left and retreats from the ego."""

    action_mode = "vpp"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.retreat_gain = float(self.config.get("retreat_gain", 0.45))
        self.lateral_gain = float(self.config.get("lateral_gain", 0.85))
        self.vertical_gain = float(self.config.get("vertical_gain", 0.20))
        self._last_action = np.zeros(3, dtype=np.float64)
        self._last_diagnostics: Dict[str, Any] = {}
        self.reset()

    def reset(self) -> None:
        self._last_action = np.zeros(3, dtype=np.float64)
        self._last_diagnostics = {"opponent_type": "scripted_evasive_vpp", "step": 0}

    def act(self, opponent_obs: Dict[str, Any]) -> np.ndarray:
        own_state, target_state = opponent_obs["own_state"], opponent_obs["target_state"]
        geometry = opponent_obs.get("physical_relative_state")
        if geometry is None:
            geometry = compute_relative_geometry(own_state, target_state)
        relative_position = np.asarray(geometry["relative_position"], dtype=np.float64)
        horizontal_norm = float(np.linalg.norm(relative_position[:2]))
        if horizontal_norm > 1e-9:
            line_of_sight = relative_position[:2] / horizontal_norm
            left_break = np.array([-line_of_sight[1], line_of_sight[0]])
        else:
            line_of_sight, left_break = np.zeros(2), np.array([0.0, 1.0])
        raw_action = np.array([
            -self.retreat_gain * line_of_sight[0] + self.lateral_gain * left_break[0],
            -self.retreat_gain * line_of_sight[1] + self.lateral_gain * left_break[1],
            -self.vertical_gain * np.sign(relative_position[2]),
        ])
        self._last_action = np.clip(raw_action, -1.0, 1.0)
        self._last_diagnostics = {"opponent_type": "scripted_evasive_vpp", "opponent_action_mode": self.action_mode, "opponent_action": self._last_action.tolist(), "raw_action": raw_action.tolist(), "relative_position_m": relative_position.tolist(), "rule": "current_relative_position_retreat_plus_left_break", "clipped": bool(not np.array_equal(raw_action, self._last_action)), "step": self._last_diagnostics.get("step", 0) + 1}
        return self._last_action.copy()

    def get_diagnostics(self) -> Dict[str, Any]:
        return dict(self._last_diagnostics)
