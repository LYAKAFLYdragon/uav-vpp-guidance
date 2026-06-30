"""Deterministic rule-guidance policy wrappers for comparison runs.

This module exposes a minimal policy interface compatible with
``scripts/run_jsbsim_hrl_comparison.py``.

The current combat-only rule-guidance comparators rely on the environment's
existing guidance stack:

- ``virtual_point.mode = zero_offset``
- ``virtual_point.anchor_mode = current_target``
- ``guidance.mode = los_rate`` or ``proportional_navigation``

Under that setup, the policy action should not perturb the virtual point.
Returning a constant zero vector keeps the comparison focused on the guidance
law itself rather than on a learned offset policy.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


class RuleGuidancePolicy:
    """Deterministic zero-action policy for rule-guidance baselines."""

    def __init__(
        self,
        action_dim: int = 3,
        constant_action: Optional[np.ndarray] = None,
        action_by_task: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        self.action_dim = int(action_dim)
        if self.action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {self.action_dim}")

        self.current_task_name: Optional[str] = None
        self._constant_action = self._normalize_action(
            np.zeros(self.action_dim, dtype=np.float32)
            if constant_action is None
            else constant_action
        )
        self._action_by_task = {
            str(task_name): self._normalize_action(action)
            for task_name, action in (action_by_task or {}).items()
        }

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.shape[0] != self.action_dim:
            raise ValueError(
                f"Expected action with dim {self.action_dim}, got {arr.shape[0]}"
            )
        return np.clip(arr, -1.0, 1.0)

    def set_task_name(self, task_name: str) -> None:
        self.current_task_name = str(task_name)

    def get_deterministic_action(self, obs: np.ndarray) -> np.ndarray:
        del obs
        if self.current_task_name is not None:
            task_action = self._action_by_task.get(self.current_task_name)
            if task_action is not None:
                return task_action.copy()
        return self._constant_action.copy()

    def load(self, path: str) -> None:
        """Compatibility no-op for policy-like wrappers without checkpoints."""
        del path
