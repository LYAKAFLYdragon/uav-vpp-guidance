"""Opponent policy interface for embedded adversarial control."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class OpponentPolicy(ABC):
    """Abstract controller used by ``CloseRangeTrackingEnv`` for target aircraft."""

    action_mode = "direct_command"

    @abstractmethod
    def act(self, opponent_obs: Dict[str, Any]) -> np.ndarray:
        """Return a 3-D continuous opponent action."""

    def reset(self) -> None:
        """Reset optional internal policy state."""
        return None

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return optional policy diagnostics from the latest action."""
        return {}
