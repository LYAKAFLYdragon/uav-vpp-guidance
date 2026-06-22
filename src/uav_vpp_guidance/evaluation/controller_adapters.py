"""
Controller adapters for unified evaluation.

Provides a common interface so the evaluation runner does not need to know
whether a controller is a trained PPO checkpoint or a fixed zero-offset PID.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class ControllerAdapter(ABC):
    """Uniform controller interface for flight-control comparison."""

    name: str

    @abstractmethod
    def reset(self) -> None:
        """Reset any per-episode controller state."""
        ...

    @abstractmethod
    def act(self, obs: dict, info: Optional[dict] = None) -> np.ndarray:
        """Return a normalized action vector for the current observation."""
        ...


class PPOAdapter(ControllerAdapter):
    """Loads a PPO / PPO+PID checkpoint and performs deterministic inference."""

    def __init__(self, agent, name: str = "ppo"):
        self.agent = agent
        self.name = name

    def reset(self) -> None:
        pass

    def act(self, obs: dict, info: Optional[dict] = None) -> np.ndarray:
        return self.agent.get_deterministic_action(obs["observation_vector"])


class ZeroOffsetAdapter(ControllerAdapter):
    """
    Pure PID controller: action is always a zero vector.

    Relies on the VPP layer being configured in ``zero_offset`` mode so that
    the guidance chain tracks the reference target directly.

    For PPO+PID compatibility, action_dim defaults to 4 so that
    aggressiveness=0 (gain_scale=1.0) is recorded in telemetry.
    """

    def __init__(self, name: str, action_dim: int = 4):
        self.name = name
        self.action_dim = int(action_dim)

    def reset(self) -> None:
        pass

    def act(self, obs: dict, info: Optional[dict] = None) -> np.ndarray:
        return np.zeros(self.action_dim, dtype=np.float32)
