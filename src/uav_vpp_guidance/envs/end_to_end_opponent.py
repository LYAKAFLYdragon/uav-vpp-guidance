"""End-to-end neural opponent loaded from a local PPO checkpoint."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from ..agents.end_to_end_ppo_agent import EndToEndPPOAgent
from .opponent_policy import OpponentPolicy


class EndToEndOpponent(OpponentPolicy):
    """Neural opponent that maps inverted observations to direct commands."""

    action_mode = "direct_command"

    def __init__(
        self,
        checkpoint_path: str,
        config: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
        invert_observation: bool = True,
    ):
        self.checkpoint_path = str(checkpoint_path)
        self.device = device
        self.invert_observation = bool(invert_observation)
        self._checkpoint = torch.load(self.checkpoint_path, map_location=device)
        ckpt_config = self._checkpoint.get("config", {})
        self.config = copy.deepcopy(ckpt_config if ckpt_config else (config or {}))
        self.obs_dim = int(self._checkpoint.get("obs_dim", 16))
        self.action_dim = int(self._checkpoint.get("action_dim", 3))
        self.agent = EndToEndPPOAgent(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            config=self.config,
            device=device,
        )
        self.agent.network.load_state_dict(
            self._checkpoint["network_state_dict"], strict=False
        )
        self.agent.network.eval()
        self._last_action = np.zeros(self.action_dim, dtype=np.float32)

    @classmethod
    def from_registry(cls, registry_config: Dict[str, Any]) -> "EndToEndOpponent":
        """Build from an ``opponent_registry`` entry."""
        checkpoint = registry_config.get("checkpoint")
        if checkpoint is None:
            raise ValueError("EndToEndOpponent requires a checkpoint path")
        return cls(
            checkpoint_path=str(Path(checkpoint)),
            config=registry_config.get("config"),
            device=registry_config.get("device", "cpu"),
            invert_observation=registry_config.get("invert_observation", True),
        )

    def act(self, opponent_obs: Dict[str, Any]) -> np.ndarray:
        obs_vec = np.asarray(opponent_obs["observation_vector"], dtype=np.float32)
        if obs_vec.shape[0] != self.obs_dim:
            raise ValueError(
                f"Opponent checkpoint expects obs_dim={self.obs_dim}, "
                f"got {obs_vec.shape[0]}"
            )
        action = self.agent.get_deterministic_action(obs_vec)
        # EndToEndPPOAgent.clip_action is a clamp to policy bounds only.
        # The environment performs the normalized [-1, 1] -> physical command
        # mapping later, so this must not denormalize the action.
        action = self.agent.clip_action(action)
        self._last_action = np.asarray(action, dtype=np.float32)
        return self._last_action

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            "opponent_type": "end_to_end",
            "opponent_checkpoint": self.checkpoint_path,
            "opponent_invert_observation": self.invert_observation,
            "opponent_action_mode": self.action_mode,
            "opponent_action": self._last_action.tolist(),
        }
