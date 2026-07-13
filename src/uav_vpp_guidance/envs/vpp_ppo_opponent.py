"""Strict continuous PPO/VPP checkpoint adapter for embedded opponents."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from ..agents.ppo_agent import PPOAgent
from .opponent_policy import OpponentPolicy


class VPPPPOOpponent(OpponentPolicy):
    """Deploy a role-reversed base-geometry PPO checkpoint through target VPP control."""

    action_mode = "vpp"
    REQUIRED_OBS_DIM = 16
    REQUIRED_ACTION_DIM = 3

    def __init__(
        self,
        checkpoint_path: str,
        config: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
    ):
        self.checkpoint_path = str(Path(checkpoint_path))
        self.device = device
        checkpoint = torch.load(self.checkpoint_path, map_location=device)
        self.config = copy.deepcopy(checkpoint.get("config") or config or {})
        self.obs_dim = int(checkpoint.get("obs_dim", 0))
        self.action_dim = int(checkpoint.get("action_dim", 0))
        if self.obs_dim != self.REQUIRED_OBS_DIM:
            raise ValueError(
                "VPPPPOOpponent requires the role-reversed base-geometry "
                f"contract ({self.REQUIRED_OBS_DIM}-D), got {self.obs_dim}-D"
            )
        if self.action_dim != self.REQUIRED_ACTION_DIM:
            raise ValueError(
                "VPPPPOOpponent requires a continuous 3-D VPP action, "
                f"got action_dim={self.action_dim}"
            )
        observation_cfg = self.config.get("observation", {}) or {}
        if observation_cfg.get("include_task_type", False):
            raise ValueError("VPPPPOOpponent checkpoint must not use task-type bits")
        if observation_cfg.get("include_opponent_stage", False):
            raise ValueError("VPPPPOOpponent checkpoint must not use opponent-stage bits")

        self.agent = PPOAgent(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            config=self.config,
            device=device,
        )
        self.agent.load(self.checkpoint_path)
        self.agent.network.eval()
        self._last_action = np.zeros(self.action_dim, dtype=np.float32)

    @classmethod
    def from_registry(cls, registry_config: Dict[str, Any]) -> "VPPPPOOpponent":
        checkpoint = registry_config.get("checkpoint")
        if checkpoint in (None, ""):
            raise ValueError("VPPPPOOpponent requires a checkpoint path")
        return cls(
            checkpoint_path=str(checkpoint),
            config=registry_config.get("config"),
            device=str(registry_config.get("device", "cpu")),
        )

    def act(self, opponent_obs: Dict[str, Any]) -> np.ndarray:
        schema = dict(opponent_obs.get("observation_schema") or {})
        if not bool(schema.get("role_reversed", False)):
            raise ValueError("VPPPPOOpponent requires a role-reversed observation")
        observation = np.asarray(opponent_obs["observation_vector"], dtype=np.float32).reshape(-1)
        if observation.shape != (self.obs_dim,):
            raise ValueError(
                "VPPPPOOpponent observation shape mismatch: "
                f"expected {(self.obs_dim,)}, got {observation.shape}"
            )
        action = np.asarray(self.agent.get_deterministic_action(observation), dtype=np.float32)
        if action.shape != (self.action_dim,) or not np.all(np.isfinite(action)):
            raise ValueError("VPPPPOOpponent produced a non-finite or malformed action")
        self._last_action = np.clip(action, -1.0, 1.0)
        return self._last_action.copy()

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            "opponent_type": "independent_ppo_vpp",
            "opponent_checkpoint": self.checkpoint_path,
            "opponent_action_mode": self.action_mode,
            "opponent_obs_dim": self.obs_dim,
            "opponent_action_dim": self.action_dim,
            "opponent_action": self._last_action.tolist(),
        }
