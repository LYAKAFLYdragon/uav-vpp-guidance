"""Frozen low-level specialist loader for hierarchical commander control."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.training.train_prediction_vpp_ppo import load_experiment_config


class FrozenSpecialistPolicy:
    """Read-only wrapper around a trained low-level PPO specialist."""

    def __init__(self, checkpoint_path: str, config_path: str, device: str = "cpu"):
        self.checkpoint_path = str(Path(checkpoint_path))
        self.config_path = str(Path(config_path))
        self.device = device
        self.config = load_experiment_config(self.config_path)
        checkpoint = torch.load(self.checkpoint_path, map_location=device)
        self.obs_dim = int(checkpoint.get("obs_dim", 0))
        self.action_dim = int(checkpoint.get("action_dim", 3))
        if self.obs_dim <= 0:
            raise ValueError(
                f"Frozen specialist checkpoint missing valid obs_dim: {checkpoint_path}"
            )
        self.agent = PPOAgent(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            config=self.config,
            device=device,
        )
        self.agent.load(self.checkpoint_path)

    def _adapt_observation(self, obs_vec: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs_vec, dtype=np.float32).flatten()
        current_dim = int(obs.shape[0])
        if current_dim == self.obs_dim:
            return obs
        if current_dim > self.obs_dim:
            return obs[: self.obs_dim]
        pad = np.zeros(self.obs_dim - current_dim, dtype=np.float32)
        return np.concatenate([obs, pad], axis=0)

    def get_deterministic_action(self, obs_vec: np.ndarray) -> np.ndarray:
        obs = self._adapt_observation(obs_vec)
        return self.agent.get_deterministic_action(obs)

    def checkpoint_metadata(self) -> Dict[str, Any]:
        return {
            "checkpoint_path": self.checkpoint_path,
            "config_path": self.config_path,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
        }


def load_frozen_specialist_registry(
    modes: List[Dict[str, Any]],
    device: str = "cpu",
) -> Dict[int, Dict[str, Any]]:
    """Load all frozen specialists keyed by commander mode id."""
    registry: Dict[int, Dict[str, Any]] = {}
    for mode_cfg in modes:
        mode_id = int(mode_cfg["id"])
        registry[mode_id] = {
            "id": mode_id,
            "name": str(mode_cfg["name"]),
            "specialist_key": str(mode_cfg.get("specialist_key", mode_cfg["name"])),
            "specialist_profile": mode_cfg.get("specialist_profile"),
            "source_specialist_key": str(
                mode_cfg.get(
                    "source_specialist_key",
                    mode_cfg.get("specialist_key", mode_cfg["name"]),
                )
            ),
            "policy": FrozenSpecialistPolicy(
                checkpoint_path=str(mode_cfg["checkpoint"]),
                config_path=str(mode_cfg["config_path"]),
                device=device,
            ),
        }
    return registry
