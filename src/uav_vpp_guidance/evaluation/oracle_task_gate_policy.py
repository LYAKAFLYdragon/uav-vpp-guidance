"""Oracle Task Gate Policy: selects a frozen specialist based on task name.

This is a hard-wired control baseline for the hierarchical commander MVP.
It routes each task to a pre-trained specialist without any learning.
"""
from typing import Any, Dict, Optional
import numpy as np
import torch

from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.training.train_prediction_vpp_ppo import load_experiment_config


class OracleTaskGatePolicy:
    """Static policy that selects a specialist based on the current task name."""

    def __init__(
        self,
        specialists_config: Dict[str, Dict[str, str]],
        device: str = "cpu",
    ):
        """
        Args:
            specialists_config: mapping from task name to dict with keys
                "checkpoint" and "config_path".
                Example:
                {
                    "head_on": {
                        "checkpoint": "path/to/head_on.pt",
                        "config_path": "path/to/head_on_config.yaml",
                    },
                    "crossing_feasible": {
                        "checkpoint": "path/to/crossing.pt",
                        "config_path": "path/to/crossing_config.yaml",
                    },
                }
            device: torch device for loading specialists.
        """
        self.device = device
        self.current_task_name: Optional[str] = None
        self._specialists: Dict[str, PPOAgent] = {}

        for task_name, spec_cfg in specialists_config.items():
            ckpt = spec_cfg.get("checkpoint")
            cfg_path = spec_cfg.get("config_path")
            if not ckpt or not cfg_path:
                raise ValueError(
                    f"Specialist for {task_name} must have 'checkpoint' and 'config_path'"
                )
            # Match the hierarchical commander specialist loader exactly so
            # oracle vs commander comparisons see the same include-resolved
            # specialist config surface.
            config = load_experiment_config(cfg_path)
            # Load checkpoint first to infer dimensions
            checkpoint = torch.load(ckpt, map_location=device)
            ckpt_obs_dim = checkpoint.get("obs_dim")
            ckpt_action_dim = checkpoint.get("action_dim")
            obs_dim = int(ckpt_obs_dim) if ckpt_obs_dim is not None else 20
            action_dim = int(ckpt_action_dim) if ckpt_action_dim is not None else 3
            specialist = PPOAgent(
                obs_dim=obs_dim,
                action_dim=action_dim,
                config=config,
                device=device,
            )
            specialist.load(ckpt)
            self._specialists[task_name] = specialist

    def set_task_name(self, task_name: str) -> None:
        """Set the current task name so the gate can select the right specialist."""
        self.current_task_name = task_name

    def get_deterministic_action(self, obs: np.ndarray) -> np.ndarray:
        """Return the deterministic action from the specialist for the current task."""
        if self.current_task_name is None:
            raise RuntimeError(
                "OracleTaskGatePolicy.current_task_name is not set. "
                "Call set_task_name() before get_deterministic_action()."
            )
        specialist = self._specialists.get(self.current_task_name)
        if specialist is None:
            # Fallback: if no specialist for this task, use the first available
            if self._specialists:
                specialist = next(iter(self._specialists.values()))
            else:
                raise RuntimeError("No specialists loaded in OracleTaskGatePolicy")
        
        # Adjust observation dimension to match specialist's expected input
        target_dim = specialist.obs_dim
        current_dim = obs.shape[-1]
        if current_dim != target_dim:
            if current_dim > target_dim:
                obs = obs[..., :target_dim]
            else:
                pad_width = [(0, 0)] * (obs.ndim - 1) + [(0, target_dim - current_dim)]
                obs = np.pad(obs, pad_width, mode="constant", constant_values=0)
        
        return specialist.get_deterministic_action(obs)

    def load(self, path: str) -> None:
        """No-op: specialists are already loaded in __init__."""
        pass
