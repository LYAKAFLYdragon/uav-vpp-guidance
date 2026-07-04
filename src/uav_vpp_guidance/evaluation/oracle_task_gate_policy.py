"""Oracle Task Gate Policy: selects a frozen specialist based on task name.

This is a hard-wired control baseline for the hierarchical commander MVP.
It routes each task to a pre-trained specialist without any learning.
"""
from typing import Any, Dict, Optional, Tuple

import numpy as np

from uav_vpp_guidance.hierarchy.specialist_policy import FrozenSpecialistPolicy


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
        self.env: Optional[Any] = None
        self._specialists: Dict[str, Dict[str, Any]] = {}
        self._last_step_metadata: Dict[str, Any] = {}

        for task_name, spec_cfg in specialists_config.items():
            ckpt = spec_cfg.get("checkpoint")
            cfg_path = spec_cfg.get("config_path")
            if not ckpt or not cfg_path:
                raise ValueError(
                    f"Specialist for {task_name} must have 'checkpoint' and 'config_path'"
                )
            specialist_key = str(spec_cfg.get("specialist_key", task_name))
            specialist_profile = spec_cfg.get("specialist_profile")
            specialist_mode_name = str(
                spec_cfg.get(
                    "specialist_mode_name",
                    self._default_specialist_mode_name(specialist_key),
                )
            )
            self._specialists[task_name] = {
                "task_name": str(task_name),
                "specialist_key": specialist_key,
                "specialist_profile": specialist_profile,
                "specialist_mode_name": specialist_mode_name,
                "policy": FrozenSpecialistPolicy(
                    checkpoint_path=str(ckpt),
                    config_path=str(cfg_path),
                    device=device,
                ),
            }

    def set_task_name(self, task_name: str) -> None:
        """Set the current task name so the gate can select the right specialist."""
        self.current_task_name = task_name

    def set_env(self, env: Any) -> None:
        self.env = env

    @staticmethod
    def _default_specialist_mode_name(specialist_key: str) -> str:
        mapping = {
            "head_on": "head_on_specialist",
            "crossing_feasible": "crossing_specialist",
            "post_merge_recovery": "post_merge_recovery_specialist",
        }
        return mapping.get(str(specialist_key), f"{specialist_key}_specialist")

    def _selected_specialist_entry(self) -> Tuple[str, Dict[str, Any]]:
        specialist_entry = self._specialists.get(self.current_task_name)
        selected_task_name = self.current_task_name
        if specialist_entry is None:
            if self._specialists:
                selected_task_name, specialist_entry = next(iter(self._specialists.items()))
            else:
                raise RuntimeError("No specialists loaded in OracleTaskGatePolicy")
        return str(selected_task_name), specialist_entry

    def _set_env_specialist_context(self, specialist_entry: Dict[str, Any]) -> None:
        if self.env is None or not hasattr(self.env, "set_runtime_specialist_context"):
            return
        self.env.set_runtime_specialist_context(
            specialist_key=specialist_entry.get("specialist_key"),
            specialist_profile=specialist_entry.get("specialist_profile"),
            specialist_mode_name=specialist_entry.get("specialist_mode_name"),
        )

    def get_deterministic_action(self, obs: np.ndarray) -> np.ndarray:
        """Return the deterministic action from the specialist for the current task."""
        if self.current_task_name is None:
            raise RuntimeError(
                "OracleTaskGatePolicy.current_task_name is not set. "
                "Call set_task_name() before get_deterministic_action()."
            )

        selected_task_name, specialist_entry = self._selected_specialist_entry()
        self._set_env_specialist_context(specialist_entry)
        action = specialist_entry["policy"].get_deterministic_action(obs)
        self._last_step_metadata = {
            "oracle_task_gate_selected_task_name": selected_task_name,
            "oracle_task_gate_selected_specialist": specialist_entry.get(
                "specialist_key"
            ),
            "oracle_task_gate_selected_specialist_profile": specialist_entry.get(
                "specialist_profile"
            ),
            "oracle_task_gate_selected_specialist_mode_name": specialist_entry.get(
                "specialist_mode_name"
            ),
        }
        return action

    def get_last_step_metadata(self) -> Dict[str, Any]:
        return dict(self._last_step_metadata)

    def load(self, path: str) -> None:
        """No-op: specialists are already loaded in __init__."""
        pass
