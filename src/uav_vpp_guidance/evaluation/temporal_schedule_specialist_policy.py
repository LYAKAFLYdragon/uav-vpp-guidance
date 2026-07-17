"""A frozen two-stage specialist schedule for routing-capacity evaluation."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np

from uav_vpp_guidance.hierarchy.specialist_policy import FrozenSpecialistPolicy


class TemporalScheduleSpecialistPolicy:
    """Use one frozen specialist for a fixed observed prefix, then another.

    The switch count is determined before execution and is independent of
    reward, terminal outcome, future state, and opponent future actions.  It
    provides a mechanically activated composition control when a scenario
    family does not reach a merge-triggered phase transition.
    """

    def __init__(
        self,
        *,
        specialists_config: Mapping[str, Mapping[str, Any]],
        initial_specialist: str,
        later_specialist: str,
        initial_high_level_steps: int,
        device: str = "cpu",
    ) -> None:
        if int(initial_high_level_steps) <= 0:
            raise ValueError("initial_high_level_steps must be positive")
        self._specialists: Dict[str, Dict[str, Any]] = {}
        for key, spec_cfg in specialists_config.items():
            checkpoint = spec_cfg.get("checkpoint")
            config_path = spec_cfg.get("config_path")
            if not checkpoint or not config_path:
                raise ValueError(f"Temporal-schedule specialist {key!r} needs checkpoint and config_path")
            self._specialists[str(key)] = {
                "specialist_key": str(spec_cfg.get("specialist_key", key)),
                "specialist_mode_name": str(spec_cfg.get("specialist_mode_name", f"{key}_specialist")),
                "specialist_profile": spec_cfg.get("specialist_profile"),
                "policy": FrozenSpecialistPolicy(
                    checkpoint_path=str(checkpoint),
                    config_path=str(config_path),
                    device=device,
                ),
            }
        self.initial_specialist = str(initial_specialist)
        self.later_specialist = str(later_specialist)
        if self.initial_specialist not in self._specialists or self.later_specialist not in self._specialists:
            raise ValueError("Temporal schedule specialist is absent from specialists_config")
        self.initial_high_level_steps = int(initial_high_level_steps)
        self.env: Optional[Any] = None
        self.reset_episode()

    def set_env(self, env: Any) -> None:
        self.env = env

    def set_task_name(self, task_name: str) -> None:
        self.current_task_name = str(task_name)

    def reset_episode(self) -> None:
        self.current_task_name: Optional[str] = None
        self._step_count = 0
        self._last_specialist: Optional[str] = None
        self._switch_count = 0
        self._first_switch_step: Optional[int] = None
        self._last_step_metadata: Dict[str, Any] = {}

    def get_deterministic_action(self, obs: np.ndarray) -> np.ndarray:
        self._step_count += 1
        phase = "initial" if self._step_count <= self.initial_high_level_steps else "later"
        selected_key = self.initial_specialist if phase == "initial" else self.later_specialist
        entry = self._specialists[selected_key]
        if self._last_specialist is not None and self._last_specialist != selected_key:
            self._switch_count += 1
            if self._first_switch_step is None:
                self._first_switch_step = self._step_count
        self._last_specialist = selected_key
        if self.env is not None and hasattr(self.env, "set_runtime_specialist_context"):
            self.env.set_runtime_specialist_context(
                specialist_key=entry["specialist_key"],
                specialist_profile=entry["specialist_profile"],
                specialist_mode_name=entry["specialist_mode_name"],
                specialist_reason="temporal_schedule_predeclared_step",
            )
        self._last_step_metadata = {
            "temporal_schedule_phase": phase,
            "temporal_schedule_selected_specialist": entry["specialist_key"],
            "temporal_schedule_initial_high_level_steps": self.initial_high_level_steps,
            "temporal_schedule_switch_count": self._switch_count,
            "temporal_schedule_first_switch_step": self._first_switch_step,
            "temporal_schedule_uses_future_state": False,
        }
        return entry["policy"].get_deterministic_action(obs)

    def get_last_step_metadata(self) -> Dict[str, Any]:
        return dict(self._last_step_metadata)
