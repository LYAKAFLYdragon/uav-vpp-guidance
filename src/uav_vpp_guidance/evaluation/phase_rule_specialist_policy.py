"""A frozen, state-only H/C schedule for routing-capacity evaluation."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np

from uav_vpp_guidance.hierarchy.specialist_policy import FrozenSpecialistPolicy


class PhaseRuleSpecialistPolicy:
    """Select one frozen specialist before and after the observed first pass.

    This is deliberately not an oracle: the only routing input is the current
    environment phase snapshot.  It cannot read a future trajectory, outcome,
    reward, or opponent action.  The policy exists to bound what the existing
    two-specialist library can achieve with one pre-registered schedule.
    """

    def __init__(
        self,
        *,
        specialists_config: Mapping[str, Mapping[str, Any]],
        pre_merge_specialist: str,
        post_merge_specialist: str,
        device: str = "cpu",
    ) -> None:
        self._specialists: Dict[str, Dict[str, Any]] = {}
        for key, spec_cfg in specialists_config.items():
            checkpoint = spec_cfg.get("checkpoint")
            config_path = spec_cfg.get("config_path")
            if not checkpoint or not config_path:
                raise ValueError(f"Phase-rule specialist {key!r} needs checkpoint and config_path")
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
        self.pre_merge_specialist = str(pre_merge_specialist)
        self.post_merge_specialist = str(post_merge_specialist)
        if self.pre_merge_specialist not in self._specialists:
            raise ValueError("pre_merge_specialist is absent from specialists_config")
        if self.post_merge_specialist not in self._specialists:
            raise ValueError("post_merge_specialist is absent from specialists_config")
        self.env: Optional[Any] = None
        self.reset_episode()

    def set_env(self, env: Any) -> None:
        self.env = env

    def set_task_name(self, task_name: str) -> None:
        """Compatibility hook; task labels do not influence this schedule."""
        self.current_task_name = str(task_name)

    def reset_episode(self) -> None:
        self.current_task_name: Optional[str] = None
        self._step_count = 0
        self._last_specialist: Optional[str] = None
        self._switch_count = 0
        self._first_switch_step: Optional[int] = None
        self._last_step_metadata: Dict[str, Any] = {}

    def _phase_snapshot(self) -> Dict[str, Any]:
        if self.env is None or not hasattr(self.env, "get_runtime_phase_state"):
            raise RuntimeError("PhaseRuleSpecialistPolicy requires env.get_runtime_phase_state()")
        snapshot = self.env.get_runtime_phase_state()
        if not isinstance(snapshot, Mapping) or "first_pass_complete" not in snapshot:
            raise RuntimeError("runtime phase snapshot lacks first_pass_complete")
        return dict(snapshot)

    def _selected_key(self, snapshot: Mapping[str, Any]) -> tuple[str, str]:
        if bool(snapshot["first_pass_complete"]):
            return "post_merge", self.post_merge_specialist
        return "pre_merge", self.pre_merge_specialist

    def get_deterministic_action(self, obs: np.ndarray) -> np.ndarray:
        snapshot = self._phase_snapshot()
        phase, selected_key = self._selected_key(snapshot)
        entry = self._specialists[selected_key]
        self._step_count += 1
        if self._last_specialist is not None and self._last_specialist != selected_key:
            self._switch_count += 1
            if self._first_switch_step is None:
                self._first_switch_step = self._step_count
        self._last_specialist = selected_key
        if hasattr(self.env, "set_runtime_specialist_context"):
            self.env.set_runtime_specialist_context(
                specialist_key=entry["specialist_key"],
                specialist_profile=entry["specialist_profile"],
                specialist_mode_name=entry["specialist_mode_name"],
                specialist_reason="phase_rule_observed_first_pass",
            )
        self._last_step_metadata = {
            "phase_rule_schedule_phase": phase,
            "phase_rule_first_pass_complete": bool(snapshot["first_pass_complete"]),
            "phase_rule_selected_specialist": entry["specialist_key"],
            "phase_rule_selected_specialist_mode_name": entry["specialist_mode_name"],
            "phase_rule_switch_count": self._switch_count,
            "phase_rule_first_switch_step": self._first_switch_step,
            "phase_rule_uses_future_state": False,
        }
        return entry["policy"].get_deterministic_action(obs)

    def get_last_step_metadata(self) -> Dict[str, Any]:
        return dict(self._last_step_metadata)
