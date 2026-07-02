"""Evaluation-time learned hierarchical commander wrapper."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from uav_vpp_guidance.agents.commander_double_dqn_agent import CommanderDoubleDQNAgent
from uav_vpp_guidance.agents.commander_ppo_agent import CommanderPPOAgent
from uav_vpp_guidance.hierarchy.commander_mode_constraints import (
    apply_head_on_post_merge_reopened_crossing_overdeep_clamp,
    apply_head_on_post_merge_reopened_crossing_secondary_clamp,
    apply_head_on_post_merge_reopened_crossing_leash,
    build_head_on_post_merge_reopened_crossing_snapshot,
    evaluate_head_on_post_merge_reopened_crossing_overdeep_clamp_state,
    evaluate_head_on_post_merge_reopened_crossing_secondary_clamp_state,
    evaluate_head_on_post_merge_reopened_crossing_leash_state,
)
from uav_vpp_guidance.hierarchy.specialist_policy import load_frozen_specialist_registry


class HierarchicalCommanderPolicy:
    """Select a frozen specialist every macro step using a learned commander."""

    def __init__(
        self,
        checkpoint_path: str,
        config: Dict[str, Any],
        obs_dim: int,
        device: str = "cpu",
    ) -> None:
        self.checkpoint_path = str(checkpoint_path)
        self.config = config
        self.device = device
        commander_cfg = config.get("commander", {})
        self.macro_action_repeat_steps = int(
            commander_cfg.get("macro_action_repeat_steps", 12)
        )
        self.min_mode_hold_steps = int(commander_cfg.get("min_mode_hold_steps", 0))
        self.head_on_post_merge_reopened_crossing_leash = dict(
            commander_cfg.get("head_on_post_merge_reopened_crossing_leash", {}) or {}
        )
        self.head_on_post_merge_reopened_crossing_secondary_clamp = dict(
            commander_cfg.get(
                "head_on_post_merge_reopened_crossing_secondary_clamp", {}
            )
            or {}
        )
        self.head_on_post_merge_reopened_crossing_overdeep_clamp = dict(
            commander_cfg.get(
                "head_on_post_merge_reopened_crossing_overdeep_clamp", {}
            )
            or {}
        )
        self.crossing_pre_merge_mode_lock = dict(
            commander_cfg.get("crossing_pre_merge_mode_lock", {}) or {}
        )
        self.mode_registry = load_frozen_specialist_registry(
            list(commander_cfg.get("modes", [])),
            device=device,
        )
        algorithm = str(commander_cfg.get("algorithm", "ppo")).lower()
        if algorithm == "double_dqn":
            self.commander = CommanderDoubleDQNAgent(
                obs_dim=int(obs_dim),
                action_dim=len(self.mode_registry),
                config=config,
                device=device,
            )
        else:
            self.commander = CommanderPPOAgent(
                obs_dim=int(obs_dim),
                action_dim=len(self.mode_registry),
                config=config,
                device=device,
            )
        self.commander.load(self.checkpoint_path)
        self.current_task_name: Optional[str] = None
        self.env = None
        self.reset_episode()

    def set_task_name(self, task_name: str) -> None:
        self.current_task_name = str(task_name)

    def set_env(self, env: Any) -> None:
        self.env = env

    def reset_episode(self) -> None:
        self._active_mode_id: Optional[int] = None
        self._step_counter = 0
        self._switch_count = 0
        self._steps_since_switch = 0
        self._first_switch_step: Optional[int] = None
        self._macro_steps_since_switch = 0
        self._hold_steps_remaining = 0
        self._head_on_post_merge_reopened_crossing_consecutive_macro_steps = 0
        self._head_on_post_merge_reopened_crossing_altitude_history_m = []
        self._last_step_metadata: Dict[str, Any] = {}

    def _select_mode(self, obs_vec: np.ndarray) -> int:
        return int(self.commander.get_deterministic_action(obs_vec))

    def _mode_name(self, mode_id: Optional[int]) -> Optional[str]:
        if mode_id is None:
            return None
        mode = self.mode_registry.get(int(mode_id))
        if mode is None:
            return None
        return str(mode.get("name"))

    def _mode_id_for_specialist_key(self, specialist_key: str) -> Optional[int]:
        for mode_id, mode in self.mode_registry.items():
            if str(mode.get("specialist_key")) == str(specialist_key):
                return int(mode_id)
        return None

    def _initial_task_bootstrap_mode_id(self) -> Optional[int]:
        """Narrow repair: guarantee crossing starts in crossing specialist."""
        if self._active_mode_id is not None:
            return None
        if str(self.current_task_name) != "crossing_feasible":
            return None
        return self._mode_id_for_specialist_key("crossing_feasible")

    def _crossing_pre_merge_mode_lock_id(
        self, commander_snapshot: Dict[str, Any]
    ) -> Optional[int]:
        cfg = self.crossing_pre_merge_mode_lock
        if not bool(cfg.get("enabled", False)):
            return None
        if str(self.current_task_name) != str(
            cfg.get("task_name", "crossing_feasible")
        ):
            return None
        if bool(commander_snapshot.get("first_pass_complete", False)):
            return None
        forced_specialist_key = str(
            cfg.get("forced_specialist_key", "crossing_feasible")
        )
        return self._mode_id_for_specialist_key(forced_specialist_key)

    def _update_active_mode(self, new_mode: int) -> bool:
        new_mode = int(new_mode)
        switched = self._active_mode_id is not None and new_mode != self._active_mode_id
        if switched and self._hold_steps_remaining > 0:
            # Enforce minimum mode hold: ignore switch request
            return False
        if switched:
            self._switch_count += 1
            if self._first_switch_step is None:
                self._first_switch_step = self._step_counter
            self._steps_since_switch = 0
            self._macro_steps_since_switch = 0
            self._hold_steps_remaining = self.min_mode_hold_steps
        self._active_mode_id = new_mode
        return switched

    def _remember_secondary_clamp_altitude(self, altitude_m: float) -> None:
        if not np.isfinite(altitude_m):
            return
        self._head_on_post_merge_reopened_crossing_altitude_history_m.append(
            float(altitude_m)
        )
        if len(self._head_on_post_merge_reopened_crossing_altitude_history_m) > 128:
            self._head_on_post_merge_reopened_crossing_altitude_history_m.pop(0)

    def get_deterministic_action(self, obs: np.ndarray) -> np.ndarray:
        commander_snapshot = build_head_on_post_merge_reopened_crossing_snapshot(
            env=self.env,
            task_name=self.current_task_name,
        )
        leash_state = evaluate_head_on_post_merge_reopened_crossing_leash_state(
            snapshot=commander_snapshot,
            leash_cfg=self.head_on_post_merge_reopened_crossing_leash,
        )
        secondary_clamp_state = (
            evaluate_head_on_post_merge_reopened_crossing_secondary_clamp_state(
                snapshot=commander_snapshot,
                altitude_history_m=(
                    self._head_on_post_merge_reopened_crossing_altitude_history_m
                ),
                secondary_cfg=(
                    self.head_on_post_merge_reopened_crossing_secondary_clamp
                ),
            )
        )
        overdeep_clamp_state = (
            evaluate_head_on_post_merge_reopened_crossing_overdeep_clamp_state(
                snapshot=commander_snapshot,
                overdeep_cfg=(
                    self.head_on_post_merge_reopened_crossing_overdeep_clamp
                ),
            )
        )
        should_select = (
            self._active_mode_id is None
            or self._step_counter % self.macro_action_repeat_steps == 0
        )
        switched = False
        requested_mode_id = self._active_mode_id
        requested_mode_name = self._mode_name(self._active_mode_id)
        constraint_triggered = False
        constraint_reason = None
        crossing_pre_merge_mode_lock_active = False
        if should_select:
            self._macro_steps_since_switch += 1
            if self._hold_steps_remaining > 0:
                self._hold_steps_remaining -= 1
            bootstrap_mode_id = self._initial_task_bootstrap_mode_id()
            if bootstrap_mode_id is not None:
                requested_mode_id = bootstrap_mode_id
            else:
                crossing_pre_merge_lock_mode_id = self._crossing_pre_merge_mode_lock_id(
                    commander_snapshot
                )
                if crossing_pre_merge_lock_mode_id is not None:
                    requested_mode_id = crossing_pre_merge_lock_mode_id
                    crossing_pre_merge_mode_lock_active = True
                else:
                    requested_mode_id = self._select_mode(obs)
            requested_mode_name = self._mode_name(requested_mode_id)
            constraint_result = apply_head_on_post_merge_reopened_crossing_leash(
                requested_mode_id=requested_mode_id,
                active_mode_id=self._active_mode_id,
                consecutive_crossing_macro_steps=(
                    self._head_on_post_merge_reopened_crossing_consecutive_macro_steps
                ),
                mode_registry=self.mode_registry,
                leash_state=leash_state,
                leash_cfg=self.head_on_post_merge_reopened_crossing_leash,
            )
            new_mode = int(constraint_result["effective_mode_id"])
            constraint_triggered = bool(constraint_result["triggered"])
            constraint_reason = str(constraint_result["reason"])
            self._head_on_post_merge_reopened_crossing_consecutive_macro_steps = int(
                constraint_result["next_consecutive_crossing_macro_steps"]
            )
            overdeep_result = (
                apply_head_on_post_merge_reopened_crossing_overdeep_clamp(
                    candidate_mode_id=new_mode,
                    mode_registry=self.mode_registry,
                    overdeep_state=overdeep_clamp_state,
                    overdeep_cfg=(
                        self.head_on_post_merge_reopened_crossing_overdeep_clamp
                    ),
                )
            )
            if bool(overdeep_result["triggered"]):
                new_mode = int(overdeep_result["effective_mode_id"])
                constraint_triggered = True
                constraint_reason = str(overdeep_result["reason"])
                self._head_on_post_merge_reopened_crossing_consecutive_macro_steps = 0
            secondary_result = (
                apply_head_on_post_merge_reopened_crossing_secondary_clamp(
                    candidate_mode_id=new_mode,
                    mode_registry=self.mode_registry,
                    secondary_state=secondary_clamp_state,
                    secondary_cfg=(
                        self.head_on_post_merge_reopened_crossing_secondary_clamp
                    ),
                )
            )
            if bool(secondary_result["triggered"]):
                new_mode = int(secondary_result["effective_mode_id"])
                constraint_triggered = True
                constraint_reason = str(secondary_result["reason"])
                self._head_on_post_merge_reopened_crossing_consecutive_macro_steps = 0
            switched = self._update_active_mode(new_mode)
        else:
            active_mode_name = self._mode_name(self._active_mode_id)
            overdeep_result = apply_head_on_post_merge_reopened_crossing_overdeep_clamp(
                candidate_mode_id=self._active_mode_id,
                mode_registry=self.mode_registry,
                overdeep_state=overdeep_clamp_state,
                overdeep_cfg=self.head_on_post_merge_reopened_crossing_overdeep_clamp,
            )
            secondary_result = apply_head_on_post_merge_reopened_crossing_secondary_clamp(
                candidate_mode_id=self._active_mode_id,
                mode_registry=self.mode_registry,
                secondary_state=secondary_clamp_state,
                secondary_cfg=self.head_on_post_merge_reopened_crossing_secondary_clamp,
            )
            if bool(overdeep_result["triggered"]):
                requested_mode_name = active_mode_name
                constraint_triggered = True
                constraint_reason = str(overdeep_result["reason"])
                self._head_on_post_merge_reopened_crossing_consecutive_macro_steps = 0
                switched = self._update_active_mode(
                    int(overdeep_result["effective_mode_id"])
                )
            elif bool(secondary_result["triggered"]):
                requested_mode_name = active_mode_name
                constraint_triggered = True
                constraint_reason = str(secondary_result["reason"])
                self._head_on_post_merge_reopened_crossing_consecutive_macro_steps = 0
                switched = self._update_active_mode(
                    int(secondary_result["effective_mode_id"])
                )
            elif (
                not leash_state.get("active", False)
                or active_mode_name != "crossing_specialist"
            ):
                self._head_on_post_merge_reopened_crossing_consecutive_macro_steps = 0

        if self._active_mode_id is None:
            raise RuntimeError("HierarchicalCommanderPolicy failed to select a mode")

        mode = self.mode_registry[self._active_mode_id]
        action = mode["policy"].get_deterministic_action(obs)
        self._step_counter += 1
        self._steps_since_switch += 1
        self._last_step_metadata = {
            "commander_mode_id": int(self._active_mode_id),
            "commander_mode_name": mode["name"],
            "commander_selected_specialist": mode["specialist_key"],
            "commander_switch_count": int(self._switch_count),
            "commander_steps_since_switch": int(self._steps_since_switch),
            "commander_macro_action_repeat_steps": int(self.macro_action_repeat_steps),
            "commander_task_oracle_gate": self.current_task_name,
            "commander_requested_mode_id": (
                int(requested_mode_id) if requested_mode_id is not None else None
            ),
            "commander_requested_mode_name": requested_mode_name,
            "commander_crossing_pre_merge_mode_lock_active": bool(
                crossing_pre_merge_mode_lock_active
            ),
            "commander_mode_constraint_triggered": bool(constraint_triggered),
            "commander_mode_constraint_reason": constraint_reason,
            "commander_head_on_post_merge_reopened_crossing_leash_active": bool(
                leash_state.get("active", False)
            ),
            "commander_head_on_post_merge_reopened_crossing_leash_range_m": float(
                leash_state.get("range_m", np.nan)
            ),
            "commander_head_on_post_merge_reopened_crossing_leash_hp_advantage": float(
                leash_state.get("hp_advantage", np.nan)
            ),
            "commander_head_on_post_merge_reopened_crossing_leash_consecutive_crossing_macro_steps": int(
                self._head_on_post_merge_reopened_crossing_consecutive_macro_steps
            ),
            "commander_head_on_post_merge_reopened_crossing_secondary_clamp_active": bool(
                secondary_clamp_state.get("active", False)
            ),
            "commander_head_on_post_merge_reopened_crossing_secondary_clamp_reason": (
                secondary_clamp_state.get("reason")
            ),
            "commander_head_on_post_merge_reopened_crossing_secondary_clamp_altitude_m": float(
                secondary_clamp_state.get("altitude_m", np.nan)
            ),
            "commander_head_on_post_merge_reopened_crossing_secondary_clamp_vp_lateral_bias_m": float(
                secondary_clamp_state.get("vp_lateral_bias_m", np.nan)
            ),
            "commander_head_on_post_merge_reopened_crossing_secondary_clamp_vp_lateral_to_range_ratio": float(
                secondary_clamp_state.get("vp_lateral_to_range_ratio", np.nan)
            ),
            "commander_head_on_post_merge_reopened_crossing_secondary_clamp_altitude_drop_m_lookback": float(
                secondary_clamp_state.get("altitude_drop_m_lookback", np.nan)
            ),
            "commander_head_on_post_merge_reopened_crossing_secondary_clamp_altitude_drop_lookback_steps": int(
                secondary_clamp_state.get("altitude_drop_lookback_steps", 0)
            ),
            "commander_head_on_post_merge_reopened_crossing_overdeep_clamp_active": bool(
                overdeep_clamp_state.get("active", False)
            ),
            "commander_head_on_post_merge_reopened_crossing_overdeep_clamp_reason": (
                overdeep_clamp_state.get("reason")
            ),
            "commander_head_on_post_merge_reopened_crossing_overdeep_clamp_range_m": float(
                overdeep_clamp_state.get("range_m", np.nan)
            ),
            "commander_head_on_post_merge_reopened_crossing_overdeep_clamp_vp_forward_bias_m": float(
                overdeep_clamp_state.get("vp_forward_bias_m", np.nan)
            ),
            "commander_head_on_post_merge_reopened_crossing_overdeep_clamp_vp_lateral_to_range_ratio": float(
                overdeep_clamp_state.get("vp_lateral_to_range_ratio", np.nan)
            ),
            "commander_mode_switched": bool(switched),
            "commander_first_switch_step": self._first_switch_step,
            "commander_macro_steps_since_switch": int(self._macro_steps_since_switch),
            "commander_hold_steps_remaining": int(self._hold_steps_remaining),
        }
        self._remember_secondary_clamp_altitude(
            float(secondary_clamp_state.get("altitude_m", np.nan))
        )
        return action

    def get_last_step_metadata(self) -> Dict[str, Any]:
        return dict(self._last_step_metadata)

    def load(self, path: str) -> None:
        del path
