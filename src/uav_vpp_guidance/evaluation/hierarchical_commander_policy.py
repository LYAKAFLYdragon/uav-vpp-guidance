"""Evaluation-time learned hierarchical commander wrapper."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

from uav_vpp_guidance.agents.commander_double_dqn_agent import CommanderDoubleDQNAgent
from uav_vpp_guidance.agents.commander_ppo_agent import CommanderPPOAgent
from uav_vpp_guidance.hierarchy.commander_mode_constraints import (
    apply_head_on_post_merge_recovery_hold,
    apply_head_on_post_merge_reopened_crossing_geometry_quality_guard,
    apply_head_on_post_merge_reopened_crossing_overdeep_clamp,
    apply_head_on_post_merge_reopened_crossing_secondary_clamp,
    apply_head_on_post_merge_reopened_crossing_target_threat_clamp,
    apply_head_on_post_merge_reopened_crossing_leash,
    build_head_on_post_merge_reopened_crossing_snapshot,
    evaluate_head_on_post_merge_recovery_hold_state,
    evaluate_head_on_post_merge_reopened_crossing_geometry_quality_guard_state,
    evaluate_head_on_post_merge_reopened_crossing_overdeep_clamp_state,
    evaluate_head_on_post_merge_reopened_crossing_secondary_clamp_state,
    evaluate_head_on_post_merge_reopened_crossing_target_threat_clamp_state,
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
        self.head_on_post_merge_reopened_crossing_target_threat_clamp = dict(
            commander_cfg.get(
                "head_on_post_merge_reopened_crossing_target_threat_clamp", {}
            )
            or {}
        )
        self.head_on_post_merge_reopened_crossing_overdeep_clamp = dict(
            commander_cfg.get(
                "head_on_post_merge_reopened_crossing_overdeep_clamp", {}
            )
            or {}
        )
        self.head_on_post_merge_reopened_crossing_geometry_quality_guard = dict(
            commander_cfg.get(
                "head_on_post_merge_reopened_crossing_geometry_quality_guard", {}
            )
            or {}
        )
        self.head_on_post_merge_first_recovery_entry_hold = dict(
            commander_cfg.get("head_on_post_merge_first_recovery_entry_hold", {}) or {}
        )
        self.head_on_post_merge_recovery_hold = dict(
            commander_cfg.get("head_on_post_merge_recovery_hold", {}) or {}
        )
        self.crossing_pre_merge_mode_lock = dict(
            commander_cfg.get("crossing_pre_merge_mode_lock", {}) or {}
        )
        self.mode_registry = load_frozen_specialist_registry(
            list(commander_cfg.get("modes", [])),
            device=device,
        )
        self.commander_action_dim = self._infer_commander_action_dim(
            self.checkpoint_path,
            fallback_action_dim=len(self.mode_registry),
        )
        if self.commander_action_dim > len(self.mode_registry):
            raise ValueError(
                "Commander checkpoint action_dim exceeds configured mode registry size: "
                f"checkpoint={self.commander_action_dim}, modes={len(self.mode_registry)}"
            )
        algorithm = str(commander_cfg.get("algorithm", "ppo")).lower()
        if algorithm == "double_dqn":
            self.commander = CommanderDoubleDQNAgent(
                obs_dim=int(obs_dim),
                action_dim=self.commander_action_dim,
                config=config,
                device=device,
            )
        else:
            self.commander = CommanderPPOAgent(
                obs_dim=int(obs_dim),
                action_dim=self.commander_action_dim,
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
        self._active_mode_entry_reason: Optional[str] = None
        self._step_counter = 0
        self._switch_count = 0
        self._steps_since_switch = 0
        self._first_switch_step: Optional[int] = None
        self._macro_steps_since_switch = 0
        self._hold_steps_remaining = 0
        self._head_on_post_merge_reopened_crossing_consecutive_macro_steps = 0
        self._head_on_post_merge_reopened_crossing_altitude_history_m = []
        self._head_on_post_merge_reopened_crossing_leash_active_steps = 0
        self._head_on_post_merge_reopened_crossing_overdeep_active_steps = 0
        self._head_on_post_merge_first_recovery_entry_hold_used = False
        self._head_on_post_merge_first_recovery_entry_hold_steps_remaining = 0
        self._head_on_post_merge_first_recovery_entry_hold_original_reason: Optional[
            str
        ] = None
        self._head_on_post_merge_recovery_mode_seen = False
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

    @staticmethod
    def _infer_commander_action_dim(
        checkpoint_path: str,
        *,
        fallback_action_dim: int,
    ) -> int:
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        except Exception:
            return int(fallback_action_dim)
        action_dim = checkpoint.get("action_dim")
        if action_dim is None:
            return int(fallback_action_dim)
        return max(1, int(action_dim))

    def _set_env_specialist_context(self, mode: Dict[str, Any]) -> None:
        if self.env is None or not hasattr(self.env, "set_runtime_specialist_context"):
            return
        self.env.set_runtime_specialist_context(
            specialist_key=mode.get("specialist_key"),
            specialist_profile=mode.get("specialist_profile"),
            specialist_mode_name=mode.get("name"),
            specialist_reason=self._active_mode_entry_reason,
        )

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

    def _update_active_mode(
        self,
        new_mode: int,
        *,
        entry_reason: Optional[str] = None,
    ) -> bool:
        new_mode = int(new_mode)
        previous_mode_id = self._active_mode_id
        switched = previous_mode_id is not None and new_mode != previous_mode_id
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
        if previous_mode_id is None or new_mode != previous_mode_id:
            self._active_mode_entry_reason = (
                None if entry_reason in (None, "") else str(entry_reason)
            )
        return switched

    def _remember_secondary_clamp_altitude(self, altitude_m: float) -> None:
        if not np.isfinite(altitude_m):
            return
        self._head_on_post_merge_reopened_crossing_altitude_history_m.append(
            float(altitude_m)
        )
        if len(self._head_on_post_merge_reopened_crossing_altitude_history_m) > 128:
            self._head_on_post_merge_reopened_crossing_altitude_history_m.pop(0)

    def _update_post_merge_guard_counters(
        self,
        *,
        commander_snapshot: Dict[str, Any],
        leash_state: Dict[str, Any],
        overdeep_clamp_state: Dict[str, Any],
    ) -> None:
        if not bool(commander_snapshot.get("first_pass_complete", False)):
            self._head_on_post_merge_reopened_crossing_leash_active_steps = 0
            self._head_on_post_merge_reopened_crossing_overdeep_active_steps = 0
            return

        if bool(leash_state.get("active", False)):
            self._head_on_post_merge_reopened_crossing_leash_active_steps += 1
        else:
            self._head_on_post_merge_reopened_crossing_leash_active_steps = 0

        if bool(overdeep_clamp_state.get("active", False)):
            self._head_on_post_merge_reopened_crossing_overdeep_active_steps += 1

    def _evaluate_head_on_first_recovery_entry_hold_state(
        self,
        *,
        commander_snapshot: Dict[str, Any],
        requested_mode_id: Optional[int],
        proposed_mode_id: Optional[int],
        original_reason: Optional[str],
    ) -> Dict[str, Any]:
        cfg = self.head_on_post_merge_first_recovery_entry_hold
        enabled = bool(cfg.get("enabled", False))
        state = {
            "configured": enabled,
            "active": False,
            "reason": "disabled" if not enabled else "inactive",
            "effective_mode_id": proposed_mode_id,
            "original_reason": (
                self._head_on_post_merge_first_recovery_entry_hold_original_reason
            ),
            "cooldown_steps_remaining": int(
                self._head_on_post_merge_first_recovery_entry_hold_steps_remaining
            ),
            "armed_this_step": False,
        }
        if not enabled:
            return state
        if not bool(commander_snapshot.get("available", False)):
            state["reason"] = str(commander_snapshot.get("reason", "unavailable"))
            return state
        if str(self.current_task_name) != str(cfg.get("task_name", "head_on")):
            state["reason"] = "task_mismatch"
            return state
        if not bool(commander_snapshot.get("first_pass_complete", False)):
            state["reason"] = "pre_merge"
            return state
        head_on_mode_id = int(cfg.get("head_on_mode_id", 0))
        recovery_mode_id = int(cfg.get("recovery_mode_id", 2))
        if requested_mode_id is None or int(requested_mode_id) != head_on_mode_id:
            state["reason"] = "requested_mode_not_head_on"
            return state
        if proposed_mode_id is None or int(proposed_mode_id) != recovery_mode_id:
            state["reason"] = "non_recovery_mode"
            return state
        if (
            self._active_mode_id is not None
            and int(self._active_mode_id) != head_on_mode_id
        ):
            state["reason"] = "active_mode_not_head_on"
            return state
        if self._head_on_post_merge_recovery_mode_seen:
            state["reason"] = "recovery_already_seen"
            return state

        allowed_reasons = [
            str(reason)
            for reason in (cfg.get("allowed_reasons", []) or [])
            if reason is not None
        ]
        normalized_original_reason = (
            str(original_reason) if original_reason is not None else None
        )
        if allowed_reasons and normalized_original_reason not in allowed_reasons:
            state["reason"] = "constraint_reason_not_eligible"
            return state
        hold_window_steps = int(cfg.get("hold_window_steps", 1) or 0)
        hold_window_steps_by_reason = (
            cfg.get("hold_window_steps_by_reason", {}) or {}
        )
        if (
            normalized_original_reason is not None
            and isinstance(hold_window_steps_by_reason, dict)
            and normalized_original_reason in hold_window_steps_by_reason
        ):
            hold_window_steps = int(
                hold_window_steps_by_reason.get(normalized_original_reason) or 0
            )
        hold_window_steps = max(0, hold_window_steps)
        if hold_window_steps <= 0:
            state["reason"] = "hold_window_disabled_for_reason"
            state["original_reason"] = normalized_original_reason
            return state

        if self._head_on_post_merge_first_recovery_entry_hold_steps_remaining > 0:
            state["active"] = True
            state["reason"] = "cooldown_active"
            state["effective_mode_id"] = head_on_mode_id
            state["original_reason"] = (
                self._head_on_post_merge_first_recovery_entry_hold_original_reason
            )
            state["cooldown_steps_remaining"] = int(
                self._head_on_post_merge_first_recovery_entry_hold_steps_remaining
            )
            return state
        if self._head_on_post_merge_first_recovery_entry_hold_used:
            state["reason"] = "already_used"
            return state
        state["active"] = True
        state["reason"] = "armed"
        state["effective_mode_id"] = head_on_mode_id
        state["original_reason"] = normalized_original_reason
        state["cooldown_steps_remaining"] = hold_window_steps
        state["armed_this_step"] = True
        self._head_on_post_merge_first_recovery_entry_hold_used = True
        self._head_on_post_merge_first_recovery_entry_hold_steps_remaining = (
            hold_window_steps
        )
        self._head_on_post_merge_first_recovery_entry_hold_original_reason = (
            normalized_original_reason
        )
        return state

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
        target_threat_clamp_state = (
            evaluate_head_on_post_merge_reopened_crossing_target_threat_clamp_state(
                snapshot=commander_snapshot,
                target_threat_cfg=(
                    self.head_on_post_merge_reopened_crossing_target_threat_clamp
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
        self._update_post_merge_guard_counters(
            commander_snapshot=commander_snapshot,
            leash_state=leash_state,
            overdeep_clamp_state=overdeep_clamp_state,
        )
        geometry_quality_guard_state = (
            evaluate_head_on_post_merge_reopened_crossing_geometry_quality_guard_state(
                snapshot=commander_snapshot,
                altitude_history_m=(
                    self._head_on_post_merge_reopened_crossing_altitude_history_m
                ),
                leash_active_steps=(
                    self._head_on_post_merge_reopened_crossing_leash_active_steps
                ),
                overdeep_active_steps=(
                    self._head_on_post_merge_reopened_crossing_overdeep_active_steps
                ),
                geometry_guard_cfg=(
                    self.head_on_post_merge_reopened_crossing_geometry_quality_guard
                ),
                leash_state=leash_state,
                overdeep_state=overdeep_clamp_state,
            )
        )
        recovery_hold_state = evaluate_head_on_post_merge_recovery_hold_state(
            snapshot=commander_snapshot,
            recovery_hold_cfg=self.head_on_post_merge_recovery_hold,
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
        first_recovery_entry_hold_state = {
            "configured": bool(
                self.head_on_post_merge_first_recovery_entry_hold.get("enabled", False)
            ),
            "active": False,
            "reason": "disabled"
            if not bool(
                self.head_on_post_merge_first_recovery_entry_hold.get("enabled", False)
            )
            else "inactive",
            "effective_mode_id": None,
            "original_reason": None,
        }
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
            target_threat_result = (
                apply_head_on_post_merge_reopened_crossing_target_threat_clamp(
                    candidate_mode_id=new_mode,
                    mode_registry=self.mode_registry,
                    target_threat_state=target_threat_clamp_state,
                    target_threat_cfg=(
                        self.head_on_post_merge_reopened_crossing_target_threat_clamp
                    ),
                    head_on_recovery_mode_seen=(
                        self._head_on_post_merge_recovery_mode_seen
                    ),
                )
            )
            if bool(target_threat_result["triggered"]):
                new_mode = int(target_threat_result["effective_mode_id"])
                constraint_triggered = True
                constraint_reason = str(target_threat_result["reason"])
                self._head_on_post_merge_reopened_crossing_consecutive_macro_steps = 0
            overdeep_result = (
                apply_head_on_post_merge_reopened_crossing_overdeep_clamp(
                    candidate_mode_id=new_mode,
                    mode_registry=self.mode_registry,
                    overdeep_state=overdeep_clamp_state,
                    overdeep_cfg=(
                        self.head_on_post_merge_reopened_crossing_overdeep_clamp
                    ),
                    head_on_recovery_mode_seen=(
                        self._head_on_post_merge_recovery_mode_seen
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
            geometry_quality_result = (
                apply_head_on_post_merge_reopened_crossing_geometry_quality_guard(
                    candidate_mode_id=new_mode,
                    mode_registry=self.mode_registry,
                    geometry_guard_state=geometry_quality_guard_state,
                    geometry_guard_cfg=(
                        self.head_on_post_merge_reopened_crossing_geometry_quality_guard
                    ),
                )
            )
            if bool(geometry_quality_result["triggered"]):
                new_mode = int(geometry_quality_result["effective_mode_id"])
                constraint_triggered = True
                constraint_reason = str(geometry_quality_result["reason"])
                self._head_on_post_merge_reopened_crossing_consecutive_macro_steps = 0
            recovery_hold_result = apply_head_on_post_merge_recovery_hold(
                candidate_mode_id=new_mode,
                active_mode_id=self._active_mode_id,
                mode_registry=self.mode_registry,
                recovery_hold_state=recovery_hold_state,
                recovery_hold_cfg=self.head_on_post_merge_recovery_hold,
            )
            if bool(recovery_hold_result["triggered"]):
                new_mode = int(recovery_hold_result["effective_mode_id"])
                constraint_triggered = True
                constraint_reason = str(recovery_hold_result["reason"])
                self._head_on_post_merge_reopened_crossing_consecutive_macro_steps = 0
            first_recovery_entry_hold_state = (
                self._evaluate_head_on_first_recovery_entry_hold_state(
                    commander_snapshot=commander_snapshot,
                    requested_mode_id=requested_mode_id,
                    proposed_mode_id=new_mode,
                    original_reason=constraint_reason,
                )
            )
            if bool(first_recovery_entry_hold_state["active"]):
                new_mode = int(first_recovery_entry_hold_state["effective_mode_id"])
                constraint_triggered = True
                constraint_reason = "head_on_first_recovery_entry_hold"
                self._head_on_post_merge_reopened_crossing_consecutive_macro_steps = 0
            switched = self._update_active_mode(
                new_mode,
                entry_reason=constraint_reason,
            )
        else:
            active_mode_name = self._mode_name(self._active_mode_id)
            candidate_mode_id = (
                int(self._active_mode_id) if self._active_mode_id is not None else None
            )
            target_threat_result = (
                apply_head_on_post_merge_reopened_crossing_target_threat_clamp(
                    candidate_mode_id=self._active_mode_id,
                    mode_registry=self.mode_registry,
                    target_threat_state=target_threat_clamp_state,
                    target_threat_cfg=(
                        self.head_on_post_merge_reopened_crossing_target_threat_clamp
                    ),
                    head_on_recovery_mode_seen=(
                        self._head_on_post_merge_recovery_mode_seen
                    ),
                )
            )
            overdeep_result = apply_head_on_post_merge_reopened_crossing_overdeep_clamp(
                candidate_mode_id=self._active_mode_id,
                mode_registry=self.mode_registry,
                overdeep_state=overdeep_clamp_state,
                overdeep_cfg=self.head_on_post_merge_reopened_crossing_overdeep_clamp,
                head_on_recovery_mode_seen=self._head_on_post_merge_recovery_mode_seen,
            )
            secondary_result = apply_head_on_post_merge_reopened_crossing_secondary_clamp(
                candidate_mode_id=self._active_mode_id,
                mode_registry=self.mode_registry,
                secondary_state=secondary_clamp_state,
                secondary_cfg=self.head_on_post_merge_reopened_crossing_secondary_clamp,
            )
            geometry_quality_result = (
                apply_head_on_post_merge_reopened_crossing_geometry_quality_guard(
                    candidate_mode_id=self._active_mode_id,
                    mode_registry=self.mode_registry,
                    geometry_guard_state=geometry_quality_guard_state,
                    geometry_guard_cfg=(
                        self.head_on_post_merge_reopened_crossing_geometry_quality_guard
                    ),
                )
            )
            if bool(target_threat_result["triggered"]):
                requested_mode_name = active_mode_name
                constraint_triggered = True
                constraint_reason = str(target_threat_result["reason"])
                self._head_on_post_merge_reopened_crossing_consecutive_macro_steps = 0
                candidate_mode_id = int(target_threat_result["effective_mode_id"])
            elif bool(overdeep_result["triggered"]):
                requested_mode_name = active_mode_name
                constraint_triggered = True
                constraint_reason = str(overdeep_result["reason"])
                self._head_on_post_merge_reopened_crossing_consecutive_macro_steps = 0
                candidate_mode_id = int(overdeep_result["effective_mode_id"])
            elif bool(secondary_result["triggered"]):
                requested_mode_name = active_mode_name
                constraint_triggered = True
                constraint_reason = str(secondary_result["reason"])
                self._head_on_post_merge_reopened_crossing_consecutive_macro_steps = 0
                candidate_mode_id = int(secondary_result["effective_mode_id"])
            elif bool(geometry_quality_result["triggered"]):
                requested_mode_name = active_mode_name
                constraint_triggered = True
                constraint_reason = str(geometry_quality_result["reason"])
                self._head_on_post_merge_reopened_crossing_consecutive_macro_steps = 0
                candidate_mode_id = int(geometry_quality_result["effective_mode_id"])
            if constraint_triggered and candidate_mode_id is not None:
                first_recovery_entry_hold_state = (
                    self._evaluate_head_on_first_recovery_entry_hold_state(
                        commander_snapshot=commander_snapshot,
                        requested_mode_id=self._active_mode_id,
                        proposed_mode_id=candidate_mode_id,
                        original_reason=constraint_reason,
                    )
                )
                if bool(first_recovery_entry_hold_state["active"]):
                    candidate_mode_id = int(
                        first_recovery_entry_hold_state["effective_mode_id"]
                    )
                    constraint_reason = "head_on_first_recovery_entry_hold"
                switched = self._update_active_mode(
                    candidate_mode_id,
                    entry_reason=constraint_reason,
                )
            elif (
                not leash_state.get("active", False)
                or active_mode_name != "crossing_specialist"
            ):
                self._head_on_post_merge_reopened_crossing_consecutive_macro_steps = 0

        if self._active_mode_id is None:
            raise RuntimeError("HierarchicalCommanderPolicy failed to select a mode")

        mode = self.mode_registry[self._active_mode_id]
        if mode.get("specialist_key") == "post_merge_recovery":
            self._head_on_post_merge_recovery_mode_seen = True
        self._set_env_specialist_context(mode)
        action = mode["policy"].get_deterministic_action(obs)
        self._step_counter += 1
        self._steps_since_switch += 1
        self._last_step_metadata = {
            "commander_mode_id": int(self._active_mode_id),
            "commander_mode_name": mode["name"],
            "commander_selected_specialist": mode["specialist_key"],
            "commander_selected_specialist_profile": mode.get("specialist_profile"),
            "commander_selected_source_specialist": mode.get(
                "source_specialist_key"
            ),
            "commander_active_mode_entry_reason": self._active_mode_entry_reason,
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
            "commander_head_on_post_merge_reopened_crossing_target_threat_clamp_active": bool(
                target_threat_clamp_state.get("active", False)
            ),
            "commander_head_on_post_merge_reopened_crossing_target_threat_clamp_reason": (
                target_threat_clamp_state.get("reason")
            ),
            "commander_head_on_post_merge_reopened_crossing_target_threat_clamp_target_in_attack_zone": bool(
                target_threat_clamp_state.get("target_in_attack_zone", False)
            ),
            "commander_head_on_post_merge_reopened_crossing_target_threat_clamp_range_m": float(
                target_threat_clamp_state.get("range_m", np.nan)
            ),
            "commander_head_on_post_merge_reopened_crossing_target_threat_clamp_range_rate_mps": float(
                target_threat_clamp_state.get("range_rate_mps", np.nan)
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
            "commander_head_on_post_merge_reopened_crossing_overdeep_clamp_min_range_so_far_m": float(
                overdeep_clamp_state.get("min_range_so_far_m", np.nan)
            ),
            "commander_head_on_post_merge_reopened_crossing_overdeep_clamp_vp_forward_bias_m": float(
                overdeep_clamp_state.get("vp_forward_bias_m", np.nan)
            ),
            "commander_head_on_post_merge_reopened_crossing_overdeep_clamp_vp_lateral_to_range_ratio": float(
                overdeep_clamp_state.get("vp_lateral_to_range_ratio", np.nan)
            ),
            "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_active": bool(
                geometry_quality_guard_state.get("active", False)
            ),
            "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_reason": (
                geometry_quality_guard_state.get("reason")
            ),
            "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_altitude_m": float(
                geometry_quality_guard_state.get("altitude_m", np.nan)
            ),
            "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_altitude_delta_m_lookback": float(
                geometry_quality_guard_state.get(
                    "altitude_delta_m_lookback", np.nan
                )
            ),
            "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_altitude_trend_lookback_steps": int(
                geometry_quality_guard_state.get(
                    "altitude_trend_lookback_steps", 0
                )
            ),
            "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_vp_forward_bias_m": float(
                geometry_quality_guard_state.get("vp_forward_bias_m", np.nan)
            ),
            "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_vp_lateral_bias_m": float(
                geometry_quality_guard_state.get("vp_lateral_bias_m", np.nan)
            ),
            "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_vp_lateral_to_range_ratio": float(
                geometry_quality_guard_state.get(
                    "vp_lateral_to_range_ratio", np.nan
                )
            ),
            "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_leash_active_steps": int(
                geometry_quality_guard_state.get("leash_active_steps", 0)
            ),
            "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_overdeep_active_steps": int(
                geometry_quality_guard_state.get("overdeep_active_steps", 0)
            ),
            "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_overdeep_seen_since_post_merge": bool(
                geometry_quality_guard_state.get(
                    "overdeep_seen_since_post_merge", False
                )
            ),
            "commander_head_on_post_merge_recovery_hold_active": bool(
                recovery_hold_state.get("active", False)
            ),
            "commander_head_on_post_merge_recovery_hold_reason": recovery_hold_state.get(
                "reason"
            ),
            "commander_head_on_post_merge_recovery_hold_range_m": float(
                recovery_hold_state.get("range_m", np.nan)
            ),
            "commander_head_on_post_merge_recovery_hold_vp_lateral_bias_m": float(
                recovery_hold_state.get("vp_lateral_bias_m", np.nan)
            ),
            "commander_head_on_post_merge_recovery_hold_vp_lateral_to_range_ratio": float(
                recovery_hold_state.get("vp_lateral_to_range_ratio", np.nan)
            ),
            "commander_head_on_post_merge_first_recovery_entry_hold_active": bool(
                first_recovery_entry_hold_state.get("active", False)
            ),
            "commander_head_on_post_merge_first_recovery_entry_hold_reason": (
                first_recovery_entry_hold_state.get("reason")
            ),
            "commander_head_on_post_merge_first_recovery_entry_hold_original_reason": (
                first_recovery_entry_hold_state.get("original_reason")
            ),
            "commander_head_on_post_merge_first_recovery_entry_hold_armed_this_step": bool(
                first_recovery_entry_hold_state.get("armed_this_step", False)
            ),
            "commander_head_on_post_merge_first_recovery_entry_hold_cooldown_steps_remaining": int(
                first_recovery_entry_hold_state.get(
                    "cooldown_steps_remaining",
                    self._head_on_post_merge_first_recovery_entry_hold_steps_remaining,
                )
            ),
            "commander_mode_switched": bool(switched),
            "commander_first_switch_step": self._first_switch_step,
            "commander_macro_steps_since_switch": int(self._macro_steps_since_switch),
            "commander_hold_steps_remaining": int(self._hold_steps_remaining),
        }
        self._remember_secondary_clamp_altitude(
            float(secondary_clamp_state.get("altitude_m", np.nan))
        )
        if self._head_on_post_merge_first_recovery_entry_hold_steps_remaining > 0:
            self._head_on_post_merge_first_recovery_entry_hold_steps_remaining -= 1
        return action

    def get_last_step_metadata(self) -> Dict[str, Any]:
        return dict(self._last_step_metadata)

    def load(self, path: str) -> None:
        del path
