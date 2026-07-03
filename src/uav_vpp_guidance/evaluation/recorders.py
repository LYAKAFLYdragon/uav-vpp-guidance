"""
Independent JSON recorders for the flight-control comparison benchmark.

Recording logic lives here so that the training/evaluation runner does not
need to inline JSON formatting or trajectory bookkeeping.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .flight_control_metrics import (
    compute_break_turn_metrics,
    compute_multi_waypoint_metrics,
    compute_sustained_turn_metrics,
)


def _tolist(arr):
    if arr is None:
        return None
    if hasattr(arr, "tolist"):
        return arr.tolist()
    return list(arr)


def _vec3(value):
    if value is None:
        return [np.nan, np.nan, np.nan]
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return [np.nan, np.nan, np.nan]
    if arr.shape[0] < 3:
        return [np.nan, np.nan, np.nan]
    return [float(arr[0]), float(arr[1]), float(arr[2])]


class EpisodeRecorder:
    """
    Records one episode for the flight-control comparison benchmark.

    The produced JSON schema contains the per-step trajectory, task-specific
    statistics, and provenance required by ``export_table3.py`` and
    ``plot_flight_control_figures.py``.
    """

    def __init__(
        self,
        run_id: str,
        task: str,
        controller: str,
        seed: int,
        episode: int,
        config: dict,
        config_sha256: str,
        git_commit: str,
        backend: str = "jsbsim",
        strict_backend: bool = True,
        save_full: bool = True,
    ):
        self.run_id = run_id
        self.task = task
        self.controller = controller
        self.seed = seed
        self.episode = episode
        self.config = config
        self.config_sha256 = config_sha256
        self.git_commit = git_commit
        self.backend = backend
        self.strict_backend = strict_backend
        self.save_full = save_full

        self.trajectory: List[Dict[str, Any]] = []
        self._prev_active_idx = 0

    def record_step(
        self,
        step: int,
        time_s: float,
        own_state: dict,
        target_state: dict,
        info: dict,
        reward: float,
    ) -> None:
        """Append one step to the trajectory."""
        if not self.save_full:
            return

        current_active_idx = info.get("active_waypoint_index", self._prev_active_idx)
        switch_event = current_active_idx != self._prev_active_idx
        self._prev_active_idx = current_active_idx
        own_pos = _vec3(own_state.get("position_m", own_state.get("position_neu")))
        target_pos = _vec3(
            target_state.get("position_m", target_state.get("position_neu"))
        )
        virtual_point = info.get("virtual_point") or {}
        vp_source = info.get("vp_position_neu")
        if vp_source is None:
            vp_source = virtual_point.get("position_neu")
        if vp_source is None:
            vp_source = virtual_point.get("position")
        vp_pos = _vec3(vp_source)
        anchor_pos = _vec3(info.get("anchor_pos"))
        offset = _vec3(info.get("vp_offset"))
        world_offset = _vec3(info.get("vp_world_offset"))
        tactical_basis_ll_world = _vec3(info.get("tactical_basis_ll_world"))
        tactical_basis_io_world = _vec3(info.get("tactical_basis_io_world"))
        tactical_basis_cd_world = _vec3(info.get("tactical_basis_cd_world"))
        tactical_basis_world_offset = _vec3(info.get("tactical_basis_world_offset"))
        offensive_anchor_lateral_world_offset = _vec3(
            info.get("offensive_anchor_lateral_world_offset")
        )

        point = {
            "step": step,
            "time_s": float(time_s),
            "task": self.task,
            "opponent_stage": info.get("opponent_stage"),
            "method": self.controller,
            "controller": self.controller,
            "seed": self.seed,
            "episode": self.episode,
            "commander_mode_id": info.get("commander_mode_id"),
            "commander_mode_name": info.get("commander_mode_name"),
            "commander_selected_specialist": info.get(
                "commander_selected_specialist"
            ),
            "commander_selected_specialist_profile": info.get(
                "commander_selected_specialist_profile"
            ),
            "commander_selected_source_specialist": info.get(
                "commander_selected_source_specialist"
            ),
            "commander_switch_count": info.get("commander_switch_count"),
            "commander_steps_since_switch": info.get(
                "commander_steps_since_switch"
            ),
            "commander_macro_action_repeat_steps": info.get(
                "commander_macro_action_repeat_steps"
            ),
            "commander_task_oracle_gate": info.get("commander_task_oracle_gate"),
            "commander_requested_mode_id": info.get("commander_requested_mode_id"),
            "commander_requested_mode_name": info.get(
                "commander_requested_mode_name"
            ),
            "commander_crossing_pre_merge_mode_lock_active": bool(
                info.get("commander_crossing_pre_merge_mode_lock_active", False)
            ),
            "commander_mode_constraint_triggered": bool(
                info.get("commander_mode_constraint_triggered", False)
            ),
            "commander_mode_constraint_reason": info.get(
                "commander_mode_constraint_reason"
            ),
            "commander_head_on_post_merge_reopened_crossing_leash_active": bool(
                info.get(
                    "commander_head_on_post_merge_reopened_crossing_leash_active",
                    False,
                )
            ),
            "commander_head_on_post_merge_reopened_crossing_leash_range_m": float(
                info.get(
                    "commander_head_on_post_merge_reopened_crossing_leash_range_m",
                    np.nan,
                )
            ),
            "commander_head_on_post_merge_reopened_crossing_leash_hp_advantage": float(
                info.get(
                    "commander_head_on_post_merge_reopened_crossing_leash_hp_advantage",
                    np.nan,
                )
            ),
            "commander_head_on_post_merge_reopened_crossing_leash_consecutive_crossing_macro_steps": int(
                info.get(
                    "commander_head_on_post_merge_reopened_crossing_leash_consecutive_crossing_macro_steps",
                    0,
                )
            ),
            "commander_head_on_post_merge_reopened_crossing_secondary_clamp_active": bool(
                info.get(
                    "commander_head_on_post_merge_reopened_crossing_secondary_clamp_active",
                    False,
                )
            ),
            "commander_head_on_post_merge_reopened_crossing_secondary_clamp_reason": info.get(
                "commander_head_on_post_merge_reopened_crossing_secondary_clamp_reason"
            ),
            "commander_head_on_post_merge_reopened_crossing_secondary_clamp_altitude_m": float(
                info.get(
                    "commander_head_on_post_merge_reopened_crossing_secondary_clamp_altitude_m",
                    np.nan,
                )
            ),
            "commander_head_on_post_merge_reopened_crossing_secondary_clamp_vp_lateral_bias_m": float(
                info.get(
                    "commander_head_on_post_merge_reopened_crossing_secondary_clamp_vp_lateral_bias_m",
                    np.nan,
                )
            ),
            "commander_head_on_post_merge_reopened_crossing_secondary_clamp_vp_lateral_to_range_ratio": float(
                info.get(
                    "commander_head_on_post_merge_reopened_crossing_secondary_clamp_vp_lateral_to_range_ratio",
                    np.nan,
                )
            ),
            "commander_head_on_post_merge_reopened_crossing_secondary_clamp_altitude_drop_m_lookback": float(
                info.get(
                    "commander_head_on_post_merge_reopened_crossing_secondary_clamp_altitude_drop_m_lookback",
                    np.nan,
                )
            ),
            "commander_head_on_post_merge_reopened_crossing_secondary_clamp_altitude_drop_lookback_steps": int(
                info.get(
                    "commander_head_on_post_merge_reopened_crossing_secondary_clamp_altitude_drop_lookback_steps",
                    0,
                )
            ),
            "commander_head_on_post_merge_reopened_crossing_target_threat_clamp_active": bool(
                info.get(
                    "commander_head_on_post_merge_reopened_crossing_target_threat_clamp_active",
                    False,
                )
            ),
            "commander_head_on_post_merge_reopened_crossing_target_threat_clamp_reason": info.get(
                "commander_head_on_post_merge_reopened_crossing_target_threat_clamp_reason"
            ),
            "commander_head_on_post_merge_reopened_crossing_target_threat_clamp_target_in_attack_zone": bool(
                info.get(
                    "commander_head_on_post_merge_reopened_crossing_target_threat_clamp_target_in_attack_zone",
                    False,
                )
            ),
            "commander_head_on_post_merge_reopened_crossing_target_threat_clamp_range_m": float(
                info.get(
                    "commander_head_on_post_merge_reopened_crossing_target_threat_clamp_range_m",
                    np.nan,
                )
            ),
            "commander_head_on_post_merge_reopened_crossing_target_threat_clamp_range_rate_mps": float(
                info.get(
                    "commander_head_on_post_merge_reopened_crossing_target_threat_clamp_range_rate_mps",
                    np.nan,
                )
            ),
            "commander_head_on_post_merge_reopened_crossing_overdeep_clamp_active": bool(
                info.get(
                    "commander_head_on_post_merge_reopened_crossing_overdeep_clamp_active",
                    False,
                )
            ),
            "commander_head_on_post_merge_reopened_crossing_overdeep_clamp_reason": info.get(
                "commander_head_on_post_merge_reopened_crossing_overdeep_clamp_reason"
            ),
            "commander_head_on_post_merge_reopened_crossing_overdeep_clamp_range_m": float(
                info.get(
                    "commander_head_on_post_merge_reopened_crossing_overdeep_clamp_range_m",
                    np.nan,
                )
            ),
            "commander_head_on_post_merge_reopened_crossing_overdeep_clamp_vp_forward_bias_m": float(
                info.get(
                    "commander_head_on_post_merge_reopened_crossing_overdeep_clamp_vp_forward_bias_m",
                    np.nan,
                )
            ),
            "commander_head_on_post_merge_reopened_crossing_overdeep_clamp_vp_lateral_to_range_ratio": float(
                info.get(
                    "commander_head_on_post_merge_reopened_crossing_overdeep_clamp_vp_lateral_to_range_ratio",
                    np.nan,
                )
            ),
            "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_active": bool(
                info.get(
                    "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_active",
                    False,
                )
            ),
            "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_reason": info.get(
                "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_reason"
            ),
            "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_altitude_m": float(
                info.get(
                    "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_altitude_m",
                    np.nan,
                )
            ),
            "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_altitude_delta_m_lookback": float(
                info.get(
                    "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_altitude_delta_m_lookback",
                    np.nan,
                )
            ),
            "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_altitude_trend_lookback_steps": int(
                info.get(
                    "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_altitude_trend_lookback_steps",
                    0,
                )
            ),
            "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_vp_forward_bias_m": float(
                info.get(
                    "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_vp_forward_bias_m",
                    np.nan,
                )
            ),
            "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_vp_lateral_bias_m": float(
                info.get(
                    "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_vp_lateral_bias_m",
                    np.nan,
                )
            ),
            "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_vp_lateral_to_range_ratio": float(
                info.get(
                    "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_vp_lateral_to_range_ratio",
                    np.nan,
                )
            ),
            "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_leash_active_steps": int(
                info.get(
                    "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_leash_active_steps",
                    0,
                )
            ),
            "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_overdeep_active_steps": int(
                info.get(
                    "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_overdeep_active_steps",
                    0,
                )
            ),
            "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_overdeep_seen_since_post_merge": bool(
                info.get(
                    "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_overdeep_seen_since_post_merge",
                    False,
                )
            ),
            "commander_mode_switched": info.get("commander_mode_switched"),
            "commander_macro_step_index": info.get("commander_macro_step_index"),
            "commander_first_switch_step": info.get("commander_first_switch_step"),
            "own_pos_m": own_pos,
            "target_pos_m": target_pos,
            "ego_pos_x": own_pos[0],
            "ego_pos_y": own_pos[1],
            "ego_pos_z": own_pos[2],
            "target_pos_x": target_pos[0],
            "target_pos_y": target_pos[1],
            "target_pos_z": target_pos[2],
            "vp_pos_x": vp_pos[0],
            "vp_pos_y": vp_pos[1],
            "vp_pos_z": vp_pos[2],
            "anchor_pos_x": anchor_pos[0],
            "anchor_pos_y": anchor_pos[1],
            "anchor_pos_z": anchor_pos[2],
            "anchor_mode": info.get("anchor_mode"),
            "anchor_mode_requested": info.get("anchor_mode_requested"),
            "close_range_anchor_mode": info.get("close_range_anchor_mode"),
            "close_range_anchor_trigger_range_m": float(
                info.get("close_range_anchor_trigger_range_m", np.nan)
            ),
            "close_range_anchor_release_on_post_merge": bool(
                info.get("close_range_anchor_release_on_post_merge", False)
            ),
            "close_range_anchor_requires_first_pass": bool(
                info.get("close_range_anchor_requires_first_pass", False)
            ),
            "close_range_anchor_offensive_anchor_blend": float(
                info.get("close_range_anchor_offensive_anchor_blend", np.nan)
            ),
            "close_range_anchor_offensive_anchor_blend_active": bool(
                info.get("close_range_anchor_offensive_anchor_blend_active", False)
            ),
            "close_range_anchor_release_alignment_angle_deg_max": float(
                info.get("close_range_anchor_release_alignment_angle_deg_max", np.nan)
            ),
            "close_range_anchor_release_alignment_satisfied": bool(
                info.get("close_range_anchor_release_alignment_satisfied", True)
            ),
            "close_range_anchor_release_ready": bool(
                info.get("close_range_anchor_release_ready", False)
            ),
            "close_range_anchor_post_merge_hold_steps": float(
                info.get("close_range_anchor_post_merge_hold_steps", np.nan)
            ),
            "close_range_anchor_post_merge_steps_since_first_pass": float(
                info.get(
                    "close_range_anchor_post_merge_steps_since_first_pass",
                    np.nan,
                )
            ),
            "close_range_anchor_post_merge_hold_remaining_steps": float(
                info.get(
                    "close_range_anchor_post_merge_hold_remaining_steps",
                    np.nan,
                )
            ),
            "close_range_anchor_window_open": bool(
                info.get("close_range_anchor_window_open", True)
            ),
            "close_range_anchor_alignment_angle_deg_max": float(
                info.get("close_range_anchor_alignment_angle_deg_max", np.nan)
            ),
            "close_range_anchor_alignment_satisfied": bool(
                info.get("close_range_anchor_alignment_satisfied", False)
            ),
            "close_range_anchor_mode_active": bool(
                info.get("close_range_anchor_mode_active", False)
            ),
            "post_merge_anchor_mode": info.get("post_merge_anchor_mode"),
            "post_merge_anchor_mode_requires_geometry_disadvantage": bool(
                info.get(
                    "post_merge_anchor_mode_requires_geometry_disadvantage",
                    False,
                )
            ),
            "post_merge_anchor_mode_condition_met": bool(
                info.get("post_merge_anchor_mode_condition_met", False)
            ),
            "post_merge_anchor_mode_active": bool(
                info.get("post_merge_anchor_mode_active", False)
            ),
            "post_merge_anchor_mode_recovery_active": bool(
                info.get("post_merge_anchor_mode_recovery_active", False)
            ),
            "post_merge_anchor_mode_release_ego_only_streak_steps": float(
                info.get("post_merge_anchor_mode_release_ego_only_streak_steps", np.nan)
            ),
            "post_merge_anchor_mode_recovery_below_altitude_m": float(
                info.get("post_merge_anchor_mode_recovery_below_altitude_m", np.nan)
            ),
            "post_merge_anchor_mode_release_reset_on_streak_break": bool(
                info.get("post_merge_anchor_mode_release_reset_on_streak_break", False)
            ),
            "post_merge_anchor_mode_offensive_anchor_blend": float(
                info.get("post_merge_anchor_mode_offensive_anchor_blend", np.nan)
            ),
            "post_merge_anchor_mode_offensive_anchor_longitudinal_blend": float(
                info.get(
                    "post_merge_anchor_mode_offensive_anchor_longitudinal_blend",
                    np.nan,
                )
            ),
            "post_merge_anchor_mode_offensive_anchor_lateral_blend": float(
                info.get(
                    "post_merge_anchor_mode_offensive_anchor_lateral_blend",
                    np.nan,
                )
            ),
            "post_merge_anchor_mode_offensive_anchor_blend_active": bool(
                info.get("post_merge_anchor_mode_offensive_anchor_blend_active", False)
            ),
            "post_merge_anchor_mode_offensive_anchor_component_blend_active": bool(
                info.get(
                    "post_merge_anchor_mode_offensive_anchor_component_blend_active",
                    False,
                )
            ),
            "post_merge_anchor_mode_lateral_world_offset_latch_on_activation": bool(
                info.get(
                    "post_merge_anchor_mode_lateral_world_offset_latch_on_activation",
                    False,
                )
            ),
            "post_merge_anchor_mode_lateral_world_offset_latch_active": bool(
                info.get(
                    "post_merge_anchor_mode_lateral_world_offset_latch_active",
                    False,
                )
            ),
            "offset_frame": info.get("offset_frame", "world_neu"),
            "configured_offset_frame": info.get(
                "configured_offset_frame",
                info.get("offset_frame", "world_neu"),
            ),
            "action_semantics": info.get("action_semantics", "cartesian_offset"),
            "configured_action_semantics": info.get(
                "configured_action_semantics",
                info.get("action_semantics", "cartesian_offset"),
            ),
            "tactical_basis_enabled": bool(
                info.get("tactical_basis_enabled", False)
            ),
            "tactical_basis_action_ll": float(
                info.get("tactical_basis_action_ll", np.nan)
            ),
            "tactical_basis_action_io": float(
                info.get("tactical_basis_action_io", np.nan)
            ),
            "tactical_basis_action_cd": float(
                info.get("tactical_basis_action_cd", np.nan)
            ),
            "tactical_basis_lead_lag_extent_m": float(
                info.get("tactical_basis_lead_lag_extent_m", np.nan)
            ),
            "tactical_basis_inside_outside_extent_m": float(
                info.get("tactical_basis_inside_outside_extent_m", np.nan)
            ),
            "tactical_basis_climb_descent_extent_m": float(
                info.get("tactical_basis_climb_descent_extent_m", np.nan)
            ),
            "tactical_basis_longitudinal_frame": info.get(
                "tactical_basis_longitudinal_frame",
                "target_velocity",
            ),
            "tactical_basis_lateral_frame": info.get(
                "tactical_basis_lateral_frame",
                "encounter_stable",
            ),
            "tactical_basis_vertical_frame": info.get(
                "tactical_basis_vertical_frame",
                "world_neu",
            ),
            "tactical_basis_lateral_sign_mode": info.get(
                "tactical_basis_lateral_sign_mode",
                "same_side",
            ),
            "tactical_basis_lateral_sign": float(
                info.get("tactical_basis_lateral_sign", np.nan)
            ),
            "tactical_basis_ll_world_x": tactical_basis_ll_world[0],
            "tactical_basis_ll_world_y": tactical_basis_ll_world[1],
            "tactical_basis_ll_world_z": tactical_basis_ll_world[2],
            "tactical_basis_io_world_x": tactical_basis_io_world[0],
            "tactical_basis_io_world_y": tactical_basis_io_world[1],
            "tactical_basis_io_world_z": tactical_basis_io_world[2],
            "tactical_basis_cd_world_x": tactical_basis_cd_world[0],
            "tactical_basis_cd_world_y": tactical_basis_cd_world[1],
            "tactical_basis_cd_world_z": tactical_basis_cd_world[2],
            "tactical_basis_world_offset_x": tactical_basis_world_offset[0],
            "tactical_basis_world_offset_y": tactical_basis_world_offset[1],
            "tactical_basis_world_offset_z": tactical_basis_world_offset[2],
            "runtime_specialist_key": info.get("runtime_specialist_key"),
            "runtime_specialist_profile": info.get("runtime_specialist_profile"),
            "runtime_specialist_mode_name": info.get(
                "runtime_specialist_mode_name"
            ),
            "post_merge_tactical_basis_recovery_profile_active": bool(
                info.get("post_merge_tactical_basis_recovery_profile_active", False)
            ),
            "post_merge_tactical_basis_recovery_profile_reason": info.get(
                "post_merge_tactical_basis_recovery_profile_reason"
            ),
            "post_merge_tactical_basis_recovery_profile_source": info.get(
                "post_merge_tactical_basis_recovery_profile_source"
            ),
            "post_merge_tactical_basis_recovery_profile_requested": info.get(
                "post_merge_tactical_basis_recovery_profile_requested"
            ),
            "post_merge_tactical_basis_recovery_profile_specialist_profile_match": bool(
                info.get(
                    "post_merge_tactical_basis_recovery_profile_specialist_profile_match",
                    False,
                )
            ),
            "post_merge_tactical_basis_recovery_profile_blend_release_recovery_active": bool(
                info.get(
                    "post_merge_tactical_basis_recovery_profile_blend_release_recovery_active",
                    False,
                )
            ),
            "post_merge_tactical_basis_recovery_profile_ll_pre": float(
                info.get(
                    "post_merge_tactical_basis_recovery_profile_ll_pre",
                    np.nan,
                )
            ),
            "post_merge_tactical_basis_recovery_profile_ll_post": float(
                info.get(
                    "post_merge_tactical_basis_recovery_profile_ll_post",
                    np.nan,
                )
            ),
            "post_merge_tactical_basis_recovery_profile_io_pre": float(
                info.get(
                    "post_merge_tactical_basis_recovery_profile_io_pre",
                    np.nan,
                )
            ),
            "post_merge_tactical_basis_recovery_profile_io_post": float(
                info.get(
                    "post_merge_tactical_basis_recovery_profile_io_post",
                    np.nan,
                )
            ),
            "post_merge_tactical_basis_recovery_profile_cd_pre": float(
                info.get(
                    "post_merge_tactical_basis_recovery_profile_cd_pre",
                    np.nan,
                )
            ),
            "post_merge_tactical_basis_recovery_profile_cd_post": float(
                info.get(
                    "post_merge_tactical_basis_recovery_profile_cd_post",
                    np.nan,
                )
            ),
            "post_merge_tactical_basis_recovery_profile_predicted_target_forward_scale_override": float(
                info.get(
                    "post_merge_tactical_basis_recovery_profile_predicted_target_forward_scale_override",
                    np.nan,
                )
            ),
            "predicted_target_blend": float(
                info.get("predicted_target_blend", 1.0)
            ),
            "predicted_target_forward_scale": float(
                info.get("predicted_target_forward_scale", 1.0)
            ),
            "offensive_anchor_blend": float(
                info.get("offensive_anchor_blend", 0.0)
            ),
            "offensive_anchor_longitudinal_blend": float(
                info.get("offensive_anchor_longitudinal_blend", np.nan)
            ),
            "offensive_anchor_lateral_blend": float(
                info.get("offensive_anchor_lateral_blend", np.nan)
            ),
            "offensive_anchor_longitudinal_m": float(
                info.get("offensive_anchor_longitudinal_m", np.nan)
            ),
            "offensive_anchor_frame": info.get(
                "offensive_anchor_frame",
                "target_velocity",
            ),
            "offensive_anchor_lateral_frame": info.get(
                "offensive_anchor_lateral_frame",
                info.get("offensive_anchor_frame", "target_velocity"),
            ),
            "offensive_anchor_encounter_stable_max_heading_delta_deg": float(
                info.get(
                    "offensive_anchor_encounter_stable_max_heading_delta_deg",
                    np.nan,
                )
            ),
            "offensive_anchor_lateral_sign_mode": info.get(
                "offensive_anchor_lateral_sign_mode",
                "same_side",
            ),
            "offensive_anchor_lateral_sign": float(
                info.get("offensive_anchor_lateral_sign", np.nan)
            ),
            "offensive_anchor_lateral_world_offset_x": (
                offensive_anchor_lateral_world_offset[0]
            ),
            "offensive_anchor_lateral_world_offset_y": (
                offensive_anchor_lateral_world_offset[1]
            ),
            "offensive_anchor_lateral_world_offset_z": (
                offensive_anchor_lateral_world_offset[2]
            ),
            "offensive_anchor_lateral_m": float(
                info.get("offensive_anchor_lateral_m", np.nan)
            ),
            "offensive_anchor_vertical_m": float(
                info.get("offensive_anchor_vertical_m", np.nan)
            ),
            "post_merge_predicted_target_blend": float(
                info.get("post_merge_predicted_target_blend", np.nan)
            ),
            "post_merge_predicted_target_forward_scale": float(
                info.get("post_merge_predicted_target_forward_scale", np.nan)
            ),
            "post_merge_predicted_target_forward_scale_release_scale": float(
                info.get(
                    "post_merge_predicted_target_forward_scale_release_scale",
                    np.nan,
                )
            ),
            "post_merge_predicted_target_forward_scale_release_ego_only_streak_steps": float(
                info.get(
                    "post_merge_predicted_target_forward_scale_release_ego_only_streak_steps",
                    np.nan,
                )
            ),
            "post_merge_predicted_target_forward_scale_release_reset_on_streak_break": bool(
                info.get(
                    "post_merge_predicted_target_forward_scale_release_reset_on_streak_break",
                    False,
                )
            ),
            "post_merge_predicted_target_forward_scale_hold_steps": float(
                info.get(
                    "post_merge_predicted_target_forward_scale_hold_steps",
                    np.nan,
                )
            ),
            "post_merge_offensive_anchor_blend": float(
                info.get("post_merge_offensive_anchor_blend", np.nan)
            ),
            "post_merge_offensive_anchor_blend_release_blend": float(
                info.get("post_merge_offensive_anchor_blend_release_blend", np.nan)
            ),
            "post_merge_offensive_anchor_blend_release_ego_only_streak_steps": float(
                info.get(
                    "post_merge_offensive_anchor_blend_release_ego_only_streak_steps",
                    np.nan,
                )
            ),
            "post_merge_offensive_anchor_blend_release_reset_on_streak_break": bool(
                info.get(
                    "post_merge_offensive_anchor_blend_release_reset_on_streak_break",
                    False,
                )
            ),
            "post_merge_offensive_anchor_blend_release_direct_track_enabled": bool(
                info.get(
                    "post_merge_offensive_anchor_blend_release_direct_track_enabled",
                    True,
                )
            ),
            "post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m": float(
                info.get(
                    "post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m",
                    np.nan,
                )
            ),
            "post_merge_offensive_anchor_blend_release_recovery_below_altitude_m": float(
                info.get(
                    "post_merge_offensive_anchor_blend_release_recovery_below_altitude_m",
                    np.nan,
                )
            ),
            "post_merge_offensive_anchor_blend_release_recovery_longitudinal_blend": float(
                info.get(
                    "post_merge_offensive_anchor_blend_release_recovery_longitudinal_blend",
                    np.nan,
                )
            ),
            "post_merge_offensive_anchor_blend_release_recovery_lateral_blend": float(
                info.get(
                    "post_merge_offensive_anchor_blend_release_recovery_lateral_blend",
                    np.nan,
                )
            ),
            "post_merge_offensive_anchor_blend_release_recovery_forward_bias_m_max": float(
                info.get(
                    "post_merge_offensive_anchor_blend_release_recovery_forward_bias_m_max",
                    np.nan,
                )
            ),
            "post_merge_offensive_anchor_blend_release_vp_forward_bias_m_min": float(
                info.get(
                    "post_merge_offensive_anchor_blend_release_vp_forward_bias_m_min",
                    np.nan,
                )
            ),
            "post_merge_offensive_anchor_blend_release_lateral_only_hold_steps": float(
                info.get(
                    "post_merge_offensive_anchor_blend_release_lateral_only_hold_steps",
                    np.nan,
                )
            ),
            "post_merge_offensive_anchor_lateral_world_offset_latch_on_activation": bool(
                info.get(
                    "post_merge_offensive_anchor_lateral_world_offset_latch_on_activation",
                    False,
                )
            ),
            "post_merge_offensive_anchor_lateral_world_offset_latch_active": bool(
                info.get(
                    "post_merge_offensive_anchor_lateral_world_offset_latch_active",
                    False,
                )
            ),
            "post_merge_offensive_anchor_blend_release_lateral_only": bool(
                info.get(
                    "post_merge_offensive_anchor_blend_release_lateral_only",
                    False,
                )
            ),
            "post_merge_offensive_anchor_blend_requires_geometry_disadvantage": bool(
                info.get(
                    "post_merge_offensive_anchor_blend_requires_geometry_disadvantage",
                    False,
                )
            ),
            "post_merge_offensive_anchor_condition_met": bool(
                info.get("post_merge_offensive_anchor_condition_met", False)
            ),
            "post_merge_offensive_anchor_target_only_attack_zone_disadvantage": bool(
                info.get(
                    "post_merge_offensive_anchor_target_only_attack_zone_disadvantage",
                    False,
                )
            ),
            "post_merge_offensive_anchor_geometry_disadvantage": bool(
                info.get(
                    "post_merge_offensive_anchor_geometry_disadvantage",
                    False,
                )
            ),
            "post_merge_offensive_anchor_alignment_disadvantage": bool(
                info.get(
                    "post_merge_offensive_anchor_alignment_disadvantage",
                    False,
                )
            ),
            "post_merge_offensive_anchor_gate_ego_attack_score": float(
                info.get("post_merge_offensive_anchor_gate_ego_attack_score", np.nan)
            ),
            "post_merge_offensive_anchor_gate_target_attack_score": float(
                info.get("post_merge_offensive_anchor_gate_target_attack_score", np.nan)
            ),
            "post_merge_offensive_anchor_gate_ego_in_attack_zone": bool(
                info.get("post_merge_offensive_anchor_gate_ego_in_attack_zone", False)
            ),
            "post_merge_offensive_anchor_gate_target_in_attack_zone": bool(
                info.get(
                    "post_merge_offensive_anchor_gate_target_in_attack_zone",
                    False,
                )
            ),
            "post_merge_offensive_anchor_gate_aa_deg_min": float(
                info.get("post_merge_offensive_anchor_gate_aa_deg_min", np.nan)
            ),
            "post_merge_offensive_anchor_gate_aa_deg": float(
                info.get("post_merge_offensive_anchor_gate_aa_deg", np.nan)
            ),
            "post_merge_offensive_anchor_gate_range_rate_mps": float(
                info.get(
                    "post_merge_offensive_anchor_gate_range_rate_mps",
                    np.nan,
                )
            ),
            "post_merge_offensive_anchor_gate_range_opening": bool(
                info.get("post_merge_offensive_anchor_gate_range_opening", False)
            ),
            "post_merge_predicted_target_blend_release_on_attack_zone": bool(
                info.get(
                    "post_merge_predicted_target_blend_release_on_attack_zone",
                    False,
                )
            ),
            "post_merge_predicted_target_blend_release_requires_target_attack_zone": bool(
                info.get(
                    "post_merge_predicted_target_blend_release_requires_target_attack_zone",
                    False,
                )
            ),
            "post_merge_predicted_target_blend_release_below_altitude_m": float(
                info.get(
                    "post_merge_predicted_target_blend_release_below_altitude_m",
                    np.nan,
                )
            ),
            "post_merge_predicted_target_blend_hold_steps": float(
                info.get("post_merge_predicted_target_blend_hold_steps", np.nan)
            ),
            "post_merge_predicted_target_blend_release_blend": float(
                info.get("post_merge_predicted_target_blend_release_blend", np.nan)
            ),
            "post_merge_predicted_target_blend_steps_since_first_pass": float(
                info.get(
                    "post_merge_predicted_target_blend_steps_since_first_pass",
                    np.nan,
                )
            ),
            "post_merge_predicted_target_blend_hold_remaining_steps": float(
                info.get(
                    "post_merge_predicted_target_blend_hold_remaining_steps",
                    np.nan,
                )
            ),
            "post_merge_predicted_target_blend_hold_window_open": bool(
                info.get("post_merge_predicted_target_blend_hold_window_open", True)
            ),
            "post_merge_predicted_target_forward_scale_steps_since_first_pass": float(
                info.get(
                    "post_merge_predicted_target_forward_scale_steps_since_first_pass",
                    np.nan,
                )
            ),
            "post_merge_predicted_target_forward_scale_hold_remaining_steps": float(
                info.get(
                    "post_merge_predicted_target_forward_scale_hold_remaining_steps",
                    np.nan,
                )
            ),
            "post_merge_predicted_target_forward_scale_hold_window_open": bool(
                info.get(
                    "post_merge_predicted_target_forward_scale_hold_window_open",
                    True,
                )
            ),
            "post_merge_predicted_target_forward_scale_ego_only_streak_steps": float(
                info.get(
                    "post_merge_predicted_target_forward_scale_ego_only_streak_steps",
                    np.nan,
                )
            ),
            "post_merge_predicted_target_blend_active": bool(
                info.get("post_merge_predicted_target_blend_active", False)
            ),
            "post_merge_predicted_target_forward_scale_active": bool(
                info.get("post_merge_predicted_target_forward_scale_active", False)
            ),
            "post_merge_predicted_target_forward_scale_release_scale_active": bool(
                info.get(
                    "post_merge_predicted_target_forward_scale_release_scale_active",
                    False,
                )
            ),
            "post_merge_predicted_target_forward_scale_hold_expired": bool(
                info.get("post_merge_predicted_target_forward_scale_hold_expired", False)
            ),
            "post_merge_predicted_target_forward_scale_release_triggered": bool(
                info.get("post_merge_predicted_target_forward_scale_release_triggered", False)
            ),
            "post_merge_predicted_target_forward_scale_release_reset_triggered": bool(
                info.get(
                    "post_merge_predicted_target_forward_scale_release_reset_triggered",
                    False,
                )
            ),
            "post_merge_predicted_target_forward_scale_released": bool(
                info.get("post_merge_predicted_target_forward_scale_released", False)
            ),
            "post_merge_anchor_mode_ego_only_streak_steps": float(
                info.get("post_merge_anchor_mode_ego_only_streak_steps", np.nan)
            ),
            "post_merge_anchor_mode_release_triggered": bool(
                info.get("post_merge_anchor_mode_release_triggered", False)
            ),
            "post_merge_anchor_mode_release_reset_triggered": bool(
                info.get("post_merge_anchor_mode_release_reset_triggered", False)
            ),
            "post_merge_anchor_mode_released": bool(
                info.get("post_merge_anchor_mode_released", False)
            ),
            "post_merge_offensive_anchor_blend_active": bool(
                info.get("post_merge_offensive_anchor_blend_active", False)
            ),
            "post_merge_offensive_anchor_blend_release_blend_active": bool(
                info.get(
                    "post_merge_offensive_anchor_blend_release_blend_active",
                    False,
                )
            ),
            "post_merge_offensive_anchor_blend_ego_only_streak_steps": float(
                info.get(
                    "post_merge_offensive_anchor_blend_ego_only_streak_steps",
                    np.nan,
                )
            ),
            "post_merge_offensive_anchor_blend_release_triggered": bool(
                info.get(
                    "post_merge_offensive_anchor_blend_release_triggered",
                    False,
                )
            ),
            "post_merge_offensive_anchor_blend_release_reset_triggered": bool(
                info.get(
                    "post_merge_offensive_anchor_blend_release_reset_triggered",
                    False,
                )
            ),
            "post_merge_offensive_anchor_blend_released": bool(
                info.get("post_merge_offensive_anchor_blend_released", False)
            ),
            "post_merge_offensive_anchor_blend_release_recovery_active": bool(
                info.get(
                    "post_merge_offensive_anchor_blend_release_recovery_active",
                    False,
                )
            ),
            "post_merge_offensive_anchor_blend_release_recovery_altitude_trigger_active": bool(
                info.get(
                    "post_merge_offensive_anchor_blend_release_recovery_altitude_trigger_active",
                    False,
                )
            ),
            "post_merge_offensive_anchor_blend_release_recovery_forward_bias_trigger_active": bool(
                info.get(
                    "post_merge_offensive_anchor_blend_release_recovery_forward_bias_trigger_active",
                    False,
                )
            ),
            "post_merge_offensive_anchor_blend_release_recovery_preview_vp_forward_bias_m": float(
                info.get(
                    "post_merge_offensive_anchor_blend_release_recovery_preview_vp_forward_bias_m",
                    np.nan,
                )
            ),
            "post_merge_offensive_anchor_blend_release_vp_forward_bias_clamp_active": bool(
                info.get(
                    "post_merge_offensive_anchor_blend_release_vp_forward_bias_clamp_active",
                    False,
                )
            ),
            "post_merge_offensive_anchor_blend_release_preclamp_vp_forward_bias_m": float(
                info.get(
                    "post_merge_offensive_anchor_blend_release_preclamp_vp_forward_bias_m",
                    np.nan,
                )
            ),
            "post_merge_offensive_anchor_blend_release_direct_track_active": bool(
                info.get(
                    "post_merge_offensive_anchor_blend_release_direct_track_active",
                    False,
                )
            ),
            "post_merge_offensive_anchor_blend_release_lateral_only_active": bool(
                info.get(
                    "post_merge_offensive_anchor_blend_release_lateral_only_active",
                    False,
                )
            ),
            "post_merge_offensive_anchor_blend_release_lateral_only_steps_since_release": float(
                info.get(
                    "post_merge_offensive_anchor_blend_release_lateral_only_steps_since_release",
                    np.nan,
                )
            ),
            "post_merge_offensive_anchor_blend_release_lateral_only_hold_remaining_steps": float(
                info.get(
                    "post_merge_offensive_anchor_blend_release_lateral_only_hold_remaining_steps",
                    np.nan,
                )
            ),
            "post_merge_offensive_anchor_blend_release_lateral_only_hold_window_open": bool(
                info.get(
                    "post_merge_offensive_anchor_blend_release_lateral_only_hold_window_open",
                    True,
                )
            ),
            "post_merge_predicted_target_blend_release_blend_active": bool(
                info.get(
                    "post_merge_predicted_target_blend_release_blend_active",
                    False,
                )
            ),
            "post_merge_predicted_target_blend_released": bool(
                info.get("post_merge_predicted_target_blend_released", False)
            ),
            "longitudinal_scale": float(info.get("longitudinal_scale", 1.0)),
            "lateral_scale": float(info.get("lateral_scale", 1.0)),
            "offset_x": offset[0],
            "offset_y": offset[1],
            "offset_z": offset[2],
            "world_offset_x": world_offset[0],
            "world_offset_y": world_offset[1],
            "world_offset_z": world_offset[2],
            "lookahead_time_s": float(info.get("lookahead_time_s", np.nan)),
            "prediction_valid": bool(info.get("prediction_valid", False)),
            "prediction_fallback": bool(info.get("prediction_fallback", False)),
            "vp_forward_bias_m": float(info.get("vp_forward_bias_m", np.nan)),
            "vp_lateral_bias_m": float(info.get("vp_lateral_bias_m", np.nan)),
            "range_m": float(info.get("range_m", np.nan)),
            "range_rate_mps": float(info.get("range_rate_mps", np.nan)),
            "ata_deg": float(info.get("ata_deg", np.nan)),
            "aa_deg": float(info.get("aa_deg", info.get("aspect_deg", np.nan))),
            "ego_attack_aoa_deg": float(info.get("ego_attack_aoa_deg", np.nan)),
            "target_attack_aoa_deg": float(info.get("target_attack_aoa_deg", np.nan)),
            "ego_in_attack_zone": bool(info.get("ego_in_attack_zone", False)),
            "target_in_attack_zone": bool(info.get("target_in_attack_zone", False)),
            "ego_hp": float(info.get("ego_hp", np.nan)),
            "target_hp": float(info.get("target_hp", np.nan)),
            "ego_attack_score": float(info.get("ego_attack_score", 0.0)),
            "target_attack_score": float(info.get("target_attack_score", 0.0)),
            "pre_merge": bool(info.get("pre_merge", True)),
            "post_merge": bool(info.get("post_merge", False)),
            "min_range_so_far_m": float(info.get("min_range_so_far_m", np.nan)),
            "first_pass_complete": bool(info.get("first_pass_complete", False)),
            "heading_deg": float(np.degrees(own_state.get("yaw_rad", 0.0))),
            "speed_mps": float(own_state.get("speed_mps", 250.0)),
            "altitude_m": float(
                own_state.get(
                    "altitude_m",
                    np.asarray(
                        own_state.get(
                            "position_m",
                            own_state.get("position_neu", [np.nan, np.nan, np.nan]),
                        ),
                        dtype=float,
                    )[2],
                )
            ),
            "nz_g": float(own_state.get("nz_g", 1.0)),
            "nz_cmd": float(info.get("nz_cmd", np.nan)),
            "roll_rad": float(own_state.get("roll_rad", np.nan)),
            "roll_deg": float(np.degrees(own_state.get("roll_rad", np.nan))),
            "roll_rate_cmd": float(info.get("roll_rate_cmd", np.nan)),
            "throttle_cmd": float(info.get("throttle_cmd", np.nan)),
            "nz_saturated": float(info.get("nz_saturated", np.nan)),
            "roll_rate_saturated": float(info.get("roll_rate_saturated", np.nan)),
            "throttle_saturated": float(info.get("throttle_saturated", np.nan)),
            "virtual_point_source": info.get("virtual_point_source"),
            "direct_track_mode_requested": bool(
                info.get("direct_track_mode_requested", False)
            ),
            "direct_track_mode_effective": bool(
                info.get("direct_track_mode_effective", False)
            ),
            "mode_switch_requested": bool(
                info.get("mode_switch_requested", False)
            ),
            "mode_switch_effective": bool(
                info.get("mode_switch_effective", False)
            ),
            "mode_switch_reason": info.get("mode_switch_reason"),
            "effective_guidance_mode": info.get("effective_guidance_mode"),
            "task_supervisor_state": info.get("task_supervisor_state"),
            "task_supervisor_pause_orbit_tracking": bool(
                info.get("task_supervisor_pause_orbit_tracking", False)
            ),
            "task_supervisor_recovery_active": bool(
                info.get("task_supervisor_recovery_active", False)
            ),
            "aggressiveness": info.get("aggressiveness"),
            "gain_scale": info.get("gain_scale"),
            "saturation_flag": bool(info.get("saturation_flag", False)),
            "active_waypoint_index": current_active_idx,
            "switch_event": bool(switch_event),
            "switch_events": copy.deepcopy(info.get("switch_events", [])),
            "waypoints": [copy.deepcopy(wp) for wp in info.get("waypoints", [])],
            "completed_waypoints": info.get("completed_waypoints", 0),
            "completed_orbits": info.get("completed_orbits", 0.0),
            "turn_radius_m": info.get("turn_radius_m", np.nan),
            "orbit_direction": info.get("orbit_direction", np.nan),
            "virtual_point_m": _tolist(
                vp_pos
            ),
            "reward": float(reward),
        }
        self.trajectory.append(point)

    def _compute_statistics(self) -> Dict[str, Any]:
        if self.task == "multi_waypoint":
            return compute_multi_waypoint_metrics(self.trajectory)
        if self.task == "sustained_turn":
            return compute_sustained_turn_metrics(self.trajectory)
        if self.task == "break_turn":
            return compute_break_turn_metrics(self.trajectory)
        return {}

    def finalize(
        self,
        steps: int,
        total_time_s: float,
        total_reward: float,
        termination_reason: str,
        success: bool,
        final_position_m: Any,
        final_speed_mps: float,
        final_altitude_m: float,
    ) -> Dict[str, Any]:
        """Build the final episode JSON."""
        statistics = self._compute_statistics()

        ep: Dict[str, Any] = {
            "run_id": self.run_id,
            "task": self.task,
            "controller": self.controller,
            "seed": self.seed,
            "episode": self.episode,
            "backend": self.backend,
            "strict_backend": self.strict_backend,
            "config_sha256": self.config_sha256,
            "git_commit": self.git_commit,
            "success": success,
            "termination_reason": termination_reason,
            "steps": steps,
            "total_time_s": total_time_s,
            "total_reward": total_reward,
            "trajectory": self.trajectory if self.save_full else [],
            "statistics": statistics,
        }

        if self.task == "multi_waypoint":
            if self.trajectory:
                ep["completed_waypoints"] = int(
                    self.trajectory[-1].get("completed_waypoints", 0)
                )
                ep["switch_events"] = self.trajectory[-1].get("switch_events", [])
                ep["waypoints"] = self.trajectory[-1].get("waypoints", [])
            else:
                ep["completed_waypoints"] = 0
                ep["switch_events"] = []
                ep["waypoints"] = []
        elif self.task == "sustained_turn":
            ep["completed_orbits"] = float(statistics.get("completed_orbits", 0.0))

        return ep


class RunRecorder:
    """
    Records all episodes in a run and writes aggregate outputs.

    Keeps a list of episode JSONs and, at the end of a run, writes:
      - per-episode JSON files under ``raw/``
      - ``aggregate/episode_records.json`` (all episodes in one file)
    """

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.raw_dir = self.run_dir / "raw"
        self.aggregate_dir = self.run_dir / "aggregate"
        self.records: List[Dict[str, Any]] = []

    def add(self, record: Dict[str, Any]) -> None:
        self.records.append(record)

    def write(self) -> None:
        """Write every episode JSON and the aggregate JSON."""
        self.aggregate_dir.mkdir(parents=True, exist_ok=True)

        for rec in self.records:
            task = rec["task"]
            controller = rec["controller"]
            seed = rec["seed"]
            episode = rec["episode"]
            out_path = (
                self.raw_dir
                / task
                / controller
                / f"seed_{seed:02d}"
                / f"episode_{episode:03d}.json"
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(rec, f, indent=2, ensure_ascii=False)

        aggregate_path = self.aggregate_dir / "episode_records.json"
        with open(aggregate_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_dir": str(self.run_dir),
                    "n_episodes": len(self.records),
                    "episodes": self.records,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
