"""Tests for the flight-control comparison recorder layer."""

import tempfile
from pathlib import Path

import numpy as np

from uav_vpp_guidance.evaluation.recorders import EpisodeRecorder, RunRecorder


def _make_info(own_pos, target_pos, completed_waypoints=0):
    return {
        "own_state": {
            "position_m": np.asarray(own_pos),
            "altitude_m": float(np.asarray(own_pos)[2]),
            "speed_mps": 250.0,
            "nz_g": 1.0,
            "roll_rad": 0.35,
        },
        "target_state": {"position_m": np.asarray(target_pos)},
        "range_m": float(np.linalg.norm(np.asarray(own_pos) - np.asarray(target_pos))),
        "nz_cmd": 1.0,
        "roll_rate_cmd": -0.4,
        "throttle_cmd": 0.95,
        "task_supervisor_state": "recovery",
        "task_supervisor_pause_orbit_tracking": True,
        "task_supervisor_recovery_active": True,
        "aggressiveness": 0.0,
        "gain_scale": 1.0,
        "saturation_flag": False,
        "active_waypoint_index": 0,
        "switch_events": [],
        "waypoints": [],
        "completed_waypoints": completed_waypoints,
        "completed_orbits": 0.0,
        "turn_radius_m": 1000.0,
        "orbit_direction": 1.0,
        "virtual_point": {"position_neu": np.asarray(target_pos)},
    }


def test_episode_recorder_multi_waypoint():
    recorder = EpisodeRecorder(
        run_id="r1",
        task="multi_waypoint",
        controller="ppo_pid",
        seed=0,
        episode=0,
        config={},
        config_sha256="abc",
        git_commit="def",
        backend="simple",
        strict_backend=False,
    )
    own = np.array([0.0, 0.0, 5000.0])
    tgt = np.array([1000.0, 0.0, 5000.0])
    info = _make_info(own, tgt)
    recorder.record_step(1, 0.2, info["own_state"], info["target_state"], info, 0.0)
    ep = recorder.finalize(
        steps=1,
        total_time_s=0.2,
        total_reward=0.0,
        termination_reason="timeout",
        success=False,
        final_position_m=own.tolist(),
        final_speed_mps=250.0,
        final_altitude_m=5000.0,
    )
    assert ep["run_id"] == "r1"
    assert ep["backend"] == "simple"
    assert ep["task"] == "multi_waypoint"
    assert "statistics" in ep
    assert "trajectory" in ep
    assert ep["completed_waypoints"] == 0
    point = ep["trajectory"][0]
    assert point["altitude_m"] == 5000.0
    assert point["roll_rad"] == 0.35
    assert point["throttle_cmd"] == 0.95
    assert point["task_supervisor_state"] == "recovery"
    assert point["task_supervisor_pause_orbit_tracking"] is True
    assert point["task_supervisor_recovery_active"] is True


def test_episode_recorder_persists_runtime_vpp_evidence():
    recorder = EpisodeRecorder(
        run_id="r2",
        task="head_on",
        controller="prediction_vpp",
        seed=1,
        episode=2,
        config={},
        config_sha256="abc",
        git_commit="def",
        backend="jsbsim",
        strict_backend=True,
    )
    own = np.array([0.0, 0.0, 5000.0])
    tgt = np.array([1000.0, 0.0, 5000.0])
    info = _make_info(own, tgt)
    info.update(
        {
            "longitudinal_scale": 0.25,
            "offset_frame": "world_neu",
            "configured_offset_frame": "target_velocity",
            "virtual_point_source": "direct_track",
            "close_range_anchor_mode": "offensive_position",
            "close_range_anchor_trigger_range_m": 1000.0,
            "close_range_anchor_release_on_post_merge": True,
            "close_range_anchor_requires_first_pass": True,
            "close_range_anchor_offensive_anchor_blend": 0.25,
            "close_range_anchor_offensive_anchor_blend_active": True,
            "close_range_anchor_release_alignment_angle_deg_max": 12.0,
            "close_range_anchor_post_merge_hold_steps": 20.0,
            "close_range_anchor_alignment_angle_deg_max": 10.0,
            "close_range_anchor_mode_active": True,
            "post_merge_anchor_mode": "offensive_position",
            "post_merge_anchor_mode_requires_geometry_disadvantage": True,
            "post_merge_anchor_mode_condition_met": True,
            "post_merge_anchor_mode_active": True,
            "post_merge_anchor_mode_recovery_active": True,
            "post_merge_anchor_mode_release_ego_only_streak_steps": 1.0,
            "post_merge_anchor_mode_recovery_below_altitude_m": 4500.0,
            "post_merge_anchor_mode_release_reset_on_streak_break": False,
            "post_merge_anchor_mode_offensive_anchor_blend": 0.25,
            "post_merge_anchor_mode_offensive_anchor_longitudinal_blend": 1.0,
            "post_merge_anchor_mode_offensive_anchor_lateral_blend": 0.25,
            "post_merge_anchor_mode_offensive_anchor_blend_active": True,
            "post_merge_anchor_mode_offensive_anchor_component_blend_active": True,
            "post_merge_anchor_mode_lateral_world_offset_latch_on_activation": True,
            "post_merge_anchor_mode_lateral_world_offset_latch_active": True,
            "post_merge_anchor_mode_ego_only_streak_steps": 1.0,
            "post_merge_anchor_mode_release_triggered": True,
            "post_merge_anchor_mode_release_reset_triggered": False,
            "post_merge_anchor_mode_released": True,
            "offensive_anchor_frame": "encounter",
            "offensive_anchor_lateral_frame": "target_velocity",
            "offensive_anchor_lateral_sign_mode": "fixed_positive",
            "offensive_anchor_lateral_sign": 1.0,
            "offensive_anchor_lateral_world_offset": np.array([120.0, 40.0, 0.0]),
            "offensive_anchor_longitudinal_blend": 1.0,
            "offensive_anchor_lateral_blend": 0.25,
            "offensive_anchor_lateral_m": 300.0,
            "offensive_anchor_vertical_m": 500.0,
            "post_merge_predicted_target_forward_scale": 0.0,
            "post_merge_predicted_target_forward_scale_release_scale": 1.0,
            "post_merge_predicted_target_forward_scale_release_ego_only_streak_steps": 1.0,
            "post_merge_predicted_target_forward_scale_release_reset_on_streak_break": True,
            "post_merge_predicted_target_forward_scale_hold_steps": 20.0,
            "post_merge_predicted_target_forward_scale_steps_since_first_pass": 4.0,
            "post_merge_predicted_target_forward_scale_hold_remaining_steps": 17.0,
            "post_merge_predicted_target_forward_scale_hold_window_open": True,
            "post_merge_predicted_target_forward_scale_ego_only_streak_steps": 1.0,
            "post_merge_predicted_target_forward_scale_active": True,
            "post_merge_predicted_target_forward_scale_release_scale_active": False,
            "post_merge_predicted_target_forward_scale_hold_expired": False,
            "post_merge_predicted_target_forward_scale_release_triggered": True,
            "post_merge_predicted_target_forward_scale_release_reset_triggered": False,
            "post_merge_predicted_target_forward_scale_released": True,
            "post_merge_offensive_anchor_blend_release_blend": 0.0,
            "post_merge_offensive_anchor_blend_release_ego_only_streak_steps": 5.0,
            "post_merge_offensive_anchor_blend_release_reset_on_streak_break": True,
            "post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m": 800.0,
            "post_merge_offensive_anchor_blend_release_recovery_below_altitude_m": 4500.0,
            "post_merge_offensive_anchor_blend_release_recovery_forward_bias_m_max": -4500.0,
            "post_merge_offensive_anchor_blend_release_vp_forward_bias_m_min": -4500.0,
            "post_merge_offensive_anchor_blend_release_recovery_longitudinal_blend": 0.0,
            "post_merge_offensive_anchor_blend_release_recovery_lateral_blend": 0.25,
            "post_merge_offensive_anchor_blend_release_lateral_only_hold_steps": 20.0,
            "post_merge_offensive_anchor_blend_release_lateral_only_steps_since_release": 4.0,
            "post_merge_offensive_anchor_blend_release_lateral_only_hold_remaining_steps": 17.0,
            "post_merge_offensive_anchor_blend_release_lateral_only_hold_window_open": True,
            "post_merge_offensive_anchor_blend_requires_geometry_disadvantage": True,
            "post_merge_offensive_anchor_condition_met": True,
            "post_merge_offensive_anchor_target_only_attack_zone_disadvantage": True,
            "post_merge_offensive_anchor_geometry_disadvantage": True,
            "post_merge_offensive_anchor_alignment_disadvantage": False,
            "post_merge_offensive_anchor_gate_ego_attack_score": 0.0,
            "post_merge_offensive_anchor_gate_target_attack_score": 0.75,
            "post_merge_offensive_anchor_gate_ego_in_attack_zone": False,
            "post_merge_offensive_anchor_gate_target_in_attack_zone": True,
            "post_merge_offensive_anchor_gate_aa_deg_min": 170.0,
            "post_merge_offensive_anchor_gate_aa_deg": 176.0,
            "post_merge_offensive_anchor_gate_range_rate_mps": 430.0,
            "post_merge_offensive_anchor_gate_range_opening": True,
            "post_merge_offensive_anchor_blend_release_blend_active": False,
            "post_merge_offensive_anchor_blend_ego_only_streak_steps": 2.0,
            "post_merge_offensive_anchor_blend_release_triggered": True,
            "post_merge_offensive_anchor_blend_release_reset_triggered": False,
            "post_merge_offensive_anchor_blend_released": True,
            "post_merge_offensive_anchor_blend_release_recovery_active": True,
            "post_merge_offensive_anchor_blend_release_recovery_altitude_trigger_active": False,
            "post_merge_offensive_anchor_blend_release_recovery_forward_bias_trigger_active": True,
            "post_merge_offensive_anchor_blend_release_recovery_preview_vp_forward_bias_m": -5200.0,
            "post_merge_offensive_anchor_blend_release_vp_forward_bias_clamp_active": True,
            "post_merge_offensive_anchor_blend_release_preclamp_vp_forward_bias_m": -5200.0,
            "post_merge_offensive_anchor_blend_release_direct_track_active": True,
            "direct_track_mode_requested": True,
            "direct_track_mode_effective": True,
            "mode_switch_requested": True,
            "mode_switch_effective": True,
            "mode_switch_reason": "gate_active",
            "effective_guidance_mode": "proportional_navigation",
        }
    )

    recorder.record_step(1, 0.2, info["own_state"], info["target_state"], info, 0.0)
    point = recorder.trajectory[0]
    assert point["longitudinal_scale"] == 0.25
    assert point["offset_frame"] == "world_neu"
    assert point["configured_offset_frame"] == "target_velocity"
    assert point["virtual_point_source"] == "direct_track"
    assert point["close_range_anchor_mode"] == "offensive_position"
    assert point["close_range_anchor_trigger_range_m"] == 1000.0
    assert point["close_range_anchor_release_on_post_merge"] is True
    assert point["close_range_anchor_requires_first_pass"] is True
    assert point["close_range_anchor_offensive_anchor_blend"] == 0.25
    assert point["close_range_anchor_offensive_anchor_blend_active"] is True
    assert point["close_range_anchor_release_alignment_angle_deg_max"] == 12.0
    assert point["close_range_anchor_post_merge_hold_steps"] == 20.0
    assert point["close_range_anchor_alignment_angle_deg_max"] == 10.0
    assert point["close_range_anchor_mode_active"] is True
    assert point["post_merge_anchor_mode"] == "offensive_position"
    assert point["post_merge_anchor_mode_requires_geometry_disadvantage"] is True
    assert point["post_merge_anchor_mode_condition_met"] is True
    assert point["post_merge_anchor_mode_active"] is True
    assert point["post_merge_anchor_mode_recovery_active"] is True
    assert point["post_merge_anchor_mode_release_ego_only_streak_steps"] == 1.0
    assert point["post_merge_anchor_mode_recovery_below_altitude_m"] == 4500.0
    assert point["post_merge_anchor_mode_release_reset_on_streak_break"] is False
    assert point["post_merge_anchor_mode_offensive_anchor_blend"] == 0.25
    assert point["post_merge_anchor_mode_offensive_anchor_longitudinal_blend"] == 1.0
    assert point["post_merge_anchor_mode_offensive_anchor_lateral_blend"] == 0.25
    assert point["post_merge_anchor_mode_offensive_anchor_blend_active"] is True
    assert point["post_merge_anchor_mode_offensive_anchor_component_blend_active"] is True
    assert (
        point["post_merge_anchor_mode_lateral_world_offset_latch_on_activation"]
        is True
    )
    assert point["post_merge_anchor_mode_lateral_world_offset_latch_active"] is True
    assert point["post_merge_anchor_mode_ego_only_streak_steps"] == 1.0
    assert point["post_merge_anchor_mode_release_triggered"] is True
    assert point["post_merge_anchor_mode_release_reset_triggered"] is False
    assert point["post_merge_anchor_mode_released"] is True
    assert point["offensive_anchor_frame"] == "encounter"
    assert point["offensive_anchor_lateral_frame"] == "target_velocity"
    assert point["offensive_anchor_lateral_sign_mode"] == "fixed_positive"
    assert point["offensive_anchor_lateral_sign"] == 1.0
    assert point["offensive_anchor_lateral_world_offset_x"] == 120.0
    assert point["offensive_anchor_lateral_world_offset_y"] == 40.0
    assert point["offensive_anchor_lateral_world_offset_z"] == 0.0
    assert point["offensive_anchor_longitudinal_blend"] == 1.0
    assert point["offensive_anchor_lateral_blend"] == 0.25
    assert point["offensive_anchor_lateral_m"] == 300.0
    assert point["offensive_anchor_vertical_m"] == 500.0
    assert point["post_merge_predicted_target_forward_scale"] == 0.0
    assert point["post_merge_predicted_target_forward_scale_release_scale"] == 1.0
    assert point["post_merge_predicted_target_forward_scale_release_ego_only_streak_steps"] == 1.0
    assert point["post_merge_predicted_target_forward_scale_release_reset_on_streak_break"] is True
    assert point["post_merge_predicted_target_forward_scale_hold_steps"] == 20.0
    assert point["post_merge_predicted_target_forward_scale_steps_since_first_pass"] == 4.0
    assert point["post_merge_predicted_target_forward_scale_hold_remaining_steps"] == 17.0
    assert point["post_merge_predicted_target_forward_scale_hold_window_open"] is True
    assert point["post_merge_predicted_target_forward_scale_ego_only_streak_steps"] == 1.0
    assert point["post_merge_predicted_target_forward_scale_active"] is True
    assert point["post_merge_predicted_target_forward_scale_release_scale_active"] is False
    assert point["post_merge_predicted_target_forward_scale_hold_expired"] is False
    assert point["post_merge_predicted_target_forward_scale_release_triggered"] is True
    assert point["post_merge_predicted_target_forward_scale_release_reset_triggered"] is False
    assert point["post_merge_predicted_target_forward_scale_released"] is True
    assert point["post_merge_offensive_anchor_blend_release_blend"] == 0.0
    assert point["post_merge_offensive_anchor_blend_release_ego_only_streak_steps"] == 5.0
    assert point["post_merge_offensive_anchor_blend_release_reset_on_streak_break"] is True
    assert point["post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m"] == 800.0
    assert point["post_merge_offensive_anchor_blend_release_recovery_below_altitude_m"] == 4500.0
    assert point["post_merge_offensive_anchor_blend_release_recovery_forward_bias_m_max"] == -4500.0
    assert point["post_merge_offensive_anchor_blend_release_vp_forward_bias_m_min"] == -4500.0
    assert point["post_merge_offensive_anchor_blend_release_recovery_longitudinal_blend"] == 0.0
    assert point["post_merge_offensive_anchor_blend_release_recovery_lateral_blend"] == 0.25
    assert point["post_merge_offensive_anchor_blend_release_recovery_active"] is True
    assert point["post_merge_offensive_anchor_blend_release_recovery_altitude_trigger_active"] is False
    assert point["post_merge_offensive_anchor_blend_release_recovery_forward_bias_trigger_active"] is True
    assert point["post_merge_offensive_anchor_blend_release_recovery_preview_vp_forward_bias_m"] == -5200.0
    assert point["post_merge_offensive_anchor_blend_release_vp_forward_bias_clamp_active"] is True
    assert point["post_merge_offensive_anchor_blend_release_preclamp_vp_forward_bias_m"] == -5200.0
    assert point["post_merge_offensive_anchor_blend_release_lateral_only_hold_steps"] == 20.0
    assert (
        point["post_merge_offensive_anchor_blend_release_lateral_only_steps_since_release"]
        == 4.0
    )
    assert (
        point[
            "post_merge_offensive_anchor_blend_release_lateral_only_hold_remaining_steps"
        ]
        == 17.0
    )
    assert (
        point["post_merge_offensive_anchor_blend_release_lateral_only_hold_window_open"]
        is True
    )
    assert point["post_merge_offensive_anchor_blend_requires_geometry_disadvantage"] is True
    assert point["post_merge_offensive_anchor_condition_met"] is True
    assert point["post_merge_offensive_anchor_target_only_attack_zone_disadvantage"] is True
    assert point["post_merge_offensive_anchor_geometry_disadvantage"] is True
    assert point["post_merge_offensive_anchor_alignment_disadvantage"] is False
    assert point["post_merge_offensive_anchor_gate_target_attack_score"] == 0.75
    assert point["post_merge_offensive_anchor_gate_ego_in_attack_zone"] is False
    assert point["post_merge_offensive_anchor_gate_target_in_attack_zone"] is True
    assert point["post_merge_offensive_anchor_gate_aa_deg_min"] == 170.0
    assert point["post_merge_offensive_anchor_gate_aa_deg"] == 176.0
    assert point["post_merge_offensive_anchor_gate_range_rate_mps"] == 430.0
    assert point["post_merge_offensive_anchor_gate_range_opening"] is True
    assert point["post_merge_offensive_anchor_blend_release_blend_active"] is False
    assert point["post_merge_offensive_anchor_blend_ego_only_streak_steps"] == 2.0
    assert point["post_merge_offensive_anchor_blend_release_triggered"] is True
    assert point["post_merge_offensive_anchor_blend_release_reset_triggered"] is False
    assert point["post_merge_offensive_anchor_blend_released"] is True
    assert point["post_merge_offensive_anchor_blend_release_direct_track_active"] is True
    assert point["direct_track_mode_requested"] is True
    assert point["direct_track_mode_effective"] is True
    assert point["mode_switch_requested"] is True
    assert point["mode_switch_effective"] is True
    assert point["mode_switch_reason"] == "gate_active"
    assert point["effective_guidance_mode"] == "proportional_navigation"


def test_recorder_flattens_tactical_basis_fields():
    recorder = EpisodeRecorder(
        run_id="r3",
        task="head_on",
        controller="prediction_vpp",
        seed=3,
        episode=0,
        config={},
        config_sha256="abc",
        git_commit="def",
    )
    own = np.array([0.0, 0.0, 5000.0])
    tgt = np.array([1000.0, 0.0, 5000.0])
    info = _make_info(own, tgt)
    info.update(
        {
            "action_semantics": "tactical_basis_v1",
            "configured_action_semantics": "tactical_basis_v1",
            "tactical_basis_enabled": True,
            "tactical_basis_action_ll": 0.25,
            "tactical_basis_action_io": -0.5,
            "tactical_basis_action_cd": 0.75,
            "tactical_basis_lead_lag_extent_m": 1200.0,
            "tactical_basis_inside_outside_extent_m": 300.0,
            "tactical_basis_climb_descent_extent_m": 600.0,
            "tactical_basis_longitudinal_frame": "target_velocity",
            "tactical_basis_lateral_frame": "encounter_stable",
            "tactical_basis_vertical_frame": "world_neu",
            "tactical_basis_lateral_sign_mode": "same_side",
            "tactical_basis_lateral_sign": 1.0,
            "tactical_basis_ll_world": np.array([0.0, 1200.0, 0.0]),
            "tactical_basis_io_world": np.array([200.0, 100.0, 0.0]),
            "tactical_basis_cd_world": np.array([0.0, 0.0, 600.0]),
            "tactical_basis_world_offset": np.array([-100.0, 250.0, 450.0]),
        }
    )

    recorder.record_step(1, 0.2, info["own_state"], info["target_state"], info, 0.0)
    point = recorder.trajectory[0]

    assert point["action_semantics"] == "tactical_basis_v1"
    assert point["configured_action_semantics"] == "tactical_basis_v1"
    assert point["tactical_basis_enabled"] is True
    assert point["tactical_basis_action_ll"] == 0.25
    assert point["tactical_basis_action_io"] == -0.5
    assert point["tactical_basis_action_cd"] == 0.75
    assert point["tactical_basis_longitudinal_frame"] == "target_velocity"
    assert point["tactical_basis_lateral_frame"] == "encounter_stable"
    assert point["tactical_basis_vertical_frame"] == "world_neu"
    assert point["tactical_basis_lateral_sign_mode"] == "same_side"
    assert point["tactical_basis_lateral_sign"] == 1.0
    assert point["tactical_basis_ll_world_y"] == 1200.0
    assert point["tactical_basis_io_world_x"] == 200.0
    assert point["tactical_basis_cd_world_z"] == 600.0
    assert point["tactical_basis_world_offset_x"] == -100.0
    assert point["tactical_basis_world_offset_y"] == 250.0
    assert point["tactical_basis_world_offset_z"] == 450.0


def test_recorder_preserves_legacy_vp_fields_when_tactical_basis_enabled():
    recorder = EpisodeRecorder(
        run_id="r4",
        task="head_on",
        controller="prediction_vpp",
        seed=4,
        episode=0,
        config={},
        config_sha256="abc",
        git_commit="def",
    )
    own = np.array([0.0, 0.0, 5000.0])
    tgt = np.array([1000.0, 0.0, 5000.0])
    info = _make_info(own, tgt)
    info.update(
        {
            "action_semantics": "tactical_basis_v1",
            "configured_action_semantics": "tactical_basis_v1",
            "tactical_basis_enabled": True,
            "vp_offset": np.array([10.0, 20.0, 30.0]),
            "vp_world_offset": np.array([40.0, 50.0, 60.0]),
            "vp_forward_bias_m": -200.0,
            "vp_lateral_bias_m": 75.0,
        }
    )

    recorder.record_step(1, 0.2, info["own_state"], info["target_state"], info, 0.0)
    point = recorder.trajectory[0]

    assert point["offset_x"] == 10.0
    assert point["offset_y"] == 20.0
    assert point["offset_z"] == 30.0
    assert point["world_offset_x"] == 40.0
    assert point["world_offset_y"] == 50.0
    assert point["world_offset_z"] == 60.0
    assert point["vp_forward_bias_m"] == -200.0
    assert point["vp_lateral_bias_m"] == 75.0


def test_episode_recorder_persists_commander_telemetry():
    recorder = EpisodeRecorder(
        run_id="r5",
        task="head_on",
        controller="hierarchical_commander_mvp_2mode",
        seed=5,
        episode=0,
        config={},
        config_sha256="abc",
        git_commit="def",
    )
    own = np.array([0.0, 0.0, 5000.0])
    tgt = np.array([1000.0, 0.0, 5000.0])
    info = _make_info(own, tgt)
    info.update(
        {
            "commander_mode_id": 1,
            "commander_mode_name": "crossing_specialist",
            "commander_selected_specialist": "crossing_feasible",
            "commander_switch_count": 2,
            "commander_steps_since_switch": 7,
            "commander_macro_action_repeat_steps": 12,
            "commander_task_oracle_gate": "head_on",
            "commander_requested_mode_id": 1,
            "commander_requested_mode_name": "crossing_specialist",
            "commander_crossing_pre_merge_mode_lock_active": True,
            "commander_mode_constraint_triggered": True,
            "commander_mode_constraint_reason": "max_consecutive_crossing_macro_steps_exceeded",
            "commander_head_on_post_merge_reopened_crossing_leash_active": True,
            "commander_head_on_post_merge_reopened_crossing_leash_range_m": 5100.0,
            "commander_head_on_post_merge_reopened_crossing_leash_hp_advantage": 10.0,
            "commander_head_on_post_merge_reopened_crossing_leash_consecutive_crossing_macro_steps": 0,
            "commander_head_on_post_merge_reopened_crossing_secondary_clamp_active": True,
            "commander_head_on_post_merge_reopened_crossing_secondary_clamp_reason": (
                "secondary_low_altitude_unresolved_lateral_descent"
            ),
            "commander_head_on_post_merge_reopened_crossing_secondary_clamp_altitude_m": 4700.0,
            "commander_head_on_post_merge_reopened_crossing_secondary_clamp_vp_lateral_bias_m": 7600.0,
            "commander_head_on_post_merge_reopened_crossing_secondary_clamp_vp_lateral_to_range_ratio": 1.52,
            "commander_head_on_post_merge_reopened_crossing_secondary_clamp_altitude_drop_m_lookback": -320.0,
            "commander_head_on_post_merge_reopened_crossing_secondary_clamp_altitude_drop_lookback_steps": 12,
            "commander_head_on_post_merge_reopened_crossing_target_threat_clamp_active": True,
            "commander_head_on_post_merge_reopened_crossing_target_threat_clamp_reason": (
                "target_attack_zone_reopened_head_on"
            ),
            "commander_head_on_post_merge_reopened_crossing_target_threat_clamp_target_in_attack_zone": True,
            "commander_head_on_post_merge_reopened_crossing_target_threat_clamp_range_m": 3600.0,
            "commander_head_on_post_merge_reopened_crossing_target_threat_clamp_range_rate_mps": 105.0,
            "commander_head_on_post_merge_reopened_crossing_overdeep_clamp_active": True,
            "commander_head_on_post_merge_reopened_crossing_overdeep_clamp_reason": (
                "overdeep_low_lateral_reopened_head_on"
            ),
            "commander_head_on_post_merge_reopened_crossing_overdeep_clamp_range_m": 4800.0,
            "commander_head_on_post_merge_reopened_crossing_overdeep_clamp_vp_forward_bias_m": -9300.0,
            "commander_head_on_post_merge_reopened_crossing_overdeep_clamp_vp_lateral_to_range_ratio": 0.42,
            "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_active": True,
            "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_reason": (
                "geometry_quality_high_side_positive_forward_lateral"
            ),
            "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_altitude_m": 5400.0,
            "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_altitude_delta_m_lookback": 400.0,
            "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_altitude_trend_lookback_steps": 24,
            "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_vp_forward_bias_m": 1500.0,
            "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_vp_lateral_bias_m": 3600.0,
            "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_vp_lateral_to_range_ratio": 0.72,
            "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_leash_active_steps": 84,
            "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_overdeep_active_steps": 19,
            "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_overdeep_seen_since_post_merge": True,
            "commander_head_on_post_merge_recovery_hold_active": True,
            "commander_head_on_post_merge_recovery_hold_reason": (
                "large_lateral_bias_close_range_recovery_hold"
            ),
            "commander_head_on_post_merge_recovery_hold_range_m": 4200.0,
            "commander_head_on_post_merge_recovery_hold_vp_lateral_bias_m": 6800.0,
            "commander_head_on_post_merge_recovery_hold_vp_lateral_to_range_ratio": 1.62,
            "commander_head_on_post_merge_first_recovery_entry_hold_active": True,
            "commander_head_on_post_merge_first_recovery_entry_hold_reason": (
                "cooldown_active"
            ),
            "commander_head_on_post_merge_first_recovery_entry_hold_original_reason": (
                "pre_threat_opening_overlateral_negative_forward_head_on"
            ),
            "commander_head_on_post_merge_first_recovery_entry_hold_armed_this_step": False,
            "commander_head_on_post_merge_first_recovery_entry_hold_cooldown_steps_remaining": 6,
            "commander_mode_switched": True,
            "commander_macro_step_index": 3,
            "commander_first_switch_step": 12,
        }
    )

    recorder.record_step(1, 0.2, info["own_state"], info["target_state"], info, 0.0)
    point = recorder.trajectory[0]

    assert point["commander_mode_id"] == 1
    assert point["commander_mode_name"] == "crossing_specialist"
    assert point["commander_selected_specialist"] == "crossing_feasible"
    assert point["commander_switch_count"] == 2
    assert point["commander_steps_since_switch"] == 7
    assert point["commander_macro_action_repeat_steps"] == 12
    assert point["commander_task_oracle_gate"] == "head_on"
    assert point["commander_requested_mode_id"] == 1
    assert point["commander_requested_mode_name"] == "crossing_specialist"
    assert point["commander_crossing_pre_merge_mode_lock_active"] is True
    assert point["commander_mode_constraint_triggered"] is True
    assert (
        point["commander_mode_constraint_reason"]
        == "max_consecutive_crossing_macro_steps_exceeded"
    )
    assert point["commander_head_on_post_merge_reopened_crossing_leash_active"] is True
    assert point["commander_head_on_post_merge_reopened_crossing_leash_range_m"] == 5100.0
    assert point["commander_head_on_post_merge_reopened_crossing_leash_hp_advantage"] == 10.0
    assert (
        point["commander_head_on_post_merge_reopened_crossing_leash_consecutive_crossing_macro_steps"]
        == 0
    )
    assert point["commander_head_on_post_merge_reopened_crossing_secondary_clamp_active"] is True
    assert (
        point["commander_head_on_post_merge_reopened_crossing_secondary_clamp_reason"]
        == "secondary_low_altitude_unresolved_lateral_descent"
    )
    assert point["commander_head_on_post_merge_reopened_crossing_secondary_clamp_altitude_m"] == 4700.0
    assert point["commander_head_on_post_merge_reopened_crossing_secondary_clamp_vp_lateral_bias_m"] == 7600.0
    assert (
        point["commander_head_on_post_merge_reopened_crossing_secondary_clamp_vp_lateral_to_range_ratio"]
        == 1.52
    )
    assert (
        point["commander_head_on_post_merge_reopened_crossing_secondary_clamp_altitude_drop_m_lookback"]
        == -320.0
    )
    assert (
        point["commander_head_on_post_merge_reopened_crossing_secondary_clamp_altitude_drop_lookback_steps"]
        == 12
    )
    assert (
        point[
            "commander_head_on_post_merge_reopened_crossing_target_threat_clamp_active"
        ]
        is True
    )
    assert (
        point[
            "commander_head_on_post_merge_reopened_crossing_target_threat_clamp_reason"
        ]
        == "target_attack_zone_reopened_head_on"
    )
    assert (
        point[
            "commander_head_on_post_merge_reopened_crossing_target_threat_clamp_target_in_attack_zone"
        ]
        is True
    )
    assert (
        point[
            "commander_head_on_post_merge_reopened_crossing_target_threat_clamp_range_m"
        ]
        == 3600.0
    )
    assert (
        point[
            "commander_head_on_post_merge_reopened_crossing_target_threat_clamp_range_rate_mps"
        ]
        == 105.0
    )
    assert (
        point["commander_head_on_post_merge_reopened_crossing_overdeep_clamp_active"]
        is True
    )
    assert (
        point["commander_head_on_post_merge_reopened_crossing_overdeep_clamp_reason"]
        == "overdeep_low_lateral_reopened_head_on"
    )
    assert (
        point["commander_head_on_post_merge_reopened_crossing_overdeep_clamp_range_m"]
        == 4800.0
    )
    assert (
        point[
            "commander_head_on_post_merge_reopened_crossing_overdeep_clamp_vp_forward_bias_m"
        ]
        == -9300.0
    )
    assert (
        point[
            "commander_head_on_post_merge_reopened_crossing_overdeep_clamp_vp_lateral_to_range_ratio"
        ]
        == 0.42
    )
    assert (
        point[
            "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_active"
        ]
        is True
    )
    assert (
        point[
            "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_reason"
        ]
        == "geometry_quality_high_side_positive_forward_lateral"
    )
    assert (
        point[
            "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_altitude_m"
        ]
        == 5400.0
    )
    assert (
        point[
            "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_altitude_delta_m_lookback"
        ]
        == 400.0
    )
    assert (
        point[
            "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_altitude_trend_lookback_steps"
        ]
        == 24
    )
    assert (
        point[
            "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_vp_forward_bias_m"
        ]
        == 1500.0
    )
    assert (
        point[
            "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_vp_lateral_bias_m"
        ]
        == 3600.0
    )
    assert (
        point[
            "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_vp_lateral_to_range_ratio"
        ]
        == 0.72
    )
    assert (
        point[
            "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_leash_active_steps"
        ]
        == 84
    )
    assert (
        point[
            "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_overdeep_active_steps"
        ]
        == 19
    )
    assert (
        point[
            "commander_head_on_post_merge_reopened_crossing_geometry_quality_guard_overdeep_seen_since_post_merge"
        ]
        is True
    )
    assert point["commander_head_on_post_merge_recovery_hold_active"] is True
    assert (
        point["commander_head_on_post_merge_recovery_hold_reason"]
        == "large_lateral_bias_close_range_recovery_hold"
    )
    assert point["commander_head_on_post_merge_recovery_hold_range_m"] == 4200.0
    assert (
        point["commander_head_on_post_merge_recovery_hold_vp_lateral_bias_m"] == 6800.0
    )
    assert (
        point["commander_head_on_post_merge_recovery_hold_vp_lateral_to_range_ratio"]
        == 1.62
    )
    assert (
        point["commander_head_on_post_merge_first_recovery_entry_hold_active"] is True
    )
    assert (
        point["commander_head_on_post_merge_first_recovery_entry_hold_reason"]
        == "cooldown_active"
    )
    assert (
        point["commander_head_on_post_merge_first_recovery_entry_hold_original_reason"]
        == "pre_threat_opening_overlateral_negative_forward_head_on"
    )
    assert (
        point["commander_head_on_post_merge_first_recovery_entry_hold_armed_this_step"]
        is False
    )
    assert (
        point[
            "commander_head_on_post_merge_first_recovery_entry_hold_cooldown_steps_remaining"
        ]
        == 6
    )
    assert point["commander_mode_switched"] is True
    assert point["commander_macro_step_index"] == 3
    assert point["commander_first_switch_step"] == 12


def test_run_recorder_writes_files():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        recorder = RunRecorder(run_dir)
        recorder.add({"task": "multi_waypoint", "controller": "ppo_pid", "seed": 0, "episode": 0})
        recorder.write()
        assert (run_dir / "aggregate" / "episode_records.json").exists()
        assert (run_dir / "raw" / "multi_waypoint" / "ppo_pid" / "seed_00" / "episode_000.json").exists()
