"""
Tests for CloseRangeTrackingEnv in no-prediction mode.
"""

import copy
import os

import pytest
import numpy as np
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.virtual_point.coordinate_transform import world_to_offset_frame


@pytest.fixture
def base_config():
    return {
        "experiment": {"name": "test_no_pred", "seed": 42, "output_root": "outputs"},
        "env": {
            "use_jsbsim": False,
            "decision_freq": 5,
            "sim_freq": 60,
            "max_high_level_steps": 512,
            "success_range_m": 900.0,
            "success_ata_deg": 25.0,
            "success_hold_time_s": 0.2,
            "hysteresis_range_m": 950.0,
            "hysteresis_ata_deg": 30.0,
            "min_altitude_m": 500.0,
            "max_altitude_m": 15000.0,
            "max_range_m": 8000.0,
            "target_mode": "constant_velocity",
        },
        "virtual_point": {
            "anchor_mode": "current_target",
            "action_dim": 3,
            "d_long_range": [-1500.0, 1500.0],
            "d_lat_range": [-800.0, 800.0],
            "d_vert_range": [-500.0, 500.0],
            "smoothing_alpha": 0.3,
        },
        "trajectory_prediction": {"enabled": False},
        "limits": {
            "nz_min": -2.0,
            "nz_max": 7.0,
            "roll_rate_min": -1.5,
            "roll_rate_max": 1.5,
            "throttle_min": 0.0,
            "throttle_max": 1.0,
        },
        "reward": {
            "w_range": 0.5,
            "w_angle": 0.8,
            "w_energy": 0.2,
            "w_safety": 2.0,
            "w_saturation": 1.0,
            "w_smooth": 0.1,
            "terminal_success": 200.0,
            "terminal_failure": -200.0,
            "terminal_crash": -300.0,
            "min_altitude_m": 500.0,
        },
        "guidance": {
            "mode": "los_rate",
            "use_gain_adapter": False,
            "gains": {
                "k_los": 1.0,
                "k_pos": 0.5,
                "k_damp": 0.2,
                "k_roll": 1.0,
                "k_speed": 0.2,
                "alpha_filter": 0.3,
            },
        },
    }


class TestCloseRangeTrackingEnvNoPrediction:
    def test_reset_returns_observation(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        obs = env.reset(seed=0)
        assert isinstance(obs, dict)
        assert "relative_state" in obs
        assert "own_state" in obs
        assert "target_state" in obs

    def test_step_returns_tuple(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(np.zeros(3))
        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        env.close()

    def test_info_contains_backend(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(3))
        assert "backend" in info
        assert info["backend"] == "simple"
        env.close()

    def test_info_contains_guidance_command(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(3))
        assert "guidance_command" in info
        assert set(info["guidance_command"].keys()) == {
            "nz_cmd",
            "roll_rate_cmd",
            "throttle_cmd",
        }
        env.close()

    def test_info_contains_combat_geometry_diagnostics(self, base_config):
        config = copy.deepcopy(base_config)
        config["attack_zone"] = {
            "enabled": True,
            "initial_hp": 100.0,
            "damage_per_step": 0.5,
            "legacy_range_enabled": False,
            "close_range_enabled": True,
            "close_range_min_km": 0.0,
            "close_range_full_score_km": 1.0,
            "close_range_max_km": 3.0,
            "close_range_max_aoa_deg": 60.0,
        }
        env = CloseRangeTrackingEnv(config)
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.array([0.1, 0.2, 0.0]))

        for field in (
            "anchor_pos",
            "vp_position_neu",
            "vp_offset",
            "vp_world_offset",
            "offset_frame",
            "range_rate_mps",
            "aa_deg",
            "ego_attack_aoa_deg",
            "target_attack_aoa_deg",
            "ego_in_attack_zone",
            "target_in_attack_zone",
            "pre_merge",
            "post_merge",
            "min_range_so_far_m",
            "first_pass_complete",
            "vp_forward_bias_m",
            "vp_lateral_bias_m",
            "nz_saturated",
            "roll_rate_saturated",
            "throttle_saturated",
        ):
            assert field in info

        assert info["offset_frame"] == "world_neu"
        for field in (
            "range_rate_mps",
            "aa_deg",
            "ego_attack_aoa_deg",
            "target_attack_aoa_deg",
            "min_range_so_far_m",
            "vp_forward_bias_m",
            "vp_lateral_bias_m",
        ):
            assert np.isfinite(float(info[field]))
        env.close()

    def test_task_conditioned_offset_frame_is_applied_from_task_name(self, base_config):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["virtual_point"]["offset_frame"] = "world_neu"
        config["virtual_point"]["offset_frame_by_task"] = {
            "head_on": "target_velocity",
            "crossing_feasible": "world_neu",
        }
        config["virtual_point"]["d_long_range"] = [-100.0, 100.0]
        config["virtual_point"]["d_lat_range"] = [-50.0, 50.0]
        config["virtual_point"]["d_vert_range"] = [-10.0, 10.0]
        env = CloseRangeTrackingEnv(config)
        scenario = {
            "own_init": {
                "position_m": [0.0, 0.0, 5000.0],
                "velocity_mps": 200.0,
                "heading_deg": 0.0,
            },
            "target_init": {
                "position_m": [1000.0, 0.0, 5000.0],
                "velocity_mps": 200.0,
                "heading_deg": 90.0,
            },
        }
        env.reset(scenario=scenario, seed=0)
        _, _, _, _, info = env.step(np.array([1.0, 0.0, 0.0]))

        assert info["offset_frame"] == "target_velocity"
        assert np.allclose(info["vp_offset"], np.array([100.0, 0.0, 0.0]))
        assert np.allclose(info["vp_world_offset"], np.array([0.0, 100.0, 0.0]))
        env.close()

    def test_mode_switch_info_preserves_configured_offset_frame(self, base_config):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["virtual_point"]["offset_frame"] = "world_neu"
        config["virtual_point"]["offset_frame_by_task"] = {
            "head_on": "target_velocity",
        }
        config["guidance"]["mode_switch"] = {
            "enabled": True,
            "aspect_threshold_deg": 180.0,
            "range_threshold_m": 10000.0,
            "closing_speed_threshold_mps": 0.0,
        }

        env = CloseRangeTrackingEnv(config)
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(3))

        assert info["configured_offset_frame"] == "target_velocity"
        assert info["offset_frame"] == "world_neu"
        assert info["virtual_point_source"] == "direct_track"
        assert info["direct_track_mode_effective"] is True
        assert info["mode_switch_effective"] is True
        assert info["effective_guidance_mode"] == "proportional_navigation"
        env.close()

    def test_info_contains_tactical_basis_defaults_in_direct_track_path(
        self,
        base_config,
    ):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["virtual_point"]["action_semantics"] = "tactical_basis_v1"
        config["guidance"]["mode_switch"] = {
            "enabled": True,
            "aspect_threshold_deg": 180.0,
            "range_threshold_m": 10000.0,
            "closing_speed_threshold_mps": 0.0,
        }

        env = CloseRangeTrackingEnv(config)
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(3))

        assert info["virtual_point_source"] == "direct_track"
        assert info["configured_action_semantics"] == "tactical_basis_v1"
        assert info["action_semantics"] == "tactical_basis_v1"
        assert info["tactical_basis_enabled"] is False
        assert "tactical_basis_ll_world" in info
        assert "tactical_basis_world_offset" in info
        env.close()

    def test_info_contains_tactical_basis_defaults_in_end_to_end_path(
        self,
        base_config,
    ):
        config = copy.deepcopy(base_config)
        config["virtual_point"]["enabled"] = False
        config["end_to_end"] = {"enabled": True}

        env = CloseRangeTrackingEnv(config)
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(3))

        assert info["virtual_point_source"] == "direct_command"
        assert info["action_semantics"] == "cartesian_offset"
        assert info["configured_action_semantics"] == "cartesian_offset"
        assert info["tactical_basis_enabled"] is False
        assert "tactical_basis_ll_world" in info
        assert "tactical_basis_world_offset" in info
        env.close()

    def test_tactical_basis_info_survives_post_merge_clamp_path(self, base_config):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["virtual_point"]["action_semantics"] = "tactical_basis_v1"
        config["virtual_point"][
            "post_merge_offensive_anchor_blend_release_vp_forward_bias_m_min"
        ] = -100.0

        env = CloseRangeTrackingEnv(config)
        env._first_pass_complete = True
        env._post_merge_offensive_anchor_blend_released = True
        own_state = {
            "position_neu": np.array([0.0, 0.0, 5000.0]),
            "altitude_m": 5000.0,
        }
        target_state = {
            "position_neu": np.array([1000.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([0.0, 250.0, 0.0]),
        }
        virtual_point = {
            "position_neu": np.array([1000.0, -400.0, 5000.0]),
            "position": np.array([1000.0, -400.0, 5000.0]),
        }
        vp_info = {
            "anchor_pos": np.array([1000.0, 0.0, 5000.0]),
            "offset": np.array([0.0, -400.0, 0.0]),
            "world_offset": np.array([0.0, -400.0, 0.0]),
            "offset_frame": "world_neu",
            "action_semantics": "tactical_basis_v1",
            "configured_action_semantics": "tactical_basis_v1",
            "tactical_basis_enabled": True,
            "tactical_basis_action_ll": 0.0,
            "tactical_basis_action_io": -1.0,
            "tactical_basis_action_cd": 0.0,
            "tactical_basis_lead_lag_extent_m": 1200.0,
            "tactical_basis_inside_outside_extent_m": 400.0,
            "tactical_basis_climb_descent_extent_m": 600.0,
            "tactical_basis_longitudinal_frame": "target_velocity",
            "tactical_basis_lateral_frame": "encounter_stable",
            "tactical_basis_vertical_frame": "world_neu",
            "tactical_basis_lateral_sign_mode": "same_side",
            "tactical_basis_lateral_sign": 1.0,
            "tactical_basis_ll_world": np.array([0.0, 1200.0, 0.0]),
            "tactical_basis_io_world": np.array([0.0, 400.0, 0.0]),
            "tactical_basis_cd_world": np.array([0.0, 0.0, 600.0]),
            "tactical_basis_world_offset": np.array([0.0, -400.0, 0.0]),
        }

        _, updated_vp_info, clamp_active, _ = (
            env._apply_post_merge_offensive_anchor_release_forward_bias_clamp(
                virtual_point,
                vp_info,
                own_state,
                target_state,
                anchor_mode="predicted_target",
                recovery_active=False,
                direct_track_mode_effective=False,
                use_command_override=False,
                post_merge_offensive_anchor_gate={
                    "range_opening": True,
                    "ego_in_attack_zone": False,
                    "target_in_attack_zone": False,
                },
            )
        )

        assert clamp_active is True
        assert updated_vp_info["action_semantics"] == "tactical_basis_v1"
        assert updated_vp_info["tactical_basis_enabled"] is True
        assert np.allclose(
            updated_vp_info["tactical_basis_world_offset"],
            np.array([0.0, -400.0, 0.0]),
        )
        env.close()

    def test_task_name_still_drives_by_task_extent_resolution(self, base_config):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["virtual_point"]["action_semantics"] = "tactical_basis_v1"
        config["virtual_point"]["tactical_basis_inside_outside_extent_m_by_task"] = {
            "head_on": 900.0,
            "crossing_feasible": 250.0,
        }

        env = CloseRangeTrackingEnv(config)

        assert (
            env.virtual_point_generator.tactical_basis_inside_outside_extent_m
            == pytest.approx(900.0)
        )
        env.close()

    def test_post_merge_anchor_mode_switches_for_task(self, base_config):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["virtual_point"]["anchor_mode"] = "predicted_target"
        config["virtual_point"]["post_merge_anchor_mode_by_task"] = {
            "head_on": "offensive_position",
        }
        config["virtual_point"]["offensive_anchor_longitudinal_m_by_task"] = {
            "head_on": 400.0,
        }
        env = CloseRangeTrackingEnv(config)
        env.reset(seed=0)
        env._first_pass_complete = True
        _, _, _, _, info = env.step(np.zeros(3))

        assert info["anchor_mode_requested"] == "predicted_target"
        assert info["post_merge_anchor_mode"] == "offensive_position"
        assert info["post_merge_anchor_mode_active"] is True
        assert info["anchor_mode"] == "offensive_position"
        env.close()

    def test_post_merge_anchor_mode_scale_overrides_only_apply_when_active(
        self,
        base_config,
    ):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["virtual_point"]["anchor_mode"] = "predicted_target"
        config["virtual_point"]["close_range_anchor_mode"] = None
        config["virtual_point"]["close_range_anchor_mode_by_task"] = {}
        config["virtual_point"]["post_merge_anchor_mode_by_task"] = {
            "head_on": "offensive_position",
        }
        config["virtual_point"][
            "post_merge_anchor_mode_requires_geometry_disadvantage_by_task"
        ] = {
            "head_on": True,
        }
        config["virtual_point"]["post_merge_anchor_mode_longitudinal_scale_by_task"] = {
            "head_on": 0.0,
        }
        config["virtual_point"]["post_merge_anchor_mode_lateral_scale_by_task"] = {
            "head_on": 0.0,
        }
        env = CloseRangeTrackingEnv(config)
        env.reset(seed=0)
        env._first_pass_complete = True

        gate_states = iter(
            [
                {
                    "requires_geometry_disadvantage": True,
                    "condition_met": False,
                    "target_only_attack_zone_disadvantage": False,
                    "geometry_disadvantage": False,
                    "alignment_disadvantage": False,
                    "ego_attack_score": 0.0,
                    "target_attack_score": 0.0,
                    "ego_in_attack_zone": False,
                    "target_in_attack_zone": False,
                    "aa_deg_min": np.nan,
                    "aa_deg": np.nan,
                    "range_rate_mps": np.nan,
                    "range_opening": False,
                },
                {
                    "requires_geometry_disadvantage": True,
                    "condition_met": True,
                    "target_only_attack_zone_disadvantage": False,
                    "geometry_disadvantage": True,
                    "alignment_disadvantage": False,
                    "ego_attack_score": 0.0,
                    "target_attack_score": 1.0,
                    "ego_in_attack_zone": False,
                    "target_in_attack_zone": True,
                    "aa_deg_min": np.nan,
                    "aa_deg": np.nan,
                    "range_rate_mps": np.nan,
                    "range_opening": False,
                },
            ]
        )
        captured = []

        def _fake_action_to_virtual_point(
            action,
            own_state,
            target_state,
            anchor_mode="current_target",
            **kwargs,
        ):
            captured.append(
                {
                    "anchor_mode": anchor_mode,
                    "longitudinal_scale_override": kwargs.get(
                        "longitudinal_scale_override"
                    ),
                    "lateral_scale_override": kwargs.get("lateral_scale_override"),
                }
            )
            target_pos = np.asarray(target_state["position_neu"], dtype=np.float64)
            longitudinal_scale = kwargs.get("longitudinal_scale_override")
            if longitudinal_scale is None:
                longitudinal_scale = 1.0
            lateral_scale = kwargs.get("lateral_scale_override")
            if lateral_scale is None:
                lateral_scale = 1.0
            return (
                {
                    "position_neu": target_pos.copy(),
                    "position": target_pos.copy(),
                },
                {
                    "anchor_mode": anchor_mode,
                    "anchor_pos": target_pos.copy(),
                    "offset": np.zeros(3, dtype=np.float64),
                    "world_offset": np.zeros(3, dtype=np.float64),
                    "offset_frame": "world_neu",
                    "predicted_target_blend": 1.0,
                    "predicted_target_forward_scale": 1.0,
                    "offensive_anchor_blend": 0.0,
                    "longitudinal_scale": longitudinal_scale,
                    "lateral_scale": lateral_scale,
                },
            )

        env._evaluate_post_merge_anchor_mode_geometry_disadvantage = (
            lambda own_state, target_state, rel_state=None: next(gate_states)
        )
        env.virtual_point_generator.action_to_virtual_point = _fake_action_to_virtual_point

        _, _, _, _, info = env.step(np.zeros(3))
        assert info["post_merge_anchor_mode_active"] is False
        assert captured[-1] == {
            "anchor_mode": "current_target",
            "longitudinal_scale_override": None,
            "lateral_scale_override": None,
        }
        assert info["longitudinal_scale"] == pytest.approx(1.0)
        assert info["lateral_scale"] == pytest.approx(1.0)

        _, _, _, _, info = env.step(np.zeros(3))
        assert info["post_merge_anchor_mode_active"] is True
        assert captured[-1] == {
            "anchor_mode": "offensive_position",
            "longitudinal_scale_override": 0.0,
            "lateral_scale_override": 0.0,
        }
        assert info["longitudinal_scale"] == pytest.approx(0.0)
        assert info["lateral_scale"] == pytest.approx(0.0)
        env.close()

    def test_post_merge_anchor_mode_can_soft_blend_offensive_anchor_without_hard_switch(
        self,
        base_config,
    ):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["virtual_point"]["anchor_mode"] = "predicted_target"
        config["virtual_point"]["close_range_anchor_mode"] = None
        config["virtual_point"]["close_range_anchor_mode_by_task"] = {}
        config["virtual_point"]["post_merge_anchor_mode_by_task"] = {
            "head_on": "offensive_position",
        }
        config["virtual_point"][
            "post_merge_anchor_mode_offensive_anchor_blend_by_task"
        ] = {
            "head_on": 0.25,
        }
        config["virtual_point"]["post_merge_anchor_mode_longitudinal_scale_by_task"] = {
            "head_on": 0.0,
        }
        config["virtual_point"]["post_merge_anchor_mode_lateral_scale_by_task"] = {
            "head_on": 0.0,
        }
        config["virtual_point"]["offensive_anchor_lateral_sign_mode_by_task"] = {
            "head_on": "fixed_positive",
        }
        config["virtual_point"]["offensive_anchor_lateral_frame_by_task"] = {
            "head_on": "encounter",
        }
        config["trajectory_prediction"] = {"enabled": True}
        env = CloseRangeTrackingEnv(config)
        env.reset(seed=0)

        class _FakePredictor:
            def predict(self, target_state):
                target_pos = np.asarray(target_state["position_m"], dtype=np.float64)
                return target_pos + np.array([100.0, 0.0, 0.0]), None, {}

        env.trajectory_predictor_adapter = _FakePredictor()
        env._first_pass_complete = True
        env._first_pass_completion_step = 0

        captured = {}

        def _fake_action_to_virtual_point(
            action,
            own_state,
            target_state,
            anchor_mode="current_target",
            **kwargs,
        ):
            captured["anchor_mode"] = anchor_mode
            captured["offensive_anchor_blend_override"] = kwargs.get(
                "offensive_anchor_blend_override"
            )
            captured["longitudinal_scale_override"] = kwargs.get(
                "longitudinal_scale_override"
            )
            captured["lateral_scale_override"] = kwargs.get(
                "lateral_scale_override"
            )
            target_pos = np.asarray(target_state["position_neu"], dtype=np.float64)
            return (
                {
                    "position_neu": target_pos.copy(),
                    "position": target_pos.copy(),
                },
                {
                    "anchor_mode": anchor_mode,
                    "anchor_pos": target_pos.copy(),
                    "offset": np.zeros(3, dtype=np.float64),
                    "world_offset": np.zeros(3, dtype=np.float64),
                    "offset_frame": "world_neu",
                    "predicted_target_blend": 1.0,
                    "predicted_target_forward_scale": 1.0,
                    "offensive_anchor_blend": (
                        kwargs["offensive_anchor_blend_override"]
                        if kwargs.get("offensive_anchor_blend_override") is not None
                        else 0.0
                    ),
                    "longitudinal_scale": (
                        kwargs["longitudinal_scale_override"]
                        if kwargs.get("longitudinal_scale_override") is not None
                        else 1.0
                    ),
                    "lateral_scale": (
                        kwargs["lateral_scale_override"]
                        if kwargs.get("lateral_scale_override") is not None
                        else 1.0
                    ),
                },
            )

        env.virtual_point_generator.action_to_virtual_point = _fake_action_to_virtual_point

        _, _, _, _, info = env.step(np.zeros(3))
        assert info["post_merge_anchor_mode_active"] is True
        assert info["post_merge_anchor_mode"] == "offensive_position"
        assert info["post_merge_anchor_mode_offensive_anchor_blend"] == pytest.approx(
            0.25
        )
        assert info["post_merge_anchor_mode_offensive_anchor_blend_active"] is True
        assert captured == {
            "anchor_mode": "predicted_target",
            "offensive_anchor_blend_override": 0.25,
            "longitudinal_scale_override": 0.0,
            "lateral_scale_override": 0.0,
        }
        assert info["anchor_mode"] == "predicted_target"
        assert info["offensive_anchor_blend"] == pytest.approx(0.25)
        env.close()

    def test_post_merge_anchor_mode_can_decompose_offensive_anchor_blend_axes(
        self,
        base_config,
    ):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["virtual_point"]["anchor_mode"] = "predicted_target"
        config["virtual_point"]["close_range_anchor_mode"] = None
        config["virtual_point"]["close_range_anchor_mode_by_task"] = {}
        config["virtual_point"]["post_merge_anchor_mode_by_task"] = {
            "head_on": "offensive_position",
        }
        config["virtual_point"][
            "post_merge_anchor_mode_offensive_anchor_longitudinal_blend_by_task"
        ] = {
            "head_on": 1.0,
        }
        config["virtual_point"][
            "post_merge_anchor_mode_offensive_anchor_lateral_blend_by_task"
        ] = {
            "head_on": 0.25,
        }
        config["virtual_point"]["post_merge_anchor_mode_longitudinal_scale_by_task"] = {
            "head_on": 0.0,
        }
        config["virtual_point"]["post_merge_anchor_mode_lateral_scale_by_task"] = {
            "head_on": 0.0,
        }
        config["trajectory_prediction"] = {"enabled": True}
        env = CloseRangeTrackingEnv(config)
        env.reset(seed=0)

        class _FakePredictor:
            def predict(self, target_state):
                target_pos = np.asarray(target_state["position_m"], dtype=np.float64)
                return target_pos + np.array([100.0, 0.0, 0.0]), None, {}

        env.trajectory_predictor_adapter = _FakePredictor()
        env._first_pass_complete = True
        env._first_pass_completion_step = 0

        captured = {}

        def _fake_action_to_virtual_point(
            action,
            own_state,
            target_state,
            anchor_mode="current_target",
            **kwargs,
        ):
            captured["anchor_mode"] = anchor_mode
            captured["offensive_anchor_blend_override"] = kwargs.get(
                "offensive_anchor_blend_override"
            )
            captured["offensive_anchor_longitudinal_blend_override"] = kwargs.get(
                "offensive_anchor_longitudinal_blend_override"
            )
            captured["offensive_anchor_lateral_blend_override"] = kwargs.get(
                "offensive_anchor_lateral_blend_override"
            )
            target_pos = np.asarray(target_state["position_neu"], dtype=np.float64)
            return (
                {
                    "position_neu": target_pos.copy(),
                    "position": target_pos.copy(),
                },
                {
                    "anchor_mode": anchor_mode,
                    "anchor_pos": target_pos.copy(),
                    "offset": np.zeros(3, dtype=np.float64),
                    "world_offset": np.zeros(3, dtype=np.float64),
                    "offset_frame": "world_neu",
                    "predicted_target_blend": 1.0,
                    "predicted_target_forward_scale": 1.0,
                    "offensive_anchor_blend": (
                        kwargs["offensive_anchor_blend_override"]
                        if kwargs.get("offensive_anchor_blend_override") is not None
                        else 0.0
                    ),
                    "offensive_anchor_longitudinal_blend": (
                        kwargs["offensive_anchor_longitudinal_blend_override"]
                        if kwargs.get("offensive_anchor_longitudinal_blend_override")
                        is not None
                        else 0.0
                    ),
                    "offensive_anchor_lateral_blend": (
                        kwargs["offensive_anchor_lateral_blend_override"]
                        if kwargs.get("offensive_anchor_lateral_blend_override")
                        is not None
                        else 0.0
                    ),
                    "offensive_anchor_longitudinal_m": 800.0,
                    "offensive_anchor_frame": "target_velocity",
                    "offensive_anchor_lateral_frame": "encounter",
                    "offensive_anchor_lateral_m": 300.0,
                    "offensive_anchor_vertical_m": 0.0,
                    "offensive_anchor_lateral_sign_mode": "fixed_positive",
                    "offensive_anchor_lateral_sign": 1.0,
                    "longitudinal_scale": (
                        kwargs["longitudinal_scale_override"]
                        if kwargs.get("longitudinal_scale_override") is not None
                        else 1.0
                    ),
                    "lateral_scale": (
                        kwargs["lateral_scale_override"]
                        if kwargs.get("lateral_scale_override") is not None
                        else 1.0
                    ),
                },
            )

        env.virtual_point_generator.action_to_virtual_point = _fake_action_to_virtual_point

        _, _, _, _, info = env.step(np.zeros(3))
        assert info["post_merge_anchor_mode_active"] is True
        assert info["post_merge_anchor_mode_offensive_anchor_blend_active"] is True
        assert (
            info["post_merge_anchor_mode_offensive_anchor_component_blend_active"]
            is True
        )
        assert (
            info["post_merge_anchor_mode_offensive_anchor_longitudinal_blend"]
            == pytest.approx(1.0)
        )
        assert (
            info["post_merge_anchor_mode_offensive_anchor_lateral_blend"]
            == pytest.approx(0.25)
        )
        assert captured == {
            "anchor_mode": "predicted_target",
            "offensive_anchor_blend_override": None,
            "offensive_anchor_longitudinal_blend_override": 1.0,
            "offensive_anchor_lateral_blend_override": 0.25,
        }
        assert info["offensive_anchor_lateral_frame"] == "encounter"
        assert info["offensive_anchor_lateral_sign_mode"] == "fixed_positive"
        assert info["offensive_anchor_lateral_sign"] == pytest.approx(1.0)
        assert info["offensive_anchor_longitudinal_blend"] == pytest.approx(1.0)
        assert info["offensive_anchor_lateral_blend"] == pytest.approx(0.25)
        env.close()

    def test_post_merge_anchor_mode_can_latch_lateral_world_offset_on_activation(
        self,
        base_config,
    ):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["virtual_point"]["anchor_mode"] = "predicted_target"
        config["virtual_point"]["close_range_anchor_mode"] = None
        config["virtual_point"]["close_range_anchor_mode_by_task"] = {}
        config["virtual_point"]["post_merge_anchor_mode_by_task"] = {
            "head_on": "offensive_position",
        }
        config["virtual_point"][
            "post_merge_anchor_mode_lateral_world_offset_latch_on_activation_by_task"
        ] = {
            "head_on": True,
        }
        env = CloseRangeTrackingEnv(config)
        env.reset(seed=0)
        env._first_pass_complete = True
        env._first_pass_completion_step = 0

        captured_overrides = []
        computed_lateral_world_offsets = [
            np.array([120.0, 40.0, 0.0], dtype=np.float64),
            np.array([-60.0, 180.0, 0.0], dtype=np.float64),
        ]
        compute_call_count = {"value": 0}

        def _fake_compute_offensive_anchor_components(
            target_state,
            own_state=None,
            *,
            lateral_world_offset_override=None,
        ):
            index = min(
                compute_call_count["value"],
                len(computed_lateral_world_offsets) - 1,
            )
            lateral_world_offset = computed_lateral_world_offsets[index].copy()
            compute_call_count["value"] += 1
            target_pos = np.asarray(target_state["position_neu"], dtype=np.float64)
            return (
                target_pos + lateral_world_offset,
                np.array([-800.0, 300.0, 0.0], dtype=np.float64),
                lateral_world_offset.copy(),
                np.array([0.0, 300.0, 0.0], dtype=np.float64),
                lateral_world_offset,
                1.0,
            )

        def _fake_action_to_virtual_point(
            action,
            own_state,
            target_state,
            anchor_mode="current_target",
            **kwargs,
        ):
            captured_overrides.append(
                None
                if kwargs.get("offensive_anchor_lateral_world_offset_override")
                is None
                else np.asarray(
                    kwargs["offensive_anchor_lateral_world_offset_override"],
                    dtype=np.float64,
                ).copy()
            )
            target_pos = np.asarray(target_state["position_neu"], dtype=np.float64)
            return (
                {
                    "position_neu": target_pos.copy(),
                    "position": target_pos.copy(),
                },
                {
                    "anchor_mode": anchor_mode,
                    "anchor_pos": target_pos.copy(),
                    "offset": np.zeros(3, dtype=np.float64),
                    "world_offset": np.zeros(3, dtype=np.float64),
                    "offset_frame": "world_neu",
                    "offensive_anchor_lateral_world_offset": (
                        kwargs.get("offensive_anchor_lateral_world_offset_override")
                    ),
                    "offensive_anchor_lateral_local_offset": np.array(
                        [0.0, 300.0, 0.0],
                        dtype=np.float64,
                    ),
                },
            )

        env.virtual_point_generator.compute_offensive_anchor_components = (
            _fake_compute_offensive_anchor_components
        )
        env.virtual_point_generator.action_to_virtual_point = (
            _fake_action_to_virtual_point
        )

        _, _, _, _, info = env.step(np.zeros(3))
        assert info["post_merge_anchor_mode_active"] is True
        assert (
            info["post_merge_anchor_mode_lateral_world_offset_latch_on_activation"]
            is True
        )
        assert info["post_merge_anchor_mode_lateral_world_offset_latch_active"] is True
        assert compute_call_count["value"] == 1
        assert np.allclose(
            captured_overrides[-1],
            computed_lateral_world_offsets[0],
        )
        assert np.allclose(
            info["offensive_anchor_lateral_world_offset"],
            computed_lateral_world_offsets[0],
        )
        assert np.allclose(
            info["offensive_anchor_lateral_local_offset"],
            np.array([0.0, 300.0, 0.0]),
        )
        assert np.allclose(
            env._post_merge_anchor_mode_lateral_world_offset_latched,
            computed_lateral_world_offsets[0],
        )

        _, _, _, _, info = env.step(np.zeros(3))
        assert info["post_merge_anchor_mode_lateral_world_offset_latch_active"] is True
        assert compute_call_count["value"] == 1
        assert np.allclose(
            captured_overrides[-1],
            computed_lateral_world_offsets[0],
        )

        env._post_merge_anchor_mode_released = True
        _, _, _, _, info = env.step(np.zeros(3))
        assert info["post_merge_anchor_mode_lateral_world_offset_latch_active"] is False
        assert env._post_merge_anchor_mode_lateral_world_offset_latched is None
        assert compute_call_count["value"] == 1

        env._post_merge_anchor_mode_released = False
        _, _, _, _, info = env.step(np.zeros(3))
        assert info["post_merge_anchor_mode_lateral_world_offset_latch_active"] is True
        assert compute_call_count["value"] == 2
        assert np.allclose(
            captured_overrides[-1],
            computed_lateral_world_offsets[1],
        )
        assert np.allclose(
            env._post_merge_anchor_mode_lateral_world_offset_latched,
            computed_lateral_world_offsets[1],
        )
        env.close()

    def test_post_merge_anchor_mode_can_require_geometry_disadvantage(
        self,
        base_config,
    ):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["virtual_point"]["anchor_mode"] = "predicted_target"
        config["virtual_point"]["close_range_anchor_mode"] = None
        config["virtual_point"]["close_range_anchor_mode_by_task"] = {}
        config["virtual_point"]["post_merge_anchor_mode_by_task"] = {
            "head_on": "offensive_position",
        }
        config["virtual_point"][
            "post_merge_anchor_mode_requires_geometry_disadvantage_by_task"
        ] = {
            "head_on": True,
        }
        env = CloseRangeTrackingEnv(config)
        env.reset(seed=0)
        env._first_pass_complete = True

        captured = []
        gate_states = iter(
            [
                {
                    "requires_geometry_disadvantage": True,
                    "condition_met": False,
                    "target_only_attack_zone_disadvantage": False,
                    "geometry_disadvantage": False,
                    "alignment_disadvantage": False,
                    "ego_attack_score": 0.0,
                    "target_attack_score": 0.0,
                    "ego_in_attack_zone": False,
                    "target_in_attack_zone": False,
                    "aa_deg_min": np.nan,
                    "aa_deg": np.nan,
                    "range_rate_mps": np.nan,
                    "range_opening": False,
                },
                {
                    "requires_geometry_disadvantage": True,
                    "condition_met": True,
                    "target_only_attack_zone_disadvantage": False,
                    "geometry_disadvantage": True,
                    "alignment_disadvantage": False,
                    "ego_attack_score": 0.0,
                    "target_attack_score": 1.0,
                    "ego_in_attack_zone": False,
                    "target_in_attack_zone": True,
                    "aa_deg_min": np.nan,
                    "aa_deg": np.nan,
                    "range_rate_mps": np.nan,
                    "range_opening": False,
                },
            ]
        )

        def _fake_action_to_virtual_point(
            action,
            own_state,
            target_state,
            anchor_mode="current_target",
            **kwargs,
        ):
            captured.append(anchor_mode)
            target_pos = np.asarray(target_state["position_neu"], dtype=np.float64)
            return (
                {
                    "position_neu": target_pos.copy(),
                    "position": target_pos.copy(),
                },
                {
                    "anchor_mode": anchor_mode,
                    "anchor_pos": target_pos.copy(),
                    "offset": np.zeros(3, dtype=np.float64),
                    "world_offset": np.zeros(3, dtype=np.float64),
                    "offset_frame": "world_neu",
                    "predicted_target_blend": 1.0,
                    "predicted_target_forward_scale": 1.0,
                    "offensive_anchor_blend": 0.0,
                    "lateral_scale": 1.0,
                },
            )

        env._evaluate_post_merge_anchor_mode_geometry_disadvantage = (
            lambda own_state, target_state, rel_state=None: next(gate_states)
        )
        env.virtual_point_generator.action_to_virtual_point = _fake_action_to_virtual_point

        _, _, _, _, info = env.step(np.zeros(3))
        assert info["post_merge_anchor_mode"] == "offensive_position"
        assert info["post_merge_anchor_mode_requires_geometry_disadvantage"] is True
        assert info["post_merge_anchor_mode_condition_met"] is False
        assert info["post_merge_anchor_mode_active"] is False
        assert captured[-1] == "current_target"

        _, _, _, _, info = env.step(np.zeros(3))
        assert info["post_merge_anchor_mode_condition_met"] is True
        assert info["post_merge_anchor_mode_active"] is True
        assert captured[-1] == "offensive_position"
        env.close()

    def test_post_merge_anchor_mode_can_release_after_ego_only_streak(
        self, base_config
    ):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["virtual_point"]["anchor_mode"] = "predicted_target"
        config["virtual_point"]["post_merge_anchor_mode_by_task"] = {
            "head_on": "offensive_position",
        }
        config["virtual_point"][
            "post_merge_anchor_mode_release_ego_only_streak_steps_by_task"
        ] = {
            "head_on": 1,
        }
        config["trajectory_prediction"] = {"enabled": True}
        env = CloseRangeTrackingEnv(config)
        env.reset(seed=0)

        class _FakePredictor:
            def predict(self, target_state):
                target_pos = np.asarray(target_state["position_m"], dtype=np.float64)
                return target_pos + np.array([100.0, 0.0, 0.0]), None, {}

        env.trajectory_predictor_adapter = _FakePredictor()
        env._first_pass_complete = True
        env._first_pass_completion_step = 0

        captured = []

        def _fake_action_to_virtual_point(
            action,
            own_state,
            target_state,
            anchor_mode="current_target",
            **kwargs,
        ):
            captured.append(anchor_mode)
            target_pos = np.asarray(target_state["position_neu"], dtype=np.float64)
            return (
                {
                    "position_neu": target_pos.copy(),
                    "position": target_pos.copy(),
                },
                {
                    "anchor_mode": anchor_mode,
                    "anchor_pos": target_pos.copy(),
                    "offset": np.zeros(3, dtype=np.float64),
                    "world_offset": np.zeros(3, dtype=np.float64),
                    "offset_frame": "world_neu",
                    "predicted_target_blend": 1.0,
                    "predicted_target_forward_scale": 1.0,
                    "offensive_anchor_blend": 0.0,
                    "lateral_scale": 1.0,
                },
            )

        env.virtual_point_generator.action_to_virtual_point = _fake_action_to_virtual_point
        env.combat_hp.step = lambda own_state, target_state: {
            "ego_in_attack_zone": True,
            "target_in_attack_zone": False,
        }

        _, _, _, _, info = env.step(np.zeros(3))
        assert info["post_merge_anchor_mode_active"] is True
        assert info["post_merge_anchor_mode_release_ego_only_streak_steps"] == pytest.approx(
            1.0
        )
        assert info["post_merge_anchor_mode_ego_only_streak_steps"] == pytest.approx(
            1.0
        )
        assert info["post_merge_anchor_mode_release_triggered"] is True
        assert info["post_merge_anchor_mode_released"] is True
        assert captured[-1] == "offensive_position"

        _, _, _, _, info = env.step(np.zeros(3))
        assert info["post_merge_anchor_mode_active"] is False
        assert info["post_merge_anchor_mode_released"] is True
        assert captured[-1] == "predicted_target"
        env.close()

    def test_post_merge_anchor_mode_release_can_reset_after_streak_break(
        self, base_config
    ):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["virtual_point"]["anchor_mode"] = "predicted_target"
        config["virtual_point"]["post_merge_anchor_mode_by_task"] = {
            "head_on": "offensive_position",
        }
        config["virtual_point"][
            "post_merge_anchor_mode_release_ego_only_streak_steps_by_task"
        ] = {
            "head_on": 1,
        }
        config["virtual_point"][
            "post_merge_anchor_mode_release_reset_on_streak_break_by_task"
        ] = {
            "head_on": True,
        }
        config["trajectory_prediction"] = {"enabled": True}
        env = CloseRangeTrackingEnv(config)
        env.reset(seed=0)

        class _FakePredictor:
            def predict(self, target_state):
                target_pos = np.asarray(target_state["position_m"], dtype=np.float64)
                return target_pos + np.array([100.0, 0.0, 0.0]), None, {}

        env.trajectory_predictor_adapter = _FakePredictor()
        env._first_pass_complete = True
        env._first_pass_completion_step = 0

        captured = []
        attack_zone_states = iter(
            [
                {"ego_in_attack_zone": True, "target_in_attack_zone": False},
                {"ego_in_attack_zone": False, "target_in_attack_zone": False},
                {"ego_in_attack_zone": False, "target_in_attack_zone": False},
            ]
        )

        def _fake_action_to_virtual_point(
            action,
            own_state,
            target_state,
            anchor_mode="current_target",
            **kwargs,
        ):
            captured.append(anchor_mode)
            target_pos = np.asarray(target_state["position_neu"], dtype=np.float64)
            return (
                {
                    "position_neu": target_pos.copy(),
                    "position": target_pos.copy(),
                },
                {
                    "anchor_mode": anchor_mode,
                    "anchor_pos": target_pos.copy(),
                    "offset": np.zeros(3, dtype=np.float64),
                    "world_offset": np.zeros(3, dtype=np.float64),
                    "offset_frame": "world_neu",
                    "predicted_target_blend": 1.0,
                    "predicted_target_forward_scale": 1.0,
                    "offensive_anchor_blend": 0.0,
                    "lateral_scale": 1.0,
                },
            )

        env.virtual_point_generator.action_to_virtual_point = _fake_action_to_virtual_point
        env.combat_hp.step = lambda own_state, target_state: next(attack_zone_states)

        _, _, _, _, info = env.step(np.zeros(3))
        assert info["post_merge_anchor_mode_release_triggered"] is True
        assert info["post_merge_anchor_mode_release_reset_triggered"] is False
        assert info["post_merge_anchor_mode_released"] is True
        assert captured[-1] == "offensive_position"

        _, _, _, _, info = env.step(np.zeros(3))
        assert info["post_merge_anchor_mode_active"] is False
        assert info["post_merge_anchor_mode_release_reset_triggered"] is True
        assert info["post_merge_anchor_mode_released"] is False
        assert captured[-1] == "predicted_target"

        _, _, _, _, info = env.step(np.zeros(3))
        assert info["post_merge_anchor_mode_active"] is True
        assert info["post_merge_anchor_mode_released"] is False
        assert captured[-1] == "offensive_position"
        env.close()

    def test_post_merge_anchor_mode_release_requires_prior_activation(
        self, base_config
    ):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["virtual_point"]["anchor_mode"] = "predicted_target"
        config["virtual_point"]["post_merge_anchor_mode_by_task"] = {
            "head_on": "offensive_position",
        }
        config["virtual_point"][
            "post_merge_anchor_mode_requires_geometry_disadvantage_by_task"
        ] = {
            "head_on": True,
        }
        config["virtual_point"][
            "post_merge_anchor_mode_release_ego_only_streak_steps_by_task"
        ] = {
            "head_on": 1,
        }
        config["trajectory_prediction"] = {"enabled": True}
        env = CloseRangeTrackingEnv(config)
        env.reset(seed=0)

        class _FakePredictor:
            def predict(self, target_state):
                target_pos = np.asarray(target_state["position_m"], dtype=np.float64)
                return target_pos + np.array([100.0, 0.0, 0.0]), None, {}

        env.trajectory_predictor_adapter = _FakePredictor()
        env._first_pass_complete = True
        env._first_pass_completion_step = 0
        env._evaluate_post_merge_anchor_mode_geometry_disadvantage = (
            lambda own_state, target_state, rel_state=None: {
                "requires_geometry_disadvantage": True,
                "condition_met": False,
                "target_only_attack_zone_disadvantage": False,
                "geometry_disadvantage": False,
                "alignment_disadvantage": False,
                "ego_attack_score": 0.0,
                "target_attack_score": 0.0,
                "ego_in_attack_zone": False,
                "target_in_attack_zone": False,
                "aa_deg_min": np.nan,
                "aa_deg": np.nan,
                "range_rate_mps": np.nan,
                "range_opening": False,
            }
        )
        env.combat_hp.step = lambda own_state, target_state: {
            "ego_in_attack_zone": True,
            "target_in_attack_zone": False,
        }

        _, _, _, _, info = env.step(np.zeros(3))
        assert info["post_merge_anchor_mode_active"] is False
        assert info["post_merge_anchor_mode_release_triggered"] is False
        assert info["post_merge_anchor_mode_released"] is False

        _, _, _, _, info = env.step(np.zeros(3))
        assert info["post_merge_anchor_mode_active"] is False
        assert info["post_merge_anchor_mode_release_triggered"] is False
        assert info["post_merge_anchor_mode_released"] is False
        env.close()

    def test_post_merge_offensive_anchor_blend_applies_to_predicted_target(self, base_config):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["virtual_point"]["anchor_mode"] = "predicted_target"
        config["virtual_point"]["close_range_anchor_mode"] = None
        config["virtual_point"]["close_range_anchor_mode_by_task"] = {}
        config["virtual_point"]["post_merge_offensive_anchor_blend_by_task"] = {
            "head_on": 0.25,
        }
        config["virtual_point"]["offensive_anchor_longitudinal_m_by_task"] = {
            "head_on": 300.0,
        }
        env = CloseRangeTrackingEnv(config)
        env.reset(seed=0)
        env._first_pass_complete = True
        _, _, _, _, info = env.step(np.zeros(3))

        assert info["anchor_mode_requested"] == "predicted_target"
        assert info["anchor_mode"] == "current_target"
        assert info["post_merge_offensive_anchor_blend"] == pytest.approx(0.25)
        assert info["post_merge_offensive_anchor_blend_active"] is True
        assert info["offensive_anchor_blend"] == pytest.approx(0.25)
        assert info["offensive_anchor_longitudinal_m"] == pytest.approx(300.0)
        assert info["offensive_anchor_vertical_m"] == pytest.approx(0.0)
        env.close()

    def test_post_merge_offensive_anchor_scale_overrides_only_apply_when_blend_is_active(
        self,
        base_config,
    ):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["virtual_point"]["anchor_mode"] = "predicted_target"
        config["virtual_point"]["close_range_anchor_mode"] = None
        config["virtual_point"]["close_range_anchor_mode_by_task"] = {}
        config["virtual_point"]["post_merge_offensive_anchor_blend_by_task"] = {
            "head_on": 0.25,
        }
        config["virtual_point"][
            "post_merge_offensive_anchor_blend_requires_geometry_disadvantage_by_task"
        ] = {
            "head_on": True,
        }
        config["virtual_point"][
            "post_merge_offensive_anchor_longitudinal_scale_by_task"
        ] = {
            "head_on": 0.0,
        }
        config["virtual_point"][
            "post_merge_offensive_anchor_lateral_scale_by_task"
        ] = {
            "head_on": 0.5,
        }
        env = CloseRangeTrackingEnv(config)
        env.reset(seed=0)
        env._first_pass_complete = True
        env._first_pass_completion_step = 0

        gate_states = iter(
            [
                {
                    "requires_geometry_disadvantage": True,
                    "condition_met": False,
                    "target_only_attack_zone_disadvantage": False,
                    "geometry_disadvantage": False,
                    "alignment_disadvantage": False,
                    "ego_attack_score": 0.0,
                    "target_attack_score": 0.0,
                    "ego_in_attack_zone": False,
                    "target_in_attack_zone": False,
                    "aa_deg_min": np.nan,
                    "aa_deg": np.nan,
                    "range_rate_mps": np.nan,
                    "range_opening": False,
                },
                {
                    "requires_geometry_disadvantage": True,
                    "condition_met": True,
                    "target_only_attack_zone_disadvantage": False,
                    "geometry_disadvantage": True,
                    "alignment_disadvantage": False,
                    "ego_attack_score": 0.0,
                    "target_attack_score": 1.0,
                    "ego_in_attack_zone": False,
                    "target_in_attack_zone": True,
                    "aa_deg_min": np.nan,
                    "aa_deg": np.nan,
                    "range_rate_mps": np.nan,
                    "range_opening": False,
                },
            ]
        )
        captured = []

        def _fake_action_to_virtual_point(
            action,
            own_state,
            target_state,
            anchor_mode="current_target",
            **kwargs,
        ):
            captured.append(
                {
                    "anchor_mode": anchor_mode,
                    "longitudinal_scale_override": kwargs.get(
                        "longitudinal_scale_override"
                    ),
                    "lateral_scale_override": kwargs.get("lateral_scale_override"),
                }
            )
            target_pos = np.asarray(target_state["position_neu"], dtype=np.float64)
            longitudinal_scale = kwargs.get("longitudinal_scale_override")
            if longitudinal_scale is None:
                longitudinal_scale = 1.0
            lateral_scale = kwargs.get("lateral_scale_override")
            if lateral_scale is None:
                lateral_scale = 1.0
            return (
                {
                    "position_neu": target_pos.copy(),
                    "position": target_pos.copy(),
                },
                {
                    "anchor_mode": anchor_mode,
                    "anchor_pos": target_pos.copy(),
                    "offset": np.zeros(3, dtype=np.float64),
                    "world_offset": np.zeros(3, dtype=np.float64),
                    "offset_frame": "world_neu",
                    "predicted_target_blend": 1.0,
                    "predicted_target_forward_scale": 1.0,
                    "offensive_anchor_blend": 0.25,
                    "longitudinal_scale": longitudinal_scale,
                    "lateral_scale": lateral_scale,
                },
            )

        env._evaluate_post_merge_offensive_anchor_geometry_disadvantage = (
            lambda own_state, target_state, rel_state=None: next(gate_states)
        )
        env.virtual_point_generator.action_to_virtual_point = _fake_action_to_virtual_point

        _, _, _, _, info = env.step(np.zeros(3))
        assert info["post_merge_offensive_anchor_blend_active"] is False
        assert captured[-1] == {
            "anchor_mode": "current_target",
            "longitudinal_scale_override": None,
            "lateral_scale_override": None,
        }
        assert info["longitudinal_scale"] == pytest.approx(1.0)
        assert info["lateral_scale"] == pytest.approx(1.0)
        assert info["post_merge_offensive_anchor_longitudinal_scale"] == pytest.approx(
            0.0
        )
        assert info["post_merge_offensive_anchor_lateral_scale"] == pytest.approx(0.5)

        _, _, _, _, info = env.step(np.zeros(3))
        assert info["post_merge_offensive_anchor_blend_active"] is True
        assert captured[-1] == {
            "anchor_mode": "current_target",
            "longitudinal_scale_override": 0.0,
            "lateral_scale_override": 0.5,
        }
        assert info["longitudinal_scale"] == pytest.approx(0.0)
        assert info["lateral_scale"] == pytest.approx(0.5)
        assert info["post_merge_offensive_anchor_longitudinal_scale"] == pytest.approx(0.0)
        assert info["post_merge_offensive_anchor_lateral_scale"] == pytest.approx(0.5)
        env.close()

    def test_post_merge_offensive_anchor_blend_reports_high_side_vertical_offset(
        self, base_config
    ):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["virtual_point"]["anchor_mode"] = "predicted_target"
        config["virtual_point"]["close_range_anchor_mode"] = None
        config["virtual_point"]["close_range_anchor_mode_by_task"] = {}
        config["virtual_point"]["post_merge_offensive_anchor_blend_by_task"] = {
            "head_on": 0.25,
        }
        config["virtual_point"]["offensive_anchor_longitudinal_m_by_task"] = {
            "head_on": 300.0,
        }
        config["virtual_point"]["offensive_anchor_vertical_m_by_task"] = {
            "head_on": 500.0,
        }
        env = CloseRangeTrackingEnv(config)
        env.reset(seed=0)
        env._first_pass_complete = True
        _, _, _, _, info = env.step(np.zeros(3))

        assert info["post_merge_offensive_anchor_blend_active"] is True
        assert info["offensive_anchor_vertical_m"] == pytest.approx(500.0)
        assert info["offensive_anchor_offset"][2] == pytest.approx(500.0)
        env.close()

    def test_post_merge_offensive_anchor_geometry_disadvantage_uses_attack_zone_scores(
        self,
        base_config,
    ):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["attack_zone"] = {
            "enabled": True,
            "legacy_range_enabled": False,
            "close_range_enabled": True,
            "close_range_min_km": 0.3,
            "close_range_full_score_km": 1.0,
            "close_range_max_km": 3.0,
            "close_range_max_aoa_deg": 60.0,
        }
        config["virtual_point"][
            "post_merge_offensive_anchor_blend_requires_geometry_disadvantage_by_task"
        ] = {
            "head_on": True,
        }
        env = CloseRangeTrackingEnv(config)

        own_state = {
            "position_m": np.array([0.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([-250.0, 0.0, 0.0]),
        }
        target_state = {
            "position_m": np.array([1000.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([-250.0, 0.0, 0.0]),
        }
        gate = env._evaluate_post_merge_offensive_anchor_geometry_disadvantage(
            own_state,
            target_state,
        )

        assert gate["requires_geometry_disadvantage"] is True
        assert gate["condition_met"] is True
        assert gate["target_only_attack_zone_disadvantage"] is True
        assert gate["geometry_disadvantage"] is True
        assert gate["target_attack_score"] > gate["ego_attack_score"]
        assert gate["target_in_attack_zone"] is True
        assert gate["ego_in_attack_zone"] is False
        env.close()

    def test_post_merge_offensive_anchor_geometry_disadvantage_can_use_post_merge_aa(
        self,
        base_config,
    ):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["attack_zone"] = {
            "enabled": True,
            "legacy_range_enabled": False,
            "close_range_enabled": True,
            "close_range_min_km": 0.3,
            "close_range_full_score_km": 1.0,
            "close_range_max_km": 3.0,
            "close_range_max_aoa_deg": 60.0,
        }
        config["virtual_point"][
            "post_merge_offensive_anchor_blend_requires_geometry_disadvantage_by_task"
        ] = {
            "head_on": True,
        }
        config["virtual_point"][
            "post_merge_offensive_anchor_geometry_disadvantage_aa_deg_min_by_task"
        ] = {
            "head_on": 170.0,
        }
        env = CloseRangeTrackingEnv(config)

        own_state = {
            "position_m": np.array([0.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([250.0, 0.0, 0.0]),
        }
        target_state = {
            "position_m": np.array([1200.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([250.0, 0.0, 0.0]),
        }
        rel_state = {
            "range_rate_mps": 420.0,
            "ata_deg": 166.0,
            "aa_deg": 174.0,
        }
        gate = env._evaluate_post_merge_offensive_anchor_geometry_disadvantage(
            own_state,
            target_state,
            rel_state=rel_state,
        )

        assert gate["requires_geometry_disadvantage"] is True
        assert gate["condition_met"] is True
        assert gate["target_only_attack_zone_disadvantage"] is False
        assert gate["geometry_disadvantage"] is False
        assert gate["alignment_disadvantage"] is True
        assert gate["aa_deg_min"] == pytest.approx(170.0)
        assert gate["aa_deg"] == pytest.approx(174.0)
        assert gate["range_rate_mps"] == pytest.approx(420.0)
        assert gate["range_opening"] is True
        env.close()

    def test_post_merge_anchor_mode_gate_is_independent_of_blend_gate(
        self,
        base_config,
    ):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["attack_zone"] = {
            "enabled": True,
            "legacy_range_enabled": False,
            "close_range_enabled": True,
            "close_range_min_km": 0.3,
            "close_range_full_score_km": 1.0,
            "close_range_max_km": 3.0,
            "close_range_max_aoa_deg": 60.0,
        }
        config["virtual_point"]["post_merge_anchor_mode_by_task"] = {
            "head_on": "offensive_position",
        }
        config["virtual_point"][
            "post_merge_anchor_mode_requires_geometry_disadvantage_by_task"
        ] = {
            "head_on": True,
        }
        env = CloseRangeTrackingEnv(config)

        own_state = {
            "position_m": np.array([0.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([250.0, 0.0, 0.0]),
        }
        target_state = {
            "position_m": np.array([1200.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([250.0, 0.0, 0.0]),
        }
        rel_state = {
            "range_rate_mps": 420.0,
            "ata_deg": 150.0,
            "aa_deg": 160.0,
        }
        gate = env._evaluate_post_merge_offensive_anchor_geometry_disadvantage(
            own_state,
            target_state,
            rel_state=rel_state,
            force_geometry_gate=True,
        )

        assert gate["requires_geometry_disadvantage"] is True
        assert gate["condition_met"] is False
        assert gate["target_only_attack_zone_disadvantage"] is False
        assert gate["geometry_disadvantage"] is False
        assert gate["alignment_disadvantage"] is False
        env.close()

    def test_post_merge_anchor_mode_recovery_window_can_delay_activation(
        self,
        base_config,
    ):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["virtual_point"]["post_merge_anchor_mode_by_task"] = {
            "head_on": "offensive_position",
        }
        config["virtual_point"][
            "post_merge_anchor_mode_requires_geometry_disadvantage_by_task"
        ] = {
            "head_on": True,
        }
        config["virtual_point"][
            "post_merge_anchor_mode_recovery_below_altitude_m_by_task"
        ] = {
            "head_on": 4500.0,
        }
        env = CloseRangeTrackingEnv(config)

        env._evaluate_post_merge_offensive_anchor_geometry_disadvantage = (
            lambda own_state, target_state, rel_state=None, force_geometry_gate=False, gate_requires_geometry_disadvantage_override=None: {
                "requires_geometry_disadvantage": True,
                "condition_met": True,
                "target_only_attack_zone_disadvantage": False,
                "geometry_disadvantage": True,
                "ego_attack_score": 0.0,
                "target_attack_score": 1.0,
                "ego_in_attack_zone": False,
                "target_in_attack_zone": False,
                "alignment_disadvantage": False,
                "aa_deg_min": 170.0,
                "aa_deg": 175.0,
                "range_rate_mps": 420.0,
                "range_opening": True,
            }
        )

        target_state = {
            "position_m": np.array([1200.0, 0.0, 4300.0]),
            "velocity_vector_mps": np.array([250.0, 0.0, 0.0]),
        }
        rel_state = {"range_rate_mps": 420.0, "ata_deg": 150.0, "aa_deg": 175.0}

        gate = env._evaluate_post_merge_anchor_mode_geometry_disadvantage(
            {
                "position_m": np.array([0.0, 0.0, 5000.0]),
                "velocity_vector_mps": np.array([250.0, 0.0, 0.0]),
                "altitude_m": 5000.0,
            },
            target_state,
            rel_state=rel_state,
        )
        assert gate["condition_met"] is False
        assert gate["recovery_active"] is False
        assert gate["recovery_below_altitude_m"] == pytest.approx(4500.0)

        gate = env._evaluate_post_merge_anchor_mode_geometry_disadvantage(
            {
                "position_m": np.array([0.0, 0.0, 4300.0]),
                "velocity_vector_mps": np.array([250.0, 0.0, 0.0]),
                "altitude_m": 4300.0,
            },
            target_state,
            rel_state=rel_state,
        )
        assert gate["condition_met"] is True
        assert gate["recovery_active"] is True
        assert gate["recovery_below_altitude_m"] == pytest.approx(4500.0)
        env.close()

    def test_post_merge_anchor_mode_recovery_window_can_switch_to_offensive_position(
        self,
        base_config,
    ):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["virtual_point"]["anchor_mode"] = "predicted_target"
        config["virtual_point"]["post_merge_anchor_mode_by_task"] = {
            "head_on": "offensive_position",
        }
        config["virtual_point"][
            "post_merge_anchor_mode_requires_geometry_disadvantage_by_task"
        ] = {
            "head_on": True,
        }
        config["virtual_point"][
            "post_merge_anchor_mode_recovery_below_altitude_m_by_task"
        ] = {
            "head_on": 4500.0,
        }
        config["trajectory_prediction"] = {"enabled": True}
        env = CloseRangeTrackingEnv(config)
        scenario = {
            "own_init": {
                "position_m": [0.0, 0.0, 4300.0],
                "velocity_mps": 200.0,
                "heading_deg": 0.0,
            },
            "target_init": {
                "position_m": [1500.0, 0.0, 4300.0],
                "velocity_mps": 200.0,
                "heading_deg": 0.0,
            },
        }
        env.reset(scenario=scenario, seed=0)
        env._first_pass_complete = True

        class _FakePredictor:
            def predict(self, target_state):
                target_pos = np.asarray(target_state["position_m"], dtype=np.float64)
                return target_pos + np.array([100.0, 0.0, 0.0]), None, {}

        env.trajectory_predictor_adapter = _FakePredictor()
        env._evaluate_post_merge_offensive_anchor_geometry_disadvantage = (
            lambda own_state, target_state, rel_state=None, force_geometry_gate=False, gate_requires_geometry_disadvantage_override=None: {
                "requires_geometry_disadvantage": True,
                "condition_met": True,
                "target_only_attack_zone_disadvantage": False,
                "geometry_disadvantage": True,
                "ego_attack_score": 0.0,
                "target_attack_score": 1.0,
                "ego_in_attack_zone": False,
                "target_in_attack_zone": False,
                "alignment_disadvantage": False,
                "aa_deg_min": 170.0,
                "aa_deg": 175.0,
                "range_rate_mps": 420.0,
                "range_opening": True,
            }
        )

        captured = []

        def _fake_action_to_virtual_point(
            action,
            own_state,
            target_state,
            anchor_mode="current_target",
            **kwargs,
        ):
            captured.append(anchor_mode)
            target_pos = np.asarray(target_state["position_neu"], dtype=np.float64)
            return (
                {
                    "position_neu": target_pos.copy(),
                    "position": target_pos.copy(),
                },
                {
                    "anchor_mode": anchor_mode,
                    "anchor_pos": target_pos.copy(),
                    "offset": np.zeros(3, dtype=np.float64),
                    "world_offset": np.zeros(3, dtype=np.float64),
                    "offset_frame": "world_neu",
                    "predicted_target_blend": 1.0,
                    "predicted_target_forward_scale": 1.0,
                    "offensive_anchor_blend": 0.0,
                    "lateral_scale": 1.0,
                },
            )

        env.virtual_point_generator.action_to_virtual_point = _fake_action_to_virtual_point

        _, _, _, _, info = env.step(np.zeros(3))
        assert captured[-1] == "offensive_position"
        assert info["anchor_mode"] == "offensive_position"
        assert info["post_merge_anchor_mode_condition_met"] is True
        assert info["post_merge_anchor_mode_active"] is True
        assert info["post_merge_anchor_mode_recovery_active"] is True
        assert info["post_merge_anchor_mode_recovery_below_altitude_m"] == pytest.approx(
            4500.0
        )
        env.close()

    def test_post_merge_anchor_mode_recovery_window_can_activate_without_geometry_gate(
        self,
        base_config,
    ):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["virtual_point"]["post_merge_anchor_mode_by_task"] = {
            "head_on": "offensive_position",
        }
        config["virtual_point"][
            "post_merge_anchor_mode_recovery_below_altitude_m_by_task"
        ] = {
            "head_on": 4500.0,
        }
        env = CloseRangeTrackingEnv(config)

        env._evaluate_post_merge_offensive_anchor_geometry_disadvantage = (
            lambda own_state, target_state, rel_state=None, force_geometry_gate=False, gate_requires_geometry_disadvantage_override=None: {
                "requires_geometry_disadvantage": False,
                "condition_met": True,
                "target_only_attack_zone_disadvantage": False,
                "geometry_disadvantage": False,
                "ego_attack_score": 0.0,
                "target_attack_score": 0.0,
                "ego_in_attack_zone": False,
                "target_in_attack_zone": False,
                "alignment_disadvantage": False,
                "aa_deg_min": np.nan,
                "aa_deg": 110.0,
                "range_rate_mps": 300.0,
                "range_opening": True,
            }
        )
        gate = env._evaluate_post_merge_anchor_mode_geometry_disadvantage(
            {
                "position_m": np.array([0.0, 0.0, 4300.0]),
                "velocity_vector_mps": np.array([250.0, 0.0, 0.0]),
                "altitude_m": 4300.0,
            },
            {
                "position_m": np.array([1200.0, 0.0, 4300.0]),
                "velocity_vector_mps": np.array([250.0, 0.0, 0.0]),
            },
            rel_state={"range_rate_mps": 300.0, "ata_deg": 140.0, "aa_deg": 110.0},
        )

        assert gate["requires_geometry_disadvantage"] is False
        assert gate["condition_met"] is True
        assert gate["recovery_active"] is True
        env.close()

    def test_post_merge_anchor_mode_gate_does_not_inherit_blend_geometry_requirement(
        self,
        base_config,
        monkeypatch,
    ):
        from uav_vpp_guidance.envs import tracking_env as tracking_env_module

        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["attack_zone"] = {
            "enabled": True,
            "legacy_range_enabled": False,
            "close_range_enabled": True,
            "close_range_min_km": 0.3,
            "close_range_full_score_km": 1.0,
            "close_range_max_km": 3.0,
            "close_range_max_aoa_deg": 60.0,
        }
        config["virtual_point"]["post_merge_anchor_mode_by_task"] = {
            "head_on": "offensive_position",
        }
        config["virtual_point"][
            "post_merge_anchor_mode_requires_geometry_disadvantage_by_task"
        ] = {
            "head_on": False,
        }
        config["virtual_point"][
            "post_merge_anchor_mode_recovery_below_altitude_m_by_task"
        ] = {
            "head_on": 4500.0,
        }
        config["virtual_point"][
            "post_merge_offensive_anchor_blend_requires_geometry_disadvantage_by_task"
        ] = {
            "head_on": True,
        }
        env = CloseRangeTrackingEnv(config)

        monkeypatch.setattr(
            tracking_env_module,
            "evaluate_attack_zone",
            lambda own_state, target_state, config=None: {
                "score": 0.0,
                "in_attack_zone": False,
            },
        )

        gate = env._evaluate_post_merge_anchor_mode_geometry_disadvantage(
            {
                "position_m": np.array([0.0, 0.0, 4300.0]),
                "velocity_vector_mps": np.array([250.0, 0.0, 0.0]),
                "altitude_m": 4300.0,
            },
            {
                "position_m": np.array([1200.0, 0.0, 4300.0]),
                "velocity_vector_mps": np.array([250.0, 0.0, 0.0]),
            },
            rel_state={"range_rate_mps": 300.0, "ata_deg": 140.0, "aa_deg": 110.0},
        )

        assert gate["requires_geometry_disadvantage"] is False
        assert gate["condition_met"] is True
        assert gate["recovery_active"] is True
        env.close()

    def test_post_merge_offensive_anchor_blend_can_require_geometry_disadvantage(
        self,
        base_config,
    ):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["virtual_point"]["anchor_mode"] = "predicted_target"
        config["virtual_point"]["close_range_anchor_mode"] = None
        config["virtual_point"]["close_range_anchor_mode_by_task"] = {}
        config["virtual_point"]["post_merge_offensive_anchor_blend_by_task"] = {
            "head_on": 0.25,
        }
        config["virtual_point"][
            "post_merge_offensive_anchor_blend_requires_geometry_disadvantage_by_task"
        ] = {
            "head_on": True,
        }
        env = CloseRangeTrackingEnv(config)
        env.reset(seed=0)
        env._first_pass_complete = True

        captured = []
        gate_states = iter(
            [
                {
                    "requires_geometry_disadvantage": True,
                    "condition_met": False,
                    "target_only_attack_zone_disadvantage": False,
                    "geometry_disadvantage": False,
                    "alignment_disadvantage": False,
                    "ego_attack_score": 0.0,
                    "target_attack_score": 0.0,
                    "ego_in_attack_zone": False,
                    "target_in_attack_zone": False,
                    "aa_deg_min": np.nan,
                    "aa_deg": np.nan,
                    "range_rate_mps": np.nan,
                    "range_opening": False,
                },
                {
                    "requires_geometry_disadvantage": True,
                    "condition_met": True,
                    "target_only_attack_zone_disadvantage": True,
                    "geometry_disadvantage": True,
                    "alignment_disadvantage": False,
                    "ego_attack_score": 0.0,
                    "target_attack_score": 1.0,
                    "ego_in_attack_zone": False,
                    "target_in_attack_zone": True,
                    "aa_deg_min": np.nan,
                    "aa_deg": np.nan,
                    "range_rate_mps": np.nan,
                    "range_opening": False,
                },
            ]
        )

        def _fake_action_to_virtual_point(
            action,
            own_state,
            target_state,
            anchor_mode="current_target",
            **kwargs,
        ):
            captured.append(kwargs.get("offensive_anchor_blend_override"))
            target_pos = np.asarray(target_state["position_neu"], dtype=np.float64)
            offensive_anchor_blend = kwargs.get("offensive_anchor_blend_override")
            return (
                {
                    "position_neu": target_pos.copy(),
                    "position": target_pos.copy(),
                },
                {
                    "anchor_mode": anchor_mode,
                    "anchor_pos": target_pos.copy(),
                    "offset": np.zeros(3, dtype=np.float64),
                    "world_offset": np.zeros(3, dtype=np.float64),
                    "offset_frame": "world_neu",
                    "predicted_target_blend": 1.0,
                    "predicted_target_forward_scale": 1.0,
                    "offensive_anchor_blend": (
                        offensive_anchor_blend
                        if offensive_anchor_blend is not None
                        else 0.0
                    ),
                    "lateral_scale": 1.0,
                },
            )

        env._evaluate_post_merge_offensive_anchor_geometry_disadvantage = (
            lambda own_state, target_state, rel_state=None: next(gate_states)
        )
        env.virtual_point_generator.action_to_virtual_point = _fake_action_to_virtual_point

        _, _, _, _, info = env.step(np.zeros(3))
        assert info["post_merge_offensive_anchor_blend_requires_geometry_disadvantage"] is True
        assert info["post_merge_offensive_anchor_condition_met"] is False
        assert info["post_merge_offensive_anchor_blend_active"] is False
        assert captured[-1] is None

        _, _, _, _, info = env.step(np.zeros(3))
        assert info["post_merge_offensive_anchor_condition_met"] is True
        assert (
            info["post_merge_offensive_anchor_target_only_attack_zone_disadvantage"]
            is True
        )
        assert info["post_merge_offensive_anchor_geometry_disadvantage"] is True
        assert info["post_merge_offensive_anchor_blend_active"] is True
        assert info["post_merge_offensive_anchor_gate_target_attack_score"] == pytest.approx(
            1.0
        )
        assert captured[-1] == pytest.approx(0.25)
        env.close()

    def test_post_merge_offensive_anchor_blend_can_activate_on_post_merge_aa_disadvantage(
        self,
        base_config,
    ):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["virtual_point"]["anchor_mode"] = "predicted_target"
        config["virtual_point"]["close_range_anchor_mode"] = None
        config["virtual_point"]["close_range_anchor_mode_by_task"] = {}
        config["virtual_point"]["post_merge_offensive_anchor_blend_by_task"] = {
            "head_on": 0.25,
        }
        config["virtual_point"][
            "post_merge_offensive_anchor_blend_requires_geometry_disadvantage_by_task"
        ] = {
            "head_on": True,
        }
        env = CloseRangeTrackingEnv(config)
        env.reset(seed=0)
        env._first_pass_complete = True

        captured = []

        def _fake_action_to_virtual_point(
            action,
            own_state,
            target_state,
            anchor_mode="current_target",
            **kwargs,
        ):
            captured.append(kwargs.get("offensive_anchor_blend_override"))
            target_pos = np.asarray(target_state["position_neu"], dtype=np.float64)
            offensive_anchor_blend = kwargs.get("offensive_anchor_blend_override")
            return (
                {
                    "position_neu": target_pos.copy(),
                    "position": target_pos.copy(),
                },
                {
                    "anchor_mode": anchor_mode,
                    "anchor_pos": target_pos.copy(),
                    "offset": np.zeros(3, dtype=np.float64),
                    "world_offset": np.zeros(3, dtype=np.float64),
                    "offset_frame": "world_neu",
                    "predicted_target_blend": 1.0,
                    "predicted_target_forward_scale": 1.0,
                    "offensive_anchor_blend": (
                        offensive_anchor_blend
                        if offensive_anchor_blend is not None
                        else 0.0
                    ),
                    "lateral_scale": 1.0,
                },
            )

        env._evaluate_post_merge_offensive_anchor_geometry_disadvantage = (
            lambda own_state, target_state, rel_state=None: {
                "requires_geometry_disadvantage": True,
                "condition_met": True,
                "target_only_attack_zone_disadvantage": False,
                "geometry_disadvantage": False,
                "alignment_disadvantage": True,
                "ego_attack_score": 0.0,
                "target_attack_score": 0.0,
                "ego_in_attack_zone": False,
                "target_in_attack_zone": False,
                "aa_deg_min": 170.0,
                "aa_deg": 175.0,
                "range_rate_mps": 430.0,
                "range_opening": True,
            }
        )
        env.virtual_point_generator.action_to_virtual_point = _fake_action_to_virtual_point

        _, _, _, _, info = env.step(np.zeros(3))
        assert info["post_merge_offensive_anchor_condition_met"] is True
        assert info["post_merge_offensive_anchor_target_only_attack_zone_disadvantage"] is False
        assert info["post_merge_offensive_anchor_geometry_disadvantage"] is False
        assert info["post_merge_offensive_anchor_alignment_disadvantage"] is True
        assert info["post_merge_offensive_anchor_gate_aa_deg_min"] == pytest.approx(
            170.0
        )
        assert info["post_merge_offensive_anchor_gate_aa_deg"] == pytest.approx(175.0)
        assert info["post_merge_offensive_anchor_gate_range_rate_mps"] == pytest.approx(
            430.0
        )
        assert info["post_merge_offensive_anchor_gate_range_opening"] is True
        assert info["post_merge_offensive_anchor_blend_active"] is True
        assert captured[-1] == pytest.approx(0.25)
        env.close()

    def test_post_merge_offensive_anchor_blend_release_requires_prior_activation(
        self, base_config
    ):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["virtual_point"]["anchor_mode"] = "predicted_target"
        config["virtual_point"]["post_merge_offensive_anchor_blend_by_task"] = {
            "head_on": 0.25,
        }
        config["virtual_point"][
            "post_merge_offensive_anchor_blend_requires_geometry_disadvantage_by_task"
        ] = {
            "head_on": True,
        }
        config["virtual_point"][
            "post_merge_offensive_anchor_blend_release_ego_only_streak_steps_by_task"
        ] = {
            "head_on": 1,
        }
        config["trajectory_prediction"] = {"enabled": True}
        env = CloseRangeTrackingEnv(config)
        env.reset(seed=0)

        class _FakePredictor:
            def predict(self, target_state):
                target_pos = np.asarray(target_state["position_m"], dtype=np.float64)
                return target_pos + np.array([100.0, 0.0, 0.0]), None, {}

        env.trajectory_predictor_adapter = _FakePredictor()
        env._first_pass_complete = True
        env._first_pass_completion_step = 0
        env._evaluate_post_merge_offensive_anchor_geometry_disadvantage = (
            lambda own_state, target_state, rel_state=None: {
                "requires_geometry_disadvantage": True,
                "condition_met": False,
                "target_only_attack_zone_disadvantage": False,
                "geometry_disadvantage": False,
                "alignment_disadvantage": False,
                "ego_attack_score": 0.0,
                "target_attack_score": 0.0,
                "ego_in_attack_zone": False,
                "target_in_attack_zone": False,
                "aa_deg_min": np.nan,
                "aa_deg": np.nan,
                "range_rate_mps": np.nan,
                "range_opening": False,
            }
        )
        env.combat_hp.step = lambda own_state, target_state: {
            "ego_in_attack_zone": True,
            "target_in_attack_zone": False,
        }

        _, _, _, _, info = env.step(np.zeros(3))
        assert info["post_merge_offensive_anchor_blend_active"] is False
        assert info["post_merge_offensive_anchor_blend_release_triggered"] is False
        assert info["post_merge_offensive_anchor_blend_released"] is False

        _, _, _, _, info = env.step(np.zeros(3))
        assert info["post_merge_offensive_anchor_blend_active"] is False
        assert info["post_merge_offensive_anchor_blend_release_triggered"] is False
        assert info["post_merge_offensive_anchor_blend_released"] is False
        env.close()

    def test_post_merge_offensive_anchor_blend_can_fall_back_to_release_blend(
        self, base_config
    ):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["virtual_point"]["anchor_mode"] = "predicted_target"
        config["virtual_point"]["post_merge_offensive_anchor_blend_by_task"] = {
            "head_on": 0.25,
        }
        config["virtual_point"][
            "post_merge_offensive_anchor_blend_release_ego_only_streak_steps_by_task"
        ] = {
            "head_on": 1,
        }
        config["virtual_point"][
            "post_merge_offensive_anchor_blend_release_blend_by_task"
        ] = {
            "head_on": 0.1,
        }
        config["trajectory_prediction"] = {"enabled": True}
        env = CloseRangeTrackingEnv(config)
        env.reset(seed=0)

        class _FakePredictor:
            def predict(self, target_state):
                target_pos = np.asarray(target_state["position_m"], dtype=np.float64)
                return target_pos + np.array([100.0, 0.0, 0.0]), None, {}

        env.trajectory_predictor_adapter = _FakePredictor()

        captured = []

        def _fake_action_to_virtual_point(
            action,
            own_state,
            target_state,
            anchor_mode="current_target",
            **kwargs,
        ):
            captured.append(kwargs.get("offensive_anchor_blend_override"))
            target_pos = np.asarray(target_state["position_neu"], dtype=np.float64)
            return (
                {
                    "position_neu": target_pos.copy(),
                    "position": target_pos.copy(),
                },
                {
                    "anchor_mode": anchor_mode,
                    "anchor_pos": target_pos.copy(),
                    "offset": np.zeros(3, dtype=np.float64),
                    "world_offset": np.zeros(3, dtype=np.float64),
                    "offset_frame": "world_neu",
                    "predicted_target_blend": 1.0,
                    "predicted_target_forward_scale": 1.0,
                    "offensive_anchor_blend": (
                        kwargs["offensive_anchor_blend_override"]
                        if kwargs.get("offensive_anchor_blend_override") is not None
                        else 0.0
                    ),
                    "lateral_scale": 1.0,
                },
            )

        env.virtual_point_generator.action_to_virtual_point = _fake_action_to_virtual_point
        env.combat_hp.step = lambda own_state, target_state: {
            "ego_in_attack_zone": True,
            "target_in_attack_zone": False,
        }

        env._first_pass_complete = True
        env._first_pass_completion_step = 0

        _, _, _, _, info = env.step(np.zeros(3))
        assert info[
            "post_merge_offensive_anchor_blend_release_ego_only_streak_steps"
        ] == pytest.approx(1.0)
        assert info["post_merge_offensive_anchor_blend_ego_only_streak_steps"] == pytest.approx(
            1.0
        )
        assert info["post_merge_offensive_anchor_blend_active"] is True
        assert info["post_merge_offensive_anchor_blend_release_triggered"] is True
        assert info["post_merge_offensive_anchor_blend_released"] is True
        assert info["post_merge_offensive_anchor_blend_release_blend_active"] is False
        assert captured[-1] == pytest.approx(0.25)

        _, _, _, _, info = env.step(np.zeros(3))
        assert info["post_merge_offensive_anchor_blend_active"] is False
        assert info["post_merge_offensive_anchor_blend_released"] is True
        assert info["post_merge_offensive_anchor_blend_release_blend_active"] is True
        assert captured[-1] == pytest.approx(0.1)
        env.close()

    def test_post_merge_offensive_anchor_blend_release_can_reset_after_streak_break(
        self, base_config
    ):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["virtual_point"]["anchor_mode"] = "predicted_target"
        config["virtual_point"]["post_merge_offensive_anchor_blend_by_task"] = {
            "head_on": 0.25,
        }
        config["virtual_point"][
            "post_merge_offensive_anchor_blend_release_ego_only_streak_steps_by_task"
        ] = {
            "head_on": 1,
        }
        config["virtual_point"][
            "post_merge_offensive_anchor_blend_release_blend_by_task"
        ] = {
            "head_on": 0.1,
        }
        config["virtual_point"][
            "post_merge_offensive_anchor_blend_release_reset_on_streak_break_by_task"
        ] = {
            "head_on": True,
        }
        config["trajectory_prediction"] = {"enabled": True}
        env = CloseRangeTrackingEnv(config)
        env.reset(seed=0)

        class _FakePredictor:
            def predict(self, target_state):
                target_pos = np.asarray(target_state["position_m"], dtype=np.float64)
                return target_pos + np.array([100.0, 0.0, 0.0]), None, {}

        env.trajectory_predictor_adapter = _FakePredictor()

        captured = []
        attack_zone_states = iter(
            [
                {"ego_in_attack_zone": True, "target_in_attack_zone": False},
                {"ego_in_attack_zone": False, "target_in_attack_zone": False},
                {"ego_in_attack_zone": False, "target_in_attack_zone": False},
            ]
        )

        def _fake_action_to_virtual_point(
            action,
            own_state,
            target_state,
            anchor_mode="current_target",
            **kwargs,
        ):
            captured.append(kwargs.get("offensive_anchor_blend_override"))
            target_pos = np.asarray(target_state["position_neu"], dtype=np.float64)
            return (
                {
                    "position_neu": target_pos.copy(),
                    "position": target_pos.copy(),
                },
                {
                    "anchor_mode": anchor_mode,
                    "anchor_pos": target_pos.copy(),
                    "offset": np.zeros(3, dtype=np.float64),
                    "world_offset": np.zeros(3, dtype=np.float64),
                    "offset_frame": "world_neu",
                    "predicted_target_blend": 1.0,
                    "predicted_target_forward_scale": 1.0,
                    "offensive_anchor_blend": (
                        kwargs["offensive_anchor_blend_override"]
                        if kwargs.get("offensive_anchor_blend_override") is not None
                        else 0.0
                    ),
                    "lateral_scale": 1.0,
                },
            )

        env.virtual_point_generator.action_to_virtual_point = _fake_action_to_virtual_point
        env.combat_hp.step = lambda own_state, target_state: next(attack_zone_states)

        env._first_pass_complete = True
        env._first_pass_completion_step = 0

        _, _, _, _, info = env.step(np.zeros(3))
        assert (
            info["post_merge_offensive_anchor_blend_release_reset_on_streak_break"]
            is True
        )
        assert info["post_merge_offensive_anchor_blend_release_triggered"] is True
        assert info["post_merge_offensive_anchor_blend_release_reset_triggered"] is False
        assert info["post_merge_offensive_anchor_blend_released"] is True
        assert captured[-1] == pytest.approx(0.25)

        _, _, _, _, info = env.step(np.zeros(3))
        assert info["post_merge_offensive_anchor_blend_release_blend_active"] is True
        assert info["post_merge_offensive_anchor_blend_release_reset_triggered"] is True
        assert info["post_merge_offensive_anchor_blend_released"] is False
        assert captured[-1] == pytest.approx(0.1)

        _, _, _, _, info = env.step(np.zeros(3))
        assert info["post_merge_offensive_anchor_blend_active"] is True
        assert info["post_merge_offensive_anchor_blend_release_blend_active"] is False
        assert info["post_merge_offensive_anchor_blend_released"] is False
        assert captured[-1] == pytest.approx(0.25)
        env.close()

    def test_post_merge_offensive_anchor_blend_release_can_keep_latched_lateral_body(
        self,
        base_config,
    ):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["virtual_point"]["anchor_mode"] = "predicted_target"
        config["virtual_point"]["post_merge_offensive_anchor_blend_by_task"] = {
            "head_on": 0.25,
        }
        config["virtual_point"][
            "post_merge_offensive_anchor_blend_release_blend_by_task"
        ] = {
            "head_on": 0.0,
        }
        config["virtual_point"][
            "post_merge_offensive_anchor_lateral_world_offset_latch_on_activation_by_task"
        ] = {
            "head_on": True,
        }
        config["virtual_point"][
            "post_merge_offensive_anchor_blend_release_lateral_only_by_task"
        ] = {
            "head_on": True,
        }
        config["virtual_point"][
            "post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m_by_task"
        ] = {
            "head_on": 0.0,
        }
        config["trajectory_prediction"] = {"enabled": True}
        env = CloseRangeTrackingEnv(config)
        env.reset(seed=0)

        class _FakePredictor:
            def predict(self, target_state):
                target_pos = np.asarray(target_state["position_m"], dtype=np.float64)
                return target_pos + np.array([100.0, 0.0, 0.0]), None, {}

        env.trajectory_predictor_adapter = _FakePredictor()
        env._first_pass_complete = True
        env._first_pass_completion_step = 0
        env._evaluate_post_merge_offensive_anchor_geometry_disadvantage = (
            lambda own_state, target_state, rel_state=None: {
                "requires_geometry_disadvantage": False,
                "condition_met": True,
                "target_only_attack_zone_disadvantage": False,
                "geometry_disadvantage": False,
                "alignment_disadvantage": True,
                "ego_attack_score": 0.0,
                "target_attack_score": 0.0,
                "ego_in_attack_zone": False,
                "target_in_attack_zone": False,
                "aa_deg_min": np.nan,
                "aa_deg": 175.0,
                "range_rate_mps": 250.0,
                "range_opening": True,
            }
        )

        captured = []
        computed_lateral_world_offset = np.array([120.0, 40.0, 0.0], dtype=np.float64)
        compute_call_count = {"value": 0}

        def _fake_compute_offensive_anchor_components(
            target_state,
            own_state=None,
            *,
            lateral_world_offset_override=None,
        ):
            compute_call_count["value"] += 1
            target_pos = np.asarray(target_state["position_neu"], dtype=np.float64)
            lateral_world_offset = (
                computed_lateral_world_offset.copy()
                if lateral_world_offset_override is None
                else np.asarray(lateral_world_offset_override, dtype=np.float64).copy()
            )
            return (
                target_pos + lateral_world_offset,
                np.array([-800.0, 300.0, 0.0], dtype=np.float64),
                lateral_world_offset.copy(),
                np.array([0.0, 300.0, 0.0], dtype=np.float64),
                lateral_world_offset.copy(),
                1.0,
            )

        def _fake_action_to_virtual_point(
            action,
            own_state,
            target_state,
            anchor_mode="current_target",
            **kwargs,
        ):
            captured.append(
                {
                    "offensive_anchor_blend_override": kwargs.get(
                        "offensive_anchor_blend_override"
                    ),
                    "offensive_anchor_longitudinal_blend_override": kwargs.get(
                        "offensive_anchor_longitudinal_blend_override"
                    ),
                    "offensive_anchor_lateral_blend_override": kwargs.get(
                        "offensive_anchor_lateral_blend_override"
                    ),
                    "offensive_anchor_lateral_world_offset_override": (
                        None
                        if kwargs.get("offensive_anchor_lateral_world_offset_override")
                        is None
                        else np.asarray(
                            kwargs[
                                "offensive_anchor_lateral_world_offset_override"
                            ],
                            dtype=np.float64,
                        ).copy()
                    ),
                }
            )
            target_pos = np.asarray(target_state["position_neu"], dtype=np.float64)
            return (
                {
                    "position_neu": target_pos.copy(),
                    "position": target_pos.copy(),
                },
                {
                    "anchor_mode": anchor_mode,
                    "anchor_pos": target_pos.copy(),
                    "offset": np.zeros(3, dtype=np.float64),
                    "world_offset": np.zeros(3, dtype=np.float64),
                    "offset_frame": "world_neu",
                    "predicted_target_blend": 1.0,
                    "predicted_target_forward_scale": 1.0,
                    "offensive_anchor_blend": (
                        kwargs["offensive_anchor_blend_override"]
                        if kwargs.get("offensive_anchor_blend_override") is not None
                        else 0.0
                    ),
                    "offensive_anchor_lateral_world_offset": (
                        kwargs.get("offensive_anchor_lateral_world_offset_override")
                    ),
                    "offensive_anchor_lateral_local_offset": np.array(
                        [0.0, 300.0, 0.0],
                        dtype=np.float64,
                    ),
                    "lateral_scale": 1.0,
                },
            )

        env.virtual_point_generator.compute_offensive_anchor_components = (
            _fake_compute_offensive_anchor_components
        )
        env.virtual_point_generator.action_to_virtual_point = (
            _fake_action_to_virtual_point
        )
        env.combat_hp.step = lambda own_state, target_state: {
            "ego_in_attack_zone": False,
            "target_in_attack_zone": False,
        }

        _, _, _, _, info = env.step(np.zeros(3))
        assert (
            info["post_merge_offensive_anchor_lateral_world_offset_latch_on_activation"]
            is True
        )
        assert info["post_merge_offensive_anchor_lateral_world_offset_latch_active"] is True
        assert info["post_merge_offensive_anchor_blend_release_lateral_only"] is True
        assert (
            info["post_merge_offensive_anchor_blend_release_lateral_only_active"]
            is False
        )
        assert info["post_merge_offensive_anchor_blend_active"] is True
        assert compute_call_count["value"] == 1
        assert captured[-1]["offensive_anchor_lateral_world_offset_override"] is None
        assert np.allclose(
            env._post_merge_offensive_anchor_lateral_world_offset_latched,
            computed_lateral_world_offset,
        )

        env._post_merge_offensive_anchor_blend_released = True
        _, _, _, _, info = env.step(np.zeros(3))
        assert (
            info["post_merge_offensive_anchor_blend_release_lateral_only_active"]
            is True
        )
        assert info["post_merge_offensive_anchor_blend_release_direct_track_active"] is False
        assert captured[-1]["offensive_anchor_blend_override"] == pytest.approx(0.0)
        assert captured[-1]["offensive_anchor_longitudinal_blend_override"] == pytest.approx(
            0.0
        )
        assert captured[-1]["offensive_anchor_lateral_blend_override"] == pytest.approx(
            1.0
        )
        assert np.allclose(
            captured[-1]["offensive_anchor_lateral_world_offset_override"],
            computed_lateral_world_offset,
        )
        assert compute_call_count["value"] == 1

        env._post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m_by_task[
            "head_on"
        ] = 100000.0
        prior_capture_count = len(captured)
        _, _, _, _, info = env.step(np.zeros(3))
        assert info["post_merge_offensive_anchor_blend_release_direct_track_active"] is True
        assert (
            info["post_merge_offensive_anchor_blend_release_lateral_only_active"]
            is False
        )
        assert info["direct_track_mode_requested"] is True
        assert len(captured) == prior_capture_count
        assert info["post_merge_offensive_anchor_lateral_world_offset_latch_active"] is False
        assert env._post_merge_offensive_anchor_lateral_world_offset_latched is None
        env.close()

    def test_post_merge_offensive_anchor_blend_release_lateral_only_can_expire_after_hold_steps(
        self,
        base_config,
    ):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["virtual_point"]["anchor_mode"] = "predicted_target"
        config["virtual_point"]["post_merge_offensive_anchor_blend_by_task"] = {
            "head_on": 0.25,
        }
        config["virtual_point"][
            "post_merge_offensive_anchor_blend_release_blend_by_task"
        ] = {
            "head_on": 0.0,
        }
        config["virtual_point"][
            "post_merge_offensive_anchor_lateral_world_offset_latch_on_activation_by_task"
        ] = {
            "head_on": True,
        }
        config["virtual_point"][
            "post_merge_offensive_anchor_blend_release_lateral_only_by_task"
        ] = {
            "head_on": True,
        }
        config["virtual_point"][
            "post_merge_offensive_anchor_blend_release_lateral_only_hold_steps_by_task"
        ] = {
            "head_on": 1,
        }
        config["virtual_point"][
            "post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m_by_task"
        ] = {
            "head_on": 0.0,
        }
        config["trajectory_prediction"] = {"enabled": True}
        env = CloseRangeTrackingEnv(config)
        env.reset(seed=0)

        class _FakePredictor:
            def predict(self, target_state):
                target_pos = np.asarray(target_state["position_m"], dtype=np.float64)
                return target_pos + np.array([100.0, 0.0, 0.0]), None, {}

        env.trajectory_predictor_adapter = _FakePredictor()
        env._first_pass_complete = True
        env._first_pass_completion_step = 0
        env._evaluate_post_merge_offensive_anchor_geometry_disadvantage = (
            lambda own_state, target_state, rel_state=None: {
                "requires_geometry_disadvantage": False,
                "condition_met": True,
                "target_only_attack_zone_disadvantage": False,
                "geometry_disadvantage": False,
                "alignment_disadvantage": True,
                "ego_attack_score": 0.0,
                "target_attack_score": 0.0,
                "ego_in_attack_zone": False,
                "target_in_attack_zone": False,
                "aa_deg_min": np.nan,
                "aa_deg": 175.0,
                "range_rate_mps": 250.0,
                "range_opening": True,
            }
        )

        captured = []
        computed_lateral_world_offset = np.array([120.0, 40.0, 0.0], dtype=np.float64)

        def _fake_compute_offensive_anchor_components(
            target_state,
            own_state=None,
            *,
            lateral_world_offset_override=None,
        ):
            target_pos = np.asarray(target_state["position_neu"], dtype=np.float64)
            lateral_world_offset = (
                computed_lateral_world_offset.copy()
                if lateral_world_offset_override is None
                else np.asarray(lateral_world_offset_override, dtype=np.float64).copy()
            )
            return (
                target_pos + lateral_world_offset,
                np.array([-800.0, 300.0, 0.0], dtype=np.float64),
                lateral_world_offset.copy(),
                np.array([0.0, 300.0, 0.0], dtype=np.float64),
                lateral_world_offset.copy(),
                1.0,
            )

        def _fake_action_to_virtual_point(
            action,
            own_state,
            target_state,
            anchor_mode="current_target",
            **kwargs,
        ):
            captured.append(
                {
                    "offensive_anchor_blend_override": kwargs.get(
                        "offensive_anchor_blend_override"
                    ),
                    "offensive_anchor_longitudinal_blend_override": kwargs.get(
                        "offensive_anchor_longitudinal_blend_override"
                    ),
                    "offensive_anchor_lateral_blend_override": kwargs.get(
                        "offensive_anchor_lateral_blend_override"
                    ),
                    "offensive_anchor_lateral_world_offset_override": (
                        None
                        if kwargs.get("offensive_anchor_lateral_world_offset_override")
                        is None
                        else np.asarray(
                            kwargs[
                                "offensive_anchor_lateral_world_offset_override"
                            ],
                            dtype=np.float64,
                        ).copy()
                    ),
                }
            )
            target_pos = np.asarray(target_state["position_neu"], dtype=np.float64)
            return (
                {
                    "position_neu": target_pos.copy(),
                    "position": target_pos.copy(),
                },
                {
                    "anchor_mode": anchor_mode,
                    "anchor_pos": target_pos.copy(),
                    "offset": np.zeros(3, dtype=np.float64),
                    "world_offset": np.zeros(3, dtype=np.float64),
                    "offset_frame": "world_neu",
                    "predicted_target_blend": 1.0,
                    "predicted_target_forward_scale": 1.0,
                    "offensive_anchor_blend": (
                        kwargs["offensive_anchor_blend_override"]
                        if kwargs.get("offensive_anchor_blend_override") is not None
                        else 0.0
                    ),
                    "offensive_anchor_lateral_world_offset": (
                        kwargs.get("offensive_anchor_lateral_world_offset_override")
                    ),
                    "offensive_anchor_lateral_local_offset": np.array(
                        [0.0, 300.0, 0.0],
                        dtype=np.float64,
                    ),
                    "lateral_scale": 1.0,
                },
            )

        env.virtual_point_generator.compute_offensive_anchor_components = (
            _fake_compute_offensive_anchor_components
        )
        env.virtual_point_generator.action_to_virtual_point = (
            _fake_action_to_virtual_point
        )
        env.combat_hp.step = lambda own_state, target_state: {
            "ego_in_attack_zone": False,
            "target_in_attack_zone": False,
        }

        _, _, _, _, info = env.step(np.zeros(3))
        assert (
            info["post_merge_offensive_anchor_blend_release_lateral_only_hold_steps"]
            == pytest.approx(1.0)
        )
        assert (
            info["post_merge_offensive_anchor_blend_release_lateral_only_active"]
            is False
        )

        env._post_merge_offensive_anchor_blend_released = True
        env._post_merge_offensive_anchor_blend_release_step = env.current_step

        _, _, _, _, info = env.step(np.zeros(3))
        assert (
            info["post_merge_offensive_anchor_blend_release_lateral_only_active"]
            is True
        )
        assert (
            info["post_merge_offensive_anchor_blend_release_lateral_only_steps_since_release"]
            == pytest.approx(1.0)
        )
        assert (
            info["post_merge_offensive_anchor_blend_release_lateral_only_hold_remaining_steps"]
            == pytest.approx(1.0)
        )
        assert (
            info["post_merge_offensive_anchor_blend_release_lateral_only_hold_window_open"]
            is True
        )
        assert np.allclose(
            captured[-1]["offensive_anchor_lateral_world_offset_override"],
            computed_lateral_world_offset,
        )
        assert captured[-1]["offensive_anchor_lateral_blend_override"] == pytest.approx(
            1.0
        )

        _, _, _, _, info = env.step(np.zeros(3))
        assert (
            info["post_merge_offensive_anchor_blend_release_lateral_only_steps_since_release"]
            == pytest.approx(2.0)
        )
        assert (
            info["post_merge_offensive_anchor_blend_release_lateral_only_hold_remaining_steps"]
            == pytest.approx(0.0)
        )
        assert (
            info["post_merge_offensive_anchor_blend_release_lateral_only_hold_window_open"]
            is False
        )
        assert (
            info["post_merge_offensive_anchor_blend_release_lateral_only_active"]
            is False
        )
        assert captured[-1]["offensive_anchor_lateral_world_offset_override"] is None
        assert captured[-1]["offensive_anchor_lateral_blend_override"] is None
        assert info["post_merge_offensive_anchor_blend_release_direct_track_active"] is False
        env.close()

    def test_post_merge_offensive_anchor_blend_zero_blend_does_not_arm_release(
        self, base_config
    ):
        config = copy.deepcopy(base_config)
        config["task"] = {
            "name": "crossing_feasible",
            "env_class": "CloseRangeTrackingEnv",
        }
        config["virtual_point"]["anchor_mode"] = "predicted_target"
        config["virtual_point"]["post_merge_offensive_anchor_blend_by_task"] = {
            "crossing_feasible": 0.0,
        }
        config["virtual_point"][
            "post_merge_offensive_anchor_blend_release_ego_only_streak_steps_by_task"
        ] = {
            "crossing_feasible": 0,
        }
        config["trajectory_prediction"] = {"enabled": True}
        env = CloseRangeTrackingEnv(config)
        env.reset(seed=0)

        class _FakePredictor:
            def predict(self, target_state):
                target_pos = np.asarray(target_state["position_m"], dtype=np.float64)
                return target_pos + np.array([100.0, 0.0, 0.0]), None, {}

        env.trajectory_predictor_adapter = _FakePredictor()
        env._first_pass_complete = True
        env._first_pass_completion_step = 0
        env.combat_hp.step = lambda own_state, target_state: {
            "ego_in_attack_zone": True,
            "target_in_attack_zone": False,
        }

        _, _, _, _, info = env.step(np.zeros(3))
        assert info["post_merge_offensive_anchor_blend"] == pytest.approx(0.0)
        assert info["post_merge_offensive_anchor_blend_active"] is False
        assert info["post_merge_offensive_anchor_blend_release_triggered"] is False
        assert info["post_merge_offensive_anchor_blend_released"] is False

        _, _, _, _, info = env.step(np.zeros(3))
        assert info["post_merge_offensive_anchor_blend_active"] is False
        assert info["post_merge_offensive_anchor_blend_release_triggered"] is False
        assert info["post_merge_offensive_anchor_blend_released"] is False
        env.close()

    def test_post_merge_offensive_anchor_blend_release_can_request_direct_track_recovery(
        self,
        base_config,
    ):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["virtual_point"]["anchor_mode"] = "predicted_target"
        config["virtual_point"]["post_merge_offensive_anchor_blend_by_task"] = {
            "head_on": 0.25,
        }
        config["virtual_point"][
            "post_merge_offensive_anchor_blend_release_blend_by_task"
        ] = {
            "head_on": 0.0,
        }
        config["virtual_point"][
            "post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m_by_task"
        ] = {
            "head_on": 2000.0,
        }
        config["trajectory_prediction"] = {"enabled": True}
        env = CloseRangeTrackingEnv(config)
        scenario = {
            "own_init": {
                "position_m": [0.0, 0.0, 1000.0],
                "velocity_mps": 200.0,
                "heading_deg": 0.0,
            },
            "target_init": {
                "position_m": [1500.0, 0.0, 1000.0],
                "velocity_mps": 200.0,
                "heading_deg": 0.0,
            },
        }
        env.reset(scenario=scenario, seed=0)

        class _FakePredictor:
            def predict(self, target_state):
                target_pos = np.asarray(target_state["position_m"], dtype=np.float64)
                return target_pos + np.array([100.0, 0.0, 0.0]), None, {}

        env.trajectory_predictor_adapter = _FakePredictor()
        env._first_pass_complete = True
        env._first_pass_completion_step = 0
        env._post_merge_offensive_anchor_blend_released = True
        env._evaluate_post_merge_offensive_anchor_geometry_disadvantage = (
            lambda own_state, target_state, rel_state=None: {
                "requires_geometry_disadvantage": True,
                "condition_met": False,
                "target_only_attack_zone_disadvantage": False,
                "geometry_disadvantage": False,
                "alignment_disadvantage": False,
                "ego_attack_score": 0.0,
                "target_attack_score": 0.0,
                "ego_in_attack_zone": False,
                "target_in_attack_zone": False,
                "aa_deg_min": np.nan,
                "aa_deg": np.nan,
                "range_rate_mps": 250.0,
                "range_opening": True,
            }
        )

        def _unexpected_vpp(*args, **kwargs):
            raise AssertionError("direct-track recovery should bypass VPP generation")

        env.virtual_point_generator.action_to_virtual_point = _unexpected_vpp

        _, _, _, _, info = env.step(np.zeros(3))
        assert (
            info["post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m"]
            == pytest.approx(2000.0)
        )
        assert info["post_merge_offensive_anchor_blend_release_direct_track_active"] is True
        assert info["direct_track_mode_requested"] is True
        assert info["direct_track_mode_effective"] is True
        assert info["virtual_point_source"] == "direct_track"
        env.close()

    def test_post_merge_offensive_anchor_blend_release_recovery_stage_can_override_direct_track(
        self,
        base_config,
    ):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["virtual_point"]["anchor_mode"] = "predicted_target"
        config["virtual_point"]["post_merge_offensive_anchor_blend_by_task"] = {
            "head_on": 0.25,
        }
        config["virtual_point"][
            "post_merge_offensive_anchor_blend_release_blend_by_task"
        ] = {
            "head_on": 0.0,
        }
        config["virtual_point"][
            "post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m_by_task"
        ] = {
            "head_on": 5000.0,
        }
        config["virtual_point"][
            "post_merge_offensive_anchor_lateral_world_offset_latch_on_activation_by_task"
        ] = {
            "head_on": True,
        }
        config["virtual_point"][
            "post_merge_offensive_anchor_blend_release_recovery_below_altitude_m_by_task"
        ] = {
            "head_on": 5000.0,
        }
        config["virtual_point"][
            "post_merge_offensive_anchor_blend_release_recovery_longitudinal_blend_by_task"
        ] = {
            "head_on": 0.0,
        }
        config["virtual_point"][
            "post_merge_offensive_anchor_blend_release_recovery_lateral_blend_by_task"
        ] = {
            "head_on": 0.25,
        }
        config["trajectory_prediction"] = {"enabled": True}
        env = CloseRangeTrackingEnv(config)
        scenario = {
            "own_init": {
                "position_m": [0.0, 0.0, 1000.0],
                "velocity_mps": 200.0,
                "heading_deg": 0.0,
            },
            "target_init": {
                "position_m": [1500.0, 0.0, 1000.0],
                "velocity_mps": 200.0,
                "heading_deg": 0.0,
            },
        }
        env.reset(scenario=scenario, seed=0)

        class _FakePredictor:
            def predict(self, target_state):
                target_pos = np.asarray(target_state["position_m"], dtype=np.float64)
                return target_pos + np.array([100.0, 0.0, 0.0]), None, {}

        env.trajectory_predictor_adapter = _FakePredictor()
        env._first_pass_complete = True
        env._first_pass_completion_step = 0
        env._post_merge_offensive_anchor_blend_released = True
        env._post_merge_offensive_anchor_lateral_world_offset_latched = np.array(
            [120.0, 40.0, 0.0],
            dtype=np.float64,
        )
        env._evaluate_post_merge_offensive_anchor_geometry_disadvantage = (
            lambda own_state, target_state, rel_state=None: {
                "requires_geometry_disadvantage": True,
                "condition_met": False,
                "target_only_attack_zone_disadvantage": False,
                "geometry_disadvantage": False,
                "alignment_disadvantage": False,
                "ego_attack_score": 0.0,
                "target_attack_score": 0.0,
                "ego_in_attack_zone": False,
                "target_in_attack_zone": False,
                "aa_deg_min": np.nan,
                "aa_deg": np.nan,
                "range_rate_mps": 250.0,
                "range_opening": True,
            }
        )
        captured = {}

        def _fake_action_to_virtual_point(
            action,
            own_state,
            target_state,
            anchor_mode="current_target",
            **kwargs,
        ):
            captured["offensive_anchor_blend_override"] = kwargs.get(
                "offensive_anchor_blend_override"
            )
            captured["offensive_anchor_longitudinal_blend_override"] = kwargs.get(
                "offensive_anchor_longitudinal_blend_override"
            )
            captured["offensive_anchor_lateral_blend_override"] = kwargs.get(
                "offensive_anchor_lateral_blend_override"
            )
            captured["offensive_anchor_lateral_world_offset_override"] = kwargs.get(
                "offensive_anchor_lateral_world_offset_override"
            )
            target_pos = np.asarray(target_state["position_neu"], dtype=np.float64)
            return (
                {
                    "position_neu": target_pos.copy(),
                    "position": target_pos.copy(),
                },
                {
                    "anchor_mode": anchor_mode,
                    "anchor_pos": target_pos.copy(),
                    "offset": np.zeros(3, dtype=np.float64),
                    "world_offset": np.zeros(3, dtype=np.float64),
                    "offset_frame": "world_neu",
                    "predicted_target_blend": 1.0,
                    "predicted_target_forward_scale": 1.0,
                    "offensive_anchor_blend": (
                        kwargs["offensive_anchor_blend_override"]
                        if kwargs.get("offensive_anchor_blend_override") is not None
                        else 0.0
                    ),
                    "longitudinal_scale": 1.0,
                    "lateral_scale": 1.0,
                },
            )

        env.virtual_point_generator.action_to_virtual_point = _fake_action_to_virtual_point

        _, _, _, _, info = env.step(np.zeros(3))

        assert (
            info["post_merge_offensive_anchor_blend_release_recovery_below_altitude_m"]
            == pytest.approx(5000.0)
        )
        assert (
            info[
                "post_merge_offensive_anchor_blend_release_recovery_longitudinal_blend"
            ]
            == pytest.approx(0.0)
        )
        assert (
            info["post_merge_offensive_anchor_blend_release_recovery_lateral_blend"]
            == pytest.approx(0.25)
        )
        assert info["post_merge_offensive_anchor_blend_release_recovery_active"] is True
        assert info["post_merge_offensive_anchor_blend_release_direct_track_active"] is False
        assert info["direct_track_mode_requested"] is False
        assert info["direct_track_mode_effective"] is False
        assert info["virtual_point_source"] == "vpp_policy"
        assert captured["offensive_anchor_blend_override"] == pytest.approx(0.0)
        assert captured["offensive_anchor_longitudinal_blend_override"] == pytest.approx(
            0.0
        )
        assert captured["offensive_anchor_lateral_blend_override"] == pytest.approx(
            0.25
        )
        assert np.allclose(
            captured["offensive_anchor_lateral_world_offset_override"],
            np.array([120.0, 40.0, 0.0]),
        )
        env.close()

    def test_post_merge_offensive_anchor_blend_release_recovery_forward_bias_trigger_can_activate_early(
        self,
        base_config,
    ):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["virtual_point"]["anchor_mode"] = "predicted_target"
        config["virtual_point"]["post_merge_offensive_anchor_blend_by_task"] = {
            "head_on": 0.25,
        }
        config["virtual_point"][
            "post_merge_offensive_anchor_blend_release_blend_by_task"
        ] = {
            "head_on": 0.0,
        }
        config["virtual_point"][
            "post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m_by_task"
        ] = {
            "head_on": 5000.0,
        }
        config["virtual_point"][
            "post_merge_offensive_anchor_lateral_world_offset_latch_on_activation_by_task"
        ] = {
            "head_on": True,
        }
        config["virtual_point"][
            "post_merge_offensive_anchor_blend_release_recovery_longitudinal_blend_by_task"
        ] = {
            "head_on": 0.0,
        }
        config["virtual_point"][
            "post_merge_offensive_anchor_blend_release_recovery_lateral_blend_by_task"
        ] = {
            "head_on": 0.25,
        }
        config["virtual_point"][
            "post_merge_offensive_anchor_blend_release_recovery_forward_bias_m_max_by_task"
        ] = {
            "head_on": -1000.0,
        }
        config["trajectory_prediction"] = {"enabled": True}
        env = CloseRangeTrackingEnv(config)
        scenario = {
            "own_init": {
                "position_m": [0.0, 0.0, 1000.0],
                "velocity_mps": 200.0,
                "heading_deg": 0.0,
            },
            "target_init": {
                "position_m": [1500.0, 0.0, 1000.0],
                "velocity_mps": 200.0,
                "heading_deg": 0.0,
            },
        }
        env.reset(scenario=scenario, seed=0)

        class _FakePredictor:
            def predict(self, target_state):
                target_pos = np.asarray(target_state["position_m"], dtype=np.float64)
                return target_pos + np.array([100.0, 0.0, 0.0]), None, {}

        env.trajectory_predictor_adapter = _FakePredictor()
        env._first_pass_complete = True
        env._first_pass_completion_step = 0
        env._post_merge_offensive_anchor_blend_released = True
        env._post_merge_offensive_anchor_lateral_world_offset_latched = np.array(
            [120.0, 40.0, 0.0],
            dtype=np.float64,
        )
        env._evaluate_post_merge_offensive_anchor_geometry_disadvantage = (
            lambda own_state, target_state, rel_state=None: {
                "requires_geometry_disadvantage": True,
                "condition_met": False,
                "target_only_attack_zone_disadvantage": False,
                "geometry_disadvantage": False,
                "alignment_disadvantage": False,
                "ego_attack_score": 0.0,
                "target_attack_score": 0.0,
                "ego_in_attack_zone": False,
                "target_in_attack_zone": False,
                "aa_deg_min": np.nan,
                "aa_deg": np.nan,
                "range_rate_mps": 250.0,
                "range_opening": True,
            }
        )
        captured = {"calls": 0}

        def _fake_action_to_virtual_point(
            action,
            own_state,
            target_state,
            anchor_mode="current_target",
            **kwargs,
        ):
            captured["calls"] += 1
            captured["offensive_anchor_blend_override"] = kwargs.get(
                "offensive_anchor_blend_override"
            )
            captured["offensive_anchor_longitudinal_blend_override"] = kwargs.get(
                "offensive_anchor_longitudinal_blend_override"
            )
            captured["offensive_anchor_lateral_blend_override"] = kwargs.get(
                "offensive_anchor_lateral_blend_override"
            )
            captured["offensive_anchor_lateral_world_offset_override"] = kwargs.get(
                "offensive_anchor_lateral_world_offset_override"
            )
            target_pos = np.asarray(target_state["position_neu"], dtype=np.float64)
            vp_pos = target_pos + np.array([-1500.0, 0.0, 0.0], dtype=np.float64)
            return (
                {
                    "position_neu": vp_pos.copy(),
                    "position": vp_pos.copy(),
                },
                {
                    "anchor_mode": anchor_mode,
                    "anchor_pos": target_pos.copy(),
                    "offset": np.zeros(3, dtype=np.float64),
                    "world_offset": vp_pos - target_pos,
                    "offset_frame": "world_neu",
                    "predicted_target_blend": 1.0,
                    "predicted_target_forward_scale": 1.0,
                    "offensive_anchor_blend": (
                        kwargs["offensive_anchor_blend_override"]
                        if kwargs.get("offensive_anchor_blend_override") is not None
                        else 0.0
                    ),
                    "longitudinal_scale": 1.0,
                    "lateral_scale": 1.0,
                },
            )

        env.virtual_point_generator.action_to_virtual_point = _fake_action_to_virtual_point

        _, _, _, _, info = env.step(np.zeros(3))

        assert (
            info["post_merge_offensive_anchor_blend_release_recovery_forward_bias_m_max"]
            == pytest.approx(-1000.0)
        )
        assert info["post_merge_offensive_anchor_blend_release_recovery_active"] is True
        assert (
            info["post_merge_offensive_anchor_blend_release_recovery_altitude_trigger_active"]
            is False
        )
        assert (
            info["post_merge_offensive_anchor_blend_release_recovery_forward_bias_trigger_active"]
            is True
        )
        assert (
            info["post_merge_offensive_anchor_blend_release_recovery_preview_vp_forward_bias_m"]
            == pytest.approx(-1500.0)
        )
        assert info["post_merge_offensive_anchor_blend_release_direct_track_active"] is False
        assert info["direct_track_mode_requested"] is False
        assert info["direct_track_mode_effective"] is False
        assert info["virtual_point_source"] == "vpp_policy"
        assert captured["calls"] == 2
        assert captured["offensive_anchor_blend_override"] == pytest.approx(0.0)
        assert captured["offensive_anchor_longitudinal_blend_override"] == pytest.approx(
            0.0
        )
        assert captured["offensive_anchor_lateral_blend_override"] == pytest.approx(
            0.25
        )
        assert np.allclose(
            captured["offensive_anchor_lateral_world_offset_override"],
            np.array([120.0, 40.0, 0.0]),
        )
        env.close()

    def test_post_merge_predicted_target_forward_scale_preserves_lateral_component(
        self, base_config
    ):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["virtual_point"]["anchor_mode"] = "predicted_target"
        config["virtual_point"]["close_range_anchor_mode"] = None
        config["virtual_point"]["close_range_anchor_mode_by_task"] = {}
        config["virtual_point"][
            "post_merge_predicted_target_forward_scale_by_task"
        ] = {
            "head_on": 0.0,
        }
        config["trajectory_prediction"]["enabled"] = True

        class _DummyPredictor:
            def reset(self):
                return None

            def predict(self, target_state):
                predicted = np.asarray(target_state["position_m"], dtype=np.float64) + np.array(
                    [-200.0, 200.0, 0.0]
                )
                return predicted, None, {"model_type": "dummy", "fallback": False}

        env = CloseRangeTrackingEnv(config)
        env.trajectory_predictor_adapter = _DummyPredictor()
        env.reset(seed=0)
        env._first_pass_complete = True
        _, _, _, _, info = env.step(np.zeros(3))

        assert info["anchor_mode_requested"] == "predicted_target"
        assert info["anchor_mode"] == "predicted_target"
        assert info["post_merge_predicted_target_forward_scale"] == pytest.approx(
            0.0
        )
        assert info["post_merge_predicted_target_forward_scale_active"] is True
        assert info["predicted_target_forward_scale"] == pytest.approx(0.0)
        env.close()

    def test_predicted_target_forward_scale_by_task_applies_before_first_pass(
        self, base_config
    ):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["virtual_point"]["anchor_mode"] = "predicted_target"
        config["virtual_point"]["close_range_anchor_mode"] = None
        config["virtual_point"]["close_range_anchor_mode_by_task"] = {}
        config["virtual_point"]["predicted_target_forward_scale_by_task"] = {
            "head_on": 0.0,
        }
        config["trajectory_prediction"]["enabled"] = True

        class _DummyPredictor:
            def reset(self):
                return None

            def predict(self, target_state):
                predicted = np.asarray(
                    target_state["position_m"], dtype=np.float64
                ) + np.array([-200.0, 200.0, 0.0])
                return predicted, None, {"model_type": "dummy", "fallback": False}

        env = CloseRangeTrackingEnv(config)
        env.trajectory_predictor_adapter = _DummyPredictor()
        env.reset(seed=0)
        env._first_pass_complete = False
        _, _, _, _, info = env.step(np.zeros(3))

        assert info["anchor_mode_requested"] == "predicted_target"
        assert info["anchor_mode"] == "predicted_target"
        assert info["predicted_target_forward_scale"] == pytest.approx(0.0)
        assert info["post_merge_predicted_target_forward_scale_active"] is False
        env.close()

    def test_post_merge_predicted_target_forward_scale_can_expire_after_hold_steps(
        self, base_config
    ):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["virtual_point"]["anchor_mode"] = "predicted_target"
        config["virtual_point"]["post_merge_predicted_target_forward_scale_by_task"] = {
            "head_on": 0.0,
        }
        config["virtual_point"][
            "post_merge_predicted_target_forward_scale_hold_steps_by_task"
        ] = {
            "head_on": 1,
        }
        config["trajectory_prediction"] = {"enabled": True}
        env = CloseRangeTrackingEnv(config)
        env.reset(seed=0)

        class _FakePredictor:
            def predict(self, target_state):
                target_pos = np.asarray(target_state["position_m"], dtype=np.float64)
                return target_pos + np.array([100.0, 0.0, 0.0]), None, {}

        env.trajectory_predictor_adapter = _FakePredictor()

        captured = {}

        def _fake_action_to_virtual_point(
            action,
            own_state,
            target_state,
            anchor_mode="current_target",
            **kwargs,
        ):
            captured["predicted_target_forward_scale_override"] = kwargs.get(
                "predicted_target_forward_scale_override"
            )
            target_pos = np.asarray(target_state["position_neu"], dtype=np.float64)
            return (
                {
                    "position_neu": target_pos.copy(),
                    "position": target_pos.copy(),
                },
                {
                    "anchor_mode": anchor_mode,
                    "anchor_pos": target_pos.copy(),
                    "offset": np.zeros(3, dtype=np.float64),
                    "world_offset": np.zeros(3, dtype=np.float64),
                    "offset_frame": "world_neu",
                    "predicted_target_blend": 1.0,
                    "predicted_target_forward_scale": (
                        kwargs["predicted_target_forward_scale_override"]
                        if kwargs.get("predicted_target_forward_scale_override")
                        is not None
                        else 1.0
                    ),
                    "lateral_scale": 1.0,
                },
            )

        env.virtual_point_generator.action_to_virtual_point = _fake_action_to_virtual_point
        env.combat_hp.step = lambda own_state, target_state: {
            "ego_in_attack_zone": False,
            "target_in_attack_zone": False,
        }

        env._first_pass_complete = True
        env._first_pass_completion_step = 0

        _, _, _, _, info = env.step(np.zeros(3))
        assert info["post_merge_predicted_target_forward_scale_hold_steps"] == pytest.approx(
            1.0
        )
        assert info[
            "post_merge_predicted_target_forward_scale_steps_since_first_pass"
        ] == pytest.approx(1.0)
        assert info[
            "post_merge_predicted_target_forward_scale_hold_remaining_steps"
        ] == pytest.approx(1.0)
        assert info["post_merge_predicted_target_forward_scale_hold_window_open"] is True
        assert info["post_merge_predicted_target_forward_scale_hold_expired"] is False
        assert info["post_merge_predicted_target_forward_scale_active"] is True
        assert captured["predicted_target_forward_scale_override"] == pytest.approx(0.0)

        _, _, _, _, info = env.step(np.zeros(3))
        assert info[
            "post_merge_predicted_target_forward_scale_steps_since_first_pass"
        ] == pytest.approx(2.0)
        assert info[
            "post_merge_predicted_target_forward_scale_hold_remaining_steps"
        ] == pytest.approx(0.0)
        assert info["post_merge_predicted_target_forward_scale_hold_window_open"] is False
        assert info["post_merge_predicted_target_forward_scale_hold_expired"] is True
        assert info["post_merge_predicted_target_forward_scale_active"] is False
        assert captured["predicted_target_forward_scale_override"] is None
        env.close()

    def test_post_merge_predicted_target_forward_scale_can_release_after_ego_only_streak(
        self, base_config
    ):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["virtual_point"]["anchor_mode"] = "predicted_target"
        config["virtual_point"]["post_merge_predicted_target_forward_scale_by_task"] = {
            "head_on": 0.0,
        }
        config["virtual_point"][
            "post_merge_predicted_target_forward_scale_release_ego_only_streak_steps_by_task"
        ] = {
            "head_on": 1,
        }
        config["trajectory_prediction"] = {"enabled": True}
        env = CloseRangeTrackingEnv(config)
        env.reset(seed=0)

        class _FakePredictor:
            def predict(self, target_state):
                target_pos = np.asarray(target_state["position_m"], dtype=np.float64)
                return target_pos + np.array([100.0, 0.0, 0.0]), None, {}

        env.trajectory_predictor_adapter = _FakePredictor()

        captured = []

        def _fake_action_to_virtual_point(
            action,
            own_state,
            target_state,
            anchor_mode="current_target",
            **kwargs,
        ):
            captured.append(kwargs.get("predicted_target_forward_scale_override"))
            target_pos = np.asarray(target_state["position_neu"], dtype=np.float64)
            return (
                {
                    "position_neu": target_pos.copy(),
                    "position": target_pos.copy(),
                },
                {
                    "anchor_mode": anchor_mode,
                    "anchor_pos": target_pos.copy(),
                    "offset": np.zeros(3, dtype=np.float64),
                    "world_offset": np.zeros(3, dtype=np.float64),
                    "offset_frame": "world_neu",
                    "predicted_target_blend": 1.0,
                    "predicted_target_forward_scale": (
                        kwargs["predicted_target_forward_scale_override"]
                        if kwargs.get("predicted_target_forward_scale_override")
                        is not None
                        else 1.0
                    ),
                    "lateral_scale": 1.0,
                },
            )

        env.virtual_point_generator.action_to_virtual_point = _fake_action_to_virtual_point
        env.combat_hp.step = lambda own_state, target_state: {
            "ego_in_attack_zone": True,
            "target_in_attack_zone": False,
        }

        env._first_pass_complete = True
        env._first_pass_completion_step = 0

        _, _, _, _, info = env.step(np.zeros(3))
        assert info[
            "post_merge_predicted_target_forward_scale_release_ego_only_streak_steps"
        ] == pytest.approx(1.0)
        assert info[
            "post_merge_predicted_target_forward_scale_ego_only_streak_steps"
        ] == pytest.approx(1.0)
        assert info["post_merge_predicted_target_forward_scale_active"] is True
        assert info["post_merge_predicted_target_forward_scale_release_triggered"] is True
        assert info["post_merge_predicted_target_forward_scale_released"] is True
        assert captured[-1] == pytest.approx(0.0)

        _, _, _, _, info = env.step(np.zeros(3))
        assert info["post_merge_predicted_target_forward_scale_active"] is False
        assert info["post_merge_predicted_target_forward_scale_released"] is True
        assert captured[-1] is None
        env.close()

    def test_post_merge_predicted_target_forward_scale_can_fall_back_to_release_scale(
        self, base_config
    ):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["virtual_point"]["anchor_mode"] = "predicted_target"
        config["virtual_point"]["post_merge_predicted_target_forward_scale_by_task"] = {
            "head_on": 0.0,
        }
        config["virtual_point"][
            "post_merge_predicted_target_forward_scale_release_ego_only_streak_steps_by_task"
        ] = {
            "head_on": 1,
        }
        config["virtual_point"][
            "post_merge_predicted_target_forward_scale_release_scale_by_task"
        ] = {
            "head_on": 0.5,
        }
        config["trajectory_prediction"] = {"enabled": True}
        env = CloseRangeTrackingEnv(config)
        env.reset(seed=0)

        class _FakePredictor:
            def predict(self, target_state):
                target_pos = np.asarray(target_state["position_m"], dtype=np.float64)
                return target_pos + np.array([100.0, 0.0, 0.0]), None, {}

        env.trajectory_predictor_adapter = _FakePredictor()

        captured = []

        def _fake_action_to_virtual_point(
            action,
            own_state,
            target_state,
            anchor_mode="current_target",
            **kwargs,
        ):
            captured.append(kwargs.get("predicted_target_forward_scale_override"))
            target_pos = np.asarray(target_state["position_neu"], dtype=np.float64)
            return (
                {
                    "position_neu": target_pos.copy(),
                    "position": target_pos.copy(),
                },
                {
                    "anchor_mode": anchor_mode,
                    "anchor_pos": target_pos.copy(),
                    "offset": np.zeros(3, dtype=np.float64),
                    "world_offset": np.zeros(3, dtype=np.float64),
                    "offset_frame": "world_neu",
                    "predicted_target_blend": 1.0,
                    "predicted_target_forward_scale": (
                        kwargs["predicted_target_forward_scale_override"]
                        if kwargs.get("predicted_target_forward_scale_override")
                        is not None
                        else 1.0
                    ),
                    "lateral_scale": 1.0,
                },
            )

        env.virtual_point_generator.action_to_virtual_point = _fake_action_to_virtual_point
        env.combat_hp.step = lambda own_state, target_state: {
            "ego_in_attack_zone": True,
            "target_in_attack_zone": False,
        }

        env._first_pass_complete = True
        env._first_pass_completion_step = 0

        _, _, _, _, info = env.step(np.zeros(3))
        assert info["post_merge_predicted_target_forward_scale_release_triggered"] is True
        assert info["post_merge_predicted_target_forward_scale_release_scale"] == pytest.approx(
            0.5
        )
        assert info["post_merge_predicted_target_forward_scale_release_scale_active"] is False
        assert captured[-1] == pytest.approx(0.0)

        _, _, _, _, info = env.step(np.zeros(3))
        assert info["post_merge_predicted_target_forward_scale_active"] is False
        assert info["post_merge_predicted_target_forward_scale_released"] is True
        assert info["post_merge_predicted_target_forward_scale_release_scale_active"] is True
        assert captured[-1] == pytest.approx(0.5)
        env.close()

    def test_post_merge_predicted_target_forward_scale_release_can_reset_after_streak_break(
        self, base_config
    ):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["virtual_point"]["anchor_mode"] = "predicted_target"
        config["virtual_point"]["post_merge_predicted_target_forward_scale_by_task"] = {
            "head_on": 0.0,
        }
        config["virtual_point"][
            "post_merge_predicted_target_forward_scale_release_ego_only_streak_steps_by_task"
        ] = {
            "head_on": 1,
        }
        config["virtual_point"][
            "post_merge_predicted_target_forward_scale_release_scale_by_task"
        ] = {
            "head_on": 0.5,
        }
        config["virtual_point"][
            "post_merge_predicted_target_forward_scale_release_reset_on_streak_break_by_task"
        ] = {
            "head_on": True,
        }
        config["trajectory_prediction"] = {"enabled": True}
        env = CloseRangeTrackingEnv(config)
        env.reset(seed=0)

        class _FakePredictor:
            def predict(self, target_state):
                target_pos = np.asarray(target_state["position_m"], dtype=np.float64)
                return target_pos + np.array([100.0, 0.0, 0.0]), None, {}

        env.trajectory_predictor_adapter = _FakePredictor()

        captured = []
        attack_zone_states = iter(
            [
                {"ego_in_attack_zone": True, "target_in_attack_zone": False},
                {"ego_in_attack_zone": False, "target_in_attack_zone": False},
                {"ego_in_attack_zone": False, "target_in_attack_zone": False},
            ]
        )

        def _fake_action_to_virtual_point(
            action,
            own_state,
            target_state,
            anchor_mode="current_target",
            **kwargs,
        ):
            captured.append(kwargs.get("predicted_target_forward_scale_override"))
            target_pos = np.asarray(target_state["position_neu"], dtype=np.float64)
            return (
                {
                    "position_neu": target_pos.copy(),
                    "position": target_pos.copy(),
                },
                {
                    "anchor_mode": anchor_mode,
                    "anchor_pos": target_pos.copy(),
                    "offset": np.zeros(3, dtype=np.float64),
                    "world_offset": np.zeros(3, dtype=np.float64),
                    "offset_frame": "world_neu",
                    "predicted_target_blend": 1.0,
                    "predicted_target_forward_scale": (
                        kwargs["predicted_target_forward_scale_override"]
                        if kwargs.get("predicted_target_forward_scale_override")
                        is not None
                        else 1.0
                    ),
                    "lateral_scale": 1.0,
                },
            )

        env.virtual_point_generator.action_to_virtual_point = _fake_action_to_virtual_point
        env.combat_hp.step = lambda own_state, target_state: next(attack_zone_states)

        env._first_pass_complete = True
        env._first_pass_completion_step = 0

        _, _, _, _, info = env.step(np.zeros(3))
        assert (
            info[
                "post_merge_predicted_target_forward_scale_release_reset_on_streak_break"
            ]
            is True
        )
        assert info["post_merge_predicted_target_forward_scale_release_triggered"] is True
        assert info["post_merge_predicted_target_forward_scale_release_reset_triggered"] is False
        assert info["post_merge_predicted_target_forward_scale_released"] is True
        assert captured[-1] == pytest.approx(0.0)

        _, _, _, _, info = env.step(np.zeros(3))
        assert info["post_merge_predicted_target_forward_scale_release_scale_active"] is True
        assert info["post_merge_predicted_target_forward_scale_release_reset_triggered"] is True
        assert info["post_merge_predicted_target_forward_scale_released"] is False
        assert captured[-1] == pytest.approx(0.5)

        _, _, _, _, info = env.step(np.zeros(3))
        assert info["post_merge_predicted_target_forward_scale_active"] is True
        assert info["post_merge_predicted_target_forward_scale_release_scale_active"] is False
        assert info["post_merge_predicted_target_forward_scale_released"] is False
        assert captured[-1] == pytest.approx(0.0)
        env.close()

    def test_close_range_anchor_mode_switches_for_task(self, base_config):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["virtual_point"]["anchor_mode"] = "predicted_target"
        config["virtual_point"]["close_range_anchor_mode_by_task"] = {
            "head_on": "current_target",
        }
        env = CloseRangeTrackingEnv(config)
        env.reset(seed=0)
        env._close_range_anchor_ready = True
        _, _, _, _, info = env.step(np.zeros(3))

        assert info["anchor_mode_requested"] == "predicted_target"
        assert info["close_range_anchor_mode"] == "current_target"
        assert info["close_range_anchor_mode_active"] is True
        assert info["anchor_mode"] == "current_target"
        env.close()

    def test_close_range_anchor_mode_respects_trigger_range_by_task(self, base_config):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["virtual_point"]["anchor_mode"] = "predicted_target"
        config["virtual_point"]["close_range_anchor_mode_by_task"] = {
            "head_on": "current_target",
        }
        config["virtual_point"]["close_range_anchor_trigger_range_m_by_task"] = {
            "head_on": 150.0,
        }
        env = CloseRangeTrackingEnv(config)
        env.reset(seed=0)
        env._merge_seen_close_range = True
        _, _, _, _, info = env.step(np.zeros(3))

        assert info["close_range_anchor_trigger_range_m"] == pytest.approx(150.0)
        assert info["close_range_anchor_mode_active"] is False

        env._close_range_anchor_ready = True
        _, _, _, _, info = env.step(np.zeros(3))
        assert info["close_range_anchor_mode_active"] is True
        assert info["anchor_mode"] == "current_target"
        env.close()

    def test_close_range_anchor_mode_can_require_alignment_gate(self, base_config):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["virtual_point"]["anchor_mode"] = "predicted_target"
        config["virtual_point"]["close_range_anchor_mode_by_task"] = {
            "head_on": "current_target",
        }
        config["virtual_point"]["close_range_anchor_alignment_angle_deg_max_by_task"] = {
            "head_on": 10.0,
        }
        env = CloseRangeTrackingEnv(config)
        env.reset(seed=0)

        env._update_merge_diagnostics(
            {
                "range_m": 500.0,
                "range_rate_mps": -100.0,
                "ata_rad": np.deg2rad(14.0),
                "aa_rad": np.deg2rad(3.0),
            }
        )
        assert env._close_range_anchor_ready is False

        env._update_merge_diagnostics(
            {
                "range_m": 500.0,
                "range_rate_mps": -100.0,
                "ata_rad": np.deg2rad(4.0),
                "aa_rad": np.deg2rad(3.0),
            }
        )
        assert env._close_range_anchor_ready is True

        _, _, _, _, info = env.step(np.zeros(3))
        assert info["close_range_anchor_alignment_angle_deg_max"] == pytest.approx(
            10.0
        )
        assert info["close_range_anchor_mode_active"] is True
        assert info["anchor_mode"] == "current_target"
        env.close()

    def test_close_range_anchor_release_alignment_gate_uses_radian_fallback(
        self, base_config
    ):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["virtual_point"][
            "close_range_anchor_release_alignment_angle_deg_max_by_task"
        ] = {
            "head_on": 10.0,
        }
        env = CloseRangeTrackingEnv(config)
        env.reset(seed=0)

        satisfied, angle_deg_max = env._evaluate_close_range_anchor_release_alignment(
            {
                "ata_rad": np.deg2rad(4.0),
                "aa_rad": np.deg2rad(3.0),
            }
        )
        assert satisfied is True
        assert angle_deg_max == pytest.approx(10.0)

        satisfied, angle_deg_max = env._evaluate_close_range_anchor_release_alignment(
            {
                "ata_rad": np.deg2rad(14.0),
                "aa_rad": np.deg2rad(3.0),
            }
        )
        assert satisfied is False
        assert angle_deg_max == pytest.approx(10.0)
        env.close()

    def test_close_range_anchor_can_release_on_post_merge(self, base_config):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["virtual_point"]["anchor_mode"] = "predicted_target"
        config["virtual_point"]["close_range_anchor_mode_by_task"] = {
            "head_on": "current_target",
        }
        config["virtual_point"]["close_range_anchor_release_on_post_merge_by_task"] = {
            "head_on": True,
        }
        env = CloseRangeTrackingEnv(config)
        env.reset(seed=0)
        env._close_range_anchor_ready = True

        _, _, _, _, info = env.step(np.zeros(3))
        assert info["close_range_anchor_release_on_post_merge"] is True
        assert info["close_range_anchor_window_open"] is True
        assert info["close_range_anchor_mode_active"] is True
        assert info["anchor_mode"] == "current_target"

        env._first_pass_complete = True
        _, _, _, _, info = env.step(np.zeros(3))
        assert info["close_range_anchor_release_on_post_merge"] is True
        assert info["close_range_anchor_window_open"] is False
        assert info["close_range_anchor_mode_active"] is False
        env.close()

    def test_close_range_anchor_can_wait_until_first_pass_before_activating(
        self, base_config
    ):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["virtual_point"]["anchor_mode"] = "predicted_target"
        config["virtual_point"]["close_range_anchor_mode_by_task"] = {
            "head_on": "current_target",
        }
        config["virtual_point"]["close_range_anchor_requires_first_pass_by_task"] = {
            "head_on": True,
        }
        env = CloseRangeTrackingEnv(config)
        env.reset(seed=0)
        env._close_range_anchor_ready = True

        _, _, _, _, info = env.step(np.zeros(3))
        assert info["close_range_anchor_requires_first_pass"] is True
        assert info["close_range_anchor_mode_active"] is False

        env._first_pass_complete = True
        env._first_pass_completion_step = env.current_step
        _, _, _, _, info = env.step(np.zeros(3))
        assert info["close_range_anchor_requires_first_pass"] is True
        assert info["close_range_anchor_mode_active"] is True
        assert info["anchor_mode"] == "current_target"
        env.close()

    def test_close_range_anchor_can_use_offensive_anchor_blend_override(
        self, base_config
    ):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["virtual_point"]["anchor_mode"] = "predicted_target"
        config["virtual_point"]["close_range_anchor_mode_by_task"] = {
            "head_on": "offensive_position",
        }
        config["virtual_point"]["close_range_anchor_offensive_anchor_blend_by_task"] = {
            "head_on": 0.25,
        }
        env = CloseRangeTrackingEnv(config)
        env.reset(seed=0)
        env._close_range_anchor_ready = True

        captured = {}

        def _fake_action_to_virtual_point(
            action,
            own_state,
            target_state,
            anchor_mode="current_target",
            **kwargs,
        ):
            captured["anchor_mode"] = anchor_mode
            captured["offensive_anchor_blend_override"] = kwargs.get(
                "offensive_anchor_blend_override"
            )
            target_pos = np.asarray(target_state["position_neu"], dtype=np.float64)
            return (
                {
                    "position_neu": target_pos.copy(),
                    "position": target_pos.copy(),
                },
                {
                    "anchor_mode": anchor_mode,
                    "anchor_pos": target_pos.copy(),
                    "offset": np.zeros(3, dtype=np.float64),
                    "world_offset": np.zeros(3, dtype=np.float64),
                    "offset_frame": "world_neu",
                    "offensive_anchor_blend": kwargs.get(
                        "offensive_anchor_blend_override", 0.0
                    )
                    or 0.0,
                    "lateral_scale": 1.0,
                },
            )

        env.virtual_point_generator.action_to_virtual_point = _fake_action_to_virtual_point

        _, _, _, _, info = env.step(np.zeros(3))
        assert info["close_range_anchor_mode_active"] is True
        assert info["close_range_anchor_offensive_anchor_blend"] == pytest.approx(0.25)
        assert info["close_range_anchor_offensive_anchor_blend_active"] is True
        assert info["anchor_mode_requested"] == "predicted_target"
        assert captured["offensive_anchor_blend_override"] == pytest.approx(0.25)
        env.close()

    def test_close_range_anchor_release_can_wait_for_post_merge_alignment_gate(
        self, base_config
    ):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["virtual_point"]["anchor_mode"] = "predicted_target"
        config["virtual_point"]["close_range_anchor_mode_by_task"] = {
            "head_on": "current_target",
        }
        config["virtual_point"]["close_range_anchor_release_on_post_merge_by_task"] = {
            "head_on": True,
        }
        config["virtual_point"][
            "close_range_anchor_release_alignment_angle_deg_max_by_task"
        ] = {
            "head_on": 120.0,
        }
        env = CloseRangeTrackingEnv(config)
        env.reset(seed=0)
        env._close_range_anchor_ready = True
        env._first_pass_complete = True
        env._first_pass_completion_step = 0
        env._evaluate_close_range_anchor_release_alignment = (
            lambda rel_state: (False, 120.0)
        )

        _, _, _, _, info = env.step(np.zeros(3))
        assert info["close_range_anchor_release_alignment_angle_deg_max"] == pytest.approx(
            120.0
        )
        assert info["close_range_anchor_release_alignment_satisfied"] is False
        assert info["close_range_anchor_release_ready"] is False
        assert info["close_range_anchor_window_open"] is True
        assert info["close_range_anchor_mode_active"] is True
        assert info["anchor_mode"] == "current_target"

        env._evaluate_close_range_anchor_release_alignment = (
            lambda rel_state: (True, 120.0)
        )
        _, _, _, _, info = env.step(np.zeros(3))
        assert info["close_range_anchor_release_alignment_satisfied"] is True
        assert info["close_range_anchor_release_ready"] is True
        assert info["close_range_anchor_window_open"] is False
        assert info["close_range_anchor_mode_active"] is False
        assert info["anchor_mode_requested"] == "predicted_target"
        env.close()

    def test_post_merge_predicted_target_blend_override_is_applied_after_first_pass(
        self, base_config
    ):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["virtual_point"]["anchor_mode"] = "predicted_target"
        config["virtual_point"]["post_merge_predicted_target_blend_by_task"] = {
            "head_on": 0.5,
        }
        config["trajectory_prediction"] = {"enabled": True}
        env = CloseRangeTrackingEnv(config)
        env.reset(seed=0)

        class _FakePredictor:
            def predict(self, target_state):
                target_pos = np.asarray(target_state["position_m"], dtype=np.float64)
                return target_pos + np.array([100.0, 0.0, 0.0]), None, {}

        env.trajectory_predictor_adapter = _FakePredictor()

        captured = {}

        def _fake_action_to_virtual_point(
            action,
            own_state,
            target_state,
            anchor_mode="current_target",
            **kwargs,
        ):
            captured["anchor_mode"] = anchor_mode
            captured["predicted_target_blend_override"] = kwargs.get(
                "predicted_target_blend_override"
            )
            target_pos = np.asarray(target_state["position_neu"], dtype=np.float64)
            return (
                {
                    "position_neu": target_pos.copy(),
                    "position": target_pos.copy(),
                },
                {
                    "anchor_mode": anchor_mode,
                    "anchor_pos": target_pos.copy(),
                    "offset": np.zeros(3, dtype=np.float64),
                    "world_offset": np.zeros(3, dtype=np.float64),
                    "offset_frame": "world_neu",
                    "predicted_target_blend": kwargs.get(
                        "predicted_target_blend_override", 1.0
                    )
                    or 1.0,
                    "lateral_scale": 1.0,
                },
            )

        env.virtual_point_generator.action_to_virtual_point = _fake_action_to_virtual_point

        env._first_pass_complete = False
        _, _, _, _, info = env.step(np.zeros(3))
        assert captured["anchor_mode"] == "predicted_target"
        assert captured["predicted_target_blend_override"] is None
        assert info["post_merge_predicted_target_blend_active"] is False

        env._first_pass_complete = True
        _, _, _, _, info = env.step(np.zeros(3))
        assert captured["predicted_target_blend_override"] == pytest.approx(0.5)
        assert info["predicted_target_blend"] == pytest.approx(0.5)
        assert info["post_merge_predicted_target_blend"] == pytest.approx(0.5)
        assert info["post_merge_predicted_target_blend_active"] is True
        env.close()

    def test_post_merge_predicted_target_blend_can_release_on_attack_zone(
        self, base_config
    ):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["virtual_point"]["anchor_mode"] = "predicted_target"
        config["virtual_point"]["post_merge_predicted_target_blend_by_task"] = {
            "head_on": 0.5,
        }
        config["virtual_point"][
            "post_merge_predicted_target_blend_release_on_attack_zone_by_task"
        ] = {
            "head_on": True,
        }
        config["trajectory_prediction"] = {"enabled": True}
        env = CloseRangeTrackingEnv(config)
        env.reset(seed=0)

        class _FakePredictor:
            def predict(self, target_state):
                target_pos = np.asarray(target_state["position_m"], dtype=np.float64)
                return target_pos + np.array([100.0, 0.0, 0.0]), None, {}

        env.trajectory_predictor_adapter = _FakePredictor()

        captured = {}

        def _fake_action_to_virtual_point(
            action,
            own_state,
            target_state,
            anchor_mode="current_target",
            **kwargs,
        ):
            captured["predicted_target_blend_override"] = kwargs.get(
                "predicted_target_blend_override"
            )
            target_pos = np.asarray(target_state["position_neu"], dtype=np.float64)
            return (
                {
                    "position_neu": target_pos.copy(),
                    "position": target_pos.copy(),
                },
                {
                    "anchor_mode": anchor_mode,
                    "anchor_pos": target_pos.copy(),
                    "offset": np.zeros(3, dtype=np.float64),
                    "world_offset": np.zeros(3, dtype=np.float64),
                    "offset_frame": "world_neu",
                    "predicted_target_blend": kwargs.get(
                        "predicted_target_blend_override", 1.0
                    )
                    or 1.0,
                    "lateral_scale": 1.0,
                },
            )

        env.virtual_point_generator.action_to_virtual_point = _fake_action_to_virtual_point
        env.combat_hp.step = lambda own_state, target_state: {
            "ego_in_attack_zone": True,
            "target_in_attack_zone": False,
        }

        env._first_pass_complete = True
        _, _, _, _, info = env.step(np.zeros(3))
        assert info["post_merge_predicted_target_blend_release_on_attack_zone"] is True
        assert info["post_merge_predicted_target_blend_active"] is True
        assert info["post_merge_predicted_target_blend_released"] is True
        assert captured["predicted_target_blend_override"] == pytest.approx(0.5)

        _, _, _, _, info = env.step(np.zeros(3))
        assert info["post_merge_predicted_target_blend_active"] is False
        assert info["post_merge_predicted_target_blend_released"] is True
        assert captured["predicted_target_blend_override"] is None
        env.close()

    def test_post_merge_predicted_target_blend_release_can_wait_for_target_attack_zone(
        self, base_config
    ):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["virtual_point"]["anchor_mode"] = "predicted_target"
        config["virtual_point"]["post_merge_predicted_target_blend_by_task"] = {
            "head_on": 0.5,
        }
        config["virtual_point"][
            "post_merge_predicted_target_blend_release_on_attack_zone_by_task"
        ] = {
            "head_on": True,
        }
        config["virtual_point"][
            "post_merge_predicted_target_blend_release_requires_target_attack_zone_by_task"
        ] = {
            "head_on": True,
        }
        config["trajectory_prediction"] = {"enabled": True}
        env = CloseRangeTrackingEnv(config)
        env.reset(seed=0)

        class _FakePredictor:
            def predict(self, target_state):
                target_pos = np.asarray(target_state["position_m"], dtype=np.float64)
                return target_pos + np.array([100.0, 0.0, 0.0]), None, {}

        env.trajectory_predictor_adapter = _FakePredictor()

        captured = {}
        attack_zone_states = iter(
            [
                {"ego_in_attack_zone": True, "target_in_attack_zone": False},
                {"ego_in_attack_zone": True, "target_in_attack_zone": True},
                {"ego_in_attack_zone": False, "target_in_attack_zone": False},
            ]
        )

        def _fake_action_to_virtual_point(
            action,
            own_state,
            target_state,
            anchor_mode="current_target",
            **kwargs,
        ):
            captured["predicted_target_blend_override"] = kwargs.get(
                "predicted_target_blend_override"
            )
            target_pos = np.asarray(target_state["position_neu"], dtype=np.float64)
            return (
                {
                    "position_neu": target_pos.copy(),
                    "position": target_pos.copy(),
                },
                {
                    "anchor_mode": anchor_mode,
                    "anchor_pos": target_pos.copy(),
                    "offset": np.zeros(3, dtype=np.float64),
                    "world_offset": np.zeros(3, dtype=np.float64),
                    "offset_frame": "world_neu",
                    "predicted_target_blend": kwargs.get(
                        "predicted_target_blend_override", 1.0
                    )
                    or 1.0,
                    "lateral_scale": 1.0,
                },
            )

        env.virtual_point_generator.action_to_virtual_point = _fake_action_to_virtual_point
        env.combat_hp.step = lambda own_state, target_state: next(attack_zone_states)

        env._first_pass_complete = True
        _, _, _, _, info = env.step(np.zeros(3))
        assert info["post_merge_predicted_target_blend_release_on_attack_zone"] is True
        assert (
            info["post_merge_predicted_target_blend_release_requires_target_attack_zone"]
            is True
        )
        assert info["post_merge_predicted_target_blend_active"] is True
        assert info["post_merge_predicted_target_blend_released"] is False
        assert captured["predicted_target_blend_override"] == pytest.approx(0.5)

        _, _, _, _, info = env.step(np.zeros(3))
        assert info["post_merge_predicted_target_blend_active"] is True
        assert info["post_merge_predicted_target_blend_released"] is True
        assert captured["predicted_target_blend_override"] == pytest.approx(0.5)

        _, _, _, _, info = env.step(np.zeros(3))
        assert info["post_merge_predicted_target_blend_active"] is False
        assert info["post_merge_predicted_target_blend_released"] is True
        assert captured["predicted_target_blend_override"] is None
        env.close()

    def test_post_merge_predicted_target_blend_can_release_below_altitude_threshold(
        self, base_config
    ):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["virtual_point"]["anchor_mode"] = "predicted_target"
        config["virtual_point"]["post_merge_predicted_target_blend_by_task"] = {
            "head_on": 0.5,
        }
        config["virtual_point"][
            "post_merge_predicted_target_blend_release_below_altitude_m_by_task"
        ] = {
            "head_on": 6000.0,
        }
        config["trajectory_prediction"] = {"enabled": True}
        env = CloseRangeTrackingEnv(config)
        env.reset(seed=0)

        class _FakePredictor:
            def predict(self, target_state):
                target_pos = np.asarray(target_state["position_m"], dtype=np.float64)
                return target_pos + np.array([100.0, 0.0, 0.0]), None, {}

        env.trajectory_predictor_adapter = _FakePredictor()

        captured = {}

        def _fake_action_to_virtual_point(
            action,
            own_state,
            target_state,
            anchor_mode="current_target",
            **kwargs,
        ):
            captured["predicted_target_blend_override"] = kwargs.get(
                "predicted_target_blend_override"
            )
            target_pos = np.asarray(target_state["position_neu"], dtype=np.float64)
            return (
                {
                    "position_neu": target_pos.copy(),
                    "position": target_pos.copy(),
                },
                {
                    "anchor_mode": anchor_mode,
                    "anchor_pos": target_pos.copy(),
                    "offset": np.zeros(3, dtype=np.float64),
                    "world_offset": np.zeros(3, dtype=np.float64),
                    "offset_frame": "world_neu",
                    "predicted_target_blend": kwargs.get(
                        "predicted_target_blend_override", 1.0
                    )
                    or 1.0,
                    "lateral_scale": 1.0,
                },
            )

        env.virtual_point_generator.action_to_virtual_point = _fake_action_to_virtual_point
        env.combat_hp.step = lambda own_state, target_state: {
            "ego_in_attack_zone": False,
            "target_in_attack_zone": False,
        }

        env._first_pass_complete = True
        _, _, _, _, info = env.step(np.zeros(3))
        assert info["post_merge_predicted_target_blend_release_below_altitude_m"] == pytest.approx(
            6000.0
        )
        assert info["post_merge_predicted_target_blend_active"] is True
        assert info["post_merge_predicted_target_blend_released"] is True
        assert captured["predicted_target_blend_override"] == pytest.approx(0.5)

        _, _, _, _, info = env.step(np.zeros(3))
        assert info["post_merge_predicted_target_blend_active"] is False
        assert info["post_merge_predicted_target_blend_released"] is True
        assert captured["predicted_target_blend_override"] is None
        env.close()

    def test_post_merge_predicted_target_blend_can_expire_after_hold_steps(
        self, base_config
    ):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["virtual_point"]["anchor_mode"] = "predicted_target"
        config["virtual_point"]["post_merge_predicted_target_blend_by_task"] = {
            "head_on": 0.5,
        }
        config["virtual_point"]["post_merge_predicted_target_blend_hold_steps_by_task"] = {
            "head_on": 1,
        }
        config["trajectory_prediction"] = {"enabled": True}
        env = CloseRangeTrackingEnv(config)
        env.reset(seed=0)

        class _FakePredictor:
            def predict(self, target_state):
                target_pos = np.asarray(target_state["position_m"], dtype=np.float64)
                return target_pos + np.array([100.0, 0.0, 0.0]), None, {}

        env.trajectory_predictor_adapter = _FakePredictor()

        captured = {}

        def _fake_action_to_virtual_point(
            action,
            own_state,
            target_state,
            anchor_mode="current_target",
            **kwargs,
        ):
            captured["predicted_target_blend_override"] = kwargs.get(
                "predicted_target_blend_override"
            )
            target_pos = np.asarray(target_state["position_neu"], dtype=np.float64)
            return (
                {
                    "position_neu": target_pos.copy(),
                    "position": target_pos.copy(),
                },
                {
                    "anchor_mode": anchor_mode,
                    "anchor_pos": target_pos.copy(),
                    "offset": np.zeros(3, dtype=np.float64),
                    "world_offset": np.zeros(3, dtype=np.float64),
                    "offset_frame": "world_neu",
                    "predicted_target_blend": kwargs.get(
                        "predicted_target_blend_override", 1.0
                    )
                    or 1.0,
                    "lateral_scale": 1.0,
                },
            )

        env.virtual_point_generator.action_to_virtual_point = _fake_action_to_virtual_point
        env.combat_hp.step = lambda own_state, target_state: {
            "ego_in_attack_zone": False,
            "target_in_attack_zone": False,
        }

        env._first_pass_complete = True
        env._first_pass_completion_step = 0

        _, _, _, _, info = env.step(np.zeros(3))
        assert info["post_merge_predicted_target_blend_hold_steps"] == pytest.approx(1.0)
        assert info["post_merge_predicted_target_blend_steps_since_first_pass"] == pytest.approx(
            1.0
        )
        assert info["post_merge_predicted_target_blend_hold_remaining_steps"] == pytest.approx(
            1.0
        )
        assert info["post_merge_predicted_target_blend_hold_window_open"] is True
        assert info["post_merge_predicted_target_blend_active"] is True
        assert info["post_merge_predicted_target_blend_released"] is False
        assert captured["predicted_target_blend_override"] == pytest.approx(0.5)

        _, _, _, _, info = env.step(np.zeros(3))
        assert info["post_merge_predicted_target_blend_steps_since_first_pass"] == pytest.approx(
            2.0
        )
        assert info["post_merge_predicted_target_blend_hold_remaining_steps"] == pytest.approx(
            0.0
        )
        assert info["post_merge_predicted_target_blend_hold_window_open"] is False
        assert info["post_merge_predicted_target_blend_active"] is False
        assert info["post_merge_predicted_target_blend_released"] is True
        assert captured["predicted_target_blend_override"] is None
        env.close()

    def test_post_merge_predicted_target_blend_can_fall_back_to_release_blend_after_hold_steps(
        self, base_config
    ):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["virtual_point"]["anchor_mode"] = "predicted_target"
        config["virtual_point"]["post_merge_predicted_target_blend_by_task"] = {
            "head_on": 0.5,
        }
        config["virtual_point"]["post_merge_predicted_target_blend_hold_steps_by_task"] = {
            "head_on": 1,
        }
        config["virtual_point"]["post_merge_predicted_target_blend_release_blend_by_task"] = {
            "head_on": 0.75,
        }
        config["trajectory_prediction"] = {"enabled": True}
        env = CloseRangeTrackingEnv(config)
        env.reset(seed=0)

        class _FakePredictor:
            def predict(self, target_state):
                target_pos = np.asarray(target_state["position_m"], dtype=np.float64)
                return target_pos + np.array([100.0, 0.0, 0.0]), None, {}

        env.trajectory_predictor_adapter = _FakePredictor()

        captured = {}

        def _fake_action_to_virtual_point(
            action,
            own_state,
            target_state,
            anchor_mode="current_target",
            **kwargs,
        ):
            captured["predicted_target_blend_override"] = kwargs.get(
                "predicted_target_blend_override"
            )
            target_pos = np.asarray(target_state["position_neu"], dtype=np.float64)
            return (
                {
                    "position_neu": target_pos.copy(),
                    "position": target_pos.copy(),
                },
                {
                    "anchor_mode": anchor_mode,
                    "anchor_pos": target_pos.copy(),
                    "offset": np.zeros(3, dtype=np.float64),
                    "world_offset": np.zeros(3, dtype=np.float64),
                    "offset_frame": "world_neu",
                    "predicted_target_blend": kwargs.get(
                        "predicted_target_blend_override", 1.0
                    )
                    or 1.0,
                    "lateral_scale": 1.0,
                },
            )

        env.virtual_point_generator.action_to_virtual_point = _fake_action_to_virtual_point
        env.combat_hp.step = lambda own_state, target_state: {
            "ego_in_attack_zone": False,
            "target_in_attack_zone": False,
        }

        env._first_pass_complete = True
        env._first_pass_completion_step = 0

        _, _, _, _, info = env.step(np.zeros(3))
        assert info["post_merge_predicted_target_blend_active"] is True
        assert info["post_merge_predicted_target_blend_release_blend"] == pytest.approx(
            0.75
        )
        assert info["post_merge_predicted_target_blend_release_blend_active"] is False
        assert captured["predicted_target_blend_override"] == pytest.approx(0.5)

        _, _, _, _, info = env.step(np.zeros(3))
        assert info["post_merge_predicted_target_blend_hold_window_open"] is False
        assert info["post_merge_predicted_target_blend_active"] is False
        assert info["post_merge_predicted_target_blend_release_blend_active"] is False
        assert info["post_merge_predicted_target_blend_released"] is True
        assert captured["predicted_target_blend_override"] is None

        _, _, _, _, info = env.step(np.zeros(3))
        assert info["post_merge_predicted_target_blend_active"] is False
        assert info["post_merge_predicted_target_blend_release_blend_active"] is True
        assert info["post_merge_predicted_target_blend_released"] is True
        assert captured["predicted_target_blend_override"] == pytest.approx(0.75)
        env.close()

    def test_close_range_anchor_can_hold_for_extra_post_merge_steps(self, base_config):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["virtual_point"]["anchor_mode"] = "predicted_target"
        config["virtual_point"]["close_range_anchor_mode_by_task"] = {
            "head_on": "current_target",
        }
        config["virtual_point"]["close_range_anchor_release_on_post_merge_by_task"] = {
            "head_on": True,
        }
        config["virtual_point"]["close_range_anchor_post_merge_hold_steps_by_task"] = {
            "head_on": 1,
        }
        env = CloseRangeTrackingEnv(config)
        env.reset(seed=0)
        env._close_range_anchor_ready = True
        env._first_pass_complete = True
        env._first_pass_completion_step = 0

        _, _, _, _, info = env.step(np.zeros(3))
        assert info["close_range_anchor_post_merge_hold_steps"] == 1
        assert info["close_range_anchor_window_open"] is True
        assert info["close_range_anchor_mode_active"] is True
        assert info["close_range_anchor_post_merge_hold_remaining_steps"] == pytest.approx(
            1.0
        )

        _, _, _, _, info = env.step(np.zeros(3))
        assert info["close_range_anchor_window_open"] is False
        assert info["close_range_anchor_mode_active"] is False
        assert info["close_range_anchor_post_merge_hold_remaining_steps"] == pytest.approx(
            0.0
        )
        env.close()

    def test_episode_runs_to_completion(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        done = False
        steps = 0
        while not done and steps < 20:
            _, _, terminated, truncated, info = env.step(np.zeros(3))
            done = terminated or truncated
            steps += 1
        assert steps > 0
        env.close()

    def test_multiple_episodes(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        for ep in range(3):
            env.reset(seed=ep)
            done = False
            steps = 0
            while not done and steps < 10:
                _, _, terminated, truncated, info = env.step(np.zeros(3))
                done = terminated or truncated
                steps += 1
        env.close()

    def test_different_actions_produce_different_commands(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        _, _, _, _, info1 = env.step(np.array([0.0, 0.0, 0.0]))
        _, _, _, _, info2 = env.step(np.array([1.0, 0.5, -0.3]))
        # Different actions should change the virtual point and guidance command
        assert not np.allclose(
            info1["guidance_command"]["nz_cmd"],
            info2["guidance_command"]["nz_cmd"],
            atol=1e-6,
        )
        env.close()

    def test_observation_vector_shape(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        obs, _, _, _, _ = env.step(np.zeros(3))
        assert "observation_vector" in obs
        assert obs["observation_vector"].ndim == 1
        assert obs["observation_vector"].shape[0] > 0
        env.close()

    def test_reward_is_finite(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        for _ in range(5):
            _, reward, _, _, _ = env.step(np.zeros(3))
            assert np.isfinite(reward)
        env.close()

    def test_commands_within_limits(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        for _ in range(10):
            _, _, _, _, info = env.step(np.zeros(3))
            cmd = info["guidance_command"]
            assert -2.0 <= cmd["nz_cmd"] <= 7.0
            assert -1.5 <= cmd["roll_rate_cmd"] <= 1.5
            assert 0.0 <= cmd["throttle_cmd"] <= 1.0
        env.close()

    def test_backend_simple_when_use_jsbsim_false(self, base_config):
        base_config["env"]["use_jsbsim"] = False
        env = CloseRangeTrackingEnv(base_config)
        assert env._backend == "simple"
        env.close()

    def test_backend_explicit_override(self, base_config):
        base_config["backend"] = "simple"
        env = CloseRangeTrackingEnv(base_config)
        assert env._backend == "simple"
        env.close()

    def test_virtual_point_in_info(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(3))
        assert "virtual_point" in info
        assert "position" in info["virtual_point"]
        env.close()

    def test_termination_reason_in_info(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        done = False
        steps = 0
        while not done and steps < 600:
            _, _, terminated, truncated, info = env.step(np.zeros(3))
            done = terminated or truncated
            steps += 1
        assert "reason" in info["termination_info"]
        env.close()

    def test_prediction_enabled_false_by_default(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(3))
        assert info["prediction_enabled"] is False
        env.close()

    def test_anchor_mode_current_target(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(3))
        assert info["anchor_mode"] == "current_target"
        env.close()

    def test_relative_state_fields(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        obs, _, _, _, _ = env.step(np.zeros(3))
        rel = obs["relative_state"]
        assert "range_m" in rel
        assert "ata_rad" in rel
        assert "aa_rad" in rel
        env.close()

    def test_reward_terms_structure(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(3))
        rt = info["reward_terms"]
        assert isinstance(rt, dict)
        assert len(rt) > 0
        env.close()

    def test_step_count_increments(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        assert env.current_step == 0
        env.step(np.zeros(3))
        assert env.current_step == 1
        env.step(np.zeros(3))
        assert env.current_step == 2
        env.close()

    def test_episode_count_increments(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        ep1 = env._episode_count
        env.reset(seed=1)
        ep2 = env._episode_count
        assert ep2 == ep1 + 1
        env.close()

    def test_close_does_not_raise(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        env.step(np.zeros(3))
        env.close()

    def test_terminal_reward_success(self, base_config):
        """成功到达时 terminal_reward 应为正数。"""
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        # 构造一个接近目标的场景
        for _ in range(50):
            _, _, terminated, truncated, info = env.step(np.zeros(3))
            if terminated:
                rt = info["reward_terms"]
                if info["termination_info"]["is_success"]:
                    assert rt["terminal_reward"] == pytest.approx(200.0, abs=1e-6)
                break
        env.close()

    def test_terminal_reward_crash(self, base_config):
        """碰撞时 terminal_reward 应为负数。"""
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        for _ in range(50):
            _, _, terminated, truncated, info = env.step(np.zeros(3))
            if terminated:
                rt = info["reward_terms"]
                if info["termination_info"]["is_crash"]:
                    assert rt["terminal_reward"] == pytest.approx(-300.0, abs=1e-6)
                break
        env.close()

    def test_terminal_reward_timeout(self, base_config):
        """超时 truncation 时 terminal_reward 应为负数。"""
        base_config["env"]["max_high_level_steps"] = 1
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        _, _, terminated, truncated, info = env.step(np.zeros(3))
        assert truncated is True
        rt = info["reward_terms"]
        assert rt["terminal_reward"] == pytest.approx(-200.0, abs=1e-6)
        env.close()

    def test_timeout_info_reason_is_timeout(self, base_config):
        """timeout 时 info['termination_info']['reason'] 应为 'timeout'。"""
        base_config["env"]["max_high_level_steps"] = 1
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        _, _, terminated, truncated, info = env.step(np.zeros(3))
        assert terminated is False
        assert truncated is True
        assert info["termination_info"]["reason"] == "timeout"
        assert info["termination_info"]["is_timeout"] is True
        assert info["is_timeout"] is True
        env.close()


class TestStrictBackend:
    """Tests for strict_backend config option."""

    def test_strict_backend_false_allows_simple_fallback(
        self, base_config, monkeypatch
    ):
        base_config["env"]["use_jsbsim"] = True
        base_config["env"]["strict_backend"] = False

        def _fail_init(*args, **kwargs):
            raise RuntimeError("Simulated JSBSim failure")

        monkeypatch.setattr(
            "uav_vpp_guidance.envs.tracking_env.JSBSimEnv",
            _fail_init,
        )
        env = CloseRangeTrackingEnv(base_config)
        assert env._backend == "simple"
        env.close()

    def test_strict_backend_true_raises_on_jsbsim_failure(
        self, base_config, monkeypatch
    ):
        base_config["env"]["use_jsbsim"] = True
        base_config["env"]["strict_backend"] = True

        def _fail_init(*args, **kwargs):
            raise RuntimeError("Simulated JSBSim failure")

        monkeypatch.setattr(
            "uav_vpp_guidance.envs.tracking_env.JSBSimEnv",
            _fail_init,
        )
        with pytest.raises(RuntimeError, match="strict_backend=True"):
            CloseRangeTrackingEnv(base_config)

    def test_backend_reflected_in_info(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(3))
        assert info["backend"] == env._backend
        env.close()


class TestCommandOverrideBackwardCompatibility:
    """Tests for the command_override parameter added in Stage 10.1."""

    def test_step_without_command_override_works(self, base_config):
        """Default step() behavior is unchanged."""
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(np.zeros(3))
        assert "guidance_command" in info
        assert info.get("effective_guidance_mode") != "command_override"
        env.close()

    def test_command_override_bypasses_policy_and_guidance(self, base_config):
        """command_override injects a command directly, bypassing VPP/guidance."""
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        override = {"nz_cmd": 2.0, "roll_rate_cmd": 0.5, "throttle_cmd": 0.8}
        obs, reward, terminated, truncated, info = env.step(
            np.zeros(3), command_override=override
        )
        assert info.get("effective_guidance_mode") == "command_override"
        assert info.get("virtual_point_source") == "command_override"
        assert info.get("command_override_active") is True
        assert info.get("raw_command") == override
        # The command should still be clipped/filtered but originate from override
        gc = info.get("guidance_command", {})
        assert abs(gc.get("nz_cmd", 0.0) - 2.0) < 0.5  # close after filter
        env.close()

    def test_command_override_clipping_still_applies(self, base_config):
        """Even with override, limits are enforced for safety."""
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        override = {"nz_cmd": 100.0, "roll_rate_cmd": 0.0, "throttle_cmd": 0.5}
        _, _, _, _, info = env.step(np.zeros(3), command_override=override)
        gc = info.get("guidance_command", {})
        assert gc["nz_cmd"] <= base_config["limits"]["nz_max"]
        env.close()


class TestScenarioPositionConversionRegression:
    """Regression tests for Stage 10.1 position conversion bug."""

    def test_scenario_to_jsbsim_init_sets_geodetic_for_nonzero_xy(self, base_config):
        """If position_m has nonzero x or y, lon/lat must be set."""
        if not os.environ.get("JSBSIM_ROOT"):
            pytest.skip("JSBSIM_ROOT not set")
        try:
            import pymap3d  # noqa: F401
        except ImportError:
            pytest.skip("pymap3d not installed")

        base_config["env"]["use_jsbsim"] = True
        base_config["env"]["strict_backend"] = True
        base_config["env"]["legacy_project_root"] = os.environ.get("JSBSIM_ROOT", "")
        base_config["env"]["origin"] = [120.0, 60.0, 0.0]

        env = CloseRangeTrackingEnv(base_config)
        if env._backend != "jsbsim":
            pytest.skip("JSBSim backend not available")

        scenario = {
            "own_init": {
                "position_m": [0.0, 0.0, 5000.0],
                "velocity_mps": 200.0,
                "heading_deg": 0.0,
            },
            "target_init": {
                "position_m": [2000.0, 0.0, 5000.0],
                "velocity_mps": 200.0,
                "heading_deg": 180.0,
            },
        }
        own_init = env._scenario_to_jsbsim_init(scenario["own_init"])
        target_init = env._scenario_to_jsbsim_init(scenario["target_init"])

        assert "ic/long-gc-deg" in own_init
        assert "ic/lat-geod-deg" in own_init
        assert "ic/long-gc-deg" in target_init
        assert "ic/lat-geod-deg" in target_init

        # Target is 2000m north of origin at lat=60deg
        assert target_init["ic/lat-geod-deg"] > own_init["ic/lat-geod-deg"]
        env.close()

    def test_zero_xy_position_keeps_default_origin(self, base_config):
        """If position_m is [0,0,z], lon/lat should still be set (to origin)."""
        if not os.environ.get("JSBSIM_ROOT"):
            pytest.skip("JSBSIM_ROOT not set")
        try:
            import pymap3d  # noqa: F401
        except ImportError:
            pytest.skip("pymap3d not installed")

        base_config["env"]["use_jsbsim"] = True
        base_config["env"]["strict_backend"] = True
        base_config["env"]["legacy_project_root"] = os.environ.get("JSBSIM_ROOT", "")
        base_config["env"]["origin"] = [120.0, 60.0, 0.0]

        env = CloseRangeTrackingEnv(base_config)
        if env._backend != "jsbsim":
            pytest.skip("JSBSim backend not available")

        init = env._scenario_to_jsbsim_init(
            {"position_m": [0.0, 0.0, 5000.0], "velocity_mps": 200.0, "heading_deg": 0.0}
        )
        assert "ic/long-gc-deg" in init
        assert "ic/lat-geod-deg" in init
        assert init["ic/long-gc-deg"] == pytest.approx(120.0, abs=1e-6)
        assert init["ic/lat-geod-deg"] == pytest.approx(60.0, abs=1e-6)
        env.close()


class TestObservationSchema:
    """Tests for configurable observation schema."""

    def test_default_observation_schema_excludes_gains_and_guidance_state(
        self, base_config
    ):
        env = CloseRangeTrackingEnv(base_config)
        obs = env.reset(seed=0)
        assert "observation_schema" in obs
        schema = obs["observation_schema"]
        assert schema["include_gains"] is False
        assert schema["include_guidance_state"] is False
        env.close()

    def test_include_gains_increases_observation_dim(self, base_config):
        env_base = CloseRangeTrackingEnv(base_config)
        base_config["observation"] = {"include_gains": True}
        env_gains = CloseRangeTrackingEnv(base_config)
        try:
            base_obs = env_base.reset(seed=0)
            gains_obs = env_gains.reset(seed=0)
            assert (
                gains_obs["observation_vector"].shape[0]
                == base_obs["observation_vector"].shape[0] + 2
            )
            assert gains_obs["observation_schema"]["include_gains"] is True
        finally:
            env_base.close()
            env_gains.close()

    def test_include_guidance_state_increases_observation_dim(self, base_config):
        env_base = CloseRangeTrackingEnv(base_config)
        base_config["observation"] = {"include_guidance_state": True}
        env_guidance = CloseRangeTrackingEnv(base_config)
        try:
            base_obs = env_base.reset(seed=0)
            guidance_obs = env_guidance.reset(seed=0)
            assert (
                guidance_obs["observation_vector"].shape[0]
                == base_obs["observation_vector"].shape[0] + 3
            )
            assert (
                guidance_obs["observation_schema"]["include_guidance_state"] is True
            )
        finally:
            env_base.close()
            env_guidance.close()

    def test_include_both_increases_observation_dim_correctly(self, base_config):
        env_base = CloseRangeTrackingEnv(base_config)
        base_config["observation"] = {
            "include_gains": True,
            "include_guidance_state": True,
        }
        env_both = CloseRangeTrackingEnv(base_config)
        try:
            base_obs = env_base.reset(seed=0)
            both_obs = env_both.reset(seed=0)
            assert (
                both_obs["observation_vector"].shape[0]
                == base_obs["observation_vector"].shape[0] + 5
            )
        finally:
            env_base.close()
            env_both.close()

    def test_backend_fallback_occurred_in_info(self, base_config, monkeypatch):
        base_config["env"]["use_jsbsim"] = True
        base_config["env"]["strict_backend"] = False

        def _fail_init(*args, **kwargs):
            raise RuntimeError("Simulated JSBSim failure")

        monkeypatch.setattr(
            "uav_vpp_guidance.envs.tracking_env.JSBSimEnv",
            _fail_init,
        )
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(3))
        assert "backend_fallback_occurred" in info
        assert info["backend_fallback_occurred"] is True
        env.close()

    def test_backend_fallback_false_when_no_fallback(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(3))
        assert "backend_fallback_occurred" in info
        assert info["backend_fallback_occurred"] is False
        env.close()

    def test_default_observation_schema_excludes_saturation(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        obs = env.reset(seed=0)
        assert obs["observation_schema"]["include_saturation"] is False
        env.close()

    def test_include_saturation_increases_observation_dim(self, base_config):
        env_base = CloseRangeTrackingEnv(base_config)
        base_config["observation"] = {"include_saturation": True}
        env_sat = CloseRangeTrackingEnv(base_config)
        try:
            base_obs = env_base.reset(seed=0)
            sat_obs = env_sat.reset(seed=0)
            assert (
                sat_obs["observation_vector"].shape[0]
                == base_obs["observation_vector"].shape[0] + 3
            )
            assert sat_obs["observation_schema"]["include_saturation"] is True
        finally:
            env_base.close()
            env_sat.close()

    def test_saturation_flags_detect_command_clipping(self, base_config):
        """Extreme override commands should be flagged in observation when enabled."""
        base_config["observation"] = {"include_saturation": True}
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        override = {
            "nz_cmd": 100.0,
            "roll_rate_cmd": -5.0,
            "throttle_cmd": 2.0,
        }
        obs, _, _, _, info = env.step(np.zeros(3), command_override=override)
        schema = obs["observation_schema"]
        assert schema["include_saturation"] is True
        # Identify saturation flags by keys in the observation schema would require
        # index knowledge; instead assert they are in the dict returned by reset.
        vec = obs["observation_vector"]
        # Saturation flags are appended at the end in build_observation order.
        assert vec[-3] == pytest.approx(1.0, abs=1e-6)  # nz_saturated
        assert vec[-2] == pytest.approx(1.0, abs=1e-6)  # roll_rate_saturated
        assert vec[-1] == pytest.approx(1.0, abs=1e-6)  # throttle_saturated
        env.close()

    def test_no_saturation_flags_for_in_range_commands(self, base_config):
        """In-range override commands should not be flagged as saturated."""
        base_config["observation"] = {"include_saturation": True}
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        override = {
            "nz_cmd": 2.0,
            "roll_rate_cmd": 0.0,
            "throttle_cmd": 0.5,
        }
        obs, _, _, _, info = env.step(np.zeros(3), command_override=override)
        vec = obs["observation_vector"]
        assert vec[-3] == pytest.approx(0.0, abs=1e-6)
        assert vec[-2] == pytest.approx(0.0, abs=1e-6)
        assert vec[-1] == pytest.approx(0.0, abs=1e-6)
        env.close()

    def test_full_observation_dim_with_all_extensions(self, base_config):
        base_config["observation"] = {
            "include_gains": True,
            "include_guidance_state": True,
            "include_saturation": True,
        }
        env = CloseRangeTrackingEnv(base_config)
        obs = env.reset(seed=0)
        schema = obs["observation_schema"]
        assert schema["include_gains"] is True
        assert schema["include_guidance_state"] is True
        assert schema["include_saturation"] is True
        # After reset (before any step) saturation flags should all be zero.
        assert obs["observation_vector"].shape[0] == 16 + 2 + 3 + 3
        env.close()

    def test_include_opponent_stage_increases_dim(self, base_config):
        base_config["observation"] = {"include_opponent_stage": True}
        env = CloseRangeTrackingEnv(base_config)
        obs = env.reset(seed=0)
        schema = obs["observation_schema"]
        assert schema["include_opponent_stage"] is True
        assert obs["observation_vector"].shape[0] == 16 + 2
        # Default opponent stage is "custom" -> both one-hot bits are 0
        assert obs["observation_vector"][-2] == 0.0
        assert obs["observation_vector"][-1] == 0.0
        assert schema["feature_names"][-2] == "opponent_is_expert"
        assert schema["feature_names"][-1] == "opponent_is_end_to_end"
        env.close()

    def test_opponent_stage_one_hot_expert(self, base_config):
        base_config["opponent"] = {"stage": "expert"}
        base_config["observation"] = {"include_opponent_stage": True}
        env = CloseRangeTrackingEnv(base_config)
        obs = env.reset(seed=0)
        assert obs["observation_vector"][-2] == 1.0
        assert obs["observation_vector"][-1] == 0.0
        env.close()

    def test_opponent_stage_one_hot_end_to_end(self, base_config):
        base_config["opponent"] = {"stage": "end_to_end"}
        base_config["observation"] = {"include_opponent_stage": True}
        env = CloseRangeTrackingEnv(base_config)
        obs = env.reset(seed=0)
        assert obs["observation_vector"][-2] == 0.0
        assert obs["observation_vector"][-1] == 1.0
        env.close()

    def test_include_task_type_increases_dim(self, base_config):
        base_config["task"] = {"name": "head_on"}
        base_config["observation"] = {"include_task_type": True}
        env = CloseRangeTrackingEnv(base_config)
        obs = env.reset(seed=0)
        schema = obs["observation_schema"]
        assert schema["include_task_type"] is True
        assert obs["observation_vector"].shape[0] == 16 + 1
        assert schema["feature_names"][-1] == "is_crossing"
        env.close()

    def test_task_type_binary_head_on(self, base_config):
        base_config["task"] = {"name": "head_on"}
        base_config["observation"] = {"include_task_type": True}
        env = CloseRangeTrackingEnv(base_config)
        obs = env.reset(seed=0)
        assert obs["observation_vector"][-1] == 0.0
        env.close()

    def test_task_type_binary_crossing(self, base_config):
        base_config["task"] = {"name": "crossing_feasible"}
        base_config["observation"] = {"include_task_type": True}
        env = CloseRangeTrackingEnv(base_config)
        obs = env.reset(seed=0)
        assert obs["observation_vector"][-1] == 1.0
        env.close()

    def test_full_observation_dim_with_all_extensions_including_opponent_stage(
        self, base_config
    ):
        base_config["task"] = {"name": "head_on"}
        base_config["observation"] = {
            "include_gains": True,
            "include_guidance_state": True,
            "include_saturation": True,
            "include_opponent_stage": True,
            "include_task_type": True,
        }
        env = CloseRangeTrackingEnv(base_config)
        obs = env.reset(seed=0)
        schema = obs["observation_schema"]
        assert schema["include_gains"] is True
        assert schema["include_guidance_state"] is True
        assert schema["include_saturation"] is True
        assert schema["include_opponent_stage"] is True
        assert schema["include_task_type"] is True
        assert obs["observation_vector"].shape[0] == 16 + 2 + 3 + 3 + 2 + 1
        env.close()

    def test_observation_schema_includes_feature_names(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        obs = env.reset(seed=0)
        schema = obs["observation_schema"]
        assert "feature_names" in schema
        assert len(schema["feature_names"]) == schema["dim"]
        assert schema["feature_names"][:4] == [
            "range_m",
            "range_rate_mps",
            "altitude_diff_m",
            "speed_diff_mps",
        ]
        env.close()

    def test_provenance_present_in_reset_and_step(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        obs = env.reset(seed=0)
        assert "provenance" in obs
        assert obs["provenance"]["backend"] == env._backend
        assert obs["provenance"]["backend_fallback_occurred"] is False
        assert "observation_schema" in obs["provenance"]
        assert "config_overrides" in obs["provenance"]

        _, _, _, _, info = env.step(np.zeros(3))
        assert "provenance" in info
        assert info["provenance"]["backend"] == env._backend
        env.close()

    def test_backend_fallback_reason_recorded(self, base_config, monkeypatch):
        base_config["env"]["use_jsbsim"] = True
        base_config["env"]["strict_backend"] = False

        def _fail_init(*args, **kwargs):
            raise RuntimeError("Simulated JSBSim failure")

        monkeypatch.setattr(
            "uav_vpp_guidance.envs.tracking_env.JSBSimEnv",
            _fail_init,
        )
        env = CloseRangeTrackingEnv(base_config)
        obs = env.reset(seed=0)
        assert obs["provenance"]["backend_fallback_occurred"] is True
        assert obs["provenance"]["backend_fallback_reason"] == "Simulated JSBSim failure"
        assert obs["provenance"]["backend"] == "simple"

        _, _, _, _, info = env.step(np.zeros(3))
        assert info["backend_fallback_reason"] == "Simulated JSBSim failure"
        env.close()

    def test_config_overrides_reflected_in_provenance(self, base_config):
        from uav_vpp_guidance.common.provenance import record_config_override

        record_config_override(
            base_config,
            "guidance.mode_switch.enabled",
            False,
            old_value=True,
            source="test_override",
        )
        env = CloseRangeTrackingEnv(base_config)
        obs = env.reset(seed=0)
        overrides = obs["provenance"]["config_overrides"]
        assert len(overrides) == 1
        assert overrides[0]["source"] == "test_override"
        assert overrides[0]["key"] == "guidance.mode_switch.enabled"
        assert overrides[0]["new_value"] is False
        env.close()

    def test_post_merge_offensive_anchor_blend_release_direct_track_enabled_defaults_true_and_can_be_disabled(
        self,
        base_config,
    ):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["virtual_point"]["anchor_mode"] = "predicted_target"
        config["virtual_point"]["close_range_anchor_mode"] = None
        config["virtual_point"]["close_range_anchor_mode_by_task"] = {}
        config["virtual_point"]["post_merge_offensive_anchor_blend_by_task"] = {
            "head_on": 0.25,
        }
        config["virtual_point"][
            "post_merge_offensive_anchor_blend_release_direct_track_enabled_by_task"
        ] = {"head_on": False}
        config["virtual_point"][
            "post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m_by_task"
        ] = {"head_on": 2000.0}

        env = CloseRangeTrackingEnv(config)
        env.reset(seed=0)
        env._first_pass_complete = True
        env._first_pass_completion_step = 0
        env._post_merge_offensive_anchor_blend_released = True

        # Fake the gate so that without the enabled flag, direct-track would be active
        env._post_merge_offensive_anchor_gate = {
            "range_opening": True,
            "ego_in_attack_zone": False,
            "target_in_attack_zone": False,
        }

        _, _, _, _, info = env.step(np.zeros(3))
        assert info["post_merge_offensive_anchor_blend_release_direct_track_enabled"] is False
        assert info["post_merge_offensive_anchor_blend_release_direct_track_active"] is False
        assert info["direct_track_mode_effective"] is False
        env.close()

    def test_post_merge_offensive_anchor_blend_release_direct_track_enabled_task_override(
        self,
        base_config,
    ):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "crossing_feasible", "env_class": "CloseRangeTrackingEnv"}
        config["virtual_point"]["anchor_mode"] = "predicted_target"
        config["virtual_point"]["close_range_anchor_mode"] = None
        config["virtual_point"]["close_range_anchor_mode_by_task"] = {}
        config["virtual_point"][
            "post_merge_offensive_anchor_blend_release_direct_track_enabled_by_task"
        ] = {"head_on": False}
        # default should be True for crossing_feasible since not overridden
        env = CloseRangeTrackingEnv(config)
        assert env._resolve_post_merge_offensive_anchor_blend_release_direct_track_enabled() is True
        env.close()

    def test_post_merge_offensive_anchor_blend_release_direct_track_enabled_matches_legacy_altitude_zero_behavior_without_recovery_or_latch(
        self,
        base_config,
    ):
        def _build_config(*, use_boolean_flag):
            config = copy.deepcopy(base_config)
            config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
            config["virtual_point"]["anchor_mode"] = "predicted_target"
            config["virtual_point"]["close_range_anchor_mode"] = None
            config["virtual_point"]["close_range_anchor_mode_by_task"] = {}
            config["virtual_point"]["post_merge_offensive_anchor_blend_by_task"] = {
                "head_on": 0.25,
            }
            config["virtual_point"][
                "post_merge_offensive_anchor_blend_release_blend_by_task"
            ] = {
                "head_on": 0.0,
            }
            config["virtual_point"][
                "post_merge_offensive_anchor_blend_release_ego_only_streak_steps_by_task"
            ] = {
                "head_on": 1,
            }
            config["virtual_point"][
                "post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m_by_task"
            ] = {
                "head_on": 0.0,
            }
            config["virtual_point"][
                "post_merge_offensive_anchor_lateral_world_offset_latch_on_activation"
            ] = False
            config["virtual_point"][
                "post_merge_offensive_anchor_lateral_world_offset_latch_on_activation_by_task"
            ] = {}
            config["virtual_point"][
                "post_merge_offensive_anchor_blend_release_recovery_below_altitude_m"
            ] = None
            config["virtual_point"][
                "post_merge_offensive_anchor_blend_release_recovery_below_altitude_m_by_task"
            ] = {}
            config["virtual_point"][
                "post_merge_offensive_anchor_blend_release_recovery_longitudinal_blend"
            ] = None
            config["virtual_point"][
                "post_merge_offensive_anchor_blend_release_recovery_longitudinal_blend_by_task"
            ] = {}
            config["virtual_point"][
                "post_merge_offensive_anchor_blend_release_recovery_lateral_blend"
            ] = None
            config["virtual_point"][
                "post_merge_offensive_anchor_blend_release_recovery_lateral_blend_by_task"
            ] = {}
            config["virtual_point"][
                "post_merge_offensive_anchor_blend_release_recovery_forward_bias_m_max"
            ] = None
            config["virtual_point"][
                "post_merge_offensive_anchor_blend_release_recovery_forward_bias_m_max_by_task"
            ] = {}
            if use_boolean_flag:
                config["virtual_point"][
                    "post_merge_offensive_anchor_blend_release_direct_track_enabled_by_task"
                ] = {"head_on": False}
            return config

        def _prepare_env(config):
            env = CloseRangeTrackingEnv(config)
            scenario = {
                "own_init": {
                    "position_m": [0.0, 0.0, 1000.0],
                    "velocity_mps": 200.0,
                    "heading_deg": 0.0,
                },
                "target_init": {
                    "position_m": [1500.0, 0.0, 1000.0],
                    "velocity_mps": 200.0,
                    "heading_deg": 0.0,
                },
            }
            env.reset(scenario=scenario, seed=0)
            env._first_pass_complete = True
            env._first_pass_completion_step = 0
            env._post_merge_offensive_anchor_blend_released = True
            env._evaluate_post_merge_offensive_anchor_geometry_disadvantage = (
                lambda own_state, target_state, rel_state=None: {
                    "requires_geometry_disadvantage": True,
                    "condition_met": False,
                    "target_only_attack_zone_disadvantage": False,
                    "geometry_disadvantage": False,
                    "alignment_disadvantage": False,
                    "ego_attack_score": 0.0,
                    "target_attack_score": 0.0,
                    "ego_in_attack_zone": False,
                    "target_in_attack_zone": False,
                    "aa_deg_min": np.nan,
                    "aa_deg": np.nan,
                    "range_rate_mps": 250.0,
                    "range_opening": True,
                }
            )

            def _fake_action_to_virtual_point(
                action,
                own_state,
                target_state,
                anchor_mode="current_target",
                **kwargs,
            ):
                target_pos = np.asarray(target_state["position_neu"], dtype=np.float64)
                vp_pos = target_pos + np.array([-1500.0, 0.0, 0.0], dtype=np.float64)
                return (
                    {
                        "position_neu": vp_pos.copy(),
                        "position": vp_pos.copy(),
                    },
                    {
                        "anchor_mode": anchor_mode,
                        "anchor_pos": target_pos.copy(),
                        "offset": np.zeros(3, dtype=np.float64),
                        "world_offset": vp_pos - target_pos,
                        "offset_frame": "world_neu",
                        "predicted_target_blend": 1.0,
                        "predicted_target_forward_scale": 1.0,
                        "offensive_anchor_blend": 0.0,
                        "longitudinal_scale": 1.0,
                        "lateral_scale": 1.0,
                    },
                )

            env.virtual_point_generator.action_to_virtual_point = _fake_action_to_virtual_point
            return env

        legacy_env = _prepare_env(_build_config(use_boolean_flag=False))
        boolean_env = _prepare_env(_build_config(use_boolean_flag=True))

        _, _, _, _, legacy_info = legacy_env.step(np.zeros(3))
        _, _, _, _, boolean_info = boolean_env.step(np.zeros(3))

        for key in (
            "post_merge_offensive_anchor_blend_release_direct_track_active",
            "post_merge_offensive_anchor_blend_release_recovery_active",
            "post_merge_offensive_anchor_lateral_world_offset_latch_active",
            "direct_track_mode_requested",
            "direct_track_mode_effective",
            "vp_forward_bias_m",
            "vp_lateral_bias_m",
        ):
            if isinstance(legacy_info[key], float):
                assert legacy_info[key] == pytest.approx(boolean_info[key])
            else:
                assert legacy_info[key] == boolean_info[key]

        legacy_env.close()
        boolean_env.close()

    def test_post_merge_offensive_anchor_blend_release_vp_forward_bias_clamp_applies_before_guidance(
        self,
        base_config,
    ):
        config = copy.deepcopy(base_config)
        config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
        config["virtual_point"]["anchor_mode"] = "predicted_target"
        config["virtual_point"]["close_range_anchor_mode"] = None
        config["virtual_point"]["close_range_anchor_mode_by_task"] = {}
        config["virtual_point"]["post_merge_offensive_anchor_blend_by_task"] = {
            "head_on": 0.25,
        }
        config["virtual_point"][
            "post_merge_offensive_anchor_blend_release_blend_by_task"
        ] = {
            "head_on": 0.0,
        }
        config["virtual_point"][
            "post_merge_offensive_anchor_blend_release_ego_only_streak_steps_by_task"
        ] = {
            "head_on": 1,
        }
        config["virtual_point"][
            "post_merge_offensive_anchor_blend_release_direct_track_enabled_by_task"
        ] = {"head_on": False}
        config["virtual_point"][
            "post_merge_offensive_anchor_blend_release_vp_forward_bias_m_min_by_task"
        ] = {"head_on": -1000.0}
        config["trajectory_prediction"] = {"enabled": True}

        env = CloseRangeTrackingEnv(config)
        scenario = {
            "own_init": {
                "position_m": [0.0, 0.0, 1000.0],
                "velocity_mps": 200.0,
                "heading_deg": 0.0,
            },
            "target_init": {
                "position_m": [1500.0, 0.0, 1000.0],
                "velocity_mps": 200.0,
                "heading_deg": 0.0,
            },
        }
        env.reset(scenario=scenario, seed=0)

        class _FakePredictor:
            def predict(self, target_state):
                target_pos = np.asarray(target_state["position_m"], dtype=np.float64)
                return target_pos + np.array([100.0, 0.0, 0.0]), None, {}

        env.trajectory_predictor_adapter = _FakePredictor()
        env._first_pass_complete = True
        env._first_pass_completion_step = 0
        env._post_merge_offensive_anchor_blend_released = True
        env._evaluate_post_merge_offensive_anchor_geometry_disadvantage = (
            lambda own_state, target_state, rel_state=None: {
                "requires_geometry_disadvantage": True,
                "condition_met": False,
                "target_only_attack_zone_disadvantage": False,
                "geometry_disadvantage": False,
                "alignment_disadvantage": False,
                "ego_attack_score": 0.0,
                "target_attack_score": 0.0,
                "ego_in_attack_zone": False,
                "target_in_attack_zone": False,
                "aa_deg_min": np.nan,
                "aa_deg": np.nan,
                "range_rate_mps": 250.0,
                "range_opening": True,
            }
        )

        def _fake_action_to_virtual_point(
            action,
            own_state,
            target_state,
            anchor_mode="current_target",
            **kwargs,
        ):
            target_pos = np.asarray(target_state["position_neu"], dtype=np.float64)
            vp_pos = target_pos + np.array([-1500.0, 0.0, 0.0], dtype=np.float64)
            return (
                {
                    "position_neu": vp_pos.copy(),
                    "position": vp_pos.copy(),
                },
                {
                    "anchor_mode": anchor_mode,
                    "anchor_pos": target_pos.copy(),
                    "offset": np.zeros(3, dtype=np.float64),
                    "world_offset": vp_pos - target_pos,
                    "offset_frame": "world_neu",
                    "predicted_target_blend": 1.0,
                    "predicted_target_forward_scale": 1.0,
                    "offensive_anchor_blend": 0.0,
                    "longitudinal_scale": 1.0,
                    "lateral_scale": 1.0,
                },
            )

        env.virtual_point_generator.action_to_virtual_point = _fake_action_to_virtual_point

        captured = {}

        def _fake_compute_command(
            own_state_arg,
            target_state_arg,
            virtual_point_arg,
            current_gains_arg,
        ):
            captured["virtual_point"] = {
                key: np.asarray(value, dtype=np.float64).copy()
                if isinstance(value, np.ndarray)
                else value
                for key, value in virtual_point_arg.items()
            }
            captured["target_state"] = {
                key: np.asarray(value, dtype=np.float64).copy()
                if isinstance(value, np.ndarray)
                else value
                for key, value in target_state_arg.items()
            }
            return {
                "nz_cmd": 0.0,
                "roll_rate_cmd": 0.0,
                "throttle_cmd": 0.5,
            }

        env.guidance.compute_command = _fake_compute_command

        _, _, _, _, info = env.step(np.zeros(3))

        target_pos = np.asarray(captured["target_state"]["position_m"], dtype=np.float64)
        clamped_vp_pos = np.asarray(
            captured["virtual_point"]["position_neu"],
            dtype=np.float64,
        )
        clamped_target_relative_vp = world_to_offset_frame(
            clamped_vp_pos - target_pos,
            "target_velocity",
            target_state=captured["target_state"],
        )

        assert (
            info["post_merge_offensive_anchor_blend_release_vp_forward_bias_m_min"]
            == pytest.approx(-1000.0)
        )
        assert (
            info["post_merge_offensive_anchor_blend_release_preclamp_vp_forward_bias_m"]
            == pytest.approx(-1500.0)
        )
        assert info["post_merge_offensive_anchor_blend_release_vp_forward_bias_clamp_active"] is True
        assert info["post_merge_offensive_anchor_blend_release_recovery_active"] is False
        assert info["post_merge_offensive_anchor_blend_release_direct_track_active"] is False
        assert info["direct_track_mode_effective"] is False
        assert clamped_target_relative_vp[0] == pytest.approx(-1000.0)
        env.close()

    def test_post_merge_offensive_anchor_blend_release_forward_bias_clamp_negative_lateral_altitude_gate(
        self,
        base_config,
    ):
        def _run_case(own_altitude_m, expect_clamped):
            config = copy.deepcopy(base_config)
            config["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
            config["virtual_point"]["anchor_mode"] = "predicted_target"
            config["virtual_point"]["close_range_anchor_mode"] = None
            config["virtual_point"]["close_range_anchor_mode_by_task"] = {}
            config["virtual_point"]["post_merge_offensive_anchor_blend_by_task"] = {
                "head_on": 0.25,
            }
            config["virtual_point"][
                "post_merge_offensive_anchor_blend_release_blend_by_task"
            ] = {
                "head_on": 0.0,
            }
            config["virtual_point"][
                "post_merge_offensive_anchor_blend_release_ego_only_streak_steps_by_task"
            ] = {
                "head_on": 1,
            }
            config["virtual_point"][
                "post_merge_offensive_anchor_blend_release_direct_track_enabled_by_task"
            ] = {"head_on": False}
            config["virtual_point"][
                "post_merge_offensive_anchor_blend_release_vp_forward_bias_m_min_by_task"
            ] = {"head_on": -1000.0}
            config["virtual_point"][
                "post_merge_offensive_anchor_blend_release_vp_forward_bias_clamp_negative_lateral_below_altitude_m_by_task"
            ] = {"head_on": 4500.0}
            config["trajectory_prediction"] = {"enabled": True}

            env = CloseRangeTrackingEnv(config)
            scenario = {
                "own_init": {
                    "position_m": [0.0, 0.0, own_altitude_m],
                    "velocity_mps": 200.0,
                    "heading_deg": 0.0,
                },
                "target_init": {
                    "position_m": [1500.0, 0.0, own_altitude_m],
                    "velocity_mps": 200.0,
                    "heading_deg": 0.0,
                },
            }
            env.reset(scenario=scenario, seed=0)

            class _FakePredictor:
                def predict(self, target_state):
                    target_pos = np.asarray(target_state["position_m"], dtype=np.float64)
                    return target_pos + np.array([100.0, 0.0, 0.0]), None, {}

            env.trajectory_predictor_adapter = _FakePredictor()
            env._first_pass_complete = True
            env._first_pass_completion_step = 0
            env._post_merge_offensive_anchor_blend_released = True
            env._evaluate_post_merge_offensive_anchor_geometry_disadvantage = (
                lambda own_state, target_state, rel_state=None: {
                    "requires_geometry_disadvantage": True,
                    "condition_met": False,
                    "target_only_attack_zone_disadvantage": False,
                    "geometry_disadvantage": False,
                    "alignment_disadvantage": False,
                    "ego_attack_score": 0.0,
                    "target_attack_score": 0.0,
                    "ego_in_attack_zone": False,
                    "target_in_attack_zone": False,
                    "aa_deg_min": np.nan,
                    "aa_deg": np.nan,
                    "range_rate_mps": 250.0,
                    "range_opening": True,
                }
            )

            def _fake_action_to_virtual_point(
                action,
                own_state,
                target_state,
                anchor_mode="current_target",
                **kwargs,
            ):
                target_pos = np.asarray(target_state["position_neu"], dtype=np.float64)
                vp_pos = target_pos + np.array(
                    [-1500.0, -500.0, 0.0],
                    dtype=np.float64,
                )
                return (
                    {
                        "position_neu": vp_pos.copy(),
                        "position": vp_pos.copy(),
                    },
                    {
                        "anchor_mode": anchor_mode,
                        "anchor_pos": target_pos.copy(),
                        "offset": np.zeros(3, dtype=np.float64),
                        "world_offset": vp_pos - target_pos,
                        "offset_frame": "world_neu",
                        "predicted_target_blend": 1.0,
                        "predicted_target_forward_scale": 1.0,
                        "offensive_anchor_blend": 0.0,
                        "longitudinal_scale": 1.0,
                        "lateral_scale": 1.0,
                    },
                )

            env.virtual_point_generator.action_to_virtual_point = _fake_action_to_virtual_point

            captured = {}

            def _fake_compute_command(
                own_state_arg,
                target_state_arg,
                virtual_point_arg,
                current_gains_arg,
            ):
                captured["virtual_point"] = {
                    key: np.asarray(value, dtype=np.float64).copy()
                    if isinstance(value, np.ndarray)
                    else value
                    for key, value in virtual_point_arg.items()
                }
                captured["target_state"] = {
                    key: np.asarray(value, dtype=np.float64).copy()
                    if isinstance(value, np.ndarray)
                    else value
                    for key, value in target_state_arg.items()
                }
                return {
                    "nz_cmd": 0.0,
                    "roll_rate_cmd": 0.0,
                    "throttle_cmd": 0.5,
                }

            env.guidance.compute_command = _fake_compute_command

            try:
                _, _, _, _, info = env.step(np.zeros(3))

                target_pos = np.asarray(
                    captured["target_state"]["position_m"],
                    dtype=np.float64,
                )
                candidate_vp_pos = np.asarray(
                    captured["virtual_point"]["position_neu"],
                    dtype=np.float64,
                )
                target_relative_vp = world_to_offset_frame(
                    candidate_vp_pos - target_pos,
                    "target_velocity",
                    target_state=captured["target_state"],
                )

                assert (
                    info[
                        "post_merge_offensive_anchor_blend_release_vp_forward_bias_clamp_negative_lateral_below_altitude_m"
                    ]
                    == pytest.approx(4500.0)
                )
                assert (
                    info["post_merge_offensive_anchor_blend_release_preclamp_vp_forward_bias_m"]
                    == pytest.approx(-1500.0)
                )
                assert target_relative_vp[1] < 0.0
                assert (
                    info["post_merge_offensive_anchor_blend_release_recovery_active"]
                    is False
                )
                assert (
                    info["post_merge_offensive_anchor_blend_release_direct_track_active"]
                    is False
                )

                if expect_clamped:
                    assert (
                        info[
                            "post_merge_offensive_anchor_blend_release_vp_forward_bias_clamp_active"
                        ]
                        is True
                    )
                    assert target_relative_vp[0] == pytest.approx(-1000.0)
                else:
                    assert (
                        info[
                            "post_merge_offensive_anchor_blend_release_vp_forward_bias_clamp_active"
                        ]
                        is False
                    )
                    assert target_relative_vp[0] == pytest.approx(-1500.0)
            finally:
                env.close()

        _run_case(own_altitude_m=5000.0, expect_clamped=False)
        _run_case(own_altitude_m=4000.0, expect_clamped=True)
