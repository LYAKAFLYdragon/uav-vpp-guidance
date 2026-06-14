"""
Tests for observation construction utilities, focusing on coordinate-frame
consistency (NEU) and velocity field extraction priority.
"""

import pytest
import numpy as np

from uav_vpp_guidance.envs.observation import (
    compute_relative_geometry,
    _get_velocity,
    build_observation,
    ObservationBuilder,
)


class TestGetVelocity:
    def test_velocity_vector_mps_priority(self):
        """velocity_vector_mps 优先于 velocity_ned。"""
        state = {
            "velocity_vector_mps": np.array([10.0, 20.0, 30.0]),
            "velocity_ned": np.array([1.0, 2.0, 3.0]),
        }
        vel = _get_velocity(state)
        assert np.allclose(vel, np.array([10.0, 20.0, 30.0]))

    def test_velocity_ned_converted_to_neu(self):
        """velocity_ned [vn, ve, vd] 应转换为 NEU [vn, ve, -vd]。"""
        state = {"velocity_ned": np.array([100.0, 50.0, -20.0])}
        vel = _get_velocity(state)
        assert np.allclose(vel, np.array([100.0, 50.0, 20.0]))

    def test_velocity_fallback(self):
        """只有 velocity 字段时直接使用。"""
        state = {"velocity": np.array([5.0, 6.0, 7.0])}
        vel = _get_velocity(state)
        assert np.allclose(vel, np.array([5.0, 6.0, 7.0]))

    def test_missing_velocity_raises(self):
        with pytest.raises(ValueError, match="missing velocity field"):
            _get_velocity({})

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError, match="must be a 3-element vector"):
            _get_velocity({"velocity_vector_mps": np.array([1.0, 2.0])})


class TestComputeRelativeGeometry:
    def test_range_rate_positive_when_target_approaching_from_above(self):
        """
        目标在上方（z=100）并以向上速度（vu=10）接近静止本机。
        在 NEU 中，rel_pos = [0,0,100]，rel_vel = [0,0,10]。
        range_rate = dot(rel_vel, rel_pos) / range = 10*100/100 = 10 > 0。
        正值表示目标正在远离本机（从本机视角，上方目标向上飞）。
        """
        own_state = {
            "position_m": np.array([0.0, 0.0, 0.0]),
            "velocity_vector_mps": np.array([0.0, 0.0, 0.0]),
        }
        target_state = {
            "position_m": np.array([0.0, 0.0, 100.0]),
            "velocity_vector_mps": np.array([0.0, 0.0, 10.0]),
        }
        rel = compute_relative_geometry(own_state, target_state)
        assert rel["range_rate_mps"] > 0.0

    def test_range_rate_negative_when_target_descending_toward_ownship(self):
        """
        目标在上方并以向下速度接近静止本机。
        rel_vel = [0,0,-10]，range_rate < 0 表示接近。
        """
        own_state = {
            "position_m": np.array([0.0, 0.0, 0.0]),
            "velocity_vector_mps": np.array([0.0, 0.0, 0.0]),
        }
        target_state = {
            "position_m": np.array([0.0, 0.0, 100.0]),
            "velocity_vector_mps": np.array([0.0, 0.0, -10.0]),
        }
        rel = compute_relative_geometry(own_state, target_state)
        assert rel["range_rate_mps"] < 0.0

    def test_velocity_ned_and_position_mixed(self):
        """position_m (NEU) + velocity_ned (自动转 NEU) 组合不应混用坐标系。"""
        own_state = {
            "position_m": np.array([0.0, 0.0, 0.0]),
            "velocity_ned": np.array([0.0, 0.0, 0.0]),
        }
        target_state = {
            "position_m": np.array([0.0, 0.0, 100.0]),
            "velocity_ned": np.array([0.0, 0.0, -10.0]),  # vd=-10 -> NEU vu=10（向上）
        }
        rel = compute_relative_geometry(own_state, target_state)
        # target 在上方且向上飞（远离本机），range_rate 应为正
        assert rel["range_rate_mps"] > 0.0

    def test_velocity_ned_descending_target_approaches(self):
        """velocity_ned vd=+10（向下）-> NEU vu=-10（向下），目标在上方下降应接近。"""
        own_state = {
            "position_m": np.array([0.0, 0.0, 0.0]),
            "velocity_ned": np.array([0.0, 0.0, 0.0]),
        }
        target_state = {
            "position_m": np.array([0.0, 0.0, 100.0]),
            "velocity_ned": np.array([0.0, 0.0, 10.0]),  # vd=+10 -> NEU vu=-10（向下）
        }
        rel = compute_relative_geometry(own_state, target_state)
        # target 在上方且向下飞（接近本机），range_rate 应为负
        assert rel["range_rate_mps"] < 0.0


class TestBuildObservation:
    def _make_states(self):
        own_state = {
            "position_m": np.array([0.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
        }
        target_state = {
            "position_m": np.array([1000.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([180.0, 0.0, 0.0]),
        }
        return own_state, target_state

    def test_base_observation_shape(self):
        own_state, target_state = self._make_states()
        obs = build_observation(own_state, target_state)
        assert isinstance(obs, np.ndarray)
        assert obs.ndim == 1
        assert obs.shape[0] == 16

    def test_ref_range_scaling(self):
        """range 应使用 3000 作为归一化参考值。"""
        own_state, target_state = self._make_states()
        obs = build_observation(own_state, target_state)
        # range = 1000m, ref_range = 3000 -> first element = 1/3
        assert obs[0] == pytest.approx(1000.0 / 3000.0)

    def test_include_temporal_features(self):
        own_state, target_state = self._make_states()
        prev_rel = compute_relative_geometry(own_state, target_state)
        prev_rel["range_rate_mps"] = 50.0
        prev_rel["ata_rad"] = 0.1
        prev_rel["aa_rad"] = 0.2
        obs = build_observation(
            own_state,
            target_state,
            prev_rel_state=prev_rel,
            include_temporal=True,
        )
        # base 16 + temporal 3 = 19
        assert obs.shape[0] == 19

    def test_include_vp_error(self):
        own_state, target_state = self._make_states()
        guidance_state = {"vp_tracking_error": np.array([100.0, 50.0, 20.0])}
        obs = build_observation(
            own_state,
            target_state,
            guidance_state=guidance_state,
            include_vp_error=True,
        )
        assert obs.shape[0] == 19

    def test_include_scenario_onehot(self):
        own_state, target_state = self._make_states()
        obs = build_observation(
            own_state,
            target_state,
            scenario_type="favorable",
            include_scenario=True,
        )
        assert obs.shape[0] == 20
        # favorable = first scenario index -> last 4 elements = [1, 0, 0, 0]
        assert np.allclose(obs[-4:], np.array([1.0, 0.0, 0.0, 0.0]))


class TestObservationBuilder:
    def _make_states(self):
        own_state = {
            "position_m": np.array([0.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
        }
        target_state = {
            "position_m": np.array([1000.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([180.0, 0.0, 0.0]),
        }
        return own_state, target_state

    def test_temporal_builder_includes_history(self):
        config = {
            "observation": {
                "temporal": {"enabled": True, "history_length": 2, "dt": 0.2},
            }
        }
        builder = ObservationBuilder(config)
        own_state, target_state = self._make_states()
        # Dimension stays constant (base + temporal slots) even without history.
        obs1 = builder.build(own_state, target_state)
        assert obs1.shape[0] == 19
        # First temporal derivatives are zero when history is missing.
        assert np.allclose(obs1[-3:], 0.0)
        # Second call should include non-zero temporal derivatives.
        obs2 = builder.build(own_state, target_state)
        assert obs2.shape[0] == 19

    def test_builder_reset_clears_history(self):
        config = {
            "observation": {
                "temporal": {"enabled": True, "history_length": 2, "dt": 0.2},
            }
        }
        builder = ObservationBuilder(config)
        own_state, target_state = self._make_states()
        builder.build(own_state, target_state)
        builder.build(own_state, target_state)  # now has temporal features
        builder.reset()
        obs = builder.build(own_state, target_state)
        # After reset, observation dimension is unchanged but temporal slots are zero.
        assert obs.shape[0] == 19
        assert np.allclose(obs[-3:], 0.0)
