"""
Unit tests for virtual point generation.
"""

import numpy as np
from uav_vpp_guidance.virtual_point.generator import VirtualPointGenerator
from uav_vpp_guidance.virtual_point.smoother import VirtualPointSmoother


class TestVirtualPointGenerator:
    def test_init(self):
        config = {"action_dim": 5}
        gen = VirtualPointGenerator(config)
        assert gen.action_dim == 5

    def test_action_to_virtual_point_current_target(self):
        config = {"action_dim": 5}
        gen = VirtualPointGenerator(config)
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        own_state = {"position_neu": np.zeros(3)}
        target_state = {"position_neu": np.array([1000.0, 0.0, 0.0])}
        vp = gen.action_to_virtual_point(action, own_state, target_state)
        assert np.allclose(vp["position"], np.array([1000.0, 0.0, 0.0]))

    def test_get_target_position_with_position_m(self):
        """_get_target_position 应支持 position_m 字段。"""
        target_state = {"position_m": np.array([500.0, 200.0, 300.0])}
        pos = VirtualPointGenerator._get_target_position(target_state)
        assert np.allclose(pos, np.array([500.0, 200.0, 300.0]))

    def test_get_target_position_priority_neu_over_m(self):
        """position_neu 优先于 position_m。"""
        target_state = {
            "position_neu": np.array([1.0, 2.0, 3.0]),
            "position_m": np.array([4.0, 5.0, 6.0]),
        }
        pos = VirtualPointGenerator._get_target_position(target_state)
        assert np.allclose(pos, np.array([1.0, 2.0, 3.0]))

    def test_constant_velocity_with_velocity_vector_mps(self):
        """constant_velocity 应支持 velocity_vector_mps (NEU)。"""
        target_state = {
            "position_neu": np.array([1000.0, 0.0, 0.0]),
            "velocity_vector_mps": np.array([50.0, 0.0, 10.0]),
        }
        pos = VirtualPointGenerator._constant_velocity_prediction(target_state, 2.0)
        assert np.allclose(pos, np.array([1100.0, 0.0, 20.0]))

    def test_constant_velocity_with_velocity_ned_converts_to_neu(self):
        """velocity_ned [vn, ve, vd] 应转换为 NEU [vn, ve, -vd]。"""
        target_state = {
            "position_neu": np.array([1000.0, 0.0, 0.0]),
            "velocity_ned": np.array([50.0, 0.0, -10.0]),  # vd=-10 -> 向上 10
        }
        pos = VirtualPointGenerator._constant_velocity_prediction(target_state, 2.0)
        assert np.allclose(pos, np.array([1100.0, 0.0, 20.0]))

    def test_constant_velocity_fallback_to_velocity(self):
        """velocity 字段作为 fallback。"""
        target_state = {
            "position_neu": np.array([1000.0, 0.0, 0.0]),
            "velocity": np.array([30.0, 0.0, 5.0]),
        }
        pos = VirtualPointGenerator._constant_velocity_prediction(target_state, 2.0)
        assert np.allclose(pos, np.array([1060.0, 0.0, 10.0]))

    def test_constant_velocity_missing_velocity_returns_current_pos(self):
        """缺少速度信息时返回当前位置。"""
        target_state = {"position_neu": np.array([1000.0, 0.0, 0.0])}
        pos = VirtualPointGenerator._constant_velocity_prediction(target_state, 2.0)
        assert np.allclose(pos, np.array([1000.0, 0.0, 0.0]))


class TestVirtualPointSmoother:
    def test_smooth_first_call(self):
        smoother = VirtualPointSmoother(alpha=0.3)
        point = np.array([1.0, 2.0, 3.0])
        out = smoother.smooth(point)
        assert np.allclose(out, point)

    def test_smooth_subsequent(self):
        smoother = VirtualPointSmoother(alpha=0.5)
        p1 = np.array([1.0, 1.0, 1.0])
        p2 = np.array([3.0, 3.0, 3.0])
        smoother.smooth(p1)
        out = smoother.smooth(p2)
        expected = 0.5 * p2 + 0.5 * p1
        assert np.allclose(out, expected)
