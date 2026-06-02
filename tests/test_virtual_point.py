"""
Unit tests for virtual point generation.
"""

import pytest
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
