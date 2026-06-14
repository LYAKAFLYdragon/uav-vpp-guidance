"""
Tests for VirtualPointGenerator action-to-VPP mapping and constraints.
"""

import numpy as np
import pytest

from uav_vpp_guidance.virtual_point.generator import VirtualPointGenerator


class TestVirtualPointGenerator:
    def _make_config(self, **overrides):
        config = {
            "action_dim": 3,
            "d_long_range": [-1500.0, 1500.0],
            "d_lat_range": [-800.0, 800.0],
            "d_vert_range": [-500.0, 500.0],
            "smoothing_alpha": 0.3,
        }
        config.update(overrides)
        return config

    def _make_states(self):
        own_state = {
            "position_neu": np.array([0.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
        }
        target_state = {
            "position_neu": np.array([1000.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([180.0, 0.0, 0.0]),
        }
        return own_state, target_state

    def test_action_to_virtual_point_default_range(self):
        gen = VirtualPointGenerator(self._make_config())
        own_state, target_state = self._make_states()
        # action = [1, 1, 1] should map to max offset
        vp = gen.action_to_virtual_point(
            np.array([1.0, 1.0, 1.0]),
            own_state,
            target_state,
            anchor_mode="current_target",
        )
        offset = vp["offset"]
        assert offset[0] == pytest.approx(1500.0)
        assert offset[1] == pytest.approx(800.0)
        assert offset[2] == pytest.approx(500.0)

    def test_dynamic_offset_scaling(self):
        gen = VirtualPointGenerator(
            self._make_config(dynamic_offset_scale=0.5)
        )
        own_state, target_state = self._make_states()
        initial_range_m = 800.0
        vp = gen.action_to_virtual_point(
            np.array([1.0, 1.0, 1.0]),
            own_state,
            target_state,
            anchor_mode="current_target",
            initial_range_m=initial_range_m,
        )
        offset = vp["offset"]
        # 50% of 800m = 400m for longitudinal; lateral/vertical scale proportionally
        assert offset[0] == pytest.approx(400.0)
        assert offset[1] == pytest.approx(400.0 * 800.0 / 1500.0)
        assert offset[2] == pytest.approx(400.0 * 500.0 / 1500.0)

    def test_dynamic_offset_scaling_ignored_without_initial_range(self):
        gen = VirtualPointGenerator(
            self._make_config(dynamic_offset_scale=0.5)
        )
        own_state, target_state = self._make_states()
        vp = gen.action_to_virtual_point(
            np.array([1.0, 1.0, 1.0]),
            own_state,
            target_state,
            anchor_mode="current_target",
        )
        offset = vp["offset"]
        assert offset[0] == pytest.approx(1500.0)

    def test_dynamics_aware_constraint(self):
        gen = VirtualPointGenerator(
            self._make_config(
                dynamics_aware=True,
                max_heading_rate=0.3,
                lookahead_steps=5,
            )
        )
        own_state = {
            "position_neu": np.array([0.0, 0.0, 5000.0]),
            # Heading is 0 deg (east); velocity east
            "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
        }
        # Target placed almost directly north -> demand ~90 deg heading change
        target_state = {
            "position_neu": np.array([0.0, 1000.0, 5000.0]),
            "velocity_vector_mps": np.array([0.0, 0.0, 0.0]),
        }
        # action = 0 -> zero offset, anchor = target
        vp = gen.action_to_virtual_point(
            np.array([0.0, 0.0, 0.0]),
            own_state,
            target_state,
            anchor_mode="current_target",
        )
        # VPP should be constrained to feasible heading sector
        vp_pos = vp["position"] - own_state["position_neu"]
        vp_heading = np.arctan2(vp_pos[1], vp_pos[0])
        # own heading is 0, max feasible = 0.3 * 0.2 * 5 = 0.3 rad
        assert abs(vp_heading) <= 0.35

    def test_dynamics_aware_disabled(self):
        gen = VirtualPointGenerator(self._make_config(dynamics_aware=False))
        own_state = {
            "position_neu": np.array([0.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
        }
        target_state = {
            "position_neu": np.array([0.0, 1000.0, 5000.0]),
            "velocity_vector_mps": np.array([0.0, 0.0, 0.0]),
        }
        vp = gen.action_to_virtual_point(
            np.array([0.0, 0.0, 0.0]),
            own_state,
            target_state,
            anchor_mode="current_target",
        )
        # No constraint: VPP should be exactly at target (north)
        assert np.allclose(vp["position"], target_state["position_neu"])
