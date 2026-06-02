"""
Tests for rule-based pursuit baseline.
"""

import pytest
import numpy as np
from uav_vpp_guidance.baselines.rule_based_pursuit import RuleBasedPursuitPolicy


class TestRuleBasedPursuitPolicy:
    def test_pure_pursuit_returns_zero_action(self):
        policy = RuleBasedPursuitPolicy(mode="pure_pursuit")
        own_state = {"position_m": np.zeros(3), "velocity_vector_mps": np.array([200.0, 0.0, 0.0])}
        target_state = {"position_m": np.array([1000.0, 0.0, 5000.0]), "velocity_vector_mps": np.array([200.0, 0.0, 0.0])}
        rel_state = {}
        action = policy.get_action(own_state, target_state, rel_state)
        assert np.allclose(action, 0.0, atol=1e-8)

    def test_lag_pursuit_nonzero(self):
        policy = RuleBasedPursuitPolicy(mode="lag_pursuit", lag_distance_m=500.0)
        own_state = {"position_m": np.zeros(3), "velocity_vector_mps": np.array([200.0, 0.0, 0.0])}
        target_state = {
            "position_m": np.array([1000.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
        }
        rel_state = {}
        action = policy.get_action(own_state, target_state, rel_state)
        assert action.shape == (3,)
        assert np.all(np.abs(action) <= 1.0)
        # Target moving along +x, lag should put point behind (negative x offset)
        assert action[0] < -0.01

    def test_lead_pursuit_nonzero(self):
        policy = RuleBasedPursuitPolicy(mode="lead_pursuit", lag_distance_m=500.0)
        own_state = {"position_m": np.zeros(3), "velocity_vector_mps": np.array([200.0, 0.0, 0.0])}
        target_state = {
            "position_m": np.array([1000.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
        }
        rel_state = {}
        action = policy.get_action(own_state, target_state, rel_state)
        assert action.shape == (3,)
        assert np.all(np.abs(action) <= 1.0)
        # Lead should put point ahead (positive x offset)
        assert action[0] > 0.01

    def test_policy_output_compatible_with_vpg(self):
        """输出格式与 VirtualPointGenerator 兼容（3维 [-1,1]）。"""
        for mode in ["pure_pursuit", "lag_pursuit", "lead_pursuit"]:
            policy = RuleBasedPursuitPolicy(mode=mode)
            own_state = {"position_m": np.zeros(3), "velocity_vector_mps": np.array([200.0, 0.0, 0.0])}
            target_state = {
                "position_m": np.array([1000.0, 0.0, 5000.0]),
                "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
            }
            action = policy.get_action(own_state, target_state, {})
            assert action.dtype == np.float64
            assert len(action) == 3
            assert np.all(action >= -1.0) and np.all(action <= 1.0)
