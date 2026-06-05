"""Tests for Stage 6G.3 oracle and rule-based anchor modes."""

import unittest
import numpy as np

from uav_vpp_guidance.virtual_point.generator import VirtualPointGenerator


class TestOracleFuturePositionAnchor(unittest.TestCase):
    def test_oracle_uses_true_velocity(self):
        gen = VirtualPointGenerator({
            "virtual_point": {
                "d_long": [-100, 100],
                "d_lat": [-100, 100],
                "d_vert": [-50, 50],
            }
        })
        own_state = {"position_neu": np.array([0.0, 0.0, 0.0])}
        target_state = {
            "position_neu": np.array([1000.0, 0.0, 100.0]),
            "velocity_vector_mps": np.array([50.0, 0.0, 0.0]),
        }
        action = np.array([0.0, 0.0, 0.0])
        vp, info = gen.action_to_virtual_point(
            action, own_state, target_state,
            anchor_mode="oracle_future_position",
            lookahead_time_s=2.0,
            return_info=True,
        )
        expected_anchor = np.array([1100.0, 0.0, 100.0])  # pos + vel * 2s
        np.testing.assert_allclose(vp["position"], expected_anchor, rtol=1e-6)
        self.assertEqual(info["anchor_mode"], "oracle_future_position")
        self.assertEqual(info["prediction_info"]["lookahead_time_s"], 2.0)

    def test_oracle_fallback_without_velocity(self):
        gen = VirtualPointGenerator({
            "virtual_point": {
                "d_long": [-100, 100],
                "d_lat": [-100, 100],
                "d_vert": [-50, 50],
            }
        })
        own_state = {"position_neu": np.array([0.0, 0.0, 0.0])}
        target_state = {
            "position_neu": np.array([1000.0, 0.0, 100.0]),
            # No velocity
        }
        action = np.array([0.0, 0.0, 0.0])
        vp, info = gen.action_to_virtual_point(
            action, own_state, target_state,
            anchor_mode="oracle_future_position",
            lookahead_time_s=2.0,
            return_info=True,
        )
        # Should fallback to current position
        np.testing.assert_allclose(vp["position"], np.array([1000.0, 0.0, 100.0]), rtol=1e-6)


class TestRuleBasedPursuitAnchor(unittest.TestCase):
    def test_rule_based_places_anchor_ahead_of_target(self):
        gen = VirtualPointGenerator({
            "virtual_point": {
                "d_long": [-100, 100],
                "d_lat": [-100, 100],
                "d_vert": [-50, 50],
            }
        })
        gen.lead_distance_m = 500.0
        own_state = {"position_neu": np.array([0.0, 0.0, 0.0])}
        target_state = {
            "position_neu": np.array([1000.0, 0.0, 100.0]),
        }
        action = np.array([0.0, 0.0, 0.0])
        vp, info = gen.action_to_virtual_point(
            action, own_state, target_state,
            anchor_mode="rule_based_pursuit",
            return_info=True,
        )
        # Anchor should be 500m ahead of target along LOS
        los = np.array([1000.0, 0.0, 100.0]) - np.array([0.0, 0.0, 0.0])
        distance = np.linalg.norm(los)
        los_unit = los / distance
        expected_anchor = np.array([1000.0, 0.0, 100.0]) + los_unit * 500.0
        np.testing.assert_allclose(vp["position"], expected_anchor, rtol=1e-6)
        self.assertEqual(info["anchor_mode"], "rule_based_pursuit")
        self.assertEqual(info["prediction_info"]["lead_distance_m"], 500.0)

    def test_rule_based_with_zero_distance(self):
        gen = VirtualPointGenerator({
            "virtual_point": {
                "d_long": [-100, 100],
                "d_lat": [-100, 100],
                "d_vert": [-50, 50],
            }
        })
        gen.lead_distance_m = 200.0
        own_state = {"position_neu": np.array([0.0, 0.0, 0.0])}
        target_state = {
            "position_neu": np.array([0.0, 0.0, 0.0]),  # Same position
        }
        action = np.array([0.0, 0.0, 0.0])
        vp, info = gen.action_to_virtual_point(
            action, own_state, target_state,
            anchor_mode="rule_based_pursuit",
            return_info=True,
        )
        # When distance is zero, default LOS unit is [1, 0, 0]
        expected_anchor = np.array([200.0, 0.0, 0.0])
        np.testing.assert_allclose(vp["position"], expected_anchor, rtol=1e-6)


class TestTrackingEnvWithNewAnchorModes(unittest.TestCase):
    def test_env_accepts_oracle_anchor_mode(self):
        from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
        config = {
            "scenarios": {
                "test": {
                    "max_range_m": 5000,
                    "ego": {"position_m": [0, 0, 1000], "velocity_mps": 150},
                    "target": {"position_m": [2000, 0, 1000], "velocity_mps": 150},
                }
            },
            "virtual_point": {
                "anchor_mode": "oracle_future_position",
                "d_long": [-100, 100],
                "d_lat": [-100, 100],
                "d_vert": [-50, 50],
            },
            "guidance": {"mode": "los_rate"},
            "limits": {"max_nz": 6.0, "max_roll_rate": 30.0},
        }
        env = CloseRangeTrackingEnv(config)
        env.reset(scenario="test")
        obs, reward, terminated, truncated, info = env.step(np.zeros(5))
        self.assertIn("anchor_mode", info)
        self.assertEqual(info["anchor_mode"], "oracle_future_position")
        env.close()

    def test_env_accepts_rule_based_anchor_mode(self):
        from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
        config = {
            "scenarios": {
                "test": {
                    "max_range_m": 5000,
                    "ego": {"position_m": [0, 0, 1000], "velocity_mps": 150},
                    "target": {"position_m": [2000, 0, 1000], "velocity_mps": 150},
                }
            },
            "virtual_point": {
                "anchor_mode": "rule_based_pursuit",
                "d_long": [-100, 100],
                "d_lat": [-100, 100],
                "d_vert": [-50, 50],
            },
            "guidance": {"mode": "los_rate"},
            "limits": {"max_nz": 6.0, "max_roll_rate": 30.0},
        }
        env = CloseRangeTrackingEnv(config)
        env.reset(scenario="test")
        obs, reward, terminated, truncated, info = env.step(np.zeros(5))
        self.assertIn("anchor_mode", info)
        self.assertEqual(info["anchor_mode"], "rule_based_pursuit")
        env.close()


if __name__ == "__main__":
    unittest.main()
