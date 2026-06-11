"""
Tests for JSBSim crossing scenario fix.

Covers:
  - Turn-rate penalty in RewardCalculator
  - Dynamics-aware constraint in VirtualPointGenerator
  - Crossing scenario feasibility on simple backend
"""

import math
import numpy as np
import pytest

from uav_vpp_guidance.envs.reward import RewardCalculator
from uav_vpp_guidance.virtual_point.generator import VirtualPointGenerator


class TestTurnRatePenalty:
    """Test turn-rate penalty in RewardCalculator."""

    def test_no_penalty_for_feasible_heading(self):
        """Small heading error should not incur penalty."""
        config = {
            "reward": {
                "w_turn_rate": 1.0,
                "max_heading_rate": 0.3,
            }
        }
        calc = RewardCalculator(config)
        info = {
            "relative_state": {
                "range_m": 2000.0,
                "los_azimuth_rad": 0.1,
                "ata_rad": 0.1,
                "aa_rad": 0.1,
            },
            "own_state": {
                "speed_mps": 200.0,
                "yaw_rad": 0.0,
                "altitude_m": 5000.0,
            },
            "command": {"nz_cmd": 1.0, "roll_rate_cmd": 0.0, "throttle_cmd": 0.7},
        }
        reward, terms = calc.compute(info)
        # Small heading error → no turn penalty
        assert terms["reward_turn"] == pytest.approx(0.0, abs=1e-6)

    def test_penalty_for_large_heading_error(self):
        """Large heading error (crossing-like) should incur penalty."""
        config = {
            "reward": {
                "w_turn_rate": 1.0,
                "max_heading_rate": 0.3,
            }
        }
        calc = RewardCalculator(config)
        info = {
            "relative_state": {
                "range_m": 2000.0,
                "los_azimuth_rad": math.pi / 2,  # 90° to the right
                "ata_rad": math.pi / 2,
                "aa_rad": math.pi / 2,
            },
            "own_state": {
                "speed_mps": 200.0,
                "yaw_rad": 0.0,
                "altitude_m": 5000.0,
            },
            "command": {"nz_cmd": 1.0, "roll_rate_cmd": 1.5, "throttle_cmd": 0.7},
        }
        reward, terms = calc.compute(info)
        # Large heading error → turn penalty should be negative
        assert terms["reward_turn"] < 0.0

    def test_penalty_scales_with_w_turn_rate(self):
        """Higher w_turn_rate → more negative reward_turn."""
        base_info = {
            "relative_state": {
                "range_m": 2000.0,
                "los_azimuth_rad": math.pi / 2,
                "ata_rad": math.pi / 2,
                "aa_rad": math.pi / 2,
            },
            "own_state": {
                "speed_mps": 200.0,
                "yaw_rad": 0.0,
                "altitude_m": 5000.0,
            },
            "command": {"nz_cmd": 1.0, "roll_rate_cmd": 1.5, "throttle_cmd": 0.7},
        }
        calc_low = RewardCalculator({"reward": {"w_turn_rate": 0.5, "max_heading_rate": 0.3}})
        calc_high = RewardCalculator({"reward": {"w_turn_rate": 2.0, "max_heading_rate": 0.3}})

        _, terms_low = calc_low.compute(base_info)
        _, terms_high = calc_high.compute(base_info)

        assert terms_high["reward_turn"] < terms_low["reward_turn"]

    def test_zero_w_turn_rate_disables_penalty(self):
        """w_turn_rate=0 should disable the penalty entirely."""
        config = {
            "reward": {
                "w_turn_rate": 0.0,
                "max_heading_rate": 0.3,
            }
        }
        calc = RewardCalculator(config)
        info = {
            "relative_state": {
                "range_m": 2000.0,
                "los_azimuth_rad": math.pi / 2,
                "ata_rad": math.pi / 2,
                "aa_rad": math.pi / 2,
            },
            "own_state": {
                "speed_mps": 200.0,
                "yaw_rad": 0.0,
                "altitude_m": 5000.0,
            },
            "command": {"nz_cmd": 1.0, "roll_rate_cmd": 1.5, "throttle_cmd": 0.7},
        }
        reward, terms = calc.compute(info)
        assert terms["reward_turn"] == pytest.approx(0.0, abs=1e-6)


class TestDynamicsAwareConstraint:
    """Test dynamics-aware constraint in VirtualPointGenerator."""

    def test_no_constraint_when_disabled(self):
        """dynamics_aware=false should not modify virtual point."""
        gen = VirtualPointGenerator({
            "dynamics_aware": False,
            "d_long_range": [-1500.0, 1500.0],
            "d_lat_range": [-800.0, 800.0],
            "d_vert_range": [-500.0, 500.0],
        })
        own_state = {
            "position_m": np.array([0.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
            "yaw_rad": 0.0,
        }
        target_state = {
            "position_neu": np.array([0.0, 2000.0, 5000.0]),
        }
        action = np.array([0.0, 0.0, 0.0])  # zero offset → VP = target position
        vp, info = gen.action_to_virtual_point(
            action, own_state, target_state, return_info=True
        )
        # Without constraint, VP should be exactly at target position
        expected = np.array([0.0, 2000.0, 5000.0])
        assert np.allclose(vp["position"], expected, atol=1e-6)

    def test_constraint_clips_large_heading_error(self):
        """dynamics_aware=true should clip VP to feasible heading sector."""
        gen = VirtualPointGenerator({
            "dynamics_aware": True,
            "max_heading_rate": 0.3,
            "d_long_range": [-1500.0, 1500.0],
            "d_lat_range": [-800.0, 800.0],
            "d_vert_range": [-500.0, 500.0],
        })
        own_state = {
            "position_m": np.array([0.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
            "yaw_rad": 0.0,
        }
        target_state = {
            "position_neu": np.array([0.0, 2000.0, 5000.0]),
        }
        action = np.array([0.0, 0.0, 0.0])  # VP at target [0, 2000, 5000]
        vp, info = gen.action_to_virtual_point(
            action, own_state, target_state, return_info=True
        )
        # With constraint, VP should be clipped to feasible sector
        # Horizontal distance should be preserved
        horiz_dist = float(np.linalg.norm(vp["position"][:2]))
        assert horiz_dist == pytest.approx(2000.0, abs=1.0)
        # Heading should be within feasible range (clipped from π/2)
        heading = math.atan2(vp["position"][1], vp["position"][0])
        assert 0.0 < heading < math.pi / 4  # Clipped from π/2 to < π/4

    def test_constraint_preserves_distance(self):
        """Constraint should preserve distance while adjusting heading."""
        gen = VirtualPointGenerator({
            "dynamics_aware": True,
            "max_heading_rate": 0.3,
            "d_long_range": [-1500.0, 1500.0],
            "d_lat_range": [-800.0, 800.0],
            "d_vert_range": [-500.0, 500.0],
        })
        own_state = {
            "position_m": np.array([0.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
            "yaw_rad": 0.0,
        }
        target_state = {
            "position_neu": np.array([0.0, 2000.0, 5000.0]),
        }
        action = np.array([0.0, 0.0, 0.0])
        vp, _ = gen.action_to_virtual_point(
            action, own_state, target_state, return_info=True
        )
        # Distance should be ~2000m (original target distance)
        dist = float(np.linalg.norm(vp["position"][:2]))
        assert dist == pytest.approx(2000.0, abs=10.0)

    def test_small_heading_error_not_clipped(self):
        """Small heading errors should not be clipped."""
        gen = VirtualPointGenerator({
            "dynamics_aware": True,
            "max_heading_rate": 0.3,
            "d_long_range": [-1500.0, 1500.0],
            "d_lat_range": [-800.0, 800.0],
            "d_vert_range": [-500.0, 500.0],
        })
        own_state = {
            "position_m": np.array([0.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
            "yaw_rad": 0.0,
        }
        # Target slightly to the right (10°)
        target_state = {
            "position_neu": np.array([2000.0 * math.cos(0.17), 2000.0 * math.sin(0.17), 5000.0]),
        }
        action = np.array([0.0, 0.0, 0.0])
        vp, _ = gen.action_to_virtual_point(
            action, own_state, target_state, return_info=True
        )
        # Should not be clipped
        assert np.allclose(vp["position"], target_state["position_neu"], atol=1.0)


class TestCrossingScenarioSimpleBackend:
    """Verify crossing scenarios still work on simple backend."""

    def test_crossing_left_smoke(self):
        """Crossing left should succeed on simple backend."""
        from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
        from uav_vpp_guidance.envs.scenario_registry import ScenarioRegistry

        config = {
            "backend": "simple",
            "env": {
                "decision_freq": 5,
                "max_high_level_steps": 512,
                "success_range_m": 900.0,
                "success_ata_deg": 25.0,
                "success_hold_time_s": 0.2,
                "hysteresis_range_m": 950.0,
                "hysteresis_ata_deg": 30.0,
                "min_altitude_m": 500.0,
                "max_altitude_m": 15000.0,
                "max_range_m": 8000.0,
            },
            "virtual_point": {
                "anchor_mode": "current_target",
                "action_dim": 3,
                "d_long_range": [-1500.0, 1500.0],
                "d_lat_range": [-800.0, 800.0],
                "d_vert_range": [-500.0, 500.0],
                "smoothing_alpha": 0.3,
            },
            "limits": {
                "nz_min": -2.0,
                "nz_max": 7.0,
                "roll_rate_min": -1.5,
                "roll_rate_max": 1.5,
                "throttle_min": 0.0,
                "throttle_max": 1.0,
            },
            "guidance": {
                "mode": "los_rate",
                "gains": {
                    "k_los": 1.0,
                    "k_pos": 0.5,
                    "k_damp": 0.2,
                    "k_roll": 1.0,
                    "k_speed": 0.2,
                    "alpha_filter": 0.3,
                }
            },
        }
        env = CloseRangeTrackingEnv(config)
        scenario = ScenarioRegistry.get("smoke_crossing_left")
        obs = env.reset(scenario=scenario, seed=0)
        
        # Run a few steps with zero action (should not crash)
        for _ in range(20):
            obs, reward, terminated, truncated, info = env.step(np.zeros(3))
            if terminated or truncated:
                break
            assert np.isfinite(reward)
        
        env.close()
