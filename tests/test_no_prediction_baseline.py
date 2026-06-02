"""
Tests for the No-Prediction VPP Baseline.

Covers:
- CloseRangeTrackingEnv full closed loop with SimplePointMassEnv
- RuleBasedPursuitPolicy modes
- Smoke rollout from train_no_prediction_vpp
- Evaluation from evaluate_no_prediction
"""

import numpy as np
import pytest
import os
import sys

# Ensure project src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.baselines.rule_based_pursuit import RuleBasedPursuitPolicy
from uav_vpp_guidance.training.train_no_prediction_vpp import smoke_rollout
from uav_vpp_guidance.evaluation.evaluate_no_prediction import evaluate


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
            "target_mode": "constant_velocity",
        },
        "virtual_point": {
            "anchor_mode": "current_target",
            "max_offset_m": 1500.0,
            "clip_action": True,
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


class TestCloseRangeTrackingEnv:
    def test_reset_returns_observation(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        obs = env.reset(seed=0)
        assert isinstance(obs, dict)
        assert "relative_state" in obs
        assert "own_state" in obs
        assert "target_state" in obs
        assert "observation_vector" in obs
        env.close()

    def test_step_with_random_action(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        obs = env.reset(seed=0)
        action = np.array([0.1, -0.2, 0.05])
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        env.close()

    def test_step_with_none_action(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        obs = env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(None)
        assert isinstance(reward, float)
        env.close()

    def test_multiple_steps_no_crash(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        obs = env.reset(seed=0)
        for _ in range(20):
            action = np.random.uniform(-1, 1, size=3)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        env.close()

    def test_observation_vector_shape(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        obs = env.reset(seed=0)
        vec = obs["observation_vector"]
        assert isinstance(vec, np.ndarray)
        assert vec.ndim == 1
        assert len(vec) > 0
        env.close()


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

    def test_policy_in_env_loop(self, base_config):
        policy = RuleBasedPursuitPolicy(mode="pure_pursuit")
        env = CloseRangeTrackingEnv(base_config)
        obs = env.reset(seed=0)
        for _ in range(20):
            rel_state = obs.get("relative_state", {})
            own_state = obs.get("own_state", {})
            target_state = obs.get("target_state", {})
            action = policy.get_action(own_state, target_state, rel_state)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        env.close()


class TestSmokeRollout:
    def test_smoke_rollout_completes(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        summary = smoke_rollout(env, num_steps=50, seed=0)
        assert isinstance(summary, dict)
        assert "num_steps" in summary
        assert "total_reward" in summary
        assert "min_range_m" in summary
        assert summary["num_steps"] > 0
        env.close()

    def test_smoke_rollout_no_crash(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        summary = smoke_rollout(env, num_steps=30, seed=0)
        assert summary.get("completed_without_crash", True)
        env.close()


class TestEvaluate:
    def test_evaluate_random_policy(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        metrics = evaluate(env, num_episodes=3, policy=None, seed=0)
        assert isinstance(metrics, dict)
        assert "success_rate" in metrics
        assert "crash_rate" in metrics
        assert "timeout_rate" in metrics
        assert "avg_return" in metrics
        assert metrics["num_episodes"] == 3
        env.close()

    def test_evaluate_rule_policy(self, base_config):
        policy = RuleBasedPursuitPolicy(mode="pure_pursuit")
        env = CloseRangeTrackingEnv(base_config)
        metrics = evaluate(env, num_episodes=3, policy=policy, seed=0)
        assert isinstance(metrics, dict)
        assert 0.0 <= metrics["success_rate"] <= 1.0
        env.close()
