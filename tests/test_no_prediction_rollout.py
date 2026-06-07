"""
Tests for no-prediction smoke rollout.
"""

import pytest
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.training.train_no_prediction_vpp import smoke_rollout
from uav_vpp_guidance.evaluation.evaluate_no_prediction import evaluate
from uav_vpp_guidance.baselines.rule_based_pursuit import RuleBasedPursuitPolicy


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
            }
        },
    }


class TestSmokeRollout:
    def test_smoke_rollout_completes(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        summary = smoke_rollout(env, num_steps=50, seed=0)
        assert isinstance(summary, dict)
        assert "total_reward" in summary
        assert "final_range_m" in summary
        assert "final_ata_deg" in summary
        assert "done" in summary
        assert "reason" in summary
        env.close()

    def test_smoke_rollout_100_steps_no_crash(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        summary = smoke_rollout(env, num_steps=100, seed=0)
        # Should complete all 100 steps or terminate for non-crash reasons
        if summary["done"]:
            assert summary["reason"] != "crash"
        env.close()


class TestEvaluate:
    def test_evaluate_random_policy(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        metrics = evaluate(env, num_episodes=3, policy=None, seed=0)
        assert isinstance(metrics, dict)
        assert "success_rate" in metrics
        assert "crash_rate" in metrics
        assert "timeout_rate" in metrics
        assert "out_of_bounds_rate" in metrics
        assert "avg_return" in metrics
        assert "avg_episode_length" in metrics
        assert "avg_min_range_m" in metrics
        assert "avg_min_ata_deg" in metrics
        assert metrics["num_episodes"] == 3
        env.close()

    def test_evaluate_rule_policy(self, base_config):
        policy = RuleBasedPursuitPolicy(mode="pure_pursuit")
        env = CloseRangeTrackingEnv(base_config)
        metrics = evaluate(env, num_episodes=3, policy=policy, seed=0)
        assert isinstance(metrics, dict)
        assert 0.0 <= metrics["success_rate"] <= 1.0
        env.close()
