"""
Tests for CloseRangeTrackingEnv in no-prediction mode.
"""

import pytest
import numpy as np
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv


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


class TestCloseRangeTrackingEnvNoPrediction:
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

    def test_step_info_contains_required_fields(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        action = np.zeros(3)
        obs, reward, terminated, truncated, info = env.step(action)

        required = [
            "virtual_point", "guidance_command", "reward_terms",
            "termination_info", "relative_state", "anchor_mode",
        ]
        for field in required:
            assert field in info, f"Missing info field: {field}"
        env.close()

    def test_step_anchor_mode_is_current_target(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(3))
        assert info["anchor_mode"] == "current_target"
        env.close()

    def test_step_no_predictor_adapter_called(self, base_config):
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        # predictor_adapter should be None when trajectory_prediction.enabled=false
        assert env.trajectory_predictor_adapter is None
        _, _, _, _, info = env.step(np.zeros(3))
        assert info["anchor_mode"] == "current_target"
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

    def test_termination_success(self, base_config):
        # Override to make success easy
        base_config["env"]["success_range_m"] = 5000.0
        base_config["env"]["success_ata_deg"] = 180.0
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        _, _, terminated, _, info = env.step(np.zeros(3))
        assert terminated is True
        assert info.get("is_success") is True
        env.close()

    def test_termination_crash(self, base_config):
        base_config["env"]["min_altitude_m"] = 6000.0
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        _, _, terminated, _, info = env.step(np.zeros(3))
        assert terminated is True
        assert info.get("is_crash") is True
        env.close()

    def test_termination_out_of_bounds(self, base_config):
        base_config["env"]["max_range_m"] = 100.0
        env = CloseRangeTrackingEnv(base_config)
        env.reset(seed=0)
        _, _, terminated, _, info = env.step(np.zeros(3))
        assert terminated is True
        assert info.get("is_out_of_bounds") is True
        env.close()
