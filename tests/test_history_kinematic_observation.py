"""
Tests for the history-kinematic observation extension (thesis Group H ablation).

Group H = 16-D base + 3-D target history-motion features, no future prediction.
These tests pin down the observation contract added for the blind-review
revision: feature keys/order, schema flag, window semantics, and the rule that
opponent observations must NOT include the extra dims.
"""

import copy

import numpy as np
import pytest

from uav_vpp_guidance.envs.observation import build_observation
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv


@pytest.fixture
def base_config():
    return {
        "experiment": {"name": "test_hist_kin", "seed": 42, "output_root": "outputs"},
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
            },
        },
    }


def _hist_config(base_config, **overrides):
    cfg = copy.deepcopy(base_config)
    cfg["observation"] = {"include_history_kinematic": True}
    cfg["observation"]["history_kinematic"] = overrides
    return cfg


class TestBuildObservationHistoryKinematic:
    def test_appends_three_dims_in_contract_order(self):
        own = {
            "position_neu": [0.0, 0.0, 1000.0],
            "velocity_vector_mps": [100.0, 0.0, 0.0],
            "attitude_rpy": [0.0, 0.0, 0.0],
        }
        target = {
            "position_neu": [1000.0, 0.0, 1000.0],
            "velocity_vector_mps": [100.0, 0.0, 0.0],
            "attitude_rpy": [0.0, 0.0, 0.0],
        }
        base_vec, base_names = build_observation(
            own, target, return_feature_names=True
        )
        assert base_vec.shape[0] == 16

        hist = {
            "hist_heading_rate": 0.1,
            "hist_speed_change": 0.2,
            "hist_los_rate": 0.3,
        }
        vec, names = build_observation(
            own, target, history_kinematic=hist, return_feature_names=True
        )
        assert vec.shape[0] == 19
        assert names[:16] == base_names
        assert names[16:] == [
            "hist_heading_rate",
            "hist_speed_change",
            "hist_los_rate",
        ]
        np.testing.assert_allclose(vec[16:], [0.1, 0.2, 0.3], atol=1e-6)

    def test_saturation_still_follows_history(self):
        own = {
            "position_neu": [0.0, 0.0, 1000.0],
            "velocity_vector_mps": [100.0, 0.0, 0.0],
        }
        target = {
            "position_neu": [1000.0, 0.0, 1000.0],
            "velocity_vector_mps": [100.0, 0.0, 0.0],
        }
        _, names = build_observation(
            own,
            target,
            history_kinematic={"hist_heading_rate": 0.0},
            saturation_features={
                "nz_saturated": 0.0,
                "roll_rate_saturated": 0.0,
                "throttle_saturated": 0.0,
            },
            return_feature_names=True,
        )
        i_hist = names.index("hist_heading_rate")
        i_sat = names.index("nz_saturated")
        assert i_hist < i_sat


class TestEnvHistoryKinematic:
    def test_schema_flag_and_dim(self, base_config):
        env = CloseRangeTrackingEnv(_hist_config(base_config))
        obs = env.reset(seed=0)
        schema = obs["observation_schema"]
        assert schema["include_history_kinematic"] is True
        assert schema["dim"] == 19
        for key in (
            "hist_heading_rate",
            "hist_speed_change",
            "hist_los_rate",
        ):
            assert key in schema["feature_names"]

    def test_disabled_by_default(self, base_config):
        env = CloseRangeTrackingEnv(copy.deepcopy(base_config))
        obs = env.reset(seed=0)
        schema = obs["observation_schema"]
        assert schema["include_history_kinematic"] is False
        assert schema["dim"] == 16

    def test_feature_semantics_synthetic_window(self, base_config):
        env = CloseRangeTrackingEnv(
            _hist_config(base_config, window_len=12, recent_len=3)
        )
        env.reset(seed=0)
        dt = float(env.env_config.get("high_level_dt", 0.2))
        n = 12
        t = np.arange(n) * dt
        # 目标航向以 0.5 rad/s 匀速转弯；LOS 以 0.3 rad/s 转动；
        # 速度从 100 线性增至 110 m/s。
        frames = [
            {
                "target_heading_rad": 0.5 * ti,
                "target_speed": 100.0 + 10.0 * (ti / t[-1]),
                "los_azimuth_rad": 0.3 * ti,
            }
            for ti in t
        ]
        env._history_kinematic_buffer.clear()
        for frame in frames:
            env._history_kinematic_buffer.append(frame)

        feats = env._build_history_kinematic_features()
        assert feats is not None
        assert abs(feats["hist_heading_rate"] - 0.5 / np.pi) < 1e-3
        assert abs(feats["hist_los_rate"] - 0.3 / np.pi) < 1e-3
        assert abs(feats["hist_speed_change"] - 0.1) < 1e-6

    def test_window_advances_once_per_action(self, base_config):
        env = CloseRangeTrackingEnv(
            _hist_config(base_config, window_len=5, recent_len=2)
        )
        env.reset(seed=0)
        assert len(env._history_kinematic_buffer) == 1
        for _ in range(4):
            env.step(np.zeros(3))
        assert len(env._history_kinematic_buffer) == 5
        env.step(np.zeros(3))
        # window 满后保持定长（deque maxlen）
        assert len(env._history_kinematic_buffer) == 5

    def test_reset_clears_window(self, base_config):
        env = CloseRangeTrackingEnv(_hist_config(base_config))
        env.reset(seed=0)
        for _ in range(3):
            env.step(np.zeros(3))
        assert len(env._history_kinematic_buffer) > 1
        env.reset(seed=1)
        assert len(env._history_kinematic_buffer) == 1

    def test_opponent_obs_excludes_history_dims(self, base_config):
        env = CloseRangeTrackingEnv(_hist_config(base_config))
        env.reset(seed=0)
        ego_dim = env._get_observation()["observation_schema"]["dim"]
        opp = env._get_opponent_obs()
        assert ego_dim == 19
        assert opp["observation_schema"]["dim"] == 16
        assert "hist_heading_rate" not in opp["observation_schema"]["feature_names"]

    def test_window_len_defaults_to_predictor_history_len(self, base_config):
        cfg = copy.deepcopy(base_config)
        cfg["trajectory_prediction"] = {
            "enabled": False,
            "history": {"history_len": 22},
        }
        cfg["observation"] = {"include_history_kinematic": True}
        env = CloseRangeTrackingEnv(cfg)
        assert env._history_kinematic_window_len == 22
        assert env._history_kinematic_buffer.maxlen == 22
        assert env._history_kinematic_recent_len == max(2, 22 // 4)
