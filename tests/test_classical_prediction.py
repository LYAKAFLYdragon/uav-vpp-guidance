"""
Tests for Classical CV/CA Prediction and P1 Hardening.

Covers:
- P1 fixes (train_log episode retention, ppo.yaml key naming, env.yaml max_range_m)
- ConstantVelocityPredictor
- ConstantAccelerationPredictor
- TrajectoryPredictorAdapter config creation
"""

import csv
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

from uav_vpp_guidance.utils.config import load_yaml_config
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.trajectory_prediction import (
    ConstantVelocityPredictor,
    ConstantAccelerationPredictor,
    TrajectoryPredictorAdapter,
    create_predictor_from_config,
    create_state_buffer_from_config,
)
from uav_vpp_guidance.trajectory_prediction.state_buffer import TrajectoryStateBuffer


# ---------------------------------------------------------------------------
# P1 Fix Tests
# ---------------------------------------------------------------------------

class TestP1Hardening:
    def test_ppo_yaml_key_naming(self):
        config = load_yaml_config("config/ppo.yaml")
        ppo = config.get("ppo", {})
        # New unified keys must exist
        assert "clip_coef" in ppo
        assert "rollout_steps" in ppo
        assert "update_epochs" in ppo
        assert "minibatch_size" in ppo
        assert "value_coef" in ppo
        # Old inconsistent keys must not exist
        assert "clip_range" not in ppo
        assert "n_steps" not in ppo
        assert "n_epochs" not in ppo

    def test_ppo_yaml_policy_key(self):
        config = load_yaml_config("config/ppo.yaml")
        assert "policy" in config
        assert "network" not in config

    def test_env_yaml_max_range_m(self):
        config = load_yaml_config("config/env.yaml")
        assert "max_range_m" in config.get("termination", {})

    def test_train_log_does_not_lose_episodes(self):
        """Smoke training must produce episode_train_log.csv with >=1 episode rows."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    sys.executable, "-m",
                    "uav_vpp_guidance.training.train_no_prediction_vpp_ppo",
                    "--config", "config/experiment/train_no_prediction_vpp_ppo.yaml",
                    "--smoke",
                    "--output-dir", tmpdir,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            assert result.returncode == 0, f"Smoke training failed: {result.stderr}"

            episode_log = os.path.join(tmpdir, "logs", "episode_train_log.csv")
            update_log = os.path.join(tmpdir, "logs", "update_train_log.csv")
            assert os.path.exists(episode_log), "episode_train_log.csv missing"
            assert os.path.exists(update_log), "update_train_log.csv missing"

            with open(episode_log, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) >= 1, "episode_train_log.csv should have at least 1 episode row"

            with open(update_log, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) >= 1, "update_train_log.csv should have at least 1 update row"


# ---------------------------------------------------------------------------
# ConstantVelocityPredictor Tests
# ---------------------------------------------------------------------------

class TestConstantVelocityPredictor:
    def test_predict_output_shape(self):
        pred = ConstantVelocityPredictor(lookahead_time_s=1.0)
        target_state = {
            "position_neu": np.array([1000.0, 0.0, 5000.0]),
            "velocity_ned": np.array([200.0, 0.0, 0.0]),
        }
        pos, var, info = pred.predict(current_target_state=target_state)
        assert pos.shape == (3,)
        assert var is None
        assert info["model"] == "constant_velocity"
        assert not info["fallback"]

    def test_predict_with_speed_heading(self):
        pred = ConstantVelocityPredictor(lookahead_time_s=2.0)
        target_state = {
            "position_m": np.array([0.0, 0.0, 5000.0]),
            "velocity_mps": 100.0,
            "heading_rad": 0.0,
        }
        pos, var, info = pred.predict(current_target_state=target_state)
        assert pos.shape == (3,)
        # Should move +200m in x direction
        assert pos[0] == pytest.approx(200.0, abs=1e-6)

    def test_predict_fallback_no_velocity(self):
        pred = ConstantVelocityPredictor(lookahead_time_s=1.0)
        target_state = {
            "position_neu": np.array([0.0, 0.0, 5000.0]),
        }
        pos, var, info = pred.predict(current_target_state=target_state)
        assert info["fallback"] is True
        assert pos is not None  # fallback returns current position

    def test_predict_uniform_motion(self):
        pred = ConstantVelocityPredictor(lookahead_time_s=1.0)
        target_state = {
            "position_neu": np.array([0.0, 0.0, 5000.0]),
            "velocity_ned": np.array([100.0, 50.0, 10.0]),
        }
        pos, _, _ = pred.predict(current_target_state=target_state)
        expected = np.array([100.0, 50.0, 5010.0])
        np.testing.assert_allclose(pos, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# ConstantAccelerationPredictor Tests
# ---------------------------------------------------------------------------

class TestConstantAccelerationPredictor:
    def test_predict_output_shape(self):
        pred = ConstantAccelerationPredictor(lookahead_time_s=1.0)
        target_state = {
            "position_neu": np.array([0.0, 0.0, 5000.0]),
            "velocity_ned": np.array([100.0, 0.0, 0.0]),
        }
        history = np.random.randn(5, 16).astype(np.float64)
        # Make velocity features consistent for last 3 frames to avoid NaN
        for i in range(3):
            history[-(i+1), 3:6] = np.array([0.333, 0.0, 0.0])  # 100/300
        pos, var, info = pred.predict(history_seq=history, current_target_state=target_state)
        assert pos.shape == (3,)
        assert info["model"] == "constant_acceleration"

    def test_fallback_to_cv_when_insufficient_history(self):
        pred = ConstantAccelerationPredictor(lookahead_time_s=1.0)
        target_state = {
            "position_neu": np.array([0.0, 0.0, 5000.0]),
            "velocity_ned": np.array([100.0, 0.0, 0.0]),
        }
        history = np.random.randn(2, 16).astype(np.float64)
        pos, var, info = pred.predict(history_seq=history, current_target_state=target_state)
        assert info["fallback"] is True
        assert "cv_fallback" in info
        # Should match CV prediction
        cv_pred = ConstantVelocityPredictor(lookahead_time_s=1.0)
        cv_pos, _, _ = cv_pred.predict(current_target_state=target_state)
        np.testing.assert_allclose(pos, cv_pos, atol=1e-6)

    def test_fallback_to_current_when_no_velocity(self):
        pred = ConstantAccelerationPredictor(lookahead_time_s=1.0)
        target_state = {
            "position_neu": np.array([0.0, 0.0, 5000.0]),
        }
        history = np.random.randn(5, 16).astype(np.float64)
        pos, var, info = pred.predict(history_seq=history, current_target_state=target_state)
        assert info["fallback"] is True
        np.testing.assert_allclose(pos, np.array([0.0, 0.0, 5000.0]), atol=1e-6)

    def test_uniform_acceleration(self):
        pred = ConstantAccelerationPredictor(lookahead_time_s=2.0)
        target_state = {
            "position_neu": np.array([0.0, 0.0, 5000.0]),
            "velocity_ned": np.array([100.0, 0.0, 0.0]),
        }
        # Construct history with constant acceleration: a = [10, 0, 0]
        # v_t = [100, 0, 0], v_{t-1} = [90, 0, 0], v_{t-2} = [80, 0, 0]
        # Feature dim 3:5 is velocity / 300
        history = np.zeros((3, 16), dtype=np.float64)
        history[0, 3:6] = np.array([80.0/300, 0.0, 0.0])
        history[1, 3:6] = np.array([90.0/300, 0.0, 0.0])
        history[2, 3:6] = np.array([100.0/300, 0.0, 0.0])

        pos, _, info = pred.predict(history_seq=history, current_target_state=target_state)
        # p = p0 + v0*T + 0.5*a*T^2 = 0 + 100*2 + 0.5*50*4 = 200 + 100 = 300
        assert not info["fallback"]
        assert pos[0] == pytest.approx(300.0, abs=5.0)


# ---------------------------------------------------------------------------
# TrajectoryPredictorAdapter Config Tests
# ---------------------------------------------------------------------------

class TestPredictorAdapterConfig:
    def test_create_cv_from_config(self):
        config = {
            "predictor_type": "constant_velocity",
            "prediction": {"lookahead_time_s": 1.5},
        }
        predictor = create_predictor_from_config(config)
        assert isinstance(predictor, ConstantVelocityPredictor)
        assert predictor.lookahead_time_s == 1.5

    def test_create_ca_from_config(self):
        config = {
            "predictor_type": "constant_acceleration",
            "prediction": {"lookahead_time_s": 2.0},
        }
        predictor = create_predictor_from_config(config)
        assert isinstance(predictor, ConstantAccelerationPredictor)
        assert predictor.lookahead_time_s == 2.0

    def test_create_buffer_from_config(self):
        config = {
            "history": {"history_len": 8, "padding_mode": "zero"},
            "model": {"input_dim": 16},
        }
        buf = create_state_buffer_from_config(config)
        assert isinstance(buf, TrajectoryStateBuffer)
        assert buf.history_len == 8

    def test_adapter_predicts_with_cv(self):
        config = {
            "predictor_type": "constant_velocity",
            "prediction": {"lookahead_time_s": 1.0, "output_mode": "absolute_position"},
            "integration": {"anchor_mode": "predicted_target"},
            "history": {"history_len": 5, "padding_mode": "repeat_first"},
            "model": {"input_dim": 16},
            "normalization": {"position_scale_m": 1000.0, "velocity_scale_mps": 300.0, "overload_scale": 9.0},
        }
        predictor = create_predictor_from_config(config)
        buf = create_state_buffer_from_config(config)
        adapter = TrajectoryPredictorAdapter(predictor, buf, config)

        target_state = {
            "position_neu": np.array([1000.0, 0.0, 5000.0]),
            "velocity_ned": np.array([200.0, 0.0, 0.0]),
        }
        pred_pos, _, info = adapter.predict(target_state)
        assert pred_pos.shape == (3,)
        assert not info["fallback"]
        assert pred_pos[0] == pytest.approx(1200.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Environment Integration Tests
# ---------------------------------------------------------------------------

class TestPredictionEnvIntegration:
    def test_cv_predicted_target_anchor(self):
        config = {
            "experiment": {"name": "test_cv", "seed": 42, "output_root": "outputs"},
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
                "anchor_mode": "predicted_target",
                "action_dim": 3,
                "d_long_range": [-1500.0, 1500.0],
                "d_lat_range": [-800.0, 800.0],
                "d_vert_range": [-500.0, 500.0],
                "smoothing_alpha": 0.3,
            },
            "trajectory_prediction": {
                "enabled": True,
                "predictor_type": "constant_velocity",
                "freeze_predictor_during_rl": True,
                "prediction": {"lookahead_time_s": 1.0, "output_mode": "absolute_position"},
                "history": {"history_len": 5, "padding_mode": "repeat_first"},
                "model": {"input_dim": 16},
                "normalization": {"position_scale_m": 1000.0, "velocity_scale_mps": 300.0, "overload_scale": 9.0},
            },
            "limits": {
                "nz_min": -2.0, "nz_max": 7.0,
                "roll_rate_min": -1.5, "roll_rate_max": 1.5,
                "throttle_min": 0.0, "throttle_max": 1.0,
            },
            "reward": {
                "w_range": 0.5, "w_angle": 0.8, "w_energy": 0.2,
                "w_safety": 2.0, "w_saturation": 1.0, "w_smooth": 0.1,
                "terminal_success": 200.0, "terminal_failure": -200.0, "terminal_crash": -300.0,
                "min_altitude_m": 500.0,
            },
            "guidance": {
                "mode": "los_rate", "use_gain_adapter": False,
                "gains": {"k_los": 1.0, "k_pos": 0.5, "k_damp": 0.2, "k_roll": 1.0, "k_speed": 0.2, "alpha_filter": 0.3},
            },
        }
        env = CloseRangeTrackingEnv(config)
        assert env.trajectory_predictor_adapter is not None
        obs = env.reset(seed=0)
        action = np.zeros(3)
        obs, reward, terminated, truncated, info = env.step(action)
        assert info["prediction_enabled"] is True
        assert info["predictor_type"] == "ConstantVelocityPredictor"
        assert info["prediction_valid"] is True
        env.close()

    def test_ca_predicted_target_anchor(self):
        config = {
            "experiment": {"name": "test_ca", "seed": 42, "output_root": "outputs"},
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
                "anchor_mode": "predicted_target",
                "action_dim": 3,
                "d_long_range": [-1500.0, 1500.0],
                "d_lat_range": [-800.0, 800.0],
                "d_vert_range": [-500.0, 500.0],
                "smoothing_alpha": 0.3,
            },
            "trajectory_prediction": {
                "enabled": True,
                "predictor_type": "constant_acceleration",
                "freeze_predictor_during_rl": True,
                "prediction": {"lookahead_time_s": 1.0, "output_mode": "absolute_position"},
                "history": {"history_len": 5, "padding_mode": "repeat_first"},
                "model": {"input_dim": 16},
                "normalization": {"position_scale_m": 1000.0, "velocity_scale_mps": 300.0, "overload_scale": 9.0},
            },
            "limits": {
                "nz_min": -2.0, "nz_max": 7.0,
                "roll_rate_min": -1.5, "roll_rate_max": 1.5,
                "throttle_min": 0.0, "throttle_max": 1.0,
            },
            "reward": {
                "w_range": 0.5, "w_angle": 0.8, "w_energy": 0.2,
                "w_safety": 2.0, "w_saturation": 1.0, "w_smooth": 0.1,
                "terminal_success": 200.0, "terminal_failure": -200.0, "terminal_crash": -300.0,
                "min_altitude_m": 500.0,
            },
            "guidance": {
                "mode": "los_rate", "use_gain_adapter": False,
                "gains": {"k_los": 1.0, "k_pos": 0.5, "k_damp": 0.2, "k_roll": 1.0, "k_speed": 0.2, "alpha_filter": 0.3},
            },
        }
        env = CloseRangeTrackingEnv(config)
        assert env.trajectory_predictor_adapter is not None
        obs = env.reset(seed=0)
        action = np.zeros(3)
        obs, reward, terminated, truncated, info = env.step(action)
        assert info["prediction_enabled"] is True
        assert info["predictor_type"] == "ConstantAccelerationPredictor"
        env.close()

    def test_prediction_fallback_when_disabled(self):
        config = {
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
                "nz_min": -2.0, "nz_max": 7.0,
                "roll_rate_min": -1.5, "roll_rate_max": 1.5,
                "throttle_min": 0.0, "throttle_max": 1.0,
            },
            "reward": {
                "w_range": 0.5, "w_angle": 0.8, "w_energy": 0.2,
                "w_safety": 2.0, "w_saturation": 1.0, "w_smooth": 0.1,
                "terminal_success": 200.0, "terminal_failure": -200.0, "terminal_crash": -300.0,
                "min_altitude_m": 500.0,
            },
            "guidance": {
                "mode": "los_rate", "use_gain_adapter": False,
                "gains": {"k_los": 1.0, "k_pos": 0.5, "k_damp": 0.2, "k_roll": 1.0, "k_speed": 0.2, "alpha_filter": 0.3},
            },
        }
        env = CloseRangeTrackingEnv(config)
        assert env.trajectory_predictor_adapter is None
        obs = env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(3))
        assert info["prediction_enabled"] is False
        assert info["predictor_type"] is None
        env.close()

    def test_action_low_high_is_normalized(self):
        """Ensure policy action remains in [-1, 1] after P0 fix."""
        config = load_yaml_config("config/experiment/train_no_prediction_vpp_ppo.yaml")
        policy = config.get("policy", {})
        low = policy.get("action_low", [])
        high = policy.get("action_high", [])
        assert all(v == -1.0 for v in low)
        assert all(v == 1.0 for v in high)
