"""
轨迹预测模型单元测试。
"""

import os

import pytest
import numpy as np
import torch

from uav_vpp_guidance.trajectory_prediction.constant_velocity import ConstantVelocityPredictor
from uav_vpp_guidance.trajectory_prediction.lstm_predictor import LSTMTrajectoryPredictor
from uav_vpp_guidance.trajectory_prediction.gru_predictor import GRUTrajectoryPredictor


class TestConstantVelocityPredictor:
    def test_predict_with_velocity_vector(self):
        predictor = ConstantVelocityPredictor(lookahead_time_s=2.0)
        state = {
            "position_neu": np.array([100.0, 200.0, 300.0]),
            "velocity_ned": np.array([10.0, 20.0, 5.0]),
        }
        pred_pos, pred_var, info = predictor.predict(current_target_state=state)
        expected = np.array([120.0, 240.0, 290.0])
        assert np.allclose(pred_pos, expected)
        assert pred_var is None
        assert not info["fallback"]

    def test_predict_with_speed_heading(self):
        predictor = ConstantVelocityPredictor(lookahead_time_s=1.0)
        state = {
            "position_neu": np.array([0.0, 0.0, 1000.0]),
            "velocity_mps": 100.0,
            "heading_rad": np.pi / 2,  # 90 deg, north -> east in NEU? Actually heading is yaw
        }
        pred_pos, pred_var, info = predictor.predict(current_target_state=state)
        # velocity = [speed*cos(heading), speed*sin(heading), 0]
        # NEU frame: north=x, east=y
        expected = np.array([0.0, 100.0, 1000.0])
        assert np.allclose(pred_pos, expected, atol=1e-6)

    def test_predict_fallback_no_state(self):
        predictor = ConstantVelocityPredictor(lookahead_time_s=1.0)
        pred_pos, pred_var, info = predictor.predict(current_target_state=None)
        assert pred_pos is None
        assert info["fallback"]


class TestLSTMTrajectoryPredictor:
    def test_output_shape(self):
        model = LSTMTrajectoryPredictor(input_dim=16, hidden_dim=32, num_layers=2)
        history_seq = torch.randn(4, 10, 16)  # [batch, history_len, input_dim]
        output = model(history_seq)
        assert output.shape == (4, 3)

    def test_predict_numpy_input(self):
        model = LSTMTrajectoryPredictor(input_dim=16, hidden_dim=32, num_layers=2)
        history_seq = np.random.randn(10, 16).astype(np.float32)
        pred_disp, pred_var, info = model.predict(history_seq)
        assert pred_disp.shape == (3,)
        assert pred_var is None
        assert info["model"] == "lstm"

    def test_predict_batch_input(self):
        model = LSTMTrajectoryPredictor(input_dim=16, hidden_dim=32, num_layers=2)
        history_seq = np.random.randn(2, 10, 16).astype(np.float32)
        pred_disp, pred_var, info = model.predict(history_seq)
        assert pred_disp.shape == (2, 3)

    def test_freeze_unfreeze(self):
        model = LSTMTrajectoryPredictor(input_dim=16, hidden_dim=32, num_layers=2)
        model.freeze()
        for param in model.parameters():
            assert not param.requires_grad
        model.unfreeze()
        for param in model.parameters():
            assert param.requires_grad

    def test_variance_output_shape(self):
        model = LSTMTrajectoryPredictor(input_dim=8, hidden_dim=16, predict_variance=True)
        history_seq = torch.randn(2, 5, 8)
        output = model(history_seq)
        assert output.shape == (2, 6)


class TestGRUTrajectoryPredictor:
    def test_output_shape(self):
        model = GRUTrajectoryPredictor(input_dim=16, hidden_dim=32, num_layers=2)
        history_seq = torch.randn(4, 10, 16)
        output = model(history_seq)
        assert output.shape == (4, 3)

    def test_predict_numpy_input(self):
        model = GRUTrajectoryPredictor(input_dim=16, hidden_dim=32, num_layers=2)
        history_seq = np.random.randn(10, 16).astype(np.float32)
        pred_disp, pred_var, info = model.predict(history_seq)
        assert pred_disp.shape == (3,)
        assert pred_var is None
        assert info["model"] == "gru"

    def test_freeze_unfreeze(self):
        model = GRUTrajectoryPredictor(input_dim=16, hidden_dim=32, num_layers=2)
        model.freeze()
        for param in model.parameters():
            assert not param.requires_grad
        model.unfreeze()
        for param in model.parameters():
            assert param.requires_grad


# ---------------------------------------------------------------------------
# Coordinate utils tests
# ---------------------------------------------------------------------------

class TestCoordinateUtils:
    def test_get_velocity_zero_speed_with_heading(self):
        """velocity_mps=0.0 + heading_rad should return [0,0,0], not raise."""
        from uav_vpp_guidance.trajectory_prediction.coordinate_utils import get_velocity_neu
        state = {"velocity_mps": 0.0, "heading_rad": 0.0}
        vel = get_velocity_neu(state)
        assert np.allclose(vel, [0.0, 0.0, 0.0])

    def test_get_velocity_speed_mps_zero(self):
        """speed_mps=0.0 should not be skipped by 'or' chain."""
        from uav_vpp_guidance.trajectory_prediction.coordinate_utils import get_velocity_neu
        state = {"speed_mps": 0.0, "heading_rad": np.pi / 4}
        vel = get_velocity_neu(state)
        assert np.allclose(vel, [0.0, 0.0, 0.0])

    def test_get_acceleration_ned_conversion(self):
        """acceleration_ned=[0,0,9.8] -> NEU [0,0,-9.8]."""
        from uav_vpp_guidance.trajectory_prediction.coordinate_utils import get_acceleration_neu
        state = {"acceleration_ned": np.array([0.0, 0.0, 9.8])}
        acc = get_acceleration_neu(state)
        assert np.allclose(acc, [0.0, 0.0, -9.8])

    def test_get_acceleration_bad_shape_raises(self):
        """Wrong-shape acceleration field should raise ValueError."""
        from uav_vpp_guidance.trajectory_prediction.coordinate_utils import get_acceleration_neu
        state = {"acceleration_vector_mps2": np.array([1.0, 2.0])}
        with pytest.raises(ValueError):
            get_acceleration_neu(state)

    def test_get_acceleration_ned_bad_shape_raises(self):
        state = {"acceleration_ned": np.array([1.0])}
        from uav_vpp_guidance.trajectory_prediction.coordinate_utils import get_acceleration_neu
        with pytest.raises(ValueError):
            get_acceleration_neu(state)

    def test_get_position_only_position_field(self):
        """Only 'position' field should work for neural displacement anchor."""
        from uav_vpp_guidance.trajectory_prediction.coordinate_utils import get_position_neu
        state = {"position": np.array([100.0, 200.0, 300.0])}
        pos = get_position_neu(state)
        assert np.allclose(pos, [100.0, 200.0, 300.0])


# ---------------------------------------------------------------------------
# Device utils tests
# ---------------------------------------------------------------------------

class TestDeviceUtils:
    def test_resolve_cpu(self):
        from uav_vpp_guidance.trajectory_prediction.device_utils import resolve_torch_device
        dev = resolve_torch_device("cpu")
        assert str(dev) == "cpu"

    def test_cuda_unavailable_fallback(self):
        import torch
        from uav_vpp_guidance.trajectory_prediction.device_utils import resolve_torch_device
        if torch.cuda.is_available():
            pytest.skip("CUDA is available on this machine")
        dev = resolve_torch_device("cuda", allow_fallback=True)
        assert str(dev) == "cpu"

    def test_cuda_unavailable_strict_raises(self):
        import torch
        from uav_vpp_guidance.trajectory_prediction.device_utils import resolve_torch_device
        if torch.cuda.is_available():
            pytest.skip("CUDA is available on this machine")
        with pytest.raises(RuntimeError):
            resolve_torch_device("cuda", allow_fallback=False)

    def test_load_checkpoint_cpu(self, tmpdir):
        import torch
        from uav_vpp_guidance.trajectory_prediction.device_utils import load_checkpoint_to_model
        from uav_vpp_guidance.trajectory_prediction.lstm_predictor import LSTMTrajectoryPredictor

        model = LSTMTrajectoryPredictor(input_dim=16, hidden_dim=32, num_layers=1, dropout=0.0)
        ckpt = os.path.join(str(tmpdir), "ckpt.pt")
        torch.save(model.state_dict(), ckpt)

        model2 = LSTMTrajectoryPredictor(input_dim=16, hidden_dim=32, num_layers=1, dropout=0.0)
        load_checkpoint_to_model(model2, ckpt, device_str="cpu", allow_device_fallback=True, strict=True)
        assert str(next(model2.parameters()).device) == "cpu"

    def test_load_checkpoint_missing_strict(self, tmpdir):
        from uav_vpp_guidance.trajectory_prediction.device_utils import load_checkpoint_to_model
        from uav_vpp_guidance.trajectory_prediction.lstm_predictor import LSTMTrajectoryPredictor

        model = LSTMTrajectoryPredictor(input_dim=16, hidden_dim=32, num_layers=1, dropout=0.0)
        bad_path = os.path.join(str(tmpdir), "nonexistent.pt")
        with pytest.raises(FileNotFoundError):
            load_checkpoint_to_model(model, bad_path, device_str="cpu", strict=True)


# ---------------------------------------------------------------------------
# Fallback mode tests
# ---------------------------------------------------------------------------

class TestFallbackModes:
    def test_fallback_constant_velocity(self):
        from uav_vpp_guidance.trajectory_prediction.predictor_adapter import _create_fallback_predictor
        fb = _create_fallback_predictor("constant_velocity", 1.0)
        assert fb is not None

    def test_fallback_constant_acceleration(self):
        from uav_vpp_guidance.trajectory_prediction.predictor_adapter import _create_fallback_predictor
        fb = _create_fallback_predictor("constant_acceleration", 1.0)
        assert fb is not None

    def test_fallback_current_target(self):
        from uav_vpp_guidance.trajectory_prediction.predictor_adapter import _create_fallback_predictor
        fb = _create_fallback_predictor("current_target", 1.0)
        assert fb is not None
        pos, _, info = fb.predict(current_target_state={"position_neu": np.array([1.0, 2.0, 3.0])})
        assert np.allclose(pos, [1.0, 2.0, 3.0])
        assert info.get("model") == "current_target"

    def test_fallback_none(self):
        from uav_vpp_guidance.trajectory_prediction.predictor_adapter import _create_fallback_predictor
        fb = _create_fallback_predictor("none", 1.0)
        assert fb is None

    def test_adapter_fallback_none_raises(self):
        from uav_vpp_guidance.trajectory_prediction.predictor_adapter import TrajectoryPredictorAdapter
        from uav_vpp_guidance.trajectory_prediction.lstm_predictor import LSTMTrajectoryPredictor
        from uav_vpp_guidance.trajectory_prediction.state_buffer import TrajectoryStateBuffer

        predictor = LSTMTrajectoryPredictor(input_dim=16, hidden_dim=32, num_layers=1, dropout=0.0)
        buffer = TrajectoryStateBuffer(history_len=5, feature_dim=16)
        config = {
            "prediction": {"lookahead_time_s": 1.0, "output_mode": "relative_displacement", "fallback_mode": "none"},
            "integration": {"anchor_mode": "predicted_target"},
            "normalization": {},
        }
        adapter = TrajectoryPredictorAdapter(predictor, buffer, config)
        with pytest.raises(RuntimeError):
            adapter.predict(current_target_state={"position_neu": np.zeros(3)})


# ---------------------------------------------------------------------------
# Checkpoint strict key resolution tests
# ---------------------------------------------------------------------------

class TestCheckpointStrictResolution:
    def test_canonical_false(self):
        from uav_vpp_guidance.trajectory_prediction.predictor_adapter import _resolve_checkpoint_strict
        assert _resolve_checkpoint_strict({"checkpoint_strict": False}) is False

    def test_alias_still_works(self):
        from uav_vpp_guidance.trajectory_prediction.predictor_adapter import _resolve_checkpoint_strict
        assert _resolve_checkpoint_strict({"strict_checkpoint": False}) is False

    def test_conflict_raises(self):
        from uav_vpp_guidance.trajectory_prediction.predictor_adapter import _resolve_checkpoint_strict
        with pytest.raises(ValueError):
            _resolve_checkpoint_strict({"checkpoint_strict": True, "strict_checkpoint": False})

    def test_default_true(self):
        from uav_vpp_guidance.trajectory_prediction.predictor_adapter import _resolve_checkpoint_strict
        assert _resolve_checkpoint_strict({}) is True


# ---------------------------------------------------------------------------
# Prediction error tracker tests
# ---------------------------------------------------------------------------

class TestPredictionErrorTracker:
    def test_cv_predictor_delayed_error_near_zero(self):
        from uav_vpp_guidance.trajectory_prediction.prediction_error_tracker import PredictionErrorTracker
        tracker = PredictionErrorTracker(high_level_dt=0.2)

        # Constant velocity target moving at [10, 0, 0] m/s
        # At t=0, predict position at t=1.0 (lookahead=1.0)
        tracker.register_prediction(
            current_time_s=0.0,
            lookahead_time_s=1.0,
            predicted_position_neu=np.array([10.0, 0.0, 0.0]),
        )
        # Update at t=1.0, actual position is [10, 0, 0]
        tracker.update(
            current_time_s=1.0,
            actual_target_position_neu=np.array([10.0, 0.0, 0.0]),
        )
        assert tracker.error_count == 1
        assert tracker.latest_error == pytest.approx(0.0, abs=1e-9)
        assert tracker.mean_error == pytest.approx(0.0, abs=1e-9)

    def test_multiple_pending_predictions(self):
        from uav_vpp_guidance.trajectory_prediction.prediction_error_tracker import PredictionErrorTracker
        tracker = PredictionErrorTracker(high_level_dt=0.2)

        tracker.register_prediction(0.0, 1.0, np.array([10.0, 0.0, 0.0]))
        tracker.register_prediction(0.2, 1.0, np.array([12.0, 0.0, 0.0]))

        tracker.update(1.0, np.array([10.0, 0.0, 0.0]))
        assert tracker.error_count == 1
        assert tracker.pending_count == 1

        tracker.update(1.2, np.array([12.0, 0.0, 0.0]))
        assert tracker.error_count == 2
        assert tracker.pending_count == 0

    def test_stats_dict(self):
        from uav_vpp_guidance.trajectory_prediction.prediction_error_tracker import PredictionErrorTracker
        tracker = PredictionErrorTracker(high_level_dt=0.2)
        stats = tracker.get_stats()
        assert stats["latest_prediction_error_m"] is None
        assert stats["mean_prediction_error_m"] is None
        assert stats["prediction_error_count"] == 0


# ---------------------------------------------------------------------------
# Config validator tests
# ---------------------------------------------------------------------------

class TestConfigValidator:
    def test_valid_cv_config(self):
        from uav_vpp_guidance.trajectory_prediction.config_validator import validate_tp_config
        config = {
            "enabled": True,
            "predictor_type": "constant_velocity",
            "prediction": {"fallback_mode": "constant_velocity"},
            "integration": {"anchor_mode": "predicted_target"},
        }
        assert validate_tp_config(config, on_unknown="warn") == []

    def test_invalid_predictor_type(self):
        from uav_vpp_guidance.trajectory_prediction.config_validator import validate_tp_config
        with pytest.raises(ValueError, match="predictor_type"):
            validate_tp_config({"predictor_type": "transformer"})

    def test_invalid_fallback_mode(self):
        from uav_vpp_guidance.trajectory_prediction.config_validator import validate_tp_config
        with pytest.raises(ValueError, match="fallback_mode"):
            validate_tp_config({"prediction": {"fallback_mode": "random_walk"}})

    def test_strict_init_missing_checkpoint(self):
        from uav_vpp_guidance.trajectory_prediction.config_validator import validate_tp_config
        with pytest.raises(ValueError, match="checkpoint_path"):
            validate_tp_config({
                "predictor_type": "lstm",
                "strict_predictor_init": True,
            })

    def test_invalid_device(self):
        from uav_vpp_guidance.trajectory_prediction.config_validator import validate_tp_config
        with pytest.raises(ValueError, match="device"):
            validate_tp_config({"device": "gpu"})

    def test_invalid_checkpoint_strict_type(self):
        from uav_vpp_guidance.trajectory_prediction.config_validator import validate_tp_config
        with pytest.raises(ValueError, match="checkpoint_strict"):
            validate_tp_config({"checkpoint_strict": "yes"})

    def test_unknown_key_warn(self):
        import warnings
        from uav_vpp_guidance.trajectory_prediction.config_validator import validate_tp_config
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_tp_config({"unkown_key_xyz": 123}, on_unknown="warn")
            assert any("unkown_key_xyz" in str(warning.message) for warning in w)

    def test_unknown_key_raise(self):
        from uav_vpp_guidance.trajectory_prediction.config_validator import validate_tp_config
        with pytest.raises(ValueError, match="Unknown"):
            validate_tp_config({"unkown_key_xyz": 123}, on_unknown="raise")


# ---------------------------------------------------------------------------
# Telemetry contract end-to-end test
# ---------------------------------------------------------------------------

class TestTelemetryContract:
    def test_env_info_contains_all_telemetry_fields(self, tmpdir):
        """CloseRangeTrackingEnv with LSTM predictor must output complete telemetry in info."""
        import os
        import torch
        from uav_vpp_guidance.trajectory_prediction.lstm_predictor import LSTMTrajectoryPredictor
        from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv

        dummy_model = LSTMTrajectoryPredictor(
            input_dim=16, hidden_dim=32, num_layers=1, dropout=0.0
        )
        ckpt_path = os.path.join(str(tmpdir), "lstm_dummy.pt")
        torch.save(dummy_model.state_dict(), ckpt_path)

        config = {
            "experiment": {"name": "test_telemetry", "seed": 42, "output_root": str(tmpdir)},
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
                "predictor_type": "lstm",
                "checkpoint_path": ckpt_path,
                "freeze_predictor_during_rl": True,
                "strict_predictor_init": True,
                "device": "cpu",
                "allow_device_fallback": True,
                "prediction": {
                    "lookahead_time_s": 1.0,
                    "output_mode": "relative_displacement",
                    "fallback_mode": "constant_velocity",
                },
                "history": {"history_len": 5, "padding_mode": "repeat_first"},
                "model": {
                    "input_dim": 16,
                    "hidden_dim": 32,
                    "num_layers": 1,
                    "dropout": 0.0,
                    "predict_variance": False,
                },
                "integration": {"anchor_mode": "predicted_target"},
                "normalization": {
                    "position_scale_m": 1000.0,
                    "velocity_scale_mps": 300.0,
                    "overload_scale": 9.0,
                },
            },
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

        env = CloseRangeTrackingEnv(config)
        env.reset(seed=0)

        # Step enough times to fill neural predictor buffer (history_len=5) + allow predictions to mature
        info = {}
        for _ in range(20):
            _, _, terminated, truncated, info = env.step(np.zeros(3))
            if terminated or truncated:
                break

        required_fields = [
            "prediction_enabled",
            "predictor_init_failed",
            "predictor_type",
            "prediction_valid",
            "prediction_fallback_reason",
            "prediction_fallback_mode",
            "prediction_fallback_model",
            "prediction_fallback_phase",
            "predicted_target_position",
            "prediction_error_m",
            "latest_prediction_error_m",
            "mean_prediction_error_m",
            "median_prediction_error_m",
            "prediction_error_count",
        ]
        for field in required_fields:
            assert field in info, f"Missing telemetry field: {field}"

        assert info["predictor_init_failed"] is False
        assert info["prediction_enabled"] is True
        env.close()

    def test_predictor_health_accumulator_rates(self):
        """PredictorHealthAccumulator must correctly classify warmup vs runtime fallback."""
        from uav_vpp_guidance.trajectory_prediction._telemetry import PredictorHealthAccumulator
        health = PredictorHealthAccumulator()
        # 3 warmup steps
        for _ in range(3):
            health.step({"prediction_enabled": True, "prediction_valid": False,
                         "prediction_fallback_phase": "warmup", "prediction_fallback_reason": "buffer not ready"})
        # 2 runtime failure steps
        for _ in range(2):
            health.step({"prediction_enabled": True, "prediction_valid": False,
                         "prediction_fallback_phase": "runtime_failure", "prediction_fallback_reason": "model error"})
        # 5 valid steps
        for _ in range(5):
            health.step({"prediction_enabled": True, "prediction_valid": True})
        rates = health.rates(10)
        assert rates["prediction_valid_rate"] == 0.5
        assert rates["fallback_rate"] == 0.5
        assert rates["warmup_fallback_rate"] == 0.3
        assert rates["runtime_fallback_rate"] == 0.2
        assert rates["post_warmup_fallback_rate"] == 0.2
