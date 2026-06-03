"""
轨迹预测模型单元测试。
"""

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
