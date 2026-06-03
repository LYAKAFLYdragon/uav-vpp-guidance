"""
Tests for TrajectoryPredictionDataset and TrajectoryPredictorTrainer.

Uses dummy trajectory data to verify:
- from_episode_logs constructs correct (history_seq, target) pairs
- Trainer train/validate/fit loop runs without errors
- Loss decreases over a few epochs on synthetic data
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch

from uav_vpp_guidance.trajectory_prediction.dataset import (
    TrajectoryPredictionDataset,
    _compute_velocity_from_position,
    _build_states_from_trajectory,
)
from uav_vpp_guidance.trajectory_prediction.trainer import TrajectoryPredictorTrainer
from uav_vpp_guidance.trajectory_prediction.lstm_predictor import (
    LSTMTrajectoryPredictor,
)
from uav_vpp_guidance.trajectory_prediction.gru_predictor import GRUTrajectoryPredictor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dummy_config():
    return {
        "history": {"history_len": 5, "padding_mode": "repeat_first"},
        "prediction": {"lookahead_time_s": 1.0, "output_mode": "relative_displacement"},
        "env": {"high_level_dt": 0.2},
        "normalization": {
            "position_scale_m": 1000.0,
            "velocity_scale_mps": 300.0,
            "overload_scale": 9.0,
        },
    }


@pytest.fixture
def dummy_trajectory_df():
    """Generate a simple straight-line trajectory with 30 timesteps."""
    np.random.seed(0)
    n = 30
    dt = 0.2
    t = np.arange(n) * dt

    # Own aircraft: stationary at origin
    ego_x = np.zeros(n)
    ego_y = np.zeros(n)
    ego_z = np.full(n, 5000.0)

    # Target: moving straight along x at 100 m/s
    target_x = 100.0 * t + 1000.0
    target_y = np.zeros(n)
    target_z = np.full(n, 5000.0)

    df = pd.DataFrame(
        {
            "step": np.arange(n),
            "time": t,
            "backend": ["simple"] * n,
            "ego_x": ego_x,
            "ego_y": ego_y,
            "ego_z": ego_z,
            "target_x": target_x,
            "target_y": target_y,
            "target_z": target_z,
            "range_m": target_x,
            "ata_deg": np.zeros(n),
            "aspect_deg": np.zeros(n),
            "action_x": np.zeros(n),
            "action_y": np.zeros(n),
            "action_z": np.zeros(n),
            "reward": np.zeros(n),
            "done": np.zeros(n, dtype=bool),
            "termination_reason": [""] * n,
        }
    )
    return df


@pytest.fixture
def dummy_dataset(dummy_trajectory_df, dummy_config):
    return TrajectoryPredictionDataset.from_episode_logs(
        [dummy_trajectory_df], dummy_config
    )


# ---------------------------------------------------------------------------
# Dataset Tests
# ---------------------------------------------------------------------------


class TestComputeVelocity:
    def test_uniform_motion(self):
        pos = np.array(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [6.0, 0.0, 0.0]]
        )
        dt = 1.0
        vel = _compute_velocity_from_position(pos, dt)
        np.testing.assert_allclose(vel[:, 0], np.array([2.0, 2.0, 2.0, 2.0]), atol=1e-6)


class TestBuildStatesFromTrajectory:
    def test_basic_shape(self, dummy_trajectory_df):
        own, target, rel = _build_states_from_trajectory(dummy_trajectory_df, 0.2)
        assert len(own) == len(dummy_trajectory_df)
        assert len(target) == len(dummy_trajectory_df)
        assert len(rel) == len(dummy_trajectory_df)
        assert "position_m" in own[0]
        assert "velocity_vector_mps" in own[0]
        assert "attitude_rpy" in own[0]

    def test_target_velocity_approximation(self, dummy_trajectory_df):
        _, target, _ = _build_states_from_trajectory(dummy_trajectory_df, 0.2)
        # target moves at ~100 m/s along x
        vx = np.array([s["velocity_vector_mps"][0] for s in target])
        assert np.mean(vx[2:-2]) == pytest.approx(100.0, abs=5.0)


class TestTrajectoryPredictionDataset:
    def test_from_episode_logs_length(self, dummy_dataset, dummy_config):
        # 30 steps, lookahead=5 (1.0s / 0.2s), history=5
        # valid t range: [4, 24] -> 21 samples
        assert len(dummy_dataset) == 21

    def test_sample_shapes(self, dummy_dataset, dummy_config):
        history_len = dummy_config["history"]["history_len"]
        feature_dim = 16
        x, y = dummy_dataset[0]
        assert x.shape == (history_len, feature_dim)
        assert y.shape == (3,)
        assert x.dtype == torch.float32
        assert y.dtype == torch.float32

    def test_target_displacement_consistency(self, dummy_dataset, dummy_config):
        # target moves at ~100 m/s along x for 1.0s => disp ~ [100, 0, 0]
        _, y = dummy_dataset[5]
        assert y[0].item() == pytest.approx(100.0, abs=10.0)
        assert abs(y[1].item()) < 10.0
        assert abs(y[2].item()) < 10.0

    def test_from_csv_file(self, dummy_trajectory_df, dummy_config, tmpdir):
        csv_path = os.path.join(tmpdir, "episode_0.csv")
        dummy_trajectory_df.to_csv(csv_path, index=False)
        ds = TrajectoryPredictionDataset.from_episode_logs([csv_path], dummy_config)
        assert len(ds) == 21
        x, y = ds[0]
        assert x.shape == (5, 16)

    def test_from_list_of_dicts(self, dummy_trajectory_df, dummy_config):
        records = dummy_trajectory_df.to_dict("records")
        ds = TrajectoryPredictionDataset.from_episode_logs([records], dummy_config)
        assert len(ds) == 21

    def test_empty_logs_raises(self, dummy_config):
        with pytest.raises(ValueError, match="No valid samples"):
            TrajectoryPredictionDataset.from_episode_logs([], dummy_config)

    def test_short_episode_skipped_raises(self, dummy_trajectory_df, dummy_config):
        """过短的 episode 应被跳过；若全部跳过则抛出 ValueError。"""
        short_df = dummy_trajectory_df.head(3)
        with pytest.raises(ValueError, match="No valid samples"):
            TrajectoryPredictionDataset.from_episode_logs([short_df], dummy_config)


# ---------------------------------------------------------------------------
# Trainer Tests
# ---------------------------------------------------------------------------


class TestTrajectoryPredictorTrainer:
    def test_train_one_epoch_lstm(self, dummy_dataset):
        model = LSTMTrajectoryPredictor(input_dim=16, hidden_dim=32, num_layers=1)
        loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=4, shuffle=True)
        config = {
            "device": "cpu",
            "learning_rate": 1.0e-3,
            "weight_decay": 1.0e-5,
            "grad_clip": 1.0,
            "output_dir": tempfile.mkdtemp(),
        }
        trainer = TrajectoryPredictorTrainer(model, loader, loader, config)
        loss = trainer.train_one_epoch()
        assert np.isfinite(loss)
        assert loss >= 0.0

    def test_validate_gru(self, dummy_dataset):
        model = GRUTrajectoryPredictor(input_dim=16, hidden_dim=32, num_layers=1)
        loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=4, shuffle=False)
        config = {
            "device": "cpu",
            "learning_rate": 1.0e-3,
            "weight_decay": 1.0e-5,
            "grad_clip": 1.0,
            "output_dir": tempfile.mkdtemp(),
        }
        trainer = TrajectoryPredictorTrainer(model, loader, loader, config)
        val_loss = trainer.validate()
        assert np.isfinite(val_loss)
        assert val_loss >= 0.0

    def test_fit_decreases_loss_on_synthetic_data(self, dummy_dataset):
        """在可预测的直线运动数据上，少量 epoch 应降低损失。"""
        model = LSTMTrajectoryPredictor(input_dim=16, hidden_dim=64, num_layers=1)
        loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=4, shuffle=True)
        config = {
            "device": "cpu",
            "learning_rate": 1.0e-3,
            "weight_decay": 1.0e-5,
            "grad_clip": 1.0,
            "epochs": 10,
            "patience": 20,
            "output_dir": tempfile.mkdtemp(),
        }
        trainer = TrajectoryPredictorTrainer(model, loader, loader, config)
        history = trainer.fit()

        assert len(history["train_loss"]) == 10
        assert len(history["val_loss"]) == 10
        # Loss should generally decrease (first vs last)
        assert history["train_loss"][-1] < history["train_loss"][0]
        # Best model checkpoint should exist
        assert os.path.exists(os.path.join(config["output_dir"], "best_model.pt"))
        assert os.path.exists(os.path.join(config["output_dir"], "latest_model.pt"))

    def test_checkpoint_save_load(self, dummy_dataset):
        model = LSTMTrajectoryPredictor(input_dim=16, hidden_dim=32, num_layers=1)
        loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=4)
        config = {
            "device": "cpu",
            "learning_rate": 1.0e-3,
            "weight_decay": 1.0e-5,
            "grad_clip": 1.0,
            "output_dir": tempfile.mkdtemp(),
        }
        trainer = TrajectoryPredictorTrainer(model, loader, loader, config)
        trainer.save_checkpoint("test.pt")
        path = os.path.join(config["output_dir"], "test.pt")
        assert os.path.exists(path)

        # Load into fresh model
        model2 = LSTMTrajectoryPredictor(input_dim=16, hidden_dim=32, num_layers=1)
        trainer2 = TrajectoryPredictorTrainer(model2, loader, loader, config)
        trainer2.load_checkpoint("test.pt")
        # Verify weights match
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)
