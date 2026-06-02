"""
TrajectoryStateBuffer 单元测试。
"""

import pytest
import numpy as np
from uav_vpp_guidance.trajectory_prediction.state_buffer import TrajectoryStateBuffer


class TestTrajectoryStateBuffer:
    def test_push_length(self):
        buf = TrajectoryStateBuffer(history_len=5, feature_dim=3)
        buf.push(np.array([1.0, 2.0, 3.0]))
        assert len(buf._buffer) == 1
        buf.push(np.array([4.0, 5.0, 6.0]))
        assert len(buf._buffer) == 2

    def test_maxlen_eviction(self):
        buf = TrajectoryStateBuffer(history_len=3, feature_dim=2)
        buf.push(np.array([1.0, 1.0]))
        buf.push(np.array([2.0, 2.0]))
        buf.push(np.array([3.0, 3.0]))
        buf.push(np.array([4.0, 4.0]))
        assert len(buf._buffer) == 3
        seq = buf.get_sequence()
        assert np.allclose(seq[0], [2.0, 2.0])
        assert np.allclose(seq[-1], [4.0, 4.0])

    def test_repeat_first_padding(self):
        buf = TrajectoryStateBuffer(history_len=5, feature_dim=3, padding_mode="repeat_first")
        buf.push(np.array([1.0, 2.0, 3.0]))
        buf.push(np.array([4.0, 5.0, 6.0]))
        seq = buf.get_sequence()
        assert seq.shape == (5, 3)
        assert np.allclose(seq[0], [1.0, 2.0, 3.0])
        assert np.allclose(seq[1], [1.0, 2.0, 3.0])
        assert np.allclose(seq[2], [1.0, 2.0, 3.0])
        assert np.allclose(seq[3], [1.0, 2.0, 3.0])
        assert np.allclose(seq[4], [4.0, 5.0, 6.0])

    def test_zero_padding(self):
        buf = TrajectoryStateBuffer(history_len=4, feature_dim=2, padding_mode="zero")
        buf.push(np.array([7.0, 8.0]))
        seq = buf.get_sequence()
        assert seq.shape == (4, 2)
        assert np.allclose(seq[0], [0.0, 0.0])
        assert np.allclose(seq[1], [0.0, 0.0])
        assert np.allclose(seq[2], [0.0, 0.0])
        assert np.allclose(seq[3], [7.0, 8.0])

    def test_empty_buffer_raises(self):
        buf = TrajectoryStateBuffer(history_len=5, feature_dim=3)
        with pytest.raises(ValueError, match="Buffer is empty"):
            buf.get_sequence()

    def test_is_ready(self):
        buf = TrajectoryStateBuffer(history_len=3, feature_dim=2)
        assert not buf.is_ready()
        buf.push(np.array([1.0, 1.0]))
        buf.push(np.array([2.0, 2.0]))
        assert not buf.is_ready()
        buf.push(np.array([3.0, 3.0]))
        assert buf.is_ready()

    def test_reset(self):
        buf = TrajectoryStateBuffer(history_len=5, feature_dim=3)
        buf.push(np.array([1.0, 2.0, 3.0]))
        buf.reset()
        assert len(buf._buffer) == 0
        with pytest.raises(ValueError):
            buf.get_sequence()

    def test_feature_dim_mismatch(self):
        buf = TrajectoryStateBuffer(history_len=5, feature_dim=3)
        with pytest.raises(ValueError, match="Feature dim mismatch"):
            buf.push(np.array([1.0, 2.0]))
