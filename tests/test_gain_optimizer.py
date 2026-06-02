"""
Unit tests for gain optimizer components.
"""

import pytest
import numpy as np
from uav_vpp_guidance.gain_optimizer.gain_space import GainSpace
from uav_vpp_guidance.gain_optimizer.cem import CEMGainOptimizer
from uav_vpp_guidance.gain_optimizer.regret import compute_empirical_regret


class TestGainSpace:
    def test_sample_shape(self):
        bounds = {"k_los": [0.1, 5.0], "k_pos": [0.0, 3.0]}
        space = GainSpace(bounds)
        samples = space.sample(10, seed=42)
        assert samples.shape == (10, 2)

    def test_clip(self):
        bounds = {"k_los": [0.0, 1.0]}
        space = GainSpace(bounds)
        clipped = space.clip(np.array([[-0.5, 2.0]]))
        assert clipped[0, 0] == 0.0

    def test_vector_to_gains(self):
        bounds = {"k_los": [0.1, 5.0], "k_pos": [0.0, 3.0]}
        space = GainSpace(bounds)
        gains = space.vector_to_gains(np.array([1.0, 2.0]))
        assert gains == {"k_los": 1.0, "k_pos": 2.0}


class TestCEMGainOptimizer:
    def test_init(self):
        bounds = {"k_los": [0.1, 5.0]}
        space = GainSpace(bounds)
        cem = CEMGainOptimizer(space, {"candidates": 12})
        assert cem.candidates == 12

    def test_sample_candidates_not_implemented(self):
        bounds = {"k_los": [0.1, 5.0]}
        space = GainSpace(bounds)
        cem = CEMGainOptimizer(space, {})
        with pytest.raises(NotImplementedError):
            cem.sample_candidates()


class TestRegret:
    def test_empirical_regret(self):
        scores = np.array([1.0, 3.0, 2.0])
        regret = compute_empirical_regret(scores, current_index=2)
        assert regret == pytest.approx(1.0)

    def test_empirical_regret_best(self):
        scores = np.array([1.0, 3.0, 2.0])
        regret = compute_empirical_regret(scores, current_index=1)
        assert regret == pytest.approx(0.0)
