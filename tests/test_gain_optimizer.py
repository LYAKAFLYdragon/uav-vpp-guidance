"""
Unit tests for gain optimizer components.
"""

import pytest
import numpy as np
from uav_vpp_guidance.gain_optimizer.gain_space import GainSpace
from uav_vpp_guidance.gain_optimizer.cem import CEMGainOptimizer
from uav_vpp_guidance.gain_optimizer.regret import compute_empirical_regret, compute_score


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

    def test_sample_candidates_returns_array(self):
        bounds = {"k_los": [0.1, 5.0]}
        space = GainSpace(bounds)
        cem = CEMGainOptimizer(space, {"candidates": 8})
        candidates = cem.sample_candidates()
        assert candidates.shape == (8, 1)
        assert np.all(candidates >= 0.1) and np.all(candidates <= 5.0)


class TestRegret:
    def test_empirical_regret(self):
        scores = np.array([1.0, 3.0, 2.0])
        regret = compute_empirical_regret(scores, current_index=2)
        assert regret == pytest.approx(1.0)

    def test_empirical_regret_best(self):
        scores = np.array([1.0, 3.0, 2.0])
        regret = compute_empirical_regret(scores, current_index=1)
        assert regret == pytest.approx(0.0)

    def test_compute_score_basic(self):
        metrics = {
            "return": -50.0,
            "success_rate": 0.8,
            "crash_rate": 0.1,
            "saturation_rate": 0.05,
        }
        weights = {
            "return_norm": 0.3,
            "success_rate": 0.4,
            "crash_rate": -0.15,
            "saturation_rate": -0.15,
        }
        score = compute_score(metrics, weights=weights)
        # return_norm = (-50 + 1000) / 2000 = 0.475
        # raw = 0.3*0.475 + 0.4*0.8 - 0.15*0.1 - 0.15*0.05 = 0.44
        # max = 0.7, min = -0.3, denom = 1.0
        expected = (0.44 + 0.3) / 1.0
        assert score == pytest.approx(expected)
        assert 0.0 <= score <= 1.0

    def test_compute_score_missing_keys(self):
        metrics = {"success_rate": 1.0, "return": 10.0}
        weights = {
            "success_rate": 0.4,
            "return_norm": 0.3,
            "crash_rate": -0.15,
        }
        with pytest.warns(UserWarning, match="crash_rate"):
            score = compute_score(metrics, weights=weights)
        # return_norm = (10 + 1000) / 2000 = 0.505
        # raw = 0.4*1.0 + 0.3*0.505 = 0.5515
        # used keys: success_rate, return_norm => max = 0.7, min = 0, denom = 0.7
        expected = 0.5515 / 0.7
        assert score == pytest.approx(expected, rel=1e-4)
        assert 0.0 <= score <= 1.0

    def test_compute_score_default_weights(self):
        metrics = {
            "return": 0.0,
            "success_rate": 1.0,
            "crash_rate": 0.0,
            "saturation_rate": 0.0,
        }
        score = compute_score(metrics, weights=None)
        # return_norm = 0.5, raw = 0.3*0.5 + 0.4*1.0 = 0.55
        # max = 0.7, min = -0.3, denom = 1.0
        expected = (0.55 + 0.3) / 1.0
        assert score > 0.0
        assert score == pytest.approx(expected, rel=1e-3)
        assert 0.0 <= score <= 1.0

    def test_compute_score_empty_metrics(self):
        with pytest.warns(UserWarning):
            score = compute_score({}, weights={"success_rate": 0.4})
        assert score == pytest.approx(0.0)

    def test_compute_score_dynamic_return_bounds(self):
        metrics = {
            "return": 100.0,
            "return_min": -200.0,
            "return_max": 200.0,
            "success_rate": 0.8,
        }
        weights = {
            "return_norm": 0.3,
            "success_rate": 0.4,
        }
        score = compute_score(metrics, weights=weights)
        # return_norm = (100 + 200) / 400 = 0.75
        # raw = 0.3*0.75 + 0.4*0.8 = 0.225 + 0.32 = 0.545
        # max = 0.7, min = 0, denom = 0.7
        expected = 0.545 / 0.7
        assert score == pytest.approx(expected, rel=1e-4)
        assert 0.0 <= score <= 1.0
