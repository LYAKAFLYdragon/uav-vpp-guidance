"""Tests for CEMGainOptimizer."""

import numpy as np
import pytest

from uav_vpp_guidance.gain_optimizer.cem import CEMGainOptimizer
from uav_vpp_guidance.gain_optimizer.gain_space import GainSpace


class TestCEMGainOptimizer:
    def test_sample_candidates_within_bounds(self):
        bounds = {"a": [0.0, 1.0], "b": [-1.0, 1.0]}
        gs = GainSpace(bounds)
        cem = CEMGainOptimizer(gs, {"candidates": 10, "elite_ratio": 0.3})
        candidates = cem.sample_candidates()
        assert candidates.shape == (10, 2)
        assert np.all(candidates[:, 0] >= 0.0) and np.all(candidates[:, 0] <= 1.0)
        assert np.all(candidates[:, 1] >= -1.0) and np.all(candidates[:, 1] <= 1.0)

    def test_update_reduces_std(self):
        bounds = {"a": [0.0, 10.0]}
        gs = GainSpace(bounds)
        cem = CEMGainOptimizer(gs, {"candidates": 20, "elite_ratio": 0.25, "noise_floor": 0.0})
        candidates = np.array([[i] for i in range(20)], dtype=np.float32)
        scores = np.array([i for i in range(20)], dtype=np.float32)  # Higher = better
        old_std = cem.std.copy()
        cem.update(candidates, scores)
        assert cem.std[0] < old_std[0]  # Std should decrease

    def test_optimize_on_quadratic(self):
        """CEM should find the maximum of a simple quadratic."""
        bounds = {"x": [-5.0, 5.0]}
        gs = GainSpace(bounds)
        cem = CEMGainOptimizer(gs, {
            "candidates": 20, "elite_ratio": 0.3,
            "noise_floor": 0.01, "convergence_tol": 0.001
        })

        def evaluator(gains_dict):
            x = gains_dict["x"]
            return -(x - 2.5) ** 2  # Max at x = 2.5

        best, history = cem.optimize(evaluator, n_iter=30)
        assert best is not None
        assert abs(best["x"] - 2.5) < 0.5  # Should be close to optimum
        # Score should improve
        assert history[-1]["best_score"] > history[0]["best_score"]

    def test_optimize_returns_history(self):
        bounds = {"a": [0.0, 1.0]}
        gs = GainSpace(bounds)
        cem = CEMGainOptimizer(gs, {"candidates": 5, "elite_ratio": 0.4})

        def evaluator(g):
            return g["a"]

        best, history = cem.optimize(evaluator, n_iter=3)
        assert len(history) == 3
        assert all("iteration" in h for h in history)
        assert all("best_score" in h for h in history)
        # best_score should be monotonically non-decreasing (global best)
        for i in range(1, len(history)):
            assert history[i]["best_score"] >= history[i - 1]["best_score"]

    def test_rng_reproducibility(self):
        """Same seed should produce identical results."""
        bounds = {"x": [0.0, 1.0]}
        gs = GainSpace(bounds)

        def evaluator(g):
            return g["x"]

        cem1 = CEMGainOptimizer(gs, {"candidates": 10, "elite_ratio": 0.3, "random_seed": 123})
        cem2 = CEMGainOptimizer(gs, {"candidates": 10, "elite_ratio": 0.3, "random_seed": 123})

        best1, hist1 = cem1.optimize(evaluator, n_iter=5)
        best2, hist2 = cem2.optimize(evaluator, n_iter=5)

        assert best1 == pytest.approx(best2, abs=1e-6)
        for h1, h2 in zip(hist1, hist2):
            assert h1["best_score"] == pytest.approx(h2["best_score"], abs=1e-6)
            np.testing.assert_allclose(h1["mean"], h2["mean"], atol=1e-6)
            np.testing.assert_allclose(h1["std"], h2["std"], atol=1e-6)
