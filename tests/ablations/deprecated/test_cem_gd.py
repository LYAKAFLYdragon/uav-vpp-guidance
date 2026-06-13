"""
Tests for the deprecated two-phase CEM-GD hybrid gain optimizer.

This optimizer has been moved to ablations/deprecated/ and is no longer
recommended for paper experiments. Use CEMEMAGainOptimizer instead.
See Theorem 3' and docs/status.md for the canonical optimizer choice.
"""

import numpy as np
import pytest

from uav_vpp_guidance.ablations.deprecated.cem_gd import CEMGDGainOptimizer
from uav_vpp_guidance.gain_optimizer.gain_space import GainSpace


class TestCEMGDGainOptimizer:
    def _make_optimizer(self, **overrides):
        bounds = {
            "k_loiter": [0.0, 2.0],
            "k_track": [0.0, 5.0],
        }
        config = {
            "candidates": 20,
            "elite_ratio": 0.25,
            "gd_ratio": 0.5,
            "gd_lr": 0.1,
            "gd_fd_eps": 1e-3,
            "noise_floor": 0.01,
            "convergence_tol": 1e-4,
        }
        config.update(overrides)
        return CEMGDGainOptimizer(GainSpace(bounds), config)

    def test_runs_with_quadratic_objective(self):
        """GD phase should refine the CEM solution on a simple quadratic."""
        opt = self._make_optimizer(gd_ratio=0.5)

        def evaluator(gains):
            x = np.array([gains["k_loiter"], gains["k_track"]])
            # Maximum at (1.5, 3.0)
            return -np.sum((x - np.array([1.5, 3.0])) ** 2)

        best, history = opt.optimize(evaluator, n_iter=40)
        assert best is not None
        assert "k_loiter" in best
        assert len(history) == 40

        best_vec = np.array([best["k_loiter"], best["k_track"]])
        assert np.allclose(best_vec, [1.5, 3.0], atol=0.2)

    def test_history_has_cem_and_gd_phases(self):
        opt = self._make_optimizer(gd_ratio=0.5)

        def evaluator(gains):
            return -(gains["k_loiter"] ** 2 + gains["k_track"] ** 2)

        best, history = opt.optimize(evaluator, n_iter=20)
        phases = {entry["phase"] for entry in history}
        assert "cem" in phases
        assert "gd" in phases

    def test_no_gd_phase_when_gd_ratio_zero(self):
        opt = self._make_optimizer(gd_ratio=0.0)

        def evaluator(gains):
            return -(gains["k_loiter"] ** 2 + gains["k_track"] ** 2)

        best, history = opt.optimize(evaluator, n_iter=10)
        assert all(entry["phase"] == "cem" for entry in history)

    def test_raises_on_nonpositive_iterations(self):
        opt = self._make_optimizer()
        with pytest.raises(ValueError):
            opt.optimize(lambda g: 0.0, n_iter=0)

    def test_best_score_improves_or_stays(self):
        opt = self._make_optimizer(gd_ratio=0.5, candidates=12)

        def evaluator(gains):
            return -(gains["k_loiter"] ** 2 + (gains["k_track"] - 2.0) ** 2)

        best, history = opt.optimize(evaluator, n_iter=30)
        final_score = evaluator(best)
        # The best score recorded across history should be at least the final score
        recorded_best = max(entry["best_score"] for entry in history)
        assert recorded_best >= final_score - 1e-6
