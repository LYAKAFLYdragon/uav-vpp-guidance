"""Tests for BilevelTrainer."""

import pytest

from uav_vpp_guidance.gain_optimizer.bilevel_trainer import BilevelTrainer
from uav_vpp_guidance.gain_optimizer.cem import CEMGainOptimizer
from uav_vpp_guidance.gain_optimizer.gain_space import GainSpace


class TestBilevelTrainer:
    def test_init(self):
        gain_space = GainSpace({"x": [0.0, 1.0]})
        cem = CEMGainOptimizer(gain_space, {"candidates": 5, "elite_ratio": 0.4})
        trainer = BilevelTrainer(None, None, cem, {
            "outer_every": 2, "inner_iter": 3, "n_episodes": 4
        })
        assert trainer.outer_every == 2
        assert trainer.inner_iter == 3
        assert trainer.n_episodes == 4

    def test_compute_regret(self):
        gain_space = GainSpace({"x": [0.0, 1.0]})
        cem = CEMGainOptimizer(gain_space, {"candidates": 5})
        trainer = BilevelTrainer(None, None, cem, {})
        trainer.history = [{"eval_success_rate": 0.8}]
        # Current eval 0.7 is worse than best_known 0.8 → regret = 0.2
        assert trainer._compute_regret(0.7) == pytest.approx(0.2)
        trainer.history.append({"eval_success_rate": 0.9})
        # Current eval 0.85 is worse than best_known 0.9 → regret = 0.1
        assert trainer._compute_regret(0.85) == pytest.approx(0.1)

    def test_regret_monotonically_non_increasing(self):
        """Regret should never increase as best_known improves."""
        gain_space = GainSpace({"x": [0.0, 1.0]})
        cem = CEMGainOptimizer(gain_space, {"candidates": 5})
        trainer = BilevelTrainer(None, None, cem, {})

        regrets = []
        for sr in [0.5, 0.6, 0.4, 0.7, 0.6, 0.8]:
            regrets.append(trainer._compute_regret(sr))
            trainer.history.append({"eval_success_rate": sr})

        for i in range(1, len(regrets)):
            assert regrets[i] <= regrets[i - 1], f"Regret increased at step {i}: {regrets[i-1]} -> {regrets[i]}"

    def test_filter_gains(self):
        raw = {"k_los": 1.0, "k_pos": 0.5, "invalid_key": 99.0}
        filtered = BilevelTrainer._filter_gains(raw)
        assert "k_los" in filtered
        assert "k_pos" in filtered
        assert "invalid_key" not in filtered
