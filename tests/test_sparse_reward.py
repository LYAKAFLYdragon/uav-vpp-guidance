"""
Unit tests for the R2SP-style sparse reward calculator.
"""

import os

import numpy as np
import pytest

from uav_vpp_guidance.envs.sparse_reward import (
    gaussian_kernel,
    linear_kernel,
    redistribute_rewards,
)
from uav_vpp_guidance.envs.reward import (
    SparseRewardCalculator,
    build_reward_calculator,
)


class TestKernels:
    def test_linear_kernel_normalises(self):
        weights = linear_kernel(0, 9)
        assert weights.shape == (10,)
        assert weights.sum() == pytest.approx(1.0)
        assert np.allclose(weights, 0.1)

    def test_linear_kernel_single_step(self):
        weights = linear_kernel(5, 5)
        assert weights.sum() == pytest.approx(1.0)
        assert weights[0] == pytest.approx(1.0)

    def test_gaussian_kernel_normalises(self):
        weights = gaussian_kernel(0, 49, sigma_ratio=0.5)
        assert weights.shape == (50,)
        assert weights.sum() == pytest.approx(1.0)

    def test_gaussian_kernel_peaks_at_event(self):
        weights = gaussian_kernel(0, 29, sigma_ratio=0.5)
        assert weights[-1] == pytest.approx(weights.max())
        # Credit should decay backwards in time.
        assert weights[0] < weights[-1]


class TestRedistributeRewards:
    def test_terminal_linear_uniform(self):
        relabelled = redistribute_rewards(10, terminal_reward=10.0)
        assert relabelled.sum() == pytest.approx(10.0)
        assert np.allclose(relabelled, 1.0)

    def test_event_backward_smoothing_conserves_total(self):
        events = [(30, 1.0)]
        relabelled = redistribute_rewards(
            60, terminal_reward=0.0, events=events, gaussian_window=50
        )
        assert relabelled.sum() == pytest.approx(1.0)
        # Only steps up to the event receive credit.
        assert np.all(relabelled[31:] == 0.0)

    def test_combined_terminal_and_events_conserves_total(self):
        events = [(20, 1.0), (70, 2.0)]
        relabelled = redistribute_rewards(
            100, terminal_reward=10.0, events=events, gaussian_window=50
        )
        assert relabelled.sum() == pytest.approx(13.0)


class TestSparseRewardCalculator:
    def _make_info(
        self,
        range_m=2000.0,
        ata_deg=30.0,
        altitude_m=5000.0,
        terminal_reward=0.0,
        in_lock=False,
    ):
        return {
            "relative_state": {
                "range_m": float(range_m),
                "ata_rad": np.deg2rad(float(ata_deg)),
            },
            "own_state": {"altitude_m": float(altitude_m)},
            "command": {"nz_cmd": 1.0, "roll_rate_cmd": 0.0, "throttle_cmd": 0.5},
            "terminal_reward": float(terminal_reward),
            "in_lock": in_lock,
        }

    def test_init_exposes_terminal_magnitudes(self):
        calc = SparseRewardCalculator({})
        assert calc.terminal_success == pytest.approx(10.0)
        assert calc.terminal_crash == pytest.approx(-10.0)
        assert calc.terminal_failure == pytest.approx(-5.0)

    def test_compute_returns_scalar_and_terms(self):
        calc = SparseRewardCalculator({})
        reward, terms = calc.compute(self._make_info())
        assert isinstance(reward, float)
        assert isinstance(terms, dict)
        assert "reward_total" in terms
        assert "reward_event_proximity" in terms
        assert "reward_lock" in terms

    def test_proximity_event_fires_once(self):
        calc = SparseRewardCalculator(
            {"sparse_reward": {"event_range_m": 1000.0, "event_ata_deg": 30.0}}
        )
        # First call enters the zone -> event reward.
        r1, t1 = calc.compute(self._make_info(range_m=900.0, ata_deg=20.0))
        assert t1["reward_event_proximity"] == pytest.approx(1.0)
        assert len(calc._events) == 1
        # Second call stays in the zone -> no additional event reward.
        r2, t2 = calc.compute(self._make_info(range_m=850.0, ata_deg=15.0))
        assert t2["reward_event_proximity"] == pytest.approx(0.0)
        assert len(calc._events) == 1

    def test_lock_reward_from_geometry(self):
        calc = SparseRewardCalculator(
            {
                "sparse_reward": {"lock_reward": 0.05},
                "env": {"success_range_m": 900.0, "success_ata_deg": 25.0},
            }
        )
        _, terms_in = calc.compute(self._make_info(range_m=800.0, ata_deg=20.0))
        assert terms_in["reward_lock"] == pytest.approx(0.05)

        calc.reset()
        _, terms_out = calc.compute(self._make_info(range_m=1200.0, ata_deg=40.0))
        assert terms_out["reward_lock"] == pytest.approx(0.0)

    def test_lock_reward_from_info_flag(self):
        calc = SparseRewardCalculator({"sparse_reward": {"lock_reward": 0.1}})
        _, terms = calc.compute(self._make_info(range_m=5000.0, in_lock=True))
        assert terms["reward_lock"] == pytest.approx(0.1)

    def test_terminal_reward_recorded_and_finalized(self):
        calc = SparseRewardCalculator({})
        for _ in range(50):
            calc.compute(self._make_info())
        # Terminal success on the last step.
        r, terms = calc.compute(
            self._make_info(terminal_reward=calc.terminal_success)
        )
        assert terms["terminal_reward"] == pytest.approx(10.0)
        assert r == pytest.approx(10.0)

        redistributed = calc.finalize()
        assert len(redistributed) == 51
        assert redistributed.sum() == pytest.approx(10.0)
        assert np.allclose(redistributed, 10.0 / 51.0)

    def test_reward_conservation_with_events_and_lock(self):
        rng = np.random.default_rng(42)
        config = {
            "sparse_reward": {
                "event_range_m": 1200.0,
                "event_ata_deg": 45.0,
                "event_reward": 1.0,
                "lock_reward": 0.05,
            }
        }
        calc = SparseRewardCalculator(config)

        base_total = 0.0
        for t in range(100):
            # Randomly wander into / out of the event zone to generate events.
            range_m = 800.0 + rng.uniform(-200.0, 800.0)
            ata_deg = rng.uniform(0.0, 60.0)
            terminal = 0.0
            if t == 99:
                terminal = calc.terminal_success
            r, _ = calc.compute(
                self._make_info(range_m=range_m, ata_deg=ata_deg, terminal_reward=terminal)
            )
            base_total += r

        redistributed = calc.finalize()
        assert redistributed.sum() == pytest.approx(base_total, rel=1e-9)

    def test_reset_clears_internal_state(self):
        calc = SparseRewardCalculator({})
        calc.compute(self._make_info(range_m=500.0, ata_deg=10.0))
        assert len(calc._events) == 1
        calc.reset()
        assert len(calc._events) == 0
        assert calc._terminal_step is None


class TestFactory:
    def test_build_reward_calculator_selects_dense_by_default(self):
        calc = build_reward_calculator({})
        from uav_vpp_guidance.envs.reward import RewardCalculator

        assert isinstance(calc, RewardCalculator)

    def test_build_reward_calculator_selects_sparse_when_enabled(self):
        calc = build_reward_calculator(
            {"reward": {"use_sparse": True}, "sparse_reward": {"enabled": True}}
        )
        assert isinstance(calc, SparseRewardCalculator)


class TestCBFCompatibility:
    def test_cbf_filter_runs_with_sparse_reward(self):
        """CBF safety filter is independent of reward signal and stays active."""
        from uav_vpp_guidance.utils.config import load_yaml_config, merge_config
        from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv

        cfg_path = "config/experiment/sparse_reward_pilot.yaml"
        base = load_yaml_config(cfg_path)
        includes = base.pop("includes", [])
        merged = {}
        for inc in includes:
            p = os.path.join(os.path.dirname(cfg_path), inc)
            if os.path.exists(p):
                merged = merge_config(merged, load_yaml_config(p))
        config = merge_config(merged, base)
        config["cbf"] = {"enabled": True, "params": {"d_min": 500.0}}

        env = CloseRangeTrackingEnv(config)
        assert isinstance(env.reward_calculator, SparseRewardCalculator)
        assert env._cbf_filter is not None

        env.reset(seed=0)
        for _ in range(3):
            _, reward, terminated, truncated, info = env.step(np.zeros(3))
            assert isinstance(reward, float)
            assert "cbf" in info
            if terminated or truncated:
                break
