"""
Unit tests for reward calculation.
"""

import numpy as np
import pytest
from uav_vpp_guidance.envs.reward import RewardCalculator


class TestRewardCalculator:
    def test_init(self):
        config = {"reward": {"w_range": 0.5, "w_angle": 0.8}}
        calc = RewardCalculator(config)
        assert calc.w_range == 0.5
        assert calc.w_angle == 0.8

    def test_compute_returns_scalar_and_terms(self):
        calc = RewardCalculator(config={})
        obs = {}
        info = {
            "relative_state": {
                "range_m": 1000.0,
                "ata_rad": np.deg2rad(10.0),
                "aa_rad": np.deg2rad(15.0),
            },
            "own_state": {"altitude_m": 5000.0},
            "command": {"nz_cmd": 2.0, "roll_rate_cmd": 0.5, "throttle_cmd": 0.7},
        }
        reward, terms = calc.compute(info)
        assert isinstance(reward, float)
        assert isinstance(terms, dict)
        assert "reward_total" in terms
        assert "reward_range" in terms
        assert "reward_angle" in terms

    def test_compute_with_saturation(self):
        calc = RewardCalculator(config={})
        info = {
            "relative_state": {"range_m": 1000.0, "ata_rad": 0.0, "aa_rad": 0.0},
            "own_state": {"altitude_m": 5000.0},
            "command": {"nz_cmd": 8.0, "roll_rate_cmd": 2.0},
        }
        reward, terms = calc.compute(info)
        # Saturation penalty should make reward more negative
        assert terms["reward_saturation"] < 0.0

    def test_reset_clears_prev_command(self):
        calc = RewardCalculator(config={})
        calc._prev_command = {"nz_cmd": 1.0}
        calc.reset()
        assert calc._prev_command is None

    def test_potential_based_shaping_zero_when_range_constant(self):
        calc = RewardCalculator(config={
            "reward": {
                "potential_based_shaping": {"enabled": True, "C": 0.1, "gamma": 0.99}
            }
        })
        info = {
            "relative_state": {"range_m": 1000.0, "ata_rad": 0.0, "aa_rad": 0.0},
            "own_state": {"altitude_m": 5000.0},
            "command": {"nz_cmd": 1.0, "roll_rate_cmd": 0.0, "throttle_cmd": 0.5},
        }
        # First call initializes previous state, no shaping
        _, terms1 = calc.compute(info)
        assert terms1["reward_potential"] == pytest.approx(0.0)
        # Second call with same range: gamma*phi - phi = (gamma-1)*phi = (gamma-1)*(-C*R)
        _, terms2 = calc.compute(info)
        expected = (0.99 - 1.0) * (-0.1 * 1000.0)
        assert terms2["reward_potential"] == pytest.approx(expected)

    def test_potential_based_shaping_positive_when_approaching(self):
        calc = RewardCalculator(config={
            "reward": {
                "potential_based_shaping": {"enabled": True, "C": 0.1, "gamma": 0.99}
            }
        })
        info_far = {
            "relative_state": {"range_m": 1000.0, "ata_rad": 0.0, "aa_rad": 0.0},
            "own_state": {"altitude_m": 5000.0},
            "command": {"nz_cmd": 1.0, "roll_rate_cmd": 0.0, "throttle_cmd": 0.5},
        }
        calc.compute(info_far)
        info_close = {
            "relative_state": {"range_m": 900.0, "ata_rad": 0.0, "aa_rad": 0.0},
            "own_state": {"altitude_m": 5000.0},
            "command": {"nz_cmd": 1.0, "roll_rate_cmd": 0.0, "throttle_cmd": 0.5},
        }
        _, terms = calc.compute(info_close)
        # Approaching target -> positive potential reward
        assert terms["reward_potential"] > 0.0

    def test_potential_based_shaping_negative_when_receding(self):
        calc = RewardCalculator(config={
            "reward": {
                "potential_based_shaping": {"enabled": True, "C": 0.1, "gamma": 0.99}
            }
        })
        info_close = {
            "relative_state": {"range_m": 900.0, "ata_rad": 0.0, "aa_rad": 0.0},
            "own_state": {"altitude_m": 5000.0},
            "command": {"nz_cmd": 1.0, "roll_rate_cmd": 0.0, "throttle_cmd": 0.5},
        }
        calc.compute(info_close)
        info_far = {
            "relative_state": {"range_m": 1000.0, "ata_rad": 0.0, "aa_rad": 0.0},
            "own_state": {"altitude_m": 5000.0},
            "command": {"nz_cmd": 1.0, "roll_rate_cmd": 0.0, "throttle_cmd": 0.5},
        }
        _, terms = calc.compute(info_far)
        # Receding from target -> negative potential reward
        assert terms["reward_potential"] < 0.0

    def test_potential_based_shaping_disabled_by_default(self):
        calc = RewardCalculator(config={})
        assert calc.pbs_enabled is False
