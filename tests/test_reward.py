"""
Unit tests for reward calculation.
"""

import pytest
import numpy as np
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
        reward, terms = calc.compute(obs, info)
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
        reward, terms = calc.compute(None, info)
        # Saturation penalty should make reward more negative
        assert terms["reward_saturation"] < 0.0

    def test_reset_clears_prev_command(self):
        calc = RewardCalculator(config={})
        calc._prev_command = {"nz_cmd": 1.0}
        calc.reset()
        assert calc._prev_command is None
