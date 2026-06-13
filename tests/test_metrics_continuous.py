"""
Tests for continuous evaluation metric helpers.
"""

import math

import numpy as np
import pytest

from uav_vpp_guidance.evaluation.metrics import compute_continuous_metrics


class TestContinuousMetrics:
    def test_empty_episodes(self):
        result = compute_continuous_metrics([])
        assert math.isnan(result["mean_final_range_m"])
        assert result["std_final_range_m"] == 0.0

    def test_basic_aggregation(self):
        episodes = [
            {"final_range_m": 100.0, "min_range_m": 50.0, "final_ata_deg": 10.0,
             "time_to_first_contact_s": 2.0, "control_effort": 30.0,
             "command_smoothness": 5.0, "return": -100.0, "length": 20},
            {"final_range_m": 200.0, "min_range_m": 80.0, "final_ata_deg": 20.0,
             "time_to_first_contact_s": 4.0, "control_effort": 50.0,
             "command_smoothness": 8.0, "return": -200.0, "length": 30},
        ]
        result = compute_continuous_metrics(episodes)
        assert result["mean_final_range_m"] == pytest.approx(150.0)
        assert result["mean_min_range_m"] == pytest.approx(65.0)
        assert result["mean_final_ata_deg"] == pytest.approx(15.0)
        assert result["mean_time_to_first_contact_s"] == pytest.approx(3.0)
        assert result["mean_control_effort"] == pytest.approx(40.0)
        assert result["mean_command_smoothness"] == pytest.approx(6.5)
        assert result["mean_return"] == pytest.approx(-150.0)
        assert result["mean_length"] == pytest.approx(25.0)

    def test_nan_ignored(self):
        episodes = [
            {"final_range_m": 100.0, "min_range_m": np.nan},
            {"final_range_m": 200.0, "min_range_m": 80.0},
        ]
        result = compute_continuous_metrics(episodes)
        assert result["mean_final_range_m"] == pytest.approx(150.0)
        assert result["mean_min_range_m"] == pytest.approx(80.0)

    def test_missing_keys_treated_as_nan(self):
        episodes = [
            {"final_range_m": 100.0},
            {"final_range_m": 300.0},
        ]
        result = compute_continuous_metrics(episodes)
        assert result["mean_final_range_m"] == pytest.approx(200.0)
        assert math.isnan(result["mean_min_range_m"])
