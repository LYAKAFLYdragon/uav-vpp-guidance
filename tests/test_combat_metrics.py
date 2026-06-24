import math

import pytest

from uav_vpp_guidance.metrics.combat_evaluator import compute_combat_metrics
from uav_vpp_guidance.metrics.compat import normalize_metric_name


def test_compute_combat_metrics_counts_hp_decided_timeouts_and_excludes_draws():
    metrics = compute_combat_metrics(
        [
            {"combat_outcome": "win", "ego_hp": 80, "target_hp": 0, "combat_time_to_kill": 4.0},
            {"combat_outcome": "loss", "combat_reason": "ego_killed", "ego_hp": 0, "target_hp": 30},
            {
                "combat_outcome": "win",
                "termination_reason": "timeout_hp_advantage",
                "ego_hp": 70,
                "target_hp": 50,
            },
            {
                "combat_outcome": "loss",
                "termination_reason": "timeout_hp_disadvantage",
                "ego_hp": 40,
                "target_hp": 60,
            },
            {"combat_outcome": "win", "termination_reason": "timeout", "ego_hp": 70, "target_hp": 50},
            {"combat_outcome": "draw", "ego_hp": 10, "target_hp": 10},
        ]
    )

    assert metrics["win_rate"] == pytest.approx(0.5)
    assert metrics["combat_success_rate"] == pytest.approx(0.5)
    assert metrics["survival_rate"] == pytest.approx(5 / 6)
    assert metrics["hp_advantage"] == pytest.approx((80 + 20 + 20) / 3)
    assert metrics["mean_time_to_kill"] == pytest.approx(4.0)
    assert len(metrics["time_to_kill"]) == 6
    assert math.isnan(metrics["time_to_kill"][1])


def test_metric_aliases_normalize_legacy_names():
    assert normalize_metric_name("success_rate") == "combat_success_rate"
    assert normalize_metric_name("mean_range_m") == "_deprecated_tracking_mean_range_m"
    assert normalize_metric_name("win_rate") == "win_rate"
