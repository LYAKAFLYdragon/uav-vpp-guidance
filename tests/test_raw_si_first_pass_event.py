from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from uav_vpp_guidance.evaluation.raw_si_first_pass_event import (
    RawSIFirstPassConfig,
    RawSIFirstPassEventError,
    RawSIFirstPassTracker,
    replay_raw_si_first_pass,
)


ROOT = Path(__file__).resolve().parent.parent
ANALYZER = ROOT / "scripts" / "analyze_thesis_neutral_postmerge_reentry_recovery_v4_phase_semantics.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_raw_si_event_requires_confirmed_closing_then_two_stable_opening_steps():
    tracker = RawSIFirstPassTracker()

    for index, rate in enumerate((-30.0, -10.0, -0.1, 0.0)):
        assert tracker.update(rate, step_index=index) is None
    event = tracker.update(0.1, step_index=4)

    assert event is not None
    assert event.event_step_index == 4
    assert event.closing_steps_before_event == 3
    assert event.opening_steps_at_event == 2
    assert tracker.first_pass_complete is True
    assert tracker.update(-100.0, step_index=5) == event


def test_raw_si_event_does_not_treat_stationary_zero_range_rate_as_a_prior_closing_pass():
    tracker = RawSIFirstPassTracker()
    for index in range(10):
        assert tracker.update(0.0, step_index=index) is None
    assert tracker.first_pass_complete is False


def test_replay_requires_raw_si_range_rate_and_validates_configuration():
    steps = [
        {"raw_si_phase_input": {"range_rate_mps": -4.0}},
        {"raw_si_phase_input": {"range_rate_mps": -2.0}},
        {"raw_si_phase_input": {"range_rate_mps": -1.0}},
        {"raw_si_phase_input": {"range_rate_mps": 0.0}},
        {"raw_si_phase_input": {"range_rate_mps": 0.2}},
    ]
    event = replay_raw_si_first_pass(steps)
    assert event is not None and event.event_step_index == 4

    with pytest.raises(RawSIFirstPassEventError, match="lacks raw_si_phase_input"):
        replay_raw_si_first_pass([{}])
    with pytest.raises(RawSIFirstPassEventError, match="non-negative"):
        RawSIFirstPassConfig(closing_rate_threshold_mps=-1.0)


def test_phase_semantics_analyzer_counts_raw_event_separately_from_environment_flag():
    analyzer = _load(ANALYZER, "v4_phase_semantics_analyzer")
    steps = []
    for index in range(25):
        steps.append(
            {
                "raw_si_phase_input": {"range_rate_mps": -10.0 if index < 3 else 0.0},
                "dynamic_state": "neutral",
                "phase": "post_merge",
                "prediction_valid": True,
                "prediction_fallback": False,
                "first_pass_complete": False,
            }
        )
    record = {
        "source_id": "THESIS-NEUTRAL-POSTMERGE-REENTRY-RECOVERY-RUNTIME-FEASIBILITY-V4",
        "opponent": "expert",
        "family_id": "close_balanced",
        "metadata": {"mirror_sign": "negative", "height_condition": "co_altitude"},
        "result": {"handoff_reached": False, "terminal_reason": "timeout"},
    }
    report = analyzer.analyze_episodes([(record, {"steps": steps})])
    item = report["opponent_family_matrix"]["expert"]["close_balanced"]

    assert item["raw_event_episodes"] == 1
    assert item["environment_first_pass_episodes"] == 0
    assert item["post_event_neutral_prediction_window_ge20"] == 1
