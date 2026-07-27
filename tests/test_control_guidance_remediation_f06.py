"""Deterministic, offline F06 multi-rate architecture and delay evidence tests."""
import hashlib
import json
from pathlib import Path

import pytest

from uav_vpp_guidance.evaluation.f06_multirate_evidence import (
    CANDIDATE_MULTI_RATE,
    LEGACY_SINGLE_RATE,
    DeterministicMultiRateScheduler,
    MultiRateSchedule,
    compare_delay_spectra,
    resample_command,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / ".kiro/specs/control-guidance-remediation"
BUNDLE = SPEC / "evidence" / "f06_multirate_bundle"


def test_exact_invocation_counts_and_timestamps_for_integer_schedule():
    trace = DeterministicMultiRateScheduler(CANDIDATE_MULTI_RATE).run(1.0, lambda _: 1.0)
    assert len(trace.high_level_timestamps_s) == 10
    assert len(trace.guidance_timestamps_s) == 20
    assert len(trace.filter_timestamps_s) == 20
    assert len(trace.low_level_timestamps_s) == 100
    assert trace.high_level_timestamps_s[:2] == [0.0, 0.1]
    assert trace.guidance_timestamps_s[:3] == [0.0, 0.05, 0.1]
    assert trace.low_level_timestamps_s[:3] == [0.0, 0.01, 0.02]


def test_command_is_zero_order_held_between_guidance_updates_and_timestamps_are_explicit():
    trace = DeterministicMultiRateScheduler(CANDIDATE_MULTI_RATE).run(0.12, lambda time_s: time_s)
    applied = dict(trace.applied_commands)
    assert applied[0.01] == pytest.approx(0.0)
    assert applied[0.04] == pytest.approx(0.0)
    # High-level actions update at 0.1 s; the later guidance tick uses that
    # held action and makes the filtered command positive.
    assert applied[0.06] == pytest.approx(0.0)
    assert applied[0.11] > 0.0
    sample = trace.telemetry[1]
    assert sample["timestamp_s"] == pytest.approx(0.01)
    assert sample["action_timestamp_s"] == pytest.approx(0.0)
    assert sample["guidance_timestamp_s"] == pytest.approx(0.0)
    assert sample["action_age_s"] == pytest.approx(0.01)


def test_linear_interpolation_requires_an_explicit_future_sample():
    assert resample_command(2.0, None, 0.5, "zero_order_hold") == pytest.approx(2.0)
    assert resample_command(2.0, 6.0, 0.5, "linear") == pytest.approx(4.0)
    with pytest.raises(ValueError, match="requires an explicit next sample"):
        resample_command(2.0, None, 0.5, "linear")


def test_reset_clears_hold_and_filter_state_before_a_fresh_trace():
    scheduler = DeterministicMultiRateScheduler(CANDIDATE_MULTI_RATE)
    scheduler.run(0.2, lambda _: 1.0)
    scheduler.reset()
    trace = scheduler.run(0.02, lambda _: 0.0)
    assert [command for _, command in trace.applied_commands] == pytest.approx([0.0, 0.0])
    assert trace.telemetry[0]["action_age_s"] == pytest.approx(0.0)


def test_non_integer_ratios_are_rejected_not_silently_drifted():
    with pytest.raises(ValueError, match="integer"):
        MultiRateSchedule(0.1, 0.06, 0.01)
    with pytest.raises(ValueError, match="integer"):
        MultiRateSchedule(0.1, 0.05, 0.03)


def test_legacy_single_rate_reproduces_synchronous_trace_and_filter_behavior():
    trace = DeterministicMultiRateScheduler(LEGACY_SINGLE_RATE).run(
        1.0, lambda time_s: 1.0 if time_s >= 0.4 else 0.0
    )
    assert trace.high_level_timestamps_s == trace.guidance_timestamps_s == trace.filter_timestamps_s == trace.low_level_timestamps_s
    assert trace.high_level_timestamps_s == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8])
    assert [value for _, value in trace.applied_commands] == pytest.approx([0.0, 0.0, 0.3, 0.51, 0.657])


def test_delay_spectrum_reports_filter_constant_as_component_not_total_latency():
    evidence = compare_delay_spectra()
    assert len(evidence["baseline"]) == len(evidence["candidate"]) == 5
    first = evidence["baseline"][0]
    assert first["filter_time_constant_s"] == pytest.approx(0.560735, rel=1e-5)
    assert "excludes actuator" in first["scope"]
    assert all(item["amplitude_ratio"] >= 0.0 for item in evidence["baseline"])


# **Validates: Requirements 2.3**
@pytest.mark.parametrize(
    ("low_tick", "guidance_ratio", "high_ratio", "ticks"),
    [
        (0.01, 1, 1, 4),
        (0.01, 2, 4, 60),
        (0.02, 3, 2, 31),
        (0.04, 1, 4, 17),
    ],
)
def test_integer_related_schedules_have_exact_counts_and_monotone_timestamps(
    low_tick, guidance_ratio, high_ratio, ticks
):
    schedule = MultiRateSchedule(
        high_level_period_s=low_tick * guidance_ratio * high_ratio,
        guidance_period_s=low_tick * guidance_ratio,
        low_level_period_s=low_tick,
    )
    duration = ticks * low_tick
    trace = DeterministicMultiRateScheduler(schedule).run(duration, lambda _: 0.0)
    assert len(trace.low_level_timestamps_s) == ticks
    assert trace.low_level_timestamps_s == sorted(trace.low_level_timestamps_s)
    assert all(time < duration for time in trace.guidance_timestamps_s)
    assert all(time < duration for time in trace.high_level_timestamps_s)
    assert len(trace.guidance_timestamps_s) == (ticks - 1) // guidance_ratio + 1
    assert len(trace.high_level_timestamps_s) == (ticks - 1) // (guidance_ratio * high_ratio) + 1


def test_f06_artifact_gate_is_blocked_and_hashes_architecture_specification():
    gate = json.loads((BUNDLE / "f06_evidence_gate.json").read_text(encoding="utf-8"))
    evidence = json.loads((BUNDLE / "f06_delay_phase_evidence.json").read_text(encoding="utf-8"))
    assert gate["status"] == "needs_more_evidence"
    assert gate["protected_path_allowlist"] == []
    assert gate["runtime_disposition"] == "preserve_legacy_single_rate"
    assert evidence["frozen_baseline"]["filter_time_constant_interpretation"] == "component only; not total end-to-end latency"
    assert hashlib.sha256((SPEC / "f06_multirate_architecture.md").read_bytes()).hexdigest() == gate["evidence_hashes"]["architecture_specification_sha256"]
