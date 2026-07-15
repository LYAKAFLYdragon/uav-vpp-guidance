from __future__ import annotations

import pytest

from uav_vpp_guidance.evaluation.thesis_v18_phase_telemetry_audit import (
    SOURCE_ID,
    V18PhaseTelemetryError,
    summarize_records,
)


def _step() -> dict:
    return {
        "base_observation_16d": [0.0] * 16,
        "raw_range_m": 1000.0, "raw_range_rate_mps": -20.0,
        "ata_rad": 0.1, "aa_rad": 0.2, "los_azimuth_rad": 0.3, "los_elevation_rad": 0.4,
        "own_speed_mps": 250.0, "target_speed_mps": 240.0,
        "own_altitude_m": 5000.0, "target_altitude_m": 4900.0,
        "own_specific_energy_j_per_kg": 80000.0, "target_specific_energy_j_per_kg": 76000.0,
        "dynamic_taxonomy": "head_on", "phase": "re_entry", "first_pass_complete": True,
        "vpp_world_offset_m": [1.0, 2.0, 3.0], "guidance_command_normalized": [0.1, 0.2, 0.3],
        "actual_nz_g": 1.1, "saturation_flags": {"nz": False, "roll_rate": False, "throttle": False},
    }


def _record(opponent: str, key: str, *, handoff: bool = True) -> dict:
    return {"source_id": SOURCE_ID, "opponent": opponent, "pair_key": key, "handoff_reached": handoff, "window_steps": 20 if handoff else 0, "steps": [_step()]}


def test_v18_summary_requires_exact_per_opponent_v17_replay():
    records = [_record(opponent, opponent) for opponent in ("expert", "end_to_end", "independent_ppo_vpp")]
    expected = {opponent: {"handoff_reached": 1, "valid_path_context_steps": 20} for opponent in ("expert", "end_to_end", "independent_ppo_vpp")}
    summary = summarize_records(records, expected=expected, records_per_opponent=1)
    assert summary["passes"] is True


def test_v18_rejects_incomplete_telemetry_and_replay_mismatch():
    bad = _record("expert", "bad")
    del bad["steps"][0]["target_speed_mps"]
    with pytest.raises(V18PhaseTelemetryError, match="missing"):
        summarize_records([bad], expected={opponent: {"handoff_reached": 0, "valid_path_context_steps": 0} for opponent in ("expert", "end_to_end", "independent_ppo_vpp")}, records_per_opponent=1)
    records = [_record(opponent, opponent) for opponent in ("expert", "end_to_end", "independent_ppo_vpp")]
    expected = {opponent: {"handoff_reached": 0, "valid_path_context_steps": 0} for opponent in ("expert", "end_to_end", "independent_ppo_vpp")}
    assert summarize_records(records, expected=expected, records_per_opponent=1)["passes"] is False
