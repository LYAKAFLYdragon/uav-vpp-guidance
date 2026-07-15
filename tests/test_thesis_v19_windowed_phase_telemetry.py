from __future__ import annotations

import pytest

from uav_vpp_guidance.evaluation.thesis_v19_windowed_phase_telemetry import SOURCE_ID, V19WindowedTelemetryError, summarize_records


def _step(*, ready: bool) -> dict:
    return {"base_observation_16d": [0.0] * 16, "phase": "re_entry", "first_pass_complete": True, "history_length": 10 if ready else 3, "prediction_valid": ready, "prediction_fallback": False, "own_specific_energy_j_per_kg": 1.0 if ready else None, "target_specific_energy_j_per_kg": 1.0 if ready else None, "actual_nz_g": 1.0 if ready else None}


def _record(opponent: str) -> dict:
    return {"source_id": SOURCE_ID, "opponent": opponent, "handoff_reached": True, "window_steps": 20, "steps": [_step(ready=False), _step(ready=True)] * 20}


def test_v19_keeps_warmup_separate_from_analysis_ready_gate():
    records = [_record(opponent) for opponent in ("expert", "end_to_end", "independent_ppo_vpp")]
    expected = {opponent: {"handoff_reached": 1, "valid_path_context_steps": 20} for opponent in ("expert", "end_to_end", "independent_ppo_vpp")}
    result = summarize_records(records, expected=expected, records_per_opponent=1, minimum_ready_steps=20)
    assert result["passes"] is True
    assert result["matrix"]["expert"]["window_steps"] == {"analysis_ready": 20, "warmup": 20}


def test_v19_rejects_fallback_even_during_warmup():
    records = [_record(opponent) for opponent in ("expert", "end_to_end", "independent_ppo_vpp")]
    records[0]["steps"][0]["prediction_fallback"] = True
    expected = {opponent: {"handoff_reached": 1, "valid_path_context_steps": 20} for opponent in ("expert", "end_to_end", "independent_ppo_vpp")}
    with pytest.raises(V19WindowedTelemetryError, match="fallback"):
        summarize_records(records, expected=expected, records_per_opponent=1, minimum_ready_steps=20)
