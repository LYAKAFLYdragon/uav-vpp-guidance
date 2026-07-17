from __future__ import annotations

import copy

from uav_vpp_guidance.evaluation.tactical_basis_lateral_runtime_probe_contract import (
    METHODS,
    OPPONENTS,
    summarize_runtime_records,
)


METRICS = {
    "response_window_steps": 40,
    "primary_pressure": {
        "candidate_minus_baseline_max": -0.05,
        "baseline_min_mean_target_attack_score": 0.01,
    },
    "secondary_geometry": {
        "range_rate_mean_min_mps": 5.0,
        "end_range_min_m": 50.0,
        "end_specific_energy_min_m": 25.0,
    },
    "execution_and_safety": {"ego_crash_oob_count_delta_vs_baseline_max": 0},
}

GATES = {
    "min_qualifying_pairs_per_required_opponent": 4,
    "min_distinct_signatures_per_required_opponent": 4,
    "min_pressure_exposed_pairs_per_required_opponent": 3,
}


def _record(method: str, opponent: str, index: int) -> dict:
    candidate = method == METHODS[1]
    window = [
        {
            "target_attack_score": 0.10 if candidate else 0.20,
            "range_rate_mps": -10.0 if candidate else -20.0,
            "range_m": 1060.0 if candidate else 1000.0,
            "own_specific_energy_m": 5100.0 if candidate else 5050.0,
        }
        for _ in range(40)
    ]
    return {
        "method": method,
        "opponent": opponent,
        "scenario_name": f"scenario-{index}",
        "scenario_signature": f"signature-{index}",
        "mirror_sign": "negative" if index % 2 == 0 else "positive",
        "program_window": window,
        "result": {
            "strict_jsbsim": True,
            "handoff_reached": True,
            "handoff_state_hash": f"handoff-{opponent}-{index}",
            "program_steps": 40,
            "program_complete": True,
            "no_reset_after_initialization": True,
            "no_backend_fallback": True,
            "no_prediction_fallback_in_program": True,
            "no_history_padding": True,
            "no_checkpoint_fallback": True,
            "continuity_hash_chain_complete": True,
            "finite_program_telemetry": True,
            "candidate_nonfinite_action_steps": 0,
            "candidate_box_limited_steps": 0,
            "candidate_slew_limited_steps": 0,
            "candidate_basis_violation_steps": 0,
            "terminal_reason": "timeout",
            "fatal_contract_violation": False,
        },
    }


def _records() -> list[dict]:
    return [
        _record(method, opponent, index)
        for method in METHODS
        for opponent in OPPONENTS
        for index in range(12)
    ]


def test_runtime_summary_requires_cross_opponent_paired_authority() -> None:
    result = summarize_runtime_records(_records(), scenarios_per_opponent=12, metrics=METRICS, gates=GATES)

    assert result["passes"] is True
    assert result["decision"] == "cross_opponent_authority_established"
    assert result["required_opponents_passed"] == ["end_to_end", "independent_ppo_vpp"]


def test_runtime_summary_freezes_on_candidate_clipping() -> None:
    records = copy.deepcopy(_records())
    candidate = next(record for record in records if record["method"] == METHODS[1])
    candidate["result"]["candidate_box_limited_steps"] = 1

    result = summarize_runtime_records(records, scenarios_per_opponent=12, metrics=METRICS, gates=GATES)

    assert result["passes"] is False
    assert result["decision"] == "runtime_contract_failure"


def test_runtime_summary_calls_zero_pressure_coverage_not_evaluable() -> None:
    records = _records()
    for record in records:
        for step in record["program_window"]:
            step["target_attack_score"] = 0.0

    result = summarize_runtime_records(records, scenarios_per_opponent=12, metrics=METRICS, gates=GATES)

    assert result["passes"] is False
    assert result["decision"] == "runtime_not_evaluable"
