from __future__ import annotations

from uav_vpp_guidance.evaluation.pressure_exposed_postmerge_contract import (
    find_first_pressure_window,
    record_passes,
    summarize_records,
)


GATES = {
    "required_window_steps": 40,
    "max_basis_condition_number": 100.0,
    "target_attack_score_min": 0.1,
    "min_pressure_steps_per_window": 10,
    "min_qualifying_episodes_per_opponent": 4,
    "min_distinct_signatures_per_opponent": 4,
    "required_opponents": ["expert", "end_to_end", "independent_ppo_vpp"],
}


def _step(*, pressure: bool = True, state: str = "disadvantage") -> dict:
    return {
        "active_action_semantics": "tactical_basis_v1",
        "configured_action_semantics": "tactical_basis_v1",
        "tactical_basis_enabled": True,
        "backend": "jsbsim",
        "backend_fallback_occurred": False,
        "prediction_fallback": False,
        "history_padding_used": False,
        "checkpoint_fallback": False,
        "phase": "post_merge",
        "dynamic_state": state,
        "history_length": 10,
        "base_observation_16d": [0.0] * 16,
        "observation_66d": [0.0] * 66,
        "reference_action": [0.0, 0.1, -0.1],
        "tactical_basis_ll_world": [1200.0, 0.0, 0.0],
        "tactical_basis_io_world": [0.0, 1600.0, 0.0],
        "tactical_basis_cd_world": [0.0, 0.0, 600.0],
        "vpp_world_offset_m": [0.0, 0.0, 0.0],
        "basis_condition_number": 2.7,
        "basis_rank": 3,
        "target_attack_score": 0.25 if pressure else 0.0,
        "range_m": 1800.0,
        "range_rate_mps": 35.0,
        "aa_deg": 120.0,
        "ata_deg": 40.0,
        "own_specific_energy_m": 4200.0,
        "altitude_m": 5100.0,
    }


def _record(opponent: str, index: int) -> dict:
    return {
        "opponent": opponent,
        "scenario_signature": f"signature-{index}",
        "mirror_sign": "negative" if index % 2 == 0 else "positive",
        "result": {
            "strict_jsbsim": True,
            "handoff_reached": True,
            "window_complete": True,
            "no_reset_after_initialization": True,
            "no_backend_fallback": True,
            "no_prediction_fallback": True,
            "no_history_padding": True,
            "no_checkpoint_fallback": True,
            "continuity_hash_chain_complete": True,
        },
        "handoff_window": [_step(pressure=index < 3 or step < 10) for step in range(40)],
    }


def test_pressure_window_accepts_sustained_real_pressure() -> None:
    record = _record("expert", 0)
    assert record_passes(record, GATES) is True


def test_pressure_window_rejects_absent_pressure() -> None:
    record = _record("expert", 3)
    record["handoff_window"] = [_step(pressure=False) for _ in range(40)]
    assert record_passes(record, GATES) is False


def test_pressure_window_allows_geometry_to_become_neutral_after_handoff() -> None:
    steps = [_step() for _ in range(40)]
    steps[12]["dynamic_state"] = "neutral"
    assert find_first_pressure_window(steps, GATES) == steps


def test_pressure_window_rejects_premerge_gap() -> None:
    steps = [_step() for _ in range(40)]
    steps[20]["phase"] = "pre_merge"
    assert find_first_pressure_window(steps, GATES) is None


def test_unpooled_gate_requires_every_registered_opponent() -> None:
    records = [
        _record(opponent, index)
        for opponent in ("expert", "end_to_end", "independent_ppo_vpp")
        for index in range(4)
    ]
    summary = summarize_records(records, GATES)
    assert summary["passes"] is True
    records[-1]["handoff_window"] = [_step(pressure=False) for _ in range(40)]
    assert summarize_records(records, GATES)["passes"] is False
