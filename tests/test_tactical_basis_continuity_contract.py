from __future__ import annotations

from uav_vpp_guidance.evaluation.raw_si_first_pass_event import RawSIFirstPassConfig
from uav_vpp_guidance.evaluation.tactical_basis_continuity_contract import (
    record_passes,
    summarize_records,
)


GATES = {
    "required_window_steps": 40,
    "max_basis_condition_number": 100.0,
    "min_qualifying_handoffs_per_opponent": 4,
    "min_distinct_signatures_per_opponent": 4,
    "opponents_required": 2,
}


def _step() -> dict:
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
        "dynamic_state": "disadvantage",
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
        "handoff_window": [_step() for _ in range(40)],
    }


def test_continuity_gate_passes_for_two_unpooled_opponents() -> None:
    records = [
        _record(opponent, index)
        for opponent in ("expert", "independent_ppo_vpp")
        for index in range(4)
    ]
    summary = summarize_records(records, GATES)
    assert summary["passes"] is True
    assert summary["passed_opponents"] == ["expert", "independent_ppo_vpp"]


def test_continuity_gate_rejects_legacy_action_semantics() -> None:
    record = _record("expert", 0)
    record["handoff_window"][0]["active_action_semantics"] = "cartesian_offset"
    assert record_passes(record, GATES) is False


def test_continuity_gate_rejects_prediction_fallback() -> None:
    record = _record("expert", 0)
    record["handoff_window"][3]["prediction_fallback"] = True
    assert record_passes(record, GATES) is False


def test_continuity_gate_rejects_insufficient_mirror_coverage() -> None:
    records = [_record("expert", index) for index in range(4)]
    for record in records:
        record["mirror_sign"] = "positive"
    assert summarize_records(records, GATES)["passes"] is False


def test_raw_si_contract_uses_the_runtime_dataclass_field_names() -> None:
    config = RawSIFirstPassConfig(
        closing_rate_threshold_mps=0.0,
        opening_rate_threshold_mps=0.0,
        minimum_closing_steps=3,
        minimum_opening_steps=2,
    )
    assert config.minimum_closing_steps == 3
