from __future__ import annotations

from uav_vpp_guidance.evaluation.tactical_basis_postmerge_p2 import (
    candidate_variants,
    summarize_rows,
)


GATES = {
    "required_window_steps": 40,
    "required_opponents": ["end_to_end", "independent_ppo_vpp"],
    "diagnostic_only_opponents": ["expert"],
    "min_states_per_required_opponent": 4,
    "min_distinct_signatures_per_required_opponent": 4,
    "max_action_box_limited_fraction": 0.0,
    "max_slew_limited_fraction": 0.0,
    "max_mean_projection_residual_m": 50.0,
    "min_peak_actual_reference_delta_m": 100.0,
    "max_basis_condition_number": 100.0,
}


REFERENCE = {
    "templates": {
        "lateral_extension": {
            "peak_action_delta": [0.20, 0.32, 0.00],
            "lateral_sign_candidates": [-1, 1],
            "vertical_sign_candidates": [1],
        },
        "vertical_extension": {
            "peak_action_delta": [0.18, 0.00, 0.28],
            "lateral_sign_candidates": [1],
            "vertical_sign_candidates": [-1, 1],
        },
        "combined_break_extension": {
            "peak_action_delta": [0.20, 0.24, 0.22],
            "lateral_sign_candidates": [-1, 1],
            "vertical_sign_candidates": [-1, 1],
        },
    }
}


def _row(opponent: str, index: int) -> dict:
    return {
        "candidate_variant": "lateral_extension:lat+1:vert+1",
        "opponent": opponent,
        "scenario_signature": f"signature-{index}",
        "mirror_sign": "negative" if index % 2 == 0 else "positive",
        "data_contract_qualified": True,
        "strict_jsbsim": True,
        "baseline_steps": 40,
        "backend_fallback": False,
        "prediction_fallback": False,
        "action_box_limited_fraction": 0.0,
        "slew_limited_fraction": 0.0,
        "mean_projection_residual_m": 10.0,
        "peak_actual_reference_delta_m": 150.0,
        "max_basis_condition_number": 3.0,
    }


def test_p2_expands_only_the_eight_frozen_p1_variants() -> None:
    variants = candidate_variants(REFERENCE)

    assert [variant["candidate_variant"] for variant in variants] == [
        "combined_break_extension:lat-1:vert-1",
        "combined_break_extension:lat-1:vert+1",
        "combined_break_extension:lat+1:vert-1",
        "combined_break_extension:lat+1:vert+1",
        "lateral_extension:lat-1:vert+1",
        "lateral_extension:lat+1:vert+1",
        "vertical_extension:lat+1:vert-1",
        "vertical_extension:lat+1:vert+1",
    ]


def test_p2_selects_variant_only_when_both_required_opponents_pass() -> None:
    rows = [
        _row(opponent, index)
        for opponent in ("end_to_end", "independent_ppo_vpp")
        for index in range(4)
    ]
    summary = summarize_rows(rows, GATES)
    assert summary["passes"] is True
    assert summary["selected_variant"] == "lateral_extension:lat+1:vert+1"
    assert summary["matrix"]["lateral_extension:lat+1:vert+1"]["expert"]["gate_role"] == "diagnostic_only"


def test_p2_rejects_slew_limited_rows() -> None:
    rows = [
        _row(opponent, index)
        for opponent in ("end_to_end", "independent_ppo_vpp")
        for index in range(4)
    ]
    rows[0]["slew_limited_fraction"] = 0.025
    assert summarize_rows(rows, GATES)["passes"] is False
