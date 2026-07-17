from __future__ import annotations

from uav_vpp_guidance.evaluation.temporal_reference_p2 import summarize_rows


GATES = {
    "min_baseline_steps_per_state": 40,
    "max_action_box_limited_fraction": 0.0,
    "max_mean_projection_residual_m": 50.0,
    "min_peak_actual_reference_delta_m": 100.0,
    "min_states_per_opponent": 4,
    "min_distinct_signatures_per_opponent": 4,
    "min_mirror_signs_per_opponent": 2,
    "candidate_opponents_required": 2,
}


def _row(variant: str, opponent: str, index: int) -> dict:
    return {
        "candidate_variant": variant,
        "opponent": opponent,
        "scenario_signature": f"{opponent}:{index}",
        "mirror_sign": "negative" if index % 2 == 0 else "positive",
        "replay_matched": True,
        "baseline_complete": True,
        "strict_jsbsim": True,
        "backend_fallback": False,
        "prediction_fallback": False,
        "baseline_steps": 40,
        "action_box_limited_fraction": 0.0,
        "mean_projection_residual_m": 1.0,
        "peak_actual_reference_delta_m": 120.0,
    }


def test_p2_requires_two_unpooled_opponents_with_four_distinct_mirrored_states() -> None:
    rows = [
        _row("lateral_extension:lat+1:vert+1", opponent, index)
        for opponent in ("expert", "independent_ppo_vpp")
        for index in range(4)
    ]
    summary = summarize_rows(rows, GATES)
    assert summary["passes"] is True
    assert summary["compiler_feasible_variants"] == ["lateral_extension:lat+1:vert+1"]


def test_p2_rejects_action_box_limited_rows() -> None:
    rows = [
        _row("vertical_extension:lat+1:vert+1", opponent, index)
        for opponent in ("expert", "independent_ppo_vpp")
        for index in range(4)
    ]
    rows[0]["action_box_limited_fraction"] = 0.1
    assert summarize_rows(rows, GATES)["passes"] is False
