from __future__ import annotations

from uav_vpp_guidance.evaluation.thesis_temporal_corridor_feasibility import summarize_rows


def _row(candidate: str, opponent: str, pair: str, residual: float = 0.0) -> dict:
    return {
        "candidate_id": candidate,
        "opponent": opponent,
        "pair_key": pair,
        "projection_residual_norm_m": residual,
        "applied_action_norm": 0.5,
        "hard_bound_active": False,
    }


def test_temporal_corridor_gate_requires_two_separate_opponents():
    rows = []
    for opponent in ("expert", "independent_ppo_vpp"):
        for index in range(4):
            rows.append(_row("candidate", opponent, str(index)))
    summary = summarize_rows(rows, min_episodes=4, max_mean_residual_m=25.0, min_peak_action_norm=0.25)
    assert summary["passes"] is True


def test_temporal_corridor_gate_rejects_large_projection_residual():
    rows = []
    for opponent in ("expert", "independent_ppo_vpp"):
        for index in range(4):
            rows.append(_row("candidate", opponent, str(index), residual=30.0))
    summary = summarize_rows(rows, min_episodes=4, max_mean_residual_m=25.0, min_peak_action_norm=0.25)
    assert summary["passes"] is False
