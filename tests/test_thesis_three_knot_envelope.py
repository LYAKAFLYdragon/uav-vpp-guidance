from __future__ import annotations

from uav_vpp_guidance.evaluation.thesis_three_knot_envelope import summarize_rows


def _row(candidate: str, opponent: str, pair: str, clipped: bool) -> dict:
    return {
        "candidate_id": candidate,
        "opponent": opponent,
        "pair_key": pair,
        "clipped": clipped,
        "clipped_axes": [clipped, False, False],
    }


def test_three_knot_gate_requires_two_unpooled_opponents():
    rows = []
    for opponent in ("expert", "independent_ppo_vpp"):
        for episode in range(4):
            rows.append(_row("candidate", opponent, str(episode), False))
    rows.append(_row("candidate", "end_to_end", "x", True))
    summary = summarize_rows(rows, min_fully_unclipped_episodes=4, min_unclipped_step_fraction=0.9)
    assert summary["passes"] is True
    assert summary["passing_candidates"] == ["candidate"]


def test_three_knot_gate_rejects_clipped_candidate():
    rows = []
    for opponent in ("expert", "independent_ppo_vpp"):
        for episode in range(4):
            rows.append(_row("candidate", opponent, str(episode), True))
    summary = summarize_rows(rows, min_fully_unclipped_episodes=4, min_unclipped_step_fraction=0.9)
    assert summary["passes"] is False
