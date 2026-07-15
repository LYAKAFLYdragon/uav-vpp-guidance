from __future__ import annotations

from pathlib import Path

from scripts.build_thesis_global_advantage_p2_capability_cards import (
    GRAVITY_MPS2,
    specific_energy_j_per_kg,
    summarize_cards,
)


def _record(opponent: str, index: int) -> dict:
    return {
        "opponent": opponent,
        "scenario_signature": f"{opponent}-{index}",
        "target_speed_mean_mps": 200.0,
        "target_altitude_mean_m": 5000.0,
        "target_specific_energy_mean_j_per_kg": 69033.25,
        "ego_attack_zone_fraction": 0.2,
        "target_attack_zone_fraction": 0.1,
        "phase_fraction": {"pre_merge": 0.2, "post_merge": 0.7, "re_entry": 0.1},
        "first_pass_reached": True,
        "first_pass_step": 12,
        "terminal_reason": "timeout",
        "ego_crash_or_oob": False,
        "target_crash_or_oob": False,
    }


def test_specific_energy_uses_recorded_si_speed_and_altitude():
    assert specific_energy_j_per_kg({"altitude_m": 100.0, "speed_mps": 20.0}) == (
        GRAVITY_MPS2 * 100.0 + 200.0
    )


def test_capability_summary_requires_separate_thirty_episode_cards():
    records = [
        _record(opponent, index)
        for opponent in ("expert", "end_to_end", "independent_ppo_vpp")
        for index in range(30)
    ]
    cards = summarize_cards(records)
    assert set(cards) == {"expert", "end_to_end", "independent_ppo_vpp"}
    assert all(card["episode_count"] == 30 for card in cards.values())
    assert all(card["terminal_reason_counts"] == {"timeout": 30} for card in cards.values())
