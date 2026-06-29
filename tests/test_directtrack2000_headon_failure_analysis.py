from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analyze_directtrack2000_headon_failure import build_failure_report  # noqa: E402


def _row(
    seed: int,
    *,
    win: bool,
    loss: bool,
    termination_reason: str,
    damage_margin: float,
    post_adv: float,
    direct_track_fraction: float,
    has_direct_track: bool,
    first_direct_step: float,
    first_direct_altitude_m: float,
    below_2000_step: float,
    min_altitude_m: float,
    final_altitude_m: float,
) -> dict:
    return {
        "seed": seed,
        "win": win,
        "loss": loss,
        "termination_reason": termination_reason,
        "damage_margin": damage_margin,
        "post_merge_attack_zone_advantage_s": post_adv,
        "direct_track_fraction": direct_track_fraction,
        "offblend_fraction": 0.1,
        "released_fraction": 0.7,
        "first_pass_step": 25.0,
        "has_direct_track": has_direct_track,
        "first_direct_step": first_direct_step,
        "first_direct_altitude_m": first_direct_altitude_m,
        "first_direct_range_m": 8400.0,
        "first_direct_range_rate_mps": 10.0,
        "below_2000_step": below_2000_step,
        "min_altitude_m": min_altitude_m,
        "final_altitude_m": final_altitude_m,
        "steps": 500,
    }


def test_build_failure_report_captures_window_bias_and_crash_after_advantage():
    formal_rows = [
        _row(
            100,
            win=True,
            loss=False,
            termination_reason="target_crash_or_out_of_bounds",
            damage_margin=14.5,
            post_adv=5.8,
            direct_track_fraction=0.0,
            has_direct_track=False,
            first_direct_step=float("nan"),
            first_direct_altitude_m=float("nan"),
            below_2000_step=float("nan"),
            min_altitude_m=4300.0,
            final_altitude_m=6600.0,
        ),
        _row(
            101,
            win=True,
            loss=False,
            termination_reason="timeout_hp_advantage",
            damage_margin=18.0,
            post_adv=7.2,
            direct_track_fraction=0.04,
            has_direct_track=True,
            first_direct_step=495.0,
            first_direct_altitude_m=870.0,
            below_2000_step=444.0,
            min_altitude_m=754.0,
            final_altitude_m=797.0,
        ),
        _row(
            102,
            win=True,
            loss=False,
            termination_reason="timeout_hp_advantage",
            damage_margin=15.0,
            post_adv=6.0,
            direct_track_fraction=0.1,
            has_direct_track=True,
            first_direct_step=492.0,
            first_direct_altitude_m=900.0,
            below_2000_step=445.0,
            min_altitude_m=810.0,
            final_altitude_m=860.0,
        ),
        _row(
            103,
            win=False,
            loss=True,
            termination_reason="ego_crash_or_out_of_bounds",
            damage_margin=18.5,
            post_adv=7.4,
            direct_track_fraction=0.05,
            has_direct_track=True,
            first_direct_step=465.0,
            first_direct_altitude_m=1080.0,
            below_2000_step=442.0,
            min_altitude_m=488.0,
            final_altitude_m=488.0,
        ),
        _row(
            104,
            win=False,
            loss=True,
            termination_reason="ego_crash_or_out_of_bounds",
            damage_margin=1.0,
            post_adv=0.4,
            direct_track_fraction=0.0,
            has_direct_track=False,
            first_direct_step=float("nan"),
            first_direct_altitude_m=float("nan"),
            below_2000_step=float("nan"),
            min_altitude_m=4530.0,
            final_altitude_m=5290.0,
        ),
        _row(
            105,
            win=False,
            loss=True,
            termination_reason="timeout_hp_disadvantage",
            damage_margin=-13.5,
            post_adv=-5.4,
            direct_track_fraction=0.0,
            has_direct_track=False,
            first_direct_step=float("nan"),
            first_direct_altitude_m=float("nan"),
            below_2000_step=float("nan"),
            min_altitude_m=4800.0,
            final_altitude_m=5100.0,
        ),
    ]
    pilot_rows = formal_rows[:3]

    report = build_failure_report(
        formal_rows,
        pilot_rows,
        pilot_window_start_seed=100,
        legacy_window_size=4,
        exemplar_rows=[formal_rows[0], formal_rows[3]],
    )

    assert report["formal_overview"]["n"] == 6
    assert report["pilot_overview"]["win_rate"] == 1.0
    assert report["window_percentiles"]["pilot_window"]["win_rate_percentile"] == 1.0
    assert report["window_percentiles"]["pilot_window"]["win_rate_at_or_above_target_count"] == 1
    assert report["window_percentiles"]["pilot_window"]["median_win_rate"] == 0.5
    assert report["window_percentiles"]["legacy_window"]["target_window"]["win_rate"] == 0.75
    assert report["loss_retention"]["losses_positive_damage_margin"] == 2
    assert report["loss_retention"]["losses_positive_damage_margin_fraction"] == 2 / 3
    assert report["termination_reasons"]["ego_crash_or_out_of_bounds"] == 2
    assert report["direct_track_split"]["with_direct"]["n"] == 3
    assert report["direct_track_split"]["losses_with_direct"]["n"] == 1
    assert report["direct_track_split"]["no_direct_losses_below_2000_count"] == 0
    assert report["exemplars"]["100"]["termination_reason"] == "target_crash_or_out_of_bounds"
    assert report["exemplars"]["103"]["final_altitude_m"] == 488.0
