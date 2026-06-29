from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analyze_directtrack2000_rearquarter_bias import (  # noqa: E402
    build_cross_split_contrast,
    build_executive_diagnosis,
    build_split_analysis,
)


def _row(
    seed: int,
    *,
    win: bool,
    loss: bool,
    termination_reason: str,
    damage_margin: float,
    post_adv: float,
    has_direct_track: bool,
    first_direct_altitude_m: float,
    vp_forward_bias_m: float,
    blend_active_fraction: float,
    direct_track_fraction: float,
    direct_track_first_step: float,
    first_active_step: float,
) -> dict:
    return {
        "seed": seed,
        "win": win,
        "loss": loss,
        "termination_reason": termination_reason,
        "damage_margin": damage_margin,
        "post_merge_attack_zone_advantage_s": post_adv,
        "direct_track_fraction": direct_track_fraction,
        "offblend_fraction": blend_active_fraction,
        "released_fraction": 0.7,
        "first_pass_step": 25.0,
        "has_direct_track": has_direct_track,
        "first_direct_step": direct_track_first_step,
        "first_direct_altitude_m": first_direct_altitude_m,
        "first_direct_range_m": 8400.0,
        "first_direct_range_rate_mps": 10.0,
        "below_2000_step": 440.0 if has_direct_track and first_direct_altitude_m < 2000.0 else float("nan"),
        "min_altitude_m": 500.0 if loss else 900.0,
        "final_altitude_m": 500.0 if loss else 1100.0,
        "steps": 500,
        "vp_forward_bias_m": vp_forward_bias_m,
        "vp_lateral_bias_m": 400.0 if win else 1800.0,
        "post_merge_offensive_anchor_blend_active_fraction": blend_active_fraction,
        "post_merge_offensive_anchor_first_active_step": first_active_step,
        "post_merge_offensive_anchor_blend_release_direct_track_active_fraction": direct_track_fraction,
        "post_merge_offensive_anchor_blend_release_direct_track_first_step": direct_track_first_step,
        "merge_min_range_m": 40.0 if has_direct_track else 31.0,
    }


def _geometry_summary(
    *,
    damage_margin: float,
    post_adv: float,
    alignment_disadvantage_fraction: float,
    blend_active_fraction: float,
    first_active_step: float,
    direct_track_fraction: float,
    vp_forward_bias_m: float,
    merge_min_range_m: float,
) -> dict:
    return {
        "first_pass_step": 25.0,
        "post_merge_attack_zone_advantage_s": post_adv,
        "damage_margin": damage_margin,
        "post_merge_offensive_anchor_condition_met_fraction": 0.12,
        "post_merge_offensive_anchor_alignment_disadvantage_fraction": alignment_disadvantage_fraction,
        "post_merge_offensive_anchor_blend_active_fraction": blend_active_fraction,
        "post_merge_offensive_anchor_first_active_step": first_active_step,
        "post_merge_offensive_anchor_blend_released_fraction": 0.7,
        "post_merge_offensive_anchor_blend_release_first_step": 160.0,
        "post_merge_offensive_anchor_blend_release_direct_track_active_fraction": direct_track_fraction,
        "post_merge_offensive_anchor_blend_release_direct_track_first_step": 445.0,
        "post_merge_predicted_target_forward_scale_active_fraction": 0.95,
        "post_merge_predicted_target_forward_scale_release_scale_active_fraction": 0.03,
        "vp_forward_bias_m": vp_forward_bias_m,
        "vp_lateral_bias_m": 450.0,
        "merge_min_range_m": merge_min_range_m,
    }


def test_build_split_analysis_identifies_pilot_bias_and_low_altitude_trap():
    formal_rows = [
        _row(
            100,
            win=True,
            loss=False,
            termination_reason="timeout_hp_advantage",
            damage_margin=18.0,
            post_adv=7.0,
            has_direct_track=True,
            first_direct_altitude_m=1600.0,
            vp_forward_bias_m=-3100.0,
            blend_active_fraction=0.09,
            direct_track_fraction=0.045,
            direct_track_first_step=462.0,
            first_active_step=27.0,
        ),
        _row(
            101,
            win=True,
            loss=False,
            termination_reason="timeout_hp_advantage",
            damage_margin=17.0,
            post_adv=6.8,
            has_direct_track=True,
            first_direct_altitude_m=1700.0,
            vp_forward_bias_m=-3050.0,
            blend_active_fraction=0.092,
            direct_track_fraction=0.046,
            direct_track_first_step=460.0,
            first_active_step=27.0,
        ),
        _row(
            102,
            win=True,
            loss=False,
            termination_reason="target_crash_or_out_of_bounds",
            damage_margin=16.5,
            post_adv=6.5,
            has_direct_track=False,
            first_direct_altitude_m=float("nan"),
            vp_forward_bias_m=-3150.0,
            blend_active_fraction=0.091,
            direct_track_fraction=0.0,
            direct_track_first_step=float("nan"),
            first_active_step=26.0,
        ),
        _row(
            103,
            win=False,
            loss=True,
            termination_reason="ego_crash_or_out_of_bounds",
            damage_margin=17.5,
            post_adv=7.1,
            has_direct_track=True,
            first_direct_altitude_m=1200.0,
            vp_forward_bias_m=-2900.0,
            blend_active_fraction=0.096,
            direct_track_fraction=0.026,
            direct_track_first_step=441.0,
            first_active_step=27.0,
        ),
        _row(
            104,
            win=False,
            loss=True,
            termination_reason="ego_crash_or_out_of_bounds",
            damage_margin=16.8,
            post_adv=6.9,
            has_direct_track=True,
            first_direct_altitude_m=1300.0,
            vp_forward_bias_m=-2920.0,
            blend_active_fraction=0.097,
            direct_track_fraction=0.025,
            direct_track_first_step=442.0,
            first_active_step=27.0,
        ),
        _row(
            105,
            win=False,
            loss=True,
            termination_reason="ego_crash_or_out_of_bounds",
            damage_margin=0.8,
            post_adv=0.3,
            has_direct_track=False,
            first_direct_altitude_m=float("nan"),
            vp_forward_bias_m=-7200.0,
            blend_active_fraction=0.133,
            direct_track_fraction=0.0,
            direct_track_first_step=float("nan"),
            first_active_step=26.0,
        ),
    ]
    pilot_rows = formal_rows[:3]

    split = build_split_analysis(
        pilot_rows=pilot_rows,
        formal_rows=formal_rows,
        pilot_geometry_summary=_geometry_summary(
            damage_margin=17.17,
            post_adv=6.77,
            alignment_disadvantage_fraction=0.09,
            blend_active_fraction=0.091,
            first_active_step=26.7,
            direct_track_fraction=0.046,
            vp_forward_bias_m=-3100.0,
            merge_min_range_m=36.7,
        ),
        formal_geometry_summary=_geometry_summary(
            damage_margin=11.1,
            post_adv=4.93,
            alignment_disadvantage_fraction=0.11,
            blend_active_fraction=0.11,
            first_active_step=26.5,
            direct_track_fraction=0.014,
            vp_forward_bias_m=-5000.0,
            merge_min_range_m=36.5,
        ),
        pilot_window_start_seed=100,
        direct_track_altitude_threshold_m=2000.0,
    )

    assert split["pilot_overview"]["win_rate"] == 1.0
    assert split["window_percentiles"]["win_rate_percentile"] == 1.0
    assert split["window_rarity"]["win_rate_at_or_above_target_count"] == 1
    assert split["window_rarity"]["median_window_win_rate"] == 0.5
    assert split["pilot_alignment_to_formal_outcomes"]["closer_to_wins_count"] == 6
    assert split["pilot_alignment_to_formal_outcomes"]["closer_to_losses_count"] == 0
    assert split["formal_modes"]["direct_losses"]["overview"]["n"] == 2
    assert split["formal_modes"]["no_direct_losses"]["overview"]["mean_damage_margin"] == 0.8
    assert split["channel_story"]["pilot_mix"]["direct_episode_count"] == 2
    assert split["channel_story"]["pilot_mix"]["no_direct_episode_count"] == 1
    assert split["channel_story"]["pilot_mix"]["members"][0]["seed"] == 100
    assert split["channel_story"]["direct_channel"]["formal_wins"]["overview"]["n"] == 2
    assert (
        split["channel_story"]["direct_channel"]["loss_minus_win"][
            "post_merge_offensive_anchor_blend_release_direct_track_active_fraction"
        ]
        < 0.0
    )
    assert (
        split["channel_story"]["direct_channel"]["loss_minus_win"][
            "post_merge_offensive_anchor_blend_release_direct_track_first_step"
        ]
        < 0.0
    )
    assert (
        split["channel_story"]["no_direct_channel"]["loss_minus_win"]["vp_forward_bias_m"]
        < 0.0
    )
    assert (
        split["channel_story"]["no_direct_channel"]["loss_minus_win"][
            "post_merge_offensive_anchor_blend_active_fraction"
        ]
        > 0.0
    )
    assert split["loss_taxonomy"]["direct_losses"]["share_of_losses"] == 2 / 3
    assert split["loss_taxonomy"]["no_direct_losses"]["mean_vp_forward_bias_m"] == -7200.0
    assert split["direct_track_low_altitude"]["below_altitude"]["overview"]["n"] == 4
    assert split["direct_track_low_altitude"]["below_altitude_loss_rate"] == 0.5
    assert split["direct_track_low_altitude"]["no_direct_loss_rate"] == 1 / 2

    diagnosis = build_executive_diagnosis(
        expert_split=split,
        contrast={
            "expert_vs_end_to_end_blend_active_ratio": 7.0,
            "end_to_end_minus_expert_first_active_step": 80.0,
            "expert_vp_forward_bias_m": -5000.0,
            "end_to_end_vp_forward_bias_m": -900.0,
        },
    )

    assert diagnosis["pilot_window_bias"]["window_count"] == 4
    assert diagnosis["pilot_alignment_bias"]["closer_to_wins_count"] == 6
    assert diagnosis["expert_failure_channels"]["direct_track_collapse_share_of_losses"] == 2 / 3
    assert diagnosis["expert_failure_channels"]["no_direct_overdeep_share_of_losses"] == 1 / 3
    assert diagnosis["expert_failure_channels"]["direct_track_collapse_positive_damage_margin_fraction"] == 1.0
    assert diagnosis["expert_failure_channels"]["direct_track_collapse_mean_final_altitude_m"] == 500.0
    assert diagnosis["expert_failure_channels"]["no_direct_overdeep_mean_damage_margin"] == 0.8
    assert diagnosis["end_to_end_guardrail"]["end_to_end_minus_expert_vp_forward_bias_m"] == 4100.0


def test_build_cross_split_contrast_shows_expert_anchor_is_more_active():
    expert_split = {
        "formal_overview": {
            "win_rate": 0.05,
            "mean_damage_margin": 9.8,
            "mean_post_merge_attack_zone_advantage_s": 3.9,
        },
        "formal_geometry": _geometry_summary(
            damage_margin=9.8,
            post_adv=3.9,
            alignment_disadvantage_fraction=0.12,
            blend_active_fraction=0.11,
            first_active_step=26.5,
            direct_track_fraction=0.014,
            vp_forward_bias_m=-4950.0,
            merge_min_range_m=36.7,
        ),
    }
    end_to_end_split = {
        "formal_overview": {
            "win_rate": 0.88,
            "mean_damage_margin": 7.8,
            "mean_post_merge_attack_zone_advantage_s": 3.1,
        },
        "formal_geometry": _geometry_summary(
            damage_margin=7.8,
            post_adv=3.1,
            alignment_disadvantage_fraction=0.0,
            blend_active_fraction=0.016,
            first_active_step=106.0,
            direct_track_fraction=0.115,
            vp_forward_bias_m=-875.0,
            merge_min_range_m=200.5,
        ),
    }

    contrast = build_cross_split_contrast(
        expert_split=expert_split,
        end_to_end_split=end_to_end_split,
    )

    assert contrast["expert_formal_win_rate"] == 0.05
    assert contrast["end_to_end_formal_win_rate"] == 0.88
    assert contrast["expert_vs_end_to_end_blend_active_ratio"] > 6.0
    assert contrast["end_to_end_minus_expert_first_active_step"] > 70.0
    assert contrast["expert_alignment_disadvantage_fraction"] == 0.12
    assert contrast["end_to_end_alignment_disadvantage_fraction"] == 0.0
