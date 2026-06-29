#!/usr/bin/env python3
"""Explain why the directtrack2000 rear-quarter lane looked good in pilot but failed in expert held-out."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from analyze_directtrack2000_headon_failure import (
    compute_window_percentiles,
    episode_to_row,
    load_episodes,
    summarize_rows,
)


PILOT_ALIGNMENT_FEATURES = [
    "damage_margin",
    "post_merge_attack_zone_advantage_s",
    "vp_forward_bias_m",
    "post_merge_offensive_anchor_blend_active_fraction",
    "post_merge_offensive_anchor_blend_release_direct_track_active_fraction",
    "post_merge_offensive_anchor_blend_release_direct_track_first_step",
]

GEOMETRY_MEAN_KEYS = [
    "vp_forward_bias_m",
    "vp_lateral_bias_m",
    "post_merge_offensive_anchor_blend_active_fraction",
    "post_merge_offensive_anchor_first_active_step",
    "post_merge_offensive_anchor_blend_release_direct_track_active_fraction",
    "post_merge_offensive_anchor_blend_release_direct_track_first_step",
    "merge_min_range_m",
    "post_merge_attack_zone_advantage_s",
    "damage_margin",
]

AGGREGATE_GEOMETRY_KEYS = [
    "first_pass_step",
    "post_merge_attack_zone_advantage_s",
    "damage_margin",
    "post_merge_offensive_anchor_condition_met_fraction",
    "post_merge_offensive_anchor_alignment_disadvantage_fraction",
    "post_merge_offensive_anchor_blend_active_fraction",
    "post_merge_offensive_anchor_first_active_step",
    "post_merge_offensive_anchor_blend_released_fraction",
    "post_merge_offensive_anchor_blend_release_first_step",
    "post_merge_offensive_anchor_blend_release_direct_track_active_fraction",
    "post_merge_offensive_anchor_blend_release_direct_track_first_step",
    "post_merge_predicted_target_forward_scale_active_fraction",
    "post_merge_predicted_target_forward_scale_release_scale_active_fraction",
    "vp_forward_bias_m",
    "vp_lateral_bias_m",
    "merge_min_range_m",
]

DIRECT_CHANNEL_KEYS = [
    "damage_margin",
    "post_merge_attack_zone_advantage_s",
    "post_merge_offensive_anchor_blend_release_direct_track_active_fraction",
    "post_merge_offensive_anchor_blend_release_direct_track_first_step",
    "first_direct_altitude_m",
    "final_altitude_m",
]

NO_DIRECT_CHANNEL_KEYS = [
    "damage_margin",
    "post_merge_attack_zone_advantage_s",
    "vp_forward_bias_m",
    "vp_lateral_bias_m",
    "post_merge_offensive_anchor_blend_active_fraction",
    "final_altitude_m",
]

PILOT_MEMBER_KEYS = [
    "seed",
    "has_direct_track",
    "termination_reason",
    "damage_margin",
    "post_merge_attack_zone_advantage_s",
    "vp_forward_bias_m",
    "vp_lateral_bias_m",
    "post_merge_offensive_anchor_blend_active_fraction",
    "post_merge_offensive_anchor_blend_release_direct_track_active_fraction",
    "post_merge_offensive_anchor_blend_release_direct_track_first_step",
    "first_direct_altitude_m",
    "final_altitude_m",
]


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _finite_mean(values: Iterable[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return float("nan")
    return sum(finite) / len(finite)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(denominator) or abs(denominator) < 1e-12:
        return float("inf") if math.isfinite(numerator) and numerator > 0.0 else float("nan")
    return numerator / denominator


def _safe_delta(lhs: float, rhs: float) -> float:
    if not math.isfinite(lhs) or not math.isfinite(rhs):
        return float("nan")
    return lhs - rhs


def _fraction(rows: Sequence[Mapping[str, Any]], predicate) -> float:
    if not rows:
        return float("nan")
    return sum(1 for row in rows if predicate(row)) / len(rows)


def _load_combat_geometry_payload(run_dir: Path) -> Dict[str, Any]:
    payload_path = run_dir / "aggregate" / "combat_geometry_diagnostics.json"
    if not payload_path.exists():
        raise FileNotFoundError(f"Combat diagnostics not found: {payload_path}")
    return json.loads(payload_path.read_text(encoding="utf-8"))


def _select_geometry_summary_row(
    payload: Mapping[str, Any],
    *,
    task: str,
    method: str,
) -> Dict[str, Any]:
    matches = [
        row
        for row in payload.get("rows", [])
        if str(row.get("task")) == task and str(row.get("method")) == method
    ]
    if not matches:
        raise ValueError(f"No combat geometry summary row for task={task!r}, method={method!r}")
    if len(matches) > 1:
        raise ValueError(f"Expected one combat geometry summary row, found {len(matches)}")
    return dict(matches[0])


def _select_geometry_episode_rows(
    payload: Mapping[str, Any],
    *,
    task: str,
    method: str,
) -> List[Dict[str, Any]]:
    rows = [
        dict(row)
        for row in payload.get("episode_rows", [])
        if str(row.get("task")) == task and str(row.get("method")) == method
    ]
    if not rows:
        raise ValueError(f"No combat geometry episode rows for task={task!r}, method={method!r}")
    return rows


def load_joined_rows(run_dir: Path, *, task: str, method: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    raw_rows = [episode_to_row(episode) for episode in load_episodes(run_dir, task, method)]
    payload = _load_combat_geometry_payload(run_dir)
    geometry_summary = _select_geometry_summary_row(payload, task=task, method=method)
    geometry_episode_rows = _select_geometry_episode_rows(payload, task=task, method=method)
    geometry_by_seed = {
        int(row.get("seed", -1)): row
        for row in geometry_episode_rows
    }

    joined_rows: List[Dict[str, Any]] = []
    for raw_row in raw_rows:
        seed = int(raw_row.get("seed", -1))
        merged = dict(raw_row)
        merged.update(geometry_by_seed.get(seed, {}))
        joined_rows.append(merged)
    return joined_rows, geometry_summary


def summarize_mode(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "overview": summarize_rows(rows),
        "geometry_means": {
            key: _finite_mean(_safe_float(row.get(key)) for row in rows)
            for key in GEOMETRY_MEAN_KEYS
        },
    }


def build_metric_means(
    rows: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
) -> Dict[str, float]:
    return {
        key: _finite_mean(_safe_float(row.get(key)) for row in rows)
        for key in keys
    }


def build_mean_deltas(
    *,
    lhs_rows: Sequence[Mapping[str, Any]],
    rhs_rows: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
) -> Dict[str, float]:
    lhs_means = build_metric_means(lhs_rows, keys)
    rhs_means = build_metric_means(rhs_rows, keys)
    return {
        key: _safe_delta(_safe_float(lhs_means.get(key)), _safe_float(rhs_means.get(key)))
        for key in keys
    }


def build_pilot_mix(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    sorted_rows = sorted(rows, key=lambda row: int(row.get("seed", -1)))
    direct_rows = [row for row in sorted_rows if bool(row.get("has_direct_track"))]
    no_direct_rows = [row for row in sorted_rows if not bool(row.get("has_direct_track"))]
    return {
        "episodes": len(sorted_rows),
        "direct_episode_count": len(direct_rows),
        "no_direct_episode_count": len(no_direct_rows),
        "direct_episode_fraction": _fraction(sorted_rows, lambda row: bool(row.get("has_direct_track"))),
        "members": [
            {key: row.get(key) for key in PILOT_MEMBER_KEYS}
            for row in sorted_rows
        ],
    }


def build_channel_story(
    *,
    pilot_rows: Sequence[Mapping[str, Any]],
    formal_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    direct_wins = [
        row for row in formal_rows if bool(row.get("win")) and bool(row.get("has_direct_track"))
    ]
    direct_losses = [
        row for row in formal_rows if bool(row.get("loss")) and bool(row.get("has_direct_track"))
    ]
    no_direct_wins = [
        row for row in formal_rows if bool(row.get("win")) and not bool(row.get("has_direct_track"))
    ]
    no_direct_losses = [
        row for row in formal_rows if bool(row.get("loss")) and not bool(row.get("has_direct_track"))
    ]
    pilot_direct_rows = [row for row in pilot_rows if bool(row.get("has_direct_track"))]
    pilot_no_direct_rows = [row for row in pilot_rows if not bool(row.get("has_direct_track"))]

    return {
        "pilot_mix": build_pilot_mix(pilot_rows),
        "direct_channel": {
            "pilot": summarize_mode(pilot_direct_rows),
            "formal_wins": summarize_mode(direct_wins),
            "formal_losses": summarize_mode(direct_losses),
            "loss_minus_win": build_mean_deltas(
                lhs_rows=direct_losses,
                rhs_rows=direct_wins,
                keys=DIRECT_CHANNEL_KEYS,
            ),
        },
        "no_direct_channel": {
            "pilot": summarize_mode(pilot_no_direct_rows),
            "formal_wins": summarize_mode(no_direct_wins),
            "formal_losses": summarize_mode(no_direct_losses),
            "loss_minus_win": build_mean_deltas(
                lhs_rows=no_direct_losses,
                rhs_rows=no_direct_wins,
                keys=NO_DIRECT_CHANNEL_KEYS,
            ),
        },
    }


def build_feature_closeness(
    pilot_rows: Sequence[Mapping[str, Any]],
    win_rows: Sequence[Mapping[str, Any]],
    loss_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    closer_to_wins_count = 0
    closer_to_losses_count = 0
    ties = 0

    for feature in PILOT_ALIGNMENT_FEATURES:
        pilot_mean = _finite_mean(_safe_float(row.get(feature)) for row in pilot_rows)
        win_mean = _finite_mean(_safe_float(row.get(feature)) for row in win_rows)
        loss_mean = _finite_mean(_safe_float(row.get(feature)) for row in loss_rows)
        if not all(math.isfinite(value) for value in (pilot_mean, win_mean, loss_mean)):
            continue

        win_gap = abs(pilot_mean - win_mean)
        loss_gap = abs(pilot_mean - loss_mean)
        if win_gap < loss_gap:
            closer_to = "wins"
            closer_to_wins_count += 1
        elif loss_gap < win_gap:
            closer_to = "losses"
            closer_to_losses_count += 1
        else:
            closer_to = "tie"
            ties += 1

        rows.append(
            {
                "feature": feature,
                "pilot_mean": pilot_mean,
                "formal_win_mean": win_mean,
                "formal_loss_mean": loss_mean,
                "pilot_to_win_gap": win_gap,
                "pilot_to_loss_gap": loss_gap,
                "closer_to": closer_to,
            }
        )

    return {
        "features": rows,
        "closer_to_wins_count": closer_to_wins_count,
        "closer_to_losses_count": closer_to_losses_count,
        "ties": ties,
        "compared_feature_count": len(rows),
    }


def build_low_altitude_direct_track_summary(
    formal_rows: Sequence[Mapping[str, Any]],
    *,
    altitude_threshold_m: float,
) -> Dict[str, Any]:
    direct_rows = [row for row in formal_rows if bool(row.get("has_direct_track"))]
    below_rows = [
        row
        for row in direct_rows
        if _safe_float(row.get("first_direct_altitude_m")) < altitude_threshold_m
    ]
    above_rows = [
        row
        for row in direct_rows
        if _safe_float(row.get("first_direct_altitude_m")) >= altitude_threshold_m
    ]
    no_direct_rows = [row for row in formal_rows if not bool(row.get("has_direct_track"))]

    def _loss_fraction(rows: Sequence[Mapping[str, Any]]) -> float:
        return _fraction(rows, lambda row: bool(row.get("loss")))

    def _ego_crash_fraction(rows: Sequence[Mapping[str, Any]]) -> float:
        return _fraction(
            rows,
            lambda row: str(row.get("termination_reason")) == "ego_crash_or_out_of_bounds",
        )

    return {
        "altitude_threshold_m": altitude_threshold_m,
        "all_direct": summarize_mode(direct_rows),
        "below_altitude": summarize_mode(below_rows),
        "above_altitude": summarize_mode(above_rows),
        "no_direct": summarize_mode(no_direct_rows),
        "all_direct_loss_rate": _loss_fraction(direct_rows),
        "below_altitude_loss_rate": _loss_fraction(below_rows),
        "above_altitude_loss_rate": _loss_fraction(above_rows),
        "no_direct_loss_rate": _loss_fraction(no_direct_rows),
        "below_altitude_ego_crash_rate": _ego_crash_fraction(below_rows),
        "no_direct_ego_crash_rate": _ego_crash_fraction(no_direct_rows),
    }


def build_window_rarity(window_percentiles: Mapping[str, Any]) -> Dict[str, Any]:
    target_window = window_percentiles["target_window"]
    return {
        "window_size": int(window_percentiles["window_size"]),
        "window_count": int(window_percentiles["window_count"]),
        "target_start_seed": int(target_window["start_seed"]),
        "target_end_seed": int(target_window["end_seed"]),
        "pilot_win_rate": _safe_float(target_window.get("win_rate")),
        "pilot_mean_damage_margin": _safe_float(target_window.get("mean_damage_margin")),
        "pilot_mean_post_merge_advantage_s": _safe_float(
            target_window.get("mean_post_merge_attack_zone_advantage_s")
        ),
        "median_window_win_rate": _safe_float(window_percentiles.get("median_win_rate")),
        "best_window_win_rate": _safe_float(window_percentiles.get("best_win_rate")),
        "win_rate_at_or_above_target_count": int(
            window_percentiles.get("win_rate_at_or_above_target_count", 0)
        ),
        "win_rate_at_or_above_target_fraction": _safe_float(
            window_percentiles.get("win_rate_at_or_above_target_fraction")
        ),
        "mean_damage_margin_at_or_above_target_count": int(
            window_percentiles.get("mean_damage_margin_at_or_above_target_count", 0)
        ),
        "mean_damage_margin_at_or_above_target_fraction": _safe_float(
            window_percentiles.get("mean_damage_margin_at_or_above_target_fraction")
        ),
        "mean_post_advantage_at_or_above_target_count": int(
            window_percentiles.get("mean_post_advantage_at_or_above_target_count", 0)
        ),
        "mean_post_advantage_at_or_above_target_fraction": _safe_float(
            window_percentiles.get("mean_post_advantage_at_or_above_target_fraction")
        ),
        "ego_crash_rate_at_or_below_target_count": int(
            window_percentiles.get("ego_crash_rate_at_or_below_target_count", 0)
        ),
        "ego_crash_rate_at_or_below_target_fraction": _safe_float(
            window_percentiles.get("ego_crash_rate_at_or_below_target_fraction")
        ),
    }


def build_loss_taxonomy(
    *,
    losses: Sequence[Mapping[str, Any]],
    direct_losses: Sequence[Mapping[str, Any]],
    no_direct_losses: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    total_losses = len(losses)

    def _channel(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        return {
            "n": len(rows),
            "share_of_losses": (len(rows) / total_losses) if total_losses else float("nan"),
            "win_rate": _fraction(rows, lambda row: bool(row.get("win"))),
            "ego_crash_rate": _fraction(
                rows,
                lambda row: str(row.get("termination_reason")) == "ego_crash_or_out_of_bounds",
            ),
            "mean_damage_margin": _finite_mean(
                _safe_float(row.get("damage_margin")) for row in rows
            ),
            "mean_post_merge_advantage_s": _finite_mean(
                _safe_float(row.get("post_merge_attack_zone_advantage_s")) for row in rows
            ),
            "mean_vp_forward_bias_m": _finite_mean(
                _safe_float(row.get("vp_forward_bias_m")) for row in rows
            ),
            "mean_direct_track_fraction": _finite_mean(
                _safe_float(row.get("post_merge_offensive_anchor_blend_release_direct_track_active_fraction"))
                for row in rows
            ),
            "mean_final_altitude_m": _finite_mean(
                _safe_float(row.get("final_altitude_m")) for row in rows
            ),
        }

    return {
        "direct_losses": _channel(direct_losses),
        "no_direct_losses": _channel(no_direct_losses),
    }


def extract_geometry_snapshot(summary_row: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: summary_row.get(key) for key in AGGREGATE_GEOMETRY_KEYS}


def build_split_analysis(
    *,
    pilot_rows: Sequence[Mapping[str, Any]],
    formal_rows: Sequence[Mapping[str, Any]],
    pilot_geometry_summary: Mapping[str, Any],
    formal_geometry_summary: Mapping[str, Any],
    pilot_window_start_seed: Optional[int] = None,
    direct_track_altitude_threshold_m: float = 2000.0,
) -> Dict[str, Any]:
    sorted_pilot = sorted(pilot_rows, key=lambda row: int(row.get("seed", -1)))
    sorted_formal = sorted(formal_rows, key=lambda row: int(row.get("seed", -1)))
    wins = [row for row in sorted_formal if bool(row.get("win"))]
    losses = [row for row in sorted_formal if bool(row.get("loss"))]
    direct_losses = [row for row in losses if bool(row.get("has_direct_track"))]
    no_direct_losses = [row for row in losses if not bool(row.get("has_direct_track"))]
    pilot_start_seed = (
        int(pilot_window_start_seed)
        if pilot_window_start_seed is not None
        else int(sorted_pilot[0]["seed"])
    )
    window_percentiles = compute_window_percentiles(
        sorted_formal,
        target_start_seed=pilot_start_seed,
        window_size=len(sorted_pilot),
    )

    return {
        "pilot_overview": summarize_rows(sorted_pilot),
        "formal_overview": summarize_rows(sorted_formal),
        "window_percentiles": window_percentiles,
        "window_rarity": build_window_rarity(window_percentiles),
        "pilot_alignment_to_formal_outcomes": build_feature_closeness(
            sorted_pilot,
            wins,
            losses,
        ),
        "pilot_geometry": extract_geometry_snapshot(pilot_geometry_summary),
        "formal_geometry": extract_geometry_snapshot(formal_geometry_summary),
        "formal_modes": {
            "wins": summarize_mode(wins),
            "losses": summarize_mode(losses),
            "direct_losses": summarize_mode(direct_losses),
            "no_direct_losses": summarize_mode(no_direct_losses),
        },
        "channel_story": build_channel_story(
            pilot_rows=sorted_pilot,
            formal_rows=sorted_formal,
        ),
        "loss_taxonomy": build_loss_taxonomy(
            losses=losses,
            direct_losses=direct_losses,
            no_direct_losses=no_direct_losses,
        ),
        "direct_track_low_altitude": build_low_altitude_direct_track_summary(
            sorted_formal,
            altitude_threshold_m=direct_track_altitude_threshold_m,
        ),
    }


def build_cross_split_contrast(
    *,
    expert_split: Mapping[str, Any],
    end_to_end_split: Mapping[str, Any],
) -> Dict[str, Any]:
    expert_formal_geometry = expert_split["formal_geometry"]
    e2e_formal_geometry = end_to_end_split["formal_geometry"]

    expert_blend = _safe_float(
        expert_formal_geometry.get("post_merge_offensive_anchor_blend_active_fraction")
    )
    e2e_blend = _safe_float(
        e2e_formal_geometry.get("post_merge_offensive_anchor_blend_active_fraction")
    )
    expert_first_active = _safe_float(
        expert_formal_geometry.get("post_merge_offensive_anchor_first_active_step")
    )
    e2e_first_active = _safe_float(
        e2e_formal_geometry.get("post_merge_offensive_anchor_first_active_step")
    )
    expert_alignment = _safe_float(
        expert_formal_geometry.get("post_merge_offensive_anchor_alignment_disadvantage_fraction")
    )
    e2e_alignment = _safe_float(
        e2e_formal_geometry.get("post_merge_offensive_anchor_alignment_disadvantage_fraction")
    )

    return {
        "expert_formal_win_rate": expert_split["formal_overview"]["win_rate"],
        "end_to_end_formal_win_rate": end_to_end_split["formal_overview"]["win_rate"],
        "expert_formal_damage_margin": expert_split["formal_overview"]["mean_damage_margin"],
        "end_to_end_formal_damage_margin": end_to_end_split["formal_overview"]["mean_damage_margin"],
        "expert_formal_post_merge_advantage_s": expert_split["formal_overview"][
            "mean_post_merge_attack_zone_advantage_s"
        ],
        "end_to_end_formal_post_merge_advantage_s": end_to_end_split["formal_overview"][
            "mean_post_merge_attack_zone_advantage_s"
        ],
        "expert_alignment_disadvantage_fraction": expert_alignment,
        "end_to_end_alignment_disadvantage_fraction": e2e_alignment,
        "expert_blend_active_fraction": expert_blend,
        "end_to_end_blend_active_fraction": e2e_blend,
        "expert_vs_end_to_end_blend_active_ratio": _safe_ratio(expert_blend, e2e_blend),
        "expert_first_active_step": expert_first_active,
        "end_to_end_first_active_step": e2e_first_active,
        "end_to_end_minus_expert_first_active_step": e2e_first_active - expert_first_active,
        "expert_vp_forward_bias_m": _safe_float(expert_formal_geometry.get("vp_forward_bias_m")),
        "end_to_end_vp_forward_bias_m": _safe_float(e2e_formal_geometry.get("vp_forward_bias_m")),
        "expert_merge_min_range_m": _safe_float(expert_formal_geometry.get("merge_min_range_m")),
        "end_to_end_merge_min_range_m": _safe_float(e2e_formal_geometry.get("merge_min_range_m")),
    }


def build_executive_diagnosis(
    *,
    expert_split: Mapping[str, Any],
    contrast: Mapping[str, Any],
) -> Dict[str, Any]:
    expert_window_rarity = expert_split["window_rarity"]
    expert_alignment = expert_split["pilot_alignment_to_formal_outcomes"]
    expert_loss_taxonomy = expert_split["loss_taxonomy"]
    expert_direct_losses = expert_split["formal_modes"]["direct_losses"]["overview"]
    expert_no_direct_losses = expert_split["formal_modes"]["no_direct_losses"]["overview"]

    return {
        "pilot_window_bias": {
            "window_size": int(expert_window_rarity["window_size"]),
            "window_count": int(expert_window_rarity["window_count"]),
            "win_rate_at_or_above_target_count": int(
                expert_window_rarity["win_rate_at_or_above_target_count"]
            ),
            "mean_damage_margin_at_or_above_target_count": int(
                expert_window_rarity["mean_damage_margin_at_or_above_target_count"]
            ),
            "mean_post_advantage_at_or_above_target_count": int(
                expert_window_rarity["mean_post_advantage_at_or_above_target_count"]
            ),
        },
        "pilot_alignment_bias": {
            "closer_to_wins_count": int(expert_alignment["closer_to_wins_count"]),
            "compared_feature_count": int(expert_alignment["compared_feature_count"]),
        },
        "expert_failure_channels": {
            "direct_track_collapse_share_of_losses": _safe_float(
                expert_loss_taxonomy["direct_losses"]["share_of_losses"]
            ),
            "no_direct_overdeep_share_of_losses": _safe_float(
                expert_loss_taxonomy["no_direct_losses"]["share_of_losses"]
            ),
            "direct_track_collapse_positive_damage_margin_fraction": _safe_float(
                expert_direct_losses["positive_damage_margin_fraction"]
            ),
            "direct_track_collapse_positive_post_advantage_fraction": _safe_float(
                expert_direct_losses["positive_post_advantage_fraction"]
            ),
            "direct_track_collapse_ego_crash_rate": _safe_float(
                expert_direct_losses["ego_crash_rate"]
            ),
            "direct_track_collapse_mean_final_altitude_m": _safe_float(
                expert_loss_taxonomy["direct_losses"]["mean_final_altitude_m"]
            ),
            "no_direct_overdeep_mean_damage_margin": _safe_float(
                expert_no_direct_losses["mean_damage_margin"]
            ),
            "no_direct_overdeep_mean_post_merge_advantage_s": _safe_float(
                expert_no_direct_losses["mean_post_merge_attack_zone_advantage_s"]
            ),
            "no_direct_overdeep_mean_vp_forward_bias_m": _safe_float(
                expert_loss_taxonomy["no_direct_losses"]["mean_vp_forward_bias_m"]
            ),
        },
        "end_to_end_guardrail": {
            "expert_vs_end_to_end_blend_active_ratio": _safe_float(
                contrast["expert_vs_end_to_end_blend_active_ratio"]
            ),
            "end_to_end_minus_expert_first_active_step": _safe_float(
                contrast["end_to_end_minus_expert_first_active_step"]
            ),
            "end_to_end_minus_expert_vp_forward_bias_m": (
                _safe_float(contrast["end_to_end_vp_forward_bias_m"])
                - _safe_float(contrast["expert_vp_forward_bias_m"])
            ),
        },
    }


def build_report(
    *,
    task: str,
    method: str,
    expert_pilot_rows: Sequence[Mapping[str, Any]],
    expert_formal_rows: Sequence[Mapping[str, Any]],
    expert_pilot_geometry: Mapping[str, Any],
    expert_formal_geometry: Mapping[str, Any],
    end_to_end_pilot_rows: Sequence[Mapping[str, Any]],
    end_to_end_formal_rows: Sequence[Mapping[str, Any]],
    end_to_end_pilot_geometry: Mapping[str, Any],
    end_to_end_formal_geometry: Mapping[str, Any],
    direct_track_altitude_threshold_m: float,
) -> Dict[str, Any]:
    expert_split = build_split_analysis(
        pilot_rows=expert_pilot_rows,
        formal_rows=expert_formal_rows,
        pilot_geometry_summary=expert_pilot_geometry,
        formal_geometry_summary=expert_formal_geometry,
        direct_track_altitude_threshold_m=direct_track_altitude_threshold_m,
    )
    end_to_end_split = build_split_analysis(
        pilot_rows=end_to_end_pilot_rows,
        formal_rows=end_to_end_formal_rows,
        pilot_geometry_summary=end_to_end_pilot_geometry,
        formal_geometry_summary=end_to_end_formal_geometry,
        direct_track_altitude_threshold_m=direct_track_altitude_threshold_m,
    )
    contrast = build_cross_split_contrast(
        expert_split=expert_split,
        end_to_end_split=end_to_end_split,
    )

    return {
        "task": task,
        "method": method,
        "pilot_alignment_features": list(PILOT_ALIGNMENT_FEATURES),
        "expert": expert_split,
        "end_to_end": end_to_end_split,
        "cross_split_contrast": contrast,
        "executive_diagnosis": build_executive_diagnosis(
            expert_split=expert_split,
            contrast=contrast,
        ),
    }


def _format_float(value: Any) -> str:
    number = _safe_float(value)
    if not math.isfinite(number):
        return "nan"
    return f"{number:.3f}"


def render_markdown(report: Mapping[str, Any]) -> str:
    expert = report["expert"]
    end_to_end = report["end_to_end"]
    contrast = report["cross_split_contrast"]
    diagnosis = report["executive_diagnosis"]
    expert_alignment = expert["pilot_alignment_to_formal_outcomes"]
    e2e_alignment = end_to_end["pilot_alignment_to_formal_outcomes"]
    expert_window_rarity = expert["window_rarity"]
    expert_loss_taxonomy = expert["loss_taxonomy"]
    expert_channel_story = expert["channel_story"]
    expert_pilot_mix = expert_channel_story["pilot_mix"]
    expert_direct_channel = expert_channel_story["direct_channel"]
    expert_no_direct_channel = expert_channel_story["no_direct_channel"]

    lines = [
        "# Directtrack2000 Rear-Quarter Pilot Bias Analysis",
        "",
        f"- task: `{report['task']}`",
        f"- method: `{report['method']}`",
        "",
        "## Diagnosis",
        "",
        (
            f"- expert pilot is a rare {diagnosis['pilot_window_bias']['window_size']}-seed window: only "
            f"{diagnosis['pilot_window_bias']['win_rate_at_or_above_target_count']}/"
            f"{diagnosis['pilot_window_bias']['window_count']} expert windows match its win rate, and only "
            f"{diagnosis['pilot_window_bias']['mean_damage_margin_at_or_above_target_count']}/"
            f"{diagnosis['pilot_window_bias']['window_count']} match its damage margin"
        ),
        (
            f"- the expert pilot centroid matches rare formal wins rather than typical held-out behavior on "
            f"{diagnosis['pilot_alignment_bias']['closer_to_wins_count']}/"
            f"{diagnosis['pilot_alignment_bias']['compared_feature_count']} selected features"
        ),
        (
            f"- expert held-out losses split across two channels instead of one: "
            f"direct_track_collapse={_format_float(diagnosis['expert_failure_channels']['direct_track_collapse_share_of_losses'])} "
            f"share of losses, no_direct_overdeep={_format_float(diagnosis['expert_failure_channels']['no_direct_overdeep_share_of_losses'])}"
        ),
        (
            f"- direct_track_collapse usually keeps offensive advantage but still crashes: "
            f"positive_damage={_format_float(diagnosis['expert_failure_channels']['direct_track_collapse_positive_damage_margin_fraction'])}, "
            f"positive_post_merge_adv={_format_float(diagnosis['expert_failure_channels']['direct_track_collapse_positive_post_advantage_fraction'])}, "
            f"ego_crash_rate={_format_float(diagnosis['expert_failure_channels']['direct_track_collapse_ego_crash_rate'])}, "
            f"mean_final_altitude_m={_format_float(diagnosis['expert_failure_channels']['direct_track_collapse_mean_final_altitude_m'])}"
        ),
        (
            f"- no_direct_overdeep is the other half of the collapse: mean_damage_margin="
            f"{_format_float(diagnosis['expert_failure_channels']['no_direct_overdeep_mean_damage_margin'])}, "
            f"mean_post_merge_adv_s={_format_float(diagnosis['expert_failure_channels']['no_direct_overdeep_mean_post_merge_advantage_s'])}, "
            f"mean_vp_forward_bias_m={_format_float(diagnosis['expert_failure_channels']['no_direct_overdeep_mean_vp_forward_bias_m'])}"
        ),
        (
            f"- end_to_end is protected because the rear-quarter branch stays much milder there: "
            f"expert/end_to_end blend-active ratio="
            f"{_format_float(diagnosis['end_to_end_guardrail']['expert_vs_end_to_end_blend_active_ratio'])}, "
            f"first-active delay={_format_float(diagnosis['end_to_end_guardrail']['end_to_end_minus_expert_first_active_step'])} steps, "
            f"forward-bias delta={_format_float(diagnosis['end_to_end_guardrail']['end_to_end_minus_expert_vp_forward_bias_m'])} m"
        ),
        "",
        "## Split Overview",
        "",
        "| split | pilot_n | pilot_win_rate | formal_n | formal_win_rate | formal_damage_margin | formal_post_merge_adv_s | formal_ego_crash_rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| expert | {expert['pilot_overview']['n']} | {_format_float(expert['pilot_overview']['win_rate'])} | "
            f"{expert['formal_overview']['n']} | {_format_float(expert['formal_overview']['win_rate'])} | "
            f"{_format_float(expert['formal_overview']['mean_damage_margin'])} | "
            f"{_format_float(expert['formal_overview']['mean_post_merge_attack_zone_advantage_s'])} | "
            f"{_format_float(expert['formal_overview']['ego_crash_rate'])} |"
        ),
        (
            f"| end_to_end | {end_to_end['pilot_overview']['n']} | {_format_float(end_to_end['pilot_overview']['win_rate'])} | "
            f"{end_to_end['formal_overview']['n']} | {_format_float(end_to_end['formal_overview']['win_rate'])} | "
            f"{_format_float(end_to_end['formal_overview']['mean_damage_margin'])} | "
            f"{_format_float(end_to_end['formal_overview']['mean_post_merge_attack_zone_advantage_s'])} | "
            f"{_format_float(end_to_end['formal_overview']['ego_crash_rate'])} |"
        ),
        "",
        "## Expert Pilot Bias",
        "",
        (
            f"- expert pilot window percentile: win_rate={_format_float(expert['window_percentiles']['win_rate_percentile'])}, "
            f"damage_margin={_format_float(expert['window_percentiles']['mean_damage_margin_percentile'])}, "
            f"post_merge_advantage={_format_float(expert['window_percentiles']['mean_post_advantage_percentile'])}, "
            f"low_crash={_format_float(expert['window_percentiles']['ego_crash_rate_percentile_low_is_good'])}"
        ),
        (
            f"- expert pilot centroid is closer to rare formal wins on "
            f"{expert_alignment['closer_to_wins_count']}/{expert_alignment['compared_feature_count']} "
            f"selected rear-quarter features"
        ),
        (
            f"- end_to_end pilot centroid is closer to formal wins on "
            f"{e2e_alignment['closer_to_wins_count']}/{e2e_alignment['compared_feature_count']} "
            f"selected features"
        ),
        "",
        "| feature | expert_pilot | expert_formal_wins | expert_formal_losses | closer_to |",
        "|---|---:|---:|---:|---|",
    ]

    for feature_row in expert_alignment["features"]:
        lines.append(
            (
                f"| {feature_row['feature']} | {_format_float(feature_row['pilot_mean'])} | "
                f"{_format_float(feature_row['formal_win_mean'])} | "
                f"{_format_float(feature_row['formal_loss_mean'])} | "
                f"{feature_row['closer_to']} |"
            )
        )

    lines.extend(
        [
            "",
            "## Expert Pilot Window Rarity",
            "",
            (
                f"- only {expert_window_rarity['win_rate_at_or_above_target_count']}/"
                f"{expert_window_rarity['window_count']} expert "
                f"{expert_window_rarity['window_size']}-seed windows match or beat the pilot win rate"
            ),
            (
                f"- only {expert_window_rarity['mean_damage_margin_at_or_above_target_count']}/"
                f"{expert_window_rarity['window_count']} windows match or beat the pilot damage margin"
            ),
            (
                f"- only {expert_window_rarity['mean_post_advantage_at_or_above_target_count']}/"
                f"{expert_window_rarity['window_count']} windows match or beat the pilot post-merge advantage"
            ),
            (
                f"- median expert {expert_window_rarity['window_size']}-seed window win_rate="
                f"{_format_float(expert_window_rarity['median_window_win_rate'])}"
            ),
            "",
            "## Expert Held-Out Failure Modes",
            "",
            "| mode | n | win_rate | ego_crash_rate | positive_damage_frac | mean_damage_margin | mean_post_merge_adv_s | mean_vp_forward_bias_m | mean_direct_track_fraction |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for key, label in [
        ("wins", "formal_wins"),
        ("losses", "formal_losses"),
        ("direct_losses", "direct_losses"),
        ("no_direct_losses", "no_direct_losses"),
    ]:
        summary = expert["formal_modes"][key]
        lines.append(
            (
                f"| {label} | {summary['overview']['n']} | {_format_float(summary['overview']['win_rate'])} | "
                f"{_format_float(summary['overview']['ego_crash_rate'])} | "
                f"{_format_float(summary['overview']['positive_damage_margin_fraction'])} | "
                f"{_format_float(summary['overview']['mean_damage_margin'])} | "
                f"{_format_float(summary['overview']['mean_post_merge_attack_zone_advantage_s'])} | "
                f"{_format_float(summary['geometry_means']['vp_forward_bias_m'])} | "
                f"{_format_float(summary['geometry_means']['post_merge_offensive_anchor_blend_release_direct_track_active_fraction'])} |"
            )
        )

    lines.extend(
        [
            "",
            "## Expert Loss Taxonomy",
            "",
            "| channel | share_of_losses | mean_damage_margin | mean_post_merge_adv_s | mean_vp_forward_bias_m | mean_final_altitude_m | ego_crash_rate |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for key, label in [
        ("direct_losses", "direct_track_collapse"),
        ("no_direct_losses", "no_direct_overdeep"),
    ]:
        channel = expert_loss_taxonomy[key]
        lines.append(
            (
                f"| {label} | {_format_float(channel['share_of_losses'])} | "
                f"{_format_float(channel['mean_damage_margin'])} | "
                f"{_format_float(channel['mean_post_merge_advantage_s'])} | "
                f"{_format_float(channel['mean_vp_forward_bias_m'])} | "
                f"{_format_float(channel['mean_final_altitude_m'])} | "
                f"{_format_float(channel['ego_crash_rate'])} |"
            )
        )

    lines.extend(
        [
            "",
            "## Expert Pilot Members",
            "",
            (
                f"- pilot mix: direct_track episodes={expert_pilot_mix['direct_episode_count']}/"
                f"{expert_pilot_mix['episodes']}, no_direct episodes={expert_pilot_mix['no_direct_episode_count']}/"
                f"{expert_pilot_mix['episodes']}"
            ),
            "",
            "| seed | direct_track | termination_reason | damage_margin | post_merge_adv_s | vp_forward_bias_m | direct_track_fraction | first_direct_step | first_direct_altitude_m |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for member in expert_pilot_mix["members"]:
        lines.append(
            (
                f"| {member['seed']} | {member['has_direct_track']} | {member['termination_reason']} | "
                f"{_format_float(member['damage_margin'])} | "
                f"{_format_float(member['post_merge_attack_zone_advantage_s'])} | "
                f"{_format_float(member['vp_forward_bias_m'])} | "
                f"{_format_float(member['post_merge_offensive_anchor_blend_release_direct_track_active_fraction'])} | "
                f"{_format_float(member['post_merge_offensive_anchor_blend_release_direct_track_first_step'])} | "
                f"{_format_float(member['first_direct_altitude_m'])} |"
            )
        )

    direct_delta = expert_direct_channel["loss_minus_win"]
    no_direct_delta = expert_no_direct_channel["loss_minus_win"]

    lines.extend(
        [
            "",
            "## Expert Channel Story",
            "",
            "| channel | formal_wins_n | formal_losses_n | win_damage | loss_damage | win_post_adv | loss_post_adv |",
            "|---|---:|---:|---:|---:|---:|---:|",
            (
                f"| direct_track | {expert_direct_channel['formal_wins']['overview']['n']} | "
                f"{expert_direct_channel['formal_losses']['overview']['n']} | "
                f"{_format_float(expert_direct_channel['formal_wins']['overview']['mean_damage_margin'])} | "
                f"{_format_float(expert_direct_channel['formal_losses']['overview']['mean_damage_margin'])} | "
                f"{_format_float(expert_direct_channel['formal_wins']['overview']['mean_post_merge_attack_zone_advantage_s'])} | "
                f"{_format_float(expert_direct_channel['formal_losses']['overview']['mean_post_merge_attack_zone_advantage_s'])} |"
            ),
            (
                f"| no_direct | {expert_no_direct_channel['formal_wins']['overview']['n']} | "
                f"{expert_no_direct_channel['formal_losses']['overview']['n']} | "
                f"{_format_float(expert_no_direct_channel['formal_wins']['overview']['mean_damage_margin'])} | "
                f"{_format_float(expert_no_direct_channel['formal_losses']['overview']['mean_damage_margin'])} | "
                f"{_format_float(expert_no_direct_channel['formal_wins']['overview']['mean_post_merge_attack_zone_advantage_s'])} | "
                f"{_format_float(expert_no_direct_channel['formal_losses']['overview']['mean_post_merge_attack_zone_advantage_s'])} |"
            ),
            "",
            (
                f"- direct_track_collapse vs direct wins: direct_track_fraction "
                f"{_format_float(expert_direct_channel['formal_wins']['geometry_means']['post_merge_offensive_anchor_blend_release_direct_track_active_fraction'])}"
                f" -> {_format_float(expert_direct_channel['formal_losses']['geometry_means']['post_merge_offensive_anchor_blend_release_direct_track_active_fraction'])}, "
                f"first_direct_step delta={_format_float(direct_delta['post_merge_offensive_anchor_blend_release_direct_track_first_step'])}, "
                f"first_direct_altitude delta={_format_float(direct_delta['first_direct_altitude_m'])}, "
                f"final_altitude delta={_format_float(direct_delta['final_altitude_m'])}"
            ),
            (
                f"- no_direct_overdeep vs no-direct wins: vp_forward_bias delta={_format_float(no_direct_delta['vp_forward_bias_m'])}, "
                f"vp_lateral_bias delta={_format_float(no_direct_delta['vp_lateral_bias_m'])}, "
                f"blend_active_fraction delta={_format_float(no_direct_delta['post_merge_offensive_anchor_blend_active_fraction'])}, "
                f"damage_margin delta={_format_float(no_direct_delta['damage_margin'])}"
            ),
            "",
            "## Cross-Split Contrast",
            "",
            "| metric | expert_formal | end_to_end_formal |",
            "|---|---:|---:|",
            f"| win_rate | {_format_float(contrast['expert_formal_win_rate'])} | {_format_float(contrast['end_to_end_formal_win_rate'])} |",
            f"| damage_margin | {_format_float(contrast['expert_formal_damage_margin'])} | {_format_float(contrast['end_to_end_formal_damage_margin'])} |",
            f"| post_merge_advantage_s | {_format_float(contrast['expert_formal_post_merge_advantage_s'])} | {_format_float(contrast['end_to_end_formal_post_merge_advantage_s'])} |",
            f"| alignment_disadvantage_fraction | {_format_float(contrast['expert_alignment_disadvantage_fraction'])} | {_format_float(contrast['end_to_end_alignment_disadvantage_fraction'])} |",
            f"| rearquarter_blend_active_fraction | {_format_float(contrast['expert_blend_active_fraction'])} | {_format_float(contrast['end_to_end_blend_active_fraction'])} |",
            f"| rearquarter_first_active_step | {_format_float(contrast['expert_first_active_step'])} | {_format_float(contrast['end_to_end_first_active_step'])} |",
            f"| vp_forward_bias_m | {_format_float(contrast['expert_vp_forward_bias_m'])} | {_format_float(contrast['end_to_end_vp_forward_bias_m'])} |",
            "",
            (
                f"- expert rear-quarter blend is {_format_float(contrast['expert_vs_end_to_end_blend_active_ratio'])}x "
                "as active as end_to_end held-out"
            ),
            (
                f"- end_to_end rear-quarter blend activates {_format_float(contrast['end_to_end_minus_expert_first_active_step'])} "
                "steps later than expert held-out"
            ),
        ]
    )

    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-pilot-run-dir", required=True)
    parser.add_argument("--expert-formal-run-dir", required=True)
    parser.add_argument("--end-to-end-pilot-run-dir", required=True)
    parser.add_argument("--end-to-end-formal-run-dir", required=True)
    parser.add_argument("--task", default="head_on")
    parser.add_argument("--method", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--direct-track-altitude-threshold-m", type=float, default=2000.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    expert_pilot_rows, expert_pilot_geometry = load_joined_rows(
        Path(args.expert_pilot_run_dir),
        task=args.task,
        method=args.method,
    )
    expert_formal_rows, expert_formal_geometry = load_joined_rows(
        Path(args.expert_formal_run_dir),
        task=args.task,
        method=args.method,
    )
    end_to_end_pilot_rows, end_to_end_pilot_geometry = load_joined_rows(
        Path(args.end_to_end_pilot_run_dir),
        task=args.task,
        method=args.method,
    )
    end_to_end_formal_rows, end_to_end_formal_geometry = load_joined_rows(
        Path(args.end_to_end_formal_run_dir),
        task=args.task,
        method=args.method,
    )

    report = build_report(
        task=args.task,
        method=args.method,
        expert_pilot_rows=expert_pilot_rows,
        expert_formal_rows=expert_formal_rows,
        expert_pilot_geometry=expert_pilot_geometry,
        expert_formal_geometry=expert_formal_geometry,
        end_to_end_pilot_rows=end_to_end_pilot_rows,
        end_to_end_formal_rows=end_to_end_formal_rows,
        end_to_end_pilot_geometry=end_to_end_pilot_geometry,
        end_to_end_formal_geometry=end_to_end_formal_geometry,
        direct_track_altitude_threshold_m=float(args.direct_track_altitude_threshold_m),
    )

    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        render_markdown(report),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
