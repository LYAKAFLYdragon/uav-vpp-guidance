#!/usr/bin/env python3
"""Analyze directtrack2000 head-on failure patterns from raw combat artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


TAIL_FIELDS = [
    "step",
    "time_s",
    "range_m",
    "range_rate_mps",
    "altitude_m",
    "heading_deg",
    "vp_forward_bias_m",
    "post_merge_offensive_anchor_blend_active",
    "post_merge_offensive_anchor_blend_released",
    "post_merge_offensive_anchor_blend_release_direct_track_active",
    "direct_track_mode_effective",
    "ego_in_attack_zone",
    "target_in_attack_zone",
]


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _mean(values: Iterable[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return float("nan")
    return sum(finite) / len(finite)


def _median(values: Iterable[float]) -> float:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return float("nan")
    mid = len(finite) // 2
    if len(finite) % 2 == 1:
        return finite[mid]
    return 0.5 * (finite[mid - 1] + finite[mid])


def _fraction(values: Sequence[Any], predicate) -> float:
    if not values:
        return float("nan")
    return sum(1 for value in values if predicate(value)) / len(values)


def _count(values: Sequence[Any], predicate) -> int:
    return sum(1 for value in values if predicate(value))


def load_episodes(run_dir: Path, task: str, method: str) -> List[Dict[str, Any]]:
    raw_dir = run_dir / "raw" / task / method
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw episode directory not found: {raw_dir}")
    episodes: List[Dict[str, Any]] = []
    for episode_path in sorted(raw_dir.glob("seed_*/episode_000.json")):
        episodes.append(json.loads(episode_path.read_text(encoding="utf-8")))
    if not episodes:
        raise FileNotFoundError(f"No episode_000.json files found under {raw_dir}")
    return episodes


def episode_to_row(episode: Dict[str, Any]) -> Dict[str, Any]:
    trajectory = episode.get("trajectory")
    if not isinstance(trajectory, list):
        trajectory = []

    first_direct = next(
        (
            point
            for point in trajectory
            if bool(point.get("post_merge_offensive_anchor_blend_release_direct_track_active"))
        ),
        None,
    )
    below_2000 = next(
        (
            point
            for point in trajectory
            if _safe_float(point.get("altitude_m")) < 2000.0
        ),
        None,
    )
    min_altitude = _mean([])
    final_altitude = _mean([])
    if trajectory:
        min_altitude = min(_safe_float(point.get("altitude_m")) for point in trajectory)
        final_altitude = _safe_float(trajectory[-1].get("altitude_m"))

    return {
        "seed": int(episode.get("seed", -1)),
        "win": bool(episode.get("win")),
        "loss": bool(episode.get("loss")),
        "termination_reason": str(episode.get("termination_reason") or "unknown"),
        "damage_margin": _safe_float(episode.get("damage_margin")),
        "post_merge_attack_zone_advantage_s": _safe_float(
            episode.get("post_merge_attack_zone_advantage_s")
        ),
        "direct_track_fraction": _safe_float(
            episode.get("post_merge_offensive_anchor_blend_release_direct_track_active_fraction")
        ),
        "offblend_fraction": _safe_float(
            episode.get("post_merge_offensive_anchor_blend_active_fraction")
        ),
        "released_fraction": _safe_float(
            episode.get("post_merge_offensive_anchor_blend_released_fraction")
        ),
        "first_pass_step": _safe_float(episode.get("first_pass_step")),
        "has_direct_track": first_direct is not None,
        "first_direct_step": _safe_float(first_direct.get("step")) if first_direct else float("nan"),
        "first_direct_altitude_m": (
            _safe_float(first_direct.get("altitude_m")) if first_direct else float("nan")
        ),
        "first_direct_range_m": (
            _safe_float(first_direct.get("range_m")) if first_direct else float("nan")
        ),
        "first_direct_range_rate_mps": (
            _safe_float(first_direct.get("range_rate_mps")) if first_direct else float("nan")
        ),
        "below_2000_step": _safe_float(below_2000.get("step")) if below_2000 else float("nan"),
        "min_altitude_m": min_altitude,
        "final_altitude_m": final_altitude,
        "steps": int(episode.get("steps", 0)),
    }


def summarize_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    damage_margins = [_safe_float(row.get("damage_margin")) for row in rows]
    post_advantages = [
        _safe_float(row.get("post_merge_attack_zone_advantage_s")) for row in rows
    ]
    direct_track = [_safe_float(row.get("direct_track_fraction")) for row in rows]

    return {
        "n": len(rows),
        "wins": sum(1 for row in rows if bool(row.get("win"))),
        "win_rate": _fraction(rows, lambda row: bool(row.get("win"))),
        "ego_crash_rate": _fraction(
            rows,
            lambda row: str(row.get("termination_reason")) == "ego_crash_or_out_of_bounds",
        ),
        "positive_damage_margin_fraction": _fraction(
            rows, lambda row: _safe_float(row.get("damage_margin")) > 0.0
        ),
        "positive_post_advantage_fraction": _fraction(
            rows, lambda row: _safe_float(row.get("post_merge_attack_zone_advantage_s")) > 0.0
        ),
        "mean_damage_margin": _mean(damage_margins),
        "median_damage_margin": _median(damage_margins),
        "mean_post_merge_attack_zone_advantage_s": _mean(post_advantages),
        "median_post_merge_attack_zone_advantage_s": _median(post_advantages),
        "mean_direct_track_fraction": _mean(direct_track),
        "median_direct_track_fraction": _median(direct_track),
        "mean_first_direct_step": _mean(
            _safe_float(row.get("first_direct_step")) for row in rows
        ),
        "mean_first_direct_altitude_m": _mean(
            _safe_float(row.get("first_direct_altitude_m")) for row in rows
        ),
        "mean_below_2000_step": _mean(
            _safe_float(row.get("below_2000_step")) for row in rows
        ),
        "mean_min_altitude_m": _mean(_safe_float(row.get("min_altitude_m")) for row in rows),
        "mean_final_altitude_m": _mean(
            _safe_float(row.get("final_altitude_m")) for row in rows
        ),
    }


def summarize_windows(
    rows: Sequence[Dict[str, Any]],
    *,
    window_size: int,
) -> List[Dict[str, Any]]:
    sorted_rows = sorted(rows, key=lambda row: int(row.get("seed", -1)))
    windows: List[Dict[str, Any]] = []
    for start in range(0, len(sorted_rows) - window_size + 1):
        chunk = sorted_rows[start : start + window_size]
        summary = summarize_rows(chunk)
        summary["start_seed"] = int(chunk[0]["seed"])
        summary["end_seed"] = int(chunk[-1]["seed"])
        windows.append(summary)
    return windows


def compute_window_percentiles(
    rows: Sequence[Dict[str, Any]],
    *,
    target_start_seed: int,
    window_size: int,
) -> Dict[str, Any]:
    windows = summarize_windows(rows, window_size=window_size)
    try:
        target = next(window for window in windows if window["start_seed"] == target_start_seed)
    except StopIteration as exc:
        raise ValueError(
            f"Target window start seed {target_start_seed} not found for size {window_size}"
        ) from exc

    win_rate_at_or_above_target_count = _count(
        windows, lambda window: window["win_rate"] >= target["win_rate"]
    )
    damage_margin_at_or_above_target_count = _count(
        windows,
        lambda window: window["mean_damage_margin"] >= target["mean_damage_margin"],
    )
    post_advantage_at_or_above_target_count = _count(
        windows,
        lambda window: (
            window["mean_post_merge_attack_zone_advantage_s"]
            >= target["mean_post_merge_attack_zone_advantage_s"]
        ),
    )
    ego_crash_rate_at_or_below_target_count = _count(
        windows,
        lambda window: window["ego_crash_rate"] <= target["ego_crash_rate"],
    )

    return {
        "window_size": window_size,
        "window_count": len(windows),
        "target_window": target,
        "win_rate_percentile": _fraction(
            windows, lambda window: window["win_rate"] <= target["win_rate"]
        ),
        "mean_damage_margin_percentile": _fraction(
            windows,
            lambda window: window["mean_damage_margin"] <= target["mean_damage_margin"],
        ),
        "mean_post_advantage_percentile": _fraction(
            windows,
            lambda window: (
                window["mean_post_merge_attack_zone_advantage_s"]
                <= target["mean_post_merge_attack_zone_advantage_s"]
            ),
        ),
        "ego_crash_rate_percentile_low_is_good": _fraction(
            windows,
            lambda window: window["ego_crash_rate"] >= target["ego_crash_rate"],
        ),
        "win_rate_at_or_above_target_count": win_rate_at_or_above_target_count,
        "win_rate_at_or_above_target_fraction": (
            win_rate_at_or_above_target_count / len(windows) if windows else float("nan")
        ),
        "mean_damage_margin_at_or_above_target_count": (
            damage_margin_at_or_above_target_count
        ),
        "mean_damage_margin_at_or_above_target_fraction": (
            damage_margin_at_or_above_target_count / len(windows)
            if windows
            else float("nan")
        ),
        "mean_post_advantage_at_or_above_target_count": (
            post_advantage_at_or_above_target_count
        ),
        "mean_post_advantage_at_or_above_target_fraction": (
            post_advantage_at_or_above_target_count / len(windows)
            if windows
            else float("nan")
        ),
        "ego_crash_rate_at_or_below_target_count": ego_crash_rate_at_or_below_target_count,
        "ego_crash_rate_at_or_below_target_fraction": (
            ego_crash_rate_at_or_below_target_count / len(windows)
            if windows
            else float("nan")
        ),
        "best_win_rate": max(window["win_rate"] for window in windows),
        "median_win_rate": _median(window["win_rate"] for window in windows),
    }


def build_failure_report(
    formal_rows: Sequence[Dict[str, Any]],
    pilot_rows: Sequence[Dict[str, Any]],
    *,
    pilot_window_start_seed: Optional[int] = None,
    legacy_window_size: int = 12,
    exemplar_rows: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    sorted_formal = sorted(formal_rows, key=lambda row: int(row.get("seed", -1)))
    sorted_pilot = sorted(pilot_rows, key=lambda row: int(row.get("seed", -1)))
    pilot_start_seed = (
        int(pilot_window_start_seed)
        if pilot_window_start_seed is not None
        else int(sorted_pilot[0]["seed"])
    )

    losses = [row for row in sorted_formal if bool(row.get("loss"))]
    direct_rows = [row for row in sorted_formal if bool(row.get("has_direct_track"))]
    no_direct_rows = [row for row in sorted_formal if not bool(row.get("has_direct_track"))]
    wins = [row for row in sorted_formal if bool(row.get("win"))]
    ego_crash = [
        row
        for row in sorted_formal
        if str(row.get("termination_reason")) == "ego_crash_or_out_of_bounds"
    ]
    timeout_advantage = [
        row
        for row in sorted_formal
        if str(row.get("termination_reason")) == "timeout_hp_advantage"
    ]

    report: Dict[str, Any] = {
        "formal_overview": summarize_rows(sorted_formal),
        "pilot_overview": summarize_rows(sorted_pilot),
        "legacy_window_size": int(legacy_window_size),
        "window_percentiles": {
            "pilot_window": compute_window_percentiles(
                sorted_formal,
                target_start_seed=pilot_start_seed,
                window_size=len(sorted_pilot),
            ),
            "legacy_window": compute_window_percentiles(
                sorted_formal,
                target_start_seed=pilot_start_seed,
                window_size=int(legacy_window_size),
            ),
        },
        "termination_reasons": {
            reason: sum(
                1
                for row in sorted_formal
                if str(row.get("termination_reason") or "unknown") == reason
            )
            for reason in sorted(
                {str(row.get("termination_reason") or "unknown") for row in sorted_formal}
            )
        },
        "loss_retention": {
            "losses_total": len(losses),
            "losses_positive_damage_margin": sum(
                1 for row in losses if _safe_float(row.get("damage_margin")) > 0.0
            ),
            "losses_positive_damage_margin_fraction": _fraction(
                losses, lambda row: _safe_float(row.get("damage_margin")) > 0.0
            ),
            "losses_positive_post_advantage": sum(
                1
                for row in losses
                if _safe_float(row.get("post_merge_attack_zone_advantage_s")) > 0.0
            ),
            "losses_positive_post_advantage_fraction": _fraction(
                losses,
                lambda row: _safe_float(row.get("post_merge_attack_zone_advantage_s")) > 0.0,
            ),
            "ego_crash_losses_positive_damage_margin": sum(
                1
                for row in ego_crash
                if _safe_float(row.get("damage_margin")) > 0.0
            ),
            "ego_crash_losses_positive_post_advantage": sum(
                1
                for row in ego_crash
                if _safe_float(row.get("post_merge_attack_zone_advantage_s")) > 0.0
            ),
        },
        "outcome_splits": {
            "wins": summarize_rows(wins),
            "losses": summarize_rows(losses),
            "ego_crash_or_out_of_bounds": summarize_rows(ego_crash),
            "timeout_hp_advantage": summarize_rows(timeout_advantage),
        },
        "direct_track_split": {
            "with_direct": summarize_rows(direct_rows),
            "without_direct": summarize_rows(no_direct_rows),
            "wins_with_direct": summarize_rows(
                [row for row in direct_rows if bool(row.get("win"))]
            ),
            "losses_with_direct": summarize_rows(
                [row for row in direct_rows if bool(row.get("loss"))]
            ),
            "losses_without_direct": summarize_rows(
                [row for row in no_direct_rows if bool(row.get("loss"))]
            ),
            "direct_losses_below_2000_count": sum(
                1
                for row in direct_rows
                if bool(row.get("loss")) and math.isfinite(_safe_float(row.get("below_2000_step")))
            ),
            "no_direct_losses_below_2000_count": sum(
                1
                for row in no_direct_rows
                if bool(row.get("loss")) and math.isfinite(_safe_float(row.get("below_2000_step")))
            ),
        },
    }

    if exemplar_rows is not None:
        report["exemplars"] = {
            str(int(row["seed"])): dict(row) for row in sorted(exemplar_rows, key=lambda row: row["seed"])
        }

    return report


def build_tail_trace(episode: Dict[str, Any], *, tail_steps: int) -> List[Dict[str, Any]]:
    trajectory = episode.get("trajectory")
    if not isinstance(trajectory, list):
        return []
    tail = trajectory[-tail_steps:]
    return [{field: point.get(field) for field in TAIL_FIELDS} for point in tail]


def render_markdown(
    report: Dict[str, Any],
    *,
    formal_run_dir: Path,
    pilot_run_dir: Path,
    task: str,
    method: str,
) -> str:
    formal = report["formal_overview"]
    pilot = report["pilot_overview"]
    pilot_window = report["window_percentiles"]["pilot_window"]
    legacy_window = report["window_percentiles"]["legacy_window"]
    loss_retention = report["loss_retention"]
    direct_track = report["direct_track_split"]

    lines = [
        "# Directtrack2000 Head-On Failure Analysis",
        "",
        f"- formal_run_dir: `{formal_run_dir}`",
        f"- pilot_run_dir: `{pilot_run_dir}`",
        f"- task: `{task}`",
        f"- method: `{method}`",
        "",
        "## Overall",
        "",
        "| split | n | win_rate | mean_damage_margin | mean_post_merge_adv_s | positive_margin_frac | ego_crash_rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| pilot | {pilot['n']} | {pilot['win_rate']:.3f} | {pilot['mean_damage_margin']:.3f} | "
            f"{pilot['mean_post_merge_attack_zone_advantage_s']:.3f} | "
            f"{pilot['positive_damage_margin_fraction']:.3f} | {pilot['ego_crash_rate']:.3f} |"
        ),
        (
            f"| formal | {formal['n']} | {formal['win_rate']:.3f} | {formal['mean_damage_margin']:.3f} | "
            f"{formal['mean_post_merge_attack_zone_advantage_s']:.3f} | "
            f"{formal['positive_damage_margin_fraction']:.3f} | {formal['ego_crash_rate']:.3f} |"
        ),
        "",
        "## Window Bias",
        "",
        "| window | size | win_rate | mean_damage_margin | mean_post_merge_adv_s | win_rate_pct | margin_pct | post_pct | low_crash_pct |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| pilot | {pilot_window['window_size']} | {pilot_window['target_window']['win_rate']:.3f} | "
            f"{pilot_window['target_window']['mean_damage_margin']:.3f} | "
            f"{pilot_window['target_window']['mean_post_merge_attack_zone_advantage_s']:.3f} | "
            f"{pilot_window['win_rate_percentile']:.3f} | "
            f"{pilot_window['mean_damage_margin_percentile']:.3f} | "
            f"{pilot_window['mean_post_advantage_percentile']:.3f} | "
            f"{pilot_window['ego_crash_rate_percentile_low_is_good']:.3f} |"
        ),
        (
            f"| legacy | {legacy_window['window_size']} | {legacy_window['target_window']['win_rate']:.3f} | "
            f"{legacy_window['target_window']['mean_damage_margin']:.3f} | "
            f"{legacy_window['target_window']['mean_post_merge_attack_zone_advantage_s']:.3f} | "
            f"{legacy_window['win_rate_percentile']:.3f} | "
            f"{legacy_window['mean_damage_margin_percentile']:.3f} | "
            f"{legacy_window['mean_post_advantage_percentile']:.3f} | "
            f"{legacy_window['ego_crash_rate_percentile_low_is_good']:.3f} |"
        ),
        "",
        "## Termination Reasons",
        "",
        "| reason | count | fraction |",
        "|---|---:|---:|",
    ]
    for reason, count in report["termination_reasons"].items():
        lines.append(f"| {reason} | {count} | {count / formal['n']:.3f} |")

    lines.extend(
        [
            "",
            "## Crash-After-Advantage Pattern",
            "",
            (
                f"- losses with positive damage margin: {loss_retention['losses_positive_damage_margin']}/"
                f"{loss_retention['losses_total']} "
                f"({loss_retention['losses_positive_damage_margin_fraction']:.3f})"
            ),
            (
                f"- losses with positive post-merge advantage: {loss_retention['losses_positive_post_advantage']}/"
                f"{loss_retention['losses_total']} "
                f"({loss_retention['losses_positive_post_advantage_fraction']:.3f})"
            ),
            (
                f"- ego-crash losses with positive damage margin: "
                f"{loss_retention['ego_crash_losses_positive_damage_margin']}/"
                f"{report['termination_reasons'].get('ego_crash_or_out_of_bounds', 0)}"
            ),
            (
                f"- ego-crash losses with positive post-merge advantage: "
                f"{loss_retention['ego_crash_losses_positive_post_advantage']}/"
                f"{report['termination_reasons'].get('ego_crash_or_out_of_bounds', 0)}"
            ),
            "",
            "## Direct-Track Split",
            "",
            "| split | n | win_rate | mean_damage_margin | mean_post_merge_adv_s | mean_first_direct_step | mean_first_direct_altitude_m | mean_final_altitude_m |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for key, label in [
        ("with_direct", "with_direct"),
        ("without_direct", "without_direct"),
        ("wins_with_direct", "wins_with_direct"),
        ("losses_with_direct", "losses_with_direct"),
        ("losses_without_direct", "losses_without_direct"),
    ]:
        item = direct_track[key]
        lines.append(
            (
                f"| {label} | {item['n']} | {item['win_rate']:.3f} | "
                f"{item['mean_damage_margin']:.3f} | "
                f"{item['mean_post_merge_attack_zone_advantage_s']:.3f} | "
                f"{item['mean_first_direct_step']:.3f} | "
                f"{item['mean_first_direct_altitude_m']:.3f} | "
                f"{item['mean_final_altitude_m']:.3f} |"
            )
        )

    lines.append("")
    lines.append(
        f"- direct-track losses below 2000m: {direct_track['direct_losses_below_2000_count']}"
    )
    lines.append(
        f"- no-direct losses below 2000m: {direct_track['no_direct_losses_below_2000_count']}"
    )

    exemplars = report.get("exemplars", {})
    if exemplars:
        lines.extend(
            [
                "",
                "## Exemplars",
                "",
                "| seed | win | termination_reason | damage_margin | post_merge_adv_s | direct_track_fraction | first_direct_step | first_direct_altitude_m | min_altitude_m | final_altitude_m |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for seed, item in exemplars.items():
            lines.append(
                (
                    f"| {seed} | {item['win']} | {item['termination_reason']} | "
                    f"{item['damage_margin']:.3f} | "
                    f"{item['post_merge_attack_zone_advantage_s']:.3f} | "
                    f"{item['direct_track_fraction']:.3f} | "
                    f"{item['first_direct_step']:.3f} | "
                    f"{item['first_direct_altitude_m']:.3f} | "
                    f"{item['min_altitude_m']:.3f} | "
                    f"{item['final_altitude_m']:.3f} |"
                )
            )

    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-run-dir", required=True, help="Formal run output directory")
    parser.add_argument("--pilot-run-dir", required=True, help="Pilot run output directory")
    parser.add_argument("--task", default="head_on")
    parser.add_argument("--method", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--legacy-window-size", type=int, default=12)
    parser.add_argument("--tail-steps", type=int, default=20)
    parser.add_argument(
        "--exemplar-seeds",
        nargs="*",
        type=int,
        default=[48000000, 48100000, 50000000],
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    formal_run_dir = Path(args.formal_run_dir)
    pilot_run_dir = Path(args.pilot_run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    formal_episodes = load_episodes(formal_run_dir, args.task, args.method)
    pilot_episodes = load_episodes(pilot_run_dir, args.task, args.method)
    formal_rows = [episode_to_row(episode) for episode in formal_episodes]
    pilot_rows = [episode_to_row(episode) for episode in pilot_episodes]

    exemplar_episode_map = {
        int(episode.get("seed", -1)): episode
        for episode in formal_episodes
        if int(episode.get("seed", -1)) in set(args.exemplar_seeds)
    }
    exemplar_rows = [
        row
        for row in formal_rows
        if int(row["seed"]) in exemplar_episode_map
    ]

    report = build_failure_report(
        formal_rows,
        pilot_rows,
        legacy_window_size=int(args.legacy_window_size),
        exemplar_rows=exemplar_rows,
    )

    report_json = output_dir / "report.json"
    report_md = output_dir / "report.md"
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report_md.write_text(
        render_markdown(
            report,
            formal_run_dir=formal_run_dir,
            pilot_run_dir=pilot_run_dir,
            task=args.task,
            method=args.method,
        ),
        encoding="utf-8",
    )

    for seed, episode in exemplar_episode_map.items():
        tail_path = output_dir / f"seed_{seed}_tail.json"
        tail_path.write_text(
            json.dumps(build_tail_trace(episode, tail_steps=int(args.tail_steps)), indent=2, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
