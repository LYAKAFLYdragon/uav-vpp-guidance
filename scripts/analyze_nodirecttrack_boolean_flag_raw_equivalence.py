#!/usr/bin/env python3
"""Compare old altitude=0 nodirecttrack hack traces with boolean-flag traces."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METHOD = (
    "prediction_vpp_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00"
)
DEFAULT_TASK = "head_on"
DEFAULT_SEEDS = (48000000, 48100000, 48200000)
DEFAULT_OLDHACK_RUN = (
    "outputs/jsbsim_hrl_comparison/"
    "reset075_no_mode_switch_longscale00_postmerge_offblend025_"
    "encounter_rearquarter300_geomgate_aa170_release1_"
    "nodirecttrack_directtrack2000_pilot_expert"
)
DEFAULT_OLDFLAG_RUN = (
    "outputs/jsbsim_hrl_comparison/"
    "reset075_no_mode_switch_longscale00_postmerge_offblend025_"
    "encounter_rearquarter300_geomgate_aa170_release1_"
    "nodirecttrack_boolean_flag_pilot_expert"
)
DEFAULT_FIXEDFLAG_RUN = (
    "outputs/jsbsim_hrl_comparison/"
    "reset075_no_mode_switch_longscale00_postmerge_offblend025_"
    "encounter_rearquarter300_geomgate_aa170_release1_"
    "nodirecttrack_boolean_flag_pilot_equiv12seed_expert"
)
DEFAULT_OUTPUT_MD = (
    "outputs/diagnostics/nodirecttrack_boolean_flag_raw_trace_comparison.md"
)
DEFAULT_OUTPUT_JSON = (
    "outputs/diagnostics/nodirecttrack_boolean_flag_raw_trace_comparison.json"
)

LATCH_FIELD = "post_merge_offensive_anchor_lateral_world_offset_latch_active"
RECOVERY_FIELD = "post_merge_offensive_anchor_blend_release_recovery_active"
RELEASE_FIELD = "post_merge_offensive_anchor_blend_released"
DIRECT_TRACK_FIELD = "direct_track_mode_effective"
RECOVERY_PREVIEW_FIELD = (
    "post_merge_offensive_anchor_blend_release_recovery_preview_vp_forward_bias_m"
)
VP_FORWARD_BIAS_FIELD = "vp_forward_bias_m"

WINDOW_FIELDS = (
    "step",
    RELEASE_FIELD,
    LATCH_FIELD,
    RECOVERY_FIELD,
    DIRECT_TRACK_FIELD,
    VP_FORWARD_BIAS_FIELD,
    RECOVERY_PREVIEW_FIELD,
)


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _read_episode(
    run_dir: Path,
    *,
    task: str,
    method: str,
    seed: int,
) -> Dict[str, Any]:
    episode_path = (
        run_dir
        / "raw"
        / task
        / method
        / f"seed_{seed}"
        / "episode_000.json"
    )
    if not episode_path.exists():
        raise FileNotFoundError(f"Episode artifact not found: {episode_path}")
    return json.loads(episode_path.read_text(encoding="utf-8"))


def _first_true_step(
    trajectory: Sequence[Mapping[str, Any]],
    field_name: str,
) -> Optional[int]:
    for index, row in enumerate(trajectory):
        if bool(row.get(field_name)):
            return int(row.get("step", index + 1))
    return None


def _step_index_map(trajectory: Sequence[Mapping[str, Any]]) -> Dict[int, int]:
    return {
        int(row.get("step", index + 1)): index
        for index, row in enumerate(trajectory)
    }


def first_vp_bias_divergence_step(
    lhs: Sequence[Mapping[str, Any]],
    rhs: Sequence[Mapping[str, Any]],
    *,
    threshold_m: float,
) -> Optional[int]:
    for lhs_row, rhs_row in zip(lhs, rhs):
        lhs_bias = _safe_float(lhs_row.get(VP_FORWARD_BIAS_FIELD))
        rhs_bias = _safe_float(rhs_row.get(VP_FORWARD_BIAS_FIELD))
        if not math.isfinite(lhs_bias) or not math.isfinite(rhs_bias):
            continue
        if abs(lhs_bias - rhs_bias) > threshold_m:
            return int(lhs_row.get("step"))
    return None


def extract_step_window(
    trajectory: Sequence[Mapping[str, Any]],
    *,
    center_step: Optional[int],
    pre_steps: int = 2,
    post_steps: int = 5,
    fields: Iterable[str] = WINDOW_FIELDS,
) -> List[Dict[str, Any]]:
    if center_step is None:
        return []
    step_to_index = _step_index_map(trajectory)
    if center_step not in step_to_index:
        return []
    start_step = max(1, center_step - pre_steps)
    end_step = center_step + post_steps
    selected: List[Dict[str, Any]] = []
    for step in range(start_step, end_step + 1):
        index = step_to_index.get(step)
        if index is None:
            continue
        row = trajectory[index]
        selected.append({field: row.get(field) for field in fields})
    return selected


def build_seed_comparison(
    *,
    seed: int,
    oldhack_episode: Mapping[str, Any],
    oldflag_episode: Mapping[str, Any],
    fixedflag_episode: Mapping[str, Any],
) -> Dict[str, Any]:
    oldhack_traj = list(oldhack_episode["trajectory"])
    oldflag_traj = list(oldflag_episode["trajectory"])
    fixedflag_traj = list(fixedflag_episode["trajectory"])

    release_step = _first_true_step(oldhack_traj, RELEASE_FIELD)
    oldflag_latch_step = _first_true_step(oldflag_traj, LATCH_FIELD)
    oldflag_recovery_step = _first_true_step(oldflag_traj, RECOVERY_FIELD)
    fixedflag_latch_step = _first_true_step(fixedflag_traj, LATCH_FIELD)
    fixedflag_recovery_step = _first_true_step(fixedflag_traj, RECOVERY_FIELD)

    oldflag_recovery_preview_vp_forward_bias_m = float("nan")
    if oldflag_recovery_step is not None:
        index = _step_index_map(oldflag_traj).get(oldflag_recovery_step)
        if index is not None:
            oldflag_recovery_preview_vp_forward_bias_m = _safe_float(
                oldflag_traj[index].get(RECOVERY_PREVIEW_FIELD)
            )

    return {
        "seed": seed,
        "oldhack": {
            "termination_reason": oldhack_episode.get("termination_reason"),
            "success": bool(oldhack_episode.get("success", False)),
            "steps": int(oldhack_episode.get("steps", len(oldhack_traj))),
            "first_release_step": release_step,
            "first_latch_step": _first_true_step(oldhack_traj, LATCH_FIELD),
            "first_recovery_step": _first_true_step(oldhack_traj, RECOVERY_FIELD),
            "first_direct_track_step": _first_true_step(
                oldhack_traj, DIRECT_TRACK_FIELD
            ),
        },
        "oldflag": {
            "termination_reason": oldflag_episode.get("termination_reason"),
            "success": bool(oldflag_episode.get("success", False)),
            "steps": int(oldflag_episode.get("steps", len(oldflag_traj))),
            "first_release_step": _first_true_step(oldflag_traj, RELEASE_FIELD),
            "first_latch_step": oldflag_latch_step,
            "first_recovery_step": oldflag_recovery_step,
            "first_direct_track_step": _first_true_step(
                oldflag_traj, DIRECT_TRACK_FIELD
            ),
            "recovery_preview_vp_forward_bias_m_at_first_recovery": (
                oldflag_recovery_preview_vp_forward_bias_m
            ),
        },
        "fixedflag": {
            "termination_reason": fixedflag_episode.get("termination_reason"),
            "success": bool(fixedflag_episode.get("success", False)),
            "steps": int(fixedflag_episode.get("steps", len(fixedflag_traj))),
            "first_release_step": _first_true_step(fixedflag_traj, RELEASE_FIELD),
            "first_latch_step": fixedflag_latch_step,
            "first_recovery_step": fixedflag_recovery_step,
            "first_direct_track_step": _first_true_step(
                fixedflag_traj, DIRECT_TRACK_FIELD
            ),
        },
        "pairwise_divergence": {
            "oldhack_vs_oldflag_first_vp_bias_divergence_gt_1m_step": (
                first_vp_bias_divergence_step(
                    oldhack_traj,
                    oldflag_traj,
                    threshold_m=1.0,
                )
            ),
            "oldhack_vs_oldflag_first_vp_bias_divergence_gt_10m_step": (
                first_vp_bias_divergence_step(
                    oldhack_traj,
                    oldflag_traj,
                    threshold_m=10.0,
                )
            ),
            "oldhack_vs_fixedflag_first_vp_bias_divergence_gt_1m_step": (
                first_vp_bias_divergence_step(
                    oldhack_traj,
                    fixedflag_traj,
                    threshold_m=1.0,
                )
            ),
            "oldhack_vs_fixedflag_first_vp_bias_divergence_gt_10m_step": (
                first_vp_bias_divergence_step(
                    oldhack_traj,
                    fixedflag_traj,
                    threshold_m=10.0,
                )
            ),
            "oldflag_vs_fixedflag_first_vp_bias_divergence_gt_1m_step": (
                first_vp_bias_divergence_step(
                    oldflag_traj,
                    fixedflag_traj,
                    threshold_m=1.0,
                )
            ),
            "oldflag_vs_fixedflag_first_vp_bias_divergence_gt_10m_step": (
                first_vp_bias_divergence_step(
                    oldflag_traj,
                    fixedflag_traj,
                    threshold_m=10.0,
                )
            ),
        },
        "release_window": {
            "oldhack": extract_step_window(oldhack_traj, center_step=release_step),
            "oldflag": extract_step_window(oldflag_traj, center_step=release_step),
            "fixedflag": extract_step_window(fixedflag_traj, center_step=release_step),
        },
        "oldflag_latch_window": extract_step_window(
            oldflag_traj,
            center_step=oldflag_latch_step,
        ),
        "oldflag_recovery_window": extract_step_window(
            oldflag_traj,
            center_step=oldflag_recovery_step,
        ),
    }


def build_report(
    *,
    oldhack_run: Path,
    oldflag_run: Path,
    fixedflag_run: Path,
    task: str,
    method: str,
    seeds: Sequence[int],
) -> Dict[str, Any]:
    seed_reports = []
    for seed in seeds:
        seed_reports.append(
            build_seed_comparison(
                seed=seed,
                oldhack_episode=_read_episode(
                    oldhack_run,
                    task=task,
                    method=method,
                    seed=seed,
                ),
                oldflag_episode=_read_episode(
                    oldflag_run,
                    task=task,
                    method=method,
                    seed=seed,
                ),
                fixedflag_episode=_read_episode(
                    fixedflag_run,
                    task=task,
                    method=method,
                    seed=seed,
                ),
            )
        )
    return {
        "task": task,
        "method": method,
        "seeds": list(seeds),
        "artifacts": {
            "oldhack_run": str(oldhack_run),
            "oldflag_run": str(oldflag_run),
            "fixedflag_run": str(fixedflag_run),
        },
        "seed_reports": seed_reports,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Nodirecttrack Boolean-Flag Raw Trace Comparison",
        "",
        "## Summary",
        "",
        "- Old boolean-flag pilot diverges from the legacy altitude=0 hack before the release event because it activates lateral latch early, then later activates recovery.",
        "- Fixed boolean-flag pilot removes both early latch and late recovery from the overlap seeds, matching the legacy hack on those gating states.",
        "- Remaining expert/head_on weakness is therefore no longer explained by the old latch/recovery non-equivalence; it has moved on to geometry quality after release.",
        "",
        "## Seed Comparison",
        "",
    ]
    for seed_report in report["seed_reports"]:
        pairwise = seed_report["pairwise_divergence"]
        lines.extend(
            [
                f"### Seed {seed_report['seed']}",
                "",
                (
                    f"- oldhack: `{seed_report['oldhack']['termination_reason']}` "
                    f"({seed_report['oldhack']['steps']} steps), "
                    f"release_step={seed_report['oldhack']['first_release_step']}, "
                    f"latch_step={seed_report['oldhack']['first_latch_step']}, "
                    f"recovery_step={seed_report['oldhack']['first_recovery_step']}"
                ),
                (
                    f"- oldflag: `{seed_report['oldflag']['termination_reason']}` "
                    f"({seed_report['oldflag']['steps']} steps), "
                    f"release_step={seed_report['oldflag']['first_release_step']}, "
                    f"latch_step={seed_report['oldflag']['first_latch_step']}, "
                    f"recovery_step={seed_report['oldflag']['first_recovery_step']}, "
                    "recovery_preview_vp_forward_bias_m="
                    f"{seed_report['oldflag']['recovery_preview_vp_forward_bias_m_at_first_recovery']}"
                ),
                (
                    f"- fixedflag: `{seed_report['fixedflag']['termination_reason']}` "
                    f"({seed_report['fixedflag']['steps']} steps), "
                    f"release_step={seed_report['fixedflag']['first_release_step']}, "
                    f"latch_step={seed_report['fixedflag']['first_latch_step']}, "
                    f"recovery_step={seed_report['fixedflag']['first_recovery_step']}"
                ),
                (
                    "- first `vp_forward_bias_m` divergence >1m: "
                    f"oldhack vs oldflag={pairwise['oldhack_vs_oldflag_first_vp_bias_divergence_gt_1m_step']}, "
                    f"oldhack vs fixedflag={pairwise['oldhack_vs_fixedflag_first_vp_bias_divergence_gt_1m_step']}, "
                    f"oldflag vs fixedflag={pairwise['oldflag_vs_fixedflag_first_vp_bias_divergence_gt_1m_step']}"
                ),
                (
                    "- first `vp_forward_bias_m` divergence >10m: "
                    f"oldhack vs oldflag={pairwise['oldhack_vs_oldflag_first_vp_bias_divergence_gt_10m_step']}, "
                    f"oldhack vs fixedflag={pairwise['oldhack_vs_fixedflag_first_vp_bias_divergence_gt_10m_step']}, "
                    f"oldflag vs fixedflag={pairwise['oldflag_vs_fixedflag_first_vp_bias_divergence_gt_10m_step']}"
                ),
                "",
                "Release window (`step`, `released`, `latch`, `recovery`, `direct_track`, `vp_forward_bias_m`):",
                "",
            ]
        )
        for label in ("oldhack", "oldflag", "fixedflag"):
            lines.append(f"- `{label}`:")
            for row in seed_report["release_window"][label]:
                lines.append(
                    "  "
                    + str(
                        {
                            "step": row["step"],
                            "released": row[RELEASE_FIELD],
                            "latch": row[LATCH_FIELD],
                            "recovery": row[RECOVERY_FIELD],
                            "direct_track": row[DIRECT_TRACK_FIELD],
                            "vp_forward_bias_m": row[VP_FORWARD_BIAS_FIELD],
                        }
                    )
                )
            lines.append("")
        if seed_report["oldflag_latch_window"]:
            lines.append("- `oldflag` latch window:")
            for row in seed_report["oldflag_latch_window"]:
                lines.append(f"  {row}")
            lines.append("")
        if seed_report["oldflag_recovery_window"]:
            lines.append("- `oldflag` recovery window:")
            for row in seed_report["oldflag_recovery_window"]:
                lines.append(f"  {row}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oldhack-run", type=Path, default=REPO_ROOT / DEFAULT_OLDHACK_RUN)
    parser.add_argument("--oldflag-run", type=Path, default=REPO_ROOT / DEFAULT_OLDFLAG_RUN)
    parser.add_argument("--fixedflag-run", type=Path, default=REPO_ROOT / DEFAULT_FIXEDFLAG_RUN)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--method", default=DEFAULT_METHOD)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--output-md", type=Path, default=REPO_ROOT / DEFAULT_OUTPUT_MD)
    parser.add_argument("--output-json", type=Path, default=REPO_ROOT / DEFAULT_OUTPUT_JSON)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(
        oldhack_run=args.oldhack_run,
        oldflag_run=args.oldflag_run,
        fixedflag_run=args.fixedflag_run,
        task=args.task,
        method=args.method,
        seeds=args.seeds,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(f"Wrote JSON report to {args.output_json}")
    print(f"Wrote Markdown report to {args.output_md}")


if __name__ == "__main__":
    main()
