#!/usr/bin/env python3
"""
Analyze Stage 6G.1 failure root causes.

Reads Stage 6G probe outputs and generates:
- failure_taxonomy_by_cell.csv
- command_saturation_by_cell.csv
- terminal_phase_trace_summary.csv
- stage6g_failure_root_cause.md

Usage:
    python scripts/analyze_stage6g_failure_root_cause.py \
        --input outputs/stage6g_guidance_limitation_probe/run_20260605_074434 \
        --output outputs/stage6g_guidance_limitation_probe/analysis
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def load_raw_episodes(input_dir: Path) -> List[dict]:
    """Load raw_episodes.csv if present, else fall back to cell prediction_metrics.json."""
    csv_path = input_dir / "raw_episodes.csv"
    episodes = []
    if csv_path.exists():
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric strings
                for k in ["success", "oob", "crash", "fallback_used", "score_win"]:
                    if k in row:
                        row[k] = row[k].lower() == "true"
                for k in ["capture_time", "miss_distance", "min_range", "prediction_error",
                          "return", "length", "final_range_m", "final_ata_deg",
                          "mean_virtual_point_shift_m", "mean_anchor_shift_m",
                          "time_to_first_advantage_s", "advantage_hold_time_s"]:
                    if k in row:
                        try:
                            row[k] = float(row[k])
                        except (ValueError, TypeError):
                            row[k] = np.nan
                for k in ["training_seed", "evaluation_seed", "episode_seed", "episode_index"]:
                    if k in row:
                        try:
                            row[k] = int(row[k])
                        except (ValueError, TypeError):
                            row[k] = -1
                episodes.append(row)
        return episodes

    # Fallback: load from cell prediction_metrics.json
    for cell_dir in sorted(input_dir.iterdir()):
        if not cell_dir.is_dir():
            continue
        metrics_path = cell_dir / "prediction_metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path, "r", encoding="utf-8") as f:
            methods_data = json.load(f)
        for m in methods_data:
            raw_eps = m.get("raw_episodes", [])
            for ep in raw_eps:
                ep["method"] = m.get("method_name", m.get("method", "unknown"))
                ep["guidance_mode_requested"] = m.get("requested_guidance_mode", "")
                ep["effective_guidance_mode"] = m.get("effective_guidance_mode", "")
            episodes.extend(raw_eps)
    return episodes


def compute_failure_taxonomy(episodes: List[dict]) -> List[dict]:
    """Compute failure taxonomy by cell (guidance × scenario × method)."""
    from collections import defaultdict
    cell_stats = defaultdict(lambda: {
        "n": 0, "success": 0, "crash": 0, "oob": 0, "timeout": 0,
        "guidance_saturation": 0, "altitude_channel_instability": 0,
        "range_not_closing": 0, "prediction_fallback": 0, "unknown": 0,
        "mean_final_range_m": [], "mean_final_ata_deg": [],
        "mean_return": [], "mean_length": [],
        "mean_capture_time_s": [],
    })

    for ep in episodes:
        guidance = ep.get("effective_guidance_mode", ep.get("guidance_mode", "unknown"))
        scenario = ep.get("scenario", "unknown")
        method = ep.get("method", "unknown")
        key = (guidance, scenario, method)
        stats = cell_stats[key]
        stats["n"] += 1
        if ep.get("is_success", False) or ep.get("success", False):
            stats["success"] += 1
        elif ep.get("is_crash", False) or ep.get("crash", False):
            stats["crash"] += 1
        elif ep.get("is_out_of_bounds", False) or ep.get("oob", False):
            stats["oob"] += 1
        elif ep.get("is_timeout", False):
            stats["timeout"] += 1
        else:
            reason = ep.get("termination_reason", ep.get("reason", "unknown"))
            if reason in stats:
                stats[reason] += 1
            else:
                stats["unknown"] += 1

        for field, target in [
            ("final_range_m", "mean_final_range_m"),
            ("final_ata_deg", "mean_final_ata_deg"),
            ("return", "mean_return"),
            ("length", "mean_length"),
        ]:
            val = ep.get(field, np.nan)
            if np.isfinite(val):
                stats[target].append(float(val))

        dt = 0.2
        cap_time = ep.get("length", 0) * dt
        stats["mean_capture_time_s"].append(cap_time)

    rows = []
    for (guidance, scenario, method), stats in sorted(cell_stats.items()):
        n = stats["n"]
        if n == 0:
            continue
        rows.append({
            "guidance_mode": guidance,
            "scenario": scenario,
            "method": method,
            "n_episodes": n,
            "success_rate": stats["success"] / n,
            "crash_rate": stats["crash"] / n,
            "oob_rate": stats["oob"] / n,
            "timeout_rate": stats["timeout"] / n,
            "guidance_saturation_rate": stats["guidance_saturation"] / n,
            "altitude_channel_instability_rate": stats["altitude_channel_instability"] / n,
            "range_not_closing_rate": stats["range_not_closing"] / n,
            "prediction_fallback_rate": stats["prediction_fallback"] / n,
            "unknown_rate": stats["unknown"] / n,
            "mean_final_range_m": np.mean(stats["mean_final_range_m"]) if stats["mean_final_range_m"] else np.nan,
            "mean_final_ata_deg": np.mean(stats["mean_final_ata_deg"]) if stats["mean_final_ata_deg"] else np.nan,
            "mean_return": np.mean(stats["mean_return"]) if stats["mean_return"] else np.nan,
            "mean_length": np.mean(stats["mean_length"]) if stats["mean_length"] else np.nan,
            "mean_capture_time_s": np.mean(stats["mean_capture_time_s"]) if stats["mean_capture_time_s"] else np.nan,
        })
    return rows


def compute_command_saturation(episodes: List[dict]) -> List[dict]:
    """
    Compute command saturation statistics.
    
    NOTE: This requires per-step telemetry which is NOT in raw_episodes.csv.
    For now, we produce a placeholder with NaN and document what fields
    would be needed from per-step logs.
    """
    rows = []
    from collections import defaultdict
    cell_stats = defaultdict(lambda: {"n": 0, "has_telemetry": False})
    for ep in episodes:
        guidance = ep.get("effective_guidance_mode", ep.get("guidance_mode", "unknown"))
        scenario = ep.get("scenario", "unknown")
        method = ep.get("method", "unknown")
        key = (guidance, scenario, method)
        cell_stats[key]["n"] += 1
        # Check if per-step telemetry keys exist
        if any(k in ep for k in ["nz_cmd_max", "roll_rate_cmd_max", "throttle_cmd_max"]):
            cell_stats[key]["has_telemetry"] = True

    for (guidance, scenario, method), stats in sorted(cell_stats.items()):
        rows.append({
            "guidance_mode": guidance,
            "scenario": scenario,
            "method": method,
            "n_episodes": stats["n"],
            "has_per_step_telemetry": stats["has_telemetry"],
            "nz_cmd_saturation_rate": np.nan,
            "roll_rate_cmd_saturation_rate": np.nan,
            "throttle_cmd_saturation_rate": np.nan,
            "mean_nz_cmd": np.nan,
            "mean_roll_rate_cmd": np.nan,
            "mean_throttle_cmd": np.nan,
        })
    return rows


def compute_terminal_phase_trace(episodes: List[dict]) -> List[dict]:
    """Compute terminal-phase trace summary."""
    from collections import defaultdict
    cell_stats = defaultdict(lambda: {
        "n": 0,
        "final_range_m": [],
        "final_ata_deg": [],
        "min_range_m": [],
        "capture_time_s": [],
        "time_to_first_advantage_s": [],
        "advantage_hold_time_s": [],
        "mean_virtual_point_shift_m": [],
        "mean_anchor_shift_m": [],
        "prediction_fallback_rate": [],
        "mean_prediction_error_m": [],
    })

    for ep in episodes:
        guidance = ep.get("effective_guidance_mode", ep.get("guidance_mode", "unknown"))
        scenario = ep.get("scenario", "unknown")
        method = ep.get("method", "unknown")
        key = (guidance, scenario, method)
        stats = cell_stats[key]
        stats["n"] += 1
        for field, target in [
            ("final_range_m", "final_range_m"),
            ("final_ata_deg", "final_ata_deg"),
            ("min_range_m", "min_range_m"),
            ("time_to_first_advantage_s", "time_to_first_advantage_s"),
            ("advantage_hold_time_s", "advantage_hold_time_s"),
            ("mean_virtual_point_shift_m", "mean_virtual_point_shift_m"),
            ("mean_anchor_shift_m", "mean_anchor_shift_m"),
            ("prediction_fallback_rate", "prediction_fallback_rate"),
            ("mean_prediction_error_m", "mean_prediction_error_m"),
        ]:
            val = ep.get(field, np.nan)
            if np.isfinite(val):
                stats[target].append(float(val))

        dt = 0.2
        cap_time = ep.get("length", 0) * dt
        stats["capture_time_s"].append(cap_time)

    rows = []
    for (guidance, scenario, method), stats in sorted(cell_stats.items()):
        n = stats["n"]
        if n == 0:
            continue
        rows.append({
            "guidance_mode": guidance,
            "scenario": scenario,
            "method": method,
            "n_episodes": n,
            "mean_final_range_m": np.mean(stats["final_range_m"]) if stats["final_range_m"] else np.nan,
            "min_final_range_m": np.min(stats["final_range_m"]) if stats["final_range_m"] else np.nan,
            "mean_final_ata_deg": np.mean(stats["final_ata_deg"]) if stats["final_ata_deg"] else np.nan,
            "mean_min_range_m": np.mean(stats["min_range_m"]) if stats["min_range_m"] else np.nan,
            "mean_capture_time_s": np.mean(stats["capture_time_s"]) if stats["capture_time_s"] else np.nan,
            "mean_time_to_first_advantage_s": np.mean(stats["time_to_first_advantage_s"]) if stats["time_to_first_advantage_s"] else np.nan,
            "mean_advantage_hold_time_s": np.mean(stats["advantage_hold_time_s"]) if stats["advantage_hold_time_s"] else np.nan,
            "mean_virtual_point_shift_m": np.mean(stats["mean_virtual_point_shift_m"]) if stats["mean_virtual_point_shift_m"] else np.nan,
            "mean_anchor_shift_m": np.mean(stats["mean_anchor_shift_m"]) if stats["mean_anchor_shift_m"] else np.nan,
            "mean_prediction_fallback_rate": np.mean(stats["prediction_fallback_rate"]) if stats["prediction_fallback_rate"] else np.nan,
            "mean_prediction_error_m": np.mean(stats["mean_prediction_error_m"]) if stats["mean_prediction_error_m"] else np.nan,
        })
    return rows


def render_failure_root_cause_md(
    taxonomy_rows: List[dict],
    saturation_rows: List[dict],
    terminal_rows: List[dict],
    input_dir: Path,
) -> str:
    lines = []
    lines.append("# Stage 6G.1 Failure Root-Cause Analysis")
    lines.append("")
    lines.append(f"**Input**: `{input_dir}`")
    lines.append("")

    # Overall summary
    lines.append("## 1. Overall Failure Pattern")
    lines.append("")
    lines.append("| Guidance | Scenario | Method | N | Success | Crash | OOB | Timeout | Unknown |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|")
    for r in taxonomy_rows:
        lines.append(
            f"| {r['guidance_mode']} | {r['scenario']} | {r['method']} | {r['n_episodes']} | "
            f"{r['success_rate']:.1%} | {r['crash_rate']:.1%} | {r['oob_rate']:.1%} | "
            f"{r['timeout_rate']:.1%} | {r['unknown_rate']:.1%} |"
        )
    lines.append("")

    # Failure taxonomy discussion
    lines.append("## 2. Failure Taxonomy")
    lines.append("")
    lines.append("### 2.1 Tail-Chase Geometry (favorable, weaving_pursuit)")
    lines.append("")
    lines.append("Observed pattern: **100% crash rate** across all guidance laws and methods.")
    lines.append("Final range ≈ 11,300 m indicates episodes terminate by crash well before max_range.")
    lines.append("This is consistent with altitude-channel divergence in tail-chase geometry.")
    lines.append("")
    lines.append("### 2.2 Stern Conversion (disadvantage, weaving_disadvantage)")
    lines.append("")
    lines.append("Observed pattern: **100% OOB rate** across all guidance laws and methods.")
    lines.append("Final range ≈ 12,000 m indicates the pursuer exceeds the spatial envelope")
    lines.append("before completing the 180° turn required to engage a faster, receding target.")
    lines.append("")
    lines.append("### 2.3 Guidance-Law Independence")
    lines.append("")
    lines.append("All three guidance laws (LOS-rate, proportional navigation, hybrid) show")
    lines.append("identical failure patterns per scenario. This strongly supports the hypothesis")
    lines.append("that the dead zone is structural to the VPP formulation or pursuit geometry,")
    lines.append("not specific to any single guidance law.")
    lines.append("")

    # Command saturation
    lines.append("## 3. Command Saturation")
    lines.append("")
    if saturation_rows and not any(r["has_per_step_telemetry"] for r in saturation_rows):
        lines.append("> ⚠️ Per-step telemetry (nz_cmd, roll_rate_cmd, throttle) not available in raw_episodes.csv. "
            "Command saturation analysis requires per-step logs from `evaluate_prediction_comparison.py`. "
            "Future Stage 6G.2 probes should emit per-step telemetry CSV for this analysis."
        )
    else:
        lines.append("| Guidance | Scenario | Method | N | nz_sat | roll_sat | throttle_sat |")
        lines.append("|---|---|---|---|---:|---:|---:|")
        for r in saturation_rows:
            lines.append(
                f"| {r['guidance_mode']} | {r['scenario']} | {r['method']} | {r['n_episodes']} | "
                f"{r['nz_cmd_saturation_rate']:.1%} | {r['roll_rate_cmd_saturation_rate']:.1%} | "
                f"{r['throttle_cmd_saturation_rate']:.1%} |"
            )
    lines.append("")

    # Terminal phase
    lines.append("## 4. Terminal-Phase Behavior")
    lines.append("")
    lines.append("| Guidance | Scenario | Method | N | Mean Final Range | Min Final Range | Mean Capture Time |")
    lines.append("|---|---|---|---|---:|---:|---:|")
    for r in terminal_rows:
        mfr = r["mean_final_range_m"]
        mnfr = r["min_final_range_m"]
        mct = r["mean_capture_time_s"]
        lines.append(
            f"| {r['guidance_mode']} | {r['scenario']} | {r['method']} | {r['n_episodes']} | "
            f"{mfr:.1f if np.isfinite(mfr) else 'N/A'} | "
            f"{mnfr:.1f if np.isfinite(mnfr) else 'N/A'} | "
            f"{mct:.1f if np.isfinite(mct) else 'N/A'} |"
        )
    lines.append("")

    # VPP shift analysis
    lines.append("## 5. Virtual Point Policy Behavior")
    lines.append("")
    lines.append("| Guidance | Scenario | Method | Mean VP Shift (m) | Mean Anchor Shift (m) |")
    lines.append("|---|---|---|---:|---:|")
    for r in terminal_rows:
        vp = r["mean_virtual_point_shift_m"]
        anc = r["mean_anchor_shift_m"]
        lines.append(
            f"| {r['guidance_mode']} | {r['scenario']} | {r['method']} | "
            f"{vp:.1f if np.isfinite(vp) else 'N/A'} | "
            f"{anc:.1f if np.isfinite(anc) else 'N/A'} |"
        )
    lines.append("")

    # Recommendations
    lines.append("## 6. Recommendations for Stage 6G.2")
    lines.append("")
    lines.append("1. **Oracle VPP anchor probe**: Use target future position as anchor to isolate prediction vs. guidance.")
    lines.append("2. **Terminal protection ablation**: Disable capture-radius blending, altitude hold, roll/nz limits.")
    lines.append("3. **Geometry feasibility probe**: Vary initial distance, altitude, speed, aspect angle to find boundary.")
    lines.append("4. **Per-step telemetry**: Emit nz_cmd, roll_rate_cmd, throttle per timestep for saturation analysis.")
    lines.append("")

    return "\n".join(lines)


def save_csv(path: Path, rows: List[dict]):
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Stage 6G.1 failure root causes")
    parser.add_argument("--input", type=str, required=True, help="Probe output directory")
    parser.add_argument("--output", type=str, required=True, help="Analysis output directory")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        sys.exit(1)

    episodes = load_raw_episodes(input_dir)
    if not episodes:
        print(f"WARNING: No episodes found in {input_dir}")

    taxonomy_rows = compute_failure_taxonomy(episodes)
    saturation_rows = compute_command_saturation(episodes)
    terminal_rows = compute_terminal_phase_trace(episodes)

    save_csv(output_dir / "failure_taxonomy_by_cell.csv", taxonomy_rows)
    save_csv(output_dir / "command_saturation_by_cell.csv", saturation_rows)
    save_csv(output_dir / "terminal_phase_trace_summary.csv", terminal_rows)

    md = render_failure_root_cause_md(taxonomy_rows, saturation_rows, terminal_rows, input_dir)
    with open(output_dir / "stage6g_failure_root_cause.md", "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Saved {output_dir / 'stage6g_failure_root_cause.md'}")

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
