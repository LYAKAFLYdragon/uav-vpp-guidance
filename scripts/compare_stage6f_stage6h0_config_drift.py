#!/usr/bin/env python3
"""
Stage 6H.0-R: Audit config drift between Stage 6F.5A (success)
and Stage 6H.0-lite baseline search (failure).

Identifies specific config deltas that could explain why the same
codebase+checkpoint found successes in 6F but zero candidates in 6H.0.

Outputs:
    docs/results/stage6h0r_config_drift_audit.json
    docs/results/stage6h0r_config_drift_audit.md
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _deep_diff(path_a, val_a, val_b, prefix=""):
    """Return flat list of diff tuples: (path, stage6f_value, stage6h0_value, severity)."""
    diffs = []

    if isinstance(val_a, dict) and isinstance(val_b, dict):
        all_keys = set(val_a) | set(val_b)
        for k in sorted(all_keys):
            new_prefix = f"{prefix}.{k}" if prefix else k
            if k not in val_a:
                diffs.append((new_prefix, "MISSING", val_b[k], "added_in_6h0"))
            elif k not in val_b:
                diffs.append((new_prefix, val_a[k], "MISSING", "removed_in_6h0"))
            else:
                diffs.extend(_deep_diff(new_prefix, val_a[k], val_b[k], new_prefix))
    elif isinstance(val_a, list) and isinstance(val_b, list):
        if len(val_a) != len(val_b) or val_a != val_b:
            diffs.append((prefix, val_a, val_b, "modified"))
    elif val_a != val_b:
        diffs.append((prefix, val_a, val_b, "modified"))

    return diffs


def _parse_scenario_geometry(scen):
    """Extract range, aspect angle, closure rate from scenario config."""
    own_pos = scen.get("own_init", {}).get("position_m", [0, 0, 0])
    tgt_pos = scen.get("target_init", {}).get("position_m", [0, 0, 0])
    own_vel = scen.get("own_init", {}).get("velocity_mps", 0)
    tgt_vel = scen.get("target_init", {}).get("velocity_mps", 0)
    own_hdg = scen.get("own_init", {}).get("heading_deg", 0)
    tgt_hdg = scen.get("target_init", {}).get("heading_deg", 0)

    import math
    dx = own_pos[0] - tgt_pos[0]
    dy = own_pos[1] - tgt_pos[1]
    range_m = math.hypot(dx, dy)

    # Approximate aspect angle (difference in heading)
    aspect_deg = abs(own_hdg - tgt_hdg) % 360
    if aspect_deg > 180:
        aspect_deg = 360 - aspect_deg

    # Approximate closure rate (component along line of sight)
    los_angle = math.atan2(dy, dx)
    own_vx = own_vel * math.cos(math.radians(own_hdg))
    own_vy = own_vel * math.sin(math.radians(own_hdg))
    tgt_vx = tgt_vel * math.cos(math.radians(tgt_hdg))
    tgt_vy = tgt_vel * math.sin(math.radians(tgt_hdg))
    closure = (own_vx - tgt_vx) * math.cos(los_angle) + (own_vy - tgt_vy) * math.sin(los_angle)

    return {
        "range_m": round(range_m, 1),
        "aspect_deg": round(aspect_deg, 1),
        "closure_rate_mps": round(closure, 1),
        "own_vel": own_vel,
        "tgt_vel": tgt_vel,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="docs/results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load configs
    s6f_path = PROJECT_ROOT / "config" / "experiment" / "stage6f5_feasible_geometry.yaml"
    s6h_path = PROJECT_ROOT / "config" / "experiment" / "stage6g5_wide_geometry_smoke.yaml"

    s6f = yaml.safe_load(s6f_path.read_text(encoding="utf-8"))
    s6h = yaml.safe_load(s6h_path.read_text(encoding="utf-8"))

    # Flat diff
    diffs = _deep_diff("", s6f, s6h)

    # Geometry analysis
    s6f_geoms = {}
    for name, scen in s6f.get("scenarios", {}).items():
        s6f_geoms[name] = _parse_scenario_geometry(scen)

    s6h_geoms = {}
    for name, scen in s6h.get("scenarios", {}).items():
        s6h_geoms[name] = _parse_scenario_geometry(scen)

    # Classify s6f geometries against s6h search space
    s6h_search = s6h.get("geometry_search", {})
    s6h_ranges = s6h_search.get("initial_range_m", [])
    s6h_aspects = s6h_search.get("aspect_angles_deg", [])
    s6h_speeds = s6h_search.get("ego_speeds_mps", [])

    classification = {}
    for name, geom in s6f_geoms.items():
        in_range = min(s6h_ranges) <= geom["range_m"] <= max(s6h_ranges) if s6h_ranges else False
        in_aspect = min(s6h_aspects) <= geom["aspect_deg"] <= max(s6h_aspects) if s6h_aspects else False
        in_speed = min(s6h_speeds) <= geom["own_vel"] <= max(s6h_speeds) if s6h_speeds else False
        classification[name] = {
            "geometry": geom,
            "in_range": in_range,
            "in_aspect": in_aspect,
            "in_speed": in_speed,
            "covered_by_search": in_range and in_aspect and in_speed,
        }

    audit = {
        "audit_date": datetime.now().isoformat(),
        "diff_summary": {
            "total_differences": len(diffs),
            "critical_keys": [],
            "moderate_keys": [],
            "cosmetic_keys": [],
        },
        "critical_differences": [],
        "stage6f_geometry_coverage": classification,
    }

    # Classify critical vs cosmetic
    critical_env_keys = {
        "success_range_m", "success_ata_deg", "success_hold_time_s",
        "hysteresis_range_m", "hysteresis_ata_deg", "max_range_m",
        "max_high_level_steps", "high_level_dt", "backend", "use_jsbsim",
    }
    critical_guidance_keys = {"guidance.mode", "guidance.gains"}

    for path, v6f, v6h, severity in diffs:
        if severity != "modified":
            continue
        is_critical = any(path.endswith(k) or path == k for k in critical_env_keys)
        is_critical = is_critical or any(path.startswith(k) for k in critical_guidance_keys)

        diff_record = {
            "path": path,
            "stage6f_value": v6f,
            "stage6h0_value": v6h,
            "critical": is_critical,
        }
        audit["critical_differences"].append(diff_record)
        if is_critical:
            audit["diff_summary"]["critical_keys"].append(path)
        else:
            audit["diff_summary"]["moderate_keys"].append(path)

    # Save JSON
    json_path = output_dir / "stage6h0r_config_drift_audit.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, default=str)

    # Save Markdown
    md_path = output_dir / "stage6h0r_config_drift_audit.md"
    lines = [
        "# Stage 6H.0-R: Config Drift Audit — 6F.5A vs 6H.0-lite",
        "",
        f"**Audit date**: {datetime.now().isoformat()}",
        "",
        "## 1. Executive Summary",
        "",
        f"- **Total differences**: {len(diffs)}",
        f"- **Critical (affects physics / success criteria)**: {len(audit['diff_summary']['critical_keys'])}",
        f"- **Moderate (naming, logging, output)**: {len(audit['diff_summary']['moderate_keys'])}",
        "",
        "## 2. Critical Differences",
        "",
        "| Config Path | Stage 6F.5A | Stage 6H.0-lite | Impact |",
        "|---|---|---|---|",
    ]
    for rec in audit["critical_differences"]:
        if rec["critical"]:
            lines.append(
                f"| `{rec['path']}` | {rec['stage6f_value']} | "
                f"{rec['stage6h0_value']} | **CRITICAL** |"
            )

    lines.extend([
        "",
        "## 3. Stage 6F Scenarios vs 6H.0 Search Space",
        "",
        "| Scenario | Range (m) | Aspect (°) | Closure (m/s) | Covered by 6H Search? |",
        "|---|---|---|---|---|",
    ])
    for name, info in classification.items():
        g = info["geometry"]
        covered = "✅ Yes" if info["covered_by_search"] else "❌ No"
        lines.append(
            f"| {name} | {g['range_m']} | {g['aspect_deg']} | {g['closure_rate_mps']} | {covered} |"
        )

    lines.extend([
        "",
        "## 4. Key Finding",
        "",
    ])

    any_covered = any(info["covered_by_search"] for info in classification.values())
    if not any_covered:
        lines.append(
            "**ALL Stage 6F scenarios fall OUTSIDE the Stage 6H.0-lite search space.**\n"
            "The baseline search did not evaluate any geometry similar to those that succeeded in 6F.5A.\n"
            "This is the root cause of the zero-candidate result — not a regression in VPP performance."
        )
    else:
        lines.append(
            "Some Stage 6F scenarios ARE within the 6H.0 search space.\n"
            "The zero-candidate result may indicate a genuine regression or checkpoint mismatch."
        )

    lines.extend([
        "",
        "## 5. Recommendations",
        "",
        "1. **Expand 6H.0 search** to include smaller ranges (800–1200m) and tail-chase aspects (0°).",
        "2. **Replay Stage 6F scenarios** with the current checkpoint to confirm no code regression.",
        "3. **Do not claim VPP has no non-tail-chase feasible region** until the search space covers historical successes.",
    ])

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Drift audit JSON: {json_path}")
    print(f"Drift audit MD: {md_path}")
    print(f"Critical diffs: {len(audit['diff_summary']['critical_keys'])}")
    print(f"Covered scenarios: {sum(1 for v in classification.values() if v['covered_by_search'])}/{len(classification)}")


if __name__ == "__main__":
    main()
