"""
Telemetry schema validator for episode-level and per-step data.

Ensures failure root-cause analysis scripts do not silently produce
weak or misleading reports when key telemetry fields are missing.
"""

from typing import Dict, List, Tuple, Optional, Set
import json


# Fields that MUST be present for any meaningful root-cause analysis
CORE_EPISODE_FIELDS = {
    "scenario",
    "method",
    "guidance_mode",
    "training_seed",
    "evaluation_seed",
    "episode_index",
    "is_success",
    "is_crash",
    "is_out_of_bounds",
    "is_timeout",
    "reason",
    "return",
    "length",
    "final_range_m",
    "final_ata_deg",
}

# Fields needed for terminal-phase / geometry analysis
TERMINAL_PHASE_FIELDS = {
    "min_range_m",
    "time_to_first_advantage_s",
    "advantage_hold_time_s",
    "mean_virtual_point_shift_m",
    "mean_anchor_shift_m",
}

# Fields needed for prediction analysis
PREDICTION_FIELDS = {
    "prediction_valid_rate",
    "prediction_fallback_rate",
    "mean_prediction_error_m",
    "prediction_error_count",
}

# Fields needed for command saturation analysis
# NOTE: These require PER-STEP telemetry, which is NOT currently emitted.
COMMAND_SATURATION_FIELDS = {
    "nz_cmd_max",
    "nz_cmd_mean",
    "nz_cmd_saturation_rate",
    "roll_rate_cmd_max",
    "roll_rate_cmd_mean",
    "roll_rate_cmd_saturation_rate",
    "throttle_cmd_max",
    "throttle_cmd_mean",
    "throttle_cmd_saturation_rate",
}

# Fields needed for altitude / energy analysis
# NOTE: These require PER-STEP telemetry.
ALTITUDE_ENERGY_FIELDS = {
    "min_altitude_m",
    "max_altitude_m",
    "final_altitude_m",
    "altitude_loss_rate",
    "energy_proxy",
}

FIELD_CATEGORIES = {
    "core": CORE_EPISODE_FIELDS,
    "terminal_phase": TERMINAL_PHASE_FIELDS,
    "prediction": PREDICTION_FIELDS,
    "command_saturation": COMMAND_SATURATION_FIELDS,
    "altitude_energy": ALTITUDE_ENERGY_FIELDS,
}


def validate_episode_telemetry(
    episode: dict,
    require_core: bool = True,
    require_terminal_phase: bool = False,
    require_prediction: bool = False,
    require_command_saturation: bool = False,
    require_altitude_energy: bool = False,
) -> Tuple[bool, List[str], Dict[str, List[str]]]:
    """
    Validate a single episode dict against the telemetry schema.

    Returns:
        (ok, critical_issues, missing_by_category)
        - ok: True if all required fields are present
        - critical_issues: List of missing core fields (always reported)
        - missing_by_category: Dict of category -> missing fields
    """
    ok = True
    critical_issues = []
    missing_by_category = {}

    required_categories = {}
    if require_core:
        required_categories["core"] = CORE_EPISODE_FIELDS
    if require_terminal_phase:
        required_categories["terminal_phase"] = TERMINAL_PHASE_FIELDS
    if require_prediction:
        required_categories["prediction"] = PREDICTION_FIELDS
    if require_command_saturation:
        required_categories["command_saturation"] = COMMAND_SATURATION_FIELDS
    if require_altitude_energy:
        required_categories["altitude_energy"] = ALTITUDE_ENERGY_FIELDS

    for category, fields in required_categories.items():
        missing = [f for f in fields if f not in episode]
        if missing:
            missing_by_category[category] = missing
            if category == "core":
                critical_issues.extend(missing)
                ok = False

    return ok, critical_issues, missing_by_category


def validate_episodes_telemetry(
    episodes: List[dict],
    require_core: bool = True,
    require_terminal_phase: bool = False,
    require_prediction: bool = False,
    require_command_saturation: bool = False,
    require_altitude_energy: bool = False,
    sample_size: int = 10,
) -> Tuple[bool, Dict]:
    """
    Validate a list of episodes. Samples up to `sample_size` episodes
    to check field presence (assumes homogeneity within a run).

    Returns:
        (ok, report)
    """
    report = {
        "total_episodes": len(episodes),
        "sampled": 0,
        "critical_issues": [],
        "missing_by_category": {},
        "unavailable_categories": [],
    }

    if not episodes:
        report["critical_issues"].append("No episodes provided")
        return False, report

    ok = True
    all_missing = {cat: set() for cat in FIELD_CATEGORIES.keys()}

    sample = episodes[:sample_size]
    report["sampled"] = len(sample)

    for ep in sample:
        ep_ok, crit, missing = validate_episode_telemetry(
            ep,
            require_core=require_core,
            require_terminal_phase=require_terminal_phase,
            require_prediction=require_prediction,
            require_command_saturation=require_command_saturation,
            require_altitude_energy=require_altitude_energy,
        )
        if not ep_ok:
            ok = False
        if crit:
            report["critical_issues"].extend(crit)
        for cat, fields in missing.items():
            all_missing[cat].update(fields)

    # Deduplicate critical issues
    report["critical_issues"] = sorted(set(report["critical_issues"]))

    # Build missing_by_category (only non-empty)
    for cat, fields in all_missing.items():
        if fields:
            report["missing_by_category"][cat] = sorted(fields)

    # Determine unavailable categories
    for cat in ["command_saturation", "altitude_energy"]:
        if cat in report["missing_by_category"]:
            report["unavailable_categories"].append(cat)

    return ok, report


def render_telemetry_validation_report(report: Dict) -> str:
    """Generate a markdown report from telemetry validation results."""
    lines = []
    lines.append("# Telemetry Schema Validation Report")
    lines.append("")
    lines.append(f"- **Total episodes**: {report['total_episodes']}")
    lines.append(f"- **Sampled for validation**: {report['sampled']}")
    lines.append("")

    if report["critical_issues"]:
        lines.append("## ❌ Critical Issues (Missing Core Fields)")
        lines.append("")
        for issue in report["critical_issues"]:
            lines.append(f"- `{issue}`")
        lines.append("")
        lines.append("> **Root-cause analysis may be unreliable without these fields.**")
        lines.append("")
    else:
        lines.append("## ✅ Core Fields")
        lines.append("")
        lines.append("All core fields present in sampled episodes.")
        lines.append("")

    if report["missing_by_category"]:
        lines.append("## Missing Fields by Category")
        lines.append("")
        for cat, fields in sorted(report["missing_by_category"].items()):
            if cat == "core":
                continue  # Already reported above
            lines.append(f"### {cat}")
            lines.append("")
            for f in fields:
                lines.append(f"- `{f}`")
            lines.append("")
    else:
        lines.append("## ✅ All Requested Fields Present")
        lines.append("")

    if report["unavailable_categories"]:
        lines.append("## ⚠️ Unavailable Categories")
        lines.append("")
        for cat in report["unavailable_categories"]:
            lines.append(f"- `{cat}`: **Not available** — per-step telemetry not emitted by current evaluation harness.")
        lines.append("")
        lines.append("> These categories require per-step telemetry (nz_cmd, roll_rate_cmd, throttle, altitude, energy). "
            "The current `evaluate_prediction_comparison.py` only emits per-episode aggregates. "
            "Do not generate root-cause claims that depend on these fields without first adding per-step telemetry output.")
        lines.append("")

    return "\n".join(lines)


def main_validate_file(
    json_path: str,
    require_terminal_phase: bool = True,
    require_prediction: bool = True,
    require_command_saturation: bool = False,
    require_altitude_energy: bool = False,
) -> Tuple[bool, Dict]:
    """Load episodes from a JSON file and validate."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    episodes = []
    if isinstance(data, list):
        for m in data:
            episodes.extend(m.get("raw_episodes", []))
    elif isinstance(data, dict):
        episodes = data.get("raw_episodes", [])

    ok, report = validate_episodes_telemetry(
        episodes,
        require_terminal_phase=require_terminal_phase,
        require_prediction=require_prediction,
        require_command_saturation=require_command_saturation,
        require_altitude_energy=require_altitude_energy,
    )
    return ok, report
