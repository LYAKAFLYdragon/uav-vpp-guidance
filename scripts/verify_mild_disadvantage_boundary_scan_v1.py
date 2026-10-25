#!/usr/bin/env python
"""Verify the completed non-learning disadvantage boundary scan."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SOURCE_ID = "THESIS-MILD-DISADVANTAGE-BOUNDARY-SCAN-V1-20260903"
SOURCE_ID_V2 = "THESIS-MILD-DISADVANTAGE-BOUNDARY-SCAN-V2-20260903"
SOURCE_ID_V3 = "THESIS-MILD-DISADVANTAGE-BOUNDARY-SCAN-V3-20260903"
SOURCE_ID_V4 = "THESIS-MILD-DISADVANTAGE-BOUNDARY-SCAN-V4-20260903"
SOURCE_ID_V5 = "THESIS-MILD-DISADVANTAGE-BOUNDARY-SCAN-V5-20260903"
EXPECTED_OPPONENTS = ("expert", "end_to_end")
EXPECTED_SCENARIOS = (
    "rear_offset_x150",
    "rear_offset_x300",
    "rear_offset_x450",
    "rear_offset_x600",
    "target_speed_200",
    "target_speed_205",
    "target_speed_210",
    "target_speed_215",
    "target_speed_220",
)
EXPECTED_SCENARIOS_V2 = (
    "offset_x100",
    "offset_x50",
    "offset_x0",
    "offset_xm50",
    "offset_xm100",
    "speed_185",
    "speed_195",
    "speed_205",
    "speed_215",
    "speed_225",
)
EXPECTED_SCENARIOS_V3 = (
    "heading_20",
    "heading_60",
    "heading_100",
    "heading_140",
    "heading_180",
    "heading_220",
    "heading_260",
)
EXPECTED_SCENARIOS_V4 = (
    "target_x_1900",
    "target_x_1000",
    "target_x_500",
    "target_x_0",
    "target_x_m300",
)
EXPECTED_SCENARIOS_V5 = (
    "target_x_1900",
    "target_x_1000",
    "target_x_875",
    "target_x_750",
    "target_x_625",
    "target_x_500",
    "target_x_0",
    "target_x_m300",
)


def verify(root: Path) -> dict:
    errors: list[str] = []
    manifest_path = root / "pilot_manifest.json"
    if not manifest_path.is_file():
        return {"ok": False, "errors": [f"missing {manifest_path}"]}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_id = manifest.get("source_id")
    if source_id == SOURCE_ID_V2:
        expected_scenarios = EXPECTED_SCENARIOS_V2
    elif source_id == SOURCE_ID_V3:
        expected_scenarios = EXPECTED_SCENARIOS_V3
    elif source_id == SOURCE_ID_V4:
        expected_scenarios = EXPECTED_SCENARIOS_V4
    elif source_id == SOURCE_ID_V5:
        expected_scenarios = EXPECTED_SCENARIOS_V5
    else:
        expected_scenarios = EXPECTED_SCENARIOS
    expected_episode_count = len(expected_scenarios) * len(EXPECTED_OPPONENTS)
    if source_id not in (SOURCE_ID, SOURCE_ID_V2, SOURCE_ID_V3, SOURCE_ID_V4, SOURCE_ID_V5):
        errors.append("wrong source_id")
    if manifest.get("training_started") is not False or manifest.get("training_permitted") is not False:
        errors.append("training was not locked off")
    if manifest.get("backend") != "jsbsim":
        errors.append("backend is not jsbsim")
    if tuple(manifest.get("opponents", [])) != EXPECTED_OPPONENTS:
        errors.append("opponent set mismatch")
    if tuple(manifest.get("scenarios", {}).keys()) != expected_scenarios:
        errors.append("scenario set mismatch")
    if manifest.get("raw_episode_count") != expected_episode_count:
        errors.append(f"expected {expected_episode_count} raw episodes")
    runs = manifest.get("runs", [])
    if len(runs) != 2:
        errors.append("expected one run manifest per opponent")

    for run in runs:
        opponent = run.get("opponent")
        if run.get("backend_effective") != "jsbsim":
            errors.append(f"{opponent}: effective backend is not jsbsim")
        if run.get("backend_fallback_occurred") is not False:
            errors.append(f"{opponent}: backend fallback occurred")
        if run.get("obs_dim") != 16 or run.get("action_dim") != 3:
            errors.append(f"{opponent}: dimension mismatch")
        run_root = root / "h0_current_geometry" / str(opponent)
        metrics_path = run_root / "metrics.json"
        if not metrics_path.is_file():
            errors.append(f"{opponent}: missing metrics.json")
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        episodes = metrics.get("raw_episodes", [])
        if len(episodes) != len(expected_scenarios):
            errors.append(f"{opponent}: expected {len(expected_scenarios)} episodes")
        if {e.get("scenario") for e in episodes} != set(expected_scenarios):
            errors.append(f"{opponent}: scenario coverage mismatch")
        for episode in episodes:
            if not episode.get("reason"):
                errors.append(f"{opponent}/{episode.get('scenario')}: missing terminal reason")
        traj_dir = run_root / "trajectories" / "h0_current_geometry"
        files = list(traj_dir.glob("*.csv")) if traj_dir.is_dir() else []
        if len(files) != len(expected_scenarios) or any(path.stat().st_size <= 0 for path in files):
            errors.append(f"{opponent}: trajectory telemetry is incomplete")

    summary = root / "metrics_by_scenario.csv"
    if not summary.is_file():
        errors.append("missing metrics_by_scenario.csv")
    else:
        with summary.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != expected_episode_count:
            errors.append(f"scenario summary does not contain {expected_episode_count} rows")
        for row in rows:
            try:
                episodes = int(row["episodes"])
                successes = int(row["successes"])
                rate = float(row["success_rate"])
            except (KeyError, TypeError, ValueError):
                errors.append("scenario summary has invalid numeric fields")
                continue
            if episodes != 1 or successes not in (0, 1) or abs(rate - successes) > 1e-9:
                errors.append(f"invalid summary values for {row.get('opponent')}/{row.get('scenario')}")

    if not (root / "boundary_analysis.json").is_file():
        errors.append("missing boundary_analysis.json")
    return {"ok": not errors, "source_id": source_id, "errors": errors}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = verify(args.root.resolve())
    if args.output:
        args.output.resolve().write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
