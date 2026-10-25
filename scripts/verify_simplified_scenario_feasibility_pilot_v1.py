#!/usr/bin/env python
"""Independently verify the simplified-scenario feasibility pilot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SOURCE_ID = "THESIS-SIMPLIFIED-SCENARIO-FEASIBILITY-PILOT-V1-20260903"
EXPECTED_OPPONENTS = ("expert", "end_to_end")
EXPECTED_SCENARIOS = (
    "head_on_reference",
    "mild_crossing_left",
    "mild_crossing_right",
    "mild_neutral",
    "mild_disadvantage",
)


def verify(root: Path) -> dict:
    errors: list[str] = []
    manifest_path = root / "pilot_manifest.json"
    if not manifest_path.is_file():
        return {"ok": False, "errors": [f"missing {manifest_path}"]}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_id") != SOURCE_ID:
        errors.append("wrong source_id")
    if manifest.get("phase") != "pilot":
        errors.append("wrong phase")
    if manifest.get("training_started") is not False or manifest.get("training_permitted") is not False:
        errors.append("training was not locked off")
    if tuple(manifest.get("opponents", [])) != EXPECTED_OPPONENTS:
        errors.append("opponent set mismatch")
    if tuple(manifest.get("scenarios", [])) != EXPECTED_SCENARIOS:
        errors.append("scenario set mismatch")
    if manifest.get("backend") != "jsbsim":
        errors.append("requested backend is not jsbsim")
    if manifest.get("raw_episode_count") != 10:
        errors.append("expected exactly 10 pilot episodes")
    runs = manifest.get("runs", [])
    if len(runs) != len(EXPECTED_OPPONENTS):
        errors.append("expected one run manifest per opponent")

    for run in runs:
        if run.get("backend_effective") != "jsbsim":
            errors.append(f"{run.get('opponent')}: effective backend is not jsbsim")
        if run.get("backend_fallback_occurred") is not False:
            errors.append(f"{run.get('opponent')}: backend fallback occurred")
        if run.get("obs_dim") != 16 or run.get("action_dim") != 3:
            errors.append(f"{run.get('opponent')}: observation/action dimensions mismatch")
        metrics_path = root / "h0_current_geometry" / str(run.get("opponent")) / "metrics.json"
        if not metrics_path.is_file():
            errors.append(f"{run.get('opponent')}: missing metrics.json")
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        episodes = metrics.get("raw_episodes", [])
        if len(episodes) != 5:
            errors.append(f"{run.get('opponent')}: expected five episodes")
        seen = set()
        for episode in episodes:
            scenario = episode.get("scenario")
            seen.add(scenario)
            if scenario not in EXPECTED_SCENARIOS:
                errors.append(f"{run.get('opponent')}: unknown scenario {scenario}")
            if not episode.get("reason"):
                errors.append(f"{run.get('opponent')}/{scenario}: missing terminal reason")
            for key in ("return", "length", "final_range_m", "final_ata_deg"):
                if key not in episode:
                    errors.append(f"{run.get('opponent')}/{scenario}: missing telemetry metric {key}")
        if seen != set(EXPECTED_SCENARIOS):
            errors.append(f"{run.get('opponent')}: scenario coverage mismatch")
        trajectory_dir = root / "h0_current_geometry" / str(run.get("opponent")) / "trajectories" / "h0_current_geometry"
        trajectory_files = list(trajectory_dir.glob("*.csv")) if trajectory_dir.is_dir() else []
        if len(trajectory_files) != 5:
            errors.append(f"{run.get('opponent')}: expected five non-empty trajectory files")
        else:
            for path in trajectory_files:
                if path.stat().st_size <= 0:
                    errors.append(f"empty trajectory file: {path.name}")

    summary_path = root / "metrics_by_scenario.csv"
    if not summary_path.is_file():
        errors.append("missing metrics_by_scenario.csv")
    else:
        with summary_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 10:
            errors.append("scenario summary does not contain 10 rows")
        for row in rows:
            try:
                episodes = int(row["episodes"])
                successes = int(row["successes"])
                rate = float(row["success_rate"])
            except (KeyError, TypeError, ValueError):
                errors.append("scenario summary has invalid numeric fields")
                continue
            if episodes != 1 or successes not in (0, 1) or abs(rate - successes / episodes) > 1e-9:
                errors.append(
                    f"invalid scenario summary values for {row.get('opponent')}/{row.get('scenario')}"
                )

    return {"ok": not errors, "source_id": SOURCE_ID, "errors": errors}


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
