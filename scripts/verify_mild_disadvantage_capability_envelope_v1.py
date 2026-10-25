#!/usr/bin/env python
"""Verify the completed two-dimensional capability-envelope scan."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SOURCE_ID = "THESIS-MILD-DISADVANTAGE-CAPABILITY-ENVELOPE-V1-20260903"
EXPECTED_OPPONENTS = ("expert", "end_to_end")
EXPECTED_SCENARIOS = tuple(
    f"x{x}_y{y}"
    for x in (625, 675, 725, 750)
    for y in (100, 150, 200, 250)
)
REPETITIONS_PER_CELL = 3


def verify(root: Path) -> dict:
    errors: list[str] = []
    manifest_path = root / "pilot_manifest.json"
    if not manifest_path.is_file():
        return {"ok": False, "errors": [f"missing {manifest_path}"]}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_id") != SOURCE_ID:
        errors.append("wrong source_id")
    if manifest.get("training_started") is not False or manifest.get("training_permitted") is not False:
        errors.append("training was not locked off")
    if manifest.get("backend") != "jsbsim":
        errors.append("backend is not jsbsim")
    if tuple(manifest.get("opponents", [])) != EXPECTED_OPPONENTS:
        errors.append("opponent set mismatch")
    if tuple(manifest.get("scenarios", {}).keys()) != EXPECTED_SCENARIOS:
        errors.append("scenario set mismatch")
    if manifest.get("repetitions_per_cell") != REPETITIONS_PER_CELL:
        errors.append("repetition count mismatch")
    expected_total = len(EXPECTED_SCENARIOS) * len(EXPECTED_OPPONENTS) * REPETITIONS_PER_CELL
    if manifest.get("raw_episode_count") != expected_total:
        errors.append(f"expected {expected_total} raw episodes")
    runs = manifest.get("runs", [])
    if len(runs) != len(EXPECTED_OPPONENTS):
        errors.append("expected one run manifest per opponent")
    for run in runs:
        opponent = run.get("opponent")
        if opponent not in EXPECTED_OPPONENTS:
            errors.append(f"unexpected opponent {opponent}")
        if run.get("backend_effective") != "jsbsim":
            errors.append(f"{opponent}: effective backend is not jsbsim")
        if run.get("backend_fallback_occurred") is not False:
            errors.append(f"{opponent}: backend fallback occurred")
        if run.get("obs_dim") != 16 or run.get("action_dim") != 3:
            errors.append(f"{opponent}: dimension mismatch")
        if run.get("episodes_per_scenario") != REPETITIONS_PER_CELL:
            errors.append(f"{opponent}: repetition count mismatch")
        if run.get("raw_episode_count") != len(EXPECTED_SCENARIOS) * REPETITIONS_PER_CELL:
            errors.append(f"{opponent}: raw episode count mismatch")
        run_root = root / "h0_current_geometry" / str(opponent)
        metrics_path = run_root / "metrics.json"
        if not metrics_path.is_file():
            errors.append(f"{opponent}: missing metrics.json")
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        episodes = metrics.get("raw_episodes", [])
        if len(episodes) != len(EXPECTED_SCENARIOS) * REPETITIONS_PER_CELL:
            errors.append(f"{opponent}: raw episode list is incomplete")
        scenario_counts = {name: 0 for name in EXPECTED_SCENARIOS}
        for episode in episodes:
            scenario = episode.get("scenario")
            if scenario not in scenario_counts:
                errors.append(f"{opponent}: unexpected scenario {scenario}")
                continue
            scenario_counts[scenario] += 1
            if not episode.get("reason"):
                errors.append(f"{opponent}/{scenario}: missing terminal reason")
        if any(count != REPETITIONS_PER_CELL for count in scenario_counts.values()):
            errors.append(f"{opponent}: unbalanced scenario repetitions")
        traj_dir = run_root / "trajectories" / "h0_current_geometry"
        files = list(traj_dir.glob("*.csv")) if traj_dir.is_dir() else []
        expected_files = len(EXPECTED_SCENARIOS) * REPETITIONS_PER_CELL
        if len(files) != expected_files or any(path.stat().st_size <= 0 for path in files):
            errors.append(f"{opponent}: trajectory telemetry is incomplete")

    cell_path = root / "metrics_by_cell_and_opponent.csv"
    if not cell_path.is_file():
        errors.append("missing metrics_by_cell_and_opponent.csv")
    else:
        with cell_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != len(EXPECTED_SCENARIOS) * len(EXPECTED_OPPONENTS):
            errors.append("cell metric row count mismatch")
        for row in rows:
            if int(row.get("episodes", 0)) != REPETITIONS_PER_CELL:
                errors.append(f"cell repetition mismatch: {row.get('opponent')}/{row.get('scenario')}")
            try:
                float(row["success_rate"])
                float(row["mean_clipping_rate"])
                float(row["mean_fallback_rate"])
            except (KeyError, TypeError, ValueError):
                errors.append(f"invalid cell metrics: {row.get('opponent')}/{row.get('scenario')}")
    envelope_path = root / "envelope_classification.csv"
    if not envelope_path.is_file():
        errors.append("missing envelope_classification.csv")
    else:
        with envelope_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != len(EXPECTED_SCENARIOS):
            errors.append("envelope classification row count mismatch")
        allowed = {"stable_feasible", "transition", "non_feasible"}
        for row in rows:
            if row.get("classification") not in allowed:
                errors.append(f"invalid envelope classification: {row.get('scenario')}")
    if not (root / "envelope_analysis.json").is_file():
        errors.append("missing envelope_analysis.json")
    if not (root / "episode_level_results.csv").is_file():
        errors.append("missing episode_level_results.csv")
    return {"ok": not errors, "source_id": manifest.get("source_id"), "errors": errors}


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
