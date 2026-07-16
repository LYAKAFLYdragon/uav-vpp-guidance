#!/usr/bin/env python3
"""Independently recompute V22's observer-isolation and release gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "scripts")]

import run_thesis_five_state_v22_observer_release_probe as runner


def _verify_row(row: dict[str, Any], opponent: str, scenario: dict[str, Any]) -> None:
    if row.get("source_id") != runner.SOURCE_ID or row.get("opponent") != opponent:
        raise RuntimeError("V22 record identity mismatch")
    if row.get("pair_key") != scenario["metadata"]["pair_key"]:
        raise RuntimeError("V22 pair identity mismatch")
    trace = row.get("observer_trace")
    if not isinstance(trace, list) or not trace:
        raise RuntimeError("V22 observer trace is missing")
    release_steps = [entry["step"] for entry in row.get("anchor_switches", []) if entry.get("mode") == "predicted_target"]
    if len(release_steps) > 1:
        raise RuntimeError("V22 switched anchors more than once")
    release_step = release_steps[0] if release_steps else None
    for receipt in trace:
        if release_step is None or int(receipt["step"]) <= release_step:
            if receipt.get("observer_prediction_consumed_by_control"):
                raise RuntimeError("V22 observer receipt entered control before release")
    post_release_valid_steps = sum(
        1
        for receipt in trace
        if release_step is not None
        and int(receipt["step"]) > release_step
        and receipt.get("prediction_valid") is True
        and not receipt.get("prediction_fallback", False)
    )
    if int(row.get("post_release_valid_steps", -1)) != post_release_valid_steps:
        raise RuntimeError("V22 post-release valid-step count mismatch")
    if bool(row.get("post_release_prediction_fallback", False)):
        raise RuntimeError("V22 record contains fail-closed post-release fallback")


def verify(config: Path, output: Path | None = None) -> dict[str, Any]:
    design = runner.validate_design(config, allow_existing_output=True)
    root = output or runner.v17._resolve(runner._load_config(config)["outputs"]["root"])
    manifest = runner.v17._load(runner.v17._resolve(runner._load_config(config)["inputs"]["manifest"]))
    rows: list[dict[str, Any]] = []
    for opponent in runner.OPPONENTS:
        for scenario in manifest["scenarios"]:
            row = json.loads(
                (root / "episodes" / opponent / f"{scenario['name']}.json").read_text(encoding="utf-8")
            )["record"]
            _verify_row(row, opponent, scenario)
            rows.append(row)
    gate = runner._summarize(rows)
    reported_gate = json.loads((root / "gate.json").read_text(encoding="utf-8"))
    if reported_gate != gate:
        raise RuntimeError("V22 reported gate mismatch")
    if int(gate["records"]) != int(design["planned_records"]):
        raise RuntimeError("V22 record count mismatch")
    return {
        "source_id": runner.SOURCE_ID,
        "records": len(rows),
        "reported_gate_matches_recomputed": True,
        "passes": gate["passes"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.config, args.output), ensure_ascii=False, indent=2))
