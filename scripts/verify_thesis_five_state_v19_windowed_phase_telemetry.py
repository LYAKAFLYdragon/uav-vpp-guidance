#!/usr/bin/env python3
"""Independently recompute V19's windowed telemetry gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_thesis_five_state_v19_windowed_phase_telemetry as runner
from uav_vpp_guidance.evaluation.thesis_v19_windowed_phase_telemetry import OPPONENTS, SOURCE_ID, summarize_records


def verify(config_path: Path, output: Path | None = None) -> dict:
    design = runner._load_design(config_path); root = output or runner.v18.v17._resolve(str(design["outputs"]["root"])); manifest = runner.v18.v17._load(runner.v18.v17._resolve(str(design["inputs"]["manifest"])))
    records = []
    for opponent in OPPONENTS:
        for scenario in manifest["scenarios"]:
            path = root / "episodes" / opponent / f"{scenario['name']}.json"; row = json.loads(path.read_text(encoding="utf-8")).get("record")
            if not isinstance(row, dict) or row.get("source_id") != SOURCE_ID or row.get("opponent") != opponent or row.get("pair_key") != scenario["metadata"]["pair_key"]: raise RuntimeError(f"V19 record identity mismatch: {path}")
            records.append(row)
    result = summarize_records(records, expected=design["expected_v17_phase_replay"], records_per_opponent=24, minimum_ready_steps=int(design["gates"]["per_opponent_analysis_ready_steps_min"]))
    reported = json.loads((root / "windowed_phase_telemetry_gate.json").read_text(encoding="utf-8"))
    if reported != {"source_id": SOURCE_ID, **result}: raise RuntimeError("V19 reported gate differs from independent recomputation")
    return {"source_id": SOURCE_ID, "records": len(records), "reported_gate_matches_recomputed": True, "passes": bool(result["passes"])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--output", type=Path); parser.add_argument("--write", action="store_true"); args = parser.parse_args()
    result = verify(args.config, args.output); root = args.output or runner.v18.v17._resolve(str(runner._load_design(args.config)["outputs"]["root"]))
    if args.write: runner.v18.v17._write_json(root / "independent_verification.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
