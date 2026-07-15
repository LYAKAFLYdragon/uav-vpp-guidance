#!/usr/bin/env python3
"""Independently recompute the V18 phase-telemetry gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_thesis_five_state_v18_phase_telemetry_audit as runner
from uav_vpp_guidance.evaluation.thesis_v18_phase_telemetry_audit import OPPONENTS, SOURCE_ID, summarize_records


def verify(config_path: Path, output: Path | None = None) -> dict:
    design = runner._load_design(config_path); root = output or runner.v17._resolve(str(design["outputs"]["root"])); manifest = runner.v17._load(runner.v17._resolve(str(design["inputs"]["manifest"])))
    records = []
    for opponent in OPPONENTS:
        for scenario in manifest["scenarios"]:
            path = root / "episodes" / opponent / f"{scenario['name']}.json"
            row = json.loads(path.read_text(encoding="utf-8")).get("record")
            if not isinstance(row, dict) or row.get("source_id") != SOURCE_ID or row.get("pair_key") != scenario["metadata"]["pair_key"]:
                raise RuntimeError(f"V18 record identity mismatch: {path}")
            records.append(row)
    recomputed = summarize_records(records, expected=design["expected_v17_phase_replay"], records_per_opponent=int(design["gates"]["records_per_opponent"]))
    reported = json.loads((root / "phase_telemetry_gate.json").read_text(encoding="utf-8"))
    if reported != {"source_id": SOURCE_ID, **recomputed}:
        raise RuntimeError("V18 reported gate does not match independent recomputation")
    return {"source_id": SOURCE_ID, "records": len(records), "reported_gate_matches_recomputed": True, "passes": bool(recomputed["passes"])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--output", type=Path); parser.add_argument("--write", action="store_true"); args = parser.parse_args()
    result = verify(args.config, args.output); root = args.output or runner.v17._resolve(str(runner._load_design(args.config)["outputs"]["root"]))
    if args.write: runner.v17._write_json(root / "independent_verification.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
