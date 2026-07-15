#!/usr/bin/env python3
"""Independently recompute the V20 warm-up-fallback telemetry gate."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import run_thesis_five_state_v20_warmup_fallback_telemetry as runner


def verify(config: Path, output: Path | None = None) -> dict:
    design = runner.base._load_design(config); root = output or runner.base.v18.v17._resolve(str(design["outputs"]["root"])); manifest = runner.base.v18.v17._load(runner.base.v18.v17._resolve(str(design["inputs"]["manifest"])))
    rows = []
    for opponent in runner.base.OPPONENTS:
        for scenario in manifest["scenarios"]:
            path = root / "episodes" / opponent / f"{scenario['name']}.json"; row = json.loads(path.read_text(encoding="utf-8")).get("record")
            if not isinstance(row, dict) or row.get("source_id") != runner.SOURCE_ID or row.get("opponent") != opponent: raise RuntimeError(f"V20 record mismatch: {path}")
            rows.append(row)
    summary = runner._summary(rows, expected=design["expected_v17_phase_replay"], records_per_opponent=24, minimum_ready_steps=int(design["gates"]["per_opponent_analysis_ready_steps_min"]))
    if json.loads((root / "windowed_phase_telemetry_gate.json").read_text(encoding="utf-8")) != {"source_id": runner.SOURCE_ID, **summary}: raise RuntimeError("V20 reported gate mismatch")
    return {"source_id": runner.SOURCE_ID, "records": len(rows), "reported_gate_matches_recomputed": True, "passes": bool(summary["passes"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--output", type=Path); parser.add_argument("--write", action="store_true"); args = parser.parse_args()
    result = verify(args.config, args.output); root = args.output or runner.base.v18.v17._resolve(str(runner.base._load_design(args.config)["outputs"]["root"]));
    if args.write: runner.base.v18.v17._write_json(root / "independent_verification.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
