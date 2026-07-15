#!/usr/bin/env python3
"""Independently recompute V13's six-objective controllability gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_thesis_five_state_reentry_v13_intent_controllability as runner
from uav_vpp_guidance.evaluation.thesis_reentry_v12_intent_controllability import METHODS, OPPONENTS, summarize_records


def _equivalent(left: Any, right: Any, *, tolerance: float = 1e-12) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(_equivalent(left[key], right[key], tolerance=tolerance) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_equivalent(a, b, tolerance=tolerance) for a, b in zip(left, right))
    if isinstance(left, (int, float)) and isinstance(right, (int, float)) and not isinstance(left, bool) and not isinstance(right, bool):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    return left == right


def verify(config_path: Path, output: Path | None = None) -> dict[str, Any]:
    config = runner._load_runtime_config(config_path)
    manifest_path = runner._resolve(config["inputs"]["manifest"])
    manifest = runner._load(manifest_path)
    root = output or runner._resolve(config["outputs"]["root"])
    if not root.is_dir():
        raise RuntimeError(f"V13 output is missing: {root}")
    records = []
    for method in METHODS:
        for opponent in OPPONENTS:
            for scenario in manifest["scenarios"]:
                path = root / "episodes" / method / opponent / f"{scenario['name']}.json"
                if not path.is_file():
                    raise RuntimeError(f"V13 episode record is missing: {path}")
                row = json.loads(path.read_text(encoding="utf-8")).get("record")
                if not isinstance(row, Mapping) or row.get("source_id") != runner.SOURCE_ID:
                    raise RuntimeError(f"V13 invalid episode record: {path}")
                if row.get("method") != method or row.get("opponent") != opponent:
                    raise RuntimeError(f"V13 episode identity drifted: {path}")
                records.append(row)
    gates = config["gates"]
    recomputed = summarize_records(records, scenarios_per_opponent=len(manifest["scenarios"]), min_window_steps=int(config["observability"]["local_window_steps"]), min_pairs=int(gates["min_qualifying_pairs_per_opponent"]), max_high_priority_worsening=float(gates["max_high_priority_component_worsening"]), max_any_component_worsening=float(gates["max_any_component_worsening"]), min_component_improvement=float(gates["min_component_improvement"]), min_improved_components=int(gates["min_improved_components"]), min_weighted_error_improvement=float(gates["min_weighted_error_improvement"]), safety_delta_max=float(gates["ego_crash_oob_delta_vs_reference_max"]), profile_name=str(config["semantic_compiler"]["profile"]))
    reported = json.loads((root / "intent_controllability_gate.json").read_text(encoding="utf-8"))
    if not _equivalent(reported, {"source_id": runner.SOURCE_ID, **recomputed}):
        raise RuntimeError("V13 reported gate does not match independent recomputation")
    run_manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    if run_manifest.get("source_id") != runner.SOURCE_ID or int(run_manifest.get("records", -1)) != len(records):
        raise RuntimeError("V13 run manifest is inconsistent")
    if run_manifest.get("manifest_sha256") != runner._hash(manifest_path):
        raise RuntimeError("V13 manifest hash drifted")
    return {"source_id": runner.SOURCE_ID, "records": len(records), "reported_gate_matches_recomputed": True, "passes": bool(recomputed["passes"])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = verify(args.config, args.output)
    root = args.output or runner._resolve(runner._load_runtime_config(args.config)["outputs"]["root"])
    if args.write:
        runner._write_json(root / "independent_verification.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
