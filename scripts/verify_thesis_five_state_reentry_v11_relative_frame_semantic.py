#!/usr/bin/env python3
"""Independently recompute V11's semantic-transfer gate from episode records."""

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

import run_thesis_five_state_reentry_v11_relative_frame_semantic as runner
from uav_vpp_guidance.evaluation.thesis_reentry_v11_relative_frame import METHODS, OPPONENTS, summarize_records


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
        raise RuntimeError(f"V11 output is missing: {root}")
    records = []
    for method in METHODS:
        for opponent in OPPONENTS:
            for scenario in manifest["scenarios"]:
                path = root / "episodes" / method / opponent / f"{scenario['name']}.json"
                if not path.is_file():
                    raise RuntimeError(f"V11 episode record is missing: {path}")
                payload = json.loads(path.read_text(encoding="utf-8"))
                row = payload.get("record")
                if not isinstance(row, Mapping) or row.get("source_id") != runner.SOURCE_ID:
                    raise RuntimeError(f"V11 episode record is invalid: {path}")
                if row.get("method") != method or row.get("opponent") != opponent:
                    raise RuntimeError(f"V11 episode identity drifted: {path}")
                records.append(row)
    gates = config["gates"]
    recomputed = summarize_records(
        records,
        scenarios_per_opponent=len(manifest["scenarios"]),
        min_window_steps=int(config["observability"]["local_window_steps"]),
        min_pairs=int(gates["min_qualifying_pairs_per_opponent"]),
        min_semantic_delta_norm_m=float(gates["min_semantic_delta_norm_m"]),
        min_vpp_delta_norm_m=float(gates["min_env_vpp_delta_norm_m"]),
        min_guidance_delta_norm=float(gates["min_guidance_delta_norm"]),
        min_range_rate_delta_mps=float(gates["min_range_rate_delta_mps"]),
        practical_intent_loss_delta=float(gates["practical_intent_loss_delta_vs_reference_max"]),
        safety_delta_max=float(gates["ego_crash_oob_delta_vs_reference_max"]),
    )
    gate_path = root / "semantic_transfer_gate.json"
    reported = json.loads(gate_path.read_text(encoding="utf-8"))
    expected_gate = {"source_id": runner.SOURCE_ID, **recomputed}
    if not _equivalent(reported, expected_gate):
        raise RuntimeError("V11 reported gate does not match independent recomputation")
    run_manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    if run_manifest.get("source_id") != runner.SOURCE_ID or int(run_manifest.get("records", -1)) != len(records):
        raise RuntimeError("V11 run manifest is inconsistent with episode records")
    if run_manifest.get("manifest_sha256") != runner._hash(manifest_path):
        raise RuntimeError("V11 run manifest has the wrong manifest hash")
    return {
        "source_id": runner.SOURCE_ID,
        "records": len(records),
        "reported_gate_matches_recomputed": True,
        "passes": bool(recomputed["passes"]),
    }


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
