#!/usr/bin/env python3
"""Independently recompute V14's temporal-corridor controllability gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_thesis_five_state_reentry_v14_temporal_corridor as runner
from uav_vpp_guidance.evaluation.thesis_reentry_v14_temporal_corridor import (
    METHODS,
    OPPONENTS,
    REFERENCE_METHOD,
    objective_profile,
    scheduled_action,
    summarize_records,
)


def _equivalent(left: Any, right: Any, *, tolerance: float = 1e-12) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(_equivalent(left[key], right[key], tolerance=tolerance) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_equivalent(a, b, tolerance=tolerance) for a, b in zip(left, right))
    if isinstance(left, (int, float)) and isinstance(right, (int, float)) and not isinstance(left, bool) and not isinstance(right, bool):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    return left == right


def _finite_vector(value: Any, *, length: int, label: str) -> None:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape != (length,) or not np.isfinite(vector).all():
        raise RuntimeError(f"V14 invalid {label}")


def _verify_window(row: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    window = row.get("window")
    window_steps = int(row.get("window_steps", 0))
    if not bool(row.get("handoff_reached")):
        if window_steps != 0 or window not in (None, []):
            raise RuntimeError("V14 non-handoff record contains a window")
        return
    if not isinstance(window, list) or len(window) != window_steps:
        raise RuntimeError("V14 window length is inconsistent")
    escape_steps = int(config["corridor"]["escape_steps"])
    corridor_steps = int(config["corridor"]["corridor_steps"])
    method = str(row["method"])
    for index, step in enumerate(window):
        if int(step.get("local_step", -1)) != index:
            raise RuntimeError("V14 local step index drifted")
        objective = objective_profile(index, escape_steps=escape_steps)
        action_profile, axes = scheduled_action(method, index, escape_steps=escape_steps, corridor_steps=corridor_steps)
        expected_stage = "escape" if index < escape_steps else "reentry"
        if step.get("objective_stage") != expected_stage or step.get("objective_profile") != objective:
            raise RuntimeError("V14 objective schedule drifted")
        if step.get("action_profile") != action_profile or step.get("semantic_axes") != axes:
            raise RuntimeError("V14 action corridor drifted")
        _finite_vector(step.get("observation_66d"), length=66, label="66-D observation")
        _finite_vector(step.get("after_component_errors"), length=6, label="component errors")
        _finite_vector(step.get("env_vpp_world_offset_m"), length=3, label="VPP offset")
        if not math.isfinite(float(step.get("after_weighted_error"))):
            raise RuntimeError("V14 non-finite weighted intent error")
        if not math.isfinite(float(step.get("actual_semantic_delta_norm_m"))):
            raise RuntimeError("V14 non-finite semantic action receipt")
        if method == REFERENCE_METHOD and float(step["actual_semantic_delta_norm_m"]) != 0.0:
            raise RuntimeError("V14 reference method applied a semantic offset")


def verify(config_path: Path, output: Path | None = None) -> dict[str, Any]:
    """Verify all episode contracts, then recompute the unpooled V14 gate."""

    config = runner._load_runtime_config(config_path)
    manifest_path = runner._resolve(config["inputs"]["manifest"])
    manifest = runner._load(manifest_path)
    root = output or runner._resolve(config["outputs"]["root"])
    if not root.is_dir():
        raise RuntimeError(f"V14 output is missing: {root}")
    records: list[Mapping[str, Any]] = []
    for method in METHODS:
        for opponent in OPPONENTS:
            for scenario in manifest["scenarios"]:
                path = root / "episodes" / method / opponent / f"{scenario['name']}.json"
                if not path.is_file():
                    raise RuntimeError(f"V14 episode record is missing: {path}")
                row = json.loads(path.read_text(encoding="utf-8")).get("record")
                if not isinstance(row, Mapping) or row.get("source_id") != runner.SOURCE_ID:
                    raise RuntimeError(f"V14 invalid episode record: {path}")
                if row.get("method") != method or row.get("opponent") != opponent:
                    raise RuntimeError(f"V14 episode identity drifted: {path}")
                if row.get("pair_key") != scenario["metadata"]["pair_key"]:
                    raise RuntimeError(f"V14 pair key drifted: {path}")
                _verify_window(row, config)
                records.append(row)
    gates = config["gates"]
    recomputed = summarize_records(
        records,
        scenarios_per_opponent=len(manifest["scenarios"]),
        min_window_steps=int(config["observability"]["local_window_steps"]),
        min_pairs=int(gates["min_qualifying_pairs_per_opponent"]),
        min_total_weighted_improvement=float(gates["min_total_weighted_improvement"]),
        min_reentry_weighted_improvement=float(gates["min_reentry_weighted_improvement"]),
        max_escape_range_rate_worsening=float(gates["max_escape_range_rate_worsening"]),
        safety_delta_max=float(gates["ego_crash_oob_delta_vs_reference_max"]),
    )
    reported = json.loads((root / "temporal_corridor_gate.json").read_text(encoding="utf-8"))
    if not _equivalent(reported, {"source_id": runner.SOURCE_ID, **recomputed}):
        raise RuntimeError("V14 reported gate does not match independent recomputation")
    run_manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    if run_manifest.get("source_id") != runner.SOURCE_ID or int(run_manifest.get("records", -1)) != len(records):
        raise RuntimeError("V14 run manifest is inconsistent")
    if run_manifest.get("config_sha256") != runner._hash(config_path):
        raise RuntimeError("V14 run manifest config hash drifted")
    if run_manifest.get("manifest_sha256") != runner._hash(manifest_path):
        raise RuntimeError("V14 run manifest manifest hash drifted")
    if run_manifest.get("training") != "prohibited" or run_manifest.get("reward_recording") != "prohibited":
        raise RuntimeError("V14 run manifest violates its non-learning contract")
    if not isinstance(run_manifest.get("authorized_code_files"), list) or not run_manifest["authorized_code_files"]:
        raise RuntimeError("V14 run manifest lacks the authorized code-file record")
    return {
        "source_id": runner.SOURCE_ID,
        "records": len(records),
        "reported_gate_matches_recomputed": True,
        "corridor_contract_replayed": True,
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
