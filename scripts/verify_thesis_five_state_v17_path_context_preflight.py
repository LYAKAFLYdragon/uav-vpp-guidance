#!/usr/bin/env python3
"""Independently verify V17's 74-D path-context sampler preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_thesis_five_state_v17_path_context_preflight as runner
import verify_thesis_five_state_reentry_v14_temporal_corridor as v14_verifier
from uav_vpp_guidance.evaluation.thesis_v17_path_context_preflight import (
    OPPONENTS,
    SOURCE_ID,
    summarize_records,
)
from uav_vpp_guidance.hierarchy.path_conditioned_observation_contract import build_path_context


def _vector(value: Any, length: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise RuntimeError(f"V17 invalid {label}")
    return result


def _verify_record(row: Mapping[str, Any]) -> None:
    if not bool(row.get("handoff_reached")):
        if int(row.get("window_steps", 0)) != 0 or row.get("window") not in (None, []):
            raise RuntimeError("V17 non-handoff record contains a window")
        return
    window = row.get("window")
    if not isinstance(window, list) or len(window) != int(row.get("window_steps", 0)):
        raise RuntimeError("V17 window length is inconsistent")
    for index, step in enumerate(window):
        if int(step.get("local_step", -1)) != index:
            raise RuntimeError("V17 local-step index drifted")
        _vector(step.get("observation_74d"), 74, "74-D observation")
        stored_context = _vector(step.get("path_context"), 8, "path context")
        rebuilt_context = build_path_context(
            requested_path_world_delta_m=step.get("requested_path_world_delta_m"),
            path_progress=float(step.get("path_progress")),
            selected_anchor_index=int(step.get("selected_anchor_index")),
        )
        if not np.allclose(stored_context, rebuilt_context, rtol=0.0, atol=1e-7):
            raise RuntimeError("V17 path context does not replay from current-only receipt")
        if not np.allclose(_vector(step.get("observation_74d"), 74, "74-D observation")[-8:], rebuilt_context, rtol=0.0, atol=1e-7):
            raise RuntimeError("V17 74-D suffix does not match its path context")
        _vector(step.get("raw_action"), 3, "raw action")
        _vector(step.get("env_vpp_world_offset_m"), 3, "VPP offset")
        _vector(step.get("guidance_command_normalized"), 3, "guidance command")
        if not bool(step.get("prediction_valid")) or bool(step.get("prediction_fallback")):
            raise RuntimeError("V17 prediction fallback occurred")


def verify(config_path: Path, output: Path | None = None) -> dict[str, Any]:
    config = runner._load_runtime_config(config_path)
    manifest_path = runner._resolve(config["inputs"]["manifest"])
    manifest = runner._load(manifest_path)
    root = output or runner._resolve(config["outputs"]["root"])
    if not root.is_dir():
        raise RuntimeError(f"V17 output is missing: {root}")
    records: list[Mapping[str, Any]] = []
    for opponent in OPPONENTS:
        for scenario in manifest["scenarios"]:
            path = root / "episodes" / opponent / f"{scenario['name']}.json"
            if not path.is_file():
                raise RuntimeError(f"V17 episode record is missing: {path}")
            row = json.loads(path.read_text(encoding="utf-8")).get("record")
            if not isinstance(row, Mapping) or row.get("source_id") != SOURCE_ID:
                raise RuntimeError(f"V17 invalid episode record: {path}")
            if row.get("opponent") != opponent or row.get("pair_key") != scenario["metadata"]["pair_key"]:
                raise RuntimeError(f"V17 record identity drifted: {path}")
            _verify_record(row)
            records.append(row)
    gates = config["gates"]
    recomputed = summarize_records(records, scenarios_per_opponent=len(manifest["scenarios"]), min_qualifying_episodes=int(gates["min_qualifying_episodes_per_opponent"]), min_valid_steps=int(gates["min_valid_path_context_steps_per_opponent"]), min_mirrors=int(gates["min_mirrors_per_opponent"]), min_signatures=int(gates["min_signatures_per_opponent"]))
    reported = json.loads((root / "path_context_preflight_gate.json").read_text(encoding="utf-8"))
    if not v14_verifier._equivalent(reported, {"source_id": SOURCE_ID, **recomputed}):
        raise RuntimeError("V17 reported gate does not match independent recomputation")
    run_manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    if run_manifest.get("source_id") != SOURCE_ID or int(run_manifest.get("records", -1)) != len(records):
        raise RuntimeError("V17 run manifest is inconsistent")
    if run_manifest.get("config_sha256") != runner._hash(config_path) or run_manifest.get("manifest_sha256") != runner._hash(manifest_path):
        raise RuntimeError("V17 run manifest hash drifted")
    if run_manifest.get("training") != "prohibited" or run_manifest.get("reward_recording") != "prohibited":
        raise RuntimeError("V17 run manifest violates the non-learning contract")
    return {"source_id": SOURCE_ID, "records": len(records), "reported_gate_matches_recomputed": True, "path_context_contract_replayed": True, "passes": bool(recomputed["passes"])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--output", type=Path); parser.add_argument("--write", action="store_true"); args = parser.parse_args()
    result = verify(args.config, args.output)
    root = args.output or runner._resolve(runner._load_runtime_config(args.config)["outputs"]["root"])
    if args.write: runner._write_json(root / "independent_verification.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
