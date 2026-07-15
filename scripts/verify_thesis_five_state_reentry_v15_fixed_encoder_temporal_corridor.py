#!/usr/bin/env python3
"""Independently recompute V15's fixed-encoder temporal-corridor gate."""

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

import run_thesis_five_state_reentry_v15_fixed_encoder_temporal_corridor as runner
import verify_thesis_five_state_reentry_v14_temporal_corridor as v14_verifier
from uav_vpp_guidance.evaluation.thesis_reentry_v15_fixed_encoder_corridor import METHODS, OPPONENTS, SOURCE_ID, summarize_records
from uav_vpp_guidance.hierarchy.shared_intent_profiles import compile_profile


def _verify_fixed_encoder_anchor(row: Mapping[str, Any], profile_name: str) -> None:
    """Confirm every serialized 66-D vector carries the declared static anchor."""

    if not bool(row.get("handoff_reached")):
        return
    profile = compile_profile(profile_name)
    for step in row.get("window", []):
        observation = np.asarray(step["observation_66d"], dtype=np.float64).reshape(-1)
        if observation.shape != (66,):
            raise RuntimeError("V15 invalid 66-D anchor observation")
        if not np.allclose(observation[22:28], profile["target_vector"], rtol=0.0, atol=1e-7):
            raise RuntimeError("V15 fixed encoder target segment drifted")
        if not np.allclose(observation[28:34], profile["weight_vector"], rtol=0.0, atol=1e-7):
            raise RuntimeError("V15 fixed encoder weight segment drifted")


def verify(config_path: Path, output: Path | None = None) -> dict[str, Any]:
    """Replay V15's fixed 66-D anchor and dynamic-corridor evidence contract."""

    config = runner._load_runtime_config(config_path)
    manifest_path = runner._resolve(config["inputs"]["manifest"])
    manifest = runner._load(manifest_path)
    root = output or runner._resolve(config["outputs"]["root"])
    if not root.is_dir():
        raise RuntimeError(f"V15 output is missing: {root}")
    records: list[Mapping[str, Any]] = []
    for method in METHODS:
        for opponent in OPPONENTS:
            for scenario in manifest["scenarios"]:
                path = root / "episodes" / method / opponent / f"{scenario['name']}.json"
                if not path.is_file():
                    raise RuntimeError(f"V15 episode record is missing: {path}")
                row = json.loads(path.read_text(encoding="utf-8")).get("record")
                if not isinstance(row, Mapping) or row.get("source_id") != SOURCE_ID:
                    raise RuntimeError(f"V15 invalid episode record: {path}")
                if row.get("method") != method or row.get("opponent") != opponent:
                    raise RuntimeError(f"V15 episode identity drifted: {path}")
                if row.get("pair_key") != scenario["metadata"]["pair_key"]:
                    raise RuntimeError(f"V15 pair key drifted: {path}")
                v14_verifier._verify_window(row, config)
                _verify_fixed_encoder_anchor(row, str(config["observation_contract"]["encoder_profile"]))
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
    reported = json.loads((root / "fixed_encoder_temporal_corridor_gate.json").read_text(encoding="utf-8"))
    if not v14_verifier._equivalent(reported, {"source_id": SOURCE_ID, **recomputed}):
        raise RuntimeError("V15 reported gate does not match independent recomputation")
    run_manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    if run_manifest.get("source_id") != SOURCE_ID or int(run_manifest.get("records", -1)) != len(records):
        raise RuntimeError("V15 run manifest is inconsistent")
    if run_manifest.get("config_sha256") != runner._hash(config_path):
        raise RuntimeError("V15 run manifest config hash drifted")
    if run_manifest.get("manifest_sha256") != runner._hash(manifest_path):
        raise RuntimeError("V15 run manifest manifest hash drifted")
    if run_manifest.get("encoder_profile") != config["observation_contract"]["encoder_profile"]:
        raise RuntimeError("V15 run manifest encoder profile drifted")
    if run_manifest.get("training") != "prohibited" or run_manifest.get("reward_recording") != "prohibited":
        raise RuntimeError("V15 run manifest violates its non-learning contract")
    if not isinstance(run_manifest.get("authorized_code_files"), list) or not run_manifest["authorized_code_files"]:
        raise RuntimeError("V15 run manifest lacks the authorized code-file record")
    return {
        "source_id": SOURCE_ID,
        "records": len(records),
        "reported_gate_matches_recomputed": True,
        "corridor_contract_replayed": True,
        "fixed_encoder_profile": config["observation_contract"]["encoder_profile"],
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
