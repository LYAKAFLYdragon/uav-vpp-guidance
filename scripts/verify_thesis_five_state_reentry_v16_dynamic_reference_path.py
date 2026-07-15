#!/usr/bin/env python3
"""Independently replay V16's current-information dynamic-path evidence gate."""

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

import run_thesis_five_state_reentry_v16_dynamic_reference_path as runner
import verify_thesis_five_state_reentry_v14_temporal_corridor as v14_verifier
from uav_vpp_guidance.evaluation.thesis_reentry_v16_dynamic_reference_path import (
    METHODS,
    OPPONENTS,
    REFERENCE_METHOD,
    SOURCE_ID,
    semantic_axes_for_method,
    semantic_profile_for_method,
    summarize_records,
)
from uav_vpp_guidance.hierarchy.shared_intent_profiles import compile_profile
from uav_vpp_guidance.virtual_point.dynamic_reference_path import (
    build_three_anchor_path,
    compile_path_anchor_action,
    path_progress_from_range_rate,
    select_path_anchor,
    semantic_profile_for_anchor,
)


def _finite_vector(value: Any, *, length: int, label: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"V16 invalid {label}") from error
    if vector.shape != (length,) or not np.isfinite(vector).all():
        raise RuntimeError(f"V16 invalid {label}")
    return vector


def _verify_fixed_encoder_anchor(row: Mapping[str, Any], profile_name: str) -> None:
    if not bool(row.get("handoff_reached")):
        return
    profile = compile_profile(profile_name)
    for step in row.get("window", []):
        observation = _finite_vector(step.get("observation_66d"), length=66, label="66-D observation")
        if not np.allclose(observation[22:28], profile["target_vector"], rtol=0.0, atol=1e-7):
            raise RuntimeError("V16 fixed 66-D target segment drifted")
        if not np.allclose(observation[28:34], profile["weight_vector"], rtol=0.0, atol=1e-7):
            raise RuntimeError("V16 fixed 66-D weight segment drifted")


def _verify_window(row: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    window = row.get("window")
    window_steps = int(row.get("window_steps", 0))
    if not bool(row.get("handoff_reached")):
        if window_steps != 0 or window not in (None, []):
            raise RuntimeError("V16 non-handoff record contains a window")
        return
    if not isinstance(window, list) or len(window) != window_steps:
        raise RuntimeError("V16 window length is inconsistent")
    method = str(row.get("method"))
    if method not in METHODS:
        raise RuntimeError("V16 record method is outside the frozen matrix")
    path_config = runner._path_config(config)
    for index, step in enumerate(window):
        if int(step.get("local_step", -1)) != index:
            raise RuntimeError("V16 local step index drifted")
        own_position = _finite_vector(step.get("own_position_neu"), length=3, label="own position")
        target_position = _finite_vector(step.get("target_position_neu"), length=3, label="target position")
        predicted_position = _finite_vector(
            step.get("predicted_target_position_neu"), length=3, label="predicted target position"
        )
        if step.get("prediction_valid") is not True or step.get("prediction_fallback") is not False:
            raise RuntimeError("V16 prediction fallback or invalid prediction occurred")
        range_rate = float(step.get("range_rate_mps"))
        if not math.isfinite(range_rate):
            raise RuntimeError("V16 range-rate input is non-finite")
        path = build_three_anchor_path(
            own_position_neu=own_position,
            target_position_neu=target_position,
            predicted_target_position_neu=predicted_position,
            config=path_config,
        )
        if not np.allclose(path, np.asarray(step.get("anchors_neu"), dtype=np.float64), rtol=0.0, atol=1e-8):
            raise RuntimeError("V16 path anchors do not replay from recorded current information")
        progress = path_progress_from_range_rate(range_rate, path_config)
        selected_index, selected_anchor = select_path_anchor(path, progress)
        if not math.isclose(float(step.get("path_progress")), progress, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("V16 path progress does not replay from range-rate")
        if int(step.get("selected_anchor_index", -1)) != selected_index:
            raise RuntimeError("V16 selected anchor index drifted")
        if step.get("selected_anchor_profile") != semantic_profile_for_anchor(selected_index):
            raise RuntimeError("V16 selected profile drifted")
        expected_stage = "escape" if selected_index < 2 else "reentry"
        if step.get("objective_stage") != expected_stage or step.get("objective_profile") != semantic_profile_for_anchor(selected_index):
            raise RuntimeError("V16 objective does not match the selected path anchor")
        expected_action_profile = semantic_profile_for_method(method, semantic_profile_for_anchor(selected_index))
        if step.get("action_profile") != expected_action_profile or step.get("semantic_axes") != semantic_axes_for_method(method):
            raise RuntimeError("V16 action/profile contract drifted")
        _finite_vector(step.get("after_component_errors"), length=6, label="component errors")
        if not math.isfinite(float(step.get("after_weighted_error"))):
            raise RuntimeError("V16 weighted intent error is non-finite")
        _finite_vector(step.get("env_vpp_world_offset_m"), length=3, label="environment VPP offset")
        _finite_vector(step.get("guidance_command_normalized"), length=3, label="guidance command")
        if not math.isfinite(float(step.get("actual_nz_g"))):
            raise RuntimeError("V16 actual normal-load response is non-finite")
        semantic_receipt = step.get("semantic_receipt")
        if not isinstance(semantic_receipt, Mapping):
            raise RuntimeError("V16 semantic receipt is missing")
        if semantic_receipt.get("profile") != expected_action_profile:
            raise RuntimeError("V16 semantic receipt profile drifted")
        _finite_vector(semantic_receipt.get("applied_action"), length=3, label="semantic action")
        path_receipt = step.get("path_action_receipt")
        if method == "progress_triggered_reference_path":
            if not isinstance(path_receipt, Mapping):
                raise RuntimeError("V16 path method lacks a path-action receipt")
            expected_receipt = compile_path_anchor_action(
                semantic_action=semantic_receipt["applied_action"],
                selected_anchor_neu=selected_anchor,
                generator_anchor_neu=predicted_position,
                raw_action_ranges=config["semantic_compiler"]["raw_action_ranges"],
                raw_action_offset_frame=str(config["semantic_compiler"]["raw_action_offset_frame"]),
                own_state={"position_neu": own_position},
                target_state={"position_neu": target_position},
            )
            if not v14_verifier._equivalent(path_receipt, expected_receipt):
                raise RuntimeError("V16 path-action receipt does not replay")
        elif path_receipt is not None:
            raise RuntimeError("V16 non-path comparator unexpectedly used the path adapter")
        if method == REFERENCE_METHOD and float(semantic_receipt.get("actual_semantic_delta_norm_m")) != 0.0:
            raise RuntimeError("V16 reference method applied a semantic offset")


def verify(config_path: Path, output: Path | None = None) -> dict[str, Any]:
    """Replay V16 path selection then recompute its unpooled mechanism gate."""

    config = runner._load_runtime_config(config_path)
    manifest_path = runner._resolve(config["inputs"]["manifest"])
    manifest = runner._load(manifest_path)
    root = output or runner._resolve(config["outputs"]["root"])
    if not root.is_dir():
        raise RuntimeError(f"V16 output is missing: {root}")
    records: list[Mapping[str, Any]] = []
    for method in METHODS:
        for opponent in OPPONENTS:
            for scenario in manifest["scenarios"]:
                path = root / "episodes" / method / opponent / f"{scenario['name']}.json"
                if not path.is_file():
                    raise RuntimeError(f"V16 episode record is missing: {path}")
                row = json.loads(path.read_text(encoding="utf-8")).get("record")
                if not isinstance(row, Mapping) or row.get("source_id") != SOURCE_ID:
                    raise RuntimeError(f"V16 invalid episode record: {path}")
                if row.get("method") != method or row.get("opponent") != opponent:
                    raise RuntimeError(f"V16 episode identity drifted: {path}")
                if row.get("pair_key") != scenario["metadata"]["pair_key"]:
                    raise RuntimeError(f"V16 pair key drifted: {path}")
                _verify_window(row, config)
                _verify_fixed_encoder_anchor(row, str(config["observation_contract"]["encoder_profile"]))
                records.append(row)
    gates = config["gates"]
    recomputed = summarize_records(
        records,
        scenarios_per_opponent=len(manifest["scenarios"]),
        min_window_steps=int(config["observability"]["local_window_steps"]),
        min_pairs=int(gates["min_qualifying_pairs_per_opponent"]),
        min_anchor_path_delta_norm_m=float(gates["min_anchor_path_delta_norm_m"]),
        min_paired_vpp_delta_norm_m=float(gates["min_paired_vpp_delta_norm_m"]),
        min_paired_guidance_delta_norm=float(gates["min_paired_guidance_delta_norm"]),
        max_escape_range_rate_worsening=float(gates["max_escape_range_rate_worsening"]),
        safety_delta_max=float(gates["ego_crash_oob_delta_vs_reference_max"]),
    )
    reported = json.loads((root / "dynamic_reference_path_gate.json").read_text(encoding="utf-8"))
    if not v14_verifier._equivalent(reported, {"source_id": SOURCE_ID, **recomputed}):
        raise RuntimeError("V16 reported gate does not match independent recomputation")
    run_manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    if run_manifest.get("source_id") != SOURCE_ID or int(run_manifest.get("records", -1)) != len(records):
        raise RuntimeError("V16 run manifest is inconsistent")
    if run_manifest.get("config_sha256") != runner._hash(config_path):
        raise RuntimeError("V16 run manifest config hash drifted")
    if run_manifest.get("manifest_sha256") != runner._hash(manifest_path):
        raise RuntimeError("V16 run manifest manifest hash drifted")
    if run_manifest.get("training") != "prohibited" or run_manifest.get("reward_recording") != "prohibited":
        raise RuntimeError("V16 run manifest violates its non-learning contract")
    if run_manifest.get("path_input") != "current_own_target_and_frozen_prediction_only":
        raise RuntimeError("V16 run manifest path-input contract drifted")
    if not isinstance(run_manifest.get("authorized_code_files"), list) or not run_manifest["authorized_code_files"]:
        raise RuntimeError("V16 run manifest lacks the authorized code-file record")
    return {
        "source_id": SOURCE_ID,
        "records": len(records),
        "reported_gate_matches_recomputed": True,
        "dynamic_path_contract_replayed": True,
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
