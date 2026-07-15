#!/usr/bin/env python3
"""Run V10's telemetry-hardened non-learning VPP action-authority probe."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_thesis_five_state_reentry_v9_action_authority as v9
from uav_vpp_guidance.evaluation.thesis_reentry_v9_action_authority import OPPONENTS, REFERENCE_METHOD, summarize_records
from uav_vpp_guidance.training import thesis_neutral_postmerge_reentry_recovery_pilot as engine
from uav_vpp_guidance.training.thesis_shared_skill_geometry import FrozenP3Encoder
from uav_vpp_guidance.evaluation.thesis_five_state_raw_si_v5_record_only import build_record_only_env


SOURCE_ID = "THESIS-FIVE-STATE-REENTRY-VPP-ACTION-AUTHORITY-V10"


class V10RuntimeError(RuntimeError):
    """Raised when V10 leaves its frozen record or telemetry contract."""


def _load(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise V10RuntimeError(f"expected mapping: {path}")
    return value


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value: Any, name: str) -> float:
    result = v9._finite_or_none(value)
    if result is None:
        raise V10RuntimeError(f"V10 required telemetry is missing or non-finite: {name}")
    return result


def _annotate_telemetry(result: dict[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """Convert backend-unobserved vertical bias into an explicit receipt."""
    contract = config["telemetry_contract"]
    required = tuple(contract["required_finite"])
    guidance_keys = tuple(contract["guidance_command_required_finite"])
    for step in result.get("window", []):
        for key in required:
            step[key] = _finite(step.get(key), key)
        guidance = step.get("guidance_command")
        if not isinstance(guidance, Mapping):
            raise V10RuntimeError("V10 guidance_command is missing")
        guidance_receipt = {key: _finite(guidance.get(key), "guidance_command." + key) for key in guidance_keys}
        vertical = v9._finite_or_none(step.get("vp_vertical_bias_m"))
        step["vp_vertical_bias_m"] = vertical
        step["vp_vertical_bias_observed"] = vertical is not None
        step["guidance_command"] = guidance_receipt
        step["vertical_response_proxy"] = {
            "applied_vertical_action": float(step["applied_delta"][2]),
            "altitude_diff_observation": _finite(step.get("altitude_diff_observation"), "altitude_diff_observation"),
            "nz_cmd": step["nz_cmd"],
            "actual_nz_g": step["actual_nz_g"],
        }
    result["telemetry_contract"] = {
        "required_finite": list(required),
        "backend_unobserved": list(contract["backend_unobserved"]),
        "vertical_response_proxies": list(contract["vertical_response_proxies"]),
    }
    return result


def _methods(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    try:
        return v9._methods(config)
    except Exception as error:
        raise V10RuntimeError(str(error)) from error


def validate_design(config_path: Path) -> dict[str, Any]:
    config = v9._load_runtime_config(config_path)
    if config.get("source_id") != SOURCE_ID:
        raise V10RuntimeError("V10 source ID drifted")
    authorization = config.get("authorization", {})
    for key in ("training_permitted", "reward_change_permitted", "candidate_policy_loading_permitted", "high_level_ppo_training_permitted"):
        if authorization.get(key) is not False:
            raise V10RuntimeError(f"V10 must prohibit {key}")
    if tuple(config.get("opponents", {}).get("order", ())) != OPPONENTS or config.get("opponents", {}).get("pooled_gate") != "prohibited":
        raise V10RuntimeError("V10 opponent contract drifted")
    methods = _methods(config)
    manifest_path = _resolve(config["inputs"]["manifest"])
    if _hash(manifest_path) != config["inputs"]["manifest_sha256"]:
        raise V10RuntimeError("V10 manifest hash mismatch")
    manifest = _load(manifest_path)
    if manifest.get("source_id") != SOURCE_ID + "-MANIFEST12" or len(manifest.get("scenarios", [])) != 12:
        raise V10RuntimeError("V10 requires its complete frozen 12-scenario manifest")
    if manifest.get("protocol", {}).get("training_permitted") is not False or manifest.get("protocol", {}).get("reward_recording") != "prohibited":
        raise V10RuntimeError("V10 manifest is not non-learning and reward-free")
    telemetry = config.get("telemetry_contract", {})
    if tuple(telemetry.get("backend_unobserved", ())) != ("vp_vertical_bias_m",):
        raise V10RuntimeError("V10 backend-unobserved telemetry contract drifted")
    output = _resolve(config["outputs"]["root"])
    if not output.is_absolute() or output.exists():
        raise V10RuntimeError("V10 output root must be absolute and absent")
    for key in ("runtime_template", "runtime_registry", "frozen_p3_encoder"):
        path = _resolve(config["inputs"][key])
        if not path.is_file() or _hash(path) != config["inputs"][key + "_sha256"]:
            raise V10RuntimeError(f"V10 frozen input mismatch: {key}")
    free_gb = shutil.disk_usage(output.parent).free / 1024 ** 3
    return {
        "source_id": SOURCE_ID,
        "mode": "nonlearning_telemetry_hardened_action_authority_preflight",
        "manifest_scenarios": len(manifest["scenarios"]),
        "planned_records": len(methods) * len(OPPONENTS) * len(manifest["scenarios"]),
        "methods": list(methods),
        "free_disk_gb": round(free_gb, 3),
        "disk_gate_would_pass": free_gb >= float(config["observability"]["min_free_disk_gb"]),
        "output_root_absent": True,
    }


def _record(scenario: Mapping[str, Any], opponent: str, method: str, result: Mapping[str, Any]) -> dict[str, Any]:
    metadata = scenario["metadata"]
    return {
        "source_id": SOURCE_ID,
        "opponent": opponent,
        "method": method,
        "pair_key": metadata["pair_key"],
        "mirror_sign": metadata["mirror_sign"],
        "height_condition": metadata["height_condition"],
        **dict(result),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = v9._load_runtime_config(args.config)
    preflight = validate_design(args.config)
    if not args.execute:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0
    v9._require_execution_authorization(config)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise V10RuntimeError("V10 requires a clean worktree")

    methods = _methods(config)
    manifest_path = _resolve(config["inputs"]["manifest"])
    manifest = _load(manifest_path)
    registry = _load(_resolve(config["inputs"]["runtime_registry"]))
    output = _resolve(config["outputs"]["root"])
    output.mkdir(parents=True)
    with (output / "authorized_config.yaml").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(yaml.safe_dump(config, sort_keys=False))
    records: list[dict[str, Any]] = []
    try:
        for method, delta in methods.items():
            for opponent in OPPONENTS:
                for scenario in manifest["scenarios"]:
                    env = build_record_only_env(config, registry, opponent)
                    policy = engine._specialist(registry, "frozen_fixed_head_on")
                    encoder = FrozenP3Encoder(Path(config["inputs"]["frozen_p3_encoder"]), config["inputs"]["frozen_p3_encoder_sha256"])
                    episode = v9.ActionAuthorityEpisode(env, policy, encoder, config)
                    try:
                        result = episode.run(scenario, int(scenario["metadata"]["scenario_seed"]), method=method, delta=delta)
                    finally:
                        episode.close()
                    if result.get("handoff_reached"):
                        result = _annotate_telemetry(result, config)
                    row = _record(scenario, opponent, method, result)
                    records.append(row)
                    v9._write_json(output / "episodes" / method / opponent / f"{scenario['name']}.json", {"record": row})
        gates = config["gates"]
        summary = summarize_records(
            records,
            method_deltas={name: delta.tolist() for name, delta in methods.items()},
            min_window_steps=int(config["observability"]["local_window_steps"]),
            min_pairs=int(gates["min_qualifying_pairs_per_opponent"]),
            min_applied_delta_norm=float(gates["min_applied_delta_norm"]),
            practical_delta=float(gates["practical_intent_loss_delta_vs_reference_max"]),
            safety_delta_max=float(gates["ego_crash_oob_delta_vs_reference_max"]),
        )
        gate = {"source_id": SOURCE_ID, **summary}
        v9._write_json(output / "action_authority_gate.json", gate)
        v9._write_json(output / "run_manifest.json", {
            "source_id": SOURCE_ID,
            "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "config_sha256": _hash(args.config),
            "manifest_sha256": _hash(manifest_path),
            "records": len(records),
            "training": "prohibited",
            "reward_recording": "prohibited",
        })
        print(json.dumps(gate, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        v9._write_json(output / "failure.json", {"source_id": SOURCE_ID, "error": repr(error), "completed_records": len(records)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
