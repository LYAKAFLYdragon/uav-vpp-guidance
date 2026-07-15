#!/usr/bin/env python3
"""Run V13's repaired non-learning six-objective controllability diagnostic."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_thesis_five_state_reentry_v12_intent_controllability as v12
from uav_vpp_guidance.evaluation.thesis_five_state_raw_si_v5_record_only import build_record_only_env
from uav_vpp_guidance.evaluation.thesis_reentry_v12_intent_controllability import METHODS, OPPONENTS, summarize_records
from uav_vpp_guidance.training import thesis_neutral_postmerge_reentry_recovery_pilot as engine
from uav_vpp_guidance.training.thesis_shared_skill_geometry import FrozenP3Encoder
from uav_vpp_guidance.virtual_point.relative_frame_semantic import RelativeFrameSemanticCompiler, RelativeFrameSemanticConfig


SOURCE_ID = "THESIS-FIVE-STATE-REENTRY-INTENT-CONTROLLABILITY-V13"


class V13RuntimeError(RuntimeError):
    """Raised when V13 departs from its frozen repaired runtime contract."""


def _load(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise V13RuntimeError(f"expected YAML mapping: {path}")
    return value


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_runtime_config(path: Path) -> dict[str, Any]:
    overlay = _load(path)
    if "base_config" not in overlay:
        return overlay
    base_path = _resolve(overlay["base_config"])
    if _hash(base_path) != str(overlay.get("base_config_sha256") or ""):
        raise V13RuntimeError("V13 authorization overlay base config hash mismatch")
    config = copy.deepcopy(_load(base_path))
    config["authorization"] = copy.deepcopy(overlay.get("authorization", {}))
    config["_authorization_execution"] = copy.deepcopy(overlay.get("execution", {}))
    return config


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _runtime_contract(config: Mapping[str, Any]) -> None:
    execution = config.get("execution")
    if not isinstance(execution, Mapping) or not execution.get("runtime_template"):
        raise V13RuntimeError("V13 requires execution.runtime_template before output creation")
    runtime_path = _resolve(str(execution["runtime_template"]))
    input_path = _resolve(str(config["inputs"]["runtime_template"]))
    if runtime_path != input_path:
        raise V13RuntimeError("V13 execution.runtime_template must equal inputs.runtime_template")
    if not runtime_path.is_file() or _hash(runtime_path) != str(config["inputs"]["runtime_template_sha256"]):
        raise V13RuntimeError("V13 runtime template SHA mismatch")


def validate_design(config_path: Path) -> dict[str, Any]:
    """Validate V13 before any output directory or JSBSim environment exists."""

    config = _load_runtime_config(config_path)
    if config.get("source_id") != SOURCE_ID:
        raise V13RuntimeError("V13 source ID drifted")
    for key in ("training_permitted", "tuning_permitted", "reward_change_permitted", "candidate_policy_loading_permitted", "high_level_ppo_training_permitted"):
        if config.get("authorization", {}).get(key) is not False:
            raise V13RuntimeError(f"V13 must prohibit {key}")
    if tuple(config.get("opponents", {}).get("order", ())) != OPPONENTS or config["opponents"].get("pooled_gate") != "prohibited":
        raise V13RuntimeError("V13 opponent contract drifted")
    methods = v12._methods(config)
    compiler_values = dict(config["semantic_compiler"])
    compiler_values.pop("profile", None)
    compiler_values.pop("raw_action_ranges", None)
    compiler = RelativeFrameSemanticCompiler(RelativeFrameSemanticConfig(**compiler_values))
    if compiler.config.raw_action_offset_frame != "world_neu":
        raise V13RuntimeError("V13 must preserve V11's frozen world-frame action interface")
    _runtime_contract(config)
    manifest_path = _resolve(config["inputs"]["manifest"])
    if _hash(manifest_path) != str(config["inputs"]["manifest_sha256"]):
        raise V13RuntimeError("V13 manifest hash mismatch")
    manifest = _load(manifest_path)
    if manifest.get("source_id") != SOURCE_ID + "-MANIFEST12" or len(manifest.get("scenarios", [])) != 12:
        raise V13RuntimeError("V13 requires its complete 12-scenario manifest")
    if manifest.get("protocol", {}).get("training_permitted") is not False or manifest["protocol"].get("reward_recording") != "prohibited":
        raise V13RuntimeError("V13 manifest is not non-learning and reward-free")
    output = _resolve(config["outputs"]["root"])
    if not output.is_absolute() or output.exists():
        raise V13RuntimeError("V13 output root must be absolute and absent")
    for key in ("runtime_template", "runtime_registry", "frozen_p3_encoder"):
        path = _resolve(config["inputs"][key])
        if not path.is_file() or _hash(path) != str(config["inputs"][key + "_sha256"]):
            raise V13RuntimeError(f"V13 frozen input mismatch: {key}")
    free_gb = shutil.disk_usage(output.parent).free / 1024**3
    return {"source_id": SOURCE_ID, "mode": "nonlearning_six_objective_controllability_preflight", "manifest_scenarios": len(manifest["scenarios"]), "planned_records": len(methods) * len(OPPONENTS) * len(manifest["scenarios"]), "methods": list(methods), "free_disk_gb": round(free_gb, 3), "disk_gate_would_pass": free_gb >= float(config["observability"]["min_free_disk_gb"]), "output_root_absent": True, "runtime_contract_passed": True}


def _require_execution_authorization(config: Mapping[str, Any], scenario_count: int) -> None:
    try:
        v12._require_execution_authorization(config, scenario_count)
    except Exception as error:
        raise V13RuntimeError(str(error)) from error


def _record(scenario: Mapping[str, Any], opponent: str, method: str, result: Mapping[str, Any]) -> dict[str, Any]:
    metadata = scenario["metadata"]
    return {"source_id": SOURCE_ID, "opponent": opponent, "method": method, "pair_key": metadata["pair_key"], "mirror_sign": metadata["mirror_sign"], "height_condition": metadata["height_condition"], **dict(result)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = _load_runtime_config(args.config)
    preflight = validate_design(args.config)
    if not args.execute:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0
    manifest = _load(_resolve(config["inputs"]["manifest"]))
    _require_execution_authorization(config, len(manifest["scenarios"]))
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise V13RuntimeError("V13 requires a clean worktree")
    methods = v12._methods(config)
    registry = _load(_resolve(config["inputs"]["runtime_registry"]))
    output = _resolve(config["outputs"]["root"])
    output.mkdir(parents=True)
    _write_json(output / "authorized_config.json", {"config": config, "config_sha256": _hash(args.config)})
    compiler_values = dict(config["semantic_compiler"])
    compiler_values.pop("profile", None)
    compiler_values.pop("raw_action_ranges", None)
    compiler = RelativeFrameSemanticCompiler(RelativeFrameSemanticConfig(**compiler_values))
    records: list[dict[str, Any]] = []
    try:
        for method, axes in methods.items():
            for opponent in OPPONENTS:
                for scenario in manifest["scenarios"]:
                    env = build_record_only_env(config, registry, opponent)
                    policy = engine._specialist(registry, "frozen_fixed_head_on")
                    encoder = FrozenP3Encoder(Path(config["inputs"]["frozen_p3_encoder"]), config["inputs"]["frozen_p3_encoder_sha256"])
                    episode = v12.IntentControllabilityEpisode(env, policy, encoder, config, compiler)
                    try:
                        result = episode.run(scenario, int(scenario["metadata"]["scenario_seed"]), method=method, enabled_axes=axes)
                    finally:
                        episode.close()
                    row = _record(scenario, opponent, method, result)
                    records.append(row)
                    _write_json(output / "episodes" / method / opponent / f"{scenario['name']}.json", {"record": row})
        gates = config["gates"]
        summary = summarize_records(records, scenarios_per_opponent=len(manifest["scenarios"]), min_window_steps=int(config["observability"]["local_window_steps"]), min_pairs=int(gates["min_qualifying_pairs_per_opponent"]), max_high_priority_worsening=float(gates["max_high_priority_component_worsening"]), max_any_component_worsening=float(gates["max_any_component_worsening"]), min_component_improvement=float(gates["min_component_improvement"]), min_improved_components=int(gates["min_improved_components"]), min_weighted_error_improvement=float(gates["min_weighted_error_improvement"]), safety_delta_max=float(gates["ego_crash_oob_delta_vs_reference_max"]), profile_name=str(config["semantic_compiler"]["profile"]))
        gate = {"source_id": SOURCE_ID, **summary}
        _write_json(output / "intent_controllability_gate.json", gate)
        _write_json(output / "run_manifest.json", {"source_id": SOURCE_ID, "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "config_sha256": _hash(args.config), "manifest_sha256": _hash(_resolve(config["inputs"]["manifest"])), "records": len(records), "training": "prohibited", "reward_recording": "prohibited"})
        print(json.dumps(gate, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        _write_json(output / "failure.json", {"source_id": SOURCE_ID, "error": repr(error), "completed_records": len(records)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
