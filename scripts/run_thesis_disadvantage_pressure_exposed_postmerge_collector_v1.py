#!/usr/bin/env python3
"""Collect one frozen pressure-exposed post-merge cohort without learning."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from uav_vpp_guidance.evaluation import thesis_five_state_raw_si_v5_record_only as raw_v5  # noqa: E402
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv  # noqa: E402
from uav_vpp_guidance.evaluation.pressure_exposed_postmerge_contract import (  # noqa: E402
    OPPONENTS,
    SOURCE_ID,
    find_first_pressure_window,
    summarize_records,
)
from uav_vpp_guidance.hierarchy.specialist_policy import FrozenSpecialistPolicy  # noqa: E402
from uav_vpp_guidance.training import thesis_neutral_postmerge_reentry_recovery_pilot as engine  # noqa: E402
from uav_vpp_guidance.training.thesis_shared_skill_geometry import FrozenP3Encoder  # noqa: E402


class PressureCollectorError(RuntimeError):
    """Raised when execution would drift from the pre-registered contract."""


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise PressureCollectorError(f"YAML mapping required: {path}")
    return value


def _merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: Path) -> dict[str, Any]:
    overlay = _load_yaml(path)
    base_path = overlay.pop("base_config", None)
    if not base_path:
        return overlay
    replace_sections = overlay.pop("replace_sections", [])
    base = load_config(_resolve(str(base_path)))
    result = _merge(base, overlay)
    for section in replace_sections:
        if section not in overlay or not isinstance(overlay[section], Mapping):
            raise PressureCollectorError(f"replacement section is missing: {section}")
        result[section] = copy.deepcopy(dict(overlay[section]))
    return result


def _require_input(config: Mapping[str, Any], key: str, *, check_file: bool) -> Path:
    path = _resolve(str(config["inputs"][key]))
    expected = str(config["inputs"][f"{key}_sha256"])
    if check_file and (not path.is_file() or _hash(path) != expected):
        raise PressureCollectorError(f"frozen input SHA mismatch: {key}")
    return path


def validate_design(path: Path, *, check_external_inputs: bool = False) -> dict[str, Any]:
    config = load_config(path)
    if (
        config.get("source_id") != SOURCE_ID
        or config.get("stage") != "nonlearning_pressure_exposed_postmerge_collection"
    ):
        raise PressureCollectorError("source identity or stage drifted")
    authorization = config.get("authorization")
    if not isinstance(authorization, Mapping):
        raise PressureCollectorError("authorization mapping is required")
    for key in (
        "training_permitted", "tuning_permitted", "reward_change_permitted",
        "candidate_action_execution_permitted", "high_level_ppo_training_permitted",
        "shared_skill_training_permitted", "guidance_pid_change_permitted",
        "snapshot_restore_permitted", "future_state_injection_permitted", "history_padding_permitted",
    ):
        if authorization.get(key) is not False:
            raise PressureCollectorError(f"collector must prohibit {key}")
    if tuple(config.get("opponents", {}).get("order", ())) != OPPONENTS:
        raise PressureCollectorError("opponent registry order drifted")
    if config.get("opponents", {}).get("pooled_gate") != "prohibited":
        raise PressureCollectorError("pooled gate must remain prohibited")
    contract = config.get("contract", {})
    if (
        contract.get("backend") != "jsbsim" or contract.get("strict_backend") is not True
        or contract.get("action_semantics") != "tactical_basis_v1"
        or contract.get("reference_controller") != "frozen_fixed_head_on"
        or contract.get("routing_enabled") is not False or contract.get("reset_at_handoff") is not False
    ):
        raise PressureCollectorError("runtime controller contract drifted")
    gates = config.get("gates", {})
    if (
        int(gates.get("required_window_steps", -1)) != 40
        or int(gates.get("min_pressure_steps_per_window", -1)) < 10
        or int(gates.get("min_qualifying_episodes_per_opponent", -1)) < 4
        or int(gates.get("min_distinct_signatures_per_opponent", -1)) < 4
        or tuple(gates.get("required_opponents", ())) != OPPONENTS
    ):
        raise PressureCollectorError("pressure gate is weaker than the pre-registration")
    manifest_path = _require_input(config, "scenario_manifest", check_file=True)
    manifest = _load_yaml(manifest_path)
    scenarios = manifest.get("scenarios", [])
    if (
        manifest.get("source_id") != config["inputs"]["scenario_manifest_source_id"]
        or len(scenarios) != 16
        or any(item.get("metadata", {}).get("initial_class") != "disadvantage" for item in scenarios)
    ):
        raise PressureCollectorError("pressure manifest contract drifted")
    signatures = {str(item["metadata"].get("mirror_pair_id")) for item in scenarios}
    mirrors = {str(item["metadata"].get("mirror_sign")) for item in scenarios}
    if len(signatures) != 8 or mirrors != {"negative", "positive"}:
        raise PressureCollectorError("pressure manifest lacks mirror/signature coverage")
    for key in ("runtime_registry", "frozen_p3_encoder", "frozen_head_on_specialist"):
        _require_input(config, key, check_file=check_external_inputs)
    return {
        "source_id": SOURCE_ID,
        "execution_permitted": bool(authorization.get("execution_permitted")),
        "planned_records": len(scenarios) * len(OPPONENTS),
        "scenarios": len(scenarios),
        "opponents": list(OPPONENTS),
        "gate": dict(gates),
    }


def _specialist(registry: Mapping[str, Any]) -> FrozenSpecialistPolicy:
    entry = registry["specialists"]["frozen_fixed_head_on"]
    return FrozenSpecialistPolicy(
        checkpoint_path=str(entry["checkpoint"]),
        config_path=str(_resolve(str(entry["config_path"]))),
        device="cpu",
    )


def _build_env(config: Mapping[str, Any], registry: Mapping[str, Any], opponent: str) -> CloseRangeTrackingEnv:
    """Construct the frozen strict-JSBSim chain from this source's registry.

    The V5 helper expects a legacy runtime-template field that this collector
    deliberately does not use.  Keeping construction explicit prevents that
    hidden mapping from changing the registered opponent or runtime.
    """

    base = copy.deepcopy(engine.load_yaml(raw_v5.legacy.BASE_ENV_CONFIG))
    runtime = config["runtime"]
    base["backend"] = "jsbsim"
    base["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
    base.setdefault("env", {}).update(
        {
            "use_jsbsim": True,
            "strict_backend": True,
            "legacy_project_root": str(runtime["legacy_project_root"]),
            "aircraft_model": str(runtime["aircraft_model"]),
            "high_level_dt": float(runtime["high_level_dt_s"]),
            "max_high_level_steps": int(config["observability"]["max_high_level_steps"]),
            "success_range_m": 0.0,
            "success_ata_deg": 0.0,
            "success_hold_time_s": 1.0,
            "max_range_m": 12000.0,
        }
    )
    base["observation"] = {
        "include_gains": True,
        "include_guidance_state": False,
        "include_saturation": False,
        "include_opponent_stage": False,
        "include_task_type": True,
    }
    base["trajectory_prediction"] = copy.deepcopy(runtime["trajectory_prediction"])
    base["virtual_point"] = copy.deepcopy(runtime["virtual_point"])
    base["opponent_stage"] = opponent
    base["attack_zone"] = copy.deepcopy(config["fixed_contract"]["attack_zone"])
    entry = registry["opponents"][opponent]
    return CloseRangeTrackingEnv(
        base,
        opponent_policy=engine._instantiate_opponent(entry),
        opponent_config={"stage": opponent, **copy.deepcopy(entry)},
    )


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_canonical(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _stable_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _normalized_steps(ledger: Mapping[str, Any], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    previous_hash = _stable_hash({"header": ledger["header"]})
    for item in ledger.get("steps", []):
        base = np.asarray(item["base_observation_16d"], dtype=np.float64)
        values = raw_v5._base_map(base)
        aa_deg = raw_v5._angle_deg(values["aa_sin"], values["aa_cos"])
        ata_deg = raw_v5._angle_deg(values["ata_sin"], values["ata_cos"])
        step = {
            "step": int(item["step"]),
            "phase": str(item["phase"]),
            "dynamic_state": str(item["dynamic_state"]),
            "history_length": int(item["history_length"]),
            "history_padding_used": bool(item["history_padding_used"]),
            "base_observation_16d": [float(value) for value in base],
            "observation_66d": item["observation_66d"],
            "reference_action": item["normalized_vpp_action"],
            "active_action_semantics": str(config["contract"]["action_semantics"]),
            "configured_action_semantics": str(config["contract"]["action_semantics"]),
            "tactical_basis_enabled": True,
            "prediction_valid": bool(item["prediction_valid"]),
            "prediction_fallback": bool(item["prediction_fallback"]),
            "vpp_world_offset_m": [
                item.get("vp_forward_bias_m"), item.get("vp_lateral_bias_m"), item.get("vp_vertical_bias_m"),
            ],
            "guidance_command": item.get("guidance_command"),
            "nz_cmd": item.get("nz_cmd"),
            "actual_nz_g": item.get("actual_nz_g"),
            "target_attack_score": 1.0 if bool(item["target_in_attack_zone"]) else 0.0,
            "target_in_attack_zone": bool(item["target_in_attack_zone"]),
            "ego_in_attack_zone": bool(item["ego_in_attack_zone"]),
            "range_m": float(values["range_m"]),
            "range_rate_mps": float(values["range_rate_mps"]),
            "aa_deg": aa_deg,
            "ata_deg": ata_deg,
            "own_specific_energy_m": float(values["own_altitude"] + values["own_speed"] ** 2 / (2.0 * 9.80665)),
            "altitude_m": float(values["own_altitude"]),
            "backend": str(item["backend"]),
            "backend_fallback_occurred": bool(item["backend_fallback_occurred"]),
            "checkpoint_fallback": False,
            "terminal_reason": str(item["terminal_reason"]),
        }
        step["continuity_prev_hash"] = previous_hash
        step["continuity_hash"] = _stable_hash(step)
        previous_hash = step["continuity_hash"]
        steps.append(step)
    return steps


def _record(
    *, source_id: str, opponent: str, scenario: Mapping[str, Any], result: Mapping[str, Any], ledger: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    steps = _normalized_steps(ledger, config)
    window = find_first_pressure_window(steps, config["gates"])
    metadata = scenario["metadata"]
    return {
        "source_id": source_id,
        "opponent": opponent,
        "scenario_name": scenario["name"],
        "scenario_signature": metadata["mirror_pair_id"],
        "mirror_sign": metadata["mirror_sign"],
        "initial_state": metadata["initial_class"],
        "phase_trace": steps,
        "handoff_window": [] if window is None else window,
        "result": {
            "strict_jsbsim": all(step["backend"] == "jsbsim" for step in steps),
            "handoff_reached": window is not None,
            "handoff_step": None if window is None else window[0]["step"],
            "window_complete": window is not None and len(window) == int(config["gates"]["required_window_steps"]),
            "no_reset_after_initialization": bool(result["no_reset_after_initialization"]),
            "no_backend_fallback": bool(result["no_backend_fallback"]),
            "no_prediction_fallback": window is not None and not any(step["prediction_fallback"] for step in window),
            "no_history_padding": bool(result["no_history_padding"]),
            "no_checkpoint_fallback": bool(result["no_checkpoint_fallback"]),
            "continuity_hash_chain_complete": bool(steps),
            "terminal_reason": str(result["terminal_reason"]),
        },
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_canonical(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _authorize_execution(config: Mapping[str, Any], expected_records: int) -> None:
    execution = config.get("execution")
    if not isinstance(execution, Mapping) or int(execution.get("expected_records", -1)) != expected_records:
        raise PressureCollectorError("execution record universe is not authorized")
    required_sha = str(execution.get("required_implementation_git_sha") or "")
    if len(required_sha) != 40 or subprocess.run(
        ["git", "merge-base", "--is-ancestor", required_sha, "HEAD"], cwd=ROOT, check=False
    ).returncode:
        raise PressureCollectorError("authorized implementation SHA is not an ancestor of HEAD")
    files = execution.get("authorized_code_files")
    if not isinstance(files, list) or not files:
        raise PressureCollectorError("authorized source file list is missing")
    for entry in files:
        if not isinstance(entry, Mapping):
            raise PressureCollectorError("authorized source entry is malformed")
        file_path = _resolve(str(entry.get("path") or ""))
        if not file_path.is_file() or _hash(file_path) != str(entry.get("sha256") or ""):
            raise PressureCollectorError(f"authorized source hash mismatch: {file_path}")
    smoke_path = _resolve(str(execution.get("required_smoke_gate") or ""))
    if not smoke_path.is_file() or _hash(smoke_path) != str(execution.get("required_smoke_gate_sha256") or ""):
        raise PressureCollectorError("required engineering smoke artifact hash mismatch")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if (
        smoke.get("smoke_id") != "ENGINEERING-PRESSURE-EXPOSED-POSTMERGE-COLLECTOR-NOT-A-SOURCE-ID"
        or smoke.get("source_id_under_test") != SOURCE_ID
        or smoke.get("passes") is not True
        or int(smoke.get("records", 0)) != len(OPPONENTS)
    ):
        raise PressureCollectorError("required engineering smoke did not pass")


def _failed_record(source_id: str, opponent: str, scenario: Mapping[str, Any], error: Exception) -> dict[str, Any]:
    metadata = scenario["metadata"]
    return {
        "source_id": source_id,
        "opponent": opponent,
        "scenario_name": scenario["name"],
        "scenario_signature": metadata["mirror_pair_id"],
        "mirror_sign": metadata["mirror_sign"],
        "initial_state": metadata["initial_class"],
        "phase_trace": [],
        "handoff_window": [],
        "result": {
            "strict_jsbsim": False,
            "handoff_reached": False,
            "window_complete": False,
            "no_reset_after_initialization": False,
            "no_backend_fallback": False,
            "no_prediction_fallback": False,
            "no_history_padding": False,
            "no_checkpoint_fallback": False,
            "continuity_hash_chain_complete": False,
            "terminal_reason": "collector_runtime_error",
            "error": repr(error),
        },
    }


def engineering_smoke(path: Path) -> dict[str, Any]:
    """Run one non-source full trajectory per opponent to validate telemetry."""

    validate_design(path, check_external_inputs=True)
    config = load_config(path)
    manifest = _load_yaml(_resolve(str(config["inputs"]["scenario_manifest"])))
    registry = _load_yaml(_resolve(str(config["inputs"]["runtime_registry"])))
    scenario = manifest["scenarios"][0]
    results = []
    for opponent in OPPONENTS:
        env = _build_env(config, registry, opponent)
        try:
            encoder = FrozenP3Encoder(
                _resolve(str(config["inputs"]["frozen_p3_encoder"])),
                str(config["inputs"]["frozen_p3_encoder_sha256"]),
            )
            episode = raw_v5.FiveStateRawSIV5Episode(env, _specialist(registry), encoder, config)
            result, ledger = episode.run(scenario, int(scenario["metadata"]["scenario_seed"]))
            normalized = _normalized_steps(ledger, config)
            results.append({
                "opponent": opponent,
                "passes": bool(normalized) and all(
                    step["backend"] == "jsbsim" and len(step["base_observation_16d"]) == 16
                    and (step["observation_66d"] is None or len(step["observation_66d"]) == 66)
                    for step in normalized
                ) and bool(result["no_backend_fallback"]),
                "steps": len(normalized),
                "prediction_valid_steps": sum(bool(step["prediction_valid"]) for step in normalized),
            })
        except Exception as error:
            results.append({"opponent": opponent, "passes": False, "error": repr(error)})
        finally:
            env.close()
    return {
        "smoke_id": "ENGINEERING-PRESSURE-EXPOSED-POSTMERGE-COLLECTOR-NOT-A-SOURCE-ID",
        "source_id_under_test": SOURCE_ID,
        "records": len(results),
        "passes": all(row["passes"] for row in results),
        "per_opponent": results,
    }


def run(path: Path) -> dict[str, Any]:
    design = validate_design(path, check_external_inputs=True)
    config = load_config(path)
    if config["authorization"].get("execution_permitted") is not True:
        raise PressureCollectorError("formal collection is not authorized by this config")
    if config["authorization"].get("require_clean_worktree") is True and subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise PressureCollectorError("formal collection requires a clean worktree")
    _authorize_execution(config, int(design["planned_records"]))
    output = _resolve(str(config["outputs"]["root"]))
    if output.exists():
        raise PressureCollectorError("output root must be absent")
    manifest = _load_yaml(_resolve(str(config["inputs"]["scenario_manifest"])))
    registry = _load_yaml(_resolve(str(config["inputs"]["runtime_registry"])))
    records = []
    output.mkdir(parents=True)
    for opponent in OPPONENTS:
        for scenario in manifest["scenarios"]:
            env = None
            try:
                env = _build_env(config, registry, opponent)
                encoder = FrozenP3Encoder(_resolve(str(config["inputs"]["frozen_p3_encoder"])), str(config["inputs"]["frozen_p3_encoder_sha256"]))
                episode = raw_v5.FiveStateRawSIV5Episode(env, _specialist(registry), encoder, config)
                result, ledger = episode.run(scenario, int(scenario["metadata"]["scenario_seed"]))
                record = _record(source_id=SOURCE_ID, opponent=opponent, scenario=scenario, result=result, ledger=ledger, config=config)
            except Exception as error:
                record = _failed_record(SOURCE_ID, opponent, scenario, error)
            finally:
                if env is not None:
                    env.close()
            records.append(record)
            _write_json(output / "episodes" / opponent / f"{scenario['name']}.json", {"record": record})
    summary = summarize_records(records, config["gates"])
    index = []
    for record in records:
        file_path = output / "episodes" / record["opponent"] / f"{record['scenario_name']}.json"
        index.append({"path": str(file_path.relative_to(output)), "sha256": _hash(file_path)})
    _write_json(output / "records_index.json", {"source_id": SOURCE_ID, "records": index})
    _write_json(output / "pressure_gate.json", {"source_id": SOURCE_ID, **summary})
    _write_json(output / "run_manifest.json", {
        "source_id": SOURCE_ID,
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "config_sha256": _hash(path),
        "scenario_manifest_sha256": _hash(_resolve(str(config["inputs"]["scenario_manifest"]))),
        "records": len(records),
        "training": "prohibited",
        "method_comparison": "prohibited_until_pressure_gate_passes",
        "design": design,
    })
    return {"output": str(output), **summary}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-output", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.smoke and args.execute:
        parser.error("--smoke and --execute are mutually exclusive")
    if args.smoke_output is not None and not args.smoke:
        parser.error("--smoke-output requires --smoke")
    if args.smoke:
        smoke = engineering_smoke(args.config)
        if args.smoke_output is not None:
            if args.smoke_output.exists():
                parser.error("refusing to overwrite existing smoke artifact")
            _write_json(args.smoke_output, smoke)
        print(json.dumps(smoke, ensure_ascii=False, indent=2))
    elif args.execute:
        print(json.dumps(run(args.config), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(validate_design(args.config), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
