#!/usr/bin/env python3
"""Collect new continuous tactical-basis baseline trajectories without learning."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.evaluation import thesis_five_state_raw_si_v5_record_only as raw_v5
from uav_vpp_guidance.evaluation.tactical_basis_continuity_contract import (
    OPPONENTS,
    record_passes,
    stable_hash,
    summarize_records,
)
from uav_vpp_guidance.hierarchy.specialist_policy import FrozenSpecialistPolicy
from uav_vpp_guidance.training import thesis_neutral_postmerge_reentry_recovery_pilot as engine
from uav_vpp_guidance.training.thesis_shared_skill_geometry import FrozenP3Encoder


SOURCE_IDS = {
    "THESIS-DISADVANTAGE-TACTICAL-BASIS-CONTINUITY-COLLECTOR-V1",
    "THESIS-DISADVANTAGE-TACTICAL-BASIS-CONTINUITY-COLLECTOR-V2",
}


class TacticalBasisContinuityCollectorError(RuntimeError):
    """Raised when the collector leaves its frozen non-learning contract."""


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise TacticalBasisContinuityCollectorError(f"YAML mapping required: {path}")
    return value


def _merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_config(path: Path) -> dict[str, Any]:
    overlay = _load(path)
    base_value = overlay.pop("base_config", None)
    if not base_value:
        return overlay
    base = _load(_resolve(str(base_value)))
    replace_sections = overlay.pop("replace_sections", [])
    if not isinstance(replace_sections, list) or any(not isinstance(item, str) for item in replace_sections):
        raise TacticalBasisContinuityCollectorError("replace_sections must be a list of mapping names")
    result = _merge(base, overlay)
    for section in replace_sections:
        if section not in overlay or not isinstance(overlay[section], Mapping):
            raise TacticalBasisContinuityCollectorError(f"replacement section is missing: {section}")
        result[section] = copy.deepcopy(dict(overlay[section]))
    return result


def _source_id(config: Mapping[str, Any]) -> str:
    source_id = str(config.get("source_id") or "")
    if source_id not in SOURCE_IDS:
        raise TacticalBasisContinuityCollectorError("source identity drifted")
    return source_id


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _vector_or_none(value: Any, size: int) -> list[float] | None:
    try:
        vector = np.asarray(value, dtype=np.float64).reshape(size)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(vector).all():
        return None
    return [float(item) for item in vector]


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _require_input(config: Mapping[str, Any], key: str) -> Path:
    inputs = config["inputs"]
    path = _resolve(str(inputs[key]))
    expected = str(inputs[key + "_sha256"])
    if not path.is_file() or _hash(path) != expected:
        raise TacticalBasisContinuityCollectorError(f"frozen input SHA mismatch: {key}")
    return path


def _validate_config(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = _load_config(path)
    _source_id(config)
    if config.get("stage") != "continuous_baseline_data_contract":
        raise TacticalBasisContinuityCollectorError("collector stage drifted")
    authorization = config.get("authorization")
    if not isinstance(authorization, Mapping):
        raise TacticalBasisContinuityCollectorError("authorization mapping is required")
    for key in (
        "training_permitted",
        "tuning_permitted",
        "reward_change_permitted",
        "candidate_action_execution_permitted",
        "high_level_ppo_training_permitted",
        "shared_skill_training_permitted",
        "guidance_pid_change_permitted",
        "snapshot_restore_permitted",
        "future_state_injection_permitted",
        "history_padding_permitted",
    ):
        if authorization.get(key) is not False:
            raise TacticalBasisContinuityCollectorError(f"collector must prohibit {key}")
    if tuple(config.get("opponents", {}).get("order", ())) != OPPONENTS:
        raise TacticalBasisContinuityCollectorError("opponent registry order drifted")
    if config["opponents"].get("pooled_gate") != "prohibited":
        raise TacticalBasisContinuityCollectorError("pooled gate must remain prohibited")
    contract = config.get("contract", {})
    if (
        contract.get("backend") != "jsbsim"
        or contract.get("strict_backend") is not True
        or contract.get("action_semantics") != "tactical_basis_v1"
        or contract.get("reference_controller") != "frozen_fixed_head_on"
        or contract.get("routing_enabled") is not False
        or contract.get("reset_at_handoff") is not False
    ):
        raise TacticalBasisContinuityCollectorError("continuous baseline contract drifted")
    if int(contract.get("reference_window_steps", -1)) != int(config["gates"]["required_window_steps"]):
        raise TacticalBasisContinuityCollectorError("window length contract drifted")
    if int(contract.get("history_window_steps", -1)) != 10:
        raise TacticalBasisContinuityCollectorError("history contract drifted")
    try:
        raw_v5.RawSIFirstPassConfig(**dict(config["observability"]["raw_si_first_pass_event"]))
    except (TypeError, ValueError) as error:
        raise TacticalBasisContinuityCollectorError("raw-SI first-pass event config is invalid") from error
    if config.get("runtime", {}).get("virtual_point", {}).get("action_semantics") != "tactical_basis_v1":
        raise TacticalBasisContinuityCollectorError("runtime does not deploy tactical_basis_v1")

    manifest_path = _require_input(config, "scenario_manifest")
    registry_path = _require_input(config, "runtime_registry")
    p3_path = _require_input(config, "frozen_p3_encoder")
    specialist_path = _require_input(config, "frozen_head_on_specialist")
    manifest = _load(manifest_path)
    registry = _load(registry_path)
    if (
        manifest.get("source_id") != config["inputs"]["scenario_manifest_source_id"]
        or len(manifest.get("scenarios", [])) != 12
    ):
        raise TacticalBasisContinuityCollectorError("collector manifest contract drifted")
    specialist = registry.get("specialists", {}).get("frozen_fixed_head_on", {})
    if Path(str(specialist.get("checkpoint", ""))) != specialist_path:
        raise TacticalBasisContinuityCollectorError("registry fixed-head-on checkpoint drifted")
    if str(specialist.get("checkpoint_sha256", "")) != str(config["inputs"]["frozen_head_on_specialist_sha256"]):
        raise TacticalBasisContinuityCollectorError("registry fixed-head-on SHA drifted")
    output = _resolve(str(config["outputs"]["root"]))
    if not output.is_absolute() or output.exists():
        raise TacticalBasisContinuityCollectorError("output root must be absolute and absent")
    return config, manifest, registry


def validate_design(path: Path) -> dict[str, Any]:
    config, manifest, _registry = _validate_config(path)
    return {
        "source_id": _source_id(config),
        "stage": config["stage"],
        "planned_records": len(manifest["scenarios"]) * len(OPPONENTS),
        "reference_window_steps": int(config["contract"]["reference_window_steps"]),
        "candidate_actions_enter_env": False,
        "training_permitted": False,
        "execution_permitted": bool(config["authorization"]["execution_permitted"]),
        "output_root_absent": True,
    }


def _build_env(config: Mapping[str, Any], registry: Mapping[str, Any], opponent: str) -> CloseRangeTrackingEnv:
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


class ContinuousTacticalBasisEpisode(raw_v5.FiveStateRawSIV5Episode):
    """One legal reset and one fixed-head-on trajectory, without a handoff reset."""

    def __init__(self, *args: Any, source_id: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.source_id = source_id

    def _step_payload(
        self,
        *,
        pre_base: np.ndarray,
        pre_history: tuple[np.ndarray, ...],
        pre_phase: str,
        pre_state: str,
        reference_66d: np.ndarray | None,
        action: np.ndarray,
        info: Mapping[str, Any],
        previous_hash: str,
    ) -> dict[str, Any]:
        own_state = info.get("own_state", {}) if isinstance(info.get("own_state"), Mapping) else {}
        basis_vectors = [
            _vector_or_none(info.get(key), 3)
            for key in (
                "tactical_basis_ll_world",
                "tactical_basis_io_world",
                "tactical_basis_cd_world",
            )
        ]
        if all(vector is not None for vector in basis_vectors):
            basis_matrix = np.column_stack(basis_vectors)
            basis_rank = int(np.linalg.matrix_rank(basis_matrix))
            basis_condition = _finite_or_none(np.linalg.cond(basis_matrix))
        else:
            basis_rank = 0
            basis_condition = None
        payload = {
            "step": self.total_steps,
            "phase": pre_phase,
            "dynamic_state": pre_state,
            "history_length": len(pre_history),
            "history_padding_used": False,
            "base_observation_16d": [float(value) for value in pre_base],
            "observation_66d": None if reference_66d is None else [float(value) for value in reference_66d],
            "reference_action": [float(value) for value in action],
            "active_action_semantics": str(info.get("action_semantics", "")),
            "configured_action_semantics": str(info.get("configured_action_semantics", "")),
            "tactical_basis_enabled": bool(info.get("tactical_basis_enabled", False)),
            "tactical_basis_ll_world": basis_vectors[0],
            "tactical_basis_io_world": basis_vectors[1],
            "tactical_basis_cd_world": basis_vectors[2],
            "basis_rank": basis_rank,
            "basis_condition_number": basis_condition,
            "prediction_valid": bool(info.get("prediction_valid", False)),
            "prediction_fallback": bool(info.get("prediction_fallback", False)),
            "vpp_world_offset_m": _vector_or_none(info.get("vp_world_offset"), 3),
            "guidance_command": _json_safe(info.get("guidance_command")),
            "aoa_rad": _finite_or_none(own_state.get("alpha_rad", own_state.get("aoa_rad", np.nan))),
            "nz_cmd": _json_safe(info.get("nz_cmd")),
            "actual_nz_g": _finite_or_none(own_state.get("nz_g", info.get("nz_g", np.nan))),
            "nz_saturated": bool(info.get("nz_saturated", False)),
            "roll_rate_saturated": bool(info.get("roll_rate_saturated", False)),
            "throttle_saturated": bool(info.get("throttle_saturated", False)),
            "backend": str(info.get("backend", "")),
            "backend_fallback_occurred": bool(info.get("backend_fallback_occurred", False)),
            "checkpoint_fallback": False,
            "terminal_reason": str(info.get("reason") or "timeout"),
        }
        payload["continuity_prev_hash"] = previous_hash
        payload["continuity_hash"] = stable_hash(payload)
        return payload

    @staticmethod
    def _trace_step(payload: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: payload[key]
            for key in (
                "step",
                "phase",
                "dynamic_state",
                "reference_action",
                "active_action_semantics",
                "configured_action_semantics",
                "tactical_basis_enabled",
                "basis_rank",
                "basis_condition_number",
                "prediction_valid",
                "prediction_fallback",
                "backend",
                "backend_fallback_occurred",
                "continuity_prev_hash",
                "continuity_hash",
            )
        }

    def run(self, scenario: Mapping[str, Any], seed: int) -> dict[str, Any]:
        self.history.clear()
        self.history_features.reset()
        self.phase_tracker = raw_v5.PhaseTracker()
        self.total_steps = 0
        self.last_info = {}
        observation = self.env.reset(scenario=copy.deepcopy(dict(scenario)), seed=int(seed))
        if self.env._backend != "jsbsim":
            raise TacticalBasisContinuityCollectorError("strict JSBSim backend was not selected")
        self.env.set_runtime_specialist_context(
            specialist_key="head_on",
            specialist_profile=None,
            specialist_mode_name="tactical_basis_continuity_baseline",
            specialist_reason="fixed_head_on_continuous_data_contract",
        )
        self.observation_schema = copy.deepcopy(observation["observation_schema"])
        self._observe(observation)
        metadata = scenario["metadata"]
        previous_hash = stable_hash(
            {
                "scenario": scenario["name"],
                "seed": int(seed),
                "initial_observation": [float(value) for value in self.observation["observation_vector"]],
            }
        )
        trace: list[dict[str, Any]] = []
        handoff_window: list[dict[str, Any]] = []
        handoff: dict[str, Any] | None = None
        backend_fallback_seen = False
        prediction_fallback_seen = False
        terminal_reason = "collector_horizon"

        while self.total_steps < self.env.max_steps:
            if self.observation is None or self.base is None:
                raise TacticalBasisContinuityCollectorError("continuous observation was lost")
            pre_base = self.base.copy()
            pre_history = tuple(item.copy() for item in self.history)
            pre_phase, pre_state = self.phase, self.dynamic_state
            reference_66d = self._reference_observation()
            action = raw_v5._finite_action(
                self.fixed_head_on.get_deterministic_action(self.observation["observation_vector"])
            )
            next_observation, _ignored_reward, terminated, truncated, info = self.env.step(action)
            self.total_steps += 1
            self.last_info = info
            self._observe(next_observation)
            payload = self._step_payload(
                pre_base=pre_base,
                pre_history=pre_history,
                pre_phase=pre_phase,
                pre_state=pre_state,
                reference_66d=reference_66d,
                action=action,
                info=info,
                previous_hash=previous_hash,
            )
            previous_hash = str(payload["continuity_hash"])
            trace.append(self._trace_step(payload))
            backend_fallback_seen |= bool(payload["backend_fallback_occurred"])
            prediction_fallback_seen |= bool(payload["prediction_fallback"])
            eligible = bool(
                handoff is None
                and pre_state == self.config["contract"]["target_dynamic_state"]
                and pre_phase in self.config["contract"]["target_phases"]
                and reference_66d is not None
                and not backend_fallback_seen
                and not prediction_fallback_seen
            )
            if eligible:
                handoff = {
                    "step": self.total_steps,
                    "phase": pre_phase,
                    "dynamic_state": pre_state,
                    "state_hash": stable_hash(
                        {
                            "base_observation_16d": payload["base_observation_16d"],
                            "observation_66d": payload["observation_66d"],
                            "continuity_prev_hash": payload["continuity_prev_hash"],
                        }
                    ),
                }
            if handoff is not None:
                handoff_window.append(payload)
            terminal_reason = str(payload["terminal_reason"])
            if backend_fallback_seen or prediction_fallback_seen:
                break
            if len(handoff_window) >= int(self.config["contract"]["reference_window_steps"]):
                terminal_reason = "collector_window_complete"
                break
            if terminated or truncated:
                break

        result = {
            "strict_jsbsim": True,
            "handoff_reached": handoff is not None,
            "handoff_step": None if handoff is None else handoff["step"],
            "handoff_state_hash": None if handoff is None else handoff["state_hash"],
            "window_steps": len(handoff_window),
            "window_complete": len(handoff_window) == int(self.config["contract"]["reference_window_steps"]),
            "no_reset_after_initialization": True,
            "no_backend_fallback": not backend_fallback_seen,
            "no_prediction_fallback": not prediction_fallback_seen,
            "no_history_padding": True,
            "no_checkpoint_fallback": True,
            "continuity_hash_chain_complete": bool(trace),
            "terminal_reason": terminal_reason,
        }
        return {
            "source_id": self.source_id,
            "opponent": str(self.env._opponent_stage),
            "scenario_name": str(scenario["name"]),
            "scenario_signature": str(metadata["mirror_pair_id"]),
            "mirror_sign": str(metadata["mirror_sign"]),
            "initial_state": str(metadata["initial_class"]),
            "initial_continuity_hash": trace[0]["continuity_prev_hash"] if trace else previous_hash,
            "continuity_trace": trace,
            "handoff_window": handoff_window,
            "result": result,
        }


def _specialist(registry: Mapping[str, Any]) -> FrozenSpecialistPolicy:
    entry = registry["specialists"]["frozen_fixed_head_on"]
    return FrozenSpecialistPolicy(
        checkpoint_path=str(entry["checkpoint"]),
        config_path=str(_resolve(str(entry["config_path"]))),
        device="cpu",
    )


def _failed_record(source_id: str, opponent: str, scenario: Mapping[str, Any], error: Exception) -> dict[str, Any]:
    metadata = scenario["metadata"]
    return {
        "source_id": source_id,
        "opponent": opponent,
        "scenario_name": str(scenario["name"]),
        "scenario_signature": str(metadata["mirror_pair_id"]),
        "mirror_sign": str(metadata["mirror_sign"]),
        "initial_state": str(metadata["initial_class"]),
        "continuity_trace": [],
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


def _authorize_execution(config: Mapping[str, Any], expected_records: int) -> None:
    execution = config.get("execution")
    if not isinstance(execution, Mapping) or int(execution.get("expected_records", -1)) != expected_records:
        raise TacticalBasisContinuityCollectorError("execution record universe is not authorized")
    required_sha = str(execution.get("required_implementation_git_sha") or "")
    if len(required_sha) != 40 or subprocess.run(
        ["git", "merge-base", "--is-ancestor", required_sha, "HEAD"], cwd=ROOT, check=False
    ).returncode:
        raise TacticalBasisContinuityCollectorError("authorized implementation SHA is not an ancestor of HEAD")
    files = execution.get("authorized_code_files")
    if not isinstance(files, list) or not files:
        raise TacticalBasisContinuityCollectorError("authorized source file list is missing")
    for item in files:
        if not isinstance(item, Mapping):
            raise TacticalBasisContinuityCollectorError("authorized source file entry is malformed")
        source = _resolve(str(item.get("path") or ""))
        expected_hash = str(item.get("sha256") or "")
        if not source.is_file() or _hash(source) != expected_hash:
            raise TacticalBasisContinuityCollectorError(f"authorized source file hash mismatch: {source}")


def run(path: Path) -> dict[str, Any]:
    config, manifest, registry = _validate_config(path)
    source_id = _source_id(config)
    if config["authorization"].get("execution_permitted") is not True:
        raise TacticalBasisContinuityCollectorError("collector execution is not authorized")
    if config["authorization"].get("require_clean_worktree") is True and subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise TacticalBasisContinuityCollectorError("collector requires a clean worktree")
    _authorize_execution(config, len(manifest["scenarios"]) * len(OPPONENTS))
    output = _resolve(str(config["outputs"]["root"]))
    output.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    try:
        for opponent in OPPONENTS:
            for scenario in manifest["scenarios"]:
                try:
                    env = _build_env(config, registry, opponent)
                    encoder = FrozenP3Encoder(
                        _resolve(str(config["inputs"]["frozen_p3_encoder"])),
                        str(config["inputs"]["frozen_p3_encoder_sha256"]),
                    )
                    episode = ContinuousTacticalBasisEpisode(
                        env, _specialist(registry), encoder, config, source_id=source_id
                    )
                    try:
                        record = episode.run(scenario, int(scenario["metadata"]["scenario_seed"]))
                    finally:
                        episode.close()
                except Exception as error:  # Preserve every failed attempt as evidence.
                    record = _failed_record(source_id, opponent, scenario, error)
                records.append(record)
                filename = str(scenario["name"]).replace("/", "_").replace("\\", "_") + ".json"
                _write_json(output / "episodes" / opponent / filename, {"record": record})
        index = []
        for record in records:
            path_value = output / "episodes" / str(record["opponent"]) / (
                str(record["scenario_name"]).replace("/", "_").replace("\\", "_") + ".json"
            )
            index.append({"path": str(path_value.relative_to(output)), "sha256": _hash(path_value)})
        summary = summarize_records(records, config["gates"])
        _write_json(output / "records_index.json", {"source_id": source_id, "records": index})
        _write_json(output / "continuity_gate.json", {"source_id": source_id, **summary})
        _write_json(
            output / "run_manifest.json",
            {
                "source_id": source_id,
                "stage": config["stage"],
                "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "config_sha256": _hash(path),
                "scenario_manifest_sha256": _hash(_resolve(str(config["inputs"]["scenario_manifest"]))),
                "runtime_registry_sha256": _hash(_resolve(str(config["inputs"]["runtime_registry"]))),
                "records": len(records),
                "training": "prohibited",
                "candidate_actions_enter_env": False,
                "reward_or_win_rate_analysis": "prohibited",
            },
        )
        return {"output": str(output), **summary}
    except Exception as error:
        _write_json(output / "failure.json", {"source_id": source_id, "error": repr(error), "completed_records": len(records)})
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(validate_design(args.config), ensure_ascii=False, indent=2))
        return 0
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
