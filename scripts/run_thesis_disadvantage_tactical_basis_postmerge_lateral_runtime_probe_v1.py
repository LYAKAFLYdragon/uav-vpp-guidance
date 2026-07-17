#!/usr/bin/env python3
"""Run the one-off paired post-merge lateral-extension runtime probe."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

# Match the existing Windows JSBSim/PyTorch runtime compatibility setting.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import run_thesis_disadvantage_tactical_basis_continuity_collector_v1 as continuity
from uav_vpp_guidance.evaluation import thesis_five_state_raw_si_v5_record_only as raw_v5
from uav_vpp_guidance.evaluation.tactical_basis_lateral_runtime_probe_contract import (
    METHODS,
    OPPONENTS,
    P2_SELECTED_VARIANT,
    SOURCE_ID,
    TacticalBasisLateralRuntimeProbeDesignError,
    is_ego_failure,
    summarize_runtime_records,
    validate_design_spec,
)
from uav_vpp_guidance.evaluation.temporal_reference_compiler import (
    TemporalReferenceConfig,
    TemporalReferenceTemplate,
    compile_temporal_reference_step,
)


class PostmergeLateralRuntimeProbeError(RuntimeError):
    """Raised when the runtime probe leaves its frozen causal contract."""


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
        raise PostmergeLateralRuntimeProbeError(f"YAML mapping required: {path}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PostmergeLateralRuntimeProbeError(f"JSON mapping required: {path}")
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _require_input(config: Mapping[str, Any], key: str) -> Path:
    inputs = config["inputs"]
    path = _resolve(str(inputs[key]))
    if not path.is_file() or _hash(path) != str(inputs[key + "_sha256"]):
        raise PostmergeLateralRuntimeProbeError(f"frozen input SHA mismatch: {key}")
    return path


def _load_inputs(config_path: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    if config.get("source_id") != SOURCE_ID or config.get("stage") != "paired_nonlearning_postmerge_runtime_probe":
        raise PostmergeLateralRuntimeProbeError("runtime probe identity drifted")
    authorization = config.get("authorization")
    if not isinstance(authorization, Mapping):
        raise PostmergeLateralRuntimeProbeError("authorization mapping is required")
    for key in (
        "training_permitted",
        "tuning_permitted",
        "reward_change_permitted",
        "high_level_ppo_training_permitted",
        "shared_skill_training_permitted",
        "guidance_pid_change_permitted",
        "action_envelope_expansion_permitted",
        "snapshot_restore_permitted",
        "future_state_injection_permitted",
        "history_padding_permitted",
    ):
        if authorization.get(key) is not False:
            raise PostmergeLateralRuntimeProbeError(f"runtime probe must prohibit {key}")

    p2_summary = _load_json(_require_input(config, "p2_summary"))
    p2_verification = _load_json(_require_input(config, "p2_verification"))
    _require_input(config, "p2_config")
    continuity_gate = _load_json(_require_input(config, "continuity_gate"))
    probe_manifest = _load_yaml(_require_input(config, "probe_manifest"))
    parent_manifest = _load_yaml(_require_input(config, "parent_manifest"))
    runtime_config_path = _require_input(config, "runtime_config")
    registry_path = _require_input(config, "runtime_registry")
    _require_input(config, "frozen_p3_encoder")
    _require_input(config, "frozen_head_on_specialist")
    design_path = _require_input(config, "temporal_reference_design")
    _require_input(config, "temporal_reference_compiler")
    if continuity_gate.get("source_id") != "THESIS-DISADVANTAGE-TACTICAL-BASIS-CONTINUITY-COLLECTOR-V3" or continuity_gate.get("passes") is not True:
        raise PostmergeLateralRuntimeProbeError("V3 continuity gate is not valid")

    design_config = copy.deepcopy(config)
    design_config["authorization"]["execution_permitted"] = False
    try:
        validate_design_spec(design_config, p2_summary, p2_verification, probe_manifest, parent_manifest)
    except TacticalBasisLateralRuntimeProbeDesignError as error:
        raise PostmergeLateralRuntimeProbeError(str(error)) from error

    runtime_config = continuity._load_config(runtime_config_path)
    registry = _load_yaml(registry_path)
    if runtime_config.get("source_id") != "THESIS-DISADVANTAGE-TACTICAL-BASIS-CONTINUITY-COLLECTOR-V3":
        raise PostmergeLateralRuntimeProbeError("runtime config must be the frozen V3 chain")
    if registry.get("specialists", {}).get("frozen_fixed_head_on", {}).get("checkpoint_sha256") != config["inputs"]["frozen_head_on_specialist_sha256"]:
        raise PostmergeLateralRuntimeProbeError("runtime registry head-on specialist drifted")

    reference_design = _load_yaml(design_path).get("temporal_reference")
    if not isinstance(reference_design, Mapping):
        raise PostmergeLateralRuntimeProbeError("P1 temporal-reference design is malformed")
    candidate = config["candidate"]
    template_mapping = reference_design.get("templates", {}).get(candidate["template"])
    if not isinstance(template_mapping, Mapping):
        raise PostmergeLateralRuntimeProbeError("selected P1 template is unavailable")
    template = TemporalReferenceTemplate.from_mapping(str(candidate["template"]), template_mapping)
    compiler_config = TemporalReferenceConfig(
        horizon_steps=int(reference_design["horizon_steps"]),
        high_level_dt_s=float(reference_design["high_level_dt_s"]),
        max_action_delta=float(reference_design["max_action_delta"]),
        max_action_slew=float(reference_design["max_action_slew"]),
        max_basis_condition_number=float(reference_design["max_basis_condition_number"]),
    )
    if (
        int(candidate["program_steps"]) != compiler_config.horizon_steps
        or float(candidate["high_level_dt_s"]) != compiler_config.high_level_dt_s
        or float(candidate["max_action_delta"]) != compiler_config.max_action_delta
        or float(candidate["max_action_slew"]) != compiler_config.max_action_slew
        or float(candidate["max_basis_condition_number"]) != compiler_config.max_basis_condition_number
        or str(candidate["p2_selected_variant"]) != P2_SELECTED_VARIANT
    ):
        raise PostmergeLateralRuntimeProbeError("candidate compiler envelope differs from P1/P2")

    requested = {str(item["name"]): int(item["scenario_seed"]) for item in probe_manifest["scenarios"]}
    parent_by_name = {str(item["name"]): item for item in parent_manifest["scenarios"]}
    scenarios: list[dict[str, Any]] = []
    for name, seed in requested.items():
        scenario = parent_by_name.get(name)
        if not isinstance(scenario, Mapping) or int(scenario["metadata"]["scenario_seed"]) != seed:
            raise PostmergeLateralRuntimeProbeError("probe manifest scenario drifted from parent")
        scenarios.append(copy.deepcopy(dict(scenario)))
    if len(scenarios) != 12:
        raise PostmergeLateralRuntimeProbeError("probe requires exactly twelve scenarios")

    episode_config = copy.deepcopy(runtime_config)
    episode_config["contract"] = copy.deepcopy(config["contract"])
    episode_config["candidate"] = copy.deepcopy(candidate)
    return {
        "config": config,
        "runtime_config": episode_config,
        "registry": registry,
        "scenarios": scenarios,
        "template": template,
        "compiler_config": compiler_config,
    }


def validate_design(config_path: Path) -> dict[str, Any]:
    values = _load_inputs(config_path)
    config = values["config"]
    output = _resolve(str(config["outputs"]["root"]))
    if not output.is_absolute() or output.exists():
        raise PostmergeLateralRuntimeProbeError("formal output root must be absolute and absent")
    return {
        "source_id": SOURCE_ID,
        "mode": "runtime_probe_preflight",
        "candidate_variant": P2_SELECTED_VARIANT,
        "planned_records": len(METHODS) * len(OPPONENTS) * len(values["scenarios"]),
        "planned_pairs": len(OPPONENTS) * len(values["scenarios"]),
        "execution_permitted": bool(config["authorization"]["execution_permitted"]),
        "output_root_absent": True,
    }


def _finite_metric(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise PostmergeLateralRuntimeProbeError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise PostmergeLateralRuntimeProbeError(f"{name} must be finite")
    return result


class PairedPostmergeEpisode(continuity.ContinuousTacticalBasisEpisode):
    """Continuous legal run-in with the selected candidate as the only action change."""

    def __init__(
        self,
        *args: Any,
        method: str,
        template: TemporalReferenceTemplate,
        compiler_config: TemporalReferenceConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if method not in METHODS:
            raise PostmergeLateralRuntimeProbeError("unknown paired runtime method")
        self.method = method
        self.template = template
        self.compiler_config = compiler_config

    def _current_basis(self) -> list[list[float]]:
        generator = getattr(self.env, "virtual_point_generator", None)
        if generator is None or str(getattr(generator, "action_semantics", "")) != "tactical_basis_v1":
            raise PostmergeLateralRuntimeProbeError("runtime does not expose tactical_basis_v1")
        own, target = self.env._get_current_states(noisy=True)
        target = self.env._task_get_target_state(target, own)
        own, target = self.env._task_pre_step(own, target)
        vectors = generator._build_tactical_basis_vectors(own, target)[:3]
        basis = [np.asarray(vector, dtype=np.float64).reshape(3).tolist() for vector in vectors]
        if not np.isfinite(np.asarray(basis, dtype=np.float64)).all():
            raise PostmergeLateralRuntimeProbeError("live tactical basis is non-finite")
        return basis

    def _after_metrics(self, info: Mapping[str, Any]) -> dict[str, float]:
        if self.base is None:
            raise PostmergeLateralRuntimeProbeError("post-step base observation is unavailable")
        base = np.asarray(self.base, dtype=np.float64)
        if base.shape != (16,) or not np.isfinite(base).all():
            raise PostmergeLateralRuntimeProbeError("post-step base observation is invalid")
        attack = info.get("attack_zone_info")
        source = attack if isinstance(attack, Mapping) else info
        own_speed = float(base[12]) * 400.0
        own_altitude = float(base[14]) * 10000.0
        return {
            "range_m": _finite_metric(float(base[0]) * 5000.0, "range_m"),
            "range_rate_mps": _finite_metric(float(base[1]) * 200.0, "range_rate_mps"),
            "ata_deg": _finite_metric(math.degrees(math.atan2(float(base[8]), float(base[9]))), "ata_deg"),
            "aa_deg": _finite_metric(math.degrees(math.atan2(float(base[10]), float(base[11]))), "aa_deg"),
            "own_speed_mps": _finite_metric(own_speed, "own_speed_mps"),
            "own_altitude_m": _finite_metric(own_altitude, "own_altitude_m"),
            "own_specific_energy_m": _finite_metric(own_altitude + own_speed ** 2 / (2.0 * 9.80665), "own_specific_energy_m"),
            "target_attack_score": _finite_metric(source.get("target_attack_score"), "target_attack_score"),
            "ego_attack_score": _finite_metric(source.get("ego_attack_score"), "ego_attack_score"),
            "target_in_attack_zone": bool(source.get("target_in_attack_zone", False)),
            "ego_in_attack_zone": bool(source.get("ego_in_attack_zone", False)),
        }

    @staticmethod
    def _trace_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: payload[key]
            for key in (
                "step",
                "phase",
                "dynamic_state",
                "reference_action",
                "applied_action",
                "method",
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
            raise PostmergeLateralRuntimeProbeError("strict JSBSim backend was not selected")
        # The context deliberately remains identical across methods; only the action changes after handoff.
        self.env.set_runtime_specialist_context(
            specialist_key="head_on",
            specialist_profile=None,
            specialist_mode_name="postmerge_lateral_runtime_probe_runin",
            specialist_reason="fixed_head_on_paired_runtime_probe",
        )
        self.observation_schema = copy.deepcopy(observation["observation_schema"])
        self._observe(observation)
        metadata = scenario["metadata"]
        previous_hash = continuity.stable_hash(
            {
                "scenario": scenario["name"],
                "seed": int(seed),
                "initial_observation": [float(value) for value in self.observation["observation_vector"]],
            }
        )
        trace: list[dict[str, Any]] = []
        program_window: list[dict[str, Any]] = []
        handoff: dict[str, Any] | None = None
        previous_applied_action: np.ndarray | None = None
        backend_fallback_seen = False
        prediction_fallback_in_program = False
        candidate_box_limited_steps = 0
        candidate_slew_limited_steps = 0
        candidate_basis_violation_steps = 0
        candidate_nonfinite_action_steps = 0
        terminal_reason = "runtime_probe_horizon"

        while self.total_steps < self.env.max_steps:
            if self.observation is None or self.base is None:
                raise PostmergeLateralRuntimeProbeError("runtime observation was lost")
            pre_base = self.base.copy()
            pre_history = tuple(item.copy() for item in self.history)
            pre_phase, pre_state = self.phase, self.dynamic_state
            reference_66d = self._reference_observation()
            reference_action = raw_v5._finite_action(
                self.fixed_head_on.get_deterministic_action(self.observation["observation_vector"])
            )
            eligible = bool(
                handoff is None
                and pre_state == self.config["contract"]["target_dynamic_state"]
                and pre_phase in self.config["contract"]["target_phases"]
                and reference_66d is not None
                and not backend_fallback_seen
                and not bool(self.last_info.get("prediction_fallback", False))
            )
            if eligible:
                handoff = {
                    "step": self.total_steps + 1,
                    "phase": pre_phase,
                    "dynamic_state": pre_state,
                    "state_hash": continuity.stable_hash(
                        {
                            "base_observation_16d": [float(value) for value in pre_base],
                            "observation_66d": [float(value) for value in reference_66d],
                            "continuity_prev_hash": previous_hash,
                        }
                    ),
                }

            receipt: dict[str, Any] | None = None
            action = reference_action.copy()
            in_program = handoff is not None and len(program_window) < self.compiler_config.horizon_steps
            if in_program and self.method == METHODS[1]:
                basis = self._current_basis()
                receipt = compile_temporal_reference_step(
                    template=self.template,
                    step_index=len(program_window),
                    reference_action=reference_action,
                    previous_applied_action=(reference_action if previous_applied_action is None else previous_applied_action),
                    basis_vectors_world_m=basis,
                    config=self.compiler_config,
                    lateral_sign=int(self.config["candidate"]["lateral_sign"]),
                    vertical_sign=int(self.config["candidate"]["vertical_sign"]),
                )
                candidate_box_limited_steps += int(bool(receipt["action_box_limited"]))
                candidate_slew_limited_steps += int(bool(receipt["slew_limited"]))
                candidate_basis_violation_steps += int(
                    int(receipt["basis_rank"]) != 3
                    or float(receipt["basis_condition_number"]) > self.compiler_config.max_basis_condition_number
                )
                action = np.asarray(receipt["applied_action"], dtype=np.float64)
                if not np.isfinite(action).all():
                    candidate_nonfinite_action_steps += 1
                if candidate_box_limited_steps or candidate_slew_limited_steps or candidate_basis_violation_steps or candidate_nonfinite_action_steps:
                    raise PostmergeLateralRuntimeProbeError("candidate violated the frozen runtime envelope")

            next_observation, _ignored_reward, terminated, truncated, info = self.env.step(action)
            self.total_steps += 1
            self.last_info = info
            self._observe(next_observation)
            backend_fallback_seen |= bool(info.get("backend_fallback_occurred", False))
            if backend_fallback_seen:
                raise PostmergeLateralRuntimeProbeError("JSBSim backend fallback occurred")
            payload = self._step_payload(
                pre_base=pre_base,
                pre_history=pre_history,
                pre_phase=pre_phase,
                pre_state=pre_state,
                reference_66d=reference_66d,
                action=reference_action,
                info=info,
                previous_hash=previous_hash,
            )
            payload["method"] = self.method
            payload["applied_action"] = [float(value) for value in action]
            payload["compiler_receipt"] = receipt
            payload["continuity_hash"] = continuity.stable_hash(
                # Method labels are provenance metadata, not simulator state. Excluding
                # them preserves the causal pre-handoff hash across paired rollouts.
                {key: value for key, value in payload.items() if key not in {"continuity_hash", "method"}}
            )
            previous_hash = str(payload["continuity_hash"])
            trace.append(self._trace_payload(payload))
            previous_applied_action = action.copy()
            terminal_reason = str(payload["terminal_reason"])

            if in_program:
                if bool(payload["prediction_fallback"]):
                    prediction_fallback_in_program = True
                    raise PostmergeLateralRuntimeProbeError("prediction fallback occurred during the paired program")
                payload.update(self._after_metrics(info))
                program_window.append(payload)
            if terminated or truncated:
                break

        finite_program_telemetry = bool(
            len(program_window) == self.compiler_config.horizon_steps
            and all(
                all(
                    math.isfinite(float(step[key]))
                    for key in (
                        "target_attack_score",
                        "ego_attack_score",
                        "range_m",
                        "range_rate_mps",
                        "own_specific_energy_m",
                        "aoa_rad",
                        "actual_nz_g",
                    )
                )
                for step in program_window
            )
        )
        result = {
            "strict_jsbsim": True,
            "handoff_reached": handoff is not None,
            "handoff_step": None if handoff is None else handoff["step"],
            "handoff_state_hash": None if handoff is None else handoff["state_hash"],
            "program_steps": len(program_window),
            "program_complete": len(program_window) == self.compiler_config.horizon_steps,
            "no_reset_after_initialization": True,
            "no_backend_fallback": not backend_fallback_seen,
            "no_prediction_fallback_in_program": not prediction_fallback_in_program,
            "no_history_padding": True,
            "no_checkpoint_fallback": True,
            "continuity_hash_chain_complete": bool(trace),
            "finite_program_telemetry": finite_program_telemetry,
            "candidate_nonfinite_action_steps": candidate_nonfinite_action_steps,
            "candidate_box_limited_steps": candidate_box_limited_steps,
            "candidate_slew_limited_steps": candidate_slew_limited_steps,
            "candidate_basis_violation_steps": candidate_basis_violation_steps,
            "terminal_reason": terminal_reason,
            "ego_failure": is_ego_failure(terminal_reason),
            "fatal_contract_violation": False,
        }
        return {
            "source_id": self.source_id,
            "method": self.method,
            "opponent": str(self.env._opponent_stage),
            "scenario_name": str(scenario["name"]),
            "scenario_signature": str(metadata["mirror_pair_id"]),
            "mirror_sign": str(metadata["mirror_sign"]),
            "initial_state": str(metadata["initial_class"]),
            "initial_continuity_hash": trace[0]["continuity_prev_hash"] if trace else previous_hash,
            "continuity_trace": trace,
            "program_window": program_window,
            "result": result,
        }


def _build_record(
    values: Mapping[str, Any],
    *,
    method: str,
    opponent: str,
    scenario: Mapping[str, Any],
) -> dict[str, Any]:
    env = continuity._build_env(values["runtime_config"], values["registry"], opponent)
    encoder = continuity.FrozenP3Encoder(
        _resolve(str(values["config"]["inputs"]["frozen_p3_encoder"])),
        str(values["config"]["inputs"]["frozen_p3_encoder_sha256"]),
    )
    episode = PairedPostmergeEpisode(
        env,
        continuity._specialist(values["registry"]),
        encoder,
        values["runtime_config"],
        source_id=SOURCE_ID,
        method=method,
        template=values["template"],
        compiler_config=values["compiler_config"],
    )
    try:
        return episode.run(scenario, int(scenario["metadata"]["scenario_seed"]))
    finally:
        episode.close()


def _write_record(output: Path, record: Mapping[str, Any]) -> Path:
    path = output / "episodes" / str(record["method"]) / str(record["opponent"]) / (str(record["scenario_name"]) + ".json")
    _write_json(path, {"record": record})
    return path


def _require_execution_authorization(config: Mapping[str, Any], expected_records: int) -> None:
    execution = config.get("execution")
    if config["authorization"].get("execution_permitted") is not True or not isinstance(execution, Mapping):
        raise PostmergeLateralRuntimeProbeError("formal runtime execution is not authorized")
    if int(execution.get("expected_records", -1)) != expected_records:
        raise PostmergeLateralRuntimeProbeError("authorized record universe drifted")
    required_sha = str(execution.get("required_implementation_git_sha") or "")
    if len(required_sha) != 40 or subprocess.run(["git", "merge-base", "--is-ancestor", required_sha, "HEAD"], cwd=ROOT, check=False).returncode:
        raise PostmergeLateralRuntimeProbeError("authorized implementation SHA is not an ancestor of HEAD")
    files = execution.get("authorized_code_files")
    if not isinstance(files, list) or not files:
        raise PostmergeLateralRuntimeProbeError("authorized source file list is missing")
    for item in files:
        path = _resolve(str(item.get("path") or ""))
        if not path.is_file() or _hash(path) != str(item.get("sha256") or ""):
            raise PostmergeLateralRuntimeProbeError(f"authorized source hash mismatch: {path}")
    smoke_path = _resolve(str(execution.get("required_smoke_gate") or ""))
    if not smoke_path.is_file() or _hash(smoke_path) != str(execution.get("required_smoke_gate_sha256") or ""):
        raise PostmergeLateralRuntimeProbeError("required engineering smoke hash mismatch")
    smoke = _load_json(smoke_path)
    if smoke.get("smoke_id") != config["engineering_smoke"]["smoke_id"] or smoke.get("passes") is not True:
        raise PostmergeLateralRuntimeProbeError("required engineering smoke did not pass")


def _write_matrix_artifacts(output: Path, records: list[dict[str, Any]], values: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    config = values["config"]
    summary = summarize_runtime_records(
        records,
        scenarios_per_opponent=len(values["scenarios"]),
        metrics=config["metrics"],
        gates=config["gates"],
    )
    index = []
    for record in records:
        record_path = _write_record(output, record)
        index.append({"path": str(record_path.relative_to(output)), "sha256": _hash(record_path)})
    _write_json(output / "records_index.json", {"source_id": SOURCE_ID, "records": index})
    gate = {key: value for key, value in summary.items() if key != "pair_rows"}
    _write_json(output / "pairing_audit.json", {"source_id": SOURCE_ID, "pairs": summary["pair_rows"]})
    fields = list(summary["pair_rows"][0]) if summary["pair_rows"] else ["opponent", "scenario_name"]
    with (output / "paired_geometry_deltas.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary["pair_rows"])
    _write_json(output / "runtime_probe_gate.json", {"source_id": SOURCE_ID, **gate})
    _write_json(
        output / "run_manifest.json",
        {
            "source_id": SOURCE_ID,
            "stage": config["stage"],
            "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "config_sha256": _hash(config_path),
            "p2_summary_sha256": _hash(_resolve(str(config["inputs"]["p2_summary"]))),
            "parent_manifest_sha256": _hash(_resolve(str(config["inputs"]["parent_manifest"]))),
            "records": len(records),
            "training": "prohibited",
            "candidate_actions_enter_env": True,
        },
    )
    return gate


def run_engineering_smoke(config_path: Path) -> dict[str, Any]:
    values = _load_inputs(config_path)
    config = values["config"]
    smoke_cfg = config["engineering_smoke"]
    output = _resolve(str(smoke_cfg["output_root"]))
    if not output.is_absolute() or output.exists():
        raise PostmergeLateralRuntimeProbeError("engineering smoke output root must be absolute and absent")
    scenario_by_name = {str(item["name"]): item for item in values["scenarios"]}
    records: list[dict[str, Any]] = []
    output.mkdir(parents=True)
    try:
        for opponent in OPPONENTS:
            scenario = scenario_by_name.get(str(smoke_cfg["scenario_by_opponent"][opponent]))
            if scenario is None:
                raise PostmergeLateralRuntimeProbeError("engineering smoke scenario is absent from probe manifest")
            baseline = _build_record(values, method=METHODS[0], opponent=opponent, scenario=scenario)
            candidate = _build_record(values, method=METHODS[1], opponent=opponent, scenario=scenario)
            records.extend((baseline, candidate))
            _write_record(output, baseline)
            _write_record(output, candidate)
        summary = summarize_runtime_records(
            records,
            scenarios_per_opponent=12,
            metrics=config["metrics"],
            gates=config["gates"],
        )
        smoke_pairs = [row for row in summary["pair_rows"] if row["baseline_present"] and row["candidate_present"]]
        passes = bool(
            len(records) == int(smoke_cfg["expected_records"])
            and len(smoke_pairs) == len(OPPONENTS)
            and all(row["baseline_window_valid"] and row["candidate_window_valid"] and row["handoff_hash_match"] for row in smoke_pairs)
            and not summary["contract_violations"]
        )
        gate = {
            "smoke_id": smoke_cfg["smoke_id"],
            "source_id_under_test": SOURCE_ID,
            "records": len(records),
            "pairs": smoke_pairs,
            "contract_violations": summary["contract_violations"],
            "passes": passes,
        }
        _write_json(output / "engineering_smoke_gate.json", gate)
        _write_json(output / "records_index.json", {"smoke_id": smoke_cfg["smoke_id"], "records": len(records)})
        if not passes:
            raise PostmergeLateralRuntimeProbeError("engineering smoke failed")
        return {"output": str(output), **gate}
    except Exception as error:
        _write_json(output / "failure.json", {"smoke_id": smoke_cfg["smoke_id"], "error": repr(error), "completed_records": len(records)})
        raise


def run(config_path: Path) -> dict[str, Any]:
    values = _load_inputs(config_path)
    config = values["config"]
    expected_records = len(METHODS) * len(OPPONENTS) * len(values["scenarios"])
    if config["authorization"].get("require_clean_worktree") is True and subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise PostmergeLateralRuntimeProbeError("formal runtime probe requires a clean worktree")
    _require_execution_authorization(config, expected_records)
    output = _resolve(str(config["outputs"]["root"]))
    if not output.is_absolute() or output.exists():
        raise PostmergeLateralRuntimeProbeError("formal output root must be absolute and absent")
    records: list[dict[str, Any]] = []
    output.mkdir(parents=True)
    try:
        for method in METHODS:
            for opponent in OPPONENTS:
                for scenario in values["scenarios"]:
                    records.append(_build_record(values, method=method, opponent=opponent, scenario=scenario))
        return {"output": str(output), **_write_matrix_artifacts(output, records, values, config_path)}
    except Exception as error:
        _write_json(output / "failure.json", {"source_id": SOURCE_ID, "error": repr(error), "completed_records": len(records)})
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.execute and args.smoke:
        parser.error("--execute and --smoke are mutually exclusive")
    if args.smoke:
        print(json.dumps(run_engineering_smoke(args.config), ensure_ascii=False, indent=2))
        return 0
    if args.execute:
        print(json.dumps(run(args.config), ensure_ascii=False, indent=2))
        return 0
    print(json.dumps(validate_design(args.config), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
