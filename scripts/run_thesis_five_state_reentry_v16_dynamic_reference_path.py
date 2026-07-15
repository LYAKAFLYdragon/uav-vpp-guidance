#!/usr/bin/env python3
"""Run V16's non-learning, current-information-only dynamic reference-path probe."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_thesis_five_state_reentry_v15_fixed_encoder_temporal_corridor as v15
from uav_vpp_guidance.evaluation.thesis_reentry_v16_dynamic_reference_path import (
    METHODS,
    OPPONENTS,
    REFERENCE_METHOD,
    SOURCE_ID,
    is_ego_failure,
    semantic_axes_for_method,
    semantic_profile_for_method,
    summarize_records,
)
from uav_vpp_guidance.envs.observation import compute_relative_geometry
from uav_vpp_guidance.hierarchy.shared_intent_profiles import compile_profile
from uav_vpp_guidance.training import thesis_neutral_postmerge_reentry_recovery_pilot as engine
from uav_vpp_guidance.training.thesis_shared_skill_geometry import FrozenP3Encoder
from uav_vpp_guidance.virtual_point.dynamic_reference_path import (
    DynamicReferencePathConfig,
    build_three_anchor_path,
    compile_path_anchor_action,
    path_progress_from_range_rate,
    select_path_anchor,
    semantic_profile_for_anchor,
)
from uav_vpp_guidance.virtual_point.relative_frame_semantic import (
    RelativeFrameSemanticCompiler,
    RelativeFrameSemanticConfig,
)


class V16RuntimeError(RuntimeError):
    """Raised when V16 leaves its frozen record-only path contract."""


def _load(path: Path) -> dict[str, Any]:
    return v15._load(path)


def _resolve(value: str | Path) -> Path:
    return v15._resolve(value)


def _hash(path: Path) -> str:
    return v15._hash(path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    v15._write_json(path, value)


def _load_runtime_config(path: Path) -> dict[str, Any]:
    overlay = _load(path)
    if "base_config" not in overlay:
        return overlay
    base_path = _resolve(str(overlay["base_config"]))
    if _hash(base_path) != str(overlay.get("base_config_sha256") or ""):
        raise V16RuntimeError("V16 authorization overlay base config hash mismatch")
    config = copy.deepcopy(_load(base_path))
    config["authorization"] = copy.deepcopy(overlay.get("authorization", {}))
    config["_authorization_execution"] = copy.deepcopy(overlay.get("execution", {}))
    return config


def _compiler(config: Mapping[str, Any]) -> RelativeFrameSemanticCompiler:
    values = dict(config["semantic_compiler"])
    values.pop("raw_action_ranges", None)
    values.pop("profile", None)
    return RelativeFrameSemanticCompiler(RelativeFrameSemanticConfig(**values))


def _path_config(config: Mapping[str, Any]) -> DynamicReferencePathConfig:
    path = config.get("path")
    if not isinstance(path, Mapping):
        raise V16RuntimeError("V16 path configuration is missing")
    return DynamicReferencePathConfig(
        escape_extent_m=float(path["escape_extent_m"]),
        connector_fraction=float(path["connector_fraction"]),
        opening_rate_start_mps=float(path["opening_rate_start_mps"]),
        opening_rate_full_mps=float(path["opening_rate_full_mps"]),
    )


def _validate_fixed_encoder_profile(config: Mapping[str, Any]) -> str:
    observation = config.get("observation_contract")
    if not isinstance(observation, Mapping) or observation.get("profile_role") != "fixed_66d_target_weight_anchor_only":
        raise V16RuntimeError("V16 fixed 66-D profile role is missing")
    if observation.get("dynamic_profile_in_66d") != "prohibited":
        raise V16RuntimeError("V16 must prohibit a dynamic profile inside 66-D")
    profile_name = str(observation.get("encoder_profile") or "")
    compiled = compile_profile(profile_name)
    if compiled["target_vector"].shape != (6,) or compiled["weight_vector"].shape != (6,):
        raise V16RuntimeError("V16 encoder profile contract is malformed")
    if "profile" in config.get("semantic_compiler", {}):
        raise V16RuntimeError("V16 semantic compiler must not define an encoder profile")
    return profile_name


def _runtime_contract(config: Mapping[str, Any]) -> None:
    execution = config.get("execution")
    if not isinstance(execution, Mapping) or not execution.get("runtime_template"):
        raise V16RuntimeError("V16 requires execution.runtime_template before output creation")
    runtime_path = _resolve(str(execution["runtime_template"]))
    input_path = _resolve(str(config["inputs"]["runtime_template"]))
    if (
        runtime_path != input_path
        or not runtime_path.is_file()
        or _hash(runtime_path) != str(config["inputs"]["runtime_template_sha256"])
    ):
        raise V16RuntimeError("V16 runtime template contract mismatch")


def validate_design(config_path: Path) -> dict[str, Any]:
    """Verify all V16 prerequisites without creating output or running JSBSim."""

    config = _load_runtime_config(config_path)
    if config.get("source_id") != SOURCE_ID:
        raise V16RuntimeError("V16 source ID drifted")
    prohibited = (
        "training_permitted",
        "tuning_permitted",
        "reward_change_permitted",
        "candidate_policy_loading_permitted",
        "high_level_ppo_training_permitted",
        "four_skill_training_permitted",
        "combat_finetune_permitted",
    )
    for key in prohibited:
        if config.get("authorization", {}).get(key) is not False:
            raise V16RuntimeError(f"V16 must prohibit {key}")
    if tuple(config.get("opponents", {}).get("order", ())) != OPPONENTS or config["opponents"].get("pooled_gate") != "prohibited":
        raise V16RuntimeError("V16 opponent contract drifted")
    if tuple(config.get("methods", {}).keys()) != METHODS:
        raise V16RuntimeError("V16 method matrix drifted")
    if config["methods"]["progress_triggered_reference_path"].get("path_adapter_enabled") is not True:
        raise V16RuntimeError("V16 path method must use the path adapter")
    if any(
        config.get("contract", {}).get(key) != "prohibited"
        for key in ("future_simulator_state", "synthetic_target")
    ):
        raise V16RuntimeError("V16 must forbid future-state and synthetic-target input")
    if config.get("contract", {}).get("guidance_pid_change") != "prohibited":
        raise V16RuntimeError("V16 must preserve guidance/PID")
    encoder_profile = _validate_fixed_encoder_profile(config)
    _runtime_contract(config)
    compiler = _compiler(config)
    if compiler.config.raw_action_offset_frame != "world_neu":
        raise V16RuntimeError("V16 must preserve the frozen world-frame VPP action interface")
    path = _path_config(config)
    if int(config["path"].get("anchor_count", 0)) != 3 or tuple(config["path"].get("semantic_profiles_by_anchor", ())) != (
        "range_extension",
        "range_extension",
        "reentry_preparation",
    ):
        raise V16RuntimeError("V16 three-anchor profile mapping drifted")
    manifest_path = _resolve(config["inputs"]["manifest"])
    if _hash(manifest_path) != str(config["inputs"]["manifest_sha256"]):
        raise V16RuntimeError("V16 manifest hash mismatch")
    manifest = _load(manifest_path)
    if manifest.get("source_id") != SOURCE_ID + "-MANIFEST12" or len(manifest.get("scenarios", [])) != 12:
        raise V16RuntimeError("V16 requires the complete 12-scenario manifest")
    if manifest.get("protocol", {}).get("training_permitted") is not False or manifest["protocol"].get("reward_recording") != "prohibited":
        raise V16RuntimeError("V16 manifest is not non-learning and reward-free")
    output = _resolve(config["outputs"]["root"])
    if not output.is_absolute() or output.exists():
        raise V16RuntimeError("V16 output root must be absolute and absent")
    for key in ("runtime_template", "runtime_registry", "frozen_p3_encoder"):
        value = _resolve(config["inputs"][key])
        if not value.is_file() or _hash(value) != str(config["inputs"][key + "_sha256"]):
            raise V16RuntimeError(f"V16 frozen input mismatch: {key}")
    free_gb = shutil.disk_usage(output.parent).free / 1024**3
    return {
        "source_id": SOURCE_ID,
        "mode": "nonlearning_current_information_dynamic_reference_path_preflight",
        "encoder_profile": encoder_profile,
        "manifest_scenarios": len(manifest["scenarios"]),
        "planned_records": len(METHODS) * len(OPPONENTS) * len(manifest["scenarios"]),
        "methods": list(METHODS),
        "path_anchor_count": 3,
        "path_progress_source": "observed_range_rate_only",
        "free_disk_gb": round(free_gb, 3),
        "disk_gate_would_pass": free_gb >= float(config["observability"]["min_free_disk_gb"]),
        "output_root_absent": True,
        "runtime_contract_passed": True,
        "path_contract_passed": path.escape_extent_m > 0.0,
        "fixed_encoder_dynamic_path_separation_passed": True,
    }


def _state_position_neu(state: Mapping[str, Any], name: str) -> np.ndarray:
    for key in ("position_neu", "position_m", "position"):
        value = state.get(key)
        try:
            position = np.asarray(value, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            continue
        if position.shape == (3,) and np.isfinite(position).all():
            return position
    raise V16RuntimeError(f"V16 lacks finite {name} position")


class DynamicReferencePathEpisode(v15.FixedEncoderTemporalCorridorEpisode):
    """Continuous V16 path transfer using V15's fixed 66-D observability anchor."""

    def _assert_path_interface(self) -> None:
        """Fail closed unless the existing predicted-target VPP anchor is identity-mapped."""

        self._assert_raw_interface()
        generator = self.env.virtual_point_generator
        if generator is None or str(self.env.config.get("virtual_point", {}).get("anchor_mode")) != "predicted_target":
            raise V16RuntimeError("V16 requires the frozen predicted-target VPP anchor")
        if not np.isclose(float(getattr(generator, "predicted_target_blend", np.nan)), 1.0):
            raise V16RuntimeError("V16 cannot reinterpret a blended predicted-target anchor")
        if not np.isclose(float(getattr(generator, "predicted_target_forward_scale", np.nan)), 1.0):
            raise V16RuntimeError("V16 cannot reinterpret a scaled predicted-target anchor")
        if bool(getattr(self.env, "_sensor_noise", {}).get("enabled", False)):
            raise V16RuntimeError("V16 current-state path receipt requires sensor noise to be disabled")

    def _current_path(self) -> dict[str, Any]:
        """Build from the exact state and predictor history used by the next action.

        ``info`` reports post-step aircraft states but a pre-step prediction.  To
        avoid mixing those timestamps, V16 synchronizes the frozen predictor to
        the current simulator state before the following action; the environment
        then reuses that same buffered frame when it executes the action.
        """

        if self.base is None:
            raise V16RuntimeError("V16 path requested without a base observation")
        own_state, target_state = self.env._get_current_states(noisy=True)
        target_state = self.env._task_get_target_state(target_state, own_state)
        own_state, target_state = self.env._task_pre_step(own_state, target_state)
        relative_state = compute_relative_geometry(own_state, target_state)
        if not self.env._sync_prediction_buffer_for_current_state(own_state, target_state, relative_state):
            raise V16RuntimeError("V16 could not synchronize the frozen predictor")
        adapter = self.env.trajectory_predictor_adapter
        if adapter is None:
            raise V16RuntimeError("V16 frozen predictor adapter is missing")
        try:
            predicted_position, _variance, prediction_info = adapter.predict(target_state)
            predicted_position = np.asarray(predicted_position, dtype=np.float64).reshape(-1)
        except Exception as error:
            raise V16RuntimeError("V16 frozen predictor call failed") from error
        if predicted_position.shape != (3,) or not np.isfinite(predicted_position).all():
            raise V16RuntimeError("V16 predicted target position is invalid")
        if bool(prediction_info.get("fallback", False)) or prediction_info.get("prediction_valid") is False:
            raise V16RuntimeError("V16 requires a valid frozen predictor without fallback")
        own_position = _state_position_neu(own_state, "own")
        target_position = _state_position_neu(target_state, "target")
        path = build_three_anchor_path(
            own_position_neu=own_position,
            target_position_neu=target_position,
            predicted_target_position_neu=predicted_position,
            config=_path_config(self.config),
        )
        range_rate_mps = float(self.base[1] * v15.v14.v11.RANGE_RATE_REFERENCE_MPS)
        progress = path_progress_from_range_rate(range_rate_mps, _path_config(self.config))
        selected_index, selected_anchor = select_path_anchor(path, progress)
        return {
            "own_state": own_state,
            "target_state": target_state,
            "own_position_neu": own_position,
            "target_position_neu": target_position,
            "predicted_target_position_neu": predicted_position,
            "prediction_valid": True,
            "prediction_fallback": False,
            "range_rate_mps": range_rate_mps,
            "path_progress": progress,
            "anchors_neu": path,
            "selected_anchor_index": selected_index,
            "selected_anchor_neu": selected_anchor,
            "selected_profile": semantic_profile_for_anchor(selected_index),
        }

    def run(self, scenario: Mapping[str, Any], seed: int, *, method: str) -> dict[str, Any]:
        if method not in METHODS:
            raise V16RuntimeError("V16 method is outside the preregistered matrix")
        self.history.clear()
        self.history_features.reset()
        self.observation = None
        self.base = None
        self.last_info = {}
        self.total_steps = 0
        self.phase_trace = []
        event = v15.v14.v11.RawSIFirstPassConfig(**dict(self.config["observability"]["raw_si_first_pass_event"]))
        self.phase_tracker = v15.v14.v11.RawSIReentryPhaseTracker(v15.v14.v11.RawSIReentryPhaseConfig(first_pass=event))
        observation = self.env.reset(scenario=copy.deepcopy(dict(scenario)), seed=int(seed))
        if self.env._backend != "jsbsim":
            raise V16RuntimeError("V16 requires strict JSBSim")
        self._assert_path_interface()
        self.env.set_runtime_specialist_context(
            specialist_key="head_on",
            specialist_profile=None,
            specialist_mode_name="v16_frozen_run_in",
            specialist_reason="v16_raw_si_reentry_run_in",
        )
        self.observation_schema = copy.deepcopy(observation["observation_schema"])
        self._observe(observation)
        done = False
        terminal_reason = "horizon_before_handoff"
        while not done and self.total_steps < int(self.config["observability"]["max_high_level_steps"]):
            if self._trigger_ready():
                break
            info, done = self._reference_step()
            terminal_reason = str(info.get("reason") or "timeout")
        if not self._trigger_ready():
            return {
                "handoff_reached": False,
                "window_steps": 0,
                "contract_passed": not bool(self.last_info.get("backend_fallback_occurred", False)),
                "terminal_reason": terminal_reason,
                "ego_failure": is_ego_failure(terminal_reason),
                "phase_trace": self.phase_trace,
            }

        self.env.set_runtime_specialist_context(
            specialist_key="head_on",
            specialist_profile="dynamic_reference_path",
            specialist_mode_name="v16_dynamic_reference_path",
            specialist_reason=method,
        )
        handoff_step = self.total_steps
        handoff_observation = self._observation66d()
        handoff_hash = v15.v14._stable_hash(
            {
                "base": self.base,
                "history": tuple(self.history),
                "observation": self.observation["observation_vector"] if self.observation else None,
                "phase": self._phase(),
            }
        )
        window: list[dict[str, Any]] = []
        target_steps = int(self.config["observability"]["local_window_steps"])
        for local_step in range(target_steps):
            if self.observation is None or self.base is None:
                raise V16RuntimeError("V16 lost observation during path window")
            path_context = self._current_path()
            objective_profile = str(path_context["selected_profile"])
            objective_stage = "escape" if int(path_context["selected_anchor_index"]) < 2 else "reentry"
            before_vector, before_errors, before_weighted = self._snapshot(objective_profile)
            reference_action = v15.v14.v11.v9.finite_action(
                self.policy.get_deterministic_action(self.observation["observation_vector"])
            )
            action_profile = semantic_profile_for_method(method, objective_profile)
            semantic_receipt = self.compiler.compile(
                reference_action=reference_action,
                base_geometry=self.base,
                own_state=path_context["own_state"],
                target_state=path_context["target_state"],
                raw_action_ranges=self.config["semantic_compiler"]["raw_action_ranges"],
                profile_name=action_profile,
                enabled_axes=semantic_axes_for_method(method),
            )
            path_action_receipt: dict[str, Any] | None = None
            if method == REFERENCE_METHOD:
                semantic_receipt["applied_action"] = reference_action.tolist()
                semantic_receipt["actual_semantic_world_delta_m"] = [0.0, 0.0, 0.0]
                semantic_receipt["actual_semantic_delta_norm_m"] = 0.0
                applied_action = reference_action
            elif method == "static_range_extension":
                applied_action = np.asarray(semantic_receipt["applied_action"], dtype=np.float64)
            else:
                path_action_receipt = compile_path_anchor_action(
                    semantic_action=semantic_receipt["applied_action"],
                    selected_anchor_neu=path_context["selected_anchor_neu"],
                    generator_anchor_neu=path_context["predicted_target_position_neu"],
                    raw_action_ranges=self.config["semantic_compiler"]["raw_action_ranges"],
                    raw_action_offset_frame=str(self.config["semantic_compiler"]["raw_action_offset_frame"]),
                    own_state=path_context["own_state"],
                    target_state=path_context["target_state"],
                )
                applied_action = np.asarray(path_action_receipt["applied_action"], dtype=np.float64)
            observation, _ignored, terminated, truncated, info = self.env.step(
                np.asarray(applied_action, dtype=np.float32)
            )
            self.total_steps += 1
            self.last_info = info
            self._observe(observation)
            if bool(info.get("backend_fallback_occurred", False)):
                raise V16RuntimeError("V16 backend fallback occurred")
            after_vector, after_errors, after_weighted = self._snapshot(objective_profile)
            own_after = info.get("own_state") if isinstance(info.get("own_state"), Mapping) else {}
            window.append(
                {
                    "local_step": local_step,
                    "objective_stage": objective_stage,
                    "objective_profile": objective_profile,
                    "action_profile": action_profile,
                    "semantic_axes": semantic_axes_for_method(method),
                    "own_position_neu": path_context["own_position_neu"].tolist(),
                    "target_position_neu": path_context["target_position_neu"].tolist(),
                    "predicted_target_position_neu": path_context["predicted_target_position_neu"].tolist(),
                    "prediction_valid": bool(path_context["prediction_valid"]),
                    "prediction_fallback": bool(path_context["prediction_fallback"]),
                    "range_rate_mps": float(path_context["range_rate_mps"]),
                    "path_progress": float(path_context["path_progress"]),
                    "anchors_neu": np.asarray(path_context["anchors_neu"], dtype=np.float64).tolist(),
                    "selected_anchor_index": int(path_context["selected_anchor_index"]),
                    "selected_anchor_profile": objective_profile,
                    "semantic_receipt": semantic_receipt,
                    "path_action_receipt": path_action_receipt,
                    "phase_before": self.phase_trace[-2]["phase"],
                    "observation_66d": handoff_observation.tolist() if not window else self._observation66d().tolist(),
                    "before_intent_vector": before_vector,
                    "after_intent_vector": after_vector,
                    "before_component_errors": before_errors,
                    "after_component_errors": after_errors,
                    "before_weighted_error": before_weighted,
                    "after_weighted_error": after_weighted,
                    "env_vpp_world_offset_m": self._finite_vector(info.get("vp_world_offset"), "vp_world_offset"),
                    "guidance_command_normalized": self._guidance_vector(info),
                    "actual_nz_g": float(own_after.get("nz_g", info.get("nz_g"))),
                    "saturated": any(
                        bool(info.get(key, False))
                        for key in ("nz_saturated", "roll_rate_saturated", "throttle_saturated")
                    ),
                }
            )
            done = bool(
                terminated
                or truncated
                or self.total_steps >= int(self.config["observability"]["max_high_level_steps"])
            )
            terminal_reason = str(info.get("reason") or "timeout")
            if done:
                break
        while not done and self.total_steps < int(self.config["observability"]["max_high_level_steps"]):
            info, done = self._reference_step()
            terminal_reason = str(info.get("reason") or "timeout")
        finite_window = all(
            len(step["observation_66d"]) == 66
            and np.isfinite(np.asarray(step["observation_66d"], dtype=np.float64)).all()
            and np.isfinite(np.asarray(step["after_component_errors"], dtype=np.float64)).all()
            and np.isfinite(float(step["after_weighted_error"]))
            and np.isfinite(np.asarray(step["env_vpp_world_offset_m"], dtype=np.float64)).all()
            and np.isfinite(np.asarray(step["guidance_command_normalized"], dtype=np.float64)).all()
            and (step["path_action_receipt"] is None or np.isfinite(np.asarray(step["path_action_receipt"]["applied_action"], dtype=np.float64)).all())
            for step in window
        )
        return {
            "handoff_reached": True,
            "handoff_state_hash": handoff_hash,
            "handoff_step": handoff_step,
            "handoff_observation_66d": handoff_observation.tolist(),
            "window_steps": len(window),
            "terminal_reason": terminal_reason,
            "ego_failure": is_ego_failure(terminal_reason),
            "contract_passed": finite_window and not bool(self.last_info.get("backend_fallback_occurred", False)),
            "window": window,
            "phase_trace": self.phase_trace,
        }


def _require_execution_authorization(config: Mapping[str, Any], scenario_count: int) -> None:
    execution = config.get("_authorization_execution")
    if config.get("authorization", {}).get("execution_permitted") is not True or not isinstance(execution, Mapping):
        raise V16RuntimeError("V16 execution is not authorized")
    expected = len(METHODS) * len(OPPONENTS) * scenario_count
    if int(execution.get("expected_records", -1)) != expected:
        raise V16RuntimeError("V16 authorization record universe drifted")
    required_sha = str(execution.get("required_implementation_git_sha") or "")
    if not required_sha or subprocess.run(["git", "cat-file", "-e", required_sha], cwd=ROOT, check=False).returncode:
        raise V16RuntimeError("V16 implementation SHA is unavailable")
    if subprocess.run(["git", "merge-base", "--is-ancestor", required_sha, "HEAD"], cwd=ROOT, check=False).returncode:
        raise V16RuntimeError("V16 implementation SHA is not an ancestor of HEAD")
    files = execution.get("authorized_code_files")
    if not isinstance(files, list) or not files:
        raise V16RuntimeError("V16 authorization file whitelist is missing")
    for item in files:
        path = _resolve(str(item.get("path") or ""))
        if not path.is_file() or _hash(path) != str(item.get("sha256") or ""):
            raise V16RuntimeError(f"V16 authorized code hash mismatch: {path}")


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
    config = _load_runtime_config(args.config)
    preflight = validate_design(args.config)
    if not args.execute:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0
    if not preflight["disk_gate_would_pass"]:
        raise V16RuntimeError("V16 free-disk gate is not satisfied")
    manifest = _load(_resolve(config["inputs"]["manifest"]))
    _require_execution_authorization(config, len(manifest["scenarios"]))
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise V16RuntimeError("V16 requires a clean worktree")
    registry = _load(_resolve(config["inputs"]["runtime_registry"]))
    output = _resolve(config["outputs"]["root"])
    output.mkdir(parents=True)
    _write_json(output / "authorized_config.json", {"config": config, "config_sha256": _hash(args.config)})
    compiler = _compiler(config)
    records: list[dict[str, Any]] = []
    try:
        for method in METHODS:
            for opponent in OPPONENTS:
                for scenario in manifest["scenarios"]:
                    env = v15.v14.build_record_only_env(config, registry, opponent)
                    policy = engine._specialist(registry, "frozen_fixed_head_on")
                    encoder = FrozenP3Encoder(
                        Path(config["inputs"]["frozen_p3_encoder"]),
                        config["inputs"]["frozen_p3_encoder_sha256"],
                    )
                    episode = DynamicReferencePathEpisode(env, policy, encoder, config, compiler)
                    try:
                        result = episode.run(scenario, int(scenario["metadata"]["scenario_seed"]), method=method)
                    finally:
                        episode.close()
                    row = _record(scenario, opponent, method, result)
                    records.append(row)
                    _write_json(output / "episodes" / method / opponent / f"{scenario['name']}.json", {"record": row})
        gates = config["gates"]
        summary = summarize_records(
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
        gate = {"source_id": SOURCE_ID, **summary}
        _write_json(output / "dynamic_reference_path_gate.json", gate)
        _write_json(
            output / "run_manifest.json",
            {
                "source_id": SOURCE_ID,
                "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "config_sha256": _hash(args.config),
                "manifest_sha256": _hash(_resolve(config["inputs"]["manifest"])),
                "records": len(records),
                "methods": list(METHODS),
                "training": "prohibited",
                "reward_recording": "prohibited",
                "path_input": "current_own_target_and_frozen_prediction_only",
                "authorized_code_files": copy.deepcopy(config.get("_authorization_execution", {}).get("authorized_code_files", [])),
            },
        )
        print(json.dumps(gate, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        _write_json(output / "failure.json", {"source_id": SOURCE_ID, "error": repr(error), "completed_records": len(records)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
