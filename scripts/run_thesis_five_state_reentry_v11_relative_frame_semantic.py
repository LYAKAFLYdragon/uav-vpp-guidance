#!/usr/bin/env python3
"""Run V11's non-learning relative-frame semantic VPP transfer probe."""

from __future__ import annotations

import argparse
from collections import deque
import copy
import hashlib
import json
import math
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
from uav_vpp_guidance.evaluation.raw_si_first_pass_event import RawSIFirstPassConfig
from uav_vpp_guidance.evaluation.raw_si_reentry_phase import RawSIReentryPhaseConfig, RawSIReentryPhaseTracker
from uav_vpp_guidance.evaluation.single_motif_continuous_66d import project_base_observation
from uav_vpp_guidance.evaluation.thesis_five_state_raw_si_v5_record_only import _stable_hash, build_record_only_env
from uav_vpp_guidance.evaluation.thesis_reentry_v11_relative_frame import (
    METHODS,
    OPPONENTS,
    REFERENCE_METHOD,
    RelativeFrameProbeError,
    is_ego_failure,
    summarize_records,
)
from uav_vpp_guidance.hierarchy.five_state_observation_contract import FiveStateObservationContract
from uav_vpp_guidance.hierarchy.shared_intent_profiles import compile_profile
from uav_vpp_guidance.hierarchy.temporal_geometry_features import TemporalGeometryFeatureExtractor
from uav_vpp_guidance.training import thesis_defext_rangeext_pilot as legacy
from uav_vpp_guidance.training import thesis_neutral_postmerge_reentry_recovery_pilot as engine
from uav_vpp_guidance.training.thesis_shared_skill_geometry import FrozenP3Encoder, intent_loss
from uav_vpp_guidance.virtual_point.relative_frame_semantic import (
    RelativeFrameSemanticCompiler,
    RelativeFrameSemanticConfig,
)


SOURCE_ID = "THESIS-FIVE-STATE-REENTRY-RELATIVE-FRAME-SEMANTIC-V11"
RANGE_REFERENCE_M = 5000.0
RANGE_RATE_REFERENCE_MPS = 200.0


class V11RuntimeError(RuntimeError):
    """Raised when V11 departs from its frozen non-learning contract."""


def _load(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise V11RuntimeError(f"expected YAML mapping: {path}")
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
        raise V11RuntimeError("V11 authorization overlay base config hash mismatch")
    config = copy.deepcopy(_load(base_path))
    config["authorization"] = copy.deepcopy(overlay.get("authorization", {}))
    config["_authorization_execution"] = copy.deepcopy(overlay.get("execution", {}))
    return config


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _methods(config: Mapping[str, Any]) -> dict[str, dict[str, bool]]:
    methods = config.get("methods")
    if not isinstance(methods, Mapping) or tuple(methods) != METHODS:
        raise V11RuntimeError(f"V11 requires methods in this order: {METHODS}")
    parsed: dict[str, dict[str, bool]] = {}
    expected_axes = {"range_extension", "signed_turn", "vertical_energy"}
    for name, axes in methods.items():
        if not isinstance(axes, Mapping) or set(axes) != expected_axes:
            raise V11RuntimeError(f"V11 method {name} must declare every semantic axis")
        parsed[str(name)] = {axis: bool(value) for axis, value in axes.items()}
    if any(parsed[REFERENCE_METHOD].values()):
        raise V11RuntimeError("reference method must not enable a semantic axis")
    if not parsed["range_extension_full"] == {axis: True for axis in expected_axes}:
        raise V11RuntimeError("V11 full profile method drifted")
    return parsed


def _compiler(config: Mapping[str, Any]) -> RelativeFrameSemanticCompiler:
    values = dict(config["semantic_compiler"])
    values.pop("profile", None)
    values.pop("raw_action_ranges", None)
    return RelativeFrameSemanticCompiler(RelativeFrameSemanticConfig(**values))


def _require_execution_authorization(config: Mapping[str, Any], scenario_count: int) -> None:
    if config.get("authorization", {}).get("execution_permitted") is not True:
        raise V11RuntimeError("V11 execution is not authorized")
    execution = config.get("_authorization_execution")
    if not isinstance(execution, Mapping):
        raise V11RuntimeError("V11 requires a hash-bound authorization overlay")
    required_sha = str(execution.get("required_implementation_git_sha") or "")
    if not required_sha or subprocess.run(["git", "cat-file", "-e", required_sha], cwd=ROOT, check=False).returncode:
        raise V11RuntimeError("V11 implementation SHA is unavailable")
    expected_records = len(METHODS) * len(OPPONENTS) * scenario_count
    if int(execution.get("expected_records", -1)) != expected_records:
        raise V11RuntimeError("V11 authorization record universe drifted")
    files = execution.get("authorized_code_files")
    if not isinstance(files, list) or not files:
        raise V11RuntimeError("V11 authorization file whitelist is missing")
    for item in files:
        if not isinstance(item, Mapping):
            raise V11RuntimeError("V11 authorization file whitelist is invalid")
        path = _resolve(str(item.get("path") or ""))
        if not path.is_file() or _hash(path) != str(item.get("sha256") or ""):
            raise V11RuntimeError(f"V11 authorized code hash mismatch: {path}")


def validate_design(config_path: Path) -> dict[str, Any]:
    """Validate V11's frozen inputs without instantiating JSBSim."""

    config = _load_runtime_config(config_path)
    if config.get("source_id") != SOURCE_ID:
        raise V11RuntimeError("V11 source ID drifted")
    for key in ("training_permitted", "tuning_permitted", "reward_change_permitted", "candidate_policy_loading_permitted", "high_level_ppo_training_permitted"):
        if config.get("authorization", {}).get(key) is not False:
            raise V11RuntimeError(f"V11 must prohibit {key}")
    if tuple(config.get("opponents", {}).get("order", ())) != OPPONENTS or config["opponents"].get("pooled_gate") != "prohibited":
        raise V11RuntimeError("V11 opponent contract drifted")
    methods = _methods(config)
    compiler = _compiler(config)
    if compiler.config.raw_action_offset_frame != "world_neu":
        raise V11RuntimeError("V11 only supports the frozen world-frame raw VPP interface")
    manifest_path = _resolve(config["inputs"]["manifest"])
    if _hash(manifest_path) != str(config["inputs"]["manifest_sha256"]):
        raise V11RuntimeError("V11 manifest hash mismatch")
    manifest = _load(manifest_path)
    if manifest.get("source_id") != SOURCE_ID + "-MANIFEST12" or len(manifest.get("scenarios", [])) != 12:
        raise V11RuntimeError("V11 requires its complete 12-scenario manifest")
    protocol = manifest.get("protocol", {})
    if protocol.get("training_permitted") is not False or protocol.get("reward_recording") != "prohibited":
        raise V11RuntimeError("V11 manifest is not non-learning and reward-free")
    output = _resolve(config["outputs"]["root"])
    if not output.is_absolute() or output.exists():
        raise V11RuntimeError("V11 output root must be absolute and absent")
    for key in ("runtime_template", "runtime_registry", "frozen_p3_encoder"):
        path = _resolve(config["inputs"][key])
        if not path.is_file() or _hash(path) != str(config["inputs"][key + "_sha256"]):
            raise V11RuntimeError(f"V11 frozen input mismatch: {key}")
    free_gb = shutil.disk_usage(output.parent).free / 1024**3
    return {
        "source_id": SOURCE_ID,
        "mode": "nonlearning_relative_frame_semantic_transfer_preflight",
        "manifest_scenarios": len(manifest["scenarios"]),
        "planned_records": len(methods) * len(OPPONENTS) * len(manifest["scenarios"]),
        "methods": list(methods),
        "free_disk_gb": round(free_gb, 3),
        "disk_gate_would_pass": free_gb >= float(config["observability"]["min_free_disk_gb"]),
        "output_root_absent": True,
    }


class SemanticTransferEpisode:
    """One continuous V11 episode with a fixed semantic compiler window."""

    def __init__(self, env: Any, policy: Any, encoder: FrozenP3Encoder, config: Mapping[str, Any], compiler: RelativeFrameSemanticCompiler) -> None:
        self.env = env
        self.policy = policy
        self.encoder = encoder
        self.config = config
        self.compiler = compiler
        self.contract = FiveStateObservationContract()
        self.history: deque[np.ndarray] = deque(maxlen=10)
        self.history_features = TemporalGeometryFeatureExtractor(10)
        self.observation_schema: Mapping[str, Any] = {}
        self.observation: Mapping[str, Any] | None = None
        self.base: np.ndarray | None = None
        self.last_info: Mapping[str, Any] = {}
        self.total_steps = 0
        event = RawSIFirstPassConfig(**dict(config["observability"]["raw_si_first_pass_event"]))
        self.phase_tracker = RawSIReentryPhaseTracker(RawSIReentryPhaseConfig(first_pass=event))
        self.phase_trace: list[dict[str, Any]] = []

    def close(self) -> None:
        self.env.close()

    def _observe(self, observation: Mapping[str, Any]) -> None:
        base = project_base_observation(observation["observation_vector"], self.observation_schema)
        if base.shape != (16,) or not np.isfinite(base).all():
            raise V11RuntimeError("V11 observed invalid 16-D base vector")
        self.history.append(base.copy())
        self.history_features.update(legacy._history_frame(base))
        raw_range = float(base[0] * RANGE_REFERENCE_M)
        raw_rate = float(base[1] * RANGE_RATE_REFERENCE_MPS)
        phase = self.phase_tracker.update(raw_range, raw_rate, step_index=self.total_steps)
        self.phase_trace.append({"step": self.total_steps, "raw_range_m": raw_range, "raw_range_rate_mps": raw_rate, "phase": phase, "first_pass_complete": self.phase_tracker.first_pass_complete})
        self.observation = observation
        self.base = base

    def _phase(self) -> str:
        if not self.phase_trace:
            raise V11RuntimeError("V11 phase trace is empty")
        return str(self.phase_trace[-1]["phase"])

    def _observation66d(self) -> np.ndarray:
        if self.base is None or len(self.history) != 10:
            raise V11RuntimeError("V11 requires exact ten-frame history")
        profile = compile_profile(str(self.config["semantic_compiler"]["profile"]))
        value = self.contract.compose(
            base_geometry=self.base,
            explicit_history=self.history_features.features(),
            intent_targets=profile["target_vector"],
            intent_weights=profile["weight_vector"],
            temporal_embedding=self.encoder.encode(np.stack(tuple(self.history), axis=0)),
        )
        if value.shape != (66,) or not np.isfinite(value).all():
            raise V11RuntimeError("V11 66-D contract failed")
        return value

    def _trigger_ready(self) -> bool:
        return bool(len(self.history) == 10 and self._phase() == "re_entry" and self.phase_tracker.first_pass_complete and self.last_info.get("prediction_valid") is True and self.last_info.get("prediction_fallback") is not True)

    def _reference_step(self) -> tuple[Mapping[str, Any], bool]:
        if self.observation is None:
            raise V11RuntimeError("V11 reference action requested before reset")
        action = v9.finite_action(self.policy.get_deterministic_action(self.observation["observation_vector"]))
        observation, _ignored, terminated, truncated, info = self.env.step(action)
        self.total_steps += 1
        self.last_info = info
        self._observe(observation)
        if bool(info.get("backend_fallback_occurred", False)):
            raise V11RuntimeError("V11 backend fallback occurred")
        return info, bool(terminated or truncated or self.total_steps >= int(self.config["observability"]["max_high_level_steps"]))

    @staticmethod
    def _finite_vector(value: Any, name: str) -> list[float]:
        try:
            result = np.asarray(value, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError) as error:
            raise V11RuntimeError(f"V11 missing {name}") from error
        if result.shape != (3,) or not np.isfinite(result).all():
            raise V11RuntimeError(f"V11 invalid {name}")
        return result.tolist()

    @staticmethod
    def _guidance_vector(info: Mapping[str, Any]) -> list[float]:
        guidance = info.get("guidance_command")
        if not isinstance(guidance, Mapping):
            raise V11RuntimeError("V11 guidance command is missing")
        try:
            vector = np.asarray([float(guidance["nz_cmd"]) / 9.0, float(guidance["roll_rate_cmd"]) / 1.5, float(guidance["throttle_cmd"])], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as error:
            raise V11RuntimeError("V11 guidance command receipt is incomplete") from error
        if not np.isfinite(vector).all():
            raise V11RuntimeError("V11 guidance command receipt is non-finite")
        return vector.tolist()

    def _assert_raw_interface(self) -> None:
        generator = self.env.virtual_point_generator
        if generator is None or str(getattr(generator, "offset_frame", "")) != "world_neu" or str(getattr(generator, "action_semantics", "")) != "cartesian_offset":
            raise V11RuntimeError("V11 requires the frozen world-frame cartesian VPP action interface")
        expected = np.asarray(self.config["semantic_compiler"]["raw_action_ranges"], dtype=np.float64)
        observed = np.asarray([generator.d_long_range, generator.d_lat_range, generator.d_vert_range], dtype=np.float64)
        if expected.shape != (3, 2) or not np.allclose(expected, observed, rtol=0.0, atol=1e-12):
            raise V11RuntimeError("V11 raw VPP action ranges drifted")

    def run(self, scenario: Mapping[str, Any], seed: int, *, method: str, enabled_axes: Mapping[str, bool]) -> dict[str, Any]:
        self.history.clear()
        self.history_features.reset()
        self.observation = None
        self.base = None
        self.last_info = {}
        self.total_steps = 0
        self.phase_trace = []
        event = RawSIFirstPassConfig(**dict(self.config["observability"]["raw_si_first_pass_event"]))
        self.phase_tracker = RawSIReentryPhaseTracker(RawSIReentryPhaseConfig(first_pass=event))
        observation = self.env.reset(scenario=copy.deepcopy(dict(scenario)), seed=int(seed))
        if self.env._backend != "jsbsim":
            raise V11RuntimeError("V11 requires strict JSBSim")
        self._assert_raw_interface()
        self.env.set_runtime_specialist_context(specialist_key="head_on", specialist_profile=None, specialist_mode_name="v11_frozen_run_in", specialist_reason="v11_raw_si_reentry_run_in")
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
            return {"handoff_reached": False, "window_steps": 0, "contract_passed": not bool(self.last_info.get("backend_fallback_occurred", False)), "terminal_reason": terminal_reason, "ego_failure": is_ego_failure(terminal_reason), "phase_trace": self.phase_trace}

        self.env.set_runtime_specialist_context(specialist_key="head_on", specialist_profile=str(self.config["semantic_compiler"]["profile"]), specialist_mode_name="v11_relative_frame_semantic", specialist_reason=method)
        handoff_step = self.total_steps
        handoff_observation = self._observation66d()
        handoff_hash = _stable_hash({"base": self.base, "history": tuple(self.history), "observation": self.observation["observation_vector"] if self.observation else None, "phase": self._phase()})
        window: list[dict[str, Any]] = []
        target_steps = int(self.config["observability"]["local_window_steps"])
        for _ in range(target_steps):
            if self.observation is None or self.base is None:
                raise V11RuntimeError("V11 lost observation during local window")
            before_loss = float(intent_loss(self.observation, str(self.config["semantic_compiler"]["profile"]), clip=2.0))
            reference_action = v9.finite_action(self.policy.get_deterministic_action(self.observation["observation_vector"]))
            own_state = self.last_info.get("own_state") if isinstance(self.last_info.get("own_state"), Mapping) else None
            target_state = self.last_info.get("target_state") if isinstance(self.last_info.get("target_state"), Mapping) else None
            if own_state is None or target_state is None:
                raise V11RuntimeError("V11 handoff lacks own/target physical state")
            receipt = self.compiler.compile(
                reference_action=reference_action,
                base_geometry=self.base,
                own_state=own_state,
                target_state=target_state,
                raw_action_ranges=self.config["semantic_compiler"]["raw_action_ranges"],
                profile_name=str(self.config["semantic_compiler"]["profile"]),
                enabled_axes=enabled_axes,
            )
            if method == REFERENCE_METHOD:
                receipt["applied_action"] = reference_action.tolist()
                receipt["actual_semantic_world_delta_m"] = [0.0, 0.0, 0.0]
                receipt["actual_semantic_delta_norm_m"] = 0.0
            observation, _ignored, terminated, truncated, info = self.env.step(np.asarray(receipt["applied_action"], dtype=np.float32))
            self.total_steps += 1
            self.last_info = info
            self._observe(observation)
            if bool(info.get("backend_fallback_occurred", False)):
                raise V11RuntimeError("V11 backend fallback occurred in local window")
            own_after = info.get("own_state") if isinstance(info.get("own_state"), Mapping) else {}
            window.append({
                "semantic_receipt": receipt,
                "actual_semantic_delta_norm_m": float(receipt["actual_semantic_delta_norm_m"]),
                "phase_before": self.phase_trace[-2]["phase"],
                "observation_66d": handoff_observation.tolist() if not window else self._observation66d().tolist(),
                "before_intent_loss": before_loss,
                "after_intent_loss": float(intent_loss(self.observation, str(self.config["semantic_compiler"]["profile"]), clip=2.0)),
                "after_range_rate_mps": float(self.base[1] * RANGE_RATE_REFERENCE_MPS),
                "env_vpp_world_offset_m": self._finite_vector(info.get("vp_world_offset"), "vp_world_offset"),
                "guidance_command_normalized": self._guidance_vector(info),
                "nz_cmd": float(info.get("nz_cmd")),
                "actual_nz_g": float(own_after.get("nz_g", info.get("nz_g"))),
                "saturated": any(bool(info.get(key, False)) for key in ("nz_saturated", "roll_rate_saturated", "throttle_saturated")),
            })
            done = bool(terminated or truncated or self.total_steps >= int(self.config["observability"]["max_high_level_steps"]))
            terminal_reason = str(info.get("reason") or "timeout")
            if done:
                break
        while not done and self.total_steps < int(self.config["observability"]["max_high_level_steps"]):
            info, done = self._reference_step()
            terminal_reason = str(info.get("reason") or "timeout")
        finite_window = all(
            len(step["observation_66d"]) == 66
            and np.isfinite(np.asarray(step["observation_66d"], dtype=np.float64)).all()
            and np.isfinite(np.asarray(step["env_vpp_world_offset_m"], dtype=np.float64)).all()
            and np.isfinite(np.asarray(step["guidance_command_normalized"], dtype=np.float64)).all()
            and math.isfinite(float(step["after_range_rate_mps"]))
            for step in window
        )
        return {
            "handoff_reached": True,
            "handoff_state_hash": handoff_hash,
            "handoff_step": handoff_step,
            "handoff_observation_66d": handoff_observation.tolist(),
            "window_steps": len(window),
            "intent_loss_auc20": float(np.mean([step["after_intent_loss"] for step in window])) if len(window) == target_steps else None,
            "terminal_reason": terminal_reason,
            "ego_failure": is_ego_failure(terminal_reason),
            "contract_passed": finite_window and not bool(self.last_info.get("backend_fallback_occurred", False)),
            "window": window,
            "phase_trace": self.phase_trace,
        }


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
        raise V11RuntimeError("V11 requires a clean worktree")
    methods = _methods(config)
    registry = _load(_resolve(config["inputs"]["runtime_registry"]))
    output = _resolve(config["outputs"]["root"])
    output.mkdir(parents=True)
    _write_json(output / "authorized_config.json", {"config": config, "config_sha256": _hash(args.config)})
    records: list[dict[str, Any]] = []
    compiler = _compiler(config)
    try:
        for method, axes in methods.items():
            for opponent in OPPONENTS:
                for scenario in manifest["scenarios"]:
                    env = build_record_only_env(config, registry, opponent)
                    policy = engine._specialist(registry, "frozen_fixed_head_on")
                    encoder = FrozenP3Encoder(Path(config["inputs"]["frozen_p3_encoder"]), config["inputs"]["frozen_p3_encoder_sha256"])
                    episode = SemanticTransferEpisode(env, policy, encoder, config, compiler)
                    try:
                        result = episode.run(scenario, int(scenario["metadata"]["scenario_seed"]), method=method, enabled_axes=axes)
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
            min_semantic_delta_norm_m=float(gates["min_semantic_delta_norm_m"]),
            min_vpp_delta_norm_m=float(gates["min_env_vpp_delta_norm_m"]),
            min_guidance_delta_norm=float(gates["min_guidance_delta_norm"]),
            min_range_rate_delta_mps=float(gates["min_range_rate_delta_mps"]),
            practical_intent_loss_delta=float(gates["practical_intent_loss_delta_vs_reference_max"]),
            safety_delta_max=float(gates["ego_crash_oob_delta_vs_reference_max"]),
        )
        gate = {"source_id": SOURCE_ID, **summary}
        _write_json(output / "semantic_transfer_gate.json", gate)
        _write_json(output / "run_manifest.json", {"source_id": SOURCE_ID, "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "config_sha256": _hash(args.config), "manifest_sha256": _hash(_resolve(config["inputs"]["manifest"])), "records": len(records), "training": "prohibited", "reward_recording": "prohibited"})
        print(json.dumps(gate, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        _write_json(output / "failure.json", {"source_id": SOURCE_ID, "error": repr(error), "completed_records": len(records)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
