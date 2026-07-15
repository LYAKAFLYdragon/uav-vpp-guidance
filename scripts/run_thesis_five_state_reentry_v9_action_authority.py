#!/usr/bin/env python3
"""Run the V9 non-learning local 3-D VPP action-authority probe."""
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

from uav_vpp_guidance.evaluation.p4_v2_sampler_feasibility import classify_dynamic_state
from uav_vpp_guidance.evaluation.raw_si_first_pass_event import RawSIFirstPassConfig
from uav_vpp_guidance.evaluation.raw_si_reentry_phase import RawSIReentryPhaseConfig, RawSIReentryPhaseTracker
from uav_vpp_guidance.evaluation.single_motif_continuous_66d import project_base_observation
from uav_vpp_guidance.evaluation.thesis_five_state_raw_si_v5_record_only import _stable_hash, build_record_only_env
from uav_vpp_guidance.evaluation.thesis_reentry_v9_action_authority import (
    OPPONENTS,
    REFERENCE_METHOD,
    ActionAuthorityContractError,
    apply_perturbation,
    finite_action,
    is_ego_failure,
    summarize_records,
)
from uav_vpp_guidance.hierarchy.five_state_observation_contract import FiveStateObservationContract
from uav_vpp_guidance.hierarchy.shared_intent_profiles import compile_profile
from uav_vpp_guidance.hierarchy.temporal_geometry_features import TemporalGeometryFeatureExtractor
from uav_vpp_guidance.training import thesis_defext_rangeext_pilot as legacy
from uav_vpp_guidance.training import thesis_neutral_postmerge_reentry_recovery_pilot as engine
from uav_vpp_guidance.training.thesis_shared_skill_geometry import FrozenP3Encoder, intent_loss


SOURCE_ID = "THESIS-FIVE-STATE-REENTRY-VPP-ACTION-AUTHORITY-V9"
RANGE_REFERENCE_M = 5000.0
RANGE_RATE_REFERENCE_MPS = 200.0


class V9RuntimeError(RuntimeError):
    """Raised when a V9 runtime record leaves its frozen physical contract."""


def _load(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise V9RuntimeError(f"expected mapping: {path}")
    return value


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _load_runtime_config(path: Path) -> dict[str, Any]:
    """Resolve a minimal authorization overlay without permitting method drift."""
    overlay = _load(path)
    if "base_config" not in overlay:
        return overlay
    base_path = _resolve(overlay["base_config"])
    if _hash(base_path) != overlay.get("base_config_sha256"):
        raise V9RuntimeError("V9 authorization overlay base config hash mismatch")
    config = copy.deepcopy(_load(base_path))
    config["authorization"] = copy.deepcopy(overlay.get("authorization", {}))
    config["_authorization_execution"] = copy.deepcopy(overlay.get("execution", {}))
    config["_authorized_overlay_path"] = str(path)
    return config


def _require_execution_authorization(config: Mapping[str, Any]) -> None:
    if config.get("authorization", {}).get("execution_permitted") is not True:
        raise V9RuntimeError("V9 execution is not authorized")
    execution = config.get("_authorization_execution")
    if not isinstance(execution, Mapping):
        raise V9RuntimeError("V9 execution requires a hash-bound authorization overlay")
    required_sha = str(execution.get("required_implementation_git_sha") or "")
    if not required_sha or subprocess.run(["git", "cat-file", "-e", required_sha], cwd=ROOT, check=False).returncode:
        raise V9RuntimeError("V9 required implementation SHA is unavailable")
    files = execution.get("authorized_code_files")
    if not isinstance(files, list) or not files:
        raise V9RuntimeError("V9 authorization code whitelist is missing")
    manifest = _load(_resolve(config["inputs"]["manifest"]))
    expected_records = len(_methods(config)) * len(OPPONENTS) * len(manifest.get("scenarios", []))
    if int(execution.get("expected_records", -1)) != expected_records:
        raise V9RuntimeError("V9 authorization expected-record universe drifted")
    for item in files:
        if not isinstance(item, Mapping):
            raise V9RuntimeError("V9 authorization code whitelist is invalid")
        code_path = _resolve(str(item.get("path") or ""))
        if not code_path.is_file() or _hash(code_path) != item.get("sha256"):
            raise V9RuntimeError(f"V9 authorized code hash mismatch: {code_path}")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _methods(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    values = config.get("methods")
    if not isinstance(values, Mapping) or REFERENCE_METHOD not in values or len(values) != 7:
        raise V9RuntimeError("V9 requires exactly one reference and six axis perturbation methods")
    parsed = {str(name): np.asarray(delta, dtype=np.float32).reshape(-1) for name, delta in values.items()}
    expected = {
        REFERENCE_METHOD,
        "forward_positive", "forward_negative",
        "lateral_positive", "lateral_negative",
        "vertical_positive", "vertical_negative",
    }
    if set(parsed) != expected:
        raise V9RuntimeError("V9 method names drifted")
    if not np.allclose(parsed[REFERENCE_METHOD], np.zeros(3, dtype=np.float32)):
        raise V9RuntimeError("reference action delta must be zero")
    for name, delta in parsed.items():
        if delta.shape != (3,) or not np.isfinite(delta).all() or np.any(np.abs(delta) > 1.0):
            raise V9RuntimeError(f"invalid V9 action delta: {name}")
    return parsed


def validate_design(config_path: Path) -> dict[str, Any]:
    """Validate V9 inputs without instantiating JSBSim or creating outputs."""
    config = _load_runtime_config(config_path)
    if config.get("source_id") != SOURCE_ID:
        raise V9RuntimeError("V9 source ID drifted")
    if config.get("authorization", {}).get("training_permitted") is not False:
        raise V9RuntimeError("V9 must prohibit training")
    if config.get("authorization", {}).get("reward_change_permitted") is not False:
        raise V9RuntimeError("V9 must prohibit reward changes")
    if tuple(config.get("opponents", {}).get("order", ())) != OPPONENTS or config.get("opponents", {}).get("pooled_gate") != "prohibited":
        raise V9RuntimeError("V9 opponent contract drifted")
    methods = _methods(config)
    manifest_path = _resolve(config["inputs"]["manifest"])
    if _hash(manifest_path) != config["inputs"]["manifest_sha256"]:
        raise V9RuntimeError("V9 manifest hash mismatch")
    manifest = _load(manifest_path)
    if manifest.get("source_id") != SOURCE_ID + "-MANIFEST12" or len(manifest.get("scenarios", [])) != 12:
        raise V9RuntimeError("V9 requires its complete frozen 12-scenario manifest")
    if manifest.get("protocol", {}).get("training_permitted") is not False or manifest.get("protocol", {}).get("reward_recording") != "prohibited":
        raise V9RuntimeError("V9 manifest is not non-learning and reward-free")
    output = _resolve(config["outputs"]["root"])
    if not output.is_absolute() or output.exists():
        raise V9RuntimeError("V9 output root must be absolute and absent")
    for key in ("runtime_template", "runtime_registry", "frozen_p3_encoder"):
        path = _resolve(config["inputs"][key])
        hash_key = key + "_sha256"
        if not path.is_file() or _hash(path) != config["inputs"][hash_key]:
            raise V9RuntimeError(f"V9 frozen input mismatch: {key}")
    free_gb = shutil.disk_usage(output.parent).free / 1024 ** 3
    return {
        "source_id": SOURCE_ID,
        "mode": "nonlearning_action_authority_preflight",
        "manifest_scenarios": len(manifest["scenarios"]),
        "planned_records": len(methods) * len(OPPONENTS) * len(manifest["scenarios"]),
        "methods": list(methods),
        "free_disk_gb": round(free_gb, 3),
        "disk_gate_would_pass": free_gb >= float(config["observability"]["min_free_disk_gb"]),
        "output_root_absent": True,
    }


class ActionAuthorityEpisode:
    """One continuous physical episode with a fixed local perturbation window."""

    def __init__(self, env: Any, policy: Any, encoder: FrozenP3Encoder, config: Mapping[str, Any]) -> None:
        self.env = env
        self.policy = policy
        self.encoder = encoder
        self.config = config
        self.contract = FiveStateObservationContract()
        self.history: deque[np.ndarray] = deque(maxlen=10)
        self.history_features = TemporalGeometryFeatureExtractor(10)
        self.observation_schema: Mapping[str, Any] = {}
        self.observation: Mapping[str, Any] | None = None
        self.base: np.ndarray | None = None
        self.last_info: Mapping[str, Any] = {}
        self.total_steps = 0
        self.phase_trace: list[dict[str, Any]] = []
        event = RawSIFirstPassConfig(**dict(config["observability"]["raw_si_first_pass_event"]))
        self.phase_tracker = RawSIReentryPhaseTracker(RawSIReentryPhaseConfig(first_pass=event))

    def close(self) -> None:
        self.env.close()

    def _observe(self, observation: Mapping[str, Any]) -> None:
        base = project_base_observation(observation["observation_vector"], self.observation_schema)
        if base.shape != (16,) or not np.isfinite(base).all():
            raise V9RuntimeError("V9 observed invalid 16-D base vector")
        self.history.append(base.copy())
        self.history_features.update(legacy._history_frame(base))
        raw_range = float(base[0] * RANGE_REFERENCE_M)
        raw_rate = float(base[1] * RANGE_RATE_REFERENCE_MPS)
        phase = self.phase_tracker.update(raw_range, raw_rate, step_index=self.total_steps)
        self.phase_trace.append({
            "step": self.total_steps,
            "raw_range_m": raw_range,
            "raw_range_rate_mps": raw_rate,
            "phase": phase,
            "first_pass_complete": self.phase_tracker.first_pass_complete,
        })
        self.observation = observation
        self.base = base

    def _phase(self) -> str:
        if not self.phase_trace:
            raise V9RuntimeError("V9 phase trace is empty")
        return str(self.phase_trace[-1]["phase"])

    def _observation66d(self) -> np.ndarray:
        if self.base is None or len(self.history) != 10:
            raise V9RuntimeError("V9 requires exact ten-frame history")
        profile = compile_profile(str(self.config["observability"]["profile"]))
        result = self.contract.compose(
            base_geometry=self.base,
            explicit_history=self.history_features.features(),
            intent_targets=profile["target_vector"],
            intent_weights=profile["weight_vector"],
            temporal_embedding=self.encoder.encode(np.stack(tuple(self.history), axis=0)),
        )
        if result.shape != (66,) or not np.isfinite(result).all():
            raise V9RuntimeError("V9 66-D contract failed")
        return result

    def _trigger_ready(self) -> bool:
        return bool(
            len(self.history) == 10
            and self._phase() == "re_entry"
            and self.phase_tracker.first_pass_complete
            and self.last_info.get("prediction_valid") is True
            and self.last_info.get("prediction_fallback") is not True
        )

    def _step_reference(self) -> tuple[Mapping[str, Any], bool]:
        if self.observation is None:
            raise V9RuntimeError("V9 reference action requested before reset")
        action = finite_action(self.policy.get_deterministic_action(self.observation["observation_vector"]))
        observation, _ignored_reward, terminated, truncated, info = self.env.step(action)
        self.total_steps += 1
        self.last_info = info
        self._observe(observation)
        if bool(info.get("backend_fallback_occurred", False)):
            raise V9RuntimeError("V9 backend fallback occurred")
        return info, bool(terminated or truncated or self.total_steps >= int(self.config["observability"]["max_high_level_steps"]))

    def run(self, scenario: Mapping[str, Any], seed: int, *, method: str, delta: np.ndarray) -> dict[str, Any]:
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
            raise V9RuntimeError("V9 requires strict JSBSim")
        self.env.set_runtime_specialist_context(
            specialist_key="head_on", specialist_profile=None,
            specialist_mode_name="v9_frozen_run_in", specialist_reason="v9_raw_si_reentry_run_in",
        )
        self.observation_schema = copy.deepcopy(observation["observation_schema"])
        self._observe(observation)
        done = False
        terminal_reason = "horizon_before_handoff"
        while not done and self.total_steps < int(self.config["observability"]["max_high_level_steps"]):
            if self._trigger_ready():
                break
            info, done = self._step_reference()
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
            specialist_key="head_on", specialist_profile=str(self.config["observability"]["profile"]),
            specialist_mode_name="v9_action_authority", specialist_reason=method,
        )
        handoff_step = self.total_steps
        handoff_observation = self._observation66d()
        handoff_hash = _stable_hash({
            "base": self.base,
            "history": tuple(self.history),
            "observation": self.observation["observation_vector"] if self.observation else None,
            "phase": self._phase(),
        })
        window: list[dict[str, Any]] = []
        target_steps = int(self.config["observability"]["local_window_steps"])
        for _ in range(target_steps):
            if self.observation is None:
                raise V9RuntimeError("V9 lost observation during local window")
            before_loss = float(intent_loss(self.observation, str(self.config["observability"]["profile"]), clip=2.0))
            reference_action = finite_action(self.policy.get_deterministic_action(self.observation["observation_vector"]))
            receipt = apply_perturbation(reference_action, delta)
            observation, _ignored_reward, terminated, truncated, info = self.env.step(np.asarray(receipt["applied_action"], dtype=np.float32))
            self.total_steps += 1
            self.last_info = info
            self._observe(observation)
            if bool(info.get("backend_fallback_occurred", False)):
                raise V9RuntimeError("V9 backend fallback occurred in local window")
            own_state = info.get("own_state") if isinstance(info.get("own_state"), Mapping) else {}
            window.append({
                **receipt,
                "phase_before": self.phase_trace[-2]["phase"],
                "observation_66d": handoff_observation.tolist() if not window else self._observation66d().tolist(),
                "before_intent_loss": before_loss,
                "after_intent_loss": float(intent_loss(self.observation, str(self.config["observability"]["profile"]), clip=2.0)),
                "vp_forward_bias_m": float(info.get("vp_forward_bias_m", math.nan)),
                "vp_lateral_bias_m": float(info.get("vp_lateral_bias_m", math.nan)),
                "vp_vertical_bias_m": float(info.get("vp_vertical_bias_m", math.nan)),
                "nz_cmd": float(info.get("nz_cmd", math.nan)),
                "actual_nz_g": float(own_state.get("nz_g", info.get("nz_g", math.nan))),
                "ego_attack_aoa_deg": float(info.get("ego_attack_aoa_deg", math.nan)),
                "saturated": any(bool(info.get(key, False)) for key in ("nz_saturated", "roll_rate_saturated", "throttle_saturated")),
            })
            done = bool(terminated or truncated or self.total_steps >= int(self.config["observability"]["max_high_level_steps"]))
            terminal_reason = str(info.get("reason") or "timeout")
            if done:
                break
        while not done and self.total_steps < int(self.config["observability"]["max_high_level_steps"]):
            info, done = self._step_reference()
            terminal_reason = str(info.get("reason") or "timeout")
        finite_window = all(
            np.isfinite(np.asarray(step["observation_66d"], dtype=np.float64)).all()
            and math.isfinite(float(step["after_intent_loss"]))
            and all(math.isfinite(float(step[key])) for key in (
                "vp_forward_bias_m", "vp_lateral_bias_m", "vp_vertical_bias_m",
                "nz_cmd", "actual_nz_g", "ego_attack_aoa_deg",
            ))
            for step in window
        )
        return {
            "handoff_reached": True,
            "handoff_state_hash": handoff_hash,
            "handoff_step": handoff_step,
            "handoff_observation_66d": handoff_observation.tolist(),
            "window_steps": len(window),
            "intent_loss_auc20": float(np.mean([step["after_intent_loss"] for step in window])) if len(window) == target_steps else None,
            "mean_applied_delta_norm": float(np.mean([step["applied_delta_norm"] for step in window])) if window else 0.0,
            "terminal_reason": terminal_reason,
            "ego_failure": is_ego_failure(terminal_reason),
            "contract_passed": finite_window and not bool(self.last_info.get("backend_fallback_occurred", False)),
            "window": window,
            "phase_trace": self.phase_trace,
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
    config = _load_runtime_config(args.config)
    preflight = validate_design(args.config)
    if not args.execute:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0
    _require_execution_authorization(config)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise V9RuntimeError("V9 requires a clean worktree")

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
                    encoder = FrozenP3Encoder(
                        Path(config["inputs"]["frozen_p3_encoder"]), config["inputs"]["frozen_p3_encoder_sha256"],
                    )
                    episode = ActionAuthorityEpisode(env, policy, encoder, config)
                    try:
                        result = episode.run(scenario, int(scenario["metadata"]["scenario_seed"]), method=method, delta=delta)
                    finally:
                        episode.close()
                    row = _record(scenario, opponent, method, result)
                    records.append(row)
                    _write_json(output / "episodes" / method / opponent / f"{scenario['name']}.json", {"record": row})
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
        _write_json(output / "action_authority_gate.json", gate)
        _write_json(output / "run_manifest.json", {
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
        _write_json(output / "failure.json", {"source_id": SOURCE_ID, "error": repr(error), "completed_records": len(records)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
