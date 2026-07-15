#!/usr/bin/env python3
"""Collect V17's non-learning 74-D path-context sampler evidence."""

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

import run_thesis_five_state_reentry_v16_dynamic_reference_path as v16
from uav_vpp_guidance.evaluation.thesis_v17_path_context_preflight import (
    OPPONENTS,
    SOURCE_ID,
    is_ego_failure,
    summarize_records,
)
from uav_vpp_guidance.hierarchy.path_conditioned_observation_contract import (
    PathConditionedObservationContract,
    build_path_context,
)
from uav_vpp_guidance.hierarchy.shared_intent_profiles import compile_profile
from uav_vpp_guidance.training import thesis_neutral_postmerge_reentry_recovery_pilot as engine
from uav_vpp_guidance.training.thesis_shared_skill_geometry import FrozenP3Encoder
from uav_vpp_guidance.virtual_point.dynamic_reference_path import compile_path_anchor_action


class V17RuntimeError(RuntimeError):
    """Raised when the V17 collector would leave its frozen data contract."""


def _load(path: Path) -> dict[str, Any]:
    return v16._load(path)


def _resolve(value: str | Path) -> Path:
    return v16._resolve(value)


def _hash(path: Path) -> str:
    return v16._hash(path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    v16._write_json(path, value)


def _load_runtime_config(path: Path) -> dict[str, Any]:
    overlay = _load(path)
    if "base_config" not in overlay:
        return overlay
    base_path = _resolve(str(overlay["base_config"]))
    if _hash(base_path) != str(overlay.get("base_config_sha256") or ""):
        raise V17RuntimeError("V17 authorization overlay base config hash mismatch")
    config = copy.deepcopy(_load(base_path))
    config["authorization"] = copy.deepcopy(overlay.get("authorization", {}))
    config["_authorization_execution"] = copy.deepcopy(overlay.get("execution", {}))
    return config


def _path_config(config: Mapping[str, Any]):
    return v16.DynamicReferencePathConfig(
        escape_extent_m=float(config["path"]["escape_extent_m"]),
        connector_fraction=float(config["path"]["connector_fraction"]),
        opening_rate_start_mps=float(config["path"]["opening_rate_start_mps"]),
        opening_rate_full_mps=float(config["path"]["opening_rate_full_mps"]),
    )


def validate_design(config_path: Path) -> dict[str, Any]:
    config = _load_runtime_config(config_path)
    if config.get("source_id") != SOURCE_ID:
        raise V17RuntimeError("V17 source ID drifted")
    for key in (
        "training_permitted",
        "tuning_permitted",
        "reward_change_permitted",
        "candidate_policy_loading_permitted",
        "combat_finetune_permitted",
    ):
        if config.get("authorization", {}).get(key) is not False:
            raise V17RuntimeError(f"V17 must prohibit {key}")
    if tuple(config.get("opponents", {}).get("order", ())) != OPPONENTS or config["opponents"].get("pooled_gate") != "prohibited":
        raise V17RuntimeError("V17 opponent contract drifted")
    if int(config.get("contract", {}).get("observation_dim", -1)) != 74:
        raise V17RuntimeError("V17 requires the new exact 74-D contract")
    for key in ("training", "reward_recording", "checkpoint_fallback", "backend_fallback", "prediction_fallback"):
        if config["contract"].get(key) != "prohibited":
            raise V17RuntimeError(f"V17 contract drifted: {key}")
    if config["contract"].get("path_context_current_only") is not True:
        raise V17RuntimeError("V17 path context must be current-only")
    _path_config(config)
    manifest_path = _resolve(config["inputs"]["manifest"])
    if _hash(manifest_path) != str(config["inputs"]["manifest_sha256"]):
        raise V17RuntimeError("V17 manifest hash mismatch")
    manifest = _load(manifest_path)
    if manifest.get("source_id") != SOURCE_ID + "-MANIFEST24" or len(manifest.get("scenarios", [])) != 24:
        raise V17RuntimeError("V17 requires the complete 24-scenario manifest")
    if manifest.get("protocol", {}).get("training_permitted") is not False or manifest["protocol"].get("reward_recording") != "prohibited":
        raise V17RuntimeError("V17 manifest is not non-learning and reward-free")
    output = _resolve(config["outputs"]["root"])
    if not output.is_absolute() or output.exists():
        raise V17RuntimeError("V17 output root must be absolute and absent")
    for key in ("runtime_template", "runtime_registry", "frozen_p3_encoder", "path_context_contract"):
        value = _resolve(config["inputs"][key])
        if not value.is_file() or _hash(value) != str(config["inputs"][key + "_sha256"]):
            raise V17RuntimeError(f"V17 frozen input mismatch: {key}")
    free_gb = shutil.disk_usage(output.parent).free / 1024**3
    return {
        "source_id": SOURCE_ID,
        "mode": "nonlearning_path_context_sampler_preflight",
        "manifest_scenarios": len(manifest["scenarios"]),
        "planned_records": len(OPPONENTS) * len(manifest["scenarios"]),
        "observation_dim": 74,
        "free_disk_gb": round(free_gb, 3),
        "disk_gate_would_pass": free_gb >= float(config["observability"]["min_free_disk_gb"]),
        "output_root_absent": True,
    }


class PathContextEpisode(v16.DynamicReferencePathEpisode):
    """Fixed-head-on rollout that records, but never acts on, V17 context."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.v17_contract = PathConditionedObservationContract()

    def _observation74d(self, path_context: np.ndarray) -> np.ndarray:
        if self.base is None or len(self.history) != 10:
            raise V17RuntimeError("V17 requires exact ten-frame history")
        profile = compile_profile("range_extension")
        value = self.v17_contract.compose(
            base_geometry=self.base,
            explicit_history=self.history_features.features(),
            intent_targets=profile["target_vector"],
            intent_weights=profile["weight_vector"],
            temporal_embedding=self.encoder.encode(np.stack(tuple(self.history), axis=0)),
            path_context=path_context,
        )
        if value.shape != (74,) or not np.isfinite(value).all():
            raise V17RuntimeError("V17 74-D observation contract failed")
        return value

    def run(self, scenario: Mapping[str, Any], seed: int) -> dict[str, Any]:
        self.history.clear(); self.history_features.reset(); self.observation = None; self.base = None; self.last_info = {}; self.total_steps = 0; self.phase_trace = []
        event = v16.v15.v14.v11.RawSIFirstPassConfig(**dict(self.config["observability"]["raw_si_first_pass_event"]))
        self.phase_tracker = v16.v15.v14.v11.RawSIReentryPhaseTracker(v16.v15.v14.v11.RawSIReentryPhaseConfig(first_pass=event))
        observation = self.env.reset(scenario=copy.deepcopy(dict(scenario)), seed=int(seed))
        if self.env._backend != "jsbsim":
            raise V17RuntimeError("V17 requires strict JSBSim")
        self._assert_path_interface()
        self.env.set_runtime_specialist_context(specialist_key="head_on", specialist_profile=None, specialist_mode_name="v17_frozen_run_in", specialist_reason="v17_path_context_collector")
        self.observation_schema = copy.deepcopy(observation["observation_schema"])
        self._observe(observation)
        done = False; terminal_reason = "horizon_before_handoff"
        while not done and self.total_steps < int(self.config["observability"]["max_high_level_steps"]):
            if self._trigger_ready():
                break
            info, done = self._reference_step(); terminal_reason = str(info.get("reason") or "timeout")
        if not self._trigger_ready():
            return {"handoff_reached": False, "window_steps": 0, "contract_passed": not bool(self.last_info.get("backend_fallback_occurred", False)), "terminal_reason": terminal_reason, "ego_failure": is_ego_failure(terminal_reason), "phase_trace": self.phase_trace}
        handoff_hash = v16.v15.v14._stable_hash({"base": self.base, "history": tuple(self.history), "observation": self.observation["observation_vector"] if self.observation else None, "phase": self._phase()})
        window: list[dict[str, Any]] = []
        target_steps = int(self.config["observability"]["local_window_steps"])
        for local_step in range(target_steps):
            if self.observation is None or self.base is None:
                raise V17RuntimeError("V17 lost observation during collection window")
            current = self._current_path()
            requested_delta = np.asarray(current["selected_anchor_neu"], dtype=np.float64) - np.asarray(current["predicted_target_position_neu"], dtype=np.float64)
            context = build_path_context(requested_path_world_delta_m=requested_delta, path_progress=float(current["path_progress"]), selected_anchor_index=int(current["selected_anchor_index"]))
            observation74d = self._observation74d(context)
            action = v16.v15.v14.v11.v9.finite_action(self.policy.get_deterministic_action(self.observation["observation_vector"]))
            feasibility = compile_path_anchor_action(semantic_action=action, selected_anchor_neu=current["selected_anchor_neu"], generator_anchor_neu=current["predicted_target_position_neu"], raw_action_ranges=self.config["semantic_compiler"]["raw_action_ranges"], raw_action_offset_frame="world_neu", own_state=current["own_state"], target_state=current["target_state"])
            observation, _ignored, terminated, truncated, info = self.env.step(action)
            self.total_steps += 1; self.last_info = info; self._observe(observation)
            if bool(info.get("backend_fallback_occurred", False)):
                raise V17RuntimeError("V17 backend fallback occurred")
            own_after = info.get("own_state") if isinstance(info.get("own_state"), Mapping) else {}
            window.append({"local_step": local_step, "observation_74d": observation74d.tolist(), "path_context": context.tolist(), "path_context_finite": True, "raw_action": action.tolist(), "raw_action_finite": True, "requested_path_world_delta_m": requested_delta.tolist(), "path_progress": float(current["path_progress"]), "selected_anchor_index": int(current["selected_anchor_index"]), "prediction_valid": True, "prediction_fallback": False, "path_feasibility_receipt_counterfactual": feasibility, "env_vpp_world_offset_m": self._finite_vector(info.get("vp_world_offset"), "vp_world_offset"), "guidance_command_normalized": self._guidance_vector(info), "actual_nz_g": float(own_after.get("nz_g", info.get("nz_g"))), "saturated": any(bool(info.get(key, False)) for key in ("nz_saturated", "roll_rate_saturated", "throttle_saturated"))})
            done = bool(terminated or truncated or self.total_steps >= int(self.config["observability"]["max_high_level_steps"])); terminal_reason = str(info.get("reason") or "timeout")
            if done: break
        while not done and self.total_steps < int(self.config["observability"]["max_high_level_steps"]):
            info, done = self._reference_step(); terminal_reason = str(info.get("reason") or "timeout")
        finite = all(len(step["observation_74d"]) == 74 and np.isfinite(np.asarray(step["observation_74d"], dtype=np.float64)).all() and np.isfinite(np.asarray(step["raw_action"], dtype=np.float64)).all() and np.isfinite(np.asarray(step["env_vpp_world_offset_m"], dtype=np.float64)).all() and np.isfinite(np.asarray(step["guidance_command_normalized"], dtype=np.float64)).all() for step in window)
        return {"handoff_reached": True, "handoff_state_hash": handoff_hash, "window_steps": len(window), "terminal_reason": terminal_reason, "ego_failure": is_ego_failure(terminal_reason), "contract_passed": finite and not bool(self.last_info.get("backend_fallback_occurred", False)), "window": window, "phase_trace": self.phase_trace}


def _require_execution_authorization(config: Mapping[str, Any], scenario_count: int) -> None:
    execution = config.get("_authorization_execution")
    if config.get("authorization", {}).get("execution_permitted") is not True or not isinstance(execution, Mapping):
        raise V17RuntimeError("V17 execution is not authorized")
    if int(execution.get("expected_records", -1)) != len(OPPONENTS) * scenario_count:
        raise V17RuntimeError("V17 authorization record universe drifted")
    required_sha = str(execution.get("required_implementation_git_sha") or "")
    if not required_sha or subprocess.run(["git", "merge-base", "--is-ancestor", required_sha, "HEAD"], cwd=ROOT, check=False).returncode:
        raise V17RuntimeError("V17 implementation SHA is not an ancestor of HEAD")
    files = execution.get("authorized_code_files")
    if not isinstance(files, list) or not files:
        raise V17RuntimeError("V17 authorization file whitelist is missing")
    for item in files:
        path = _resolve(str(item.get("path") or ""))
        if not path.is_file() or _hash(path) != str(item.get("sha256") or ""):
            raise V17RuntimeError(f"V17 authorized code hash mismatch: {path}")


def _record(scenario: Mapping[str, Any], opponent: str, result: Mapping[str, Any]) -> dict[str, Any]:
    meta = scenario["metadata"]
    return {"source_id": SOURCE_ID, "opponent": opponent, "pair_key": meta["pair_key"], "mirror_sign": meta["mirror_sign"], "scenario_signature": meta["scenario_signature"], **dict(result)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--execute", action="store_true"); args = parser.parse_args()
    config = _load_runtime_config(args.config); preflight = validate_design(args.config)
    if not args.execute:
        print(json.dumps(preflight, ensure_ascii=False, indent=2)); return 0
    if not preflight["disk_gate_would_pass"]: raise V17RuntimeError("V17 free-disk gate is not satisfied")
    manifest = _load(_resolve(config["inputs"]["manifest"])); _require_execution_authorization(config, len(manifest["scenarios"]))
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip(): raise V17RuntimeError("V17 requires a clean worktree")
    registry = _load(_resolve(config["inputs"]["runtime_registry"])); output = _resolve(config["outputs"]["root"]); output.mkdir(parents=True)
    _write_json(output / "authorized_config.json", {"config": config, "config_sha256": _hash(args.config)})
    records: list[dict[str, Any]] = []
    try:
        for opponent in OPPONENTS:
            for scenario in manifest["scenarios"]:
                env = v16.v15.v14.build_record_only_env(config, registry, opponent); policy = engine._specialist(registry, "frozen_fixed_head_on"); encoder = FrozenP3Encoder(Path(config["inputs"]["frozen_p3_encoder"]), config["inputs"]["frozen_p3_encoder_sha256"]); episode = PathContextEpisode(env, policy, encoder, config, v16._compiler(config))
                try: result = episode.run(scenario, int(scenario["metadata"]["scenario_seed"]))
                finally: episode.close()
                row = _record(scenario, opponent, result); records.append(row); _write_json(output / "episodes" / opponent / f"{scenario['name']}.json", {"record": row})
        gates = config["gates"]; summary = summarize_records(records, scenarios_per_opponent=len(manifest["scenarios"]), min_qualifying_episodes=int(gates["min_qualifying_episodes_per_opponent"]), min_valid_steps=int(gates["min_valid_path_context_steps_per_opponent"]), min_mirrors=int(gates["min_mirrors_per_opponent"]), min_signatures=int(gates["min_signatures_per_opponent"]))
        _write_json(output / "path_context_preflight_gate.json", {"source_id": SOURCE_ID, **summary})
        _write_json(output / "run_manifest.json", {"source_id": SOURCE_ID, "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "config_sha256": _hash(args.config), "manifest_sha256": _hash(_resolve(config["inputs"]["manifest"])), "records": len(records), "training": "prohibited", "reward_recording": "prohibited", "authorized_code_files": copy.deepcopy(config.get("_authorization_execution", {}).get("authorized_code_files", []))})
        print(json.dumps({"source_id": SOURCE_ID, **summary}, ensure_ascii=False, indent=2)); return 0
    except Exception as error:
        _write_json(output / "failure.json", {"source_id": SOURCE_ID, "error": repr(error), "completed_records": len(records)}); raise


if __name__ == "__main__": raise SystemExit(main())
