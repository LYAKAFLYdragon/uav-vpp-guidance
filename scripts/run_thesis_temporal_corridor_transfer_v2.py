#!/usr/bin/env python3
"""Run the preregistered V2 non-learning temporal-corridor transfer probe."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Optional

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import run_thesis_five_state_reentry_v16_dynamic_reference_path as v16
from uav_vpp_guidance.envs.observation import compute_relative_geometry
from uav_vpp_guidance.evaluation.thesis_temporal_corridor_transfer_v2 import (
    METHODS,
    OPPONENTS,
    REFERENCE_METHOD,
    SOURCE_ID,
    is_ego_failure,
    summarize_records,
)
from uav_vpp_guidance.training import thesis_neutral_postmerge_reentry_recovery_pilot as engine
from uav_vpp_guidance.training.thesis_shared_skill_geometry import FrozenP3Encoder
from uav_vpp_guidance.virtual_point.temporal_corridor import (
    TemporalCorridorConfig,
    compile_temporal_corridor_action,
)


class TemporalCorridorTransferV2Error(RuntimeError):
    """Raised when V2 would leave its preregistered runtime contract."""


def _load(path: Path) -> dict[str, Any]:
    return v16._load(path)


def _resolve(value: str | Path) -> Path:
    return v16._resolve(value)


def _hash(path: Path) -> str:
    return v16._hash(path)


def _write(path: Path, value: Mapping[str, Any]) -> None:
    v16._write_json(path, value)


def _config(path: Path) -> dict[str, Any]:
    overlay = _load(path)
    if "base_config" not in overlay:
        return overlay
    base = _resolve(str(overlay["base_config"]))
    if _hash(base) != str(overlay.get("base_config_sha256") or ""):
        raise TemporalCorridorTransferV2Error("V2 authorization base hash mismatch")
    result = copy.deepcopy(_load(base))
    result["authorization"] = copy.deepcopy(overlay.get("authorization", {}))
    result["_execution"] = copy.deepcopy(overlay.get("execution", {}))
    return result


def environment_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the Source config after binding its runtime to the frozen input."""

    inputs = config.get("inputs")
    execution = config.get("execution")
    if not isinstance(inputs, Mapping) or not isinstance(execution, Mapping):
        raise TemporalCorridorTransferV2Error("V2 requires explicit inputs and execution runtime mappings")
    runtime_value = execution.get("runtime_template")
    input_value = inputs.get("runtime_template")
    if not runtime_value or not input_value:
        raise TemporalCorridorTransferV2Error("V2 execution.runtime_template is required before environment construction")
    runtime_path = _resolve(str(runtime_value))
    input_path = _resolve(str(input_value))
    expected_hash = str(inputs.get("runtime_template_sha256") or "")
    if (
        runtime_path != input_path
        or not runtime_path.is_file()
        or _hash(runtime_path) != expected_hash
        or str(execution.get("runtime_template_sha256") or "") != expected_hash
    ):
        raise TemporalCorridorTransferV2Error("V2 execution runtime mapping does not match the frozen input")
    return config


def build_record_only_env(config: Mapping[str, Any], registry: Mapping[str, Any], opponent: str):
    """Build V16's strict JSBSim chain from the explicitly validated mapping."""

    return v16.v15.v14.build_record_only_env(environment_config(config), registry, opponent)


def validate_design(path: Path) -> dict[str, Any]:
    design = _config(path)
    if design.get("source_id") != SOURCE_ID:
        raise TemporalCorridorTransferV2Error("V2 source ID drifted")
    if tuple(design.get("opponents", {}).get("order", ())) != OPPONENTS or design["opponents"].get("pooled_gate") != "prohibited":
        raise TemporalCorridorTransferV2Error("V2 opponent contract drifted")
    if tuple(design.get("methods", {})) != METHODS:
        raise TemporalCorridorTransferV2Error("V2 method contract drifted")
    for key in (
        "training_permitted",
        "tuning_permitted",
        "reward_change_permitted",
        "candidate_policy_loading_permitted",
        "high_level_ppo_training_permitted",
        "four_skill_training_permitted",
        "combat_finetune_permitted",
    ):
        if design["authorization"].get(key) is not False:
            raise TemporalCorridorTransferV2Error(f"V2 must prohibit {key}")
    environment_config(design)
    for key in ("manifest", "runtime_template", "runtime_registry", "frozen_p3_encoder"):
        frozen = _resolve(str(design["inputs"][key]))
        expected = str(design["inputs"].get(key + "_sha256") or "")
        if not frozen.is_file() or _hash(frozen) != expected:
            raise TemporalCorridorTransferV2Error(f"V2 frozen input mismatch: {key}")
    manifest = _load(_resolve(str(design["inputs"]["manifest"])))
    if manifest.get("source_id") != SOURCE_ID + "-MANIFEST12" or len(manifest.get("scenarios", [])) != 12:
        raise TemporalCorridorTransferV2Error("V2 manifest contract drifted")
    if manifest.get("protocol", {}).get("training_permitted") is not False:
        raise TemporalCorridorTransferV2Error("V2 manifest must remain non-learning")
    output = _resolve(str(design["outputs"]["root"]))
    if not output.is_absolute() or output.exists():
        raise TemporalCorridorTransferV2Error("V2 output root must be absolute and absent")
    return {
        "source_id": SOURCE_ID,
        "planned_records": len(METHODS) * len(OPPONENTS) * len(manifest["scenarios"]),
        "output_root_absent": True,
        "runtime_mapping_passed": True,
        "execution_permitted": bool(design["authorization"].get("execution_permitted")),
    }


def _authorize(config: Mapping[str, Any], expected_records: int) -> None:
    execution = config.get("_execution")
    if (
        config["authorization"].get("execution_permitted") is not True
        or not isinstance(execution, Mapping)
        or int(execution.get("expected_records", -1)) != expected_records
    ):
        raise TemporalCorridorTransferV2Error("V2 execution is not authorized")
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if sha != str(execution.get("required_implementation_git_sha") or ""):
        raise TemporalCorridorTransferV2Error("V2 implementation SHA is not exactly authorized")
    files = execution.get("authorized_code_files")
    if not isinstance(files, list) or not files:
        raise TemporalCorridorTransferV2Error("V2 authorized file list is missing")
    for item in files:
        source = _resolve(str(item.get("path") or ""))
        if not source.is_file() or _hash(source) != str(item.get("sha256") or ""):
            raise TemporalCorridorTransferV2Error(f"V2 authorized code mismatch: {source}")


class Episode(v16.DynamicReferencePathEpisode):
    """V16's run-in/phase/telemetry chain with only the post-handoff compiler replaced."""

    def _context(self):
        own, target = self.env._get_current_states(noisy=True)
        target = self.env._task_get_target_state(target, own)
        own, target = self.env._task_pre_step(own, target)
        geometry = compute_relative_geometry(own, target)
        if not self.env._sync_prediction_buffer_for_current_state(own, target, geometry):
            raise TemporalCorridorTransferV2Error("V2 predictor synchronization failed")
        predicted, _, info = self.env.trajectory_predictor_adapter.predict(target)
        predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
        if predicted.shape != (3,) or not np.isfinite(predicted).all() or bool(info.get("fallback", False)):
            raise TemporalCorridorTransferV2Error("V2 prediction fallback")
        return own, target, predicted

    def run(
        self,
        scenario: Mapping[str, Any],
        seed: int,
        method: str,
        compiler: Optional[TemporalCorridorConfig],
    ) -> dict[str, Any]:
        self.history.clear()
        self.history_features.reset()
        self.observation = None
        self.base = None
        self.last_info = {}
        self.total_steps = 0
        self.phase_trace = []
        event = v16.v15.v14.v11.RawSIFirstPassConfig(
            **dict(self.config["observability"]["raw_si_first_pass_event"])
        )
        self.phase_tracker = v16.v15.v14.v11.RawSIReentryPhaseTracker(
            v16.v15.v14.v11.RawSIReentryPhaseConfig(first_pass=event)
        )
        observation = self.env.reset(scenario=copy.deepcopy(dict(scenario)), seed=int(seed))
        if self.env._backend != "jsbsim":
            raise TemporalCorridorTransferV2Error("V2 requires strict JSBSim")
        self._assert_path_interface()
        self.env.set_runtime_specialist_context(
            specialist_key="head_on",
            specialist_profile=None,
            specialist_mode_name="tc_v2_frozen_run_in",
            specialist_reason="tc_v2_run_in",
        )
        self.observation_schema = copy.deepcopy(observation["observation_schema"])
        self._observe(observation)
        done = False
        terminal = "horizon_before_handoff"
        while not done and self.total_steps < int(self.config["observability"]["max_high_level_steps"]):
            if self._trigger_ready():
                break
            info, done = self._reference_step()
            terminal = str(info.get("reason") or "timeout")
        if not self._trigger_ready():
            return {
                "handoff_reached": False,
                "window_steps": 0,
                "contract_passed": not bool(self.last_info.get("backend_fallback_occurred", False)),
                "terminal_reason": terminal,
                "ego_failure": is_ego_failure(terminal),
                "phase_trace": self.phase_trace,
            }

        handoff_hash = v16.v15.v14._stable_hash(
            {"base": self.base, "history": tuple(self.history), "phase": self._phase()}
        )
        previous = np.zeros(3)
        window: list[dict[str, Any]] = []
        self.env.set_runtime_specialist_context(
            specialist_key="head_on",
            specialist_profile="temporal_corridor" if compiler else None,
            specialist_mode_name="tc_v2_window",
            specialist_reason=method,
        )
        for local_step in range(int(self.config["observability"]["local_window_steps"])):
            profile = "range_extension" if local_step < 10 else "reentry_preparation"
            _, errors, weighted = self._snapshot(profile)
            reference = v16.v15.v14.v11.v9.finite_action(
                self.policy.get_deterministic_action(self.observation["observation_vector"])
            )
            receipt = None
            if compiler is None:
                action = reference
            else:
                own, _target, predicted = self._context()
                receipt = compile_temporal_corridor_action(
                    own_position_neu=own["position_neu"],
                    predicted_target_position_neu=predicted,
                    phase_step=local_step,
                    previous_action=previous,
                    raw_action_ranges=self.config["raw_action_ranges"],
                    config=compiler,
                )
                action = np.asarray(receipt["applied_action"])
                previous = action.copy()
            observation, _ignored, terminated, truncated, info = self.env.step(np.asarray(action, dtype=np.float32))
            self.total_steps += 1
            self.last_info = info
            self._observe(observation)
            if bool(info.get("backend_fallback_occurred", False)) or info.get("prediction_fallback") is True:
                raise TemporalCorridorTransferV2Error("V2 runtime fallback")
            _after, after_errors, after_weighted = self._snapshot(profile)
            own_after = info.get("own_state", {})
            window.append(
                {
                    "local_step": local_step,
                    "objective_profile": profile,
                    "before_component_errors": errors,
                    "after_component_errors": after_errors,
                    "before_weighted_error": weighted,
                    "after_weighted_error": after_weighted,
                    "temporal_corridor_receipt": receipt,
                    "applied_action": np.asarray(action).tolist(),
                    "env_vpp_world_offset_m": self._finite_vector(info.get("vp_world_offset"), "vp_world_offset"),
                    "guidance_command_normalized": self._guidance_vector(info),
                    "actual_nz_g": float(own_after.get("nz_g", info.get("nz_g"))),
                    "prediction_valid": info.get("prediction_valid") is True,
                    "prediction_fallback": bool(info.get("prediction_fallback", False)),
                }
            )
            done = bool(
                terminated
                or truncated
                or self.total_steps >= int(self.config["observability"]["max_high_level_steps"])
            )
            terminal = str(info.get("reason") or "timeout")
            if done:
                break
        while not done and self.total_steps < int(self.config["observability"]["max_high_level_steps"]):
            info, done = self._reference_step()
            terminal = str(info.get("reason") or "timeout")
        finite = all(
            np.isfinite(np.asarray(step["after_component_errors"], dtype=float)).all()
            and np.isfinite(float(step["after_weighted_error"]))
            for step in window
        )
        return {
            "handoff_reached": True,
            "handoff_state_hash": handoff_hash,
            "window_steps": len(window),
            "terminal_reason": terminal,
            "ego_failure": is_ego_failure(terminal),
            "contract_passed": finite,
            "window": window,
            "phase_trace": self.phase_trace,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    design = _config(args.config)
    preflight = validate_design(args.config)
    if not args.execute:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0
    _authorize(design, int(preflight["planned_records"]))
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise TemporalCorridorTransferV2Error("V2 requires a clean worktree")

    manifest = _load(_resolve(str(design["inputs"]["manifest"])))
    registry = _load(_resolve(str(design["inputs"]["runtime_registry"])))
    output = _resolve(str(design["outputs"]["root"]))
    output.mkdir(parents=True)
    _write(output / "authorized_config.json", {"config": design, "config_sha256": _hash(args.config)})
    records: list[dict[str, Any]] = []
    try:
        for method in METHODS:
            compiler = None if method == REFERENCE_METHOD else TemporalCorridorConfig(**design["methods"][method]["compiler"])
            for opponent in OPPONENTS:
                for scenario in manifest["scenarios"]:
                    env = build_record_only_env(design, registry, opponent)
                    policy = engine._specialist(registry, "frozen_fixed_head_on")
                    encoder = FrozenP3Encoder(
                        Path(design["inputs"]["frozen_p3_encoder"]),
                        design["inputs"]["frozen_p3_encoder_sha256"],
                    )
                    episode = Episode(env, policy, encoder, design, v16._compiler(design))
                    try:
                        result = episode.run(
                            scenario,
                            int(scenario["metadata"]["scenario_seed"]),
                            method,
                            compiler,
                        )
                    finally:
                        episode.close()
                    record = {
                        "source_id": SOURCE_ID,
                        "method": method,
                        "opponent": opponent,
                        "pair_key": scenario["metadata"]["pair_key"],
                        **result,
                    }
                    records.append(record)
                    _write(output / "episodes" / method / opponent / f"{scenario['name']}.json", {"record": record})
        gate = summarize_records(
            records,
            scenarios_per_opponent=len(manifest["scenarios"]),
            window_steps=int(design["observability"]["local_window_steps"]),
            gates=design["gates"],
        )
        _write(output / "gate.json", {"source_id": SOURCE_ID, **gate})
        _write(
            output / "run_manifest.json",
            {
                "source_id": SOURCE_ID,
                "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "config_sha256": _hash(args.config),
                "manifest_sha256": _hash(_resolve(str(design["inputs"]["manifest"]))),
                "records": len(records),
                "training": "prohibited",
                "reward_recording": "prohibited",
                "authorized_code_files": design["_execution"]["authorized_code_files"],
            },
        )
        print(json.dumps({"source_id": SOURCE_ID, **gate}, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        _write(output / "failure.json", {"source_id": SOURCE_ID, "error": repr(error), "completed_records": len(records)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
