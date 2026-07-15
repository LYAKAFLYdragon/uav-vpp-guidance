"""Reward-free continuous collector for the V5 five-state observability run.

The fixed head-on specialist remains the only controller.  The 66-D vector is
recorded as a data-contract probe, not supplied to that legacy specialist.
"""

from __future__ import annotations

from collections import deque
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.evaluation.p4_v2_sampler_feasibility import classify_dynamic_state
from uav_vpp_guidance.evaluation.raw_si_first_pass_event import (
    RawSIFirstPassConfig,
    RawSIFirstPassTracker,
    replay_raw_si_first_pass,
)
from uav_vpp_guidance.evaluation.single_motif_continuous_66d import project_base_observation
from uav_vpp_guidance.hierarchy.five_state_observation_contract import FiveStateObservationContract
from uav_vpp_guidance.hierarchy.shared_intent_profiles import compile_profile
from uav_vpp_guidance.hierarchy.specialist_policy import FrozenSpecialistPolicy
from uav_vpp_guidance.hierarchy.temporal_geometry_features import TemporalGeometryFeatureExtractor
from uav_vpp_guidance.training import thesis_defext_rangeext_pilot as legacy
from uav_vpp_guidance.training import thesis_neutral_postmerge_reentry_recovery_pilot as neutral_engine
from uav_vpp_guidance.training.thesis_shared_skill_geometry import FrozenP3Encoder, PhaseTracker


SOURCE_ID = "THESIS-FIVE-STATE-RAW-SI-PHASE-OBSERVABILITY-V5"
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")


class FiveStateRawSIV5RecordError(RuntimeError):
    """Raised when a V5 record would leave the frozen observability contract."""


def _finite_action(action: Any) -> np.ndarray:
    result = np.asarray(action, dtype=np.float32).reshape(-1)
    if result.shape != (3,) or not np.isfinite(result).all() or np.any(np.abs(result) > 1.0 + 1e-6):
        raise FiveStateRawSIV5RecordError("V5 action is not a finite normalized 3-D VPP action")
    return result


def _angle_deg(sine: float, cosine: float) -> float:
    return abs(math.degrees(math.atan2(float(sine), float(cosine))))


def _base_map(base: np.ndarray) -> dict[str, float]:
    return {name: float(base[index]) for index, name in enumerate(legacy.BASE_GEOMETRY_FEATURES)}


def _history_frame(base: np.ndarray) -> dict[str, float]:
    values = _base_map(base)
    return {
        "range_rate_mps": values["range_rate_mps"],
        "aa_deg": _angle_deg(values["aa_sin"], values["aa_cos"]),
        "ata_deg": _angle_deg(values["ata_sin"], values["ata_cos"]),
        "specific_energy_height_m": values["own_altitude"] + values["own_speed"] ** 2 / (2.0 * 9.80665),
        "altitude_m": values["own_altitude"],
    }


def _canonical(value: Any) -> Any:
    """Make telemetry hashable without changing the recorded physical values."""

    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, np.ndarray):
        return _canonical(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if math.isnan(number):
            return "NaN"
        if math.isinf(number):
            return "Infinity" if number > 0.0 else "-Infinity"
        return number
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (str, bool)) or value is None:
        return value
    return repr(value)


def _stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(_canonical(payload), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_record_only_env(
    config: Mapping[str, Any], registry: Mapping[str, Any], opponent_id: str
) -> CloseRangeTrackingEnv:
    """Create the strict-JSBSim chain without loading a candidate policy."""

    if opponent_id not in OPPONENTS:
        raise FiveStateRawSIV5RecordError("V5 opponent is outside the frozen registry")
    runtime = neutral_engine._runtime(config)
    base = copy.deepcopy(neutral_engine.load_yaml(legacy.BASE_ENV_CONFIG))
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
    base["opponent_stage"] = opponent_id
    base["attack_zone"] = {"enabled": True, "initial_hp": 100.0, "damage_per_step": 0.5,
                           "close_range_max_km": 3.0, "close_range_max_aoa_deg": 60.0}
    entry = registry["opponents"][opponent_id]
    return CloseRangeTrackingEnv(
        base,
        opponent_policy=neutral_engine._instantiate_opponent(entry),
        opponent_config={"stage": opponent_id, **copy.deepcopy(entry)},
    )


class FiveStateRawSIV5Episode:
    """One legal reset and continuous fixed-head-on rollout with a V5 ledger."""

    def __init__(
        self,
        env: CloseRangeTrackingEnv,
        fixed_head_on: FrozenSpecialistPolicy,
        encoder: FrozenP3Encoder,
        config: Mapping[str, Any],
    ) -> None:
        self.env = env
        self.fixed_head_on = fixed_head_on
        self.encoder = encoder
        self.config = config
        self.contract = FiveStateObservationContract()
        self.history: deque[np.ndarray] = deque(maxlen=10)
        self.history_features = TemporalGeometryFeatureExtractor(10)
        self.phase_tracker = PhaseTracker()
        event = config["observability"]["raw_si_first_pass_event"]
        self.raw_tracker = RawSIFirstPassTracker(RawSIFirstPassConfig(**event))
        self.profile = compile_profile(str(config["observability"]["reference_profile"]))
        self.observation_schema: Mapping[str, Any] = {}
        self.observation: Mapping[str, Any] | None = None
        self.base: np.ndarray | None = None
        self.phase = "pre_merge"
        self.dynamic_state = "unknown"
        self.total_steps = 0
        self.last_info: Mapping[str, Any] = {}
        self.steps: list[dict[str, Any]] = []

    def _observe(self, observation: Mapping[str, Any]) -> None:
        base = project_base_observation(observation["observation_vector"], self.observation_schema)
        if base.shape != (16,) or not np.isfinite(base).all():
            raise FiveStateRawSIV5RecordError("V5 base observation is not finite 16-D")
        self.history.append(base.copy())
        self.history_features.update(_history_frame(base))
        values = _base_map(base)
        self.phase = self.phase_tracker.update(values["range_m"], values["range_rate_mps"])
        self.dynamic_state = classify_dynamic_state(
            _angle_deg(values["ata_sin"], values["ata_cos"]),
            _angle_deg(values["aa_sin"], values["aa_cos"]),
        )
        if self.dynamic_state == "unknown":
            raise FiveStateRawSIV5RecordError("V5 dynamic taxonomy became unknown")
        self.observation = observation
        self.base = base

    def _reference_observation(self) -> np.ndarray | None:
        if self.base is None:
            raise FiveStateRawSIV5RecordError("V5 66-D observation requested before reset")
        if len(self.history) < 10:
            return None
        if len(self.history) != 10:
            raise FiveStateRawSIV5RecordError("V5 history length drifted")
        result = self.contract.compose(
            base_geometry=self.base,
            explicit_history=self.history_features.features(),
            intent_targets=self.profile["target_vector"],
            intent_weights=self.profile["weight_vector"],
            temporal_embedding=self.encoder.encode(np.stack(tuple(self.history), axis=0)),
        )
        if result.shape != (66,) or not np.isfinite(result).all():
            raise FiveStateRawSIV5RecordError("V5 reference 66-D observation is invalid")
        return result

    def run(self, scenario: Mapping[str, Any], seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
        self.history.clear()
        self.history_features.reset()
        self.phase_tracker = PhaseTracker()
        event = self.config["observability"]["raw_si_first_pass_event"]
        self.raw_tracker = RawSIFirstPassTracker(RawSIFirstPassConfig(**event))
        self.total_steps = 0
        self.last_info = {}
        self.steps = []
        observation = self.env.reset(scenario=copy.deepcopy(dict(scenario)), seed=int(seed))
        if self.env._backend != "jsbsim":
            raise FiveStateRawSIV5RecordError("V5 strict JSBSim backend was not selected")
        self.env.set_runtime_specialist_context(
            specialist_key="head_on",
            specialist_profile=None,
            specialist_mode_name="frozen_fixed_head_on",
            specialist_reason="v5_raw_si_observability_reference_controller",
        )
        self.observation_schema = copy.deepcopy(observation["observation_schema"])
        self._observe(observation)
        terminal_reason = "horizon"
        while self.total_steps < self.env.max_steps:
            if self.observation is None or self.base is None:
                raise FiveStateRawSIV5RecordError("V5 lost its continuous observation")
            pre_base = self.base.copy()
            pre_history = tuple(item.copy() for item in self.history)
            pre_phase, pre_state = self.phase, self.dynamic_state
            values = _base_map(pre_base)
            raw_event = self.raw_tracker.update(values["range_rate_mps"], step_index=self.total_steps)
            reference_66d = self._reference_observation()
            action = _finite_action(self.fixed_head_on.get_deterministic_action(self.observation["observation_vector"]))
            next_observation, _environment_reward, terminated, truncated, info = self.env.step(action)
            self.total_steps += 1
            self.last_info = info
            self._observe(next_observation)
            own_state = info.get("own_state", {}) if isinstance(info.get("own_state"), Mapping) else {}
            target_state = info.get("target_state", {}) if isinstance(info.get("target_state"), Mapping) else {}
            post_raw_event = raw_event is not None
            record = {
                "stage": "continuous_fixed_head_on",
                "step": self.total_steps,
                "phase": pre_phase,
                "phase_source": "raw_si_range_rate_phase_tracker",
                "dynamic_state": pre_state,
                "taxonomy_angle_order": ["ATA", "AA"],
                "environment_first_pass_complete": bool(info.get("first_pass_complete", False)),
                "raw_si_phase_input": {"range_m": values["range_m"], "range_rate_mps": values["range_rate_mps"]},
                "raw_si_first_pass_latched": post_raw_event,
                "raw_si_first_pass_event_step": None if raw_event is None else raw_event.event_step_index,
                "base_observation_16d": [float(value) for value in pre_base],
                "history_length": len(pre_history),
                "history_padding_used": False,
                "observation_66d": None if reference_66d is None else [float(value) for value in reference_66d],
                "reference_profile": str(self.config["observability"]["reference_profile"]),
                "profile_changes_actions": False,
                "profile_validity_gate": False,
                "normalized_vpp_action": [float(value) for value in action],
                "post_event_66d_valid": bool(post_raw_event and reference_66d is not None),
                "prediction_valid": bool(info.get("prediction_valid", False)),
                "prediction_fallback": bool(info.get("prediction_fallback", False)),
                "vp_forward_bias_m": info.get("vp_forward_bias_m"),
                "vp_lateral_bias_m": info.get("vp_lateral_bias_m"),
                "vp_vertical_bias_m": info.get("vp_vertical_bias_m"),
                "guidance_command": info.get("guidance_command"),
                "nz_cmd": info.get("nz_cmd"),
                "actual_nz_g": own_state.get("nz_g", info.get("nz_g")),
                "nz_saturated": bool(info.get("nz_saturated", False)),
                "roll_rate_saturated": bool(info.get("roll_rate_saturated", False)),
                "throttle_saturated": bool(info.get("throttle_saturated", False)),
                "target_in_attack_zone": bool(info.get("target_in_attack_zone", False)),
                "ego_in_attack_zone": bool(info.get("ego_in_attack_zone", False)),
                "ego_hp": info.get("ego_hp"),
                "target_hp": info.get("target_hp"),
                "terminal_reason": str(info.get("reason") or "timeout"),
                "backend": info.get("backend", "jsbsim"),
                "backend_fallback_occurred": bool(info.get("backend_fallback_occurred", False)),
                "own_state": dict(own_state),
                "target_state": dict(target_state),
            }
            record["step_sha256"] = _stable_hash(record)
            self.steps.append(record)
            if record["backend_fallback_occurred"]:
                raise FiveStateRawSIV5RecordError("V5 reported a JSBSim backend fallback")
            if terminated or truncated:
                terminal_reason = record["terminal_reason"]
                break
        replay = replay_raw_si_first_pass(self.steps, self.raw_tracker.config)
        tracker_event = self.raw_tracker.event
        if (replay is None) != (tracker_event is None) or (
            replay is not None and tracker_event is not None and replay.event_step_index != tracker_event.event_step_index
        ):
            raise FiveStateRawSIV5RecordError("V5 raw-SI event replay disagrees with the live collector")
        post_event_66d_steps = sum(bool(step["post_event_66d_valid"]) for step in self.steps)
        prediction_valid_post_event_steps = sum(
            bool(step["post_event_66d_valid"] and step["prediction_valid"] and not step["prediction_fallback"])
            for step in self.steps
        )
        full_telemetry = all(
            len(step["base_observation_16d"]) == 16
            and len(step["normalized_vpp_action"]) == 3
            and np.isfinite(np.asarray(step["base_observation_16d"], dtype=np.float64)).all()
            and np.isfinite(np.asarray(step["normalized_vpp_action"], dtype=np.float64)).all()
            and (step["observation_66d"] is None or (
                len(step["observation_66d"]) == 66 and np.isfinite(np.asarray(step["observation_66d"], dtype=np.float64)).all()
            ))
            for step in self.steps
        )
        ledger = {
            "source_id": SOURCE_ID,
            "header": {
                "scenario": scenario["name"],
                "scenario_seed": scenario["metadata"]["scenario_seed"],
                "initial_state": scenario["metadata"]["initial_class"],
                "mirror_sign": scenario["metadata"]["mirror_sign"],
                "height_condition": scenario["metadata"]["height_condition"],
                "reference_controller": "frozen_fixed_head_on",
                "reference_profile": str(self.config["observability"]["reference_profile"]),
            },
            "steps": self.steps,
            "summary": {
                "step_count": len(self.steps),
                "raw_event_reached": tracker_event is not None,
                "raw_event_step_index": None if tracker_event is None else tracker_event.event_step_index,
                "post_event_66d_steps": post_event_66d_steps,
                "prediction_valid_post_event_steps": prediction_valid_post_event_steps,
                "full_telemetry": full_telemetry,
                "no_backend_fallback": all(not bool(step["backend_fallback_occurred"]) for step in self.steps),
                "no_reset_after_initialization": True,
                "no_history_padding": all(not bool(step["history_padding_used"]) for step in self.steps),
                "no_checkpoint_fallback": True,
                "raw_si_event_replay_passed": True,
                "terminal_reason": terminal_reason,
            },
        }
        result = {
            "raw_event_reached": tracker_event is not None,
            "post_event_66d_steps": post_event_66d_steps,
            "prediction_valid_post_event_steps": prediction_valid_post_event_steps,
            "telemetry_complete": full_telemetry,
            "no_backend_fallback": ledger["summary"]["no_backend_fallback"],
            "no_reset_after_initialization": True,
            "no_history_padding": ledger["summary"]["no_history_padding"],
            "no_checkpoint_fallback": True,
            "terminal_reason": terminal_reason,
            "training_reward": "prohibited",
            "environment_reward_recording": "prohibited",
        }
        return result, ledger

    def close(self) -> None:
        self.env.close()


def runtime_record(
    *, opponent: str, scenario: Mapping[str, Any], result: Mapping[str, Any], ledger: Mapping[str, Any], runtime_evidence: Mapping[str, Any]
) -> dict[str, Any]:
    """Serialize one V5 record with no reward, score, or controller substitution."""

    if opponent not in OPPONENTS:
        raise FiveStateRawSIV5RecordError("V5 runtime record opponent drifted")
    metadata = scenario.get("metadata")
    if not isinstance(metadata, Mapping) or not isinstance(ledger.get("summary"), Mapping):
        raise FiveStateRawSIV5RecordError("V5 runtime record lacks metadata or ledger summary")
    required = (
        "raw_event_reached", "post_event_66d_steps", "telemetry_complete", "no_backend_fallback",
        "no_reset_after_initialization", "no_history_padding", "no_checkpoint_fallback",
        "training_reward", "environment_reward_recording",
    )
    if any(key not in result for key in required) or "reward" in result:
        raise FiveStateRawSIV5RecordError("V5 runtime result drifted from the reward-free contract")
    if result["training_reward"] != "prohibited" or result["environment_reward_recording"] != "prohibited":
        raise FiveStateRawSIV5RecordError("V5 runtime result exposes a reward channel")
    for key in ("scenario_application_passed", "raw_si_event_replay_passed", "fixed_head_on_action_only"):
        if runtime_evidence.get(key) is not True:
            raise FiveStateRawSIV5RecordError("V5 runtime evidence failed: {0}".format(key))
    if not all(bool(result[key]) for key in (
        "telemetry_complete", "no_backend_fallback", "no_reset_after_initialization", "no_history_padding", "no_checkpoint_fallback"
    )):
        raise FiveStateRawSIV5RecordError("V5 runtime record has an invalid continuity contract")
    return {
        "source_id": SOURCE_ID,
        "opponent": opponent,
        "initial_state": metadata.get("initial_class"),
        "pair_key": metadata.get("pair_key"),
        "metadata": dict(metadata),
        "result": dict(result),
        "runtime_evidence": dict(runtime_evidence),
        "paired_delta_eligible": False,
        "role": "nonlearning_raw_si_phase_observability",
    }
