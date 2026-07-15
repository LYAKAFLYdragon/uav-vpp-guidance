"""One-time neutral/post-merge reentry-recovery feasibility pilot runtime.

This module is deliberately narrower than the future five-skill system.  It
trains one 66-D-to-3-D VPP policy with routing disabled and preserves full
run-in and post-handoff telemetry for every dev/heldout episode.
"""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
import yaml

from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.evaluation.global_advantage_runin_contract import stable_hash
from uav_vpp_guidance.evaluation.thesis_neutral_postmerge_reentry_recovery_pilot import (
    validate_design,
)
from uav_vpp_guidance.hierarchy.specialist_policy import FrozenSpecialistPolicy
from uav_vpp_guidance.training import thesis_defext_rangeext_pilot as legacy
from uav_vpp_guidance.training.thesis_shared_skill_geometry import FrozenP3Encoder
from uav_vpp_guidance.training.thesis_shared_skill_policy import ThesisSharedSkillPPOAgent


ROOT = Path(__file__).resolve().parents[3]
SOURCE_ID = "THESIS-NEUTRAL-POSTMERGE-REENTRY-RECOVERY-PILOT-V1"
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
METHODS = (
    "candidate_fixed_reentry_recovery",
    "frozen_fixed_head_on",
    "frozen_fixed_crossing",
)
TARGET_SKILL = "reentry_recovery"
TARGET_PROFILE = "reentry_preparation"
TARGET_STATE = "neutral"
TARGET_PHASE = "post_merge"


class NeutralPostMergePilotError(RuntimeError):
    """Raised when the authorized pilot breaks a frozen contract."""


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise NeutralPostMergePilotError(f"expected YAML mapping: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_authorized_config(path: Path) -> tuple[dict[str, Any], Path]:
    overlay = load_yaml(path)
    base_value = overlay.pop("base_config", None)
    if not base_value:
        raise NeutralPostMergePilotError("execution config must declare base_config")
    base_path = _repo_path(str(base_value))
    expected_hash = str(overlay.get("execution", {}).get("base_config_sha256", "")).lower()
    if not base_path.is_file() or sha256_file(base_path) != expected_hash:
        raise NeutralPostMergePilotError("execution config base_config SHA mismatch")
    return _merge(load_yaml(base_path), overlay), base_path


def _git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _git_changed_paths_since(commit: str) -> set[str]:
    output = _git_value("diff", "--name-only", f"{commit}..HEAD")
    return {line.replace("\\", "/") for line in output.splitlines() if line.strip()}


def _require_hash(path: Path, expected: Any, label: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64:
        raise NeutralPostMergePilotError(f"{label} must declare a SHA-256")
    if not path.is_file() or sha256_file(path) != expected.lower():
        raise NeutralPostMergePilotError(f"{label} SHA mismatch")


def _finite_action(action: Any, label: str) -> np.ndarray:
    result = np.asarray(action, dtype=np.float32).reshape(-1)
    if result.shape != (3,) or not np.isfinite(result).all() or np.any(np.abs(result) > 1.0 + 1e-6):
        raise NeutralPostMergePilotError(f"{label} is not a finite normalized 3-D VPP action")
    return result


@contextmanager
def _legacy_target_contract() -> Iterator[None]:
    """Reuse only the frozen 66-D/VPP mechanics, never its task identity."""

    old_skill, old_profile = legacy.TARGET_SKILL, legacy.TARGET_PROFILE
    legacy.TARGET_SKILL, legacy.TARGET_PROFILE = TARGET_SKILL, TARGET_PROFILE
    try:
        yield
    finally:
        legacy.TARGET_SKILL, legacy.TARGET_PROFILE = old_skill, old_profile


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if hasattr(value, "tolist"):
        return _canonical(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if math.isnan(number):
            return "NaN"
        if math.isinf(number):
            return "Infinity" if number > 0 else "-Infinity"
        return number
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (str, bool)) or value is None:
        return value
    return repr(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_canonical(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _runtime(config: Mapping[str, Any]) -> Mapping[str, Any]:
    runtime_path = _repo_path(str(config["execution"]["runtime_template"]))
    return load_yaml(runtime_path)["runtime"]


def _instantiate_opponent(entry: Mapping[str, Any]) -> object:
    return legacy._instantiate_opponent(entry)


def _build_env(config: Mapping[str, Any], registry: Mapping[str, Any], opponent_id: str) -> CloseRangeTrackingEnv:
    base = copy.deepcopy(load_yaml(legacy.BASE_ENV_CONFIG))
    runtime = _runtime(config)
    training = config["proposed_training"]
    base["backend"] = "jsbsim"
    base["task"] = {"name": "head_on", "env_class": "CloseRangeTrackingEnv"}
    base.setdefault("env", {}).update(
        {
            "use_jsbsim": True,
            "strict_backend": True,
            "legacy_project_root": str(runtime["legacy_project_root"]),
            "aircraft_model": str(runtime["aircraft_model"]),
            "high_level_dt": float(runtime["high_level_dt_s"]),
            "max_high_level_steps": int(training["max_high_level_steps_per_episode"]),
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
    base["attack_zone"] = copy.deepcopy(config["fixed_contract"]["attack_zone"])
    entry = registry["opponents"][opponent_id]
    return CloseRangeTrackingEnv(
        base,
        opponent_policy=_instantiate_opponent(entry),
        opponent_config={"stage": opponent_id, **copy.deepcopy(entry)},
    )


def _specialist(registry: Mapping[str, Any], name: str) -> FrozenSpecialistPolicy:
    entry = registry["specialists"][name]
    checkpoint = Path(str(entry["checkpoint"]))
    _require_hash(checkpoint, entry.get("checkpoint_sha256"), f"specialist {name}")
    return FrozenSpecialistPolicy(
        checkpoint_path=str(checkpoint),
        config_path=str(_repo_path(str(entry["config_path"]))),
        device="cpu",
    )


class NeutralPostMergeEpisode(legacy.ContinuousHandoffEpisode):
    """Fresh JSBSim episode with a neutral/post-merge handoff trigger."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.runin_steps: list[dict[str, Any]] = []
        self.post_handoff_steps: list[dict[str, Any]] = []
        self.handoff_state_sha256: str | None = None

    def _trigger_ready(self) -> bool:
        return bool(
            len(self.history) == 10
            and self.first_pass_complete
            and self.dynamic_state == TARGET_STATE
            and self.phase == TARGET_PHASE
            and self._target_pair_allowed()
            and self.last_info.get("prediction_valid") is True
            and self.last_info.get("prediction_fallback") is not True
        )

    def _step_record(
        self,
        *,
        stage: str,
        action: np.ndarray,
        phase: str,
        dynamic_state: str,
        observation_66d: np.ndarray | None,
        info: Mapping[str, Any],
        base_vector: np.ndarray | None = None,
        history: Sequence[np.ndarray] | None = None,
    ) -> dict[str, Any]:
        if self.base is None and base_vector is None:
            raise NeutralPostMergePilotError("telemetry requested without a base observation")
        base_array = self.base if base_vector is None else base_vector
        assert base_array is not None
        history_values = tuple(self.history) if history is None else tuple(history)
        base = legacy._base_map(base_array)
        own_state = info.get("own_state") if isinstance(info.get("own_state"), Mapping) else {}
        target_state = info.get("target_state") if isinstance(info.get("target_state"), Mapping) else {}
        record = {
            "stage": stage,
            "step": int(self.total_steps),
            "phase": phase,
            "dynamic_state": dynamic_state,
            "first_pass_complete": bool(self.first_pass_complete),
            "raw_si_phase_input": {"range_m": base["range_m"], "range_rate_mps": base["range_rate_mps"]},
            "base_observation_16d": [float(value) for value in base_array],
            "history_length": len(history_values),
            "history_sha256": stable_hash([list(item) for item in history_values], 6),
            "observation_66d": None if observation_66d is None else [float(value) for value in observation_66d],
            "normalized_vpp_action": [float(value) for value in action],
            "vp_forward_bias_m": info.get("vp_forward_bias_m"),
            "vp_lateral_bias_m": info.get("vp_lateral_bias_m"),
            "vp_vertical_bias_m": info.get("vp_vertical_bias_m"),
            "guidance_command": info.get("guidance_command"),
            "nz_cmd": info.get("nz_cmd"),
            "actual_nz_g": own_state.get("nz_g", info.get("nz_g")),
            "ego_attack_aoa_deg": info.get("ego_attack_aoa_deg"),
            "own_speed_mps": base["own_speed"],
            "own_altitude_m": base["own_altitude"],
            "target_speed_mps": base["target_speed"],
            "target_altitude_m": base["target_altitude"],
            "prediction_valid": bool(info.get("prediction_valid", False)),
            "prediction_fallback": bool(info.get("prediction_fallback", False)),
            "backend": info.get("backend", "jsbsim"),
            "backend_fallback_occurred": bool(info.get("backend_fallback_occurred", False)),
            "nz_saturated": bool(info.get("nz_saturated", False)),
            "roll_rate_saturated": bool(info.get("roll_rate_saturated", False)),
            "throttle_saturated": bool(info.get("throttle_saturated", False)),
            "target_in_attack_zone": bool(info.get("target_in_attack_zone", False)),
            "ego_in_attack_zone": bool(info.get("ego_in_attack_zone", False)),
            "ego_hp": info.get("ego_hp"),
            "target_hp": info.get("target_hp"),
            "terminal_reason": info.get("reason"),
            "own_state": dict(own_state),
            "target_state": dict(target_state),
        }
        record["step_sha256"] = stable_hash(record, 6)
        return record

    def reset_to_handoff(self, scenario: Mapping[str, Any], seed: int) -> tuple[np.ndarray | None, dict[str, Any]]:
        self.history.clear()
        self.history_features.reset()
        self.phase_tracker = legacy.PhaseTracker()
        self.first_pass_complete = False
        self.last_info = {}
        self.total_steps = 0
        self.handoff_step = None
        self.handoff_state_sha256 = None
        self.runin_steps = []
        self.post_handoff_steps = []
        observation = self.env.reset(scenario=copy.deepcopy(dict(scenario)), seed=int(seed))
        if self.env._backend != "jsbsim":
            raise NeutralPostMergePilotError("strict JSBSim backend was not selected")
        self.env.set_runtime_specialist_context(
            specialist_key="head_on",
            specialist_profile=None,
            specialist_mode_name="run_in_head_on_specialist",
            specialist_reason="neutral_postmerge_frozen_runin",
        )
        self.observation_schema = copy.deepcopy(observation["observation_schema"])
        self._observe(observation)
        while self.total_steps < self.env.max_steps:
            if self._trigger_ready():
                self.handoff_step = self.total_steps + 1
                policy_obs = self.policy_observation()
                self.handoff_state_sha256 = stable_hash(
                    {
                        "base": [float(value) for value in self.base],
                        "history": [list(item) for item in self.history],
                        "phase": self.phase,
                        "dynamic_state": self.dynamic_state,
                        "policy_observation": [float(value) for value in policy_obs],
                    },
                    6,
                )
                return policy_obs, {
                    "handoff_reached": True,
                    "handoff_step": self.handoff_step,
                    "phase": self.phase,
                    "dynamic_state": self.dynamic_state,
                    "handoff_state_sha256": self.handoff_state_sha256,
                }
            if self.observation is None or self.base is None:
                raise NeutralPostMergePilotError("run-in lost its current observation")
            pre_phase, pre_state = self.phase, self.dynamic_state
            pre_base, pre_history = self.base.copy(), tuple(item.copy() for item in self.history)
            action = _finite_action(self.run_in_policy.get_deterministic_action(self.observation["observation_vector"]), "run-in action")
            observation, _reward, terminated, truncated, info = self.env.step(action)
            self.total_steps += 1
            self.last_info = info
            self.first_pass_complete = bool(info.get("first_pass_complete", False))
            self._observe(observation)
            self.runin_steps.append(
                self._step_record(stage="run_in", action=action, phase=pre_phase, dynamic_state=pre_state, observation_66d=None, info=info, base_vector=pre_base, history=pre_history)
            )
            if bool(info.get("backend_fallback_occurred", False)):
                raise NeutralPostMergePilotError("run-in reported a backend fallback")
            if terminated or truncated:
                return None, {"handoff_reached": False, "terminal_reason": str(info.get("reason") or "terminated_before_handoff"), "steps": self.total_steps}
        return None, {"handoff_reached": False, "terminal_reason": "horizon_before_handoff", "steps": self.total_steps}

    def step(self, action: np.ndarray) -> tuple[np.ndarray | None, float, bool, dict[str, Any]]:
        if self.observation is None or self.base is None:
            raise NeutralPostMergePilotError("post-handoff step has no current observation")
        pre_phase, pre_state = self.phase, self.dynamic_state
        pre_first_pass = self.first_pass_complete
        pre_pair_allowed = self._target_pair_allowed()
        pre_prediction_valid = self.last_info.get("prediction_valid") is True
        pre_prediction_fallback = self.last_info.get("prediction_fallback") is True
        policy_obs = self.policy_observation()
        pre_base, pre_history = self.base.copy(), tuple(item.copy() for item in self.history)
        normalized = _finite_action(action, "post-handoff action")
        next_observation, reward, done, metric = super().step(normalized)
        metric["valid_target_step"] = bool(
            pre_first_pass
            and pre_state == TARGET_STATE
            and pre_phase == TARGET_PHASE
            and pre_pair_allowed
            and pre_prediction_valid
            and not pre_prediction_fallback
        )
        metric["handoff_state_sha256"] = self.handoff_state_sha256
        self.post_handoff_steps.append(
            self._step_record(stage="post_handoff", action=normalized, phase=pre_phase, dynamic_state=pre_state, observation_66d=policy_obs, info=self.last_info, base_vector=pre_base, history=pre_history)
        )
        if bool(self.last_info.get("backend_fallback_occurred", False)):
            raise NeutralPostMergePilotError("post-handoff reported a backend fallback")
        return next_observation, reward, done, metric

    def ledger(self, *, opponent: str, method: str, scenario: Mapping[str, Any], handoff: Mapping[str, Any]) -> dict[str, Any]:
        steps = self.runin_steps + self.post_handoff_steps
        return {
            "source_id": SOURCE_ID,
            "header": {
                "opponent": opponent,
                "method": method,
                "scenario": scenario["name"],
                "scenario_signature": scenario["metadata"]["scenario_signature"],
                "scenario_seed": scenario["metadata"]["scenario_seed"],
                "mirror_sign": scenario["metadata"]["mirror_sign"],
            },
            "handoff": dict(handoff),
            "steps": steps,
            "summary": {
                "step_count": len(steps),
                "handoff_reached": bool(handoff.get("handoff_reached")),
                "no_backend_fallback": all(not bool(item.get("backend_fallback_occurred")) for item in steps),
                "full_post_handoff_telemetry": all(
                    item.get("observation_66d") is not None and len(item["observation_66d"]) == 66
                    for item in self.post_handoff_steps
                ),
            },
        }


def sample_train_scenario(distribution: Mapping[str, Any], episode_index: int) -> dict[str, Any]:
    contract = distribution["sampling_contract"]
    rng = np.random.default_rng(int(contract["seed"]) + int(episode_index))
    own_altitude = float(rng.uniform(*contract["own_altitude_m"]))
    altitude_diff = float(rng.uniform(*contract["target_minus_own_altitude_m"]))
    initial_range = float(rng.uniform(*contract["initial_range_m"]))
    own_speed = float(rng.uniform(*contract["own_speed_mps"]))
    target_speed = max(float(rng.uniform(*contract["target_speed_mps"])), own_speed + float(contract["require_target_faster_than_own_by_mps"]))
    mirror = contract["mirror_signs"][episode_index % len(contract["mirror_signs"])]
    sign = -1 if mirror == "negative" else 1
    los_azimuth = sign * float(rng.uniform(*contract["los_azimuth_abs_deg"]))
    own_angle = float(rng.uniform(*contract["own_to_target_los_angle_deg"]))
    target_angle = float(rng.uniform(*contract["target_velocity_to_own_los_angle_deg"]))
    horizontal = math.sqrt(initial_range**2 - altitude_diff**2)
    fraction = horizontal / initial_range
    feasible_angle_limit = math.degrees(math.acos(-fraction)) - 1e-6
    if own_angle > feasible_angle_limit or target_angle > feasible_angle_limit:
        raise NeutralPostMergePilotError("frozen neutral sampler requested an infeasible 3-D angle")
    own_heading = legacy._horizontal_heading(los_azimuth, fraction, own_angle, -sign)
    target_heading = legacy._horizontal_heading(legacy._normalize_heading(los_azimuth + 180.0), fraction, target_angle, sign)
    azimuth = math.radians(los_azimuth)
    return {
        "name": f"neutral_postmerge_reentry_train_{episode_index:06d}",
        "own_init": {"position_m": [0.0, 0.0, own_altitude], "velocity_mps": own_speed, "heading_deg": own_heading},
        "target_init": {"position_m": [horizontal * math.cos(azimuth), horizontal * math.sin(azimuth), own_altitude + altitude_diff], "velocity_mps": target_speed, "heading_deg": target_heading},
        "metadata": {
            "split": "train", "initial_class": TARGET_STATE, "height_condition": "continuous", "mirror_sign": mirror,
            "scenario_signature": f"neutral_train_{episode_index:06d}", "scenario_seed": int(contract["seed"]) + int(episode_index),
        },
    }


def _agent_config(config: Mapping[str, Any]) -> dict[str, Any]:
    training = config["proposed_training"]
    return {"training": {"ppo": copy.deepcopy(training["ppo"]), "policy": copy.deepcopy(training["policy"])}}


def _checkpoint_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_id": SOURCE_ID, "skill_name": TARGET_SKILL, "profile_name": TARGET_PROFILE,
        "observation_dim": 66, "action_dim": 3, "p3_encoder_sha256": config["fixed_contract"]["encoder"]["sha256"],
        "encoder_trainable": False, "routing_enabled": False, "training_seed": int(config["proposed_training"]["training_seed"]),
    }


def _episode_result(opponent: str, method: str, scenario: Mapping[str, Any], handoff: Mapping[str, Any], metrics: Sequence[Mapping[str, Any]], ledger: Mapping[str, Any]) -> dict[str, Any]:
    valid = [item for item in metrics if item.get("valid_target_step")][:20]
    final = metrics[-1] if metrics else {}
    return {
        "opponent": opponent, "method": method, "scenario": scenario["name"],
        "scenario_signature": scenario["metadata"]["scenario_signature"], "mirror_sign": scenario["metadata"]["mirror_sign"],
        "scenario_seed": scenario["metadata"]["scenario_seed"], "handoff_reached": bool(handoff.get("handoff_reached")),
        "handoff_step": handoff.get("handoff_step"), "handoff_state_sha256": handoff.get("handoff_state_sha256"),
        "valid_target_steps": sum(bool(item.get("valid_target_step")) for item in metrics), "qualifying": len(valid) == 20,
        "intent_loss_auc20": float(np.mean([item["after_intent_loss"] for item in valid])) if len(valid) == 20 else None,
        "range_opening_fraction_auc20": float(np.mean([float(item["range_rate_mps"] > 0.0) for item in valid])) if valid else None,
        "specific_energy_height_delta_m_at_20": float(valid[-1]["specific_energy_height_delta_m"]) if len(valid) == 20 else None,
        "target_attack_zone_exposure_fraction_auc20": float(np.mean([float(item["target_in_attack_zone"]) for item in valid])) if valid else None,
        "ego_attack_zone_exposure_fraction_auc20": float(np.mean([float(item["ego_in_attack_zone"]) for item in valid])) if valid else None,
        "terminal_reason": final.get("terminal_reason", handoff.get("terminal_reason", "no_handoff")),
        "ego_failure": bool(final.get("ego_failure", False)), "win": bool(final.get("win", False)),
        "loss": bool(final.get("loss", False)), "draw": bool(final.get("draw", False)),
        "ego_hp": final.get("ego_hp"), "target_hp": final.get("target_hp"),
        "telemetry_complete": bool(ledger["summary"]["full_post_handoff_telemetry"]),
    }


def _run_episode(config: Mapping[str, Any], registry: Mapping[str, Any], scenario: Mapping[str, Any], opponent: str, method: str, *, agent: ThesisSharedSkillPPOAgent | None = None, collect_training: bool = False) -> tuple[dict[str, Any], dict[str, Any], list[tuple[np.ndarray, np.ndarray, float, float, bool, float]]]:
    if method not in METHODS:
        raise NeutralPostMergePilotError(f"unknown method: {method}")
    with _legacy_target_contract():
        encoder = FrozenP3Encoder(Path(config["fixed_contract"]["encoder"]["checkpoint"]), str(config["fixed_contract"]["encoder"]["sha256"]))
        env = _build_env(config, registry, opponent)
        episode = NeutralPostMergeEpisode(env, _specialist(registry, "run_in_head_on"), encoder, config)
        baseline = None
        if method == "frozen_fixed_head_on":
            baseline = _specialist(registry, method)
        elif method == "frozen_fixed_crossing":
            baseline = _specialist(registry, method)
        elif agent is None:
            raise NeutralPostMergePilotError("candidate method needs a loaded agent")
        transitions: list[tuple[np.ndarray, np.ndarray, float, float, bool, float]] = []
        try:
            observation, handoff = episode.reset_to_handoff(scenario, int(scenario["metadata"]["scenario_seed"]))
            metrics: list[dict[str, Any]] = []
            if observation is not None:
                if method == "candidate_fixed_reentry_recovery":
                    episode.set_handoff_context(specialist_key=TARGET_SKILL, specialist_profile=TARGET_PROFILE, specialist_mode_name="reentry_recovery_specialist")
                elif method == "frozen_fixed_head_on":
                    episode.set_handoff_context(specialist_key="head_on", specialist_profile=None, specialist_mode_name="head_on_specialist")
                else:
                    episode.set_handoff_context(specialist_key="crossing_feasible", specialist_profile=None, specialist_mode_name="crossing_specialist")
                done = False
                while not done:
                    if agent is not None:
                        action, log_prob, value = agent.select_action(observation, deterministic=not collect_training)
                    else:
                        if episode.observation is None or baseline is None:
                            raise NeutralPostMergePilotError("baseline lost its source observation")
                        action = _finite_action(baseline.get_deterministic_action(episode.observation["observation_vector"]), "baseline action")
                        log_prob, value = 0.0, 0.0
                    next_observation, reward, done, metric = episode.step(action)
                    metrics.append(metric)
                    if collect_training:
                        transitions.append((observation.copy(), action.copy(), float(log_prob), float(reward), bool(done), float(value)))
                    if not done:
                        if next_observation is None:
                            raise NeutralPostMergePilotError("unfinished episode lost its 66-D observation")
                        observation = next_observation
            ledger = episode.ledger(opponent=opponent, method=method, scenario=scenario, handoff=handoff)
            return _episode_result(opponent, method, scenario, handoff, metrics, ledger), ledger, transitions
        finally:
            episode.close()


def _evaluate(config: Mapping[str, Any], registry: Mapping[str, Any], scenarios: Sequence[Mapping[str, Any]], opponent: str, method: str, checkpoint: Path | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    agent = None
    if method == "candidate_fixed_reentry_recovery":
        if checkpoint is None:
            raise NeutralPostMergePilotError("candidate evaluation requires a checkpoint")
        agent = ThesisSharedSkillPPOAgent(_agent_config(config), device="cpu")
        agent.load_strict(checkpoint, _checkpoint_metadata(config))
    results, ledgers = [], []
    for scenario in scenarios:
        result, ledger, _transitions = _run_episode(config, registry, scenario, opponent, method, agent=agent)
        results.append(result)
        ledgers.append(ledger)
    return results, ledgers


def _summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    qualifying = [item for item in records if item.get("qualifying")]
    return {
        "episodes": len(records), "handoff_episodes": sum(bool(item.get("handoff_reached")) for item in records),
        "qualifying_episodes": len(qualifying), "valid_target_steps": sum(int(item.get("valid_target_steps", 0)) for item in records),
        "mean_intent_loss_auc20": float(np.mean([item["intent_loss_auc20"] for item in qualifying])) if qualifying else None,
        "ego_crash_oob_rate": sum(bool(item.get("ego_failure")) for item in records) / max(len(records), 1),
        "telemetry_complete": all(bool(item.get("telemetry_complete")) for item in records),
    }


def _paired(candidate: Sequence[Mapping[str, Any]], baseline: Sequence[Mapping[str, Any]]) -> list[float]:
    baseline_map = {item["scenario_signature"]: item for item in baseline if item.get("qualifying") and item.get("intent_loss_auc20") is not None}
    deltas = []
    for item in candidate:
        other = baseline_map.get(item["scenario_signature"])
        if item.get("qualifying") and item.get("intent_loss_auc20") is not None and other is not None:
            if item.get("handoff_state_sha256") != other.get("handoff_state_sha256"):
                raise NeutralPostMergePilotError("paired methods have non-equivalent handoff state hashes")
            deltas.append(float(item["intent_loss_auc20"]) - float(other["intent_loss_auc20"]))
    return deltas


def _evaluate_dev(config: Mapping[str, Any], candidate: Mapping[str, Sequence[Mapping[str, Any]]], baseline: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]) -> dict[str, Any]:
    per_opponent, worst, safety_pass = {}, -float("inf"), True
    for opponent in OPPONENTS:
        candidate_records, head = candidate[opponent], baseline[opponent]["frozen_fixed_head_on"]
        deltas = _paired(candidate_records, head)
        candidate_summary, head_summary = _summary(candidate_records), _summary(head)
        mean = float(np.mean(deltas)) if deltas else None
        crash_delta = candidate_summary["ego_crash_oob_rate"] - head_summary["ego_crash_oob_rate"]
        passes = crash_delta <= float(config["gates"]["safety_noninferiority_vs_head_on"]["ego_crash_oob_rate_delta_max"])
        safety_pass = safety_pass and passes
        worst = max(worst, mean if mean is not None else float("inf"))
        per_opponent[opponent] = {"paired_count": len(deltas), "mean_paired_delta_vs_head_on": mean, "ego_crash_oob_delta_vs_head_on": crash_delta, "safety_pass": passes}
    return {"selection_score_worst_opponent_mean_delta": worst, "safety_pass": safety_pass, "per_opponent": per_opponent}


def analyze_heldout(config: Mapping[str, Any], records: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]) -> dict[str, Any]:
    gates, per_opponent = config["gates"], {}
    head_improvements = best_improvements = 0
    all_contract = all_safety = all_head_noninferior = remaining_best_noninferior = True
    for opponent in OPPONENTS:
        candidate = records[opponent]["candidate_fixed_reentry_recovery"]
        head, crossing = records[opponent]["frozen_fixed_head_on"], records[opponent]["frozen_fixed_crossing"]
        candidate_summary, head_summary, crossing_summary = _summary(candidate), _summary(head), _summary(crossing)
        head_deltas, crossing_deltas = _paired(candidate, head), _paired(candidate, crossing)
        head_delta = float(np.mean(head_deltas)) if head_deltas else None
        crossing_delta = float(np.mean(crossing_deltas)) if crossing_deltas else None
        baselines = {"frozen_fixed_head_on": head_summary["mean_intent_loss_auc20"], "frozen_fixed_crossing": crossing_summary["mean_intent_loss_auc20"]}
        viable = {name: value for name, value in baselines.items() if value is not None}
        best_name = min(viable, key=viable.get) if viable else None
        best_delta = head_delta if best_name == "frozen_fixed_head_on" else crossing_delta if best_name else None
        minimum, claim_ready = gates["per_opponent_contract_minimum"], gates["per_opponent_claim_ready_coverage"]
        signatures = {item["scenario_signature"] for item in candidate if item.get("qualifying")}
        mirrors = {item["mirror_sign"] for item in candidate if item.get("qualifying")}
        contract = bool(
            candidate_summary["qualifying_episodes"] >= int(claim_ready["qualifying_paired_episodes"])
            and len(head_deltas) >= int(claim_ready["qualifying_paired_episodes"])
            and len(crossing_deltas) >= int(claim_ready["qualifying_paired_episodes"])
            and candidate_summary["valid_target_steps"] >= int(claim_ready["valid_target_steps"])
            and len(signatures) >= int(minimum["distinct_scenario_signatures"])
            and len(mirrors) >= int(minimum["distinct_mirror_signs"])
            and candidate_summary["telemetry_complete"] and head_summary["telemetry_complete"] and crossing_summary["telemetry_complete"]
        )
        crash_delta = candidate_summary["ego_crash_oob_rate"] - head_summary["ego_crash_oob_rate"]
        safety = crash_delta <= float(gates["safety_noninferiority_vs_head_on"]["ego_crash_oob_rate_delta_max"])
        head_noninferior = head_delta is not None and head_delta <= float(gates["geometry_noninferiority_vs_head_on"]["paired_intent_loss_auc20_delta_max"])
        head_improved = head_delta is not None and head_delta <= float(gates["practical_improvement_vs_head_on"]["paired_intent_loss_auc20_delta_max"])
        best_improved = best_delta is not None and best_delta <= float(gates["practical_improvement_vs_best_existing_specialist"]["paired_intent_loss_auc20_delta_max"])
        best_noninferior = best_delta is not None and best_delta <= float(gates["remaining_opponent_noninferiority_vs_best_existing_specialist"]["paired_intent_loss_auc20_delta_max"])
        head_improvements += int(head_improved); best_improvements += int(best_improved)
        all_contract = all_contract and contract; all_safety = all_safety and safety
        all_head_noninferior = all_head_noninferior and head_noninferior; remaining_best_noninferior = remaining_best_noninferior and best_noninferior
        per_opponent[opponent] = {
            "candidate": candidate_summary, "head_on": head_summary, "crossing": crossing_summary,
            "paired_head_count": len(head_deltas), "paired_crossing_count": len(crossing_deltas),
            "mean_paired_delta_vs_head_on": head_delta, "mean_paired_delta_vs_crossing": crossing_delta,
            "best_existing_specialist": best_name, "mean_paired_delta_vs_best_existing": best_delta,
            "ego_crash_oob_delta_vs_head_on": crash_delta, "contract_pass": contract, "safety_pass": safety,
            "head_on_noninferiority_pass": head_noninferior, "head_on_practical_improvement_pass": head_improved,
            "best_existing_practical_improvement_pass": best_improved, "best_existing_noninferiority_pass": best_noninferior,
        }
    supported = bool(all_contract and all_safety and all_head_noninferior and remaining_best_noninferior and head_improvements >= int(gates["practical_improvement_vs_head_on"]["minimum_opponents_passing"]) and best_improvements >= int(gates["practical_improvement_vs_best_existing_specialist"]["minimum_opponents_passing"]))
    crossing_sufficient = any(item["best_existing_specialist"] == "frozen_fixed_crossing" and item["mean_paired_delta_vs_crossing"] is not None and item["mean_paired_delta_vs_crossing"] >= -0.02 for item in per_opponent.values())
    verdict = "safety_or_contract_no_go" if not all_contract or not all_safety else "new_reentry_recovery_skill_gap_preliminarily_supported" if supported else "existing_library_routing_or_composition_gap" if crossing_sufficient else "reentry_recovery_hypothesis_not_supported"
    return {"source_id": SOURCE_ID, "verdict": verdict, "new_skill_gap_supported": supported, "training_authorized_after_pilot": False, "per_opponent": per_opponent, "aggregate_decision_counts": {"head_on_practical_improvement_opponents": head_improvements, "best_existing_practical_improvement_opponents": best_improvements}}


def _validate_authorization(config: Mapping[str, Any], config_path: Path, base_path: Path) -> tuple[Mapping[str, Any], Path]:
    if config.get("source_id") != SOURCE_ID:
        raise NeutralPostMergePilotError("unexpected source ID")
    auth = config.get("authorization", {})
    for key in ("execution_permitted", "training_permitted", "baseline_evaluation_permitted", "heldout_evaluation_permitted"):
        if auth.get(key) is not True:
            raise NeutralPostMergePilotError(f"authorization.{key} must be true")
    for key in ("high_level_ppo_training_permitted", "four_skill_training_permitted", "combat_finetune_permitted", "tuning_permitted", "vpp_change_permitted", "guidance_change_permitted", "pid_change_permitted", "snapshot_restore", "future_state_injection", "history_padding_permitted"):
        if auth.get(key) is not False:
            raise NeutralPostMergePilotError(f"authorization.{key} must remain false")
    execution = config.get("execution", {})
    required_sha = str(execution.get("required_implementation_git_sha", ""))
    check = subprocess.run(["git", "merge-base", "--is-ancestor", required_sha, "HEAD"], cwd=ROOT, check=False)
    if check.returncode != 0 or _git_value("status", "--porcelain"):
        raise NeutralPostMergePilotError("execution requires a clean worktree and frozen implementation ancestor")
    allowed = {str(item) for item in execution.get("authorization_delta_paths", [])}
    changed = _git_changed_paths_since(required_sha)
    if changed - allowed:
        raise NeutralPostMergePilotError(f"unexpected changes after implementation freeze: {sorted(changed - allowed)}")
    authorized_files = execution.get("authorized_code_files", [])
    if not isinstance(authorized_files, Sequence) or not authorized_files:
        raise NeutralPostMergePilotError("execution requires explicit authorized code hashes")
    for entry in authorized_files:
        _require_hash(_repo_path(str(entry.get("path", ""))), entry.get("sha256"), "authorized code")
    _require_hash(base_path, execution.get("base_config_sha256"), "base config")
    validate_design(base_path)
    _require_hash(_repo_path(str(execution["runtime_template"])), execution.get("runtime_template_sha256"), "runtime template")
    registry_path = _repo_path(str(execution["runtime_registry"]))
    _require_hash(registry_path, execution.get("runtime_registry_sha256"), "runtime registry")
    registry = load_yaml(registry_path)
    if tuple(registry.get("opponents", ())) != OPPONENTS:
        raise NeutralPostMergePilotError("execution runtime registry opponent order drifted")
    for opponent in OPPONENTS:
        entry = registry["opponents"][opponent]
        if opponent != "expert":
            _require_hash(Path(str(entry["checkpoint"])), entry.get("checkpoint_sha256"), f"opponent {opponent}")
    for method in ("run_in_head_on", "frozen_fixed_head_on", "frozen_fixed_crossing"):
        entry = registry["specialists"][method]
        _require_hash(Path(str(entry["checkpoint"])), entry.get("checkpoint_sha256"), f"specialist {method}")
    output = Path(str(config["outputs"]["root"]))
    if output.exists() or not output.is_absolute():
        raise NeutralPostMergePilotError("authorized output root must be absolute, absent, and fresh")
    free_gb = shutil.disk_usage(output.parent).free / (1024**3)
    if free_gb < float(config["contract"]["min_free_disk_gb"]):
        raise NeutralPostMergePilotError("authorized output root failed disk gate")
    return registry, output


def _write_evaluation(output_root: Path, split: str, opponent: str, method: str, records: Sequence[Mapping[str, Any]], ledgers: Sequence[Mapping[str, Any]]) -> None:
    write_json(output_root / split / "records" / opponent / f"{method}.json", list(records))
    for ledger in ledgers:
        name = str(ledger["header"]["scenario"])
        write_json(output_root / split / "telemetry" / opponent / method / f"{name}.json", ledger)


def _baseline_dev(config: Mapping[str, Any], registry: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    scenarios = load_yaml(_repo_path(str(config["sources"]["dev_manifest"]))) ["scenarios"]
    all_records: dict[str, Any] = {}
    for opponent in OPPONENTS:
        all_records[opponent] = {}
        for method in ("frozen_fixed_head_on", "frozen_fixed_crossing"):
            records, ledgers = _evaluate(config, registry, scenarios, opponent, method)
            all_records[opponent][method] = records
            _write_evaluation(output_root, "dev", opponent, method, records, ledgers)
    write_json(output_root / "dev" / "baseline_summary.json", {opponent: {method: _summary(records) for method, records in methods.items()} for opponent, methods in all_records.items()})
    return all_records


def _train(config: Mapping[str, Any], registry: Mapping[str, Any], output_root: Path, baseline_dev: Mapping[str, Any]) -> dict[str, Any]:
    training = config["proposed_training"]
    random.seed(int(training["training_seed"])); np.random.seed(int(training["training_seed"])); torch.manual_seed(int(training["training_seed"]))
    agent = ThesisSharedSkillPPOAgent(_agent_config(config), device="cpu")
    distribution = load_yaml(_repo_path(str(config["sources"]["train_distribution"])))
    dev_scenarios = load_yaml(_repo_path(str(config["sources"]["dev_manifest"]))) ["scenarios"]
    checkpoint_root = output_root / "checkpoints"
    eval_steps = list(training["checkpoint_evaluation_steps"])
    next_eval = attempt = opponent_index = 0
    summaries, dev_evaluations = [], []
    stopped = False; stop_reason = None
    while agent.total_timesteps < int(training["total_timesteps"]):
        if attempt >= int(training["max_episode_attempts"]):
            stopped, stop_reason = True, "max_episode_attempts_before_training_budget"; break
        opponent = OPPONENTS[opponent_index % len(OPPONENTS)]; opponent_index += 1
        scenario = sample_train_scenario(distribution, attempt); attempt += 1
        result, _ledger, transitions = _run_episode(config, registry, scenario, opponent, "candidate_fixed_reentry_recovery", agent=agent, collect_training=True)
        last_next: np.ndarray | None = None
        for observation, action, log_prob, reward, done, value in transitions:
            agent.store_transition(observation, action, log_prob, reward, done, value)
            if agent.buffer.full:
                agent.update(None)
        summaries.append({"episode_attempt": attempt, "opponent": opponent, "scenario_signature": result["scenario_signature"], "handoff_reached": result["handoff_reached"], "candidate_steps": len(transitions), "valid_target_steps": result["valid_target_steps"], "terminal_reason": result["terminal_reason"]})
        while next_eval < len(eval_steps) and agent.total_timesteps >= int(eval_steps[next_eval]):
            if len(agent.buffer): agent.update(last_next)
            step = int(eval_steps[next_eval]); checkpoint = checkpoint_root / f"step_{step}.pt"
            agent.save(checkpoint, _checkpoint_metadata(config))
            candidates: dict[str, Any] = {}
            for opponent_id in OPPONENTS:
                records, ledgers = _evaluate(config, registry, dev_scenarios, opponent_id, "candidate_fixed_reentry_recovery", checkpoint)
                candidates[opponent_id] = records
                _write_evaluation(output_root, "dev", opponent_id, f"candidate_step_{step}", records, ledgers)
            evaluation = _evaluate_dev(config, candidates, baseline_dev)
            evaluation.update({"step": step, "checkpoint": str(checkpoint)})
            dev_evaluations.append(evaluation)
            write_json(output_root / "dev" / f"candidate_step_{step}_summary.json", evaluation)
            if not evaluation["safety_pass"]:
                stopped, stop_reason = True, f"dev_safety_stop_at_{step}"
            next_eval += 1
            if stopped: break
        if stopped: break
    if len(agent.buffer): agent.update(None)
    final_path = checkpoint_root / "last.pt"; agent.save(final_path, _checkpoint_metadata(config))
    eligible = [item for item in dev_evaluations if item["safety_pass"] and math.isfinite(float(item["selection_score_worst_opponent_mean_delta"]))]
    selected = min(eligible, key=lambda item: (float(item["selection_score_worst_opponent_mean_delta"]), int(item["step"]))) if eligible else None
    selected_path = None
    if selected is not None:
        selected_path = checkpoint_root / "best.pt"; shutil.copy2(Path(selected["checkpoint"]), selected_path)
    report = {"training_started": True, "training_steps": agent.total_timesteps, "episode_attempts": attempt, "stopped": stopped, "stop_reason": stop_reason, "train_episode_summaries": summaries, "dev_evaluations": dev_evaluations, "selected_checkpoint": str(selected_path) if selected_path else None, "selected_dev": selected, "heldout_permitted": bool(selected_path and not stopped and agent.total_timesteps == int(training["total_timesteps"]))}
    write_json(output_root / "train" / "training_summary.json", report)
    return report


def _heldout(config: Mapping[str, Any], registry: Mapping[str, Any], output_root: Path, checkpoint: Path) -> dict[str, Any]:
    scenarios = load_yaml(_repo_path(str(config["sources"]["heldout_manifest"]))) ["scenarios"]
    records: dict[str, Any] = {}
    for opponent in OPPONENTS:
        records[opponent] = {}
        for method in METHODS:
            values, ledgers = _evaluate(config, registry, scenarios, opponent, method, checkpoint if method == "candidate_fixed_reentry_recovery" else None)
            records[opponent][method] = values
            _write_evaluation(output_root, "heldout", opponent, method, values, ledgers)
    decision = analyze_heldout(config, records)
    write_json(output_root / "heldout" / "records.json", records)
    write_json(output_root / "heldout" / "decision.json", decision)
    return decision


def run(config_path: Path) -> dict[str, Any]:
    config, base_path = load_authorized_config(config_path)
    registry, output_root = _validate_authorization(config, config_path, base_path)
    output_root.mkdir(parents=True, exist_ok=False)
    try:
        write_json(output_root / "authorization_preflight.json", {"source_id": SOURCE_ID, "git_sha": _git_value("rev-parse", "HEAD"), "config_sha256": sha256_file(config_path), "base_config_sha256": sha256_file(base_path), "execution_permitted": True})
        write_json(output_root / "resolved_config.json", config)
        baseline = _baseline_dev(config, registry, output_root)
        training = _train(config, registry, output_root, baseline)
        decision = _heldout(config, registry, output_root, Path(training["selected_checkpoint"])) if training["heldout_permitted"] else None
        final = {"source_id": SOURCE_ID, "training": training, "heldout_executed": decision is not None, "decision": decision, "paper_safe": False}
        write_json(output_root / "pilot_summary.json", final)
        return final
    except Exception as error:
        write_json(output_root / "runner_failure_manifest.json", {"source_id": SOURCE_ID, "status": "execution_failed_output_preserved", "error": repr(error)})
        raise


def preflight(config_path: Path) -> dict[str, Any]:
    """Validate an authorized configuration without creating an output root."""

    config, base_path = load_authorized_config(config_path)
    registry, output_root = _validate_authorization(config, config_path, base_path)
    return {
        "source_id": SOURCE_ID,
        "mode": "authorized_preflight_no_jsbsim_no_output_creation",
        "git_sha": _git_value("rev-parse", "HEAD"),
        "execution_permitted": bool(config["authorization"]["execution_permitted"]),
        "training_permitted": bool(config["authorization"]["training_permitted"]),
        "output_root": str(output_root),
        "output_root_absent": not output_root.exists(),
        "opponents": list(registry["opponents"]),
        "methods": list(METHODS),
        "total_timesteps": int(config["proposed_training"]["total_timesteps"]),
    }
