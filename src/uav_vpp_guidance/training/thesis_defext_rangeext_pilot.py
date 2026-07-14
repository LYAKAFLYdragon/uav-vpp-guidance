"""Authorized single-skill defensive-extension feasibility pilot."""

from __future__ import annotations

from collections import deque
import copy
import hashlib
import importlib
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.evaluation.p4_v2_sampler_feasibility import classify_dynamic_state
from uav_vpp_guidance.evaluation.single_motif_continuous_66d import project_base_observation
from uav_vpp_guidance.hierarchy.five_state_observation_contract import (
    BASE_GEOMETRY_FEATURES,
    FiveStateObservationContract,
)
from uav_vpp_guidance.hierarchy.shared_intent_profiles import PROFILE_NAMES, compile_profile
from uav_vpp_guidance.hierarchy.shared_intent_validity import (
    SKILL_NAMES,
    TacticalContext,
    build_validity_mask,
)
from uav_vpp_guidance.hierarchy.specialist_policy import FrozenSpecialistPolicy
from uav_vpp_guidance.hierarchy.temporal_geometry_features import TemporalGeometryFeatureExtractor
from uav_vpp_guidance.training.thesis_shared_skill_geometry import (
    FrozenP3Encoder,
    PhaseTracker,
    intent_loss,
    physical_intent_vector,
)
from uav_vpp_guidance.training.thesis_shared_skill_policy import ThesisSharedSkillPPOAgent


ROOT = Path(__file__).resolve().parents[3]
BASE_ENV_CONFIG = ROOT / "config" / "experiment" / "jsbsim_hrl_comparison.yaml"
SOURCE_ID = "THESIS-DEFEXT-RANGEEXT-FEASIBILITY-PILOT-V1"
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
METHODS = (
    "candidate_fixed_defensive_extension",
    "frozen_fixed_head_on",
    "frozen_fixed_crossing",
)
TARGET_SKILL = "defensive_extension"
TARGET_PROFILE = "range_extension"
GRAVITY_MPS2 = 9.80665


class PilotContractError(RuntimeError):
    """Raised when an authorized pilot violates its frozen contract."""


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge_config(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_authorized_config(path: Path) -> dict[str, Any]:
    overlay = load_yaml(path)
    base_value = overlay.pop("base_config", None)
    if not base_value:
        return overlay
    base_path = Path(str(base_value))
    if not base_path.is_absolute():
        base_path = Path(path).parent / base_path
    expected = str(overlay.get("authorization", {}).get("base_config_sha256", "")).lower()
    actual = sha256_file(base_path)
    if actual != expected:
        raise PilotContractError("authorized pilot base_config SHA mismatch")
    return merge_config(load_yaml(base_path), overlay)


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _git_clean() -> bool:
    return not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()


def _git_is_ancestor(commit: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", str(commit), "HEAD"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def validate_authorization(config: Mapping[str, Any], *, require_fresh_output: bool = True) -> dict[str, Any]:
    if config.get("source_id") != SOURCE_ID:
        raise PilotContractError("unexpected pilot source_id")
    authorization = config.get("authorization", {})
    required_true = (
        "training_permitted",
        "pilot_execution_permitted",
        "baseline_evaluation_permitted",
        "heldout_evaluation_permitted",
    )
    if not all(authorization.get(key) is True for key in required_true):
        raise PilotContractError("pilot authorization is incomplete")
    required_false = (
        "high_level_ppo_training_permitted",
        "four_skill_training_permitted",
        "combat_finetune_permitted",
    )
    if not all(authorization.get(key) is False for key in required_false):
        raise PilotContractError("pilot authorization expands beyond the single skill")
    if authorization.get("scope") != "single_defensive_extension_feasibility_pilot_only":
        raise PilotContractError("pilot authorization scope mismatch")
    current_sha = _git_sha()
    required_sha = str(authorization.get("required_implementation_git_sha", ""))
    if not required_sha or not _git_is_ancestor(required_sha) or not _git_clean():
        raise PilotContractError("authorized pilot requires the frozen implementation ancestor and a clean HEAD")
    code_files = authorization.get("authorized_code_files", [])
    if not isinstance(code_files, Sequence) or not code_files:
        raise PilotContractError("authorized pilot requires explicit code-file hashes")
    for entry in code_files:
        if not isinstance(entry, Mapping):
            raise PilotContractError("authorized code-file entry must be a mapping")
        path = _repo_path(str(entry.get("path", "")))
        if not path.is_file() or sha256_file(path) != entry.get("sha256"):
            raise PilotContractError(f"authorized code-file SHA mismatch: {path}")

    fixed = config["fixed_contract"]
    if (
        fixed.get("skill") != TARGET_SKILL
        or fixed.get("profile") != TARGET_PROFILE
        or fixed.get("routing_enabled") is not False
        or fixed.get("high_level_policy_present") is not False
        or fixed.get("encoder", {}).get("trainable") is not False
    ):
        raise PilotContractError("fixed skill/profile/encoder/routing contract drifted")
    if tuple(config["opponents"]["order"]) != OPPONENTS or set(config["methods"]) != set(METHODS):
        raise PilotContractError("method or opponent matrix drifted")
    if int(config["proposed_training"]["total_timesteps"]) != 50000:
        raise PilotContractError("training budget drifted")
    output_root = Path(config["outputs"]["root"])
    if require_fresh_output and output_root.exists():
        raise PilotContractError("pilot output root must be fresh and absent")

    paths = {
        "p3": Path(fixed["encoder"]["checkpoint"]),
        "runtime_registry": _repo_path(config["sources"]["pilot_runtime_registry"]),
        "runtime_template": _repo_path(config["sources"]["runtime_template"]),
        "dev_manifest": _repo_path(config["sources"]["dev_manifest"]),
        "heldout_manifest": _repo_path(config["sources"]["heldout_manifest"]),
    }
    expected_hashes = {
        "p3": fixed["encoder"]["sha256"],
        "runtime_registry": config["sources"]["pilot_runtime_registry_sha256"],
        "runtime_template": config["sources"]["runtime_template_sha256"],
    }
    for key, expected in expected_hashes.items():
        if not paths[key].is_file() or sha256_file(paths[key]) != expected:
            raise PilotContractError(f"pilot asset SHA mismatch: {key}")
    return {
        "source_id": SOURCE_ID,
        "git_sha": current_sha,
        "git_clean": True,
        "output_root": str(output_root),
        "paths": {key: str(value) for key, value in paths.items()},
    }


def _import_class(class_path: str):
    module_name, class_name = str(class_path).replace("src.", "", 1).rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


def _instantiate_opponent(entry: Mapping[str, Any]) -> object:
    cls = _import_class(str(entry["class"]))
    kwargs: dict[str, Any] = {}
    if entry.get("checkpoint"):
        kwargs["checkpoint_path"] = str(entry["checkpoint"])
    if "config" in entry:
        kwargs["config"] = copy.deepcopy(entry["config"])
    if "device" in entry:
        kwargs["device"] = str(entry["device"])
    if "invert_observation" in entry:
        kwargs["invert_observation"] = bool(entry["invert_observation"])
    return cls(**kwargs)


def _runtime(config: Mapping[str, Any]) -> Mapping[str, Any]:
    template = load_yaml(_repo_path(config["sources"]["runtime_template"]))
    return template["runtime"]


def _environment_config(config: Mapping[str, Any], opponent_id: str) -> dict[str, Any]:
    base = copy.deepcopy(load_yaml(BASE_ENV_CONFIG))
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
    return base


def _build_env(config: Mapping[str, Any], registry: Mapping[str, Any], opponent_id: str) -> CloseRangeTrackingEnv:
    entry = registry["opponents"][opponent_id]
    env = CloseRangeTrackingEnv(
        _environment_config(config, opponent_id),
        opponent_policy=_instantiate_opponent(entry),
        opponent_config={"stage": opponent_id, **copy.deepcopy(entry)},
    )
    return env


def _specialist(registry: Mapping[str, Any], key: str, *, device: str = "cpu") -> FrozenSpecialistPolicy:
    entry = registry["specialists"][key]
    path = Path(entry["checkpoint"])
    if sha256_file(path) != entry["checkpoint_sha256"]:
        raise PilotContractError(f"specialist checkpoint SHA mismatch: {key}")
    return FrozenSpecialistPolicy(
        checkpoint_path=str(path),
        config_path=str(_repo_path(entry["config_path"])),
        device=device,
    )


def _base_map(base: np.ndarray) -> dict[str, float]:
    return {name: float(base[index]) for index, name in enumerate(BASE_GEOMETRY_FEATURES)}


def _angle_deg(sine: float, cosine: float) -> float:
    return abs(math.degrees(math.atan2(float(sine), float(cosine))))


def _history_frame(base: np.ndarray) -> dict[str, float]:
    values = _base_map(base)
    return {
        "range_rate_mps": values["range_rate_mps"],
        "aa_deg": _angle_deg(values["aa_sin"], values["aa_cos"]),
        "ata_deg": _angle_deg(values["ata_sin"], values["ata_cos"]),
        "specific_energy_height_m": values["own_altitude"] + values["own_speed"] ** 2 / (2.0 * GRAVITY_MPS2),
        "altitude_m": values["own_altitude"],
    }


def _tactical_context(base: np.ndarray, state: str, phase: str) -> TacticalContext:
    values = _base_map(base)
    own_energy = values["own_altitude"] + values["own_speed"] ** 2 / (2.0 * GRAVITY_MPS2)
    target_energy = values["target_altitude"] + values["target_speed"] ** 2 / (2.0 * GRAVITY_MPS2)
    energy_delta = own_energy - target_energy
    energy_state = "advantage" if energy_delta > 100.0 else "disadvantage" if energy_delta < -100.0 else "balanced"
    altitude_delta = values["own_altitude"] - values["target_altitude"]
    altitude_state = "own_above" if altitude_delta > 100.0 else "own_below" if altitude_delta < -100.0 else "co_altitude"
    return TacticalContext(state, phase, energy_state, altitude_state)


def _is_ego_failure(info: Mapping[str, Any]) -> bool:
    reason = str(info.get("reason") or info.get("combat_reason") or "").lower()
    return any(token in reason for token in ("ego_crash", "ego_out_of_bounds", "crash", "out_of_bounds")) and not reason.startswith("target_")


class ContinuousHandoffEpisode:
    """One legal reset, frozen run-in, then one fixed post-handoff controller."""

    def __init__(
        self,
        env: CloseRangeTrackingEnv,
        run_in_policy: FrozenSpecialistPolicy,
        encoder: FrozenP3Encoder,
        config: Mapping[str, Any],
    ) -> None:
        self.env = env
        self.run_in_policy = run_in_policy
        self.encoder = encoder
        self.config = config
        self.contract = FiveStateObservationContract()
        self.history: deque[np.ndarray] = deque(maxlen=10)
        self.history_features = TemporalGeometryFeatureExtractor(10)
        self.phase_tracker = PhaseTracker()
        self.observation_schema: Mapping[str, Any] = {}
        self.observation: Mapping[str, Any] | None = None
        self.base: np.ndarray | None = None
        self.phase = "pre_merge"
        self.dynamic_state = "unknown"
        self.first_pass_complete = False
        self.last_info: Mapping[str, Any] = {}
        self.total_steps = 0
        self.handoff_step: int | None = None

    def _observe(self, observation: Mapping[str, Any]) -> None:
        base = project_base_observation(observation["observation_vector"], self.observation_schema)
        self.history.append(base.copy())
        self.history_features.update(_history_frame(base))
        values = _base_map(base)
        self.phase = self.phase_tracker.update(values["range_m"], values["range_rate_mps"])
        self.dynamic_state = classify_dynamic_state(
            _angle_deg(values["aa_sin"], values["aa_cos"]),
            _angle_deg(values["ata_sin"], values["ata_cos"]),
        )
        if self.dynamic_state == "unknown":
            raise PilotContractError("dynamic taxonomy became unknown")
        self.observation = observation
        self.base = base

    def _target_pair_allowed(self) -> bool:
        assert self.base is not None
        context = _tactical_context(self.base, self.dynamic_state, self.phase)
        validity = build_validity_mask(context)["mask"]
        return bool(validity[SKILL_NAMES.index(TARGET_SKILL), PROFILE_NAMES.index(TARGET_PROFILE)])

    def _trigger_ready(self) -> bool:
        return bool(
            len(self.history) == 10
            and self.first_pass_complete
            and self.dynamic_state == "disadvantage"
            and self.phase in {"post_merge", "re_entry"}
            and self._target_pair_allowed()
            and self.last_info.get("prediction_valid") is True
            and self.last_info.get("prediction_fallback") is not True
        )

    def policy_observation(self) -> np.ndarray:
        if self.observation is None or self.base is None or len(self.history) != 10:
            raise PilotContractError("candidate policy requested before exact 10-frame history")
        profile = compile_profile(TARGET_PROFILE)
        result = self.contract.compose(
            base_geometry=self.base,
            explicit_history=self.history_features.features(),
            intent_targets=profile["target_vector"],
            intent_weights=profile["weight_vector"],
            temporal_embedding=self.encoder.encode(np.stack(tuple(self.history), axis=0)),
        )
        if result.shape != (66,) or not np.isfinite(result).all():
            raise PilotContractError("candidate 66-D observation is invalid")
        return result

    def reset_to_handoff(self, scenario: Mapping[str, Any], seed: int) -> tuple[np.ndarray | None, dict[str, Any]]:
        self.history.clear()
        self.history_features.reset()
        self.phase_tracker = PhaseTracker()
        self.first_pass_complete = False
        self.last_info = {}
        self.total_steps = 0
        self.handoff_step = None
        observation = self.env.reset(scenario=copy.deepcopy(dict(scenario)), seed=int(seed))
        if self.env._backend != "jsbsim":
            raise PilotContractError("pilot prohibits JSBSim backend fallback")
        if hasattr(self.env, "set_runtime_specialist_context"):
            self.env.set_runtime_specialist_context(
                specialist_key="head_on",
                specialist_profile=None,
                specialist_mode_name="head_on_specialist",
            )
        self.observation_schema = copy.deepcopy(observation["observation_schema"])
        self._observe(observation)
        while self.total_steps < self.env.max_steps:
            if self._trigger_ready():
                self.handoff_step = self.total_steps + 1
                return self.policy_observation(), {
                    "handoff_reached": True,
                    "handoff_step": self.handoff_step,
                    "phase": self.phase,
                    "dynamic_state": self.dynamic_state,
                }
            assert self.observation is not None
            action = self.run_in_policy.get_deterministic_action(self.observation["observation_vector"])
            observation, _reward, terminated, truncated, info = self.env.step(action)
            self.total_steps += 1
            self.last_info = info
            self.first_pass_complete = bool(info.get("first_pass_complete", False))
            self._observe(observation)
            if bool(info.get("backend_fallback_occurred", False)):
                raise PilotContractError("pilot run-in reported backend fallback")
            if terminated or truncated:
                return None, {
                    "handoff_reached": False,
                    "terminal_reason": str(info.get("reason") or "terminated_before_handoff"),
                    "steps": self.total_steps,
                }
        return None, {"handoff_reached": False, "terminal_reason": "horizon_before_handoff", "steps": self.total_steps}

    def set_handoff_context(
        self,
        *,
        specialist_key: str,
        specialist_profile: str | None,
        specialist_mode_name: str,
    ) -> None:
        if self.handoff_step is None:
            raise PilotContractError("cannot set handoff context before handoff")
        if hasattr(self.env, "set_runtime_specialist_context"):
            self.env.set_runtime_specialist_context(
                specialist_key=specialist_key,
                specialist_profile=specialist_profile,
                specialist_mode_name=specialist_mode_name,
            )

    def step(self, action: np.ndarray) -> tuple[np.ndarray | None, float, bool, dict[str, Any]]:
        if self.observation is None or self.base is None or self.handoff_step is None:
            raise PilotContractError("episode has not reached handoff")
        normalized_action = np.asarray(action, dtype=np.float32).reshape(-1)
        if normalized_action.shape != (3,) or not np.isfinite(normalized_action).all():
            raise PilotContractError("non-finite candidate/baseline 3-D action")
        if np.any(np.abs(normalized_action) > 1.0 + 1e-6):
            raise PilotContractError("candidate/baseline action exceeded normalized bounds")
        pre_phase = self.phase
        pre_state = self.dynamic_state
        pre_first_pass = self.first_pass_complete
        pre_pair_allowed = self._target_pair_allowed()
        pre_prediction_valid = self.last_info.get("prediction_valid") is True
        pre_prediction_fallback = self.last_info.get("prediction_fallback") is True
        before_loss = intent_loss(self.observation, TARGET_PROFILE, clip=2.0)
        before_intent = physical_intent_vector(self.observation)
        observation, _base_reward, terminated, truncated, info = self.env.step(normalized_action)
        self.total_steps += 1
        self.last_info = info
        self.first_pass_complete = bool(info.get("first_pass_complete", False))
        self._observe(observation)
        after_loss = intent_loss(self.observation, TARGET_PROFILE, clip=2.0)
        after_intent = physical_intent_vector(self.observation)
        reward_cfg = self.config["proposed_training"]["geometry_reward"]
        progress = float(np.clip(before_loss - after_loss, -float(reward_cfg["intent_error_clip"]), float(reward_cfg["intent_error_clip"])))
        alignment = max(0.0, 1.0 - min(after_loss, float(reward_cfg["intent_error_clip"])) / float(reward_cfg["intent_error_clip"]))
        saturation_count = sum(bool(info.get(key, False)) for key in ("nz_saturated", "roll_rate_saturated", "throttle_saturated"))
        ego_failure = _is_ego_failure(info)
        reward = (
            float(reward_cfg["progress_weight"]) * progress
            + float(reward_cfg["alignment_weight"]) * alignment
            - float(reward_cfg["physical_command_saturation_penalty"]) * saturation_count
            - (float(reward_cfg["terminal_ego_crash_oob_penalty"]) if ego_failure else 0.0)
        )
        valid_target_step = bool(
            pre_first_pass
            and pre_state == "disadvantage"
            and pre_phase in {"post_merge", "re_entry"}
            and pre_pair_allowed
            and pre_prediction_valid
            and not pre_prediction_fallback
        )
        done = bool(terminated or truncated or self.total_steps >= self.env.max_steps)
        next_observation = None if done else self.policy_observation()
        own_state = info.get("own_state", {}) if isinstance(info.get("own_state", {}), Mapping) else {}
        metric = {
            "step": self.total_steps,
            "phase": pre_phase,
            "dynamic_state": pre_state,
            "valid_target_step": valid_target_step,
            "before_intent_loss": float(before_loss),
            "after_intent_loss": float(after_loss),
            "intent_progress": progress,
            "range_m": float(before_intent[2]),
            "range_rate_mps": float(before_intent[3]),
            "specific_energy_height_delta_m": float(after_intent[4]),
            "altitude_delta_m": float(after_intent[5]),
            "target_in_attack_zone": bool(info.get("target_in_attack_zone", False)),
            "ego_in_attack_zone": bool(info.get("ego_in_attack_zone", False)),
            "ego_hp": float(info.get("ego_hp", 100.0)),
            "target_hp": float(info.get("target_hp", 100.0)),
            "vp_forward_bias_m": float(info.get("vp_forward_bias_m", np.nan)),
            "vp_lateral_bias_m": float(info.get("vp_lateral_bias_m", np.nan)),
            "vp_vertical_bias_m": float(info.get("vp_vertical_bias_m", np.nan)),
            "nz_cmd": float(info.get("nz_cmd", np.nan)),
            "actual_nz_g": float(own_state.get("nz_g", info.get("nz_g", np.nan))),
            "ego_attack_aoa_deg": float(info.get("ego_attack_aoa_deg", np.nan)),
            "saturation_flag": bool(info.get("saturation_flag", False)),
            "reward": float(reward),
            "terminal_reason": str(info.get("reason") or "timeout"),
            "ego_failure": ego_failure,
            "combat_outcome": info.get("combat_outcome"),
            "win": bool(info.get("win", False)),
            "loss": bool(info.get("loss", False)),
            "draw": bool(info.get("draw", False)),
            "backend_fallback_occurred": bool(info.get("backend_fallback_occurred", False)),
        }
        if metric["backend_fallback_occurred"]:
            raise PilotContractError("pilot reported JSBSim backend fallback")
        return next_observation, float(reward), done, metric

    def close(self) -> None:
        self.env.close()


def _normalize_heading(value: float) -> float:
    return float(value % 360.0)


def _horizontal_heading(los_azimuth: float, horizontal_fraction: float, desired_angle: float, side: int) -> float:
    ratio = math.cos(math.radians(desired_angle)) / horizontal_fraction
    if ratio < -1.0 or ratio > 1.0:
        raise PilotContractError("sampled train geometry is physically infeasible")
    delta = math.degrees(math.acos(max(-1.0, min(1.0, ratio))))
    return _normalize_heading(los_azimuth + side * delta)


def sample_train_scenario(distribution: Mapping[str, Any], episode_index: int) -> dict[str, Any]:
    contract = distribution["sampling_contract"]
    rng = np.random.default_rng(int(contract["seed"]) + int(episode_index))
    own_altitude = float(rng.uniform(*contract["own_altitude_m"]))
    altitude_diff = float(rng.uniform(*contract["target_minus_own_altitude_m"]))
    initial_range = float(rng.uniform(*contract["initial_range_m"]))
    own_speed = float(rng.uniform(*contract["own_speed_mps"]))
    target_speed = float(rng.uniform(*contract["target_speed_mps"]))
    if target_speed - own_speed < float(contract["require_target_faster_than_own_by_mps"]):
        target_speed = own_speed + float(contract["require_target_faster_than_own_by_mps"])
    mirror = contract["mirror_signs"][episode_index % len(contract["mirror_signs"])]
    sign = -1 if mirror == "negative" else 1
    los_azimuth = sign * float(rng.uniform(*contract["los_azimuth_abs_deg"]))
    own_angle = float(rng.uniform(*contract["own_to_target_los_angle_deg"]))
    target_angle = float(rng.uniform(*contract["target_velocity_to_own_los_angle_deg"]))
    horizontal = math.sqrt(initial_range**2 - altitude_diff**2)
    fraction = horizontal / initial_range
    own_heading = _horizontal_heading(los_azimuth, fraction, own_angle, -sign)
    target_heading = _horizontal_heading(_normalize_heading(los_azimuth + 180.0), fraction, target_angle, sign)
    azimuth = math.radians(los_azimuth)
    return {
        "name": f"defext_rangeext_train_{episode_index:06d}",
        "own_init": {
            "position_m": [0.0, 0.0, own_altitude],
            "velocity_mps": own_speed,
            "heading_deg": own_heading,
        },
        "target_init": {
            "position_m": [horizontal * math.cos(azimuth), horizontal * math.sin(azimuth), own_altitude + altitude_diff],
            "velocity_mps": target_speed,
            "heading_deg": target_heading,
        },
        "metadata": {
            "split": "train",
            "initial_class": "disadvantage",
            "height_condition": "continuous",
            "mirror_sign": mirror,
            "scenario_signature": f"train_{episode_index:06d}",
            "scenario_seed": int(contract["seed"]) + int(episode_index),
        },
    }


def _agent_config(config: Mapping[str, Any]) -> dict[str, Any]:
    training = config["proposed_training"]
    return {"training": {"ppo": copy.deepcopy(training["ppo"]), "policy": copy.deepcopy(training["policy"])}}


def _checkpoint_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_id": SOURCE_ID,
        "skill_name": TARGET_SKILL,
        "profile_name": TARGET_PROFILE,
        "observation_dim": 66,
        "action_dim": 3,
        "p3_encoder_sha256": config["fixed_contract"]["encoder"]["sha256"],
        "encoder_trainable": False,
        "routing_enabled": False,
        "training_seed": int(config["proposed_training"]["training_seed"]),
    }


def _episode_result(
    *,
    opponent: str,
    method: str,
    scenario: Mapping[str, Any],
    handoff_meta: Mapping[str, Any],
    metrics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    valid = [item for item in metrics if item.get("valid_target_step")]
    window = valid[:20]
    final = metrics[-1] if metrics else {}
    return {
        "opponent": opponent,
        "method": method,
        "scenario": scenario["name"],
        "scenario_signature": scenario["metadata"].get("scenario_signature"),
        "mirror_sign": scenario["metadata"].get("mirror_sign"),
        "scenario_seed": scenario["metadata"].get("scenario_seed"),
        "handoff_reached": bool(handoff_meta.get("handoff_reached", False)),
        "handoff_step": handoff_meta.get("handoff_step"),
        "valid_target_steps": len(valid),
        "qualifying": len(window) == 20,
        "intent_loss_auc20": float(np.mean([item["after_intent_loss"] for item in window])) if len(window) == 20 else None,
        "range_opening_fraction_auc20": float(np.mean([float(item["range_rate_mps"] > 0.0) for item in window])) if window else None,
        "specific_energy_height_delta_m_at_20": float(window[-1]["specific_energy_height_delta_m"]) if len(window) == 20 else None,
        "target_attack_zone_exposure_fraction_auc20": float(np.mean([float(item["target_in_attack_zone"]) for item in window])) if window else None,
        "ego_attack_zone_exposure_fraction_auc20": float(np.mean([float(item["ego_in_attack_zone"]) for item in window])) if window else None,
        "terminal_reason": final.get("terminal_reason", handoff_meta.get("terminal_reason", "no_handoff")),
        "ego_failure": bool(final.get("ego_failure", False)),
        "win": bool(final.get("win", False)),
        "loss": bool(final.get("loss", False)),
        "draw": bool(final.get("draw", False)),
        "ego_hp": final.get("ego_hp"),
        "target_hp": final.get("target_hp"),
    }


def evaluate_method(
    config: Mapping[str, Any],
    registry: Mapping[str, Any],
    scenarios: Sequence[Mapping[str, Any]],
    opponent_id: str,
    method: str,
    *,
    checkpoint: Path | None = None,
) -> list[dict[str, Any]]:
    env = _build_env(config, registry, opponent_id)
    run_in = _specialist(registry, "run_in_head_on")
    encoder = FrozenP3Encoder(
        Path(config["fixed_contract"]["encoder"]["checkpoint"]),
        config["fixed_contract"]["encoder"]["sha256"],
    )
    episode = ContinuousHandoffEpisode(env, run_in, encoder, config)
    candidate = None
    baseline = None
    if method == "candidate_fixed_defensive_extension":
        if checkpoint is None:
            raise PilotContractError("candidate evaluation requires a checkpoint")
        candidate = ThesisSharedSkillPPOAgent(_agent_config(config), device="cpu")
        candidate.load_strict(checkpoint, _checkpoint_metadata(config))
    elif method == "frozen_fixed_head_on":
        baseline = _specialist(registry, "frozen_fixed_head_on")
    elif method == "frozen_fixed_crossing":
        baseline = _specialist(registry, "frozen_fixed_crossing")
    else:
        raise PilotContractError(f"unknown evaluation method: {method}")
    results: list[dict[str, Any]] = []
    try:
        for scenario in scenarios:
            seed = int(scenario["metadata"]["scenario_seed"])
            observation, handoff = episode.reset_to_handoff(scenario, seed)
            metrics: list[dict[str, Any]] = []
            if observation is not None:
                if method == "candidate_fixed_defensive_extension":
                    episode.set_handoff_context(
                        specialist_key=TARGET_SKILL,
                        specialist_profile=TARGET_PROFILE,
                        specialist_mode_name="defensive_extension_specialist",
                    )
                elif method == "frozen_fixed_head_on":
                    episode.set_handoff_context(
                        specialist_key="head_on",
                        specialist_profile=None,
                        specialist_mode_name="head_on_specialist",
                    )
                else:
                    episode.set_handoff_context(
                        specialist_key="crossing_feasible",
                        specialist_profile=None,
                        specialist_mode_name="crossing_specialist",
                    )
                done = False
                while not done:
                    if candidate is not None:
                        action = candidate.select_action(observation, deterministic=True)[0]
                    else:
                        assert baseline is not None and episode.observation is not None
                        action = baseline.get_deterministic_action(episode.observation["observation_vector"])
                    next_observation, _reward, done, metric = episode.step(action)
                    metrics.append(metric)
                    if not done:
                        if next_observation is None:
                            raise PilotContractError("unfinished evaluation lost 66-D observation")
                        observation = next_observation
            results.append(
                _episode_result(
                    opponent=opponent_id,
                    method=method,
                    scenario=scenario,
                    handoff_meta=handoff,
                    metrics=metrics,
                )
            )
    finally:
        episode.close()
    return results


def summarize_method(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    qualifying = [item for item in records if item.get("qualifying")]
    return {
        "episodes": len(records),
        "handoff_episodes": sum(bool(item.get("handoff_reached")) for item in records),
        "qualifying_episodes": len(qualifying),
        "valid_target_steps": sum(int(item.get("valid_target_steps", 0)) for item in records),
        "mean_intent_loss_auc20": float(np.mean([item["intent_loss_auc20"] for item in qualifying])) if qualifying else None,
        "ego_crash_oob_rate": sum(bool(item.get("ego_failure")) for item in records) / max(len(records), 1),
        "win_rate_descriptive": sum(bool(item.get("win")) for item in records) / max(len(records), 1),
    }


def _paired_deltas(
    candidate: Sequence[Mapping[str, Any]], baseline: Sequence[Mapping[str, Any]]
) -> list[float]:
    baseline_map = {
        item["scenario_signature"]: item
        for item in baseline
        if item.get("qualifying") and item.get("intent_loss_auc20") is not None
    }
    values = []
    for item in candidate:
        other = baseline_map.get(item["scenario_signature"])
        if item.get("qualifying") and item.get("intent_loss_auc20") is not None and other is not None:
            values.append(float(item["intent_loss_auc20"]) - float(other["intent_loss_auc20"]))
    return values


def evaluate_dev_checkpoint(
    candidate_records: Mapping[str, Sequence[Mapping[str, Any]]],
    baseline_records: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> dict[str, Any]:
    per_opponent = {}
    worst_delta = -float("inf")
    safety_pass = True
    for opponent in OPPONENTS:
        candidate = candidate_records[opponent]
        head = baseline_records[opponent]["frozen_fixed_head_on"]
        deltas = _paired_deltas(candidate, head)
        mean_delta = float(np.mean(deltas)) if deltas else None
        candidate_summary = summarize_method(candidate)
        head_summary = summarize_method(head)
        crash_delta = candidate_summary["ego_crash_oob_rate"] - head_summary["ego_crash_oob_rate"]
        opponent_safety = crash_delta <= 0.05
        safety_pass = safety_pass and opponent_safety
        if mean_delta is None:
            worst_delta = float("inf")
        else:
            worst_delta = max(worst_delta, mean_delta)
        per_opponent[opponent] = {
            "candidate": candidate_summary,
            "head_on": head_summary,
            "paired_count": len(deltas),
            "mean_paired_delta_vs_head_on": mean_delta,
            "ego_crash_oob_delta_vs_head_on": crash_delta,
            "safety_pass": opponent_safety,
        }
    return {
        "selection_score_worst_opponent_mean_delta": worst_delta,
        "safety_pass": safety_pass,
        "per_opponent": per_opponent,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_baseline_dev(
    config: Mapping[str, Any], registry: Mapping[str, Any], output_root: Path
) -> dict[str, Any]:
    scenarios = load_yaml(_repo_path(config["sources"]["dev_manifest"]))["scenarios"]
    result: dict[str, Any] = {}
    for opponent in OPPONENTS:
        result[opponent] = {}
        for method in ("frozen_fixed_head_on", "frozen_fixed_crossing"):
            records = evaluate_method(config, registry, scenarios, opponent, method)
            result[opponent][method] = records
    write_json(output_root / "dev" / "baseline_records.json", result)
    write_json(
        output_root / "dev" / "baseline_summary.json",
        {
            opponent: {method: summarize_method(records) for method, records in methods.items()}
            for opponent, methods in result.items()
        },
    )
    return result


def train_candidate(
    config: Mapping[str, Any],
    registry: Mapping[str, Any],
    output_root: Path,
    baseline_dev: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> dict[str, Any]:
    training = config["proposed_training"]
    total_timesteps = int(training["total_timesteps"])
    eval_steps = list(training["checkpoint_evaluation_steps"])
    distribution = load_yaml(_repo_path(config["sources"]["train_distribution"]))
    dev_scenarios = load_yaml(_repo_path(config["sources"]["dev_manifest"]))["scenarios"]
    random.seed(int(training["training_seed"]))
    np.random.seed(int(training["training_seed"]))
    torch.manual_seed(int(training["training_seed"]))
    agent = ThesisSharedSkillPPOAgent(_agent_config(config), device="cpu")
    encoder = FrozenP3Encoder(
        Path(config["fixed_contract"]["encoder"]["checkpoint"]),
        config["fixed_contract"]["encoder"]["sha256"],
    )
    checkpoint_root = output_root / "checkpoints"
    train_summaries = []
    dev_evaluations = []
    next_eval_index = 0
    episode_attempt = 0
    opponent_index = 0
    stopped = False
    stop_reason = None
    episodes = {
        opponent: ContinuousHandoffEpisode(
            _build_env(config, registry, opponent),
            _specialist(registry, "run_in_head_on"),
            encoder,
            config,
        )
        for opponent in OPPONENTS
    }
    try:
        while agent.total_timesteps < total_timesteps:
            if episode_attempt >= int(training["max_episode_attempts"]):
                stopped = True
                stop_reason = "max_episode_attempts_before_training_budget"
                break
            opponent = OPPONENTS[opponent_index % len(OPPONENTS)]
            opponent_index += 1
            scenario = sample_train_scenario(distribution, episode_attempt)
            episode_attempt += 1
            episode = episodes[opponent]
            metrics: list[dict[str, Any]] = []
            observation, handoff = episode.reset_to_handoff(
                scenario, int(scenario["metadata"]["scenario_seed"])
            )
            if observation is not None:
                episode.set_handoff_context(
                    specialist_key=TARGET_SKILL,
                    specialist_profile=TARGET_PROFILE,
                    specialist_mode_name="defensive_extension_specialist",
                )
                done = False
                next_observation: np.ndarray | None = None
                while not done and agent.total_timesteps < total_timesteps:
                    action, log_prob, value = agent.select_action(observation, deterministic=False)
                    next_observation, reward, done, metric = episode.step(action)
                    agent.store_transition(observation, action, log_prob, reward, done, value)
                    metrics.append(metric)
                    if agent.buffer.full:
                        agent.update(None if done else next_observation)
                    if not done:
                        if next_observation is None:
                            raise PilotContractError("unfinished training rollout lost 66-D observation")
                        observation = next_observation
                    while next_eval_index < len(eval_steps) and agent.total_timesteps >= eval_steps[next_eval_index]:
                        if len(agent.buffer):
                            agent.update(None if done else next_observation)
                        step = int(eval_steps[next_eval_index])
                        checkpoint = checkpoint_root / f"step_{step}.pt"
                        agent.save(checkpoint, _checkpoint_metadata(config))
                        candidate_records = {
                            item: evaluate_method(config, registry, dev_scenarios, item, "candidate_fixed_defensive_extension", checkpoint=checkpoint)
                            for item in OPPONENTS
                        }
                        evaluation = evaluate_dev_checkpoint(candidate_records, baseline_dev)
                        evaluation["step"] = step
                        evaluation["checkpoint"] = str(checkpoint)
                        dev_evaluations.append(evaluation)
                        write_json(output_root / "dev" / f"candidate_step_{step}_records.json", candidate_records)
                        write_json(output_root / "dev" / f"candidate_step_{step}_summary.json", evaluation)
                        if not evaluation["safety_pass"]:
                            stopped = True
                            stop_reason = f"dev_safety_stop_at_{step}"
                        next_eval_index += 1
                        if stopped:
                            break
                    if stopped:
                        break
            train_summaries.append(
                {
                    "episode_attempt": episode_attempt,
                    "opponent": opponent,
                    "scenario_signature": scenario["metadata"]["scenario_signature"],
                    "handoff_reached": bool(handoff.get("handoff_reached", False)),
                    "candidate_steps": len(metrics),
                    "valid_target_steps": sum(bool(item["valid_target_step"]) for item in metrics),
                    "terminal_reason": metrics[-1]["terminal_reason"] if metrics else handoff.get("terminal_reason"),
                }
            )
            if stopped:
                break
    finally:
        for episode in episodes.values():
            episode.close()
    if len(agent.buffer):
        agent.update(None)
    final_checkpoint = checkpoint_root / "last.pt"
    agent.save(final_checkpoint, _checkpoint_metadata(config))
    eligible = [item for item in dev_evaluations if item["safety_pass"] and math.isfinite(float(item["selection_score_worst_opponent_mean_delta"]))]
    selected = min(eligible, key=lambda item: (float(item["selection_score_worst_opponent_mean_delta"]), int(item["step"]))) if eligible else None
    selected_path = None
    if selected is not None:
        selected_path = checkpoint_root / "best.pt"
        shutil.copy2(Path(selected["checkpoint"]), selected_path)
    report = {
        "training_started": True,
        "training_steps": agent.total_timesteps,
        "episode_attempts": episode_attempt,
        "stopped": stopped,
        "stop_reason": stop_reason,
        "train_episode_summaries": train_summaries,
        "dev_evaluations": dev_evaluations,
        "selected_checkpoint": str(selected_path) if selected_path else None,
        "selected_dev": selected,
        "heldout_permitted": bool(selected_path is not None and not stopped and agent.total_timesteps == total_timesteps),
    }
    write_json(output_root / "training_summary.json", report)
    return report


def _mean_delta(candidate: Sequence[Mapping[str, Any]], baseline: Sequence[Mapping[str, Any]]) -> tuple[int, float | None]:
    deltas = _paired_deltas(candidate, baseline)
    return len(deltas), float(np.mean(deltas)) if deltas else None


def analyze_heldout(
    config: Mapping[str, Any],
    records: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> dict[str, Any]:
    gates = config["gates"]
    per_opponent = {}
    head_improvements = 0
    best_improvements = 0
    all_contract = True
    all_safety = True
    all_head_noninferior = True
    remaining_best_noninferior = True
    for opponent in OPPONENTS:
        candidate = records[opponent]["candidate_fixed_defensive_extension"]
        head = records[opponent]["frozen_fixed_head_on"]
        crossing = records[opponent]["frozen_fixed_crossing"]
        candidate_summary = summarize_method(candidate)
        head_summary = summarize_method(head)
        crossing_summary = summarize_method(crossing)
        paired_head_count, head_delta = _mean_delta(candidate, head)
        paired_cross_count, crossing_delta = _mean_delta(candidate, crossing)
        baseline_means = {
            "frozen_fixed_head_on": head_summary["mean_intent_loss_auc20"],
            "frozen_fixed_crossing": crossing_summary["mean_intent_loss_auc20"],
        }
        valid_baselines = {key: value for key, value in baseline_means.items() if value is not None}
        best_name = min(valid_baselines, key=valid_baselines.get) if valid_baselines else None
        best_delta = head_delta if best_name == "frozen_fixed_head_on" else crossing_delta if best_name else None
        contract_min = gates["per_opponent_contract_minimum"]
        claim_ready = gates["per_opponent_claim_ready_coverage"]
        signatures = {item["scenario_signature"] for item in candidate if item.get("qualifying")}
        mirrors = {item["mirror_sign"] for item in candidate if item.get("qualifying")}
        contract_pass = (
            candidate_summary["qualifying_episodes"] >= int(contract_min["qualifying_paired_episodes"])
            and candidate_summary["valid_target_steps"] >= int(contract_min["valid_target_steps"])
            and len(signatures) >= int(contract_min["distinct_scenario_signatures"])
            and len(mirrors) >= int(contract_min["distinct_mirror_signs"])
            and candidate_summary["qualifying_episodes"] >= int(claim_ready["qualifying_paired_episodes"])
            and candidate_summary["valid_target_steps"] >= int(claim_ready["valid_target_steps"])
        )
        crash_delta = candidate_summary["ego_crash_oob_rate"] - head_summary["ego_crash_oob_rate"]
        safety_pass = crash_delta <= float(gates["safety_noninferiority_vs_head_on"]["ego_crash_oob_rate_delta_max"])
        head_noninferior = head_delta is not None and head_delta <= float(gates["geometry_noninferiority_vs_head_on"]["paired_intent_loss_auc20_delta_max"])
        head_improved = head_delta is not None and head_delta <= float(gates["practical_improvement_vs_head_on"]["paired_intent_loss_auc20_delta_max"])
        best_improved = best_delta is not None and best_delta <= float(gates["practical_improvement_vs_best_existing_specialist"]["paired_intent_loss_auc20_delta_max"])
        best_noninferior = best_delta is not None and best_delta <= float(gates["remaining_opponent_noninferiority_vs_best_existing_specialist"]["paired_intent_loss_auc20_delta_max"])
        head_improvements += int(head_improved)
        best_improvements += int(best_improved)
        all_contract = all_contract and contract_pass
        all_safety = all_safety and safety_pass
        all_head_noninferior = all_head_noninferior and head_noninferior
        remaining_best_noninferior = remaining_best_noninferior and best_noninferior
        per_opponent[opponent] = {
            "candidate": candidate_summary,
            "head_on": head_summary,
            "crossing": crossing_summary,
            "paired_head_count": paired_head_count,
            "paired_crossing_count": paired_cross_count,
            "mean_paired_delta_vs_head_on": head_delta,
            "mean_paired_delta_vs_crossing": crossing_delta,
            "best_existing_specialist": best_name,
            "mean_paired_delta_vs_best_existing": best_delta,
            "ego_crash_oob_delta_vs_head_on": crash_delta,
            "contract_pass": contract_pass,
            "safety_pass": safety_pass,
            "head_on_noninferiority_pass": head_noninferior,
            "head_on_practical_improvement_pass": head_improved,
            "best_existing_practical_improvement_pass": best_improved,
            "best_existing_noninferiority_pass": best_noninferior,
        }
    new_skill_supported = bool(
        all_contract
        and all_safety
        and all_head_noninferior
        and head_improvements >= int(gates["practical_improvement_vs_head_on"]["minimum_opponents_passing"])
        and best_improvements >= int(gates["practical_improvement_vs_best_existing_specialist"]["minimum_opponents_passing"])
        and remaining_best_noninferior
    )
    crossing_matches_candidate = any(
        item["best_existing_specialist"] == "frozen_fixed_crossing"
        and item["mean_paired_delta_vs_crossing"] is not None
        and item["mean_paired_delta_vs_crossing"] >= -0.02
        for item in per_opponent.values()
    )
    if not all_contract or not all_safety:
        verdict = "safety_or_contract_no_go"
    elif new_skill_supported:
        verdict = "new_defensive_extension_skill_gap_preliminarily_supported"
    elif crossing_matches_candidate:
        verdict = "existing_library_routing_or_composition_gap"
    else:
        verdict = "defensive_extension_hypothesis_not_supported"
    return {
        "source_id": SOURCE_ID,
        "verdict": verdict,
        "new_skill_gap_supported": new_skill_supported,
        "training_authorized_after_pilot": False,
        "per_opponent": per_opponent,
        "aggregate_decision_counts": {
            "head_on_practical_improvement_opponents": head_improvements,
            "best_existing_practical_improvement_opponents": best_improvements,
        },
    }


def run_heldout(
    config: Mapping[str, Any],
    registry: Mapping[str, Any],
    output_root: Path,
    checkpoint: Path,
) -> dict[str, Any]:
    scenarios = load_yaml(_repo_path(config["sources"]["heldout_manifest"]))["scenarios"]
    records: dict[str, Any] = {}
    for opponent in OPPONENTS:
        records[opponent] = {}
        for method in METHODS:
            records[opponent][method] = evaluate_method(
                config,
                registry,
                scenarios,
                opponent,
                method,
                checkpoint=checkpoint if method == "candidate_fixed_defensive_extension" else None,
            )
    write_json(output_root / "heldout" / "records.json", records)
    decision = analyze_heldout(config, records)
    write_json(output_root / "heldout" / "decision.json", decision)
    return decision


def run_pilot(config: Mapping[str, Any]) -> dict[str, Any]:
    authorization = validate_authorization(config, require_fresh_output=True)
    output_root = Path(config["outputs"]["root"])
    output_root.mkdir(parents=True)
    write_json(output_root / "authorization_preflight.json", authorization)
    write_json(output_root / "resolved_config.json", config)
    registry = load_yaml(_repo_path(config["sources"]["pilot_runtime_registry"]))
    baseline_dev = run_baseline_dev(config, registry, output_root)
    training = train_candidate(config, registry, output_root, baseline_dev)
    decision = None
    if training["heldout_permitted"]:
        decision = run_heldout(
            config,
            registry,
            output_root,
            Path(training["selected_checkpoint"]),
        )
    final = {
        "source_id": SOURCE_ID,
        "authorization": authorization,
        "training": training,
        "heldout_executed": decision is not None,
        "decision": decision,
    }
    write_json(output_root / "pilot_summary.json", final)
    return final
