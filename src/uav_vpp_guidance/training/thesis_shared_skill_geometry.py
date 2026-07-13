"""P4 geometry-pretraining implementation for four frozen-interface skills.

This module owns only the new non-canonical geometry-pretraining lane.  It
never trains the P3 encoder, never emits task/opponent bits into the 66-D
policy input, and never saves training raw telemetry.  The caller must still
explicitly authorize the YAML configuration before :func:`run_geometry_pretrain`
will create an output directory or step JSBSim.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
import copy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import random
import shutil
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.hierarchy.five_state_observation_contract import FiveStateObservationContract
from uav_vpp_guidance.hierarchy.shared_intent_profiles import compile_profile
from uav_vpp_guidance.hierarchy.temporal_geometry_features import TemporalGeometryFeatureExtractor
from uav_vpp_guidance.training.thesis_shared_skill_policy import ThesisSharedSkillPPOAgent
from uav_vpp_guidance.training.thesis_temporal_geometry_encoder import TemporalGeometryEncoder


REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_ENV_CONFIG = REPO_ROOT / "config" / "experiment" / "jsbsim_hrl_comparison.yaml"
GRAVITY_MPS2 = 9.80665


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return payload


def _resolve_repo_path(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _velocity(state: Mapping[str, Any]) -> np.ndarray:
    for key in ("velocity_vector_mps", "velocity"):
        if key in state:
            vector = np.asarray(state[key], dtype=np.float64).reshape(-1)
            if vector.shape == (3,):
                return vector
    if "velocity_ned" in state:
        vector = np.asarray(state["velocity_ned"], dtype=np.float64).reshape(-1)
        if vector.shape == (3,):
            return np.array([vector[0], vector[1], -vector[2]], dtype=np.float64)
    raise ValueError("State is missing a finite 3-D velocity")


def _altitude(state: Mapping[str, Any]) -> float:
    if state.get("altitude_m") is not None:
        return float(state["altitude_m"])
    for key in ("position_m", "position_neu", "position"):
        if key in state:
            position = np.asarray(state[key], dtype=np.float64).reshape(-1)
            if position.shape == (3,):
                return float(position[2])
    raise ValueError("State is missing altitude")


def _specific_energy_height(state: Mapping[str, Any]) -> float:
    speed = float(np.linalg.norm(_velocity(state)))
    return speed * speed / (2.0 * GRAVITY_MPS2) + _altitude(state)


class FrozenP3Encoder:
    """Past-only frozen P3 v2 encoder with its train-only normalisation."""

    def __init__(self, checkpoint_path: Path, expected_sha256: str):
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Frozen P3 checkpoint is missing: {checkpoint_path}")
        actual_sha256 = _sha256_file(checkpoint_path)
        if actual_sha256 != str(expected_sha256).lower():
            raise ValueError("Frozen P3 checkpoint SHA mismatch")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_config = dict(checkpoint["model_config"])
        if (
            int(model_config.get("input_dim", -1)) != 16
            or int(model_config.get("embedding_dim", -1)) != 32
            or not bool(checkpoint.get("frozen_for_high_level_ppo", False))
        ):
            raise ValueError("P3 checkpoint does not satisfy the frozen 16-D to 32-D contract")
        self.model = TemporalGeometryEncoder(
            input_dim=16,
            hidden_dim=int(model_config["hidden_dim"]),
            embedding_dim=32,
            horizons=checkpoint["dataset"]["prediction_horizons_steps"],
        )
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        normalization = checkpoint["normalization"]
        self.mean = np.asarray(normalization["mean"], dtype=np.float32).reshape(-1)
        self.std = np.asarray(normalization["std"], dtype=np.float32).reshape(-1)
        if self.mean.shape != (16,) or self.std.shape != (16,) or np.any(self.std <= 0.0):
            raise ValueError("P3 checkpoint normalisation contract is invalid")
        self.checkpoint_sha256 = actual_sha256

    def encode(self, observed_history: np.ndarray) -> np.ndarray:
        history = np.asarray(observed_history, dtype=np.float32)
        if history.shape != (10, 16):
            raise ValueError(
                "P3 encoding requires exactly ten observed 16-D frames; "
                "implicit padding or truncation is prohibited"
            )
        if not np.isfinite(history).all():
            raise ValueError("P3 observed history contains non-finite values")
        normalized = (history - self.mean) / self.std
        embedding = self.model.encode_history(normalized)
        if embedding.shape != (1, 32) or not np.isfinite(embedding).all():
            raise ValueError("Frozen P3 encoder emitted an invalid embedding")
        return embedding[0].astype(np.float32, copy=False)


class PhaseTracker:
    """Training-time phase telemetry consistent with P3 v2 definitions."""

    def __init__(self, merge_range_m: float = 1000.0, reentry_rate_mps: float = -25.0):
        self.merge_range_m = float(merge_range_m)
        self.reentry_rate_mps = float(reentry_rate_mps)
        self._seen_merge = False
        self._post_merge_steps = 0

    def update(self, range_m: float, range_rate_mps: float) -> str:
        if not self._seen_merge and float(range_m) <= self.merge_range_m:
            self._seen_merge = True
        if not self._seen_merge:
            return "pre_merge"
        self._post_merge_steps += 1
        if self._post_merge_steps >= 5 and float(range_rate_mps) <= self.reentry_rate_mps:
            return "re_entry"
        return "post_merge"


def physical_intent_vector(observation: Mapping[str, Any]) -> np.ndarray:
    relative = observation["relative_state"]
    own = observation["own_state"]
    target = observation["target_state"]
    return np.asarray(
        [
            float(np.rad2deg(relative["aa_rad"])),
            float(np.rad2deg(relative["ata_rad"])),
            float(relative["range_m"]),
            float(relative["range_rate_mps"]),
            _specific_energy_height(own) - _specific_energy_height(target),
            _altitude(own) - _altitude(target),
        ],
        dtype=np.float32,
    )


def intent_loss(observation: Mapping[str, Any], profile_name: str, *, clip: float = 2.0) -> float:
    profile = compile_profile(profile_name)
    targets = np.asarray(profile["physical_targets"], dtype=np.float32)
    weights = np.asarray(profile["weight_vector"], dtype=np.float32)
    scales = np.asarray([180.0, 180.0, 5000.0, 200.0, 1000.0, 1000.0], dtype=np.float32)
    residual = np.abs(physical_intent_vector(observation) - targets) / scales
    weighted = float(np.sum(weights * np.minimum(residual, float(clip))) / max(float(np.sum(weights)), 1e-6))
    return weighted


def shaped_geometry_reward(
    before_loss: float,
    after_loss: float,
    info: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, float]:
    """Profile-weighted geometric progress with an explicit ego safety penalty."""

    reward_config = config["geometry_reward"]
    clip = float(reward_config["intent_error_clip"])
    progress = float(np.clip(before_loss - after_loss, -clip, clip))
    alignment = max(0.0, 1.0 - min(after_loss, clip) / clip)
    saturation = sum(
        float(bool(info.get(key, False)))
        for key in ("nz_saturated", "roll_rate_saturated", "throttle_saturated")
    )
    reason = str(info.get("reason") or info.get("termination_info", {}).get("reason") or "")
    ego_terminal = reason in {"crash", "out_of_bounds", "ego_crash", "ego_out_of_bounds"}
    reward = (
        float(reward_config["progress_weight"]) * progress
        + float(reward_config["alignment_weight"]) * alignment
        - float(reward_config["physical_command_saturation_penalty"]) * saturation
        - (float(reward_config["terminal_ego_crash_oob_penalty"]) if ego_terminal else 0.0)
    )
    return {
        "reward": float(reward),
        "intent_progress": progress,
        "intent_alignment": alignment,
        "saturation_count": saturation,
        "ego_terminal_penalty": float(ego_terminal),
    }


def _history_frame(observation: Mapping[str, Any]) -> dict[str, float]:
    relative = observation["relative_state"]
    own = observation["own_state"]
    return {
        "range_rate_mps": float(relative["range_rate_mps"]),
        "aa_deg": float(np.rad2deg(relative["aa_rad"])),
        "ata_deg": float(np.rad2deg(relative["ata_rad"])),
        "specific_energy_height_m": _specific_energy_height(own),
        "altitude_m": _altitude(own),
    }


def _actual_response_metrics(observation: Mapping[str, Any]) -> dict[str, float]:
    """Extract actual JSBSim response, not merely requested guidance command."""

    own = observation["own_state"]
    actual_nz_g = float(own.get("nz_g", np.nan))
    body_velocity = np.asarray(own.get("velocity_body", [np.nan, np.nan, np.nan]), dtype=np.float64).reshape(-1)
    if body_velocity.shape == (3,) and np.isfinite(body_velocity[[0, 2]]).all():
        actual_aoa_deg = float(np.rad2deg(np.arctan2(body_velocity[2], body_velocity[0])))
    else:
        actual_aoa_deg = float("nan")
    return {"actual_nz_g": actual_nz_g, "actual_aoa_deg": actual_aoa_deg}


def _environment_config(config: Mapping[str, Any], opponent_id: str) -> dict[str, Any]:
    base = copy.deepcopy(_load_yaml(BASE_ENV_CONFIG))
    runtime = config["runtime"]
    base["backend"] = "jsbsim"
    base.setdefault("env", {}).update(
        {
            "use_jsbsim": True,
            "strict_backend": True,
            "legacy_project_root": str(runtime["legacy_project_root"]),
            "aircraft_model": str(runtime["aircraft_model"]),
            "high_level_dt": float(runtime["high_level_dt_s"]),
            "max_high_level_steps": int(config["training"]["max_high_level_steps_per_episode"]),
            "success_range_m": 0.0,
            "success_ata_deg": 0.0,
            "success_hold_time_s": 1.0,
            "max_range_m": 12000.0,
        }
    )
    base["trajectory_prediction"] = copy.deepcopy(runtime["trajectory_prediction"])
    base["virtual_point"] = copy.deepcopy(runtime["virtual_point"])
    base["opponent_stage"] = str(opponent_id)
    return base


class ProfileConditionedSkillEpisode:
    """A no-padding 66-D P4 rollout around one strict JSBSim environment."""

    def __init__(
        self,
        env: CloseRangeTrackingEnv,
        encoder: FrozenP3Encoder,
        profile_name: str,
        config: Mapping[str, Any],
    ):
        self.env = env
        self.encoder = encoder
        self.profile_name = str(profile_name)
        self.config = config
        self.contract = FiveStateObservationContract()
        self.history = deque(maxlen=10)
        self.history_features = TemporalGeometryFeatureExtractor(window_steps=10)
        self.phase_tracker = PhaseTracker()
        self.current_observation: dict[str, Any] | None = None
        self.current_phase = "pre_merge"

    def _observe(self, observation: Mapping[str, Any]) -> None:
        base = np.asarray(observation["observation_vector"], dtype=np.float32).reshape(-1)
        if base.shape != (16,) or not np.isfinite(base).all():
            raise ValueError("P4 base geometry must remain exactly 16-D and finite")
        self.history.append(base.copy())
        self.history_features.update(_history_frame(observation))
        relative = observation["relative_state"]
        self.current_phase = self.phase_tracker.update(
            float(relative["range_m"]), float(relative["range_rate_mps"])
        )
        self.current_observation = dict(observation)

    def _policy_observation(self) -> np.ndarray:
        if self.current_observation is None or len(self.history) != 10:
            raise RuntimeError("P4 policy cannot act before ten real observations are available")
        profile = compile_profile(self.profile_name)
        result = self.contract.compose(
            base_geometry=np.asarray(self.current_observation["observation_vector"], dtype=np.float32),
            explicit_history=self.history_features.features(),
            intent_targets=profile["target_vector"],
            intent_weights=profile["weight_vector"],
            temporal_embedding=self.encoder.encode(np.stack(tuple(self.history), axis=0)),
        )
        if result.shape != (66,):
            raise AssertionError("P4 policy observation dimension drifted")
        return result

    def reset(self, scenario: Mapping[str, Any], seed: int) -> tuple[np.ndarray, dict[str, Any]]:
        self.history.clear()
        self.history_features.reset()
        self.phase_tracker = PhaseTracker()
        observation = self.env.reset(scenario=copy.deepcopy(dict(scenario)), seed=int(seed))
        if self.env._backend != "jsbsim":
            raise RuntimeError("P4 geometry pretraining prohibits backend fallback")
        self._observe(observation)
        bootstrap_steps = int(self.config["history_bootstrap"]["policy_action_start_after_observed_steps"]) - 1
        action = np.asarray(self.config["history_bootstrap"]["bootstrap_action"], dtype=np.float32)
        if action.shape != (3,) or not np.allclose(action, 0.0):
            raise ValueError("P4 only permits the declared zero-VPP history bootstrap")
        for _ in range(bootstrap_steps):
            observation, _reward, terminated, truncated, info = self.env.step(action)
            self._observe(observation)
            if terminated or truncated:
                raise RuntimeError(
                    "P4 history bootstrap terminated before an exact ten-step history; "
                    "the scenario must be rejected rather than padded"
                )
        return self._policy_observation(), {
            "phase": self.current_phase,
            "intent_loss": intent_loss(self.current_observation, self.profile_name, clip=float(self.config["geometry_reward"]["intent_error_clip"])),
        }

    def step(self, action: np.ndarray) -> tuple[np.ndarray | None, float, bool, dict[str, Any]]:
        if self.current_observation is None:
            raise RuntimeError("P4 episode must be reset before stepping")
        normalized_action = np.asarray(action, dtype=np.float32).reshape(-1)
        if normalized_action.shape != (3,) or not np.isfinite(normalized_action).all():
            raise ValueError("P4 policy action must be finite normalized 3-D VPP")
        if np.any(np.abs(normalized_action) > 1.0 + 1e-6):
            raise ValueError("P4 policy action is outside normalized VPP bounds")
        before = intent_loss(self.current_observation, self.profile_name, clip=float(self.config["geometry_reward"]["intent_error_clip"]))
        observation, _base_reward, terminated, truncated, info = self.env.step(normalized_action)
        self._observe(observation)
        after = intent_loss(self.current_observation, self.profile_name, clip=float(self.config["geometry_reward"]["intent_error_clip"]))
        reward_terms = shaped_geometry_reward(before, after, info, self.config)
        done = bool(terminated or truncated)
        next_observation = None if done else self._policy_observation()
        metrics = {
            "phase": self.current_phase,
            "before_intent_loss": before,
            "after_intent_loss": after,
            "terminal_reason": str(info.get("reason") or info.get("termination_info", {}).get("reason") or "timeout"),
            "vp_forward_bias_m": float(info.get("vp_forward_bias_m", np.nan)),
            "vp_lateral_bias_m": float(info.get("vp_lateral_bias_m", np.nan)),
            "nz_cmd": float(info.get("nz_cmd", np.nan)),
            "ego_attack_aoa_deg": float(info.get("ego_attack_aoa_deg", np.nan)),
            **_actual_response_metrics(self.current_observation),
            **reward_terms,
        }
        return next_observation, float(reward_terms["reward"]), done, metrics

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if callable(close):
            close()


def _p3_sampler_module():
    path = REPO_ROOT / "scripts" / "train_thesis_five_state_temporal_encoder.py"
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("p3_geometry_sampler", path)
    if spec is None or spec.loader is None:  # pragma: no cover - installation failure
        raise RuntimeError("Unable to import frozen P3 train sampler")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_geometry_training_plan(
    config: Mapping[str, Any], registry: Mapping[str, Any]
) -> dict[str, Any]:
    """Return a no-side-effect plan used by dry-run and unit tests."""

    distribution = _load_yaml(_resolve_repo_path(config["sources"]["train_distribution"]))
    sampler = _p3_sampler_module()
    candidates = sampler.sample_train_scenarios(distribution)
    skills = registry["skills"]
    plan: dict[str, Any] = {"skills": {}, "training_started": False}
    for skill_name, definition in skills.items():
        eligible = [
            scenario
            for scenario in candidates
            if scenario["metadata"]["initial_class"] in definition["geometry_phase_coverage"]["geometry_states"]
        ]
        plan["skills"][skill_name] = {
            "allowed_profiles": list(definition["allowed_profiles"]),
            "eligible_train_scenarios": len(eligible),
            "declared_geometry_states": list(definition["geometry_phase_coverage"]["geometry_states"]),
            "declared_phases": list(definition["geometry_phase_coverage"]["phases"]),
            "total_timesteps": int(config["training"]["total_timesteps_per_skill"]),
        }
    return plan


def _make_episode(
    config: Mapping[str, Any],
    registry: Mapping[str, Any],
    encoder: FrozenP3Encoder,
    opponent_id: str,
    profile_name: str,
) -> ProfileConditionedSkillEpisode:
    opponent_registry = _load_yaml(_resolve_repo_path(config["sources"]["opponent_registry"]))
    entry = opponent_registry["opponents"][opponent_id]
    env = CloseRangeTrackingEnv(
        _environment_config(config, opponent_id),
        opponent_policy=_instantiate_opponent(entry),
        opponent_config={"stage": opponent_id, **copy.deepcopy(entry)},
    )
    return ProfileConditionedSkillEpisode(env, encoder, profile_name, config)


def _record_episode_summary(
    aggregate: dict[str, Any], skill_name: str, profile_name: str, initial_state: str, metrics: Sequence[Mapping[str, Any]]) -> None:
    aggregate["episodes"] += 1
    aggregate["profile_episodes"][profile_name] += 1
    for metric in metrics:
        aggregate["phase_steps"][metric["phase"]] += 1
        aggregate["state_phase_steps"][f"{initial_state}:{metric['phase']}"] += 1
        aggregate["intent_progress_steps"] += int(float(metric["intent_progress"]) > 0.0)
        aggregate["policy_steps"] += 1
        for name in (
            "vp_forward_bias_m",
            "vp_lateral_bias_m",
            "nz_cmd",
            "ego_attack_aoa_deg",
            "actual_nz_g",
            "actual_aoa_deg",
        ):
            value = float(metric[name])
            if np.isfinite(value):
                aggregate["response_samples"][name].append(value)
    reason = str(metrics[-1]["terminal_reason"]) if metrics else "bootstrap_rejected"
    aggregate["terminal_reasons"][reason] += 1


def _empty_aggregate() -> dict[str, Any]:
    return {
        "episodes": 0,
        "profile_episodes": Counter(),
        "phase_steps": Counter(),
        "state_phase_steps": Counter(),
        "intent_progress_steps": 0,
        "policy_steps": 0,
        "terminal_reasons": Counter(),
        "response_samples": defaultdict(list),
    }


def _serialize_aggregate(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    response = {}
    for name, values in aggregate["response_samples"].items():
        finite = np.asarray(values, dtype=np.float64)
        response[name] = {
            "count": int(finite.size),
            "mean": float(np.mean(finite)) if finite.size else None,
            "max_abs": float(np.max(np.abs(finite))) if finite.size else None,
        }
    return {
        "episodes": int(aggregate["episodes"]),
        "profile_episodes": dict(aggregate["profile_episodes"]),
        "phase_steps": dict(aggregate["phase_steps"]),
        "state_phase_steps": dict(aggregate["state_phase_steps"]),
        "intent_progress_fraction": (
            float(aggregate["intent_progress_steps"] / aggregate["policy_steps"])
            if aggregate["policy_steps"] else 0.0
        ),
        "terminal_reasons": dict(aggregate["terminal_reasons"]),
        "response_summary": response,
    }


def evaluate_skill_gate(
    aggregate_by_opponent: Mapping[str, Mapping[str, Any]],
    skill_definition: Mapping[str, Any],
    config: Mapping[str, Any],
    zero_vpp_terminal_by_opponent: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    """Evaluate the pre-registered per-skill geometry-pretrain gate."""

    gate = config["selection"]["per_skill_gate"]
    all_profiles = tuple(skill_definition["allowed_profiles"])
    required_states = tuple(skill_definition["geometry_phase_coverage"]["geometry_states"])
    required_phases = tuple(skill_definition["geometry_phase_coverage"]["phases"])
    combined = _empty_aggregate()
    safety_by_opponent: dict[str, Any] = {}
    for opponent_id, raw in aggregate_by_opponent.items():
        for key in ("episodes", "intent_progress_steps", "policy_steps"):
            combined[key] += int(raw[key])
        combined["profile_episodes"].update(raw["profile_episodes"])
        combined["state_phase_steps"].update(raw["state_phase_steps"])
        for reason, count in raw["terminal_reasons"].items():
            combined["terminal_reasons"][reason] += count
        total = max(int(raw["episodes"]), 1)
        ego_failures = sum(raw["terminal_reasons"].get(reason, 0) for reason in ("crash", "out_of_bounds", "ego_crash", "ego_out_of_bounds"))
        zero = zero_vpp_terminal_by_opponent.get(opponent_id, {})
        zero_failures = sum(zero.get(reason, 0) for reason in ("crash", "out_of_bounds", "ego_crash", "ego_out_of_bounds"))
        zero_total = max(sum(zero.values()), 1)
        delta = ego_failures / total - zero_failures / zero_total
        safety_by_opponent[opponent_id] = {
            "policy_crash_oob_rate": ego_failures / total,
            "zero_vpp_crash_oob_rate": zero_failures / zero_total,
            "delta": delta,
            "passed": delta <= float(gate["max_ego_crash_oob_delta_vs_zero_vpp_reference"]),
        }
    coverage = {
        f"{state}:{phase}": int(combined["state_phase_steps"].get(f"{state}:{phase}", 0))
        for state in required_states
        for phase in required_phases
    }
    profile_coverage = {profile: int(combined["profile_episodes"].get(profile, 0)) for profile in all_profiles}
    progress_fraction = (
        combined["intent_progress_steps"] / combined["policy_steps"]
        if combined["policy_steps"]
        else 0.0
    )
    checks = {
        "profile_conditioning_coverage": all(
            count >= int(gate["minimum_profile_conditioning_coverage_per_allowed_profile"])
            for count in profile_coverage.values()
        ),
        "geometry_phase_coverage": all(
            count >= int(gate["minimum_geometry_phase_coverage_per_declared_cell"])
            for count in coverage.values()
        ),
        "intent_progress": progress_fraction >= float(gate["minimum_intent_progress_fraction"]),
        "safety": all(entry["passed"] for entry in safety_by_opponent.values()),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "profile_coverage": profile_coverage,
        "geometry_phase_coverage": coverage,
        "intent_progress_fraction": float(progress_fraction),
        "safety_by_opponent": safety_by_opponent,
        "status": "ready_for_combat_finetune" if all(checks.values()) else "not_ready_do_not_start_combat_finetune",
    }


def _ensure_training_authorized(config: Mapping[str, Any], output_root: Path) -> None:
    if config["experiment"].get("training_permitted") is not True:
        raise PermissionError("P4 geometry pretraining is disabled by training_permitted: false")
    if config["experiment"].get("execution_mode") != "authorized_training":
        raise PermissionError("P4 geometry pretraining requires execution_mode: authorized_training")
    authorization = config.get("authorization", {})
    if (
        authorization.get("scope") != "geometry_pretrain_only"
        or not str(authorization.get("run_id", "")).strip()
        or authorization.get("combat_finetune_permitted") is not False
        or authorization.get("high_level_ppo_permitted") is not False
    ):
        raise PermissionError("P4 geometry pretraining authorization is incomplete or expands scope")
    if output_root.exists():
        raise FileExistsError("P4 geometry output root must be fresh; refusing to reuse or overwrite it")
    if shutil.disk_usage(output_root.anchor).free / (1024**3) < 120.0:
        raise RuntimeError("P4 geometry pretraining requires at least 120 GB free disk space")


def _eligible_train_scenarios(
    config: Mapping[str, Any], skill_definition: Mapping[str, Any], *, seed_offset: int
) -> list[dict[str, Any]]:
    distribution = _load_yaml(_resolve_repo_path(config["sources"]["train_distribution"]))
    candidates = _p3_sampler_module().sample_train_scenarios(
        distribution, seed_offset=int(seed_offset)
    )
    allowed_states = set(skill_definition["geometry_phase_coverage"]["geometry_states"])
    selected = [item for item in candidates if item["metadata"]["initial_class"] in allowed_states]
    if not selected:
        raise RuntimeError("P4 skill has no train scenarios matching its declared geometry coverage")
    return selected


def _run_rollout(
    runner: ProfileConditionedSkillEpisode,
    scenario: Mapping[str, Any],
    *,
    seed: int,
    action_provider,
    max_steps: int,
) -> tuple[list[dict[str, Any]], str]:
    """Execute one full episode and return compact metrics, never raw states."""

    try:
        observation, _meta = runner.reset(scenario, seed)
    except RuntimeError as exc:
        if "history bootstrap terminated" not in str(exc):
            raise
        return [], "bootstrap_rejected"
    records: list[dict[str, Any]] = []
    for _ in range(int(max_steps)):
        action = action_provider(observation)
        next_observation, _reward, done, metric = runner.step(action)
        records.append(metric)
        if done:
            return records, str(metric["terminal_reason"])
        if next_observation is None:  # Defensive invariant.
            raise AssertionError("P4 unfinished rollout returned no next observation")
        observation = next_observation
    return records, "forced_horizon"


def _evaluate_skill_on_dev30(
    agent: ThesisSharedSkillPPOAgent,
    skill_name: str,
    skill_definition: Mapping[str, Any],
    config: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Counter]]:
    """Evaluate policy and paired zero-VPP reference separately per opponent."""

    manifest = _load_yaml(_resolve_repo_path(config["sources"]["dev30_manifest"]))
    opponent_registry = _load_yaml(_resolve_repo_path(config["sources"]["opponent_registry"]))
    allowed_states = set(skill_definition["geometry_phase_coverage"]["geometry_states"])
    scenarios = [item for item in manifest["scenarios"] if item["metadata"]["initial_class"] in allowed_states]
    if not scenarios:
        raise RuntimeError(f"No dev30 scenarios match {skill_name} coverage")
    profiles = tuple(skill_definition["allowed_profiles"])
    by_opponent: dict[str, dict[str, Any]] = {}
    zero_terminals: dict[str, Counter] = {}
    for opponent_index, opponent_id in enumerate(config["selection"]["dev30_opponents"]):
        if opponent_id not in opponent_registry["opponents"]:
            raise KeyError(f"P4 dev registry lacks opponent {opponent_id}")
        aggregate = _empty_aggregate()
        zero = Counter()
        runner = _make_episode(config, registry, FrozenP3Encoder(
            Path(config["encoder"]["checkpoint"]), config["encoder"]["sha256"]
        ), opponent_id, profiles[0])
        try:
            for scenario_index, scenario in enumerate(scenarios):
                profile_name = profiles[scenario_index % len(profiles)]
                runner.profile_name = profile_name
                records, _reason = _run_rollout(
                    runner,
                    scenario,
                    seed=int(scenario["metadata"]["scenario_seed"]) + opponent_index * 100000,
                    action_provider=lambda observation: agent.select_action(observation, deterministic=True)[0],
                    max_steps=int(config["training"]["max_high_level_steps_per_episode"]),
                )
                _record_episode_summary(
                    aggregate,
                    skill_name,
                    profile_name,
                    str(scenario["metadata"]["initial_class"]),
                    records,
                )
                runner.profile_name = profile_name
                _zero_records, zero_reason = _run_rollout(
                    runner,
                    scenario,
                    seed=int(scenario["metadata"]["scenario_seed"]) + opponent_index * 100000,
                    action_provider=lambda _observation: np.zeros(3, dtype=np.float32),
                    max_steps=int(config["training"]["max_high_level_steps_per_episode"]),
                )
                zero[zero_reason] += 1
        finally:
            runner.close()
        by_opponent[opponent_id] = aggregate
        zero_terminals[opponent_id] = zero
    gate = evaluate_skill_gate(by_opponent, skill_definition, config, zero_terminals)
    return {
        "skill": skill_name,
        "by_opponent": {key: _serialize_aggregate(value) for key, value in by_opponent.items()},
        "zero_vpp_terminal_reasons_by_opponent": {key: dict(value) for key, value in zero_terminals.items()},
        "gate": gate,
    }, zero_terminals


def _checkpoint_metadata(
    skill_name: str,
    registry: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "skill_name": skill_name,
        "skill_id": int(registry["skills"][skill_name]["skill_id"]),
        "allowed_profiles": list(registry["skills"][skill_name]["allowed_profiles"]),
        "observation_dim": 66,
        "action_dim": 3,
        "p3_encoder_sha256": str(config["encoder"]["sha256"]),
        "encoder_trainable": False,
        "checkpoint_fallback": "prohibited",
        "stage": "geometry_pretrain",
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_geometry_pretrain(
    config: Mapping[str, Any], registry: Mapping[str, Any], *, output_root: Path
) -> dict[str, Any]:
    """Run sequential four-skill geometry pretraining after explicit authorization.

    This is intentionally not invoked by the current P4 patch.  It is kept
    here so the later authorization changes only gates, never the training
    algorithm or frozen P3/profile/opponent contracts.
    """

    _ensure_training_authorized(config, output_root)
    frozen = registry["frozen_inputs"]
    encoder_info = frozen["p3_temporal_encoder"]
    encoder = FrozenP3Encoder(Path(encoder_info["checkpoint"]), encoder_info["sha256"])
    output_root.mkdir(parents=True)
    (output_root / "resolved_geometry_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plan = build_geometry_training_plan(config, registry)
    _write_json(output_root / "training_plan.json", plan)
    random.seed(int(config["training"]["seed"]))
    np.random.seed(int(config["training"]["seed"]))
    torch.manual_seed(int(config["training"]["seed"]))
    final_report: dict[str, Any] = {
        "stage": "p4_geometry_pretrain",
        "training_started": True,
        "raw_train_trajectory_persistence": "prohibited",
        "p3_encoder_sha256": encoder.checkpoint_sha256,
        "skills": {},
    }
    for skill_offset, (skill_name, skill_definition) in enumerate(registry["skills"].items()):
        skill_root = output_root / "skills" / skill_name
        checkpoint_root = skill_root / "checkpoints"
        agent = ThesisSharedSkillPPOAgent(config, device=str(config["training"]["device"]))
        scenarios = _eligible_train_scenarios(config, skill_definition, seed_offset=skill_offset * 1000)
        profiles = tuple(skill_definition["allowed_profiles"])
        runner = _make_episode(config, registry, encoder, str(config["training"]["train_opponent"]), profiles[0])
        train_aggregate = _empty_aggregate()
        evaluation_records: list[dict[str, Any]] = []
        candidates: list[tuple[float, Path, dict[str, Any]]] = []
        next_observation: np.ndarray | None = None
        scenario_index = 0
        profile_index = 0
        episode_index = 0
        next_eval = int(config["selection"]["evaluation_interval_steps"])
        try:
            while agent.total_timesteps < int(config["training"]["total_timesteps_per_skill"]):
                scenario = scenarios[scenario_index % len(scenarios)]
                profile_name = profiles[profile_index % len(profiles)]
                scenario_index += 1
                profile_index += 1
                episode_index += 1
                runner.profile_name = profile_name
                try:
                    observation, _meta = runner.reset(
                        scenario,
                        seed=int(config["training"]["seed"]) + skill_offset * 1_000_000 + episode_index,
                    )
                except RuntimeError as exc:
                    if "history bootstrap terminated" not in str(exc):
                        raise
                    train_aggregate["episodes"] += 1
                    train_aggregate["terminal_reasons"]["bootstrap_rejected"] += 1
                    continue
                records: list[dict[str, Any]] = []
                done = False
                while not done and agent.total_timesteps < int(config["training"]["total_timesteps_per_skill"]):
                    action, log_prob, value = agent.select_action(observation, deterministic=False)
                    next_observation, reward, done, metric = runner.step(action)
                    agent.store_transition(observation, action, log_prob, reward, done, value)
                    records.append(metric)
                    if agent.buffer.full:
                        agent.update(None if done else next_observation)
                    if not done:
                        if next_observation is None:
                            raise AssertionError("P4 unfinished training rollout lost its next observation")
                        observation = next_observation
                _record_episode_summary(
                    train_aggregate,
                    skill_name,
                    profile_name,
                    str(scenario["metadata"]["initial_class"]),
                    records,
                )
                if agent.total_timesteps >= next_eval or agent.total_timesteps >= int(config["training"]["total_timesteps_per_skill"]):
                    if len(agent.buffer):
                        agent.update(None if done else next_observation)
                    evaluation, _zero = _evaluate_skill_on_dev30(agent, skill_name, skill_definition, config, registry)
                    evaluation["step"] = agent.total_timesteps
                    evaluation_records.append(evaluation)
                    gate = evaluation["gate"]
                    score = float(gate["intent_progress_fraction"])
                    candidate_path = checkpoint_root / f"step_{agent.total_timesteps}.pt"
                    agent.save(candidate_path, _checkpoint_metadata(skill_name, registry, config))
                    candidates.append((score, candidate_path, gate))
                    candidates.sort(key=lambda item: (bool(item[2]["passed"]), item[0]), reverse=True)
                    for _score, stale_path, _gate in candidates[3:]:
                        if stale_path.exists():
                            stale_path.unlink()
                    candidates = candidates[:3]
                    next_eval += int(config["selection"]["evaluation_interval_steps"])
        finally:
            runner.close()
        if len(agent.buffer):
            agent.update(next_observation)
        final_checkpoint = checkpoint_root / "last.pt"
        agent.save(final_checkpoint, _checkpoint_metadata(skill_name, registry, config))
        selected = candidates[0] if candidates else None
        if selected is not None:
            best_path = checkpoint_root / "best.pt"
            shutil.copy2(selected[1], best_path)
            selected_gate = selected[2]
        else:
            selected_gate = {"passed": False, "status": "not_ready_no_dev_evaluation"}
        skill_report = {
            "training_steps": agent.total_timesteps,
            "train_aggregate": _serialize_aggregate(train_aggregate),
            "dev_evaluations": evaluation_records,
            "selected_gate": selected_gate,
            "selected_checkpoint": str((checkpoint_root / "best.pt") if selected is not None else final_checkpoint),
            "status": "geometry_pretrain_gate_passed" if selected_gate.get("passed") else "geometry_pretrain_gate_failed_do_not_start_combat_finetune",
        }
        _write_json(skill_root / "geometry_pretrain_summary.json", skill_report)
        final_report["skills"][skill_name] = skill_report
    final_report["all_skills_ready_for_combat_finetune"] = all(
        entry["selected_gate"].get("passed", False) for entry in final_report["skills"].values()
    )
    final_report["next_stage"] = (
        "combat_finetune_may_be_authorized"
        if final_report["all_skills_ready_for_combat_finetune"]
        else "do_not_start_combat_finetune"
    )
    _write_json(output_root / "p4_geometry_pretrain_gate.json", final_report)
    return final_report
