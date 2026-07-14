"""Observe-only 66-D contract collector for one defensive-extension motif."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np

from uav_vpp_guidance.evaluation.p4_v2_sampler_feasibility import (
    classify_dynamic_state,
)
from uav_vpp_guidance.evaluation.phase_reachability_handoff_contract import (
    telemetry_sha256,
)
from uav_vpp_guidance.hierarchy.five_state_observation_contract import (
    BASE_GEOMETRY_FEATURES,
    FiveStateObservationContract,
)
from uav_vpp_guidance.hierarchy.shared_intent_profiles import compile_profile
from uav_vpp_guidance.hierarchy.shared_intent_validity import (
    PROFILE_NAMES,
    SKILL_NAMES,
    TacticalContext,
    build_validity_mask,
)
from uav_vpp_guidance.hierarchy.temporal_geometry_features import (
    TemporalGeometryFeatureExtractor,
)
from uav_vpp_guidance.training.thesis_shared_skill_geometry import (
    FrozenP3Encoder,
    PhaseTracker,
)


SOURCE_ID = "THESIS-SINGLE-MOTIF-CONTINUOUS-66D-R1"
REQUIRED_OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
TARGET_SKILL = "defensive_extension"
TARGET_PROFILE = "range_extension"
TARGET_STATE = "disadvantage"
TARGET_PHASES = ("post_merge", "re_entry")
GRAVITY_MPS2 = 9.80665


class SingleMotifContractError(ValueError):
    """Raised when the observe-only collector contract is violated."""


@dataclass(frozen=True)
class SingleMotifPlan:
    source_id: str
    p3_checkpoint: Path
    p3_checkpoint_sha256: str
    opponents: tuple[str, ...]
    target_skill: str
    target_profile: str
    target_state: str
    target_phases: tuple[str, ...]
    history_window_steps: int
    min_episodes_per_opponent: int
    min_target_steps_per_opponent: int
    min_signatures_per_opponent: int
    min_mirror_signs_per_opponent: int
    energy_band_m: float
    altitude_band_m: float


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SingleMotifContractError(f"{name} must be a mapping")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SingleMotifContractError(f"{name} must be a positive integer")
    return int(value)


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SingleMotifContractError(f"{name} must be a positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise SingleMotifContractError(f"{name} must be a positive number")
    return result


def build_single_motif_plan(config: Mapping[str, Any]) -> SingleMotifPlan:
    """Validate the independent, non-learning collector configuration."""

    protocol = _mapping(
        config.get("single_motif_continuous_66d"),
        "single_motif_continuous_66d",
    )
    if str(protocol.get("source_id", "")) != SOURCE_ID:
        raise SingleMotifContractError("unexpected single-motif source_id")
    authorization = _mapping(protocol.get("authorization"), "authorization")
    required_false = (
        "training_permitted",
        "tuning_permitted",
        "action_replacement_permitted",
        "policy_change_permitted",
        "vpp_change_permitted",
        "guidance_change_permitted",
        "pid_change_permitted",
        "reset_after_first_pass",
        "snapshot_restore",
        "future_state_injection",
        "history_padding_permitted",
    )
    for key in required_false:
        if authorization.get(key) is not False:
            raise SingleMotifContractError(f"authorization.{key} must remain false")
    if authorization.get("execution_permitted") is not True:
        raise SingleMotifContractError("authorization.execution_permitted must be true")

    reference = _mapping(protocol.get("reference"), "reference")
    if reference.get("controller") != "legacy_static_oracle_task_gate":
        raise SingleMotifContractError("collector must use the frozen oracle task gate")
    if reference.get("learning") is not False:
        raise SingleMotifContractError("reference.learning must remain false")

    target = _mapping(protocol.get("target_motif"), "target_motif")
    if target.get("skill") != TARGET_SKILL or target.get("profile") != TARGET_PROFILE:
        raise SingleMotifContractError("target skill/profile drifted")
    if target.get("dynamic_state") != TARGET_STATE:
        raise SingleMotifContractError("target dynamic state drifted")
    phases = tuple(target.get("phases", ()))
    if phases != TARGET_PHASES:
        raise SingleMotifContractError("target phases drifted")

    observation = _mapping(protocol.get("observation_contract"), "observation_contract")
    if int(observation.get("base_dim", -1)) != 16:
        raise SingleMotifContractError("base observation must remain 16-D")
    if int(observation.get("temporal_embedding_dim", -1)) != 32:
        raise SingleMotifContractError("P3 embedding must remain 32-D")
    if int(observation.get("composed_dim", -1)) != 66:
        raise SingleMotifContractError("composed observation must remain 66-D")
    history_window = _positive_int(
        observation.get("history_window_steps"),
        "observation_contract.history_window_steps",
    )
    if history_window != 10:
        raise SingleMotifContractError("history window must remain exactly ten real frames")

    p3 = _mapping(protocol.get("frozen_p3_encoder"), "frozen_p3_encoder")
    checkpoint = Path(str(p3.get("checkpoint", "")))
    expected_sha = str(p3.get("sha256", "")).lower()
    if not checkpoint.is_absolute() or len(expected_sha) != 64:
        raise SingleMotifContractError("frozen P3 checkpoint path/SHA is invalid")

    opponents = tuple(protocol.get("opponents", ()))
    if opponents != REQUIRED_OPPONENTS:
        raise SingleMotifContractError("collector must preserve the three-opponent order")
    gate = _mapping(protocol.get("gate"), "gate")
    context = _mapping(protocol.get("context_bands"), "context_bands")
    return SingleMotifPlan(
        source_id=SOURCE_ID,
        p3_checkpoint=checkpoint,
        p3_checkpoint_sha256=expected_sha,
        opponents=opponents,
        target_skill=TARGET_SKILL,
        target_profile=TARGET_PROFILE,
        target_state=TARGET_STATE,
        target_phases=TARGET_PHASES,
        history_window_steps=history_window,
        min_episodes_per_opponent=_positive_int(
            gate.get("minimum_qualifying_episodes_per_opponent"),
            "gate.minimum_qualifying_episodes_per_opponent",
        ),
        min_target_steps_per_opponent=_positive_int(
            gate.get("minimum_valid_target_steps_per_opponent"),
            "gate.minimum_valid_target_steps_per_opponent",
        ),
        min_signatures_per_opponent=_positive_int(
            gate.get("minimum_distinct_scenario_signatures_per_opponent"),
            "gate.minimum_distinct_scenario_signatures_per_opponent",
        ),
        min_mirror_signs_per_opponent=_positive_int(
            gate.get("minimum_distinct_mirror_signs_per_opponent"),
            "gate.minimum_distinct_mirror_signs_per_opponent",
        ),
        energy_band_m=_positive_float(
            context.get("specific_energy_height_band_m"),
            "context_bands.specific_energy_height_band_m",
        ),
        altitude_band_m=_positive_float(
            context.get("altitude_band_m"),
            "context_bands.altitude_band_m",
        ),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _vector_list(value: Any, expected_dim: int | None = None) -> list[float]:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if expected_dim is not None and vector.shape != (expected_dim,):
        raise SingleMotifContractError(
            f"vector shape mismatch: expected {(expected_dim,)}, got {vector.shape}"
        )
    if not np.isfinite(vector).all():
        raise SingleMotifContractError("vector contains non-finite values")
    return [float(item) for item in vector]


def project_base_observation(
    observation_vector: Any,
    observation_schema: Mapping[str, Any],
) -> np.ndarray:
    """Project the exact 16 base features by name, never by positional slicing."""

    vector = np.asarray(observation_vector, dtype=np.float32).reshape(-1)
    feature_names = tuple(str(name) for name in observation_schema.get("feature_names", ()))
    if len(feature_names) != len(set(feature_names)):
        raise SingleMotifContractError("observation schema contains duplicate feature names")
    if vector.shape != (len(feature_names),):
        raise SingleMotifContractError("observation vector/schema dimension mismatch")
    index = {name: position for position, name in enumerate(feature_names)}
    missing = [name for name in BASE_GEOMETRY_FEATURES if name not in index]
    if missing:
        raise SingleMotifContractError(f"observation schema is missing base features: {missing}")
    base = np.asarray([vector[index[name]] for name in BASE_GEOMETRY_FEATURES], dtype=np.float32)
    if base.shape != (16,) or not np.isfinite(base).all():
        raise SingleMotifContractError("projected base observation is not finite 16-D")
    return base


def _base_map(base: np.ndarray) -> dict[str, float]:
    return {name: float(base[index]) for index, name in enumerate(BASE_GEOMETRY_FEATURES)}


def _angle_deg(sine: float, cosine: float) -> float:
    angle = math.degrees(math.atan2(float(sine), float(cosine)))
    return abs(angle)


def _history_frame(base: np.ndarray) -> dict[str, float]:
    values = _base_map(base)
    speed = values["own_speed"]
    altitude = values["own_altitude"]
    return {
        "range_rate_mps": values["range_rate_mps"],
        "aa_deg": _angle_deg(values["aa_sin"], values["aa_cos"]),
        "ata_deg": _angle_deg(values["ata_sin"], values["ata_cos"]),
        "specific_energy_height_m": altitude + speed * speed / (2.0 * GRAVITY_MPS2),
        "altitude_m": altitude,
    }


def _context(base: np.ndarray, dynamic_state: str, phase: str, plan: SingleMotifPlan) -> TacticalContext:
    values = _base_map(base)
    own_energy = values["own_altitude"] + values["own_speed"] ** 2 / (2.0 * GRAVITY_MPS2)
    target_energy = values["target_altitude"] + values["target_speed"] ** 2 / (2.0 * GRAVITY_MPS2)
    energy_delta = own_energy - target_energy
    if energy_delta > plan.energy_band_m:
        energy_state = "advantage"
    elif energy_delta < -plan.energy_band_m:
        energy_state = "disadvantage"
    else:
        energy_state = "balanced"
    altitude_delta = values["own_altitude"] - values["target_altitude"]
    if altitude_delta > plan.altitude_band_m:
        altitude_state = "own_above"
    elif altitude_delta < -plan.altitude_band_m:
        altitude_state = "own_below"
    else:
        altitude_state = "co_altitude"
    return TacticalContext(
        geometry_state=dynamic_state,
        phase=phase,
        energy_state=energy_state,
        altitude_state=altitude_state,
    )


def _scenario_signature(metadata: Mapping[str, Any], scenario_name: str) -> str:
    fields = (
        metadata.get("distance_speed_package"),
        metadata.get("height_condition"),
        metadata.get("mirror_sign"),
    )
    if all(value not in (None, "") for value in fields):
        return "|".join(str(value) for value in fields)
    return str(scenario_name)


def _safe_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return normalized or "unnamed"


class SingleMotifContinuous66DCollector:
    """Observe a real rollout without changing any control value."""

    def __init__(
        self,
        *,
        plan: SingleMotifPlan,
        run_id: str,
        method_name: str,
        task_name: str,
        opponent_id: str,
        seed: int,
        episode: int,
        scenario_name: str,
        scenario_metadata: Mapping[str, Any],
        observation_schema: Mapping[str, Any],
        environment_episode: int,
        encoder: Any | None = None,
    ) -> None:
        if method_name != "legacy_static_oracle_task_gate":
            raise SingleMotifContractError("collector requires the frozen oracle task gate")
        if opponent_id not in plan.opponents:
            raise SingleMotifContractError(f"unexpected opponent: {opponent_id}")
        self.plan = plan
        self.contract = FiveStateObservationContract(
            temporal_embedding_dim=32,
            history_window_steps=plan.history_window_steps,
        )
        self.encoder = encoder or FrozenP3Encoder(
            plan.p3_checkpoint,
            plan.p3_checkpoint_sha256,
        )
        self.history: deque[np.ndarray] = deque(maxlen=plan.history_window_steps)
        self.history_features = TemporalGeometryFeatureExtractor(plan.history_window_steps)
        self.phase_tracker = PhaseTracker()
        self.observation_schema = dict(observation_schema)
        self.environment_episode = int(environment_episode)
        self.first_pass_complete = False
        self.pending: dict[str, Any] | None = None
        self.steps: list[dict[str, Any]] = []
        self.lineage_id = telemetry_sha256(
            {
                "source_id": plan.source_id,
                "run_id": run_id,
                "method": method_name,
                "task": task_name,
                "opponent": opponent_id,
                "seed": int(seed),
                "episode": int(episode),
                "scenario": scenario_name,
                "environment_episode": int(environment_episode),
            }
        )
        self.header = {
            "source_id": plan.source_id,
            "mode": "observe_only",
            "training_permitted": False,
            "action_replacement_permitted": False,
            "run_id": run_id,
            "method": method_name,
            "task": task_name,
            "opponent": opponent_id,
            "seed": int(seed),
            "episode": int(episode),
            "scenario": scenario_name,
            "scenario_metadata": dict(scenario_metadata),
            "scenario_signature": _scenario_signature(scenario_metadata, scenario_name),
            "mirror_sign": scenario_metadata.get("mirror_sign"),
            "lineage_id": self.lineage_id,
            "environment_episode": int(environment_episode),
            "observation_schema": dict(observation_schema),
            "observation_schema_sha256": telemetry_sha256(observation_schema),
            "feature_names_66d": list(self.contract.feature_names),
            "p3_checkpoint": str(plan.p3_checkpoint),
            "p3_checkpoint_sha256": plan.p3_checkpoint_sha256,
            "target_skill": plan.target_skill,
            "target_profile": plan.target_profile,
            "target_state": plan.target_state,
            "target_phases": list(plan.target_phases),
        }

    def begin_step(
        self,
        *,
        step: int,
        observation: Mapping[str, Any],
        action: Any,
        environment_episode: int,
    ) -> None:
        """Freeze the exact pre-step observation/action without mutation."""

        if self.pending is not None:
            raise SingleMotifContractError("previous collector step was not completed")
        if int(environment_episode) != self.environment_episode:
            raise SingleMotifContractError("environment episode lineage changed before step")
        if action is None:
            raise SingleMotifContractError("collector requires an explicit 3-D VPP action")
        base = project_base_observation(
            observation.get("observation_vector"),
            self.observation_schema,
        )
        self.history.append(base.copy())
        explicit = self.history_features.update(_history_frame(base))
        values = _base_map(base)
        aa_deg = _angle_deg(values["aa_sin"], values["aa_cos"])
        ata_deg = _angle_deg(values["ata_sin"], values["ata_cos"])
        dynamic_state = classify_dynamic_state(aa_deg, ata_deg)
        if dynamic_state == "unknown":
            raise SingleMotifContractError("dynamic taxonomy is unknown")
        phase = self.phase_tracker.update(values["range_m"], values["range_rate_mps"])
        context = _context(base, dynamic_state, phase, self.plan)
        validity = build_validity_mask(context)
        skill_index = SKILL_NAMES.index(self.plan.target_skill)
        profile_index = PROFILE_NAMES.index(self.plan.target_profile)
        target_pair_allowed = bool(validity["mask"][skill_index, profile_index])

        action_vector = np.asarray(action, dtype=np.float32).reshape(-1)
        action_finite = action_vector.shape == (3,) and bool(np.isfinite(action_vector).all())
        action_in_bounds = action_finite and bool(np.all(np.abs(action_vector) <= 1.0 + 1e-6))
        if not action_finite or not action_in_bounds:
            raise SingleMotifContractError("reference action is not finite normalized 3-D VPP")

        contract_ready = len(self.history) == self.plan.history_window_steps
        item: dict[str, Any] = {
            "step": int(step),
            "pre_step_time_s": float(max(0, int(step) - 1) * 0.2),
            "lineage_id": self.lineage_id,
            "environment_episode": self.environment_episode,
            "reset_occurred": False,
            "future_state_injected": False,
            "history_padding_used": False,
            "history_size": len(self.history),
            "contract_ready": contract_ready,
            "base_observation_16d": _vector_list(base, 16),
            "explicit_history_6d": _vector_list(explicit, 6),
            "dynamic_state": dynamic_state,
            "phase": phase,
            "first_pass_complete_before_step": self.first_pass_complete,
            "aa_deg": aa_deg,
            "ata_deg": ata_deg,
            "range_m": values["range_m"],
            "range_rate_mps": values["range_rate_mps"],
            "validity_context": {
                "geometry_state": context.geometry_state,
                "phase": context.phase,
                "energy_state": context.energy_state,
                "altitude_state": context.altitude_state,
            },
            "validity_mask": validity["mask"].astype(bool).tolist(),
            "target_pair_allowed": target_pair_allowed,
            "normalized_vpp_action_3d": _vector_list(action_vector, 3),
            "action_sha256_before_step": telemetry_sha256(action_vector),
            "action_finite": action_finite,
            "action_in_bounds": action_in_bounds,
        }
        if contract_ready:
            history = np.stack(tuple(self.history), axis=0)
            profile = compile_profile(self.plan.target_profile)
            embedding = np.asarray(self.encoder.encode(history), dtype=np.float32)
            observation_66d = self.contract.compose(
                base_geometry=base,
                explicit_history=explicit,
                intent_targets=profile["target_vector"],
                intent_weights=profile["weight_vector"],
                temporal_embedding=embedding,
            )
            item.update(
                {
                    "observed_history_10x16": history.astype(np.float32).tolist(),
                    "temporal_embedding_32d": _vector_list(embedding, 32),
                    "intent_targets_6d": _vector_list(profile["target_vector"], 6),
                    "intent_weights_6d": _vector_list(profile["weight_vector"], 6),
                    "observation_66d": _vector_list(observation_66d, 66),
                    "observation_66d_sha256": telemetry_sha256(observation_66d),
                    "observation_66d_finite": True,
                }
            )
        self.pending = {
            "record": item,
            "action_copy": action_vector.copy(),
        }

    def finish_step(
        self,
        *,
        step: int,
        action: Any,
        observation: Mapping[str, Any],
        info: Mapping[str, Any],
        environment_episode: int,
    ) -> None:
        """Attach real post-step prediction, VPP, guidance, and PID response."""

        if self.pending is None:
            raise SingleMotifContractError("collector finish_step has no pending step")
        if int(environment_episode) != self.environment_episode:
            raise SingleMotifContractError("environment episode lineage changed after step")
        item = self.pending["record"]
        if int(item["step"]) != int(step):
            raise SingleMotifContractError("collector step alignment drifted")
        action_after = np.asarray(action, dtype=np.float32).reshape(-1)
        action_identity_preserved = bool(
            action_after.shape == (3,)
            and np.array_equal(action_after, self.pending["action_copy"])
        )
        backend = str(info.get("backend", ""))
        backend_fallback = bool(info.get("backend_fallback_occurred", False))
        prediction_valid = bool(info.get("prediction_valid", False))
        prediction_fallback = bool(info.get("prediction_fallback", False))
        own_state = info.get("own_state", {})
        if not isinstance(own_state, Mapping):
            own_state = {}
        item.update(
            {
                "action_sha256_after_step": telemetry_sha256(action_after),
                "action_identity_preserved": action_identity_preserved,
                "backend": backend,
                "backend_fallback_occurred": backend_fallback,
                "strict_jsbsim_lineage": backend == "jsbsim" and not backend_fallback,
                "first_pass_complete_after_step": bool(info.get("first_pass_complete", False)),
                "prediction_valid": prediction_valid,
                "prediction_fallback": prediction_fallback,
                "prediction_state": {
                    "predicted_target_position": info.get("predicted_target_position"),
                    "prediction_error_m": _finite_or_none(info.get("prediction_error_m")),
                    "prediction_lookahead_time_s": _finite_or_none(info.get("prediction_lookahead_time_s")),
                },
                "vpp_state": {
                    "virtual_point_source": info.get("virtual_point_source"),
                    "vp_position_neu": info.get("vp_position_neu"),
                    "vp_offset": info.get("vp_offset"),
                    "vp_world_offset": info.get("vp_world_offset"),
                    "vp_forward_bias_m": _finite_or_none(info.get("vp_forward_bias_m")),
                    "vp_lateral_bias_m": _finite_or_none(info.get("vp_lateral_bias_m")),
                    "effective_guidance_mode": info.get("effective_guidance_mode"),
                },
                "pid_command_response": {
                    "nz_cmd": _finite_or_none(info.get("nz_cmd")),
                    "roll_rate_cmd": _finite_or_none(info.get("roll_rate_cmd")),
                    "throttle_cmd": _finite_or_none(info.get("throttle_cmd")),
                    "actual_nz_g": _finite_or_none(own_state.get("nz_g", info.get("nz_g"))),
                    "actual_aoa_deg": _finite_or_none(info.get("ego_attack_aoa_deg")),
                    "own_speed_mps": _finite_or_none(own_state.get("speed_mps")),
                    "own_altitude_m": _finite_or_none(own_state.get("altitude_m")),
                    "nz_saturated": bool(info.get("nz_saturated", False)),
                    "roll_rate_saturated": bool(info.get("roll_rate_saturated", False)),
                    "throttle_saturated": bool(info.get("throttle_saturated", False)),
                    "saturation_flag": bool(info.get("saturation_flag", False)),
                },
                "post_observation_sha256": telemetry_sha256(observation.get("observation_vector")),
            }
        )
        structural_valid = bool(
            item["contract_ready"]
            and item.get("observation_66d_finite", False)
            and item["action_finite"]
            and item["action_in_bounds"]
            and item["action_identity_preserved"]
            and item["strict_jsbsim_lineage"]
            and not item["reset_occurred"]
            and not item["future_state_injected"]
            and not item["history_padding_used"]
        )
        target_motif = bool(
            item["dynamic_state"] == self.plan.target_state
            and item["phase"] in self.plan.target_phases
            and item["first_pass_complete_before_step"]
            and item["target_pair_allowed"]
        )
        item["structural_contract_valid"] = structural_valid
        item["target_motif_step"] = target_motif
        item["valid_target_step"] = bool(
            structural_valid
            and target_motif
            and prediction_valid
            and not prediction_fallback
        )
        self.first_pass_complete = bool(item["first_pass_complete_after_step"])
        self.steps.append(item)
        self.pending = None

    def ledger(self) -> dict[str, Any]:
        if self.pending is not None:
            raise SingleMotifContractError("collector ledger requested with an unfinished step")
        ready = [item for item in self.steps if item["contract_ready"]]
        target = [item for item in ready if item["target_motif_step"]]
        valid_target = [item for item in target if item["valid_target_step"]]
        return {
            "header": dict(self.header),
            "summary": {
                "recorded_steps": len(self.steps),
                "warmup_steps_without_padding": len(self.steps) - len(ready),
                "contract_ready_steps": len(ready),
                "target_motif_steps": len(target),
                "valid_target_steps": len(valid_target),
                "all_ready_structural_contract_valid": bool(ready) and all(
                    item["structural_contract_valid"] for item in ready
                ),
                "all_ready_prediction_without_fallback": bool(ready) and all(
                    item["prediction_valid"] and not item["prediction_fallback"]
                    for item in ready
                ),
                "all_actions_identity_preserved": bool(self.steps) and all(
                    item["action_identity_preserved"] for item in self.steps
                ),
                "no_reset": bool(self.steps) and all(not item["reset_occurred"] for item in self.steps),
                "no_padding": bool(self.steps) and all(not item["history_padding_used"] for item in self.steps),
                "strict_jsbsim_lineage": bool(self.steps) and all(
                    item["strict_jsbsim_lineage"] for item in self.steps
                ),
            },
            "steps": self.steps,
        }


def write_single_motif_artifacts(
    run_dir: Path,
    records: Sequence[MutableMapping[str, Any]],
) -> Path | None:
    """Move large collector ledgers into independent per-episode artifacts."""

    entries: list[dict[str, Any]] = []
    root = Path(run_dir) / "single_motif_continuous_66d"
    for record in records:
        ledger = record.pop("single_motif_continuous_66d", None)
        if not isinstance(ledger, Mapping):
            continue
        header = _mapping(ledger.get("header"), "collector.header")
        opponent = _safe_name(str(header.get("opponent", "unknown")))
        scenario = _safe_name(str(header.get("scenario", "scenario")))
        seed = int(header.get("seed", 0))
        episode = int(header.get("episode", 0))
        relative = Path(opponent) / scenario / f"seed_{seed}_episode_{episode:03d}.json"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(ledger, indent=2, ensure_ascii=False), encoding="utf-8")
        digest = _sha256_file(path)
        continuity = record.get("phase_reachability_handoff_ledger", {})
        continuity_valid = bool(
            isinstance(continuity, Mapping) and continuity.get("continuity_valid", False)
        )
        entry = {
            "source_id": SOURCE_ID,
            "opponent": header.get("opponent"),
            "scenario": header.get("scenario"),
            "scenario_signature": header.get("scenario_signature"),
            "mirror_sign": header.get("mirror_sign"),
            "seed": seed,
            "episode": episode,
            "artifact_path": str(Path("single_motif_continuous_66d") / relative).replace("\\", "/"),
            "sha256": digest,
            "size_bytes": path.stat().st_size,
            "summary": dict(ledger.get("summary", {})),
            "v2_continuity_valid": continuity_valid,
        }
        entries.append(entry)
        record["single_motif_continuous_66d_artifact"] = dict(entry)
    if not entries:
        return None
    root.mkdir(parents=True, exist_ok=True)
    index_path = root / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "source_id": SOURCE_ID,
                "mode": "observe_only",
                "n_episodes": len(entries),
                "entries": entries,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return index_path
