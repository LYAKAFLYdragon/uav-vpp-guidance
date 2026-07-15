"""Fail-closed observe-only collector for the P2-B5 phase-feasible sampler.

The sampler answers a data-contract question only: whether a frozen head-on
reference rollout can reach a continuous ``disadvantage -> post_merge/re_entry``
motif with valid 66-D observations for every frozen opponent.  It never
replaces the reference action or trains a policy.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from uav_vpp_guidance.evaluation.p4_v2_sampler_feasibility import classify_dynamic_state
from uav_vpp_guidance.evaluation.phase_reachability_handoff_contract import telemetry_sha256
from uav_vpp_guidance.evaluation.single_motif_continuous_66d import (
    _base_map,
    _context,
    _history_frame,
    _vector_list,
    project_base_observation,
)
from uav_vpp_guidance.hierarchy.five_state_observation_contract import FiveStateObservationContract
from uav_vpp_guidance.hierarchy.shared_intent_profiles import compile_profile
from uav_vpp_guidance.hierarchy.shared_intent_validity import (
    PROFILE_NAMES,
    SKILL_NAMES,
    build_validity_mask,
)
from uav_vpp_guidance.hierarchy.temporal_geometry_features import TemporalGeometryFeatureExtractor
from uav_vpp_guidance.training.thesis_shared_skill_geometry import FrozenP3Encoder, PhaseTracker


SOURCE_ID = "THESIS-GLOBAL-ADVANTAGE-V1-P2-PHASE-FEASIBLE-SAMPLER-B5-R1"
MANIFEST_SOURCE_ID = "THESIS-GLOBAL-ADVANTAGE-V1-P2-PHASE-FEASIBLE-SAMPLER-B5-MANIFEST12-R1"
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
TARGET_SKILL = "defensive_extension"
TARGET_PROFILE = "range_extension"
TARGET_STATE = "disadvantage"
TARGET_PHASES = ("post_merge", "re_entry")
RAW_SI_CONTRACT = "raw_SI_from_observation.relative_state"


class PhaseFeasibleSamplerError(ValueError):
    """Raised when the B5 sampler leaves its preregistered boundary."""


@dataclass(frozen=True)
class PhaseFeasibleSamplerPlan:
    source_id: str
    execution_permitted: bool
    p3_checkpoint: Path
    p3_checkpoint_sha256: str
    reference_specialist: str
    reference_specialist_sha256: str
    opponents: tuple[str, ...]
    history_window_steps: int
    min_episodes_per_opponent: int
    min_target_steps_per_opponent: int
    min_signatures_per_opponent: int
    min_mirror_signs_per_opponent: int
    max_high_level_steps: int
    min_free_disk_gb: float
    energy_band_m: float
    altitude_band_m: float
    range_scale_m: float
    range_rate_scale_mps: float
    range_tolerance_m: float
    range_rate_tolerance_mps: float


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PhaseFeasibleSamplerError(f"{name} must be a mapping")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PhaseFeasibleSamplerError(f"{name} must be a positive integer")
    return int(value)


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise PhaseFeasibleSamplerError(f"{name} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise PhaseFeasibleSamplerError(f"{name} must be a positive finite number")
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_phase_feasible_sampler_plan(config: Mapping[str, Any]) -> PhaseFeasibleSamplerPlan:
    """Validate the design or a separately authorised B5 execution config."""

    if str(config.get("source_id", "")) != SOURCE_ID:
        raise PhaseFeasibleSamplerError("unexpected B5 source ID")
    protocol = _mapping(config.get("phase_feasible_sampler_b5"), "phase_feasible_sampler_b5")
    if str(protocol.get("source_id", "")) != SOURCE_ID:
        raise PhaseFeasibleSamplerError("protocol source ID drifted")
    authorization = _mapping(config.get("authorization"), "authorization")
    for key in (
        "training_permitted",
        "tuning_permitted",
        "candidate_policy_loading_permitted",
        "action_replacement_permitted",
        "policy_change_permitted",
        "vpp_change_permitted",
        "guidance_change_permitted",
        "pid_change_permitted",
        "reset_after_first_pass",
        "snapshot_restore",
        "future_state_injection",
        "history_padding_permitted",
        "heldout_claims_permitted",
    ):
        if authorization.get(key) is not False:
            raise PhaseFeasibleSamplerError(f"authorization.{key} must remain false")
    if not isinstance(authorization.get("execution_permitted"), bool):
        raise PhaseFeasibleSamplerError("authorization.execution_permitted must be explicit")

    reference = _mapping(protocol.get("reference"), "phase_feasible_sampler_b5.reference")
    if reference.get("controller") != "frozen_fixed_head_on_specialist":
        raise PhaseFeasibleSamplerError("B5 must retain the frozen head-on reference")
    if reference.get("learning") is not False or reference.get("routing") is not False:
        raise PhaseFeasibleSamplerError("B5 reference must remain fixed and non-learning")
    reference_sha = str(reference.get("checkpoint_sha256", "")).lower()
    if len(reference_sha) != 64:
        raise PhaseFeasibleSamplerError("reference checkpoint SHA-256 is invalid")

    target = _mapping(protocol.get("target_motif"), "phase_feasible_sampler_b5.target_motif")
    if (
        target.get("skill") != TARGET_SKILL
        or target.get("profile") != TARGET_PROFILE
        or target.get("dynamic_state") != TARGET_STATE
        or tuple(target.get("phases", ())) != TARGET_PHASES
    ):
        raise PhaseFeasibleSamplerError("B5 target motif drifted")

    phase_input = _mapping(protocol.get("phase_input"), "phase_feasible_sampler_b5.phase_input")
    if phase_input.get("unit_contract") != RAW_SI_CONTRACT:
        raise PhaseFeasibleSamplerError("B5 phase tracking must use raw SI relative state")

    observation = _mapping(protocol.get("observation_contract"), "phase_feasible_sampler_b5.observation_contract")
    if (
        int(observation.get("base_dim", -1)) != 16
        or int(observation.get("temporal_embedding_dim", -1)) != 32
        or int(observation.get("composed_dim", -1)) != 66
        or int(observation.get("history_window_steps", -1)) != 10
    ):
        raise PhaseFeasibleSamplerError("B5 must preserve the frozen 66-D observation contract")

    p3 = _mapping(protocol.get("frozen_p3_encoder"), "phase_feasible_sampler_b5.frozen_p3_encoder")
    checkpoint = Path(str(p3.get("checkpoint", "")))
    p3_sha = str(p3.get("sha256", "")).lower()
    if not checkpoint.is_absolute() or len(p3_sha) != 64:
        raise PhaseFeasibleSamplerError("B5 P3 checkpoint path/SHA is invalid")

    opponents = tuple(protocol.get("opponents", ()))
    if opponents != OPPONENTS:
        raise PhaseFeasibleSamplerError("B5 must report the frozen three opponents separately")
    gate = _mapping(protocol.get("gate"), "phase_feasible_sampler_b5.gate")
    contract = _mapping(config.get("contract"), "contract")
    if (
        contract.get("backend") != "jsbsim"
        or contract.get("strict_backend") is not True
        or contract.get("fresh_environment_per_episode") is not True
        or contract.get("phase_input_unit_contract") != RAW_SI_CONTRACT
    ):
        raise PhaseFeasibleSamplerError("B5 strict-JSBSim/raw-SI contract drifted")
    bands = _mapping(protocol.get("context_bands"), "phase_feasible_sampler_b5.context_bands")
    normalization = _mapping(
        phase_input.get("policy_normalization_diagnostic"),
        "phase_feasible_sampler_b5.phase_input.policy_normalization_diagnostic",
    )
    return PhaseFeasibleSamplerPlan(
        source_id=SOURCE_ID,
        execution_permitted=bool(authorization["execution_permitted"]),
        p3_checkpoint=checkpoint,
        p3_checkpoint_sha256=p3_sha,
        reference_specialist=str(reference.get("registry_key", "")),
        reference_specialist_sha256=reference_sha,
        opponents=opponents,
        history_window_steps=10,
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
        max_high_level_steps=_positive_int(contract.get("max_high_level_steps"), "contract.max_high_level_steps"),
        min_free_disk_gb=_positive_float(contract.get("min_free_disk_gb"), "contract.min_free_disk_gb"),
        energy_band_m=_positive_float(
            bands.get("specific_energy_height_band_m"),
            "context_bands.specific_energy_height_band_m",
        ),
        altitude_band_m=_positive_float(bands.get("altitude_band_m"), "context_bands.altitude_band_m"),
        range_scale_m=_positive_float(normalization.get("range_m_scale"), "range_m_scale"),
        range_rate_scale_mps=_positive_float(
            normalization.get("range_rate_mps_scale"), "range_rate_mps_scale"
        ),
        range_tolerance_m=_positive_float(normalization.get("range_tolerance_m"), "range_tolerance_m"),
        range_rate_tolerance_mps=_positive_float(
            normalization.get("range_rate_tolerance_mps"), "range_rate_tolerance_mps"
        ),
    )


def raw_si_phase_input(observation: Mapping[str, Any]) -> dict[str, float | str]:
    """Return the only values allowed to drive B5 phase thresholds."""

    relative = observation.get("relative_state")
    if not isinstance(relative, Mapping):
        raise PhaseFeasibleSamplerError("observation is missing raw relative_state")
    try:
        range_m = float(relative["range_m"])
        range_rate_mps = float(relative["range_rate_mps"])
    except (KeyError, TypeError, ValueError) as error:
        raise PhaseFeasibleSamplerError("raw relative range/range-rate is invalid") from error
    if not math.isfinite(range_m) or not math.isfinite(range_rate_mps):
        raise PhaseFeasibleSamplerError("raw relative range/range-rate must be finite")
    return {
        "range_m": range_m,
        "range_rate_mps": range_rate_mps,
        "unit_contract": RAW_SI_CONTRACT,
    }


def policy_normalization_diagnostic(
    base_values: Mapping[str, float], plan: PhaseFeasibleSamplerPlan
) -> dict[str, float | bool]:
    """Persist the policy-vector scale check without feeding it to phase logic."""

    normalized_range = float(base_values["range_m"])
    normalized_rate = float(base_values["range_rate_mps"])
    return {
        "range_m_normalized": normalized_range,
        "range_rate_mps_normalized": normalized_rate,
        "range_m_scale": plan.range_scale_m,
        "range_rate_mps_scale": plan.range_rate_scale_mps,
    }


def normalization_is_consistent(
    raw: Mapping[str, Any], diagnostic: Mapping[str, Any], plan: PhaseFeasibleSamplerPlan
) -> bool:
    """Check the frozen diagnostic scale; never infer phase from this vector."""

    try:
        range_error = abs(
            float(diagnostic["range_m_normalized"]) * plan.range_scale_m - float(raw["range_m"])
        )
        rate_error = abs(
            float(diagnostic["range_rate_mps_normalized"]) * plan.range_rate_scale_mps
            - float(raw["range_rate_mps"])
        )
    except (KeyError, TypeError, ValueError):
        return False
    return range_error <= plan.range_tolerance_m and rate_error <= plan.range_rate_tolerance_mps


def _scenario_signature(metadata: Mapping[str, Any], scenario_name: str) -> str:
    fields = (
        metadata.get("distance_speed_package"),
        metadata.get("height_condition"),
    )
    return "|".join(str(value) for value in fields) if all(fields) else str(scenario_name)


class PhaseFeasibleSamplerCollector:
    """Record a continuous, action-preserving B5 rollout in raw-SI phase units."""

    def __init__(
        self,
        *,
        plan: PhaseFeasibleSamplerPlan,
        opponent: str,
        scenario_name: str,
        scenario_metadata: Mapping[str, Any],
        seed: int,
        environment_episode: int,
        observation_schema: Mapping[str, Any],
        encoder: Any | None = None,
    ) -> None:
        if opponent not in plan.opponents:
            raise PhaseFeasibleSamplerError(f"unexpected opponent: {opponent}")
        self.plan = plan
        self.opponent = opponent
        self.environment_episode = int(environment_episode)
        self.schema = dict(observation_schema)
        self.contract = FiveStateObservationContract(temporal_embedding_dim=32, history_window_steps=10)
        self.encoder = encoder or FrozenP3Encoder(plan.p3_checkpoint, plan.p3_checkpoint_sha256)
        self.history: deque[np.ndarray] = deque(maxlen=plan.history_window_steps)
        self.history_features = TemporalGeometryFeatureExtractor(plan.history_window_steps)
        self.phase_tracker = PhaseTracker()
        self.first_pass_complete = False
        self.first_pass_boundary: dict[str, Any] | None = None
        self.continuity_receipt: dict[str, Any] | None = None
        self.pending: dict[str, Any] | None = None
        self.steps: list[dict[str, Any]] = []
        self.lineage_id = telemetry_sha256(
            {
                "source_id": plan.source_id,
                "opponent": opponent,
                "scenario": scenario_name,
                "seed": int(seed),
                "environment_episode": self.environment_episode,
            }
        )
        self.header = {
            "source_id": plan.source_id,
            "mode": "observe_only_action_preserving",
            "opponent": opponent,
            "scenario": scenario_name,
            "scenario_metadata": dict(scenario_metadata),
            "scenario_signature": _scenario_signature(scenario_metadata, scenario_name),
            "mirror_sign": scenario_metadata.get("mirror_sign"),
            "seed": int(seed),
            "environment_episode": self.environment_episode,
            "lineage_id": self.lineage_id,
            "observation_schema": self.schema,
            "observation_schema_sha256": telemetry_sha256(self.schema),
            "phase_input_unit_contract": RAW_SI_CONTRACT,
            "reference_specialist": plan.reference_specialist,
            "reference_specialist_sha256": plan.reference_specialist_sha256,
            "p3_checkpoint": str(plan.p3_checkpoint),
            "p3_checkpoint_sha256": plan.p3_checkpoint_sha256,
            "target_skill": TARGET_SKILL,
            "target_profile": TARGET_PROFILE,
            "target_state": TARGET_STATE,
            "target_phases": list(TARGET_PHASES),
            "training_permitted": False,
            "action_replacement_permitted": False,
        }

    def begin_step(self, *, step: int, observation: Mapping[str, Any], action: Any) -> None:
        """Freeze pre-step data; this method does not mutate the environment or action."""

        if self.pending is not None:
            raise PhaseFeasibleSamplerError("previous sampler step was not completed")
        base = project_base_observation(observation.get("observation_vector"), self.schema)
        values = _base_map(base)
        raw_phase = raw_si_phase_input(observation)
        diagnostic = policy_normalization_diagnostic(values, self.plan)
        self.history.append(base.copy())
        explicit_history = self.history_features.update(_history_frame(base))
        aa_deg = abs(math.degrees(math.atan2(values["aa_sin"], values["aa_cos"])))
        ata_deg = abs(math.degrees(math.atan2(values["ata_sin"], values["ata_cos"])))
        dynamic_state = classify_dynamic_state(aa_deg, ata_deg)
        if dynamic_state == "unknown":
            raise PhaseFeasibleSamplerError("dynamic taxonomy is unknown")
        phase = self.phase_tracker.update(raw_phase["range_m"], raw_phase["range_rate_mps"])
        context = _context(base, dynamic_state, phase, self.plan)
        validity = build_validity_mask(context)
        target_allowed = bool(
            validity["mask"][SKILL_NAMES.index(TARGET_SKILL), PROFILE_NAMES.index(TARGET_PROFILE)]
        )
        action_vector = np.asarray(action, dtype=np.float32).reshape(-1)
        action_finite = action_vector.shape == (3,) and bool(np.isfinite(action_vector).all())
        action_in_bounds = action_finite and bool(np.all(np.abs(action_vector) <= 1.0 + 1e-6))
        if not action_in_bounds:
            raise PhaseFeasibleSamplerError("reference action is not finite normalized 3-D VPP")
        ready = len(self.history) == self.plan.history_window_steps
        record: dict[str, Any] = {
            "step": int(step),
            "lineage_id": self.lineage_id,
            "environment_episode": self.environment_episode,
            "reset_occurred": False,
            "future_state_injected": False,
            "history_padding_used": False,
            "history_size": len(self.history),
            "contract_ready": ready,
            "base_observation_16d": _vector_list(base, 16),
            "explicit_history_6d": _vector_list(explicit_history, 6),
            "raw_si_phase_input": raw_phase,
            "policy_normalization_diagnostic": diagnostic,
            "normalization_consistent": normalization_is_consistent(raw_phase, diagnostic, self.plan),
            "dynamic_state": dynamic_state,
            "phase": phase,
            "aa_deg": aa_deg,
            "ata_deg": ata_deg,
            "first_pass_complete_before_step": self.first_pass_complete,
            "validity_context": {
                "geometry_state": context.geometry_state,
                "phase": context.phase,
                "energy_state": context.energy_state,
                "altitude_state": context.altitude_state,
            },
            "target_pair_allowed": target_allowed,
            "normalized_vpp_action_3d": _vector_list(action_vector, 3),
            "action_sha256_before_step": telemetry_sha256(action_vector),
            "action_finite": action_finite,
            "action_in_bounds": action_in_bounds,
        }
        if ready:
            history = np.stack(tuple(self.history), axis=0)
            profile = compile_profile(TARGET_PROFILE)
            embedding = np.asarray(self.encoder.encode(history), dtype=np.float32)
            observation_66d = self.contract.compose(
                base_geometry=base,
                explicit_history=explicit_history,
                intent_targets=profile["target_vector"],
                intent_weights=profile["weight_vector"],
                temporal_embedding=embedding,
            )
            record.update(
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
        self.pending = {"record": record, "action": action_vector.copy()}

    def finish_step(self, *, step: int, action: Any, info: Mapping[str, Any]) -> None:
        """Attach physical response data after one real, uninterrupted JSBSim step."""

        if self.pending is None:
            raise PhaseFeasibleSamplerError("sampler finish_step has no pending step")
        record = self.pending["record"]
        if int(record["step"]) != int(step):
            raise PhaseFeasibleSamplerError("sampler step alignment drifted")
        action_after = np.asarray(action, dtype=np.float32).reshape(-1)
        action_preserved = bool(
            action_after.shape == (3,) and np.array_equal(action_after, self.pending["action"])
        )
        backend = str(info.get("backend", ""))
        fallback = bool(info.get("backend_fallback_occurred", False))
        record.update(
            {
                "action_sha256_after_step": telemetry_sha256(action_after),
                "action_identity_preserved": action_preserved,
                "backend": backend,
                "backend_fallback_occurred": fallback,
                "strict_jsbsim_lineage": backend == "jsbsim" and not fallback,
                "prediction_valid": bool(info.get("prediction_valid", False)),
                "prediction_fallback": bool(info.get("prediction_fallback", False)),
                "first_pass_complete_after_step": bool(info.get("first_pass_complete", False)),
                "terminal_reason": info.get("reason"),
                "vpp_guidance_pid": {
                    "virtual_point_source": info.get("virtual_point_source"),
                    "vp_position_neu": info.get("vp_position_neu"),
                    "vp_offset": info.get("vp_offset"),
                    "effective_guidance_mode": info.get("effective_guidance_mode"),
                    "nz_cmd": info.get("nz_cmd"),
                    "roll_rate_cmd": info.get("roll_rate_cmd"),
                    "throttle_cmd": info.get("throttle_cmd"),
                    "nz_saturated": bool(info.get("nz_saturated", False)),
                    "roll_rate_saturated": bool(info.get("roll_rate_saturated", False)),
                    "throttle_saturated": bool(info.get("throttle_saturated", False)),
                },
            }
        )
        record["structural_contract_valid"] = bool(
            record["contract_ready"]
            and record.get("observation_66d_finite", False)
            and record["action_finite"]
            and record["action_in_bounds"]
            and record["action_identity_preserved"]
            and record["strict_jsbsim_lineage"]
            and record["normalization_consistent"]
            and not record["reset_occurred"]
            and not record["future_state_injected"]
            and not record["history_padding_used"]
        )
        record["target_motif_step"] = bool(
            record["dynamic_state"] == TARGET_STATE
            and record["phase"] in TARGET_PHASES
            and record["first_pass_complete_before_step"]
            and record["target_pair_allowed"]
        )
        record["valid_target_step"] = bool(
            record["structural_contract_valid"]
            and record["target_motif_step"]
            and record["prediction_valid"]
            and not record["prediction_fallback"]
        )
        if record["first_pass_complete_after_step"] and self.first_pass_boundary is None:
            self.first_pass_boundary = {
                "step": int(step),
                "lineage_id": self.lineage_id,
                "environment_episode": self.environment_episode,
                "raw_si_phase_input_sha256": telemetry_sha256(record["raw_si_phase_input"]),
                "action_sha256": record["action_sha256_after_step"],
            }
        elif self.first_pass_boundary is not None and self.continuity_receipt is None:
            parent_step = int(self.first_pass_boundary["step"])
            if int(step) == parent_step + 1:
                self.continuity_receipt = {
                    "passed": True,
                    "boundary_step": parent_step,
                    "sampler_step": int(step),
                    "same_lineage": record["lineage_id"] == self.first_pass_boundary["lineage_id"],
                    "same_environment_episode": record["environment_episode"]
                    == self.first_pass_boundary["environment_episode"],
                    "no_reset": not record["reset_occurred"],
                    "no_future_state_injection": not record["future_state_injected"],
                }
                self.continuity_receipt["passed"] = all(
                    value for key, value in self.continuity_receipt.items() if key != "passed" and isinstance(value, bool)
                )
        self.first_pass_complete = bool(record["first_pass_complete_after_step"])
        self.steps.append(record)
        self.pending = None

    def ledger(self) -> dict[str, Any]:
        if self.pending is not None:
            raise PhaseFeasibleSamplerError("sampler ledger requested with an unfinished step")
        ready = [record for record in self.steps if record["contract_ready"]]
        valid_target = [record for record in ready if record["valid_target_step"]]
        return {
            "header": dict(self.header),
            "summary": {
                "recorded_steps": len(self.steps),
                "contract_ready_steps": len(ready),
                "valid_target_steps": len(valid_target),
                "first_pass_observed": self.first_pass_boundary is not None,
                "continuity_valid": bool(self.continuity_receipt and self.continuity_receipt["passed"]),
                "all_ready_structural_contract_valid": bool(ready)
                and all(record["structural_contract_valid"] for record in ready),
                "all_actions_identity_preserved": bool(self.steps)
                and all(record["action_identity_preserved"] for record in self.steps),
                "strict_jsbsim_lineage": bool(self.steps)
                and all(record["strict_jsbsim_lineage"] for record in self.steps),
                "no_reset": bool(self.steps) and all(not record["reset_occurred"] for record in self.steps),
                "no_padding": bool(self.steps)
                and all(not record["history_padding_used"] for record in self.steps),
            },
            "first_pass_boundary": self.first_pass_boundary,
            "continuity_receipt": self.continuity_receipt,
            "steps": self.steps,
        }


def evaluate_gate(ledgers: Sequence[Mapping[str, Any]], plan: PhaseFeasibleSamplerPlan) -> dict[str, Any]:
    """Apply B5 prerequisites per opponent without pooling evidence."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    for ledger in ledgers:
        header = _mapping(ledger.get("header"), "ledger.header")
        summary = _mapping(ledger.get("summary"), "ledger.summary")
        opponent = str(header.get("opponent", ""))
        if opponent not in plan.opponents:
            raise PhaseFeasibleSamplerError("ledger opponent is outside the B5 registry")
        valid_steps = int(summary.get("valid_target_steps", 0))
        episode_contract = all(
            (
                bool(summary.get("all_ready_structural_contract_valid", False)),
                bool(summary.get("all_actions_identity_preserved", False)),
                bool(summary.get("strict_jsbsim_lineage", False)),
                bool(summary.get("no_reset", False)),
                bool(summary.get("no_padding", False)),
                bool(summary.get("continuity_valid", False)),
                bool(summary.get("scenario_application_passed", False)),
                bool(summary.get("raw_si_phase_replay_passed", False)),
                bool(summary.get("initial_pre_merge_semantics_passed", False)),
            )
        )
        row = {
            "opponent": opponent,
            "scenario": header.get("scenario"),
            "scenario_signature": header.get("scenario_signature"),
            "mirror_sign": header.get("mirror_sign"),
            "valid_target_steps": valid_steps,
            "episode_contract_valid": episode_contract,
            "qualifying_episode": episode_contract and valid_steps > 0,
        }
        grouped[opponent].append(row)
        rows.append(row)

    opponents: dict[str, Any] = {}
    for opponent in plan.opponents:
        opponent_rows = grouped[opponent]
        qualifying = [row for row in opponent_rows if row["qualifying_episode"]]
        steps = sum(int(row["valid_target_steps"]) for row in qualifying)
        signatures = sorted({str(row["scenario_signature"]) for row in qualifying})
        mirrors = sorted({str(row["mirror_sign"]) for row in qualifying})
        checks = {
            "episodes_present": bool(opponent_rows),
            "minimum_qualifying_episodes": len(qualifying) >= plan.min_episodes_per_opponent,
            "minimum_valid_target_steps": steps >= plan.min_target_steps_per_opponent,
            "minimum_distinct_scenario_signatures": len(signatures) >= plan.min_signatures_per_opponent,
            "minimum_distinct_mirror_signs": len(mirrors) >= plan.min_mirror_signs_per_opponent,
        }
        opponents[opponent] = {
            "episodes": len(opponent_rows),
            "qualifying_episodes": len(qualifying),
            "valid_target_steps": steps,
            "distinct_scenario_signatures": signatures,
            "distinct_mirror_signs": mirrors,
            "checks": checks,
            "passed": all(checks.values()),
        }
    passed = all(item["passed"] for item in opponents.values())
    return {
        "source_id": plan.source_id,
        "gate_passed": passed,
        "training_unlocked": False,
        "pilot_preregistration_may_be_drafted": passed,
        "verdict": "phase_feasible_data_contract_established" if passed else "phase_feasible_data_contract_not_established",
        "opponents": opponents,
        "episode_rows": rows,
        "pooling_prohibited": True,
    }


def sha256_file(path: Path) -> str:
    """Public hash helper for the B5 runner and verifier."""

    return _sha256_file(Path(path))
