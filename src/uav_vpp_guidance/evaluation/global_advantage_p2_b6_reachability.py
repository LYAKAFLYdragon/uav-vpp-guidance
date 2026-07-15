"""Profile-free continuous collector and selector for B6 reachability evidence."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import hashlib
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
from uav_vpp_guidance.hierarchy.shared_intent_validity import PROFILE_NAMES, build_validity_mask
from uav_vpp_guidance.hierarchy.temporal_geometry_features import TemporalGeometryFeatureExtractor
from uav_vpp_guidance.training.thesis_shared_skill_geometry import FrozenP3Encoder, PhaseTracker


SOURCE_ID = "THESIS-GLOBAL-ADVANTAGE-V1-P2-OPPONENT-CONDITIONAL-REACHABILITY-B6-R1"
MANIFEST_SOURCE_ID = "THESIS-GLOBAL-ADVANTAGE-V1-P2-OPPONENT-CONDITIONAL-REACHABILITY-B6-MANIFEST60-R1"
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
STATES = ("advantage", "head_on", "disadvantage", "neutral", "crossing_entry")
PHASES = ("post_merge", "re_entry")
RAW_SI_CONTRACT = "raw_SI_from_observation.relative_state"


class B6ReachabilityError(ValueError):
    """Raised when B6 leaves its preregistered, non-learning contract."""


@dataclass(frozen=True)
class B6ReachabilityPlan:
    source_id: str
    execution_permitted: bool
    p3_checkpoint: Path
    p3_checkpoint_sha256: str
    reference_specialist: str
    reference_specialist_sha256: str
    opponents: tuple[str, ...]
    history_window_steps: int
    max_high_level_steps: int
    min_free_disk_gb: float
    min_episodes: int
    min_steps: int
    min_signatures: int
    min_mirrors: int
    energy_band_m: float
    altitude_band_m: float


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise B6ReachabilityError(f"{name} must be a mapping")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise B6ReachabilityError(f"{name} must be a positive integer")
    return int(value)


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise B6ReachabilityError(f"{name} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise B6ReachabilityError(f"{name} must be a positive finite number")
    return result


def build_b6_plan(config: Mapping[str, Any]) -> B6ReachabilityPlan:
    """Validate the B6 design, or a future separately authorised run config."""

    if config.get("source_id") != SOURCE_ID:
        raise B6ReachabilityError("unexpected B6 source ID")
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
        "snapshot_restore",
        "future_state_injection",
        "history_padding_permitted",
        "heldout_claims_permitted",
    ):
        if authorization.get(key) is not False:
            raise B6ReachabilityError(f"authorization.{key} must remain false")
    if not isinstance(authorization.get("execution_permitted"), bool):
        raise B6ReachabilityError("authorization.execution_permitted must be explicit")
    inputs = _mapping(config.get("inputs"), "inputs")
    checkpoint = Path(str(inputs.get("frozen_p3_encoder", "")))
    p3_sha = str(inputs.get("frozen_p3_encoder_sha256", "")).lower()
    reference_sha = str(inputs.get("reference_specialist_sha256", "")).lower()
    if not checkpoint.is_absolute() or len(p3_sha) != 64 or len(reference_sha) != 64:
        raise B6ReachabilityError("B6 frozen asset path/SHA is invalid")
    contract = _mapping(config.get("contract"), "contract")
    if (
        contract.get("backend") != "jsbsim"
        or contract.get("strict_backend") is not True
        or contract.get("fresh_environment_per_episode") is not True
        or contract.get("phase_input_unit_contract") != RAW_SI_CONTRACT
        or tuple(contract.get("taxonomy_angle_order", ())) != ("ATA", "AA")
    ):
        raise B6ReachabilityError("B6 physical/raw-SI/taxonomy contract drifted")
    capture = _mapping(contract.get("observation_capture"), "contract.observation_capture")
    if (
        int(capture.get("base_dim", -1)) != 16
        or int(capture.get("real_history_window_steps", -1)) != 10
        or int(capture.get("p3_embedding_dim", -1)) != 32
        or int(capture.get("raw_action_dim", -1)) != 3
        or capture.get("record_profile_free") is not True
        or capture.get("record_all_profile_masks") is not True
    ):
        raise B6ReachabilityError("B6 profile-free observation contract drifted")
    gate = _mapping(contract.get("candidate_gate_per_opponent"), "contract.candidate_gate_per_opponent")
    bands = _mapping(contract.get("context_bands"), "contract.context_bands")
    return B6ReachabilityPlan(
        source_id=SOURCE_ID,
        execution_permitted=bool(authorization["execution_permitted"]),
        p3_checkpoint=checkpoint,
        p3_checkpoint_sha256=p3_sha,
        reference_specialist=str(inputs.get("reference_specialist", "")),
        reference_specialist_sha256=reference_sha,
        opponents=OPPONENTS,
        history_window_steps=10,
        max_high_level_steps=_positive_int(contract.get("max_high_level_steps"), "contract.max_high_level_steps"),
        min_free_disk_gb=_positive_float(contract.get("min_free_disk_gb"), "contract.min_free_disk_gb"),
        min_episodes=_positive_int(gate.get("qualifying_episodes"), "candidate_gate_per_opponent.qualifying_episodes"),
        min_steps=_positive_int(gate.get("valid_steps"), "candidate_gate_per_opponent.valid_steps"),
        min_signatures=_positive_int(gate.get("distinct_signatures"), "candidate_gate_per_opponent.distinct_signatures"),
        min_mirrors=_positive_int(gate.get("distinct_mirrors"), "candidate_gate_per_opponent.distinct_mirrors"),
        energy_band_m=_positive_float(
            bands.get("specific_energy_height_band_m"), "context_bands.specific_energy_height_band_m"
        ),
        altitude_band_m=_positive_float(bands.get("altitude_band_m"), "context_bands.altitude_band_m"),
    )


def raw_si_phase_input(observation: Mapping[str, Any]) -> dict[str, float | str]:
    """Read the only geometry values allowed to change the phase tracker."""

    relative = observation.get("relative_state")
    if not isinstance(relative, Mapping):
        raise B6ReachabilityError("observation is missing raw relative_state")
    try:
        range_m = float(relative["range_m"])
        range_rate_mps = float(relative["range_rate_mps"])
    except (KeyError, TypeError, ValueError) as error:
        raise B6ReachabilityError("raw relative range/range-rate is invalid") from error
    if not math.isfinite(range_m) or not math.isfinite(range_rate_mps):
        raise B6ReachabilityError("raw relative range/range-rate must be finite")
    return {"range_m": range_m, "range_rate_mps": range_rate_mps, "unit_contract": RAW_SI_CONTRACT}


def _scenario_signature(metadata: Mapping[str, Any], scenario_name: str) -> str:
    fields = (metadata.get("distance_speed_package"), metadata.get("height_condition"))
    return "|".join(str(value) for value in fields) if all(fields) else str(scenario_name)


class ProfileFreeReachabilityCollector:
    """Capture all dynamic state/phase samples without selecting a skill or profile."""

    def __init__(
        self,
        *,
        plan: B6ReachabilityPlan,
        opponent: str,
        scenario_name: str,
        scenario_metadata: Mapping[str, Any],
        seed: int,
        environment_episode: int,
        observation_schema: Mapping[str, Any],
        encoder: Any | None = None,
    ) -> None:
        if opponent not in plan.opponents:
            raise B6ReachabilityError(f"unexpected B6 opponent: {opponent}")
        self.plan = plan
        self.environment_episode = int(environment_episode)
        self.schema = dict(observation_schema)
        self.contract = FiveStateObservationContract(temporal_embedding_dim=32, history_window_steps=10)
        self.encoder = encoder or FrozenP3Encoder(plan.p3_checkpoint, plan.p3_checkpoint_sha256)
        self.history: deque[np.ndarray] = deque(maxlen=plan.history_window_steps)
        self.history_features = TemporalGeometryFeatureExtractor(plan.history_window_steps)
        self.phase_tracker = PhaseTracker()
        self.first_pass_complete = False
        self.boundary: dict[str, Any] | None = None
        self.continuity: dict[str, Any] | None = None
        self.pending: dict[str, Any] | None = None
        self.steps: list[dict[str, Any]] = []
        self.lineage_id = telemetry_sha256(
            {
                "source_id": SOURCE_ID,
                "opponent": opponent,
                "scenario": scenario_name,
                "seed": int(seed),
                "environment_episode": self.environment_episode,
            }
        )
        self.header = {
            "source_id": SOURCE_ID,
            "mode": "profile_free_observe_only_action_preserving",
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
            "training_permitted": False,
            "action_replacement_permitted": False,
        }

    def begin_step(self, *, step: int, observation: Mapping[str, Any], action: Any) -> None:
        if self.pending is not None:
            raise B6ReachabilityError("previous B6 collector step was not completed")
        base = project_base_observation(observation.get("observation_vector"), self.schema)
        values = _base_map(base)
        raw = raw_si_phase_input(observation)
        self.history.append(base.copy())
        explicit = self.history_features.update(_history_frame(base))
        ata_deg = abs(math.degrees(math.atan2(values["ata_sin"], values["ata_cos"])))
        aa_deg = abs(math.degrees(math.atan2(values["aa_sin"], values["aa_cos"])))
        dynamic_state = classify_dynamic_state(ata_deg, aa_deg)
        if dynamic_state not in STATES:
            raise B6ReachabilityError("dynamic taxonomy is unknown")
        phase = self.phase_tracker.update(raw["range_m"], raw["range_rate_mps"])
        context = _context(base, dynamic_state, phase, self.plan)
        validity = build_validity_mask(context)
        action_vector = np.asarray(action, dtype=np.float32).reshape(-1)
        action_finite = action_vector.shape == (3,) and bool(np.isfinite(action_vector).all())
        action_in_bounds = action_finite and bool(np.all(np.abs(action_vector) <= 1.0 + 1e-6))
        if not action_in_bounds:
            raise B6ReachabilityError("frozen reference action is not finite normalized 3-D VPP")
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
            "explicit_history_6d": _vector_list(explicit, 6),
            "raw_si_phase_input": raw,
            "dynamic_state": dynamic_state,
            "phase": phase,
            "ata_deg": ata_deg,
            "aa_deg": aa_deg,
            "first_pass_complete_before_step": self.first_pass_complete,
            "validity_context": {
                "geometry_state": context.geometry_state,
                "phase": context.phase,
                "energy_state": context.energy_state,
                "altitude_state": context.altitude_state,
            },
            "validity_mask": validity["mask"].astype(bool).tolist(),
            "validity_reasons": validity["reasons"],
            "normalized_vpp_action_3d": _vector_list(action_vector, 3),
            "action_sha256_before_step": telemetry_sha256(action_vector),
            "action_finite": action_finite,
            "action_in_bounds": action_in_bounds,
        }
        if ready:
            history = np.stack(tuple(self.history), axis=0)
            embedding = np.asarray(self.encoder.encode(history), dtype=np.float32)
            profile_free = np.concatenate((base, explicit, embedding)).astype(np.float32, copy=False)
            profile_66d: dict[str, dict[str, Any]] = {}
            for profile_name in PROFILE_NAMES:
                profile = compile_profile(profile_name)
                vector = self.contract.compose(
                    base_geometry=base,
                    explicit_history=explicit,
                    intent_targets=profile["target_vector"],
                    intent_weights=profile["weight_vector"],
                    temporal_embedding=embedding,
                )
                profile_66d[profile_name] = {
                    "finite": bool(np.isfinite(vector).all()),
                    "sha256": telemetry_sha256(vector),
                }
            record.update(
                {
                    "observed_history_10x16": history.astype(np.float32).tolist(),
                    "temporal_embedding_32d": _vector_list(embedding, 32),
                    "profile_free_observation_54d": _vector_list(profile_free, 54),
                    "profile_free_observation_54d_sha256": telemetry_sha256(profile_free),
                    "profile_free_observation_54d_finite": True,
                    "profiles_66d": profile_66d,
                    "all_profiles_66d_finite": all(item["finite"] for item in profile_66d.values()),
                }
            )
        self.pending = {"record": record, "action": action_vector.copy()}

    def finish_step(self, *, step: int, action: Any, info: Mapping[str, Any]) -> None:
        if self.pending is None:
            raise B6ReachabilityError("B6 collector finish_step has no pending record")
        record = self.pending["record"]
        if int(record["step"]) != int(step):
            raise B6ReachabilityError("B6 collector step alignment drifted")
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
            and record.get("profile_free_observation_54d_finite", False)
            and record.get("all_profiles_66d_finite", False)
            and record["action_finite"]
            and record["action_in_bounds"]
            and record["action_identity_preserved"]
            and record["strict_jsbsim_lineage"]
            and not record["reset_occurred"]
            and not record["future_state_injected"]
            and not record["history_padding_used"]
        )
        record["profile_free_eligible_step"] = bool(
            record["structural_contract_valid"]
            and record["first_pass_complete_before_step"]
            and record["prediction_valid"]
            and not record["prediction_fallback"]
            and record["phase"] in PHASES
        )
        if record["first_pass_complete_after_step"] and self.boundary is None:
            self.boundary = {
                "step": int(step),
                "lineage_id": self.lineage_id,
                "environment_episode": self.environment_episode,
                "raw_si_phase_input_sha256": telemetry_sha256(record["raw_si_phase_input"]),
                "action_sha256": record["action_sha256_after_step"],
            }
        elif self.boundary is not None and self.continuity is None and int(step) == int(self.boundary["step"]) + 1:
            self.continuity = {
                "passed": True,
                "boundary_step": int(self.boundary["step"]),
                "sampler_step": int(step),
                "same_lineage": record["lineage_id"] == self.boundary["lineage_id"],
                "same_environment_episode": record["environment_episode"] == self.boundary["environment_episode"],
                "no_reset": not record["reset_occurred"],
                "no_future_state_injection": not record["future_state_injected"],
            }
            self.continuity["passed"] = all(
                value for key, value in self.continuity.items() if key != "passed" and isinstance(value, bool)
            )
        self.first_pass_complete = bool(record["first_pass_complete_after_step"])
        self.steps.append(record)
        self.pending = None

    def ledger(self) -> dict[str, Any]:
        if self.pending is not None:
            raise B6ReachabilityError("B6 ledger requested with an unfinished step")
        ready = [record for record in self.steps if record["contract_ready"]]
        eligible = [record for record in self.steps if record.get("profile_free_eligible_step", False)]
        return {
            "header": dict(self.header),
            "summary": {
                "recorded_steps": len(self.steps),
                "contract_ready_steps": len(ready),
                "profile_free_eligible_steps": len(eligible),
                "first_pass_observed": self.boundary is not None,
                "continuity_valid": bool(self.continuity and self.continuity["passed"]),
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
            "first_pass_boundary": self.boundary,
            "continuity_receipt": self.continuity,
            "steps": self.steps,
        }


def evaluate_candidate_atlas(ledgers: Sequence[Mapping[str, Any]], plan: B6ReachabilityPlan) -> dict[str, Any]:
    """Apply profile-free state/phase coverage gates separately per opponent."""

    buckets: dict[tuple[str, str, str], dict[str, Any]] = {}
    episodes: dict[str, int] = defaultdict(int)
    for ledger in ledgers:
        header = _mapping(ledger.get("header"), "ledger.header")
        summary = _mapping(ledger.get("summary"), "ledger.summary")
        opponent = str(header.get("opponent", ""))
        if opponent not in plan.opponents:
            raise B6ReachabilityError("ledger contains an unexpected opponent")
        episodes[opponent] += 1
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
        if not episode_contract:
            continue
        for step in ledger.get("steps", []):
            if not isinstance(step, Mapping) or not step.get("profile_free_eligible_step", False):
                continue
            state = str(step.get("dynamic_state", ""))
            phase = str(step.get("phase", ""))
            if state not in STATES or phase not in PHASES:
                continue
            key = (opponent, state, phase)
            bucket = buckets.setdefault(
                key,
                {"steps": 0, "episodes": set(), "signatures": set(), "mirrors": set()},
            )
            bucket["steps"] += 1
            bucket["episodes"].add(str(header.get("scenario")))
            bucket["signatures"].add(str(header.get("scenario_signature")))
            bucket["mirrors"].add(str(header.get("mirror_sign")))
    matrix: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for state in STATES:
        for phase in PHASES:
            per_opponent: dict[str, dict[str, int]] = {}
            for opponent in plan.opponents:
                bucket = buckets.get((opponent, state, phase))
                item = {
                    "eligible_steps": int(bucket["steps"]) if bucket else 0,
                    "qualifying_episodes": len(bucket["episodes"]) if bucket else 0,
                    "distinct_signatures": len(bucket["signatures"]) if bucket else 0,
                    "distinct_mirrors": len(bucket["mirrors"]) if bucket else 0,
                }
                per_opponent[opponent] = item
                matrix.append({"state": state, "phase": phase, "opponent": opponent, **item})
            minimums = {key: min(item[key] for item in per_opponent.values()) for key in next(iter(per_opponent.values()))}
            candidates.append(
                {
                    "state": state,
                    "phase": phase,
                    "per_opponent": per_opponent,
                    "cross_opponent_minimums": minimums,
                    "pilot_input_candidate": (
                        minimums["eligible_steps"] >= plan.min_steps
                        and minimums["qualifying_episodes"] >= plan.min_episodes
                        and minimums["distinct_signatures"] >= plan.min_signatures
                        and minimums["distinct_mirrors"] >= plan.min_mirrors
                    ),
                }
            )
    phase_rank = {"post_merge": 1, "re_entry": 0}
    state_rank = {name: len(STATES) - index for index, name in enumerate(("neutral", "crossing_entry", "advantage", "head_on", "disadvantage"))}
    candidates.sort(
        key=lambda item: (
            item["cross_opponent_minimums"]["eligible_steps"],
            item["cross_opponent_minimums"]["qualifying_episodes"],
            item["cross_opponent_minimums"]["distinct_signatures"],
            item["cross_opponent_minimums"]["distinct_mirrors"],
            phase_rank[item["phase"]],
            state_rank[item["state"]],
        ),
        reverse=True,
    )
    selected = next((item for item in candidates if item["pilot_input_candidate"]), None)
    return {
        "source_id": SOURCE_ID,
        "pooling_prohibited": True,
        "episodes_per_opponent": {opponent: episodes[opponent] for opponent in plan.opponents},
        "matrix": matrix,
        "candidates": candidates,
        "selected_candidate": selected,
        "pilot_authorized": False,
        "verdict": "b6_pilot_input_candidate_identified" if selected else "b6_no_cross_opponent_pilot_input_candidate",
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
