"""Contracts for the V5 five-state raw-SI observability collector."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


SOURCE_ID = "THESIS-FIVE-STATE-RAW-SI-PHASE-OBSERVABILITY-V5"
MANIFEST_SOURCE_ID = f"{SOURCE_ID}-MANIFEST60"
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
INITIAL_STATES = ("advantage", "head_on", "disadvantage", "neutral", "crossing_entry")


class FiveStateRawSIV5ContractError(ValueError):
    """Raised when V5 leaves its preregistered observability contract."""


@dataclass(frozen=True)
class ObservabilityPlan:
    source_id: str
    opponents: tuple[str, ...]
    initial_states: tuple[str, ...]
    max_high_level_steps: int
    min_free_disk_gb: float
    min_raw_event_episodes: int
    min_post_event_66d_steps: int
    min_mirror_signs: int
    min_height_conditions: int


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FiveStateRawSIV5ContractError(f"{name} must be a mapping")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise FiveStateRawSIV5ContractError(f"{name} must be a positive integer")
    return int(value)


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FiveStateRawSIV5ContractError(f"{name} must be a positive finite number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise FiveStateRawSIV5ContractError(f"{name} must be a positive finite number")
    return number


def build_plan(config: Mapping[str, Any]) -> ObservabilityPlan:
    """Validate V5 as a non-learning, raw-SI-only collector."""

    if config.get("source_id") != SOURCE_ID:
        raise FiveStateRawSIV5ContractError("unexpected V5 source ID")
    if "proposed_training" in config:
        raise FiveStateRawSIV5ContractError("V5 observability config must not define proposed_training")
    authorization = _mapping(config.get("authorization"), "authorization")
    authorization_keys = (
        "execution_permitted",
        "training_permitted",
        "tuning_permitted",
        "candidate_policy_loading_permitted",
        "high_level_ppo_training_permitted",
        "four_skill_training_permitted",
        "combat_finetune_permitted",
        "vpp_change_permitted",
        "guidance_change_permitted",
        "pid_change_permitted",
        "snapshot_restore",
        "future_state_injection",
        "history_padding_permitted",
        "heldout_claims_permitted",
    )
    for key in authorization_keys:
        if key not in authorization or not isinstance(authorization.get(key), bool):
            raise FiveStateRawSIV5ContractError(f"authorization.{key} must be explicit boolean")
    for key in authorization_keys:
        if key != "execution_permitted" and authorization.get(key) is not False:
            raise FiveStateRawSIV5ContractError(f"authorization.{key} must remain false")
    protocol = _mapping(config.get("observability"), "observability")
    if tuple(config.get("opponents", {}).get("order", ())) != OPPONENTS:
        raise FiveStateRawSIV5ContractError("V5 must preserve the frozen three-opponent order")
    if tuple(protocol.get("initial_state_order", ())) != INITIAL_STATES:
        raise FiveStateRawSIV5ContractError("V5 initial-state order drifted")
    raw_event = _mapping(protocol.get("raw_si_first_pass_event"), "raw_si_first_pass_event")
    if raw_event != {
        "closing_rate_threshold_mps": 0.0,
        "opening_rate_threshold_mps": 0.0,
        "minimum_closing_steps": 3,
        "minimum_opening_steps": 2,
    }:
        raise FiveStateRawSIV5ContractError("V5 raw-SI first-pass event contract drifted")
    if protocol.get("reference_controller") != "frozen_fixed_head_on":
        raise FiveStateRawSIV5ContractError("V5 must retain the frozen fixed head-on controller")
    if protocol.get("reference_profile") != "front_intercept":
        raise FiveStateRawSIV5ContractError("V5 must retain the fixed front-intercept reference profile")
    if protocol.get("profile_changes_actions") is not False or protocol.get("profile_validity_gate") is not False:
        raise FiveStateRawSIV5ContractError("V5 reference profile must not change actions or gate records")
    if protocol.get("training_reward") != "prohibited" or protocol.get("environment_reward_recording") != "prohibited":
        raise FiveStateRawSIV5ContractError("V5 must remain reward-unobserved")
    gate = _mapping(protocol.get("per_initial_state_per_opponent_gate"), "per_initial_state_per_opponent_gate")
    plan = ObservabilityPlan(
        source_id=SOURCE_ID,
        opponents=OPPONENTS,
        initial_states=INITIAL_STATES,
        max_high_level_steps=_positive_int(protocol.get("max_high_level_steps"), "max_high_level_steps"),
        min_free_disk_gb=_positive_float(protocol.get("min_free_disk_gb"), "min_free_disk_gb"),
        min_raw_event_episodes=_positive_int(gate.get("raw_event_episodes"), "raw_event_episodes"),
        min_post_event_66d_steps=_positive_int(gate.get("post_event_66d_steps"), "post_event_66d_steps"),
        min_mirror_signs=_positive_int(gate.get("distinct_mirror_signs"), "distinct_mirror_signs"),
        min_height_conditions=_positive_int(gate.get("distinct_height_conditions"), "distinct_height_conditions"),
    )
    if (
        plan.min_raw_event_episodes > 12
        or plan.min_post_event_66d_steps > plan.max_high_level_steps * 12
        or plan.min_mirror_signs > 2
        or plan.min_height_conditions > 3
    ):
        raise FiveStateRawSIV5ContractError("V5 observability gate exceeds its fixed 12-scenario cell support")
    return plan


def evaluation_pair_key(scenario: Mapping[str, Any]) -> str:
    metadata = _mapping(scenario.get("metadata"), "scenario.metadata")
    if metadata.get("identity_kind") != "evaluation":
        raise FiveStateRawSIV5ContractError("V5 scenario must use evaluation identity")
    cell, seed, serialized = metadata.get("geometry_cell_id"), metadata.get("scenario_seed"), metadata.get("pair_key")
    if not isinstance(cell, str) or not cell or isinstance(seed, bool) or not isinstance(seed, int):
        raise FiveStateRawSIV5ContractError("V5 scenario lacks geometry_cell_id/integer scenario_seed")
    expected = f"{cell}::seed={seed}"
    if serialized != expected:
        raise FiveStateRawSIV5ContractError("V5 pair key drifted")
    return expected


def validate_manifest(manifest: Mapping[str, Any]) -> list[str]:
    """Validate 5 initial states x 2 packages x 3 heights x 2 mirrors."""

    if manifest.get("source_id") != MANIFEST_SOURCE_ID:
        raise FiveStateRawSIV5ContractError("unexpected V5 manifest source")
    scenarios = list(manifest.get("scenarios") or [])
    if len(scenarios) != 60:
        raise FiveStateRawSIV5ContractError("V5 manifest must contain 60 scenarios")
    keys = [evaluation_pair_key(item) for item in scenarios]
    if len(set(keys)) != 60:
        raise FiveStateRawSIV5ContractError("V5 manifest pair keys must be unique")
    grouped: dict[str, list[Mapping[str, Any]]] = {state: [] for state in INITIAL_STATES}
    for scenario in scenarios:
        metadata = _mapping(scenario.get("metadata"), "scenario.metadata")
        state = metadata.get("initial_class")
        if state not in grouped:
            raise FiveStateRawSIV5ContractError("V5 scenario has unknown initial state")
        if metadata.get("phase_at_reset") != "pre_merge" or metadata.get("task_registry_key") != "head_on":
            raise FiveStateRawSIV5ContractError("V5 must use pre-merge frozen head-on run-in")
        expected_taxonomy_state = "transition" if state == "crossing_entry" else state
        if metadata.get("taxonomy_geometry_state") != expected_taxonomy_state:
            raise FiveStateRawSIV5ContractError("V5 initial taxonomy state drifted")
        if state == "crossing_entry" and metadata.get("crossing_context") is not True:
            raise FiveStateRawSIV5ContractError("V5 crossing-entry context drifted")
        grouped[str(state)].append(scenario)
    for state, items in grouped.items():
        if len(items) != 12:
            raise FiveStateRawSIV5ContractError(f"{state} must contain 12 scenarios")
        mirrors = {item["metadata"].get("mirror_sign") for item in items}
        heights = {item["metadata"].get("height_condition") for item in items}
        packages = {item["metadata"].get("distance_speed_package") for item in items}
        if mirrors != {"negative", "positive"} or len(heights) != 3 or len(packages) != 2:
            raise FiveStateRawSIV5ContractError(f"{state} lacks 2-package x 3-height x 2-mirror support")
    return keys


def expected_pairs_by_state(manifest: Mapping[str, Any]) -> dict[str, set[str]]:
    validate_manifest(manifest)
    result = {state: set() for state in INITIAL_STATES}
    for scenario in manifest["scenarios"]:
        result[str(scenario["metadata"]["initial_class"])].add(evaluation_pair_key(scenario))
    return result


def evaluate_observability(
    records: Sequence[Mapping[str, Any]], plan: ObservabilityPlan, expected_pairs: Mapping[str, set[str]]
) -> dict[str, Any]:
    """Apply V5 gates independently for each initial-state/opponent cell."""

    expected: set[tuple[str, str]] = set()
    seen: set[tuple[str, str]] = set()
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for state in plan.initial_states:
        keys = expected_pairs.get(state)
        if not isinstance(keys, set) or len(keys) != 12:
            raise FiveStateRawSIV5ContractError("V5 expected pair index must contain 12 keys per state")
        expected.update((opponent, pair) for opponent in plan.opponents for pair in keys)
    for record in records:
        opponent, state, pair = record.get("opponent"), record.get("initial_state"), record.get("pair_key")
        if opponent not in plan.opponents or state not in plan.initial_states or not isinstance(pair, str):
            raise FiveStateRawSIV5ContractError("V5 record identity drifted")
        metadata = _mapping(record.get("metadata"), "V5 record metadata")
        if metadata.get("initial_class") != state or evaluation_pair_key({"metadata": metadata}) != pair:
            raise FiveStateRawSIV5ContractError("V5 record metadata drifted")
        identity = (str(opponent), pair)
        if identity not in expected or identity in seen:
            raise FiveStateRawSIV5ContractError("V5 record is extra or duplicate")
        seen.add(identity)
        result = _mapping(record.get("result"), "V5 record result")
        group = groups.setdefault(
            (str(opponent), str(state)),
            {"records": 0, "raw_event_episodes": 0, "post_event_66d_steps": 0, "mirrors": set(), "heights": set(), "contract_failures": 0},
        )
        group["records"] += 1
        complete = (
            bool(result.get("telemetry_complete"))
            and bool(result.get("no_backend_fallback"))
            and bool(result.get("no_reset_after_initialization"))
            and bool(result.get("no_history_padding"))
            and bool(result.get("no_checkpoint_fallback"))
        )
        if not complete:
            group["contract_failures"] += 1
            continue
        if bool(result.get("raw_event_reached")):
            group["raw_event_episodes"] += 1
            group["post_event_66d_steps"] += int(result.get("post_event_66d_steps", 0))
            group["mirrors"].add(metadata.get("mirror_sign"))
            group["heights"].add(metadata.get("height_condition"))
    if seen != expected:
        raise FiveStateRawSIV5ContractError(
            f"V5 record universe incomplete or extra: missing={len(expected - seen)}, extra={len(seen - expected)}"
        )
    matrix: dict[str, dict[str, Any]] = {opponent: {} for opponent in plan.opponents}
    for opponent in plan.opponents:
        for state in plan.initial_states:
            group = groups.get((opponent, state), {})
            item = {
                "records": int(group.get("records", 0)),
                "raw_event_episodes": int(group.get("raw_event_episodes", 0)),
                "post_event_66d_steps": int(group.get("post_event_66d_steps", 0)),
                "distinct_mirror_signs": len(group.get("mirrors", set())),
                "distinct_height_conditions": len(group.get("heights", set())),
                "contract_failures": int(group.get("contract_failures", 0)),
            }
            item["passes"] = bool(
                item["records"] == 12
                and item["raw_event_episodes"] >= plan.min_raw_event_episodes
                and item["post_event_66d_steps"] >= plan.min_post_event_66d_steps
                and item["distinct_mirror_signs"] >= plan.min_mirror_signs
                and item["distinct_height_conditions"] >= plan.min_height_conditions
                and item["contract_failures"] == 0
            )
            matrix[opponent][state] = item
    return {
        "source_id": plan.source_id,
        "pooling_prohibited": True,
        "matrix": matrix,
        "all_initial_state_opponent_cells_pass": all(
            item["passes"] for by_state in matrix.values() for item in by_state.values()
        ),
        "training_unlocked": False,
        "verdict": "v5_five_state_observability_established"
        if all(item["passes"] for by_state in matrix.values() for item in by_state.values())
        else "v5_five_state_observability_incomplete",
    }
