"""Contracts for the V4 reward-unobserved runtime-feasibility record."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


SOURCE_ID = "THESIS-NEUTRAL-POSTMERGE-REENTRY-RECOVERY-RUNTIME-FEASIBILITY-V4"
MANIFEST_SOURCE_ID = f"{SOURCE_ID}-MANIFEST48"
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
FAMILY_ORDER = (
    "close_balanced",
    "close_high_closure",
    "extended_balanced",
    "extended_high_closure",
)


class NeutralPostMergeV4ContractError(ValueError):
    """Raised when V4 drifts from its preregistered record-only contract."""


@dataclass(frozen=True)
class RuntimeFeasibilityPlan:
    source_id: str
    execution_permitted: bool
    opponents: tuple[str, ...]
    family_order: tuple[str, ...]
    max_high_level_steps: int
    min_free_disk_gb: float
    min_qualifying_episodes: int
    min_valid_target_steps: int
    min_mirror_signs: int
    min_height_conditions: int


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NeutralPostMergeV4ContractError(f"{label} must be a mapping")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise NeutralPostMergeV4ContractError(f"{label} must be a positive integer")
    return int(value)


def _positive_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NeutralPostMergeV4ContractError(f"{label} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise NeutralPostMergeV4ContractError(f"{label} must be a positive finite number")
    return result


def build_plan(config: Mapping[str, Any]) -> RuntimeFeasibilityPlan:
    """Validate V4 without accepting a learning or reward contract."""

    if config.get("source_id") != SOURCE_ID:
        raise NeutralPostMergeV4ContractError("unexpected V4 source ID")
    if "proposed_training" in config:
        raise NeutralPostMergeV4ContractError("V4 record-only config must not define proposed_training")
    authorization = _mapping(config.get("authorization"), "authorization")
    for key in (
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
    ):
        if key not in authorization or bool(authorization.get(key)):
            raise NeutralPostMergeV4ContractError(f"authorization.{key} must remain false")
    opponents = tuple(config.get("opponents", {}).get("order", ()))
    if opponents != OPPONENTS:
        raise NeutralPostMergeV4ContractError("V4 must preserve the frozen three-opponent order")
    protocol = _mapping(config.get("runtime_feasibility"), "runtime_feasibility")
    family_order = tuple(protocol.get("family_order", ()))
    if family_order != FAMILY_ORDER:
        raise NeutralPostMergeV4ContractError("V4 family order drifted")
    if protocol.get("training_reward") != "prohibited" or protocol.get("environment_reward_recording") != "prohibited":
        raise NeutralPostMergeV4ContractError("V4 must prohibit training-reward use and environment-reward recording")
    runtime_contract = _mapping(config.get("contract"), "contract")
    if runtime_contract.get("training_reward") != "prohibited" or runtime_contract.get("environment_reward_recording") != "prohibited":
        raise NeutralPostMergeV4ContractError("V4 runtime contract must prohibit reward use and recording")
    gate = _mapping(protocol.get("per_family_per_opponent_gate"), "per_family_per_opponent_gate")
    return RuntimeFeasibilityPlan(
        source_id=SOURCE_ID,
        execution_permitted=bool(authorization.get("execution_permitted", False)),
        opponents=opponents,
        family_order=family_order,
        max_high_level_steps=_positive_int(protocol.get("max_high_level_steps"), "max_high_level_steps"),
        min_free_disk_gb=_positive_float(protocol.get("min_free_disk_gb"), "min_free_disk_gb"),
        min_qualifying_episodes=_positive_int(gate.get("qualifying_episodes"), "qualifying_episodes"),
        min_valid_target_steps=_positive_int(gate.get("valid_target_steps"), "valid_target_steps"),
        min_mirror_signs=_positive_int(gate.get("distinct_mirror_signs"), "distinct_mirror_signs"),
        min_height_conditions=_positive_int(gate.get("distinct_height_conditions"), "distinct_height_conditions"),
    )


def evaluation_pair_key(scenario: Mapping[str, Any]) -> str:
    """Return the sole evaluation identity permitted in a V4 record."""

    metadata = _mapping(scenario.get("metadata"), "scenario.metadata")
    if metadata.get("identity_kind") != "evaluation":
        raise NeutralPostMergeV4ContractError("V4 runtime requires evaluation identity")
    cell, seed, serialized = (
        metadata.get("geometry_cell_id"),
        metadata.get("scenario_seed"),
        metadata.get("pair_key"),
    )
    if not isinstance(cell, str) or not cell or isinstance(seed, bool) or not isinstance(seed, int):
        raise NeutralPostMergeV4ContractError("evaluation scenario lacks geometry_cell_id/integer scenario_seed")
    expected = f"{cell}::seed={seed}"
    if serialized != expected:
        raise NeutralPostMergeV4ContractError("serialized evaluation pair_key drifted")
    return expected


def validate_feasibility_manifest(manifest: Mapping[str, Any]) -> list[str]:
    """Validate the V4 48-cell runtime-only envelope before JSBSim is touched."""

    if manifest.get("source_id") != MANIFEST_SOURCE_ID:
        raise NeutralPostMergeV4ContractError("unexpected V4 feasibility manifest source")
    scenarios = list(manifest.get("scenarios") or [])
    if len(scenarios) != 48:
        raise NeutralPostMergeV4ContractError("V4 feasibility manifest must contain 48 scenarios")
    pair_keys = [evaluation_pair_key(item) for item in scenarios]
    if len(set(pair_keys)) != len(pair_keys):
        raise NeutralPostMergeV4ContractError("V4 feasibility pair keys are not unique")
    by_family: dict[str, list[Mapping[str, Any]]] = {name: [] for name in FAMILY_ORDER}
    for scenario in scenarios:
        metadata = _mapping(scenario.get("metadata"), "scenario.metadata")
        family = metadata.get("feasibility_family_id")
        if family not in by_family:
            raise NeutralPostMergeV4ContractError("scenario references an unknown feasibility family")
        if metadata.get("initial_class") != "neutral" or metadata.get("phase_at_reset") != "pre_merge":
            raise NeutralPostMergeV4ContractError("V4 scenarios must start neutral/pre_merge")
        if metadata.get("task_registry_key") != "head_on":
            raise NeutralPostMergeV4ContractError("V4 must use the frozen head-on run-in engine")
        by_family[str(family)].append(scenario)
    for family, items in by_family.items():
        if len(items) != 12:
            raise NeutralPostMergeV4ContractError(f"{family} must contain exactly 12 scenarios")
        mirrors = {item["metadata"].get("mirror_sign") for item in items}
        heights = {item["metadata"].get("height_condition") for item in items}
        packages = {item["metadata"].get("distance_speed_package") for item in items}
        if mirrors != {"negative", "positive"} or len(heights) != 3 or len(packages) != 2:
            raise NeutralPostMergeV4ContractError(f"{family} lacks 2-package x 3-height x 2-mirror support")
    return pair_keys


def runtime_identity_index(manifest: Mapping[str, Any]) -> dict[str, set[str]]:
    """Return the exact family/pair-key universe required from a V4 run."""

    validate_feasibility_manifest(manifest)
    result: dict[str, set[str]] = {family: set() for family in FAMILY_ORDER}
    for scenario in manifest["scenarios"]:
        metadata = _mapping(scenario.get("metadata"), "scenario.metadata")
        result[str(metadata["feasibility_family_id"])].add(evaluation_pair_key(scenario))
    return result


def evaluate_family_coverage(
    records: Sequence[Mapping[str, Any]],
    plan: RuntimeFeasibilityPlan,
    expected_pairs_by_family: Mapping[str, set[str]],
) -> dict[str, Any]:
    """Apply V4 physical-support gates separately for every family/opponent."""

    expected: set[tuple[str, str]] = set()
    seen: set[tuple[str, str]] = set()
    buckets: dict[tuple[str, str], dict[str, Any]] = {}
    for family in plan.family_order:
        keys = expected_pairs_by_family.get(family)
        if not isinstance(keys, set) or len(keys) != 12:
            raise NeutralPostMergeV4ContractError("runtime identity index must provide 12 pair keys per family")
        expected.update((opponent, key) for opponent in plan.opponents for key in keys)
    for record in records:
        opponent, family, key = record.get("opponent"), record.get("family_id"), record.get("pair_key")
        if opponent not in plan.opponents or family not in plan.family_order or not isinstance(key, str):
            raise NeutralPostMergeV4ContractError("runtime record identity drifted")
        metadata = _mapping(record.get("metadata"), "runtime record metadata")
        if metadata.get("feasibility_family_id") != family or evaluation_pair_key({"metadata": metadata}) != key:
            raise NeutralPostMergeV4ContractError("runtime record metadata drifted")
        identity = (str(opponent), key)
        if identity not in expected or identity in seen:
            raise NeutralPostMergeV4ContractError("runtime record is extra or duplicate")
        seen.add(identity)
        result = _mapping(record.get("result"), "runtime record result")
        bucket = buckets.setdefault(
            (str(family), str(opponent)),
            {"records": 0, "qualifying_episodes": 0, "valid_target_steps": 0, "mirrors": set(), "heights": set(), "contract_failures": 0},
        )
        bucket["records"] += 1
        valid = bool(result.get("handoff_reached")) and bool(result.get("qualifying")) and bool(result.get("telemetry_complete"))
        if not valid:
            bucket["contract_failures"] += int(not bool(result.get("telemetry_complete")))
            continue
        bucket["qualifying_episodes"] += 1
        bucket["valid_target_steps"] += int(result.get("valid_target_steps", 0))
        bucket["mirrors"].add(metadata.get("mirror_sign"))
        bucket["heights"].add(metadata.get("height_condition"))
    if seen != expected:
        raise NeutralPostMergeV4ContractError(
            f"V4 runtime record universe is incomplete or contains extras: missing={len(expected - seen)}, extra={len(seen - expected)}"
        )
    candidates: list[dict[str, Any]] = []
    for family in plan.family_order:
        per_opponent: dict[str, dict[str, Any]] = {}
        for opponent in plan.opponents:
            bucket = buckets.get((family, opponent), {})
            item = {
                "records": int(bucket.get("records", 0)),
                "qualifying_episodes": int(bucket.get("qualifying_episodes", 0)),
                "valid_target_steps": int(bucket.get("valid_target_steps", 0)),
                "distinct_mirror_signs": len(bucket.get("mirrors", set())),
                "distinct_height_conditions": len(bucket.get("heights", set())),
                "contract_failures": int(bucket.get("contract_failures", 0)),
            }
            item["passes"] = bool(
                item["records"] == 12
                and item["qualifying_episodes"] >= plan.min_qualifying_episodes
                and item["valid_target_steps"] >= plan.min_valid_target_steps
                and item["distinct_mirror_signs"] >= plan.min_mirror_signs
                and item["distinct_height_conditions"] >= plan.min_height_conditions
                and item["contract_failures"] == 0
            )
            per_opponent[opponent] = item
        minimums = {
            field: min(int(item[field]) for item in per_opponent.values())
            for field in ("qualifying_episodes", "valid_target_steps", "distinct_mirror_signs", "distinct_height_conditions")
        }
        candidates.append(
            {
                "family_id": family,
                "per_opponent": per_opponent,
                "cross_opponent_minimums": minimums,
                "runtime_feasibility_pass": all(bool(item["passes"]) for item in per_opponent.values()),
            }
        )
    rank = {name: len(plan.family_order) - index for index, name in enumerate(plan.family_order)}
    candidates.sort(
        key=lambda item: (
            item["cross_opponent_minimums"]["valid_target_steps"],
            item["cross_opponent_minimums"]["qualifying_episodes"],
            item["cross_opponent_minimums"]["distinct_height_conditions"],
            item["cross_opponent_minimums"]["distinct_mirror_signs"],
            rank[item["family_id"]],
        ),
        reverse=True,
    )
    selected = next((item for item in candidates if item["runtime_feasibility_pass"]), None)
    return {
        "source_id": plan.source_id,
        "pooling_prohibited": True,
        "training_reward": "prohibited",
        "environment_reward_recording": "prohibited",
        "candidates": candidates,
        "selected_family": selected,
        "future_pilot_authorized": False,
        "verdict": "v4_runtime_feasibility_family_identified" if selected else "v4_no_cross_opponent_runtime_feasible_family",
    }
