"""Pure contracts for the V3 neutral/post-merge runtime-feasibility gate.

V3 deliberately separates unpaired continuous train identities from paired
evaluation identities.  The latter are the only records that may contribute
to a future paired comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


SOURCE_ID = "THESIS-NEUTRAL-POSTMERGE-REENTRY-RECOVERY-RUNTIME-FEASIBILITY-V3"
MANIFEST_SOURCE_ID = f"{SOURCE_ID}-MANIFEST48"
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
FAMILY_ORDER = (
    "midrange_balanced",
    "midrange_high_closure",
    "far_balanced",
    "far_high_closure",
)


class NeutralPostMergeV3ContractError(ValueError):
    """Raised when V3 leaves its preregistered identity or gate contract."""


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
        raise NeutralPostMergeV3ContractError(f"{label} must be a mapping")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise NeutralPostMergeV3ContractError(f"{label} must be a positive integer")
    return int(value)


def _positive_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NeutralPostMergeV3ContractError(f"{label} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise NeutralPostMergeV3ContractError(f"{label} must be a positive finite number")
    return result


def build_plan(config: Mapping[str, Any]) -> RuntimeFeasibilityPlan:
    """Validate a V3 design config or a future separately authorised overlay."""

    if config.get("source_id") != SOURCE_ID:
        raise NeutralPostMergeV3ContractError("unexpected V3 source ID")
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
        if authorization.get(key) is not False:
            raise NeutralPostMergeV3ContractError(f"authorization.{key} must remain false")
    opponents = tuple(config.get("opponents", {}).get("order", ()))
    if opponents != OPPONENTS:
        raise NeutralPostMergeV3ContractError("V3 must preserve the frozen three-opponent order")
    protocol = _mapping(config.get("runtime_feasibility"), "runtime_feasibility")
    order = tuple(protocol.get("family_order", ()))
    if order != FAMILY_ORDER:
        raise NeutralPostMergeV3ContractError("V3 family order drifted")
    gate = _mapping(protocol.get("per_family_per_opponent_gate"), "per_family_per_opponent_gate")
    return RuntimeFeasibilityPlan(
        source_id=SOURCE_ID,
        execution_permitted=bool(authorization.get("execution_permitted", False)),
        opponents=opponents,
        family_order=order,
        max_high_level_steps=_positive_int(protocol.get("max_high_level_steps"), "max_high_level_steps"),
        min_free_disk_gb=_positive_float(protocol.get("min_free_disk_gb"), "min_free_disk_gb"),
        min_qualifying_episodes=_positive_int(gate.get("qualifying_episodes"), "qualifying_episodes"),
        min_valid_target_steps=_positive_int(gate.get("valid_target_steps"), "valid_target_steps"),
        min_mirror_signs=_positive_int(gate.get("distinct_mirror_signs"), "distinct_mirror_signs"),
        min_height_conditions=_positive_int(gate.get("distinct_height_conditions"), "distinct_height_conditions"),
    )


def evaluation_pair_key(scenario: Mapping[str, Any]) -> str:
    """Return the sole key allowed to identify a paired evaluation episode."""

    metadata = _mapping(scenario.get("metadata"), "scenario.metadata")
    if metadata.get("identity_kind") != "evaluation":
        raise NeutralPostMergeV3ContractError("paired key requested for a non-evaluation scenario")
    cell, seed, serialized = (
        metadata.get("geometry_cell_id"),
        metadata.get("scenario_seed"),
        metadata.get("pair_key"),
    )
    if not isinstance(cell, str) or not cell or isinstance(seed, bool) or not isinstance(seed, int):
        raise NeutralPostMergeV3ContractError("evaluation scenario lacks geometry_cell_id/integer scenario_seed")
    expected = f"{cell}::seed={seed}"
    if serialized != expected:
        raise NeutralPostMergeV3ContractError("serialized evaluation pair_key drifted")
    return expected


def train_episode_key(scenario: Mapping[str, Any]) -> str:
    """Return an unpaired train identity that cannot enter paired deltas."""

    metadata = _mapping(scenario.get("metadata"), "scenario.metadata")
    if metadata.get("identity_kind") != "train":
        raise NeutralPostMergeV3ContractError("train key requested for a non-train scenario")
    stream, seed = metadata.get("train_stream_id"), metadata.get("scenario_seed")
    if not isinstance(stream, str) or not stream or isinstance(seed, bool) or not isinstance(seed, int):
        raise NeutralPostMergeV3ContractError("train scenario lacks train_stream_id/integer scenario_seed")
    if metadata.get("pair_key") is not None or metadata.get("geometry_cell_id") is not None:
        raise NeutralPostMergeV3ContractError("train scenarios must not carry evaluation pairing metadata")
    return f"{stream}::seed={seed}"


def episode_identity(scenario: Mapping[str, Any]) -> dict[str, Any]:
    """Make paired-delta eligibility explicit at every serializer boundary."""

    metadata = _mapping(scenario.get("metadata"), "scenario.metadata")
    kind = metadata.get("identity_kind")
    if kind == "evaluation":
        return {"kind": kind, "key": evaluation_pair_key(scenario), "paired_delta_eligible": True}
    if kind == "train":
        return {"kind": kind, "key": train_episode_key(scenario), "paired_delta_eligible": False}
    raise NeutralPostMergeV3ContractError("scenario has no recognised V3 identity kind")


def validate_feasibility_manifest(manifest: Mapping[str, Any]) -> list[str]:
    """Validate the static 48-cell evaluation envelope before JSBSim is touched."""

    if manifest.get("source_id") != MANIFEST_SOURCE_ID:
        raise NeutralPostMergeV3ContractError("unexpected V3 feasibility manifest source")
    scenarios = list(manifest.get("scenarios") or [])
    if len(scenarios) != 48:
        raise NeutralPostMergeV3ContractError("V3 feasibility manifest must contain 48 scenarios")
    keys = [evaluation_pair_key(item) for item in scenarios]
    if len(set(keys)) != len(keys):
        raise NeutralPostMergeV3ContractError("V3 feasibility pair keys are not unique")
    families: dict[str, list[Mapping[str, Any]]] = {name: [] for name in FAMILY_ORDER}
    for scenario in scenarios:
        metadata = _mapping(scenario.get("metadata"), "scenario.metadata")
        family = metadata.get("feasibility_family_id")
        if family not in families:
            raise NeutralPostMergeV3ContractError("scenario references an unknown feasibility family")
        if metadata.get("initial_class") != "neutral" or metadata.get("phase_at_reset") != "pre_merge":
            raise NeutralPostMergeV3ContractError("V3 feasibility scenarios must be neutral/pre_merge")
        if metadata.get("task_registry_key") != "head_on":
            raise NeutralPostMergeV3ContractError("V3 feasibility must match the future pilot head-on run-in engine")
        families[family].append(scenario)
    for family, items in families.items():
        if len(items) != 12:
            raise NeutralPostMergeV3ContractError(f"{family} must contain exactly 12 scenarios")
        mirrors = {item["metadata"].get("mirror_sign") for item in items}
        heights = {item["metadata"].get("height_condition") for item in items}
        packages = {item["metadata"].get("distance_speed_package") for item in items}
        if mirrors != {"negative", "positive"} or len(heights) != 3 or len(packages) != 2:
            raise NeutralPostMergeV3ContractError(f"{family} lacks the frozen 2-package x 3-height x 2-mirror support")
    return keys


def runtime_identity_index(manifest: Mapping[str, Any]) -> dict[str, set[str]]:
    """Return the exact family/pair-key universe required from a V3 run."""

    validate_feasibility_manifest(manifest)
    index: dict[str, set[str]] = {family: set() for family in FAMILY_ORDER}
    for scenario in manifest["scenarios"]:
        metadata = _mapping(scenario.get("metadata"), "scenario.metadata")
        index[str(metadata["feasibility_family_id"])].add(evaluation_pair_key(scenario))
    return index


def evaluate_family_coverage(
    records: Sequence[Mapping[str, Any]],
    plan: RuntimeFeasibilityPlan,
    expected_pairs_by_family: Mapping[str, set[str]],
) -> dict[str, Any]:
    """Apply V3 support gates separately for every family and opponent."""

    buckets: dict[tuple[str, str], dict[str, Any]] = {}
    seen: set[tuple[str, str]] = set()
    expected: set[tuple[str, str]] = set()
    for family in plan.family_order:
        keys = expected_pairs_by_family.get(family)
        if not isinstance(keys, set) or len(keys) != 12:
            raise NeutralPostMergeV3ContractError("runtime identity index must provide 12 pair keys per family")
        expected.update((opponent, key) for opponent in plan.opponents for key in keys)
    for record in records:
        opponent, family, key = (
            record.get("opponent"),
            record.get("family_id"),
            record.get("pair_key"),
        )
        if opponent not in plan.opponents or family not in plan.family_order or not isinstance(key, str):
            raise NeutralPostMergeV3ContractError("runtime record identity drifted")
        metadata = _mapping(record.get("metadata"), "runtime record metadata")
        if metadata.get("feasibility_family_id") != family:
            raise NeutralPostMergeV3ContractError("runtime record family does not match metadata")
        if evaluation_pair_key({"metadata": metadata}) != key or (str(opponent), key) not in expected:
            raise NeutralPostMergeV3ContractError("runtime record is not in the frozen V3 manifest universe")
        identity = (str(opponent), key)
        if identity in seen:
            raise NeutralPostMergeV3ContractError("duplicate opponent/pair-key runtime record")
        seen.add(identity)
        result = _mapping(record.get("result"), "runtime record result")
        bucket = buckets.setdefault(
            (str(family), str(opponent)),
            {"valid_target_steps": 0, "qualifying_episodes": 0, "mirrors": set(), "heights": set(), "records": 0, "contract_failures": 0},
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
        missing, extra = expected - seen, seen - expected
        raise NeutralPostMergeV3ContractError(
            f"V3 runtime record universe is incomplete or contains extras: missing={len(missing)}, extra={len(extra)}"
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
        "candidates": candidates,
        "selected_family": selected,
        "future_pilot_authorized": False,
        "verdict": "v3_runtime_feasibility_family_identified" if selected else "v3_no_cross_opponent_runtime_feasible_family",
    }
