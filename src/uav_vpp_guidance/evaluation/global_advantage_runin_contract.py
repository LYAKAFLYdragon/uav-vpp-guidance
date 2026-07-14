"""Fail-closed contracts for the global-advantage P1 run-in preflight."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np

from uav_vpp_guidance.evaluation.p4_v2_sampler_feasibility import (
    classify_dynamic_state,
)
from uav_vpp_guidance.hierarchy.five_state_observation_contract import (
    BASE_GEOMETRY_FEATURES,
)


SOURCE_ID = "THESIS-GLOBAL-ADVANTAGE-V1-P1-RUNIN-TELEMETRY-R2"
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
ORDER_VARIANTS = ("forward", "reverse", "mirror_interleaved")
REQUIRED_CORE_TELEMETRY = (
    "base_observation",
    "normalized_vpp_action",
    "vp_forward_bias_m",
    "vp_lateral_bias_m",
    "vpp_vertical_component",
    "nz_cmd",
    "actual_nz_g",
    "ego_attack_aoa_deg",
    "own_speed_mps",
    "own_altitude_m",
    "saturation_flag",
)


class GlobalAdvantageP1ContractError(ValueError):
    """Raised when P1 inputs or outputs violate the frozen contract."""


@dataclass(frozen=True)
class GlobalAdvantageP1Plan:
    source_id: str
    opponents: tuple[str, ...]
    order_variants: tuple[str, ...]
    scenario_count: int
    max_high_level_steps: int
    hash_decimal_places: int
    history_window_steps: int
    min_free_disk_gb: float


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GlobalAdvantageP1ContractError(f"{name} must be a mapping")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise GlobalAdvantageP1ContractError(f"{name} must be a positive integer")
    return int(value)


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GlobalAdvantageP1ContractError(f"{name} must be a positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise GlobalAdvantageP1ContractError(f"{name} must be a positive number")
    return result


def build_p1_plan(config: Mapping[str, Any]) -> GlobalAdvantageP1Plan:
    """Validate the authorised non-learning P1 preflight configuration."""

    protocol = _mapping(config.get("global_advantage_p1"), "global_advantage_p1")
    if protocol.get("source_id") != SOURCE_ID:
        raise GlobalAdvantageP1ContractError("unexpected P1 source_id")
    authorization = _mapping(protocol.get("authorization"), "authorization")
    if authorization.get("execution_permitted") is not True:
        raise GlobalAdvantageP1ContractError("P1 execution is not authorised")
    for key in (
        "training_permitted",
        "tuning_permitted",
        "policy_change_permitted",
        "vpp_change_permitted",
        "guidance_change_permitted",
        "pid_change_permitted",
        "heldout_evaluation_permitted",
        "snapshot_restore_permitted",
        "future_state_injection_permitted",
    ):
        if authorization.get(key) is not False:
            raise GlobalAdvantageP1ContractError(
                f"authorization.{key} must remain false"
            )
    run_in = _mapping(protocol.get("run_in"), "run_in")
    if run_in.get("controller") != "frozen_run_in_head_on_specialist":
        raise GlobalAdvantageP1ContractError("P1 must use the frozen run-in specialist")
    if run_in.get("backend") != "jsbsim" or run_in.get("strict_backend") is not True:
        raise GlobalAdvantageP1ContractError("P1 requires strict JSBSim")
    opponents = tuple(protocol.get("opponents", ()))
    if opponents != OPPONENTS:
        raise GlobalAdvantageP1ContractError("P1 must preserve the three-opponent order")
    repeats = _mapping(protocol.get("repeats"), "repeats")
    variants = tuple(repeats.get("order_variants", ()))
    if variants != ORDER_VARIANTS:
        raise GlobalAdvantageP1ContractError("P1 must use the three frozen order variants")
    if _positive_int(repeats.get("count"), "repeats.count") != len(ORDER_VARIANTS):
        raise GlobalAdvantageP1ContractError("repeat count must equal the order-variant count")
    telemetry = _mapping(protocol.get("telemetry"), "telemetry")
    if telemetry.get("persist_raw_steps") is not True:
        raise GlobalAdvantageP1ContractError("P1 must persist raw step telemetry")
    return GlobalAdvantageP1Plan(
        source_id=SOURCE_ID,
        opponents=opponents,
        order_variants=variants,
        scenario_count=_positive_int(
            protocol.get("expected_scenario_count"), "expected_scenario_count"
        ),
        max_high_level_steps=_positive_int(
            protocol.get("max_high_level_steps"), "max_high_level_steps"
        ),
        hash_decimal_places=_positive_int(
            telemetry.get("hash_decimal_places"), "telemetry.hash_decimal_places"
        ),
        history_window_steps=_positive_int(
            telemetry.get("history_window_steps"), "telemetry.history_window_steps"
        ),
        min_free_disk_gb=_positive_float(
            protocol.get("min_free_disk_gb"), "min_free_disk_gb"
        ),
    )


def _json_safe(value: Any, decimal_places: int) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item, decimal_places)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, decimal_places) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist(), decimal_places)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if math.isnan(number):
            return "NaN"
        if math.isinf(number):
            return "Infinity" if number > 0 else "-Infinity"
        return round(number, decimal_places)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (str, bool)) or value is None:
        return value
    return repr(value)


def stable_hash(value: Any, decimal_places: int) -> str:
    """Hash a canonical, quantised telemetry payload."""

    payload = json.dumps(
        _json_safe(value, decimal_places),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def base_observation(
    observation: Mapping[str, Any],
) -> dict[str, float]:
    """Project the immutable 16-D base state by feature name."""

    schema = _mapping(observation.get("observation_schema"), "observation_schema")
    names = tuple(str(item) for item in schema.get("feature_names", ()))
    vector = np.asarray(observation.get("observation_vector"), dtype=np.float64).reshape(-1)
    if vector.shape != (len(names),) or len(names) != len(set(names)):
        raise GlobalAdvantageP1ContractError("observation vector/schema mismatch")
    indices = {name: index for index, name in enumerate(names)}
    missing = [name for name in BASE_GEOMETRY_FEATURES if name not in indices]
    if missing:
        raise GlobalAdvantageP1ContractError(f"missing base observation features: {missing}")
    result = {name: float(vector[indices[name]]) for name in BASE_GEOMETRY_FEATURES}
    if not all(math.isfinite(value) for value in result.values()):
        raise GlobalAdvantageP1ContractError("base observation contains non-finite values")
    return result


def angle_deg(sine: float, cosine: float) -> float:
    return abs(math.degrees(math.atan2(float(sine), float(cosine))))


def classify_step(base: Mapping[str, float], phase: str) -> str:
    state = classify_dynamic_state(
        angle_deg(base["aa_sin"], base["aa_cos"]),
        angle_deg(base["ata_sin"], base["ata_cos"]),
    )
    if state == "unknown":
        raise GlobalAdvantageP1ContractError(
            f"dynamic taxonomy is unknown in phase {phase}"
        )
    return state


def telemetry_record(
    *,
    step: int,
    base: Mapping[str, float],
    phase: str,
    dynamic_state: str,
    history: Sequence[Mapping[str, float]],
    temporal_embedding: Sequence[float] | None,
    action: Sequence[float] | None,
    info: Mapping[str, Any],
    decimal_places: int,
) -> dict[str, Any]:
    """Build one persisted step record without mutating any control object."""

    own_state = info.get("own_state") if isinstance(info.get("own_state"), Mapping) else {}
    target_state = (
        info.get("target_state") if isinstance(info.get("target_state"), Mapping) else {}
    )
    action_values = None
    if action is not None:
        action_values = [float(value) for value in np.asarray(action).reshape(-1)]
        if len(action_values) != 3 or not all(math.isfinite(value) for value in action_values):
            raise GlobalAdvantageP1ContractError("run-in action is not finite 3-D")
    record = {
        "step": int(step),
        "base_observation": dict(base),
        "phase": str(phase),
        "dynamic_state": str(dynamic_state),
        "first_pass_complete": bool(info.get("first_pass_complete", False)),
        "history_length": len(history),
        "history_sha256": stable_hash(list(history), decimal_places),
        "temporal_embedding": temporal_embedding,
        "temporal_embedding_sha256": (
            stable_hash(temporal_embedding, decimal_places)
            if temporal_embedding is not None
            else None
        ),
        "proposed_skill": "run_in_head_on",
        "effective_skill": "run_in_head_on",
        "profile": None,
        "mask_reason": "reference_run_in_no_policy_mask",
        "normalized_vpp_action": action_values,
        "vp_forward_bias_m": info.get("vp_forward_bias_m"),
        "vp_lateral_bias_m": info.get("vp_lateral_bias_m"),
        "vp_vertical_bias_m": info.get("vp_vertical_bias_m"),
        "vpp_vertical_component": (
            action_values[2] if action_values is not None else None
        ),
        "guidance_command": info.get("guidance_command"),
        "nz_cmd": info.get("nz_cmd"),
        "actual_nz_g": own_state.get("nz_g", info.get("nz_g")),
        "ego_attack_aoa_deg": info.get("ego_attack_aoa_deg"),
        "own_speed_mps": base["own_speed"],
        "own_altitude_m": base["own_altitude"],
        "target_speed_mps": base["target_speed"],
        "target_altitude_m": base["target_altitude"],
        "saturation_flag": bool(info.get("saturation_flag", False)),
        "nz_saturated": bool(info.get("nz_saturated", False)),
        "roll_rate_saturated": bool(info.get("roll_rate_saturated", False)),
        "throttle_saturated": bool(info.get("throttle_saturated", False)),
        "prediction_valid": bool(info.get("prediction_valid", False)),
        "prediction_fallback": bool(info.get("prediction_fallback", False)),
        "prediction_fallback_reason": info.get("prediction_fallback_reason"),
        "prediction_fallback_phase": info.get("prediction_fallback_phase"),
        "target_in_attack_zone": bool(info.get("target_in_attack_zone", False)),
        "ego_in_attack_zone": bool(info.get("ego_in_attack_zone", False)),
        "ego_hp": info.get("ego_hp"),
        "target_hp": info.get("target_hp"),
        "terminal_reason": info.get("reason"),
        "backend": info.get("backend", "jsbsim"),
        "backend_fallback_occurred": bool(info.get("backend_fallback_occurred", False)),
        "own_state": dict(own_state),
        "target_state": dict(target_state),
    }
    record["physical_state_sha256"] = stable_hash(
        {
            "base": record["base_observation"],
            "own_state": record["own_state"],
            "target_state": record["target_state"],
        },
        decimal_places,
    )
    record["action_sha256"] = stable_hash(action_values, decimal_places)
    record["step_sha256"] = stable_hash(record, decimal_places)
    return record


def summarize_episode(
    *,
    header: Mapping[str, Any],
    reset: Mapping[str, Any],
    steps: Sequence[Mapping[str, Any]],
    terminal_reason: str,
    decimal_places: int,
) -> dict[str, Any]:
    """Produce a compact, comparable fingerprint for one frozen rollout."""

    if not steps:
        raise GlobalAdvantageP1ContractError("P1 episode has no recorded steps")
    first_pass = next(
        (record for record in steps if bool(record.get("first_pass_complete", False))),
        None,
    )
    boundary = first_pass or steps[-1]
    core_complete = all(
        all(record.get(field) is not None for field in REQUIRED_CORE_TELEMETRY)
        for record in steps
    )
    no_fallback = all(
        record.get("backend") == "jsbsim"
        and not bool(record.get("backend_fallback_occurred", False))
        and not (
            bool(record.get("prediction_fallback", False))
            and record.get("prediction_fallback_phase") != "warmup"
        )
        for record in steps
    )
    return {
        "source_id": SOURCE_ID,
        "opponent": header["opponent"],
        "scenario_signature": header["scenario_signature"],
        "scenario_name": header["scenario_name"],
        "repeat_index": header["repeat_index"],
        "order_variant": header["order_variant"],
        "reset_state_sha256": reset["physical_state_sha256"],
        "reset_observation_sha256": reset["step_sha256"],
        "trajectory_sha256": stable_hash(
            [record["step_sha256"] for record in steps], decimal_places
        ),
        "boundary_kind": "first_pass" if first_pass is not None else "terminal",
        "boundary_step": boundary["step"],
        "boundary_state_sha256": boundary["physical_state_sha256"],
        "boundary_history_sha256": boundary["history_sha256"],
        "terminal_reason": str(terminal_reason),
        "step_count": len(steps),
        "telemetry_complete": core_complete,
        "no_backend_or_prediction_fallback": no_fallback,
        "prediction_warmup_steps": sum(
            bool(record.get("prediction_fallback", False))
            and record.get("prediction_fallback_phase") == "warmup"
            for record in steps
        ),
    }


def _first_trajectory_difference(
    episodes: Sequence[Mapping[str, Any]],
) -> int | None:
    sequences = [list(item.get("step_hashes", ())) for item in episodes]
    for index, values in enumerate(zip(*sequences)):
        if len(set(values)) != 1:
            return index + 1
    if len({len(sequence) for sequence in sequences}) != 1:
        return min(len(sequence) for sequence in sequences) + 1
    return None


def compare_repeats(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare repeat fingerprints without recomputing or editing telemetry."""

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for episode in episodes:
        grouped[(str(episode["opponent"]), str(episode["scenario_signature"]))].append(
            episode
        )
    rows: list[dict[str, Any]] = []
    for (opponent, signature), group in sorted(grouped.items()):
        repeat_indices = {int(item["repeat_index"]) for item in group}
        if len(group) != len(ORDER_VARIANTS) or repeat_indices != {0, 1, 2}:
            raise GlobalAdvantageP1ContractError(
                f"{opponent}/{signature}: missing or duplicate repeat"
            )
        reset_equal = len({str(item["reset_state_sha256"]) for item in group}) == 1
        trajectory_equal = len({str(item["trajectory_sha256"]) for item in group}) == 1
        boundary_equal = len({str(item["boundary_state_sha256"]) for item in group}) == 1
        terminal_equal = len({str(item["terminal_reason"]) for item in group}) == 1
        telemetry_complete = all(bool(item["telemetry_complete"]) for item in group)
        no_fallback = all(bool(item["no_backend_or_prediction_fallback"]) for item in group)
        passed = all(
            (
                reset_equal,
                trajectory_equal,
                boundary_equal,
                terminal_equal,
                telemetry_complete,
                no_fallback,
            )
        )
        rows.append(
            {
                "opponent": opponent,
                "scenario_signature": signature,
                "repeat_count": len(group),
                "reset_state_equal": reset_equal,
                "trajectory_equal": trajectory_equal,
                "boundary_state_equal": boundary_equal,
                "terminal_equal": terminal_equal,
                "telemetry_complete": telemetry_complete,
                "no_backend_or_prediction_fallback": no_fallback,
                "first_trajectory_difference_step": (
                    _first_trajectory_difference(group)
                    if not trajectory_equal
                    else None
                ),
                "passed": passed,
            }
        )
    return {
        "source_id": SOURCE_ID,
        "comparison_rows": rows,
        "passed": bool(rows) and all(bool(row["passed"]) for row in rows),
        "failure_count": sum(not bool(row["passed"]) for row in rows),
        "total_cells": len(rows),
    }
