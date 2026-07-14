"""Fail-closed contracts for the P1 R3 fresh-process reproducibility probe."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


SOURCE_ID = "THESIS-GLOBAL-ADVANTAGE-V1-P1-R3-FRESH-ENV-REPRO-V1"
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
ORDER_VARIANTS = ("forward", "reverse", "mirror_interleaved")

# This is a versioned, explicit JSBSim state view. It deliberately contains the
# physical states and actuator properties that can affect the next integration step.
FDM_PROPERTY_NAMES = (
    "position/long-gc-deg",
    "position/lat-geod-deg",
    "position/h-sl-ft",
    "attitude/roll-rad",
    "attitude/pitch-rad",
    "attitude/heading-true-rad",
    "velocities/v-north-fps",
    "velocities/v-east-fps",
    "velocities/v-down-fps",
    "velocities/vt-fps",
    "velocities/u-fps",
    "velocities/v-fps",
    "velocities/w-fps",
    "velocities/p-rad_sec",
    "velocities/q-rad_sec",
    "velocities/r-rad_sec",
    "accelerations/Nz",
    "accelerations/a-z-ft_sec2",
    "fcs/elevator-cmd-norm",
    "fcs/aileron-cmd-norm",
    "fcs/rudder-cmd-norm",
    "fcs/throttle-cmd-norm",
)

_STATIC_OBJECT_FIELDS = {
    "config",
    "agent",
    "actuator",
    "predictor",
    "_fallback_predictor",
    "_checkpoint",
    "checkpoint_path",
    "config_path",
    "device",
    "expected_feature_names",
    "obs_dim",
    "action_dim",
    "action_mode",
}

_ENV_RUNTIME_FIELDS = (
    "current_step",
    "_episode_count",
    "_sim_time_s",
    "_last_aggressiveness",
    "_last_pid_gain_deltas",
    "_prev_delayed_command",
    "_prediction_buffer_synced_for_action",
    "_last_virtual_point",
    "_last_observation_schema",
    "_last_command_saturation",
    "_merge_seen_close_range",
    "_close_range_anchor_ready",
    "_first_pass_complete",
    "_first_pass_completion_step",
    "_mode_switch_latched",
    "_runtime_specialist_key",
    "_runtime_specialist_profile",
    "_runtime_specialist_mode_name",
    "_runtime_specialist_reason",
    "_post_merge_anchor_mode_released",
    "_post_merge_anchor_mode_has_activated",
    "_post_merge_anchor_mode_ego_only_streak_steps",
    "_post_merge_anchor_mode_lateral_world_offset_latched",
    "_post_merge_anchor_mode_lateral_world_offset_latch_activation_range_m",
    "_post_merge_predicted_target_blend_released",
    "_post_merge_offensive_anchor_blend_released",
    "_post_merge_offensive_anchor_blend_release_step",
    "_post_merge_offensive_anchor_blend_has_activated",
    "_post_merge_offensive_anchor_blend_ego_only_streak_steps",
    "_post_merge_offensive_anchor_lateral_world_offset_latched",
    "_post_merge_predicted_target_forward_scale_released",
    "_post_merge_predicted_target_forward_scale_ego_only_streak_steps",
    "_post_merge_tactical_basis_recovery_profile_was_active",
    "_post_merge_tactical_basis_recovery_profile_entry_lateral_sign",
    "_post_merge_tactical_basis_recovery_profile_entry_window_steps_remaining",
)

_RESET_SENTINEL_FIELDS = {
    "_merge_min_range_so_far_m",
    "_post_merge_tactical_basis_recovery_profile_previous_vp_lateral_bias_m",
    "_post_merge_tactical_basis_recovery_profile_previous_vp_forward_bias_m",
    "_post_merge_tactical_basis_recovery_profile_previous_altitude_m",
}


class GlobalAdvantageP1R3ContractError(ValueError):
    """Raised when R3 cannot establish a complete reproducibility contract."""


@dataclass(frozen=True)
class GlobalAdvantageP1R3Plan:
    source_id: str
    execution_permitted: bool
    opponents: tuple[str, ...]
    order_variants: tuple[str, ...]
    scenario_count: int
    max_high_level_steps: int
    hash_decimal_places: int
    history_window_steps: int
    child_timeout_seconds: int
    min_free_disk_gb: float


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GlobalAdvantageP1R3ContractError(f"{name} must be a mapping")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise GlobalAdvantageP1R3ContractError(f"{name} must be a positive integer")
    return int(value)


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GlobalAdvantageP1R3ContractError(f"{name} must be a positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise GlobalAdvantageP1R3ContractError(f"{name} must be a positive number")
    return result


def build_p1_r3_plan(config: Mapping[str, Any]) -> GlobalAdvantageP1R3Plan:
    """Validate the R3 plan without authorising training or evaluation."""

    protocol = _mapping(config.get("global_advantage_p1_r3"), "global_advantage_p1_r3")
    if config.get("source_id") != SOURCE_ID or protocol.get("source_id") != SOURCE_ID:
        raise GlobalAdvantageP1R3ContractError("unexpected R3 source_id")
    authorization = _mapping(protocol.get("authorization"), "authorization")
    execution_permitted = authorization.get("execution_permitted")
    if not isinstance(execution_permitted, bool):
        raise GlobalAdvantageP1R3ContractError("execution_permitted must be explicit")
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
            raise GlobalAdvantageP1R3ContractError(
                f"authorization.{key} must remain false"
            )
    run_in = _mapping(protocol.get("run_in"), "run_in")
    if run_in.get("controller") != "frozen_run_in_head_on_specialist":
        raise GlobalAdvantageP1R3ContractError("R3 must use the frozen run-in specialist")
    if run_in.get("backend") != "jsbsim" or run_in.get("strict_backend") is not True:
        raise GlobalAdvantageP1R3ContractError("R3 requires strict JSBSim")
    isolation = _mapping(protocol.get("isolation"), "isolation")
    if isolation.get("fresh_process_per_episode") is not True:
        raise GlobalAdvantageP1R3ContractError("R3 requires one fresh process per episode")
    if isolation.get("fresh_environment_per_episode") is not True:
        raise GlobalAdvantageP1R3ContractError("R3 requires one fresh environment per episode")
    opponents = tuple(protocol.get("opponents", ()))
    if opponents != OPPONENTS:
        raise GlobalAdvantageP1R3ContractError("R3 must preserve the three-opponent order")
    repeats = _mapping(protocol.get("repeats"), "repeats")
    variants = tuple(repeats.get("order_variants", ()))
    if variants != ORDER_VARIANTS:
        raise GlobalAdvantageP1R3ContractError("R3 must use the three frozen order variants")
    if _positive_int(repeats.get("count"), "repeats.count") != len(ORDER_VARIANTS):
        raise GlobalAdvantageP1R3ContractError("repeat count must equal the order-variant count")
    telemetry = _mapping(protocol.get("telemetry"), "telemetry")
    if telemetry.get("persist_raw_steps") is not True:
        raise GlobalAdvantageP1R3ContractError("R3 must persist raw step telemetry")
    if telemetry.get("persist_full_reset_envelope") is not True:
        raise GlobalAdvantageP1R3ContractError("R3 must persist the full reset envelope")
    child_timeout_seconds = _positive_int(
        protocol.get("child_timeout_seconds"), "child_timeout_seconds"
    )
    return GlobalAdvantageP1R3Plan(
        source_id=SOURCE_ID,
        execution_permitted=execution_permitted,
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
        child_timeout_seconds=child_timeout_seconds,
        min_free_disk_gb=_positive_float(
            protocol.get("min_free_disk_gb"), "min_free_disk_gb"
        ),
    )


def canonicalize(value: Any, *, path: str = "root") -> Any:
    """Return a finite JSON-safe value or fail instead of hiding state drift."""

    if isinstance(value, Mapping):
        return {
            str(key): canonicalize(item, path=f"{path}.{key}")
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple, deque)):
        return [canonicalize(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, np.ndarray):
        return canonicalize(value.tolist(), path=path)
    if isinstance(value, np.generic):
        return canonicalize(value.item(), path=path)
    if isinstance(value, np.random.Generator):
        return {
            "bit_generator": value.bit_generator.__class__.__name__,
            "state": canonicalize(value.bit_generator.state, path=f"{path}.state"),
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise GlobalAdvantageP1R3ContractError(f"non-finite value at {path}")
        return float(value)
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        return canonicalize(value.detach().cpu().numpy(), path=path)
    if hasattr(value, "__dict__"):
        fields = {
            key: item
            for key, item in vars(value).items()
            if key not in _STATIC_OBJECT_FIELDS
        }
        return {
            "class": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
            "fields": canonicalize(fields, path=f"{path}.__dict__"),
        }
    raise GlobalAdvantageP1R3ContractError(
        f"unsupported runtime value at {path}: {value.__class__.__module__}.{value.__class__.__qualname__}"
    )


def snapshot_hash(payload: Mapping[str, Any], decimal_places: int) -> str:
    """Hash a complete, finite snapshot with fixed decimal precision."""

    canonical = canonicalize(payload)

    def quantize(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): quantize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [quantize(item) for item in value]
        if isinstance(value, float):
            return round(value, decimal_places)
        return value

    encoded = json.dumps(
        quantize(canonical), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _component_state(component: Any, *, name: str) -> Mapping[str, Any] | None:
    if component is None:
        return None
    try:
        raw_fields = vars(component)
    except TypeError as error:
        raise GlobalAdvantageP1R3ContractError(
            f"cannot inspect runtime component {name}"
        ) from error
    fields = {
        key: value
        for key, value in raw_fields.items()
        if key not in _STATIC_OBJECT_FIELDS
    }
    return {
        "class": f"{component.__class__.__module__}.{component.__class__.__qualname__}",
        "fields": canonicalize(fields, path=name),
    }


def _observation_snapshot(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    schema = _mapping(observation.get("observation_schema"), "observation_schema")
    names = tuple(str(name) for name in schema.get("feature_names", ()))
    vector = np.asarray(observation.get("observation_vector"), dtype=np.float64).reshape(-1)
    if not names or vector.shape != (len(names),) or len(names) != len(set(names)):
        raise GlobalAdvantageP1R3ContractError("full observation/schema contract mismatch")
    if not np.isfinite(vector).all():
        raise GlobalAdvantageP1R3ContractError("full observation contains non-finite values")
    return {
        "feature_names": list(names),
        "observation_vector": [float(item) for item in vector],
        "observation_schema": canonicalize(schema, path="observation_schema"),
    }


def _sentinel_snapshot(value: Any, *, field: str) -> Mapping[str, Any]:
    numeric = float(value)
    if math.isnan(numeric):
        return {"field": field, "state": "nan_unset_sentinel"}
    if math.isinf(numeric):
        return {
            "field": field,
            "state": "positive_inf_unset_sentinel" if numeric > 0 else "negative_inf_sentinel",
        }
    return {"field": field, "state": "finite", "value": numeric}


def _environment_runtime_state(env: Any) -> Mapping[str, Any]:
    values: dict[str, Any] = {}
    for field in _ENV_RUNTIME_FIELDS:
        if not hasattr(env, field):
            raise GlobalAdvantageP1R3ContractError(
                f"environment is missing required runtime field {field}"
            )
        values[field] = getattr(env, field)
    sentinels: dict[str, Any] = {}
    for field in _RESET_SENTINEL_FIELDS:
        if not hasattr(env, field):
            raise GlobalAdvantageP1R3ContractError(
                f"environment is missing required reset sentinel {field}"
            )
        sentinels[field] = _sentinel_snapshot(getattr(env, field), field=field)
    for rng_name in ("_robustness_rng", "_prediction_noise_rng", "_domain_rand_rng"):
        if not hasattr(env, rng_name):
            raise GlobalAdvantageP1R3ContractError(
                f"environment is missing required RNG {rng_name}"
            )
        values[rng_name] = getattr(env, rng_name)
    return {"runtime": canonicalize(values, path="environment"), "sentinels": sentinels}


def _fdm_snapshot(env: Any) -> Mapping[str, Any]:
    jsbsim_env = getattr(env, "jsbsim_env", None)
    aircraft = getattr(jsbsim_env, "_aircraft", None)
    if not isinstance(aircraft, Mapping) or not aircraft:
        raise GlobalAdvantageP1R3ContractError("R3 requires initialized JSBSim aircraft")
    snapshots: dict[str, Any] = {}
    for uid, vehicle in sorted(aircraft.items(), key=lambda pair: str(pair[0])):
        if not hasattr(vehicle, "get_property_value") or not hasattr(vehicle, "get_state"):
            raise GlobalAdvantageP1R3ContractError(f"invalid JSBSim aircraft interface: {uid}")
        properties: dict[str, float] = {}
        for property_name in FDM_PROPERTY_NAMES:
            try:
                value = float(vehicle.get_property_value(property_name))
            except Exception as error:
                raise GlobalAdvantageP1R3ContractError(
                    f"cannot read required JSBSim property {uid}:{property_name}"
                ) from error
            if not math.isfinite(value):
                raise GlobalAdvantageP1R3ContractError(
                    f"non-finite JSBSim property {uid}:{property_name}"
                )
            properties[property_name] = value
        snapshots[str(uid)] = {
            "sim_time_s": float(vehicle.jsbsim_exec.get_sim_time()),
            "properties": properties,
            "derived_state": canonicalize(vehicle.get_state(), path=f"fdm.{uid}.derived_state"),
        }
    return snapshots


def capture_runtime_envelope(
    *,
    env: Any,
    observation: Mapping[str, Any],
    history: Sequence[Mapping[str, float]],
    phase_tracker: Any,
    reference_metadata: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Capture every mutable input that can affect the next high-level step."""

    if getattr(env, "_backend", None) != "jsbsim":
        raise GlobalAdvantageP1R3ContractError("R3 backend is not JSBSim")
    if getattr(env, "trajectory_predictor_adapter", None) is None:
        raise GlobalAdvantageP1R3ContractError("R3 requires an initialized predictor adapter")
    components = {
        "guidance": _component_state(getattr(env, "guidance", None), name="guidance"),
        "target_guidance": _component_state(
            getattr(env, "_target_guidance", None), name="target_guidance"
        ),
        "guidance_pn": _component_state(
            getattr(env, "_guidance_pn", None), name="guidance_pn"
        ),
        "command_post_processor": _component_state(
            getattr(env, "command_post_processor", None), name="command_post_processor"
        ),
        "current_gains": _component_state(
            getattr(env, "current_gains", None), name="current_gains"
        ),
        "command_filter": _component_state(
            getattr(env, "_command_filter", None), name="command_filter"
        ),
        "target_command_filter": _component_state(
            getattr(env, "_target_command_filter", None), name="target_command_filter"
        ),
        "low_level_controller": _component_state(
            getattr(env, "_low_level_controller", None), name="low_level_controller"
        ),
        "target_low_level_controller": _component_state(
            getattr(env, "_target_low_level_controller", None),
            name="target_low_level_controller",
        ),
        "trajectory_predictor_adapter": _component_state(
            getattr(env, "trajectory_predictor_adapter", None),
            name="trajectory_predictor_adapter",
        ),
        "prediction_error_tracker": _component_state(
            getattr(env, "_prediction_error_tracker", None),
            name="prediction_error_tracker",
        ),
        "virtual_point_generator": _component_state(
            getattr(env, "virtual_point_generator", None), name="virtual_point_generator"
        ),
        "reward_calculator": _component_state(
            getattr(env, "reward_calculator", None), name="reward_calculator"
        ),
        "termination_checker": _component_state(
            getattr(env, "termination_checker", None), name="termination_checker"
        ),
        "combat_hp": _component_state(getattr(env, "combat_hp", None), name="combat_hp"),
        "opponent_policy": _component_state(
            getattr(env, "opponent_policy", None), name="opponent_policy"
        ),
    }
    envelope = {
        "snapshot_version": 1,
        "observation": _observation_snapshot(observation),
        "history": canonicalize(history, path="history"),
        "phase_tracker": _component_state(phase_tracker, name="phase_tracker"),
        "environment": _environment_runtime_state(env),
        "components": components,
        "fdm": _fdm_snapshot(env),
        "reference": canonicalize(reference_metadata, path="reference"),
    }
    return canonicalize(envelope, path="runtime_envelope")


def finite_action(value: Any, *, name: str) -> list[float]:
    action = np.asarray(value, dtype=np.float64).reshape(-1)
    if action.shape != (3,) or not np.isfinite(action).all():
        raise GlobalAdvantageP1R3ContractError(f"{name} must be a finite 3-D action")
    return [float(item) for item in action]


def compare_r3_repeats(episodes: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Apply the preregistered all-fields equivalence gate to R3 summaries."""

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for episode in episodes:
        grouped[(str(episode["opponent"]), str(episode["scenario_signature"]))].append(
            episode
        )
    rows: list[dict[str, Any]] = []
    required_hashes = (
        "reset_envelope_sha256",
        "first_reference_action_sha256",
        "first_opponent_action_sha256",
        "trajectory_sha256",
        "boundary_envelope_sha256",
        "terminal_reason",
    )
    for (opponent, signature), group in sorted(grouped.items()):
        repeat_indices = {int(item["repeat_index"]) for item in group}
        if len(group) != len(ORDER_VARIANTS) or repeat_indices != {0, 1, 2}:
            raise GlobalAdvantageP1R3ContractError(
                f"{opponent}/{signature}: missing or duplicate repeat"
            )
        equal = {
            field: len({str(item[field]) for item in group}) == 1
            for field in required_hashes
        }
        fresh_processes = len({int(item["process_id"]) for item in group}) == len(group)
        telemetry_complete = all(bool(item["telemetry_complete"]) for item in group)
        fallback_free = all(bool(item["no_backend_or_prediction_fallback"]) for item in group)
        passed = all(equal.values()) and fresh_processes and telemetry_complete and fallback_free
        rows.append(
            {
                "opponent": opponent,
                "scenario_signature": signature,
                "repeat_count": len(group),
                **{f"{field}_equal": value for field, value in equal.items()},
                "fresh_processes": fresh_processes,
                "telemetry_complete": telemetry_complete,
                "no_backend_or_prediction_fallback": fallback_free,
                "passed": passed,
            }
        )
    return {
        "source_id": SOURCE_ID,
        "comparison_rows": rows,
        "passed": bool(rows) and all(bool(row["passed"]) for row in rows),
        "failure_count": sum(not bool(row["passed"]) for row in rows),
        "total_cells": len(rows),
        "verdict": (
            "runin_protocol_reproducible"
            if rows and all(bool(row["passed"]) for row in rows)
            else "runin_protocol_not_reproducible_do_not_train"
        ),
    }
