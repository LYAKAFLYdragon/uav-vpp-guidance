"""Shared commander mode-constraint helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from uav_vpp_guidance.envs.observation import compute_relative_geometry
from uav_vpp_guidance.virtual_point.coordinate_transform import (
    world_to_offset_frame,
)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _extract_position_vector(state: Any) -> np.ndarray:
    if isinstance(state, dict):
        for key in ("position_m", "position_neu", "position"):
            value = state.get(key)
            if value is not None:
                vector = np.asarray(value, dtype=np.float64)
                if vector.shape == (3,):
                    return vector
    return np.full(3, np.nan, dtype=np.float64)


def _extract_altitude_m(state: Any) -> float:
    if isinstance(state, dict):
        altitude_m = _safe_float(state.get("altitude_m"))
        if np.isfinite(altitude_m):
            return altitude_m
    position = _extract_position_vector(state)
    if position.shape == (3,) and np.isfinite(position[2]):
        return float(position[2])
    return float("nan")


def _extract_vp_position_vector(virtual_point: Any) -> np.ndarray:
    if isinstance(virtual_point, dict):
        for key in ("position_neu", "position"):
            value = virtual_point.get(key)
            if value is not None:
                vector = np.asarray(value, dtype=np.float64)
                if vector.shape == (3,):
                    return vector
    return np.full(3, np.nan, dtype=np.float64)


def build_head_on_post_merge_reopened_crossing_snapshot(
    *,
    env: Any,
    task_name: Optional[str],
) -> Dict[str, Any]:
    """Capture one consistent commander-side snapshot for leash evaluation."""
    snapshot = {
        "task_name": str(task_name) if task_name is not None else None,
        "available": False,
        "reason": "inactive",
        "first_pass_complete": False,
        "range_m": np.nan,
        "hp_advantage": np.nan,
        "ego_in_attack_zone": False,
        "target_in_attack_zone": False,
        "altitude_m": np.nan,
        "vp_forward_bias_m": np.nan,
        "vp_lateral_bias_m": np.nan,
        "vp_lateral_to_range_ratio": np.nan,
    }
    if env is None:
        snapshot["reason"] = "no_env"
        return snapshot
    if not hasattr(env, "_get_current_states"):
        snapshot["reason"] = "missing_state_accessor"
        return snapshot

    first_pass_complete = bool(getattr(env, "_first_pass_complete", False))
    snapshot["first_pass_complete"] = first_pass_complete
    own_state, target_state = env._get_current_states(noisy=False)
    rel_state = compute_relative_geometry(own_state, target_state)
    range_m = _safe_float(rel_state.get("range_m"))
    snapshot["range_m"] = range_m

    combat_info = getattr(getattr(env, "combat_hp", None), "last_info", {}) or {}
    snapshot["ego_in_attack_zone"] = bool(combat_info.get("ego_in_attack_zone", False))
    snapshot["target_in_attack_zone"] = bool(
        combat_info.get("target_in_attack_zone", False)
    )

    hp_advantage = combat_info.get("hp_advantage")
    if hp_advantage is None:
        ego_hp = _safe_float(combat_info.get("ego_hp"))
        target_hp = _safe_float(combat_info.get("target_hp"))
        if np.isfinite(ego_hp) and np.isfinite(target_hp):
            hp_advantage = ego_hp - target_hp
    snapshot["hp_advantage"] = _safe_float(hp_advantage)
    snapshot["altitude_m"] = _extract_altitude_m(own_state)

    virtual_point = getattr(env, "_last_virtual_point", None)
    vp_pos = _extract_vp_position_vector(virtual_point)
    target_pos = _extract_position_vector(target_state)
    if (
        vp_pos.shape == (3,)
        and target_pos.shape == (3,)
        and np.all(np.isfinite(vp_pos))
        and np.all(np.isfinite(target_pos))
    ):
        target_relative_vp = world_to_offset_frame(
            vp_pos - target_pos,
            "target_velocity",
            own_state=own_state,
            target_state=target_state,
        )
        snapshot["vp_forward_bias_m"] = _safe_float(target_relative_vp[0])
        snapshot["vp_lateral_bias_m"] = _safe_float(target_relative_vp[1])
        if np.isfinite(range_m) and range_m > 1e-6:
            snapshot["vp_lateral_to_range_ratio"] = abs(
                float(snapshot["vp_lateral_bias_m"])
            ) / float(range_m)
    else:
        snapshot["reason"] = "missing_virtual_point"
        return snapshot

    snapshot["available"] = True
    snapshot["reason"] = "available"
    return snapshot


def evaluate_head_on_post_merge_reopened_crossing_leash_state(
    *,
    snapshot: Dict[str, Any],
    leash_cfg: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate whether the narrow head_on reopened-state leash is active."""
    cfg = leash_cfg or {}
    enabled = bool(cfg.get("enabled", False))
    state = {
        "configured": enabled,
        "active": False,
        "reason": "disabled" if not enabled else "inactive",
        "range_m": np.nan,
        "hp_advantage": np.nan,
        "ego_in_attack_zone": False,
        "target_in_attack_zone": False,
        "first_pass_complete": False,
        "altitude_m": np.nan,
        "vp_forward_bias_m": np.nan,
        "vp_lateral_bias_m": np.nan,
        "vp_lateral_to_range_ratio": np.nan,
    }
    if not enabled:
        return state
    if not bool(snapshot.get("available", False)):
        state["reason"] = str(snapshot.get("reason", "unavailable"))
        return state

    target_task_name = str(cfg.get("task_name", "head_on"))
    if str(snapshot.get("task_name")) != target_task_name:
        state["reason"] = "task_mismatch"
        return state

    first_pass_complete = bool(snapshot.get("first_pass_complete", False))
    state["first_pass_complete"] = first_pass_complete
    if not first_pass_complete:
        state["reason"] = "pre_merge"
        return state

    range_m = _safe_float(snapshot.get("range_m"))
    state["range_m"] = range_m
    state["hp_advantage"] = _safe_float(snapshot.get("hp_advantage"))
    state["ego_in_attack_zone"] = bool(snapshot.get("ego_in_attack_zone", False))
    state["target_in_attack_zone"] = bool(snapshot.get("target_in_attack_zone", False))
    state["altitude_m"] = _safe_float(snapshot.get("altitude_m"))
    state["vp_forward_bias_m"] = _safe_float(snapshot.get("vp_forward_bias_m"))
    state["vp_lateral_bias_m"] = _safe_float(snapshot.get("vp_lateral_bias_m"))
    state["vp_lateral_to_range_ratio"] = _safe_float(
        snapshot.get("vp_lateral_to_range_ratio")
    )
    min_range_m = float(cfg.get("min_range_m", 4500.0))
    if not np.isfinite(range_m):
        state["reason"] = "nonfinite_range"
        return state
    if range_m < min_range_m:
        state["reason"] = "below_min_range"
        return state

    if bool(cfg.get("require_no_attack_zone", True)) and (
        state["ego_in_attack_zone"] or state["target_in_attack_zone"]
    ):
        state["reason"] = "attack_zone_active"
        return state

    state["active"] = True
    state["reason"] = "active"
    return state


def evaluate_head_on_post_merge_reopened_crossing_secondary_clamp_state(
    *,
    snapshot: Dict[str, Any],
    altitude_history_m: List[float],
    secondary_cfg: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate the low-altitude unresolved-lateral secondary clamp."""
    cfg = secondary_cfg or {}
    enabled = bool(cfg.get("enabled", False))
    lookback_steps = max(1, int(cfg.get("altitude_drop_lookback_steps", 12)))
    state = {
        "configured": enabled,
        "active": False,
        "reason": "disabled" if not enabled else "inactive",
        "range_m": _safe_float(snapshot.get("range_m")),
        "hp_advantage": _safe_float(snapshot.get("hp_advantage")),
        "ego_in_attack_zone": bool(snapshot.get("ego_in_attack_zone", False)),
        "target_in_attack_zone": bool(snapshot.get("target_in_attack_zone", False)),
        "first_pass_complete": bool(snapshot.get("first_pass_complete", False)),
        "altitude_m": _safe_float(snapshot.get("altitude_m")),
        "vp_forward_bias_m": _safe_float(snapshot.get("vp_forward_bias_m")),
        "vp_lateral_bias_m": _safe_float(snapshot.get("vp_lateral_bias_m")),
        "vp_lateral_to_range_ratio": _safe_float(
            snapshot.get("vp_lateral_to_range_ratio")
        ),
        "altitude_drop_lookback_steps": lookback_steps,
        "altitude_drop_m_lookback": np.nan,
    }
    if not enabled:
        return state
    if not bool(snapshot.get("available", False)):
        state["reason"] = str(snapshot.get("reason", "unavailable"))
        return state

    target_task_name = str(cfg.get("task_name", "head_on"))
    if str(snapshot.get("task_name")) != target_task_name:
        state["reason"] = "task_mismatch"
        return state
    if not state["first_pass_complete"]:
        state["reason"] = "pre_merge"
        return state

    altitude_m = state["altitude_m"]
    if not np.isfinite(altitude_m):
        state["reason"] = "nonfinite_altitude"
        return state
    max_altitude_m = float(cfg.get("max_altitude_m", 5000.0))
    if altitude_m >= max_altitude_m:
        state["reason"] = "above_max_altitude"
        return state

    vp_lateral_to_range_ratio = state["vp_lateral_to_range_ratio"]
    if not np.isfinite(vp_lateral_to_range_ratio):
        state["reason"] = "nonfinite_vp_lateral_to_range_ratio"
        return state
    min_abs_vp_lateral_to_range_ratio = float(
        cfg.get("min_abs_vp_lateral_to_range_ratio", 1.5)
    )
    if vp_lateral_to_range_ratio <= min_abs_vp_lateral_to_range_ratio:
        state["reason"] = "low_vp_lateral_to_range_ratio"
        return state

    if len(altitude_history_m) < lookback_steps:
        state["reason"] = "insufficient_altitude_history"
        return state
    altitude_drop_m_lookback = altitude_m - float(altitude_history_m[-lookback_steps])
    state["altitude_drop_m_lookback"] = altitude_drop_m_lookback
    min_altitude_drop_m = float(cfg.get("min_altitude_drop_m", 250.0))
    if altitude_drop_m_lookback > -min_altitude_drop_m:
        state["reason"] = "insufficient_altitude_drop"
        return state

    state["active"] = True
    state["reason"] = "active"
    return state


def apply_head_on_post_merge_reopened_crossing_leash(
    *,
    requested_mode_id: int,
    active_mode_id: Optional[int],
    consecutive_crossing_macro_steps: int,
    mode_registry: Dict[int, Dict[str, Any]],
    leash_state: Dict[str, Any],
    leash_cfg: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Apply a narrow cap on sustained crossing mode inside head_on reopenings."""
    cfg = leash_cfg or {}
    requested_mode_id = int(requested_mode_id)
    if not bool(cfg.get("enabled", False)):
        return {
            "effective_mode_id": requested_mode_id,
            "triggered": False,
            "reason": "disabled",
            "next_consecutive_crossing_macro_steps": 0,
        }

    crossing_mode_id = int(cfg.get("crossing_mode_id", 1))
    forced_mode_id = int(cfg.get("forced_mode_id", 0))
    max_consecutive_crossing_macro_steps = max(
        1,
        int(cfg.get("max_consecutive_crossing_macro_steps", 2)),
    )

    if not leash_state.get("active", False):
        return {
            "effective_mode_id": requested_mode_id,
            "triggered": False,
            "reason": leash_state.get("reason", "inactive"),
            "next_consecutive_crossing_macro_steps": 0,
        }

    if requested_mode_id != crossing_mode_id:
        return {
            "effective_mode_id": requested_mode_id,
            "triggered": False,
            "reason": "non_crossing_mode",
            "next_consecutive_crossing_macro_steps": 0,
        }

    proposed_consecutive = (
        int(consecutive_crossing_macro_steps) + 1
        if active_mode_id is not None and int(active_mode_id) == crossing_mode_id
        else 1
    )
    if proposed_consecutive > max_consecutive_crossing_macro_steps:
        effective_mode_id = forced_mode_id
        if effective_mode_id not in mode_registry:
            effective_mode_id = requested_mode_id
        return {
            "effective_mode_id": int(effective_mode_id),
            "triggered": int(effective_mode_id) != requested_mode_id,
            "reason": "max_consecutive_crossing_macro_steps_exceeded",
            "next_consecutive_crossing_macro_steps": 0,
        }

    return {
        "effective_mode_id": requested_mode_id,
        "triggered": False,
        "reason": "allowed",
        "next_consecutive_crossing_macro_steps": proposed_consecutive,
    }


def apply_head_on_post_merge_reopened_crossing_secondary_clamp(
    *,
    candidate_mode_id: Optional[int],
    mode_registry: Dict[int, Dict[str, Any]],
    secondary_state: Dict[str, Any],
    secondary_cfg: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Force head_on when low-altitude unresolved-lateral crossing becomes unsafe."""
    cfg = secondary_cfg or {}
    if candidate_mode_id is None:
        return {
            "effective_mode_id": None,
            "triggered": False,
            "reason": "no_candidate_mode",
        }
    candidate_mode_id = int(candidate_mode_id)
    if not bool(cfg.get("enabled", False)):
        return {
            "effective_mode_id": candidate_mode_id,
            "triggered": False,
            "reason": "disabled",
        }

    crossing_mode_id = int(cfg.get("crossing_mode_id", 1))
    forced_mode_id = int(cfg.get("forced_mode_id", 0))
    if not secondary_state.get("active", False):
        return {
            "effective_mode_id": candidate_mode_id,
            "triggered": False,
            "reason": secondary_state.get("reason", "inactive"),
        }
    if candidate_mode_id != crossing_mode_id:
        return {
            "effective_mode_id": candidate_mode_id,
            "triggered": False,
            "reason": "non_crossing_mode",
        }

    effective_mode_id = forced_mode_id
    if effective_mode_id not in mode_registry:
        effective_mode_id = candidate_mode_id
    return {
        "effective_mode_id": int(effective_mode_id),
        "triggered": int(effective_mode_id) != candidate_mode_id,
        "reason": "secondary_low_altitude_unresolved_lateral_descent",
    }
