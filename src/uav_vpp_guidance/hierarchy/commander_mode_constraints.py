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
        "range_rate_mps": np.nan,
        "min_range_so_far_m": np.nan,
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
    snapshot["range_rate_mps"] = _safe_float(rel_state.get("range_rate_mps"))
    snapshot["min_range_so_far_m"] = _safe_float(
        getattr(env, "_merge_min_range_so_far_m", np.nan)
    )

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
        "range_rate_mps": np.nan,
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
    state["range_rate_mps"] = _safe_float(snapshot.get("range_rate_mps"))
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


def evaluate_head_on_post_merge_reopened_crossing_target_threat_clamp_state(
    *,
    snapshot: Dict[str, Any],
    target_threat_cfg: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate whether a head_on crossing request should stop under target threat."""
    cfg = target_threat_cfg or {}
    enabled = bool(cfg.get("enabled", False))
    state = {
        "configured": enabled,
        "active": False,
        "reason": "disabled" if not enabled else "inactive",
        "range_m": _safe_float(snapshot.get("range_m")),
        "range_rate_mps": _safe_float(snapshot.get("range_rate_mps")),
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

    range_m = state["range_m"]
    min_range_m = cfg.get("min_range_m")
    if min_range_m is not None:
        if not np.isfinite(range_m):
            state["reason"] = "nonfinite_range"
            return state
        if range_m < float(min_range_m):
            state["reason"] = "below_min_range"
            return state

    if not state["target_in_attack_zone"]:
        if bool(cfg.get("pre_threat_entry_guard_enabled", False)):
            if bool(cfg.get("pre_threat_require_no_attack_zone", True)) and (
                state["ego_in_attack_zone"] or state["target_in_attack_zone"]
            ):
                state["reason"] = "pre_threat_attack_zone_active"
                return state

            min_pre_threat_range_m = float(
                cfg.get("pre_threat_min_range_m", cfg.get("min_range_m", 4500.0))
            )
            if not np.isfinite(state["range_m"]):
                state["reason"] = "pre_threat_nonfinite_range"
                return state
            if state["range_m"] < min_pre_threat_range_m:
                state["reason"] = "pre_threat_below_min_range"
                return state

            min_range_rate_mps = float(
                cfg.get("pre_threat_min_range_rate_mps", 50.0)
            )
            if not np.isfinite(state["range_rate_mps"]):
                state["reason"] = "pre_threat_nonfinite_range_rate"
                return state
            if state["range_rate_mps"] < min_range_rate_mps:
                state["reason"] = "pre_threat_not_opening_fast_enough"
                return state

            max_vp_forward_bias_m = float(
                cfg.get("pre_threat_max_vp_forward_bias_m", -2500.0)
            )
            if not np.isfinite(state["vp_forward_bias_m"]):
                state["reason"] = "pre_threat_nonfinite_vp_forward_bias"
                return state
            if state["vp_forward_bias_m"] > max_vp_forward_bias_m:
                state["reason"] = "pre_threat_vp_forward_not_negative_enough"
                return state

            min_abs_lateral_ratio = float(
                cfg.get("pre_threat_min_abs_vp_lateral_to_range_ratio", 1.5)
            )
            if not np.isfinite(state["vp_lateral_to_range_ratio"]):
                state["reason"] = "pre_threat_nonfinite_lateral_ratio"
                return state
            if state["vp_lateral_to_range_ratio"] < min_abs_lateral_ratio:
                state["reason"] = "pre_threat_lateral_ratio_below_min"
                return state

            positive_lateral_min_vp_lateral_bias_m = cfg.get(
                "pre_threat_positive_lateral_min_vp_lateral_bias_m"
            )
            positive_lateral_max_vp_forward_bias_m = cfg.get(
                "pre_threat_positive_lateral_max_vp_forward_bias_m"
            )
            if (
                positive_lateral_min_vp_lateral_bias_m is not None
                and positive_lateral_max_vp_forward_bias_m is not None
            ):
                if not np.isfinite(state["vp_lateral_bias_m"]):
                    state["reason"] = "pre_threat_nonfinite_vp_lateral_bias"
                    return state
                if state["vp_lateral_bias_m"] >= float(
                    positive_lateral_min_vp_lateral_bias_m
                ) and state["vp_forward_bias_m"] > float(
                    positive_lateral_max_vp_forward_bias_m
                ):
                    state["reason"] = (
                        "pre_threat_positive_lateral_forward_bias_not_negative_enough"
                    )
                    return state

            state["active"] = True
            state["reason"] = "pre_threat_opening_overlateral_negative_forward_head_on"
            return state

        state["reason"] = "target_not_in_attack_zone"
        return state

    state["active"] = True
    state["reason"] = "target_attack_zone_reopened_head_on"
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
        "min_range_so_far_m": _safe_float(snapshot.get("min_range_so_far_m")),
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


def evaluate_head_on_post_merge_reopened_crossing_overdeep_clamp_state(
    *,
    snapshot: Dict[str, Any],
    overdeep_cfg: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate whether reopened head_on is too overdeep for a crossing detour."""
    cfg = overdeep_cfg or {}
    enabled = bool(cfg.get("enabled", False))
    state = {
        "configured": enabled,
        "active": False,
        "reason": "disabled" if not enabled else "inactive",
        "range_m": _safe_float(snapshot.get("range_m")),
        "min_range_so_far_m": _safe_float(snapshot.get("min_range_so_far_m")),
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

    range_m = state["range_m"]
    if not np.isfinite(range_m):
        state["reason"] = "nonfinite_range"
        return state
    close_range_max_range_m = _safe_float(cfg.get("close_range_max_range_m"))
    if np.isfinite(close_range_max_range_m) and range_m <= close_range_max_range_m:
        state["active"] = True
        state["reason"] = "close_range_reengagement_crossing"
        return state

    min_range_m = float(cfg.get("min_range_m", 4500.0))
    if range_m < min_range_m:
        state["reason"] = "below_min_range"
        return state

    vp_forward_bias_m = state["vp_forward_bias_m"]
    if not np.isfinite(vp_forward_bias_m):
        state["reason"] = "nonfinite_vp_forward_bias"
        return state
    min_negative_vp_forward_bias_m = float(
        cfg.get("min_negative_vp_forward_bias_m", 9000.0)
    )
    if vp_forward_bias_m > -min_negative_vp_forward_bias_m:
        state["reason"] = "forward_bias_not_overdeep"
        return state

    vp_lateral_to_range_ratio = state["vp_lateral_to_range_ratio"]
    if not np.isfinite(vp_lateral_to_range_ratio):
        state["reason"] = "nonfinite_vp_lateral_to_range_ratio"
        return state
    max_abs_vp_lateral_to_range_ratio = float(
        cfg.get("max_abs_vp_lateral_to_range_ratio", 0.5)
    )
    if vp_lateral_to_range_ratio > max_abs_vp_lateral_to_range_ratio:
        state["reason"] = "lateral_ratio_large_enough"
        return state

    state["active"] = True
    state["reason"] = "overdeep_low_lateral_reopened_head_on"
    return state


def evaluate_head_on_post_merge_reopened_crossing_geometry_quality_guard_state(
    *,
    snapshot: Dict[str, Any],
    altitude_history_m: List[float],
    leash_active_steps: int,
    overdeep_active_steps: int,
    geometry_guard_cfg: Optional[Dict[str, Any]],
    leash_state: Optional[Dict[str, Any]] = None,
    overdeep_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate a narrow post-merge head_on geometry-quality guard state."""
    cfg = geometry_guard_cfg or {}
    enabled = bool(cfg.get("enabled", False))
    lookback_steps = max(1, int(cfg.get("altitude_trend_lookback_steps", 24)))
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
        "altitude_trend_lookback_steps": lookback_steps,
        "altitude_delta_m_lookback": np.nan,
        "leash_active_steps": int(max(0, leash_active_steps)),
        "overdeep_active_steps": int(max(0, overdeep_active_steps)),
        "leash_active": bool((leash_state or {}).get("active", False)),
        "overdeep_active": bool((overdeep_state or {}).get("active", False)),
        "overdeep_seen_since_post_merge": int(max(0, overdeep_active_steps)) > 0,
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

    if bool(cfg.get("require_leash_active", True)) and not state["leash_active"]:
        state["reason"] = "leash_inactive"
        return state

    min_leash_active_steps = max(1, int(cfg.get("min_leash_active_steps", 72)))
    if state["leash_active_steps"] < min_leash_active_steps:
        state["reason"] = "leash_not_sustained"
        return state

    if bool(cfg.get("require_overdeep_seen", True)):
        min_overdeep_active_steps = max(
            1, int(cfg.get("min_overdeep_active_steps", 1))
        )
        if state["overdeep_active_steps"] < min_overdeep_active_steps:
            state["reason"] = "overdeep_not_seen"
            return state

    if len(altitude_history_m) < lookback_steps:
        state["reason"] = "insufficient_altitude_history"
        return state

    altitude_m = state["altitude_m"]
    if not np.isfinite(altitude_m):
        state["reason"] = "nonfinite_altitude"
        return state

    altitude_delta_m_lookback = altitude_m - float(altitude_history_m[-lookback_steps])
    state["altitude_delta_m_lookback"] = altitude_delta_m_lookback

    vp_forward_bias_m = state["vp_forward_bias_m"]
    if not np.isfinite(vp_forward_bias_m):
        state["reason"] = "nonfinite_vp_forward_bias"
        return state

    vp_lateral_bias_m = state["vp_lateral_bias_m"]
    if not np.isfinite(vp_lateral_bias_m):
        state["reason"] = "nonfinite_vp_lateral_bias"
        return state

    vp_lateral_to_range_ratio = state["vp_lateral_to_range_ratio"]
    if not np.isfinite(vp_lateral_to_range_ratio):
        state["reason"] = "nonfinite_vp_lateral_to_range_ratio"
        return state

    min_abs_vp_lateral_bias_m = float(cfg.get("min_abs_vp_lateral_bias_m", 2500.0))
    min_abs_vp_lateral_to_range_ratio = float(
        cfg.get("min_abs_vp_lateral_to_range_ratio", 0.5)
    )
    if (
        abs(vp_lateral_bias_m) < min_abs_vp_lateral_bias_m
        and vp_lateral_to_range_ratio < min_abs_vp_lateral_to_range_ratio
    ):
        state["reason"] = "lateral_bias_too_small"
        return state

    min_negative_vp_forward_bias_m = float(
        cfg.get("min_negative_vp_forward_bias_m", 8000.0)
    )
    min_positive_vp_forward_bias_m = float(
        cfg.get("min_positive_vp_forward_bias_m", 800.0)
    )
    min_altitude_drop_m = float(cfg.get("min_altitude_drop_m", 250.0))
    min_altitude_gain_m = float(cfg.get("min_altitude_gain_m", 250.0))

    if (
        altitude_delta_m_lookback <= -min_altitude_drop_m
        and vp_forward_bias_m <= -min_negative_vp_forward_bias_m
    ):
        state["active"] = True
        state["reason"] = "geometry_quality_low_side_negative_forward_lateral"
        return state

    if altitude_delta_m_lookback >= min_altitude_gain_m:
        if vp_forward_bias_m >= min_positive_vp_forward_bias_m:
            state["active"] = True
            state["reason"] = "geometry_quality_high_side_positive_forward_lateral"
            return state
        if vp_forward_bias_m <= -min_negative_vp_forward_bias_m:
            state["active"] = True
            state["reason"] = "geometry_quality_high_side_negative_forward_lateral"
            return state

    state["reason"] = "geometry_quality_not_degraded"
    return state


def evaluate_head_on_post_merge_recovery_hold_state(
    *,
    snapshot: Dict[str, Any],
    recovery_hold_cfg: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Keep recovery active while reopened head_on geometry remains severely over-lateral."""
    cfg = recovery_hold_cfg or {}
    enabled = bool(cfg.get("enabled", False))
    state = {
        "configured": enabled,
        "active": False,
        "reason": "disabled" if not enabled else "inactive",
        "range_m": _safe_float(snapshot.get("range_m")),
        "range_rate_mps": _safe_float(snapshot.get("range_rate_mps")),
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

    if bool(cfg.get("require_no_attack_zone", True)) and (
        state["ego_in_attack_zone"] or state["target_in_attack_zone"]
    ):
        state["reason"] = "attack_zone_active"
        return state

    range_m = state["range_m"]
    if not np.isfinite(range_m):
        state["reason"] = "nonfinite_range"
        return state
    min_range_m = float(cfg.get("min_range_m", 4500.0))
    if range_m < min_range_m:
        state["reason"] = "below_min_range"
        return state

    vp_lateral_bias_m = state["vp_lateral_bias_m"]
    if not np.isfinite(vp_lateral_bias_m):
        state["reason"] = "nonfinite_vp_lateral_bias"
        return state
    vp_lateral_to_range_ratio = state["vp_lateral_to_range_ratio"]
    if not np.isfinite(vp_lateral_to_range_ratio):
        state["reason"] = "nonfinite_vp_lateral_to_range_ratio"
        return state

    min_abs_vp_lateral_bias_m = float(cfg.get("min_abs_vp_lateral_bias_m", 6000.0))
    if abs(vp_lateral_bias_m) < min_abs_vp_lateral_bias_m:
        state["reason"] = "vp_lateral_bias_not_large_enough"
        return state

    min_abs_vp_lateral_to_range_ratio = float(
        cfg.get("min_abs_vp_lateral_to_range_ratio", 1.4)
    )
    if abs(vp_lateral_to_range_ratio) < min_abs_vp_lateral_to_range_ratio:
        state["reason"] = "vp_lateral_to_range_ratio_not_large_enough"
        return state

    state["active"] = True
    state["reason"] = "recovery_reopened_geometry_still_overlateral"
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
        0,
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


def apply_head_on_post_merge_recovery_hold(
    *,
    candidate_mode_id: Optional[int],
    active_mode_id: Optional[int],
    mode_registry: Dict[int, Dict[str, Any]],
    recovery_hold_state: Dict[str, Any],
    recovery_hold_cfg: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Keep the recovery specialist active while reopened geometry remains severely over-lateral."""
    cfg = recovery_hold_cfg or {}
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
    if not recovery_hold_state.get("active", False):
        return {
            "effective_mode_id": candidate_mode_id,
            "triggered": False,
            "reason": recovery_hold_state.get("reason", "inactive"),
        }

    recovery_mode_id = int(cfg.get("recovery_mode_id", 2))
    if active_mode_id is None or int(active_mode_id) != recovery_mode_id:
        return {
            "effective_mode_id": candidate_mode_id,
            "triggered": False,
            "reason": "non_recovery_mode",
        }
    if candidate_mode_id == recovery_mode_id:
        return {
            "effective_mode_id": candidate_mode_id,
            "triggered": False,
            "reason": "already_recovery_mode",
        }

    effective_mode_id = recovery_mode_id
    if effective_mode_id not in mode_registry:
        effective_mode_id = candidate_mode_id
    return {
        "effective_mode_id": int(effective_mode_id),
        "triggered": int(effective_mode_id) != candidate_mode_id,
        "reason": str(
            recovery_hold_state.get(
                "reason", "recovery_reopened_geometry_still_overlateral"
            )
        ),
    }


def apply_head_on_post_merge_reopened_crossing_target_threat_clamp(
    *,
    candidate_mode_id: Optional[int],
    mode_registry: Dict[int, Dict[str, Any]],
    target_threat_state: Dict[str, Any],
    target_threat_cfg: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Force head_on when a head_on crossing detour reopens target threat."""
    cfg = target_threat_cfg or {}
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

    if not target_threat_state.get("active", False):
        return {
            "effective_mode_id": candidate_mode_id,
            "triggered": False,
            "reason": target_threat_state.get("reason", "inactive"),
        }
    crossing_mode_id = int(cfg.get("crossing_mode_id", 1))
    head_on_mode_id = cfg.get("head_on_mode_id")
    if candidate_mode_id == crossing_mode_id:
        effective_mode_id = int(cfg.get("forced_mode_id", 0))
    elif head_on_mode_id is not None and candidate_mode_id == int(head_on_mode_id):
        effective_mode_id = int(
            cfg.get("head_on_forced_mode_id", cfg.get("forced_mode_id", 0))
        )
    else:
        return {
            "effective_mode_id": candidate_mode_id,
            "triggered": False,
            "reason": "non_crossing_mode",
        }

    if effective_mode_id not in mode_registry:
        effective_mode_id = candidate_mode_id
    return {
        "effective_mode_id": int(effective_mode_id),
        "triggered": int(effective_mode_id) != candidate_mode_id,
        "reason": str(
            target_threat_state.get(
                "reason",
                "target_attack_zone_reopened_head_on",
            )
        ),
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


def apply_head_on_post_merge_reopened_crossing_overdeep_clamp(
    *,
    candidate_mode_id: Optional[int],
    mode_registry: Dict[int, Dict[str, Any]],
    overdeep_state: Dict[str, Any],
    overdeep_cfg: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Force head_on when reopened head_on is already overdeep with little lateral spread."""
    cfg = overdeep_cfg or {}
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

    if not overdeep_state.get("active", False):
        return {
            "effective_mode_id": candidate_mode_id,
            "triggered": False,
            "reason": overdeep_state.get("reason", "inactive"),
        }
    crossing_mode_id = int(cfg.get("crossing_mode_id", 1))
    head_on_mode_id = cfg.get("head_on_mode_id")
    if candidate_mode_id == crossing_mode_id:
        effective_mode_id = int(cfg.get("forced_mode_id", 0))
    elif head_on_mode_id is not None and candidate_mode_id == int(head_on_mode_id):
        if str(overdeep_state.get("reason")) == "close_range_reengagement_crossing":
            head_on_min_range_so_far_m = _safe_float(
                cfg.get("head_on_min_range_so_far_m")
            )
            min_range_so_far_m = _safe_float(overdeep_state.get("min_range_so_far_m"))
            if np.isfinite(head_on_min_range_so_far_m):
                if not np.isfinite(min_range_so_far_m):
                    return {
                        "effective_mode_id": candidate_mode_id,
                        "triggered": False,
                        "reason": "head_on_min_range_so_far_unavailable",
                    }
                if min_range_so_far_m < head_on_min_range_so_far_m:
                    return {
                        "effective_mode_id": candidate_mode_id,
                        "triggered": False,
                        "reason": "head_on_close_range_reengagement_min_range_so_far_too_small",
                    }
        else:
            head_on_min_vp_lateral_bias_m = _safe_float(
                cfg.get("head_on_min_vp_lateral_bias_m")
            )
            vp_lateral_bias_m = _safe_float(overdeep_state.get("vp_lateral_bias_m"))
            if np.isfinite(head_on_min_vp_lateral_bias_m):
                if not np.isfinite(vp_lateral_bias_m):
                    return {
                        "effective_mode_id": candidate_mode_id,
                        "triggered": False,
                        "reason": "head_on_overdeep_vp_lateral_bias_unavailable",
                    }
                if vp_lateral_bias_m < head_on_min_vp_lateral_bias_m:
                    return {
                        "effective_mode_id": candidate_mode_id,
                        "triggered": False,
                        "reason": "head_on_overdeep_vp_lateral_bias_too_negative",
                    }
        effective_mode_id = int(
            cfg.get("head_on_forced_mode_id", cfg.get("forced_mode_id", 0))
        )
    else:
        return {
            "effective_mode_id": candidate_mode_id,
            "triggered": False,
            "reason": "non_crossing_mode",
        }

    if effective_mode_id not in mode_registry:
        effective_mode_id = candidate_mode_id
    return {
        "effective_mode_id": int(effective_mode_id),
        "triggered": int(effective_mode_id) != candidate_mode_id,
        "reason": str(
            overdeep_state.get("reason", "overdeep_low_lateral_reopened_head_on")
        ),
    }


def apply_head_on_post_merge_reopened_crossing_geometry_quality_guard(
    *,
    candidate_mode_id: Optional[int],
    mode_registry: Dict[int, Dict[str, Any]],
    geometry_guard_state: Dict[str, Any],
    geometry_guard_cfg: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Force head_on when post-merge head_on geometry quality has degraded."""
    cfg = geometry_guard_cfg or {}
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
    if not geometry_guard_state.get("active", False):
        return {
            "effective_mode_id": candidate_mode_id,
            "triggered": False,
            "reason": geometry_guard_state.get("reason", "inactive"),
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
        "reason": str(
            geometry_guard_state.get(
                "reason",
                "geometry_quality_low_side_negative_forward_lateral",
            )
        ),
    }
