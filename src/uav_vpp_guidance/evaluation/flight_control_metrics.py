"""
Flight-control comparison metrics.

Computes per-episode statistics for the multi-waypoint and sustained-turn tasks.
Canonical metric names follow the task specification (sections 7.5 and 8.5).
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


# Physical plausibility caps. Values beyond these limits are treated as
# simulator divergence / numerical garbage and excluded from statistics.
MAX_RANGE_M = 50_000.0          # close-range tasks stay well below this
MAX_TURN_RADIUS_M = 50_000.0
MAX_SPEED_MPS = 1_000.0
MAX_NZ_G = 20.0
MIN_NZ_G = -5.0


def _safe_mean(values: List[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float(np.nan)
    return float(np.mean(arr))


def _safe_max(values: List[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float(np.nan)
    return float(np.max(arr))


def _safe_median(values: List[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float(np.nan)
    return float(np.median(arr))


def _safe_std(values: List[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return float(np.nan)
    return float(np.std(arr, ddof=1))


def _mask_physical(values: List[float], lo: float, hi: float) -> np.ndarray:
    """Return a boolean mask of finite values within [lo, hi]."""
    arr = np.asarray(values, dtype=float)
    return np.isfinite(arr) & (arr >= lo) & (arr <= hi)


def _clip_to_physical(values: List[float], lo: float, hi: float) -> List[float]:
    """Replace values outside [lo, hi] or non-finite with NaN."""
    arr = np.asarray(values, dtype=float)
    mask = _mask_physical(values, lo, hi)
    out = np.full_like(arr, np.nan, dtype=float)
    out[mask] = arr[mask]
    return out.tolist()


def _energy_loss_rate_mps2(speeds: List[float], times: List[float]) -> float:
    """Linear regression slope of speed vs time, returned as a loss rate."""
    s = np.asarray(speeds, dtype=float)
    t = np.asarray(times, dtype=float)
    valid = np.isfinite(s) & np.isfinite(t)
    s = s[valid]
    t = t[valid]
    if len(s) < 2:
        return float(np.nan)
    # Fit speed = a + b * t; energy loss rate = -b (clip negative to 0).
    # Use the closed-form slope to avoid np.linalg.lstsq aborts on Windows
    # when multiple OpenMP runtimes are present.
    t_mean = float(np.mean(t))
    var_t = float(np.mean((t - t_mean) ** 2))
    if var_t <= 1e-12:
        return float(np.nan)
    cov = float(np.mean((t - t_mean) * (s - np.mean(s))))
    b = cov / var_t
    return max(0.0, float(-b))


def _safe_rmse(values: List[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float(np.nan)
    return float(np.sqrt(np.mean(arr ** 2)))


def _deadband_sign(value: float, deadband: float) -> int:
    if not np.isfinite(value):
        return 0
    if value > deadband:
        return 1
    if value < -deadband:
        return -1
    return 0


def _recovery_delay_steps(
    cmd_dev: np.ndarray,
    actual_dev: np.ndarray,
    command_deadband: float = 0.2,
    response_deadband: float = 0.03,
    max_window_steps: int = 20,
) -> float:
    """Mean step delay from nz command reversal to actual nz reversal."""
    delays = []
    for idx in range(1, len(cmd_dev)):
        prev_sign = _deadband_sign(cmd_dev[idx - 1], command_deadband)
        next_sign = _deadband_sign(cmd_dev[idx], command_deadband)
        if prev_sign == 0 or next_sign == 0 or prev_sign == next_sign:
            continue

        stop = min(len(actual_dev), idx + max_window_steps + 1)
        for j in range(idx, stop):
            if j == 0:
                continue
            actual_delta = actual_dev[j] - actual_dev[j - 1]
            if _deadband_sign(actual_delta, response_deadband) == next_sign:
                delays.append(float(j - idx))
                break
            if _deadband_sign(actual_dev[j], command_deadband) == next_sign:
                delays.append(float(j - idx))
                break

    return _safe_mean(delays)


def _overshoot_pct(
    cmd_dev: np.ndarray,
    actual_dev: np.ndarray,
    command_deadband: float = 0.2,
) -> float:
    """Maximum nz overshoot relative to commanded deviation from 1g."""
    overshoots = []
    start = None
    active_sign = 0

    def close_segment(end: int) -> None:
        if start is None or end <= start:
            return
        target_peak = float(np.nanmax(np.abs(cmd_dev[start:end])))
        if not np.isfinite(target_peak) or target_peak <= command_deadband:
            return
        actual_peak = float(np.nanmax(np.abs(actual_dev[start:end])))
        if not np.isfinite(actual_peak):
            return
        overshoots.append(max(0.0, (actual_peak - target_peak) / target_peak * 100.0))

    for idx, value in enumerate(cmd_dev):
        sign = _deadband_sign(value, command_deadband)
        if sign == 0:
            close_segment(idx)
            start = None
            active_sign = 0
            continue
        if start is None:
            start = idx
            active_sign = sign
            continue
        if sign != active_sign:
            close_segment(idx)
            start = idx
            active_sign = sign

    close_segment(len(cmd_dev))
    return _safe_max(overshoots) if overshoots else np.nan


def compute_multi_waypoint_metrics(trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute per-episode statistics for the multi-waypoint task."""
    ranges = _clip_to_physical(
        [float(r.get("range_m", np.nan)) for r in trajectory], 0.0, MAX_RANGE_M
    )
    speeds = _clip_to_physical(
        [float(r.get("speed_mps", np.nan)) for r in trajectory], 0.0, MAX_SPEED_MPS
    )
    nz = _clip_to_physical(
        [float(r.get("nz_g", np.nan)) for r in trajectory], MIN_NZ_G, MAX_NZ_G
    )
    aggressiveness = [
        float(r.get("aggressiveness"))
        for r in trajectory
        if r.get("aggressiveness") is not None
    ]
    gain_scales = [
        float(r.get("gain_scale"))
        for r in trajectory
        if r.get("gain_scale") is not None
    ]
    saturation_flags = [bool(r.get("saturation_flag", False)) for r in trajectory]

    stats = {
        # Canonical names from the task specification (section 7.5).
        "mean_track_error_m": _safe_mean(ranges),
        "std_track_error_m": _safe_std(ranges),
        "median_track_error_m": _safe_median(ranges),
        "max_track_error_m": _safe_max(ranges),
        "mean_speed_mps": _safe_mean(speeds),
        "max_speed_mps": _safe_max(speeds),
        "mean_nz_g": _safe_mean(nz),
        "max_nz_g": _safe_max(nz),
        "saturation_ratio": float(np.mean(saturation_flags)) if saturation_flags else 0.0,
    }

    if aggressiveness:
        stats["mean_aggressiveness"] = _safe_mean(aggressiveness)
        stats["mean_gain_scale"] = _safe_mean(gain_scales)
    else:
        stats["mean_aggressiveness"] = None
        stats["mean_gain_scale"] = None

    return stats


def compute_break_turn_metrics(trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute per-episode statistics for the break-turn / yo-yo task."""
    ranges = _clip_to_physical(
        [float(r.get("range_m", np.nan)) for r in trajectory], 0.0, MAX_RANGE_M
    )
    speeds = _clip_to_physical(
        [float(r.get("speed_mps", np.nan)) for r in trajectory], 0.0, MAX_SPEED_MPS
    )
    nz = _clip_to_physical(
        [float(r.get("nz_g", np.nan)) for r in trajectory], MIN_NZ_G, MAX_NZ_G
    )
    nz_cmd = _clip_to_physical(
        [float(r.get("nz_cmd", np.nan)) for r in trajectory], MIN_NZ_G, MAX_NZ_G
    )
    times = [float(r.get("time_s", np.nan)) for r in trajectory]
    aggressiveness = [
        float(r.get("aggressiveness"))
        for r in trajectory
        if r.get("aggressiveness") is not None
    ]
    gain_scales = [
        float(r.get("gain_scale"))
        for r in trajectory
        if r.get("gain_scale") is not None
    ]
    saturation_flags = [bool(r.get("saturation_flag", False)) for r in trajectory]

    nz_arr = np.asarray(nz, dtype=float)
    nz_cmd_arr = np.asarray(nz_cmd, dtype=float)
    nz_pair_mask = np.isfinite(nz_arr) & np.isfinite(nz_cmd_arr)
    if np.any(nz_pair_mask):
        nz_errors = (nz_cmd_arr[nz_pair_mask] - nz_arr[nz_pair_mask]).tolist()
        cmd_dev = nz_cmd_arr[nz_pair_mask] - 1.0
        actual_dev = nz_arr[nz_pair_mask] - 1.0
        nz_tracking_rmse = _safe_rmse(nz_errors)
        recovery_delay = _recovery_delay_steps(cmd_dev, actual_dev)
        overshoot_pct = _overshoot_pct(cmd_dev, actual_dev)
    else:
        nz_tracking_rmse = float(np.nan)
        recovery_delay = float(np.nan)
        overshoot_pct = float(np.nan)

    stats = {
        "mean_track_error_m": _safe_mean(ranges),
        "std_track_error_m": _safe_std(ranges),
        "median_track_error_m": _safe_median(ranges),
        "max_track_error_m": _safe_max(ranges),
        "mean_speed_mps": _safe_mean(speeds),
        "max_speed_mps": _safe_max(speeds),
        "mean_nz_g": _safe_mean(nz),
        "max_nz_g": _safe_max(nz),
        "nz_tracking_rmse": nz_tracking_rmse,
        "recovery_delay": recovery_delay,
        "overshoot_pct": overshoot_pct,
        "energy_loss_rate_mps2": _energy_loss_rate_mps2(speeds, times),
        "saturation_ratio": float(np.mean(saturation_flags)) if saturation_flags else 0.0,
    }

    if aggressiveness:
        stats["mean_aggressiveness"] = _safe_mean(aggressiveness)
        stats["mean_gain_scale"] = _safe_mean(gain_scales)
    else:
        stats["mean_aggressiveness"] = None
        stats["mean_gain_scale"] = None

    return stats


def compute_sustained_turn_metrics(trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute per-episode statistics for the sustained-turn task."""
    radii = _clip_to_physical(
        [float(r.get("turn_radius_m", np.nan)) for r in trajectory], 0.0, MAX_TURN_RADIUS_M
    )
    speeds = _clip_to_physical(
        [float(r.get("speed_mps", np.nan)) for r in trajectory], 0.0, MAX_SPEED_MPS
    )
    nz = _clip_to_physical(
        [float(r.get("nz_g", np.nan)) for r in trajectory], MIN_NZ_G, MAX_NZ_G
    )
    headings = [float(r.get("heading_deg", np.nan)) for r in trajectory]
    times = [float(r.get("time_s", np.nan)) for r in trajectory]
    aggressiveness = [
        float(r.get("aggressiveness"))
        for r in trajectory
        if r.get("aggressiveness") is not None
    ]
    gain_scales = [
        float(r.get("gain_scale"))
        for r in trajectory
        if r.get("gain_scale") is not None
    ]
    saturation_flags = [bool(r.get("saturation_flag", False)) for r in trajectory]

    # Exclude first 5 s from radius statistics (transient).
    transient_steps = sum(1 for tt in times if tt < 5.0)
    radii_steady = radii[transient_steps:] if len(radii) > transient_steps else radii

    # Turn rate from heading (deg/s).
    turn_rates = []
    for i in range(1, len(headings)):
        if np.isfinite(headings[i]) and np.isfinite(headings[i - 1]) and np.isfinite(times[i]) and np.isfinite(times[i - 1]):
            dt = times[i] - times[i - 1]
            if dt > 0:
                dh = headings[i] - headings[i - 1]
                # Normalize to [-180, 180]
                dh = (dh + 180.0) % 360.0 - 180.0
                turn_rates.append(abs(dh) / dt)

    if trajectory:
        last_orbits = trajectory[-1].get("completed_orbits", 0.0)
        completed_orbits = float(last_orbits) if last_orbits is not None and np.isfinite(float(last_orbits)) else 0.0
    else:
        completed_orbits = 0.0

    avg_turn_rate = _safe_mean(turn_rates)
    std_turn_rate = _safe_std(turn_rates)
    avg_turn_radius = _safe_mean(radii_steady)
    std_turn_radius = _safe_std(radii_steady)
    avg_speed = _safe_mean(speeds)
    std_speed = _safe_std(speeds)

    stats = {
        # Canonical names from the task specification (section 8.5).
        "completed_orbits": completed_orbits,
        "avg_turn_rate_deg_s": avg_turn_rate,
        "std_turn_rate_deg_s": std_turn_rate,
        "avg_turn_radius_m": avg_turn_radius,
        "std_turn_radius_m": std_turn_radius,
        "radius_std_m": std_turn_radius,
        "avg_speed_mps": avg_speed,
        "std_speed_mps": std_speed,
        "mean_nz_g": _safe_mean(nz),
        "max_nz_g": _safe_max(nz),
        "energy_loss_rate_mps2": _energy_loss_rate_mps2(speeds, times),
        "saturation_ratio": float(np.mean(saturation_flags)) if saturation_flags else 0.0,
    }

    if aggressiveness:
        stats["mean_aggressiveness"] = _safe_mean(aggressiveness)
        stats["mean_gain_scale"] = _safe_mean(gain_scales)
    else:
        stats["mean_aggressiveness"] = None
        stats["mean_gain_scale"] = None

    return stats
