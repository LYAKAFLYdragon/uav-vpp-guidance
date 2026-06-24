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

    stats = {
        "mean_track_error_m": _safe_mean(ranges),
        "std_track_error_m": _safe_std(ranges),
        "median_track_error_m": _safe_median(ranges),
        "max_track_error_m": _safe_max(ranges),
        "mean_speed_mps": _safe_mean(speeds),
        "max_speed_mps": _safe_max(speeds),
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
