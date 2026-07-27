"""Pure offline F05 VPP semantic comparisons; no runtime VPP dependency."""
from math import atan2, radians, tan
import numpy as np

LATERAL_BOUND_M = 800.0


def fixed_metre(action: float) -> float:
    return float(np.clip(action, -1.0, 1.0) * LATERAL_BOUND_M)


def normalized_fraction(action: float) -> dict:
    bounded = float(np.clip(action, -1.0, 1.0))
    return {"normalized_action": bounded, "offset_m": fixed_metre(bounded)}


def distance_scaled(action: float, range_m: float, *, max_angle_deg: float = 30.0, min_range_m: float = 50.0) -> float:
    if not np.isfinite(range_m) or range_m < 0.0:
        raise ValueError("range_m must be finite and non-negative")
    bounded_action = float(np.clip(action, -1.0, 1.0))
    effective_range = max(float(range_m), min_range_m)
    requested = effective_range * tan(radians(max_angle_deg) * bounded_action)
    return float(np.clip(requested, -LATERAL_BOUND_M, LATERAL_BOUND_M))


def angular_meaning_deg(offset_m: float, range_m: float) -> float:
    return float(np.degrees(atan2(offset_m, max(float(range_m), 1.0e-9))))


def compare_semantics(actions=(-1.0, -0.5, 0.0, 0.5, 1.0), ranges_m=(0.0, 50.0, 800.0, 2500.0, 5000.0)) -> dict:
    rows = []
    for range_m in ranges_m:
        for action in actions:
            fixed = fixed_metre(action)
            scaled = distance_scaled(action, range_m)
            rows.append({"range_m": float(range_m), "action": float(action), "fixed_metre_offset_m": fixed, "fixed_metre_angle_deg": angular_meaning_deg(fixed, range_m), "normalized_fraction": normalized_fraction(action), "distance_scaled_offset_m": scaled, "distance_scaled_angle_deg": angular_meaning_deg(scaled, range_m)})
    return {"action_bounds": [-1.0, 1.0], "lateral_bound_m": LATERAL_BOUND_M, "distance_scaled_parameters": {"max_angle_deg": 30.0, "min_range_m": 50.0}, "rows": rows, "baseline_examples": {"fixed_800m_at_2500m_deg": angular_meaning_deg(800.0, 2500.0), "fixed_800m_at_800m_deg": angular_meaning_deg(800.0, 800.0)}, "scope": "offline action-to-offset geometry only; tactical, safety, tracking, policy stability, checkpoint migration, and backend evidence unavailable"}
