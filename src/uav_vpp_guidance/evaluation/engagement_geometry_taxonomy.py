"""Analysis-only relative-geometry taxonomy with explicit angle semantics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Optional


@dataclass(frozen=True)
class GeometryTaxonomyConfig:
    hemisphere_boundary_deg: float = 90.0
    transition_half_width_deg: float = 5.0


def _finite_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def normalize_unsigned_angle_deg(value: Any) -> Optional[float]:
    """Normalize a signed or wrapped angle to the interval [0, 180]."""

    angle = _finite_float(value)
    if angle is None:
        return None
    wrapped = abs(angle) % 360.0
    return min(wrapped, 360.0 - wrapped)


def _hemisphere(angle_deg: Any, config: GeometryTaxonomyConfig) -> str:
    angle = normalize_unsigned_angle_deg(angle_deg)
    if angle is None:
        return "unknown"
    lower = config.hemisphere_boundary_deg - config.transition_half_width_deg
    upper = config.hemisphere_boundary_deg + config.transition_half_width_deg
    if angle < lower:
        return "front"
    if angle > upper:
        return "rear"
    return "transition"


def classify_relative_geometry(
    own_to_target_los_angle_deg: Any,
    target_velocity_to_own_los_angle_deg: Any,
    config: GeometryTaxonomyConfig | None = None,
) -> str:
    """Classify the four geometry quadrants while preserving a transition band."""

    cfg = config or GeometryTaxonomyConfig()
    own_hemi = _hemisphere(own_to_target_los_angle_deg, cfg)
    target_hemi = _hemisphere(target_velocity_to_own_los_angle_deg, cfg)
    if "unknown" in (own_hemi, target_hemi):
        return "unknown"
    if "transition" in (own_hemi, target_hemi):
        return "transition"
    return {
        ("front", "rear"): "advantage",
        ("front", "front"): "head_on",
        ("rear", "front"): "disadvantage",
        ("rear", "rear"): "neutral",
    }[(own_hemi, target_hemi)]
