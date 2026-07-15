"""A non-learning, current-information-only dynamic reference-path core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class DynamicReferencePathError(ValueError):
    """Raised when a V16 reference path violates its geometry contract."""


def _vector(value: Any, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise DynamicReferencePathError(f"{name} must be a finite 3-D vector")
    return result


@dataclass(frozen=True)
class DynamicReferencePathConfig:
    """Frozen V16 path geometry; all values are physical SI units."""

    escape_extent_m: float = 1200.0
    connector_fraction: float = 0.5
    opening_rate_start_mps: float = 0.0
    opening_rate_full_mps: float = 80.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.escape_extent_m) or self.escape_extent_m <= 0.0:
            raise DynamicReferencePathError("escape_extent_m must be finite and positive")
        if not 0.0 < self.connector_fraction < 1.0:
            raise DynamicReferencePathError("connector_fraction must lie in (0, 1)")
        if not self.opening_rate_full_mps > self.opening_rate_start_mps:
            raise DynamicReferencePathError("opening-rate bounds must be increasing")


def build_three_anchor_path(
    *,
    own_position_neu: Any,
    target_position_neu: Any,
    predicted_target_position_neu: Any,
    config: DynamicReferencePathConfig,
) -> np.ndarray:
    """Build escape, connector, and predicted-reentry anchors without future state.

    The only future-looking quantity is the already-frozen predictor output.
    The path is not a synthetic target and does not change terminal semantics.
    """

    own = _vector(own_position_neu, name="own_position_neu")
    target = _vector(target_position_neu, name="target_position_neu")
    predicted = _vector(predicted_target_position_neu, name="predicted_target_position_neu")
    los_to_ego = own - target
    distance = float(np.linalg.norm(los_to_ego))
    if distance <= 1e-6:
        raise DynamicReferencePathError("own and target positions must be distinct")
    escape = target + config.escape_extent_m * los_to_ego / distance
    connector = (1.0 - config.connector_fraction) * escape + config.connector_fraction * predicted
    path = np.vstack((escape, connector, predicted))
    if not np.isfinite(path).all():
        raise DynamicReferencePathError("reference path contains non-finite values")
    return path


def path_progress_from_range_rate(range_rate_mps: float, config: DynamicReferencePathConfig) -> float:
    """Map observed opening rate to a deterministic [0, 1] path progress."""

    value = float(range_rate_mps)
    if not np.isfinite(value):
        raise DynamicReferencePathError("range_rate_mps must be finite")
    return float(
        np.clip(
            (value - config.opening_rate_start_mps)
            / (config.opening_rate_full_mps - config.opening_rate_start_mps),
            0.0,
            1.0,
        )
    )


def select_path_anchor(path_neu: Any, progress: float) -> tuple[int, np.ndarray]:
    """Select escape, connector, or re-entry anchor from observed progress."""

    path = np.asarray(path_neu, dtype=np.float64)
    if path.shape != (3, 3) or not np.isfinite(path).all():
        raise DynamicReferencePathError("path_neu must be a finite [3, 3] matrix")
    if not np.isfinite(progress) or not 0.0 <= float(progress) <= 1.0:
        raise DynamicReferencePathError("progress must lie in [0, 1]")
    index = 0 if progress < 1.0 / 3.0 else 1 if progress < 2.0 / 3.0 else 2
    return index, path[index].copy()


def semantic_profile_for_anchor(index: int) -> str:
    """Map observable path progress to an existing, frozen VPP intent profile."""

    if int(index) in (0, 1):
        return "range_extension"
    if int(index) == 2:
        return "reentry_preparation"
    raise DynamicReferencePathError("anchor index must be 0, 1, or 2")
