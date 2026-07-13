"""Global tactical-intent profile compiler for five-state v1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


PROFILE_NAMES: Tuple[str, ...] = (
    "front_intercept",
    "rear_quarter_alignment",
    "lateral_displacement",
    "defensive_break",
    "range_extension",
    "energy_altitude_recovery",
    "reentry_preparation",
)
INTENT_FIELDS: Tuple[str, ...] = (
    "aa_target_deg",
    "ata_target_deg",
    "range_target_m",
    "range_rate_target_mps",
    "specific_energy_delta_m",
    "altitude_delta_m",
)


@dataclass(frozen=True)
class TacticalIntentProfile:
    name: str
    targets: Tuple[float, ...]
    weights: Tuple[float, ...]

    def compile(self) -> Dict[str, np.ndarray]:
        targets = np.asarray(self.targets, dtype=np.float32)
        weights = np.asarray(self.weights, dtype=np.float32)
        if targets.shape != (len(INTENT_FIELDS),) or weights.shape != targets.shape:
            raise ValueError(f"Invalid profile shape for {self.name}")
        if not np.all(np.isfinite(targets)) or not np.all(np.isfinite(weights)):
            raise ValueError(f"Non-finite intent values for {self.name}")
        if np.any(weights < 0.0) or np.any(weights > 1.0):
            raise ValueError(f"Intent weights must be in [0, 1] for {self.name}")
        scales = np.asarray([180.0, 180.0, 5000.0, 200.0, 1000.0, 1000.0])
        return {
            "target_vector": np.clip(targets / scales, -1.0, 1.0),
            "weight_vector": weights,
            "physical_targets": targets,
        }


_PROFILE_TABLE = {
    "front_intercept": TacticalIntentProfile(
        "front_intercept", (10.0, 20.0, 900.0, -100.0, 0.0, 0.0), (1, 1, 1, 1, 0.3, 0.2)
    ),
    "rear_quarter_alignment": TacticalIntentProfile(
        "rear_quarter_alignment", (15.0, 160.0, 750.0, -40.0, 0.0, 0.0), (1, 1, 1, 0.8, 0.3, 0.2)
    ),
    "lateral_displacement": TacticalIntentProfile(
        "lateral_displacement", (45.0, 90.0, 1500.0, 0.0, 0.0, 0.0), (0.7, 0.7, 0.8, 0.5, 0.4, 0.3)
    ),
    "defensive_break": TacticalIntentProfile(
        "defensive_break", (80.0, 20.0, 2500.0, 80.0, 150.0, 200.0), (0.7, 0.8, 1, 1, 0.8, 0.6)
    ),
    "range_extension": TacticalIntentProfile(
        "range_extension", (70.0, 10.0, 4000.0, 120.0, 200.0, 100.0), (0.5, 0.7, 1, 1, 0.8, 0.5)
    ),
    "energy_altitude_recovery": TacticalIntentProfile(
        "energy_altitude_recovery", (60.0, 60.0, 2800.0, 20.0, 300.0, 300.0), (0.4, 0.4, 0.7, 0.5, 1, 1)
    ),
    "reentry_preparation": TacticalIntentProfile(
        "reentry_preparation", (30.0, 130.0, 2200.0, -30.0, 100.0, 0.0), (0.8, 0.8, 1, 0.8, 0.7, 0.4)
    ),
}


def get_profile(name: str) -> TacticalIntentProfile:
    try:
        return _PROFILE_TABLE[str(name)]
    except KeyError as exc:
        raise KeyError(f"Unknown tactical-intent profile: {name}") from exc


def compile_profile(name: str) -> Dict[str, np.ndarray]:
    return get_profile(name).compile()
