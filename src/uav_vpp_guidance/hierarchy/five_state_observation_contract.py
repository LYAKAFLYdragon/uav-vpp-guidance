"""Versioned observation contract for the five-state shared-intent family."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np


CONTRACT_VERSION = "five_state_shared_intent_v1"
BASE_GEOMETRY_FEATURES: Tuple[str, ...] = (
    "range_m",
    "range_rate_mps",
    "altitude_diff_m",
    "speed_diff_mps",
    "los_azimuth_sin",
    "los_azimuth_cos",
    "los_elevation_sin",
    "los_elevation_cos",
    "ata_sin",
    "ata_cos",
    "aa_sin",
    "aa_cos",
    "own_speed",
    "target_speed",
    "own_altitude",
    "target_altitude",
)
EXPLICIT_HISTORY_FEATURES: Tuple[str, ...] = (
    "range_rate_mean_10",
    "range_rate_slope_per_step_10",
    "aa_delta_deg_10",
    "ata_delta_deg_10",
    "specific_energy_height_delta_m_10",
    "altitude_delta_m_10",
)
INTENT_TARGET_FEATURES: Tuple[str, ...] = (
    "intent_aa_target",
    "intent_ata_target",
    "intent_range_target",
    "intent_closure_target",
    "intent_specific_energy_target",
    "intent_altitude_target",
)
INTENT_WEIGHT_FEATURES: Tuple[str, ...] = tuple(
    name.replace("_target", "_weight") for name in INTENT_TARGET_FEATURES
)


def _require_unique(names: Iterable[str], label: str) -> Tuple[str, ...]:
    normalized = tuple(str(name) for name in names)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{label} contains duplicate feature names")
    return normalized


@dataclass(frozen=True)
class FiveStateObservationContract:
    """One strict input schema shared by all newly trained low-level skills."""

    temporal_embedding_dim: int = 32
    history_window_steps: int = 10

    @property
    def feature_names(self) -> Tuple[str, ...]:
        embedding = tuple(
            f"temporal_embedding_{index:02d}"
            for index in range(self.temporal_embedding_dim)
        )
        return (
            BASE_GEOMETRY_FEATURES
            + EXPLICIT_HISTORY_FEATURES
            + INTENT_TARGET_FEATURES
            + INTENT_WEIGHT_FEATURES
            + embedding
        )

    @property
    def obs_dim(self) -> int:
        return len(self.feature_names)

    def compose(
        self,
        *,
        base_geometry: np.ndarray,
        explicit_history: np.ndarray,
        intent_targets: np.ndarray,
        intent_weights: np.ndarray,
        temporal_embedding: np.ndarray,
    ) -> np.ndarray:
        segments = {
            "base_geometry": (base_geometry, len(BASE_GEOMETRY_FEATURES)),
            "explicit_history": (explicit_history, len(EXPLICIT_HISTORY_FEATURES)),
            "intent_targets": (intent_targets, len(INTENT_TARGET_FEATURES)),
            "intent_weights": (intent_weights, len(INTENT_WEIGHT_FEATURES)),
            "temporal_embedding": (temporal_embedding, self.temporal_embedding_dim),
        }
        values = []
        for name, (segment, expected_dim) in segments.items():
            vector = np.asarray(segment, dtype=np.float32).reshape(-1)
            if vector.shape != (expected_dim,):
                raise ValueError(
                    f"{name} shape mismatch: expected {(expected_dim,)}, got {vector.shape}"
                )
            if not np.all(np.isfinite(vector)):
                raise ValueError(f"{name} contains non-finite values")
            values.append(vector)
        observation = np.concatenate(values, axis=0)
        if observation.shape != (self.obs_dim,):  # Defensive invariant.
            raise AssertionError("Five-state observation dimension drifted")
        return observation


def build_feature_name_transplant_map(
    source_feature_names: Iterable[str],
    target_feature_names: Iterable[str],
) -> Dict[int, int]:
    """Map only identically named feature columns; never infer by position."""

    source = _require_unique(source_feature_names, "source_feature_names")
    target = _require_unique(target_feature_names, "target_feature_names")
    target_index = {name: index for index, name in enumerate(target)}
    return {
        source_index: target_index[name]
        for source_index, name in enumerate(source)
        if name in target_index
    }
