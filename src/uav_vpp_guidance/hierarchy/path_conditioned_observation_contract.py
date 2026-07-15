"""Strict path-conditioned v2 observation contract for the five-state family.

V1's 66-D contract remains frozen.  This separate contract adds only current
dynamic-reference-path context derived from the frozen predictor and current
geometry; it never exposes a task id, opponent id, future simulator state, or
the policy's own not-yet-selected action.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

import numpy as np

from .five_state_observation_contract import FiveStateObservationContract


CONTRACT_VERSION = "five_state_shared_intent_v2_path_context"
PATH_CONTEXT_FEATURES: Tuple[str, ...] = (
    "path_delta_forward_norm_clipped2",
    "path_delta_lateral_norm_clipped2",
    "path_delta_vertical_norm_clipped2",
    "path_progress",
    "path_anchor_escape",
    "path_anchor_connector",
    "path_anchor_predicted_reentry",
    "path_delta_exceeds_vpp_envelope",
)
PATH_CONTEXT_DIM = len(PATH_CONTEXT_FEATURES)
VPP_WORLD_SCALES_M = np.asarray((1500.0, 800.0, 500.0), dtype=np.float32)


class PathConditionedObservationError(ValueError):
    """Raised when a path context leaves the v2 no-leakage contract."""


def _finite_vector(value: Any, *, shape: tuple[int, ...], label: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError) as error:
        raise PathConditionedObservationError(f"{label} must be numeric") from error
    if result.shape != shape or not np.isfinite(result).all():
        raise PathConditionedObservationError(f"{label} must be finite with shape {shape}")
    return result


def build_path_context(
    *,
    requested_path_world_delta_m: Any,
    path_progress: float,
    selected_anchor_index: int,
) -> np.ndarray:
    """Encode current path geometry without consuming policy action or future state.

    The first three values retain direction and bounded overshoot (up to twice
    the frozen VPP envelope); the final binary value tells a new policy that
    the desired path point cannot be represented exactly by the current head.
    """

    delta = _finite_vector(
        requested_path_world_delta_m,
        shape=(3,),
        label="requested_path_world_delta_m",
    )
    progress = float(path_progress)
    index = int(selected_anchor_index)
    if not np.isfinite(progress) or not 0.0 <= progress <= 1.0:
        raise PathConditionedObservationError("path_progress must lie in [0, 1]")
    if index not in (0, 1, 2):
        raise PathConditionedObservationError("selected_anchor_index must be 0, 1, or 2")
    normalized = delta / VPP_WORLD_SCALES_M
    context = np.concatenate(
        (
            np.clip(normalized, -2.0, 2.0),
            np.asarray((progress,), dtype=np.float32),
            np.eye(3, dtype=np.float32)[index],
            np.asarray((float(np.any(np.abs(normalized) > 1.0)),), dtype=np.float32),
        )
    )
    if context.shape != (PATH_CONTEXT_DIM,) or not np.isfinite(context).all():
        raise AssertionError("path-context dimension drifted")
    return context.astype(np.float32, copy=False)


@dataclass(frozen=True)
class PathConditionedObservationContract:
    """Versioned 74-D contract for newly trained path-conditioned policies only."""

    base_contract: FiveStateObservationContract = FiveStateObservationContract()

    @property
    def feature_names(self) -> Tuple[str, ...]:
        return self.base_contract.feature_names + PATH_CONTEXT_FEATURES

    @property
    def obs_dim(self) -> int:
        return len(self.feature_names)

    def compose(
        self,
        *,
        base_geometry: Any,
        explicit_history: Any,
        intent_targets: Any,
        intent_weights: Any,
        temporal_embedding: Any,
        path_context: Any,
    ) -> np.ndarray:
        """Join a frozen V1 prefix with the auditable current-only path segment."""

        prefix = self.base_contract.compose(
            base_geometry=np.asarray(base_geometry, dtype=np.float32),
            explicit_history=np.asarray(explicit_history, dtype=np.float32),
            intent_targets=np.asarray(intent_targets, dtype=np.float32),
            intent_weights=np.asarray(intent_weights, dtype=np.float32),
            temporal_embedding=np.asarray(temporal_embedding, dtype=np.float32),
        )
        context = _finite_vector(path_context, shape=(PATH_CONTEXT_DIM,), label="path_context")
        value = np.concatenate((prefix, context), axis=0)
        if value.shape != (self.obs_dim,) or not np.isfinite(value).all():
            raise AssertionError("path-conditioned observation dimension drifted")
        return value

    def compose_from_path_receipt(
        self,
        *,
        base_geometry: Any,
        explicit_history: Any,
        intent_targets: Any,
        intent_weights: Any,
        temporal_embedding: Any,
        path_action_receipt: Mapping[str, Any],
        path_progress: float,
        selected_anchor_index: int,
    ) -> np.ndarray:
        """Compose from the pre-action V16-compatible receipt, never post-action telemetry."""

        if not isinstance(path_action_receipt, Mapping):
            raise PathConditionedObservationError("path_action_receipt must be a mapping")
        context = build_path_context(
            requested_path_world_delta_m=path_action_receipt.get("requested_path_world_delta_m"),
            path_progress=path_progress,
            selected_anchor_index=selected_anchor_index,
        )
        return self.compose(
            base_geometry=base_geometry,
            explicit_history=explicit_history,
            intent_targets=intent_targets,
            intent_weights=intent_weights,
            temporal_embedding=temporal_embedding,
            path_context=context,
        )
