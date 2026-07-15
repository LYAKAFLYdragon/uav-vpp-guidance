"""Explicit legacy-observation projections for frozen baseline specialists.

The shared-intent research lane never permits ``FrozenSpecialistPolicy`` to
silently truncate or pad an environment vector.  A legacy crossing checkpoint
is the sole approved exception: it expects the 18 features preceding the
newer task bit ``is_crossing``.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np


CROSSING_LEGACY_FEATURES = (
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
    "gain_k_los",
    "gain_k_pos",
)


class FrozenSpecialistProjectionError(ValueError):
    """Raised when a frozen specialist has no declared observation adapter."""


def _names(feature_names: Iterable[Any]) -> tuple[str, ...]:
    names = tuple(str(item) for item in feature_names)
    if len(names) != len(set(names)):
        raise FrozenSpecialistProjectionError("environment feature names must be unique")
    return names


def project_crossing_observation(
    observation: Any, feature_names: Iterable[Any], checkpoint_obs_dim: int
) -> tuple[np.ndarray, dict[str, Any]]:
    """Project the 19-D V5 environment vector into the declared 18-D legacy view."""

    names = _names(feature_names)
    values = np.asarray(observation, dtype=np.float32).reshape(-1)
    if values.shape != (len(names),) or not np.isfinite(values).all():
        raise FrozenSpecialistProjectionError("environment observation is not finite and schema-aligned")
    if int(checkpoint_obs_dim) != len(CROSSING_LEGACY_FEATURES):
        raise FrozenSpecialistProjectionError("crossing checkpoint must declare the frozen 18-D contract")
    expected_environment = CROSSING_LEGACY_FEATURES + ("is_crossing",)
    if names != expected_environment:
        raise FrozenSpecialistProjectionError(
            "crossing projection accepts only the declared 19-D V5 feature order"
        )
    projected = values[: len(CROSSING_LEGACY_FEATURES)].copy()
    return projected, {
        "adapter": "explicit_drop_is_crossing_task_bit_v1",
        "source_feature_names": list(names),
        "target_feature_names": list(CROSSING_LEGACY_FEATURES),
        "dropped_feature_names": ["is_crossing"],
        "source_dim": len(names),
        "target_dim": int(projected.shape[0]),
        "implicit_truncation_or_padding": False,
    }


def action_for_frozen_specialist(
    policy: Any, observation: Any, observation_schema: Mapping[str, Any], method: str
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return an action only when the frozen policy receives its exact input width."""

    names = _names(observation_schema.get("feature_names", ()))
    values = np.asarray(observation, dtype=np.float32).reshape(-1)
    if method == "frozen_fixed_head_on":
        if int(getattr(policy, "obs_dim", -1)) != len(names) or values.shape != (len(names),):
            raise FrozenSpecialistProjectionError("head-on checkpoint must receive the exact environment schema")
        forwarded = values
        receipt = {
            "adapter": "identity_exact_schema_v1",
            "source_feature_names": list(names),
            "target_feature_names": list(names),
            "dropped_feature_names": [],
            "source_dim": len(names),
            "target_dim": len(names),
            "implicit_truncation_or_padding": False,
        }
    elif method == "frozen_fixed_crossing":
        forwarded, receipt = project_crossing_observation(values, names, int(getattr(policy, "obs_dim", -1)))
    else:
        raise FrozenSpecialistProjectionError("method has no approved frozen-specialist projection")
    action = np.asarray(policy.get_deterministic_action(forwarded), dtype=np.float32).reshape(-1)
    if action.shape != (3,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1.0 + 1e-6):
        raise FrozenSpecialistProjectionError("frozen specialist returned an invalid normalized 3-D action")
    return action, receipt
