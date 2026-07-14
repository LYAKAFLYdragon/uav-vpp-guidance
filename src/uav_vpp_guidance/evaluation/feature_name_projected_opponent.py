"""Feature-name projection adapter for frozen embedded opponents.

The environment may expose ego policies with gains and task bits while older
embedded opponent checkpoints accept only the base 16-D role-reversed combat
observation.  This adapter performs an explicit, audited projection by feature
name.  It intentionally rejects missing, duplicated, or reordered contracts
instead of silently slicing or padding the vector.
"""

from __future__ import annotations

import importlib
from typing import Any, Mapping, Sequence

import numpy as np

from ..envs.opponent_policy import OpponentPolicy


BASE_GEOMETRY_FEATURE_NAMES: tuple[str, ...] = (
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


def project_role_reversed_base_observation(
    opponent_obs: Mapping[str, Any],
    *,
    expected_feature_names: Sequence[str] = BASE_GEOMETRY_FEATURE_NAMES,
) -> dict[str, Any]:
    """Return an exact role-reversed base-geometry projection.

    The caller owns the role reversal.  This function only selects declared
    feature names, preserving their requested order and all non-vector fields.
    """

    schema = dict(opponent_obs.get("observation_schema") or {})
    if not bool(schema.get("role_reversed", False)):
        raise ValueError("Projected opponent requires a role-reversed observation")

    feature_names = [str(name) for name in schema.get("feature_names", [])]
    vector = np.asarray(opponent_obs.get("observation_vector"), dtype=np.float32).reshape(-1)
    if len(feature_names) != vector.shape[0]:
        raise ValueError(
            "Opponent observation feature-name/vector mismatch: "
            f"names={len(feature_names)}, vector={vector.shape[0]}"
        )
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("Opponent observation feature names must be unique")

    expected = tuple(str(name) for name in expected_feature_names)
    if len(expected) != len(set(expected)):
        raise ValueError("Expected opponent feature names must be unique")
    positions = {name: index for index, name in enumerate(feature_names)}
    missing = [name for name in expected if name not in positions]
    if missing:
        raise ValueError(
            "Opponent base-geometry projection is missing required features: "
            + ", ".join(missing)
        )

    projected = np.asarray([vector[positions[name]] for name in expected], dtype=np.float32)
    if projected.shape != (len(expected),) or not np.all(np.isfinite(projected)):
        raise ValueError("Projected opponent observation is malformed or non-finite")

    result = dict(opponent_obs)
    result["observation_vector"] = projected
    result["observation_schema"] = {
        "dim": int(projected.shape[0]),
        "feature_names": list(expected),
        "role_reversed": True,
        "adapter": "feature_name_base16_projection",
    }
    return result


def _import_class(class_path: str):
    normalized = str(class_path)
    if normalized.startswith("src."):
        normalized = normalized[len("src.") :]
    module_name, class_name = normalized.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


class FeatureNameProjectedOpponent(OpponentPolicy):
    """Wrap a frozen 16-D opponent without changing the global environment."""

    def __init__(
        self,
        checkpoint_path: str,
        config: Mapping[str, Any] | None = None,
        device: str = "cpu",
        invert_observation: bool | None = None,
    ):
        adapter_config = dict(config or {})
        delegate_class_path = adapter_config.get("delegate_class")
        if not delegate_class_path:
            raise ValueError("FeatureNameProjectedOpponent requires delegate_class")
        delegate_class = _import_class(str(delegate_class_path))
        delegate_kwargs = dict(adapter_config.get("delegate_kwargs") or {})
        delegate_kwargs.update(
            {
                "checkpoint_path": str(checkpoint_path),
                "config": dict(adapter_config.get("delegate_config") or {}),
                "device": str(device),
            }
        )
        self.delegate = delegate_class(**delegate_kwargs)
        self.expected_feature_names = tuple(
            str(name)
            for name in adapter_config.get("expected_feature_names", BASE_GEOMETRY_FEATURE_NAMES)
        )
        if self.expected_feature_names != BASE_GEOMETRY_FEATURE_NAMES:
            raise ValueError("Held-out opponent adapter must use the frozen base 16-D contract")
        self.action_mode = str(getattr(self.delegate, "action_mode", "direct_command"))
        self.checkpoint_path = str(checkpoint_path)
        # The comparison runner forwards legacy registry flags. Role reversal is
        # still enforced from the runtime schema rather than trusted here.
        self.registry_invert_observation = invert_observation
        self._projection_count = 0

    def reset(self) -> None:
        self.delegate.reset()

    def act(self, opponent_obs: Mapping[str, Any]) -> np.ndarray:
        projected = project_role_reversed_base_observation(
            opponent_obs,
            expected_feature_names=self.expected_feature_names,
        )
        self._projection_count += 1
        action = np.asarray(self.delegate.act(projected), dtype=np.float32)
        if action.shape != (3,) or not np.all(np.isfinite(action)):
            raise ValueError("Projected opponent delegate returned a malformed action")
        return action

    def get_diagnostics(self) -> dict[str, Any]:
        diagnostics = dict(self.delegate.get_diagnostics())
        diagnostics.update(
            {
                "opponent_adapter": "feature_name_base16_projection",
                "opponent_adapter_expected_feature_names": list(self.expected_feature_names),
                "opponent_adapter_projection_count": int(self._projection_count),
                "opponent_adapter_checkpoint": self.checkpoint_path,
                "opponent_adapter_registry_invert_observation": self.registry_invert_observation,
            }
        )
        return diagnostics
