"""
Action-space schema validator for the flight-control comparison tasks.

Centralises the interpretation of policy action vectors so that env.step,
generator, adapters, and configs all speak the same language:

- 3D: ``[dx, dy, dz]`` — standard PPO with virtual pursuit point.
- 4D: ``[dx, dy, dz, aggressiveness]`` — PPO+PID hybrid.
  ``aggressiveness`` in [-1, 1] scales PID gains via
  ``scale = 1.0 + 0.5 * aggressiveness``.
- 6D: APIC-PID per-gain deltas
  ``[delta_Kp_nz, delta_Ki_nz, delta_Kd_nz,
     delta_Kp_roll, delta_Ki_roll, delta_Kd_roll]``.

5D actions are explicitly rejected because no canonical semantics are defined.
"""

from __future__ import annotations

from enum import Enum
from typing import Tuple

import numpy as np


class ActionSchema(Enum):
    """Supported action-space schemas."""

    PPO_3D = 3
    PPO_PID_4D = 4
    APIC_PID_6D = 6

    @property
    def description(self) -> str:
        descriptions = {
            ActionSchema.PPO_3D: "[dx, dy, dz]",
            ActionSchema.PPO_PID_4D: "[dx, dy, dz, aggressiveness]",
            ActionSchema.APIC_PID_6D: "[dKp_nz, dKi_nz, dKd_nz, dKp_roll, dKi_roll, dKd_roll]",
        }
        return descriptions[self]

    @property
    def has_aggressiveness(self) -> bool:
        return self == ActionSchema.PPO_PID_4D

    @property
    def is_apic(self) -> bool:
        return self == ActionSchema.APIC_PID_6D


def infer_action_schema(action: np.ndarray) -> ActionSchema:
    """
    Infer the action schema from the action vector length.

    Raises:
        ValueError: if the length is not one of the supported schemas.
    """
    action = np.asarray(action)
    dim = int(action.shape[0]) if action.ndim == 1 else int(action.shape[-1])
    try:
        return ActionSchema(dim)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported action dimension {dim}. "
            "Supported schemas are 3D [dx,dy,dz], "
            "4D [dx,dy,dz,aggressiveness], and "
            "6D APIC [dKp_nz,dKi_nz,dKd_nz,dKp_roll,dKi_roll,dKd_roll]."
        ) from exc


def validate_action(action, expected_dim: int = None) -> Tuple[np.ndarray, ActionSchema]:
    """
    Validate and normalise an action vector.

    Args:
        action: array-like action.
        expected_dim: if provided, also assert that the inferred schema matches.

    Returns:
        tuple of (action as 1-D float ndarray, inferred ActionSchema).
    """
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    schema = infer_action_schema(action)
    if expected_dim is not None and schema.value != int(expected_dim):
        raise ValueError(
            f"Action dimension {schema.value} does not match expected {expected_dim}."
        )
    return action, schema


def split_action(action: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Split a PPO+PID 4D action into the 3D VPP offset and aggressiveness.

    Returns:
        (vpp_offset, aggressiveness) where aggressiveness is clipped to [-1, 1].
    """
    action = np.asarray(action, dtype=np.float64)
    schema = infer_action_schema(action)
    if schema == ActionSchema.PPO_PID_4D:
        return action[:3].copy(), float(np.clip(action[3], -1.0, 1.0))
    if schema == ActionSchema.PPO_3D:
        return action[:3].copy(), 0.0
    raise ValueError(
        f"Cannot extract aggressiveness from {schema.name} action; "
        "expected 3D or 4D."
    )
