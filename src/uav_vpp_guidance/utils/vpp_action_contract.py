"""Canonical, versioned action contract for virtual-pursuit-point policies."""
from __future__ import annotations
from typing import Mapping
import numpy as np

VPP_ACTION_DIMENSION = 3
VPP_ACTION_SCHEMA_VERSION = "vpp-action-contract-1"
LEGACY_5D_COMPATIBILITY_MODE = "vpp_5d_legacy"


def resolve_vpp_action_dimension(config: Mapping | None) -> int:
    """Require the canonical VPP dimension; never infer a silent fallback."""
    config = config or {}
    if "action_dim" not in config:
        raise ValueError("virtual_point.action_dim is required by the F07 canonical VPP action contract; silent fallback is prohibited.")
    dimension = int(config["action_dim"])
    if dimension != VPP_ACTION_DIMENSION:
        raise ValueError(f"virtual_point.action_dim={dimension} is incompatible with canonical VPP dimension {VPP_ACTION_DIMENSION}; use a separately approved migration.")
    return dimension


def validate_vpp_action(action, *, expected_dim: int = VPP_ACTION_DIMENSION, legacy_compatibility_mode: str | None = None) -> np.ndarray:
    """Validate a VPP policy action before any action-to-offset interpretation."""
    values = np.asarray(action, dtype=np.float64).reshape(-1)
    if values.size == expected_dim:
        return values
    if legacy_compatibility_mode == LEGACY_5D_COMPATIBILITY_MODE and values.size == 5:
        return values
    raise ValueError(f"VPP action dimension {values.size} does not match canonical dimension {expected_dim}. Set legacy_compatibility_mode={LEGACY_5D_COMPATIBILITY_MODE!r} only for an explicitly versioned legacy artifact.")


def validate_vpp_checkpoint_metadata(checkpoint: Mapping, *, expected_dim: int) -> int:
    """Reject checkpoints with absent, stale, or incompatible action metadata."""
    if "action_dim" not in checkpoint:
        raise ValueError("VPP checkpoint metadata missing action_dim; legacy fallback is prohibited.")
    dimension = int(checkpoint["action_dim"])
    if dimension != int(expected_dim):
        raise ValueError(f"VPP checkpoint action_dim={dimension} does not match expected canonical dimension {expected_dim}.")
    return dimension
