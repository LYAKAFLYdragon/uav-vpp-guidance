"""Compile interpretable relative-frame intents into an existing 3-D VPP action.

The compiler deliberately sits in front of the legacy normalized VPP action
head.  It does not alter the VPP generator, guidance law, PID, or policy.  A
semantic request is first expressed in LOS/target-velocity coordinates and is
then projected back to the configured raw action frame.  This makes the
requested geometry and every clipping decision explicit in telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from uav_vpp_guidance.hierarchy.five_state_observation_contract import (
    BASE_GEOMETRY_FEATURES,
)
from uav_vpp_guidance.hierarchy.shared_intent_profiles import compile_profile
from uav_vpp_guidance.virtual_point.coordinate_transform import (
    offset_frame_basis,
    offset_to_world,
    world_to_offset_frame,
)


RANGE_REFERENCE_M = 5000.0
RANGE_RATE_REFERENCE_MPS = 200.0
ALTITUDE_REFERENCE_M = 10000.0
SPEED_REFERENCE_MPS = 400.0
GRAVITY_MPS2 = 9.80665
SEMANTIC_AXES = ("range_extension", "signed_turn", "vertical_energy")


class RelativeFrameSemanticError(ValueError):
    """Raised when a semantic VPP request cannot be compiled safely."""


def _finite_vector(value: Any, *, name: str, length: int) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as error:
        raise RelativeFrameSemanticError(f"{name} must be numeric") from error
    if result.shape != (length,) or not np.isfinite(result).all():
        raise RelativeFrameSemanticError(f"{name} must be a finite {length}-D vector")
    return result


def _axis_to_world(action: np.ndarray, ranges: Sequence[Sequence[float]], frame: str, own: Mapping[str, Any], target: Mapping[str, Any]) -> np.ndarray:
    if len(ranges) != 3:
        raise RelativeFrameSemanticError("raw_action_ranges must contain three ranges")
    local = []
    for value, limits in zip(action, ranges):
        if len(limits) != 2:
            raise RelativeFrameSemanticError("each raw action range must contain [min, max]")
        lower, upper = float(limits[0]), float(limits[1])
        if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
            raise RelativeFrameSemanticError("raw action ranges must be finite and increasing")
        local.append(lower + (float(value) + 1.0) * (upper - lower) / 2.0)
    return offset_to_world(np.asarray(local, dtype=np.float64), frame, own_state=own, target_state=target)


def _world_to_action(world_offset: np.ndarray, ranges: Sequence[Sequence[float]], frame: str, own: Mapping[str, Any], target: Mapping[str, Any]) -> np.ndarray:
    local = world_to_offset_frame(world_offset, frame, own_state=own, target_state=target)
    action = []
    for value, limits in zip(local, ranges):
        lower, upper = float(limits[0]), float(limits[1])
        action.append(2.0 * (float(value) - lower) / (upper - lower) - 1.0)
    return np.asarray(action, dtype=np.float64)


def _base_values(base_geometry: Any) -> dict[str, float]:
    base = _finite_vector(base_geometry, name="base_geometry", length=16)
    return {name: float(base[index]) for index, name in enumerate(BASE_GEOMETRY_FEATURES)}


def _signed_angle_deg(sine: float, cosine: float) -> float:
    return float(math.degrees(math.atan2(float(sine), float(cosine))))


@dataclass(frozen=True)
class RelativeFrameSemanticConfig:
    """Frozen physical scales for a semantic VPP transfer probe."""

    range_extent_m: float = 1200.0
    turn_extent_m: float = 900.0
    vertical_extent_m: float = 400.0
    range_error_scale_m: float = 2000.0
    range_rate_error_scale_mps: float = 200.0
    turn_error_scale_deg: float = 90.0
    energy_error_scale_m: float = 1000.0
    altitude_error_scale_m: float = 1000.0
    raw_action_offset_frame: str = "world_neu"

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if name == "raw_action_offset_frame":
                continue
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise RelativeFrameSemanticError(f"{name} must be finite and positive")


class RelativeFrameSemanticCompiler:
    """Convert a profile-conditioned relative intent into one bounded VPP action."""

    def __init__(self, config: RelativeFrameSemanticConfig) -> None:
        self.config = config

    def compile(
        self,
        *,
        reference_action: Any,
        base_geometry: Any,
        own_state: Mapping[str, Any],
        target_state: Mapping[str, Any],
        raw_action_ranges: Sequence[Sequence[float]],
        profile_name: str,
        enabled_axes: Mapping[str, bool],
    ) -> dict[str, Any]:
        """Return the action plus a complete receipt for a semantic request.

        Positive ``range_extension`` means "request greater separation": it
        shifts the VPP toward the ego along the LOS, rather than farther past
        the target.  Positive vertical energy means an upward NEU VPP shift.
        """

        reference = _finite_vector(reference_action, name="reference_action", length=3)
        if np.any(np.abs(reference) > 1.0 + 1e-6):
            raise RelativeFrameSemanticError("reference_action is outside [-1, 1]")
        values = _base_values(base_geometry)
        if not isinstance(own_state, Mapping) or not isinstance(target_state, Mapping):
            raise RelativeFrameSemanticError("own_state and target_state must be mappings")
        unknown = set(enabled_axes).difference(SEMANTIC_AXES)
        if unknown:
            raise RelativeFrameSemanticError(f"unknown semantic axes: {sorted(unknown)}")

        profile = compile_profile(profile_name)
        targets = np.asarray(profile["physical_targets"], dtype=np.float64)
        weights = np.asarray(profile["weight_vector"], dtype=np.float64)
        if targets.shape != (6,) or weights.shape != (6,):
            raise RelativeFrameSemanticError("profile compiler returned an invalid intent vector")

        range_m = values["range_m"] * RANGE_REFERENCE_M
        range_rate_mps = values["range_rate_mps"] * RANGE_RATE_REFERENCE_MPS
        own_speed = values["own_speed"] * SPEED_REFERENCE_MPS
        target_speed = values["target_speed"] * SPEED_REFERENCE_MPS
        own_altitude = values["own_altitude"] * ALTITUDE_REFERENCE_M
        target_altitude = values["target_altitude"] * ALTITUDE_REFERENCE_M
        own_energy = own_altitude + own_speed**2 / (2.0 * GRAVITY_MPS2)
        target_energy = target_altitude + target_speed**2 / (2.0 * GRAVITY_MPS2)
        own_minus_target_altitude = own_altitude - target_altitude
        own_minus_target_energy = own_energy - target_energy
        signed_ata_deg = _signed_angle_deg(values["ata_sin"], values["ata_cos"])

        range_components = np.asarray(
            [
                (targets[2] - range_m) / self.config.range_error_scale_m,
                (targets[3] - range_rate_mps) / self.config.range_rate_error_scale_mps,
            ],
            dtype=np.float64,
        )
        range_weights = weights[2:4]
        range_command = float(np.clip(np.average(range_components, weights=range_weights), -1.0, 1.0))
        turn_error = max(0.0, abs(signed_ata_deg) - float(targets[1]))
        turn_command = float(
            -math.copysign(min(turn_error / self.config.turn_error_scale_deg, 1.0), signed_ata_deg)
            if abs(signed_ata_deg) > 1e-9
            else 0.0
        )
        vertical_components = np.asarray(
            [
                (targets[4] - own_minus_target_energy) / self.config.energy_error_scale_m,
                (targets[5] - own_minus_target_altitude) / self.config.altitude_error_scale_m,
            ],
            dtype=np.float64,
        )
        vertical_command = float(np.clip(np.average(vertical_components, weights=weights[4:6]), -1.0, 1.0))

        requested_local = {
            "range_extension": range_command if enabled_axes.get("range_extension", False) else 0.0,
            "signed_turn": turn_command if enabled_axes.get("signed_turn", False) else 0.0,
            "vertical_energy": vertical_command if enabled_axes.get("vertical_energy", False) else 0.0,
        }
        los_basis = offset_frame_basis("los_relative", own_state=own_state, target_state=target_state)
        target_basis = offset_frame_basis("target_velocity", own_state=own_state, target_state=target_state)
        range_world = -requested_local["range_extension"] * self.config.range_extent_m * los_basis[:, 0]
        turn_world = requested_local["signed_turn"] * self.config.turn_extent_m * target_basis[:, 1]
        vertical_world = np.array([0.0, 0.0, requested_local["vertical_energy"] * self.config.vertical_extent_m], dtype=np.float64)
        requested_semantic_world = range_world + turn_world + vertical_world

        reference_world = _axis_to_world(reference, raw_action_ranges, self.config.raw_action_offset_frame, own_state, target_state)
        requested_world = reference_world + requested_semantic_world
        unclipped_action = _world_to_action(requested_world, raw_action_ranges, self.config.raw_action_offset_frame, own_state, target_state)
        applied_action = np.clip(unclipped_action, -1.0, 1.0)
        applied_world = _axis_to_world(applied_action, raw_action_ranges, self.config.raw_action_offset_frame, own_state, target_state)
        actual_semantic_world = applied_world - reference_world

        return {
            "profile": str(profile_name),
            "enabled_axes": {axis: bool(enabled_axes.get(axis, False)) for axis in SEMANTIC_AXES},
            "reference_action": reference.tolist(),
            "applied_action": applied_action.tolist(),
            "unclipped_action": unclipped_action.tolist(),
            "clipped": bool(not np.allclose(unclipped_action, applied_action, rtol=0.0, atol=1e-8)),
            "state": {
                "range_m": range_m,
                "range_rate_mps": range_rate_mps,
                "signed_ata_deg": signed_ata_deg,
                "own_minus_target_energy_m": own_minus_target_energy,
                "own_minus_target_altitude_m": own_minus_target_altitude,
            },
            "commands": {
                "range_extension": requested_local["range_extension"],
                "signed_turn": requested_local["signed_turn"],
                "vertical_energy": requested_local["vertical_energy"],
            },
            "reference_world_offset_m": reference_world.tolist(),
            "requested_semantic_world_delta_m": requested_semantic_world.tolist(),
            "actual_semantic_world_delta_m": actual_semantic_world.tolist(),
            "requested_semantic_delta_norm_m": float(np.linalg.norm(requested_semantic_world)),
            "actual_semantic_delta_norm_m": float(np.linalg.norm(actual_semantic_world)),
        }
