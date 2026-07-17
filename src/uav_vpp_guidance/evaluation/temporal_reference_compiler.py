"""Bounded C1 temporal VPP-reference programs for reachability screening.

This module intentionally differs from the earlier raw-frame corridor designs:
it compiles a finite VPP-reference *program* through the deployed
``tactical_basis_v1`` action semantics, relative to a frozen specialist action.
It does not learn, inspect simulator future state, alter the VPP/guidance/PID
chain, or expand the instantaneous action box.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


TEMPLATE_NAMES = (
    "lateral_extension",
    "vertical_extension",
    "combined_break_extension",
)


class TemporalReferenceCompilerError(ValueError):
    """Raised when a temporal reference violates its frozen interface contract."""


def _vector(value: Any, *, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as error:
        raise TemporalReferenceCompilerError(f"{name} must be a finite 3-D vector") from error
    if result.shape != (3,) or not np.isfinite(result).all():
        raise TemporalReferenceCompilerError(f"{name} must be a finite 3-D vector")
    return result


def _basis_matrix(value: Sequence[Any], *, max_condition_number: float) -> tuple[np.ndarray, int, float]:
    if len(value) != 3:
        raise TemporalReferenceCompilerError("tactical basis requires exactly three world vectors")
    basis = np.column_stack(
        [_vector(vector, name=f"basis_vectors_world_m[{index}]") for index, vector in enumerate(value)]
    )
    rank = int(np.linalg.matrix_rank(basis))
    condition = float(np.linalg.cond(basis)) if rank == 3 else float("inf")
    if rank != 3 or not np.isfinite(condition) or condition > max_condition_number:
        raise TemporalReferenceCompilerError(
            "tactical basis is not a finite rank-3 envelope: "
            f"rank={rank}, condition={condition}"
        )
    return basis, rank, condition


@dataclass(frozen=True)
class TemporalReferenceConfig:
    """Frozen envelope for one current-information temporal reference program."""

    horizon_steps: int = 40
    high_level_dt_s: float = 0.2
    max_action_delta: float = 0.35
    max_action_slew: float = 0.10
    max_basis_condition_number: float = 100.0

    def __post_init__(self) -> None:
        if self.horizon_steps < 4:
            raise TemporalReferenceCompilerError("horizon_steps must be at least four")
        for name in ("high_level_dt_s", "max_action_delta", "max_action_slew", "max_basis_condition_number"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise TemporalReferenceCompilerError(f"{name} must be finite and positive")
        if self.max_action_delta > 1.0:
            raise TemporalReferenceCompilerError("max_action_delta must not expand the normalized action box")
        if self.max_action_slew > 2.0:
            raise TemporalReferenceCompilerError("max_action_slew cannot exceed the normalized action-box diameter")


@dataclass(frozen=True)
class TemporalReferenceTemplate:
    """Peak tactical-basis action delta for a named, interpretable template."""

    name: str
    peak_action_delta: tuple[float, float, float]

    def __post_init__(self) -> None:
        if self.name not in TEMPLATE_NAMES:
            raise TemporalReferenceCompilerError(f"unknown temporal-reference template: {self.name}")
        vector = _vector(self.peak_action_delta, name=f"{self.name}.peak_action_delta")
        object.__setattr__(self, "peak_action_delta", tuple(float(item) for item in vector))

    @classmethod
    def from_mapping(cls, name: str, value: Mapping[str, Any]) -> "TemporalReferenceTemplate":
        if not isinstance(value, Mapping):
            raise TemporalReferenceCompilerError(f"template {name} must be a mapping")
        return cls(name=str(name), peak_action_delta=tuple(_vector(value.get("peak_action_delta"), name=f"{name}.peak_action_delta")))


def c1_window(step_index: int, config: TemporalReferenceConfig) -> float:
    """Return the C1 zero-endpoint sine-squared window for one program step."""

    if not 0 <= int(step_index) < config.horizon_steps:
        raise TemporalReferenceCompilerError("step_index is outside the frozen reference horizon")
    progress = float(step_index) / float(config.horizon_steps - 1)
    return float(np.sin(np.pi * progress) ** 2)


def _signed_peak(
    template: TemporalReferenceTemplate,
    *,
    lateral_sign: int,
    vertical_sign: int,
    config: TemporalReferenceConfig,
) -> np.ndarray:
    if lateral_sign not in (-1, 1) or vertical_sign not in (-1, 1):
        raise TemporalReferenceCompilerError("lateral_sign and vertical_sign must each be -1 or 1")
    peak = _vector(template.peak_action_delta, name=f"{template.name}.peak_action_delta")
    peak[1] *= float(lateral_sign)
    peak[2] *= float(vertical_sign)
    if np.any(np.abs(peak) > config.max_action_delta + 1e-12):
        raise TemporalReferenceCompilerError(
            f"{template.name} exceeds the frozen max_action_delta={config.max_action_delta}"
        )
    return peak


def compile_temporal_reference_step(
    *,
    template: TemporalReferenceTemplate,
    step_index: int,
    reference_action: Any,
    previous_applied_action: Any,
    basis_vectors_world_m: Sequence[Any],
    config: TemporalReferenceConfig,
    lateral_sign: int = 1,
    vertical_sign: int = 1,
) -> dict[str, Any]:
    """Compile one C1 reference knot into the frozen tactical-basis action box.

    ``reference_action`` must come from the frozen head-on specialist at the
    same simulator step.  The compiler reads no future state; the caller may
    rebuild the active basis from the current state at every 0.2-s step.
    """

    reference = _vector(reference_action, name="reference_action")
    previous = _vector(previous_applied_action, name="previous_applied_action")
    if np.any(np.abs(reference) > 1.0 + 1e-8) or np.any(np.abs(previous) > 1.0 + 1e-8):
        raise TemporalReferenceCompilerError("reference and previous actions must lie in [-1, 1]")
    basis, rank, condition = _basis_matrix(
        basis_vectors_world_m,
        max_condition_number=config.max_basis_condition_number,
    )
    window = c1_window(step_index, config)
    requested_delta = window * _signed_peak(
        template,
        lateral_sign=lateral_sign,
        vertical_sign=vertical_sign,
        config=config,
    )
    requested_action = reference + requested_delta
    box_projected_action = np.clip(requested_action, -1.0, 1.0)
    slew_lower = previous - config.max_action_slew
    slew_upper = previous + config.max_action_slew
    applied_action = np.minimum(np.maximum(box_projected_action, slew_lower), slew_upper)
    applied_action = np.clip(applied_action, -1.0, 1.0)
    actual_delta = applied_action - reference
    requested_world = basis @ requested_delta
    actual_world = basis @ actual_delta
    residual = requested_world - actual_world
    return {
        "template": template.name,
        "step_index": int(step_index),
        "program_horizon_steps": config.horizon_steps,
        "program_duration_s": float(config.horizon_steps * config.high_level_dt_s),
        "temporal_profile": "sin_squared_c1",
        "c1_window": window,
        "lateral_sign": int(lateral_sign),
        "vertical_sign": int(vertical_sign),
        "uses_future_state": False,
        "basis_rank": rank,
        "basis_condition_number": condition,
        "basis_vectors_world_m": basis.T.tolist(),
        "reference_action": reference.tolist(),
        "requested_action_delta": requested_delta.tolist(),
        "requested_action": requested_action.tolist(),
        "box_projected_action": box_projected_action.tolist(),
        "previous_applied_action": previous.tolist(),
        "applied_action": applied_action.tolist(),
        "actual_action_delta": actual_delta.tolist(),
        "requested_reference_world_delta_m": requested_world.tolist(),
        "actual_reference_world_delta_m": actual_world.tolist(),
        "projection_residual_world_m": residual.tolist(),
        "projection_residual_norm_m": float(np.linalg.norm(residual)),
        "action_box_limited": bool(
            not np.allclose(requested_action, box_projected_action, rtol=0.0, atol=1e-10)
        ),
        "slew_limited": bool(
            not np.allclose(box_projected_action, applied_action, rtol=0.0, atol=1e-10)
        ),
    }


def compile_temporal_reference_program(
    *,
    template: TemporalReferenceTemplate,
    reference_actions: Sequence[Any],
    basis_vectors_by_step: Sequence[Sequence[Any]],
    config: TemporalReferenceConfig,
    initial_applied_action: Any | None = None,
    lateral_sign: int = 1,
    vertical_sign: int = 1,
) -> list[dict[str, Any]]:
    """Compile a full finite program for offline P2 envelope analysis only.

    The runtime counterpart must call :func:`compile_temporal_reference_step`
    with the current state-derived basis.  This sequence helper is deliberately
    explicit about receiving a recorded basis per step, so it cannot hide any
    future-state dependency in the online control contract.
    """

    if len(reference_actions) != config.horizon_steps or len(basis_vectors_by_step) != config.horizon_steps:
        raise TemporalReferenceCompilerError("program inputs must equal the frozen horizon_steps")
    previous = _vector(
        reference_actions[0] if initial_applied_action is None else initial_applied_action,
        name="initial_applied_action",
    )
    receipts: list[dict[str, Any]] = []
    for step_index, (reference, basis) in enumerate(zip(reference_actions, basis_vectors_by_step)):
        receipt = compile_temporal_reference_step(
            template=template,
            step_index=step_index,
            reference_action=reference,
            previous_applied_action=previous,
            basis_vectors_world_m=basis,
            config=config,
            lateral_sign=lateral_sign,
            vertical_sign=vertical_sign,
        )
        receipts.append(receipt)
        previous = np.asarray(receipt["applied_action"], dtype=np.float64)
    return receipts
