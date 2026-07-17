"""Bounded, non-learning VPP motif pulses for capability discrimination.

The policy keeps the deployed tactical-basis VPP action semantics intact.  It
uses a frozen head-on specialist as a reference and overlays one finite
world-frame geometry request through the *active* tactical basis.  The full
projection receipt is emitted with each step so an evaluator can distinguish
an unavailable interface direction from a geometrically ineffective command.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from uav_vpp_guidance.hierarchy.specialist_policy import FrozenSpecialistPolicy


MOTIF_NAMES = (
    "lateral_break_same_side",
    "range_extension_los",
    "vertical_separation",
)


class TacticalBasisMotifError(ValueError):
    """Raised when a motif cannot be represented by the active VPP basis."""


def _vector(value: Any, *, name: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as error:
        raise TacticalBasisMotifError(f"{name} must be a finite 3-D vector") from error
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise TacticalBasisMotifError(f"{name} must be a finite 3-D vector")
    return vector


def _position(state: Mapping[str, Any], *, name: str) -> np.ndarray:
    for key in ("position_neu", "position_m", "position"):
        if key in state:
            return _vector(state[key], name=f"{name}.{key}")
    raise TacticalBasisMotifError(f"{name} has no 3-D position")


@dataclass(frozen=True)
class MotifPulseConfig:
    """Frozen bounded pulse geometry for the Plan 2 non-learning probe."""

    warmup_high_level_steps: int = 10
    intervention_high_level_steps: int = 20
    ramp_high_level_steps: int = 3
    max_action_delta: float = 0.35
    range_extension_extent_m: float = 700.0
    vertical_separation_extent_m: float = 300.0
    max_basis_condition_number: float = 100.0

    def __post_init__(self) -> None:
        if self.warmup_high_level_steps < 1:
            raise TacticalBasisMotifError("warmup_high_level_steps must be positive")
        if self.intervention_high_level_steps < 3:
            raise TacticalBasisMotifError("intervention_high_level_steps must be at least three")
        if not 1 <= self.ramp_high_level_steps <= self.intervention_high_level_steps // 2:
            raise TacticalBasisMotifError("ramp_high_level_steps is incompatible with intervention length")
        if not 0.0 < self.max_action_delta <= 1.0:
            raise TacticalBasisMotifError("max_action_delta must lie in (0, 1]")
        for name in (
            "range_extension_extent_m",
            "vertical_separation_extent_m",
            "max_basis_condition_number",
        ):
            if not np.isfinite(float(getattr(self, name))) or float(getattr(self, name)) <= 0.0:
                raise TacticalBasisMotifError(f"{name} must be finite and positive")


def _pulse_scale(step: int, config: MotifPulseConfig) -> float:
    """Return the predeclared trapezoidal scale for one intervention step."""

    if not 1 <= int(step) <= config.intervention_high_level_steps:
        raise TacticalBasisMotifError("intervention step is outside the configured pulse")
    if step <= config.ramp_high_level_steps:
        return float(step) / float(config.ramp_high_level_steps)
    ramp_down_start = config.intervention_high_level_steps - config.ramp_high_level_steps + 1
    if step >= ramp_down_start:
        return float(config.intervention_high_level_steps - step + 1) / float(
            config.ramp_high_level_steps
        )
    return 1.0


def _basis_matrix(basis_vectors: Sequence[Any], config: MotifPulseConfig) -> tuple[np.ndarray, int, float]:
    if len(basis_vectors) != 3:
        raise TacticalBasisMotifError("tactical basis requires exactly three vectors")
    basis = np.column_stack(
        [_vector(vector, name=f"basis[{index}]") for index, vector in enumerate(basis_vectors)]
    )
    # This is always a 3-D basis.  Avoid an SVD here: the runtime contract
    # only needs an invertible basis and the residual of its explicit inverse.
    # The direct solve is also the operation used to compile the action below.
    try:
        inverse = np.linalg.inv(basis)
    except np.linalg.LinAlgError:
        inverse = None
    rank = 3 if inverse is not None else 0
    condition = (
        float(np.linalg.norm(basis, ord=np.inf) * np.linalg.norm(inverse, ord=np.inf))
        if inverse is not None
        else float("inf")
    )
    inverse_residual = (
        float(np.linalg.norm(basis @ inverse - np.eye(3), ord=np.inf))
        if inverse is not None
        else float("inf")
    )
    if (
        rank != 3
        or not np.isfinite(condition)
        or condition > config.max_basis_condition_number
        or not np.isfinite(inverse_residual)
        or inverse_residual > 1e-8
    ):
        raise TacticalBasisMotifError(
            f"active tactical basis is not well-conditioned rank-3: rank={rank}, condition={condition}"
        )
    return basis, rank, condition


def _desired_world_delta(
    *,
    motif: str,
    basis: np.ndarray,
    own_state: Mapping[str, Any],
    target_state: Mapping[str, Any],
    scale: float,
    config: MotifPulseConfig,
) -> np.ndarray:
    if motif == "lateral_break_same_side":
        # The active IO vector already carries the encounter-stable same-side sign.
        return scale * config.max_action_delta * basis[:, 1]
    own_position = _position(own_state, name="own_state")
    target_position = _position(target_state, name="target_state")
    if motif == "range_extension_los":
        toward_ego = own_position - target_position
        norm = float(np.linalg.norm(toward_ego))
        if norm <= 1e-8:
            raise TacticalBasisMotifError("LOS direction is undefined")
        return scale * config.range_extension_extent_m * toward_ego / norm
    if motif == "vertical_separation":
        direction = float(np.sign(own_position[2] - target_position[2]))
        if direction == 0.0:
            direction = 1.0
        return np.array(
            [0.0, 0.0, scale * direction * config.vertical_separation_extent_m],
            dtype=np.float64,
        )
    raise TacticalBasisMotifError(f"unknown motif: {motif}")


def compile_tactical_basis_motif_action(
    *,
    motif: str,
    intervention_step: int,
    reference_action: Any,
    basis_vectors: Sequence[Any],
    own_state: Mapping[str, Any],
    target_state: Mapping[str, Any],
    config: MotifPulseConfig,
) -> dict[str, Any]:
    """Project one interpretable world-frame pulse into the active VPP action head."""

    if motif not in MOTIF_NAMES:
        raise TacticalBasisMotifError(f"unknown motif: {motif}")
    reference = _vector(reference_action, name="reference_action")
    if np.any(np.abs(reference) > 1.0 + 1e-6):
        raise TacticalBasisMotifError("reference_action is outside [-1, 1]")
    basis, rank, condition = _basis_matrix(basis_vectors, config)
    scale = _pulse_scale(int(intervention_step), config)
    requested_world = _desired_world_delta(
        motif=motif,
        basis=basis,
        own_state=own_state,
        target_state=target_state,
        scale=scale,
        config=config,
    )
    requested_action_delta = np.linalg.solve(basis, requested_world)
    bounded_action_delta = np.clip(
        requested_action_delta,
        -config.max_action_delta,
        config.max_action_delta,
    )
    unclipped_action = reference + bounded_action_delta
    applied_action = np.clip(unclipped_action, -1.0, 1.0)
    actual_action_delta = applied_action - reference
    actual_world = basis @ actual_action_delta
    residual = requested_world - actual_world
    return {
        "motif": motif,
        "intervention_step": int(intervention_step),
        "pulse_scale": float(scale),
        "basis_rank": rank,
        "basis_condition_number": condition,
        "basis_vectors_world_m": basis.T.tolist(),
        "reference_action": reference.tolist(),
        "requested_world_delta_m": requested_world.tolist(),
        "requested_action_delta": requested_action_delta.tolist(),
        "bounded_action_delta": bounded_action_delta.tolist(),
        "applied_action": applied_action.tolist(),
        "actual_action_delta": actual_action_delta.tolist(),
        "actual_world_delta_m": actual_world.tolist(),
        "projection_residual_world_m": residual.tolist(),
        "projection_residual_norm_m": float(np.linalg.norm(residual)),
        "delta_limited": bool(
            not np.allclose(requested_action_delta, bounded_action_delta, rtol=0.0, atol=1e-8)
        ),
        "action_clipped": bool(
            not np.allclose(unclipped_action, applied_action, rtol=0.0, atol=1e-8)
        ),
    }


class TacticalBasisMotifSpecialistPolicy:
    """Overlay one bounded non-learning motif on a frozen head-on specialist."""

    def __init__(
        self,
        *,
        specialist_config: Mapping[str, Any],
        motif: str,
        pulse_config: Mapping[str, Any] | None = None,
        device: str = "cpu",
    ) -> None:
        if motif not in MOTIF_NAMES:
            raise TacticalBasisMotifError(f"unknown motif: {motif}")
        checkpoint = specialist_config.get("checkpoint")
        config_path = specialist_config.get("config_path")
        if not checkpoint or not config_path:
            raise TacticalBasisMotifError("motif specialist requires frozen checkpoint and config_path")
        self.policy = FrozenSpecialistPolicy(
            checkpoint_path=str(checkpoint),
            config_path=str(config_path),
            device=device,
        )
        if self.policy.action_dim != 3:
            raise TacticalBasisMotifError("motif specialist requires a 3-D VPP action")
        self.motif = motif
        self.config = MotifPulseConfig(**dict(pulse_config or {}))
        self.specialist_key = str(specialist_config.get("specialist_key", "head_on"))
        self.specialist_mode_name = str(
            specialist_config.get("specialist_mode_name", "head_on_motif_probe")
        )
        self.env: Optional[Any] = None
        self.reset_episode()

    def set_env(self, env: Any) -> None:
        self.env = env

    def set_task_name(self, task_name: str) -> None:
        self.current_task_name = str(task_name)

    def reset_episode(self) -> None:
        self.current_task_name: Optional[str] = None
        self._step_count = 0
        self._last_step_metadata: Dict[str, Any] = {}

    def _active_basis(self, own_state: Mapping[str, Any], target_state: Mapping[str, Any]) -> Sequence[Any]:
        if self.env is None:
            raise TacticalBasisMotifError("motif specialist requires set_env() before evaluation")
        generator = getattr(self.env, "virtual_point_generator", None)
        if generator is None or str(getattr(generator, "action_semantics", "")) != "tactical_basis_v1":
            raise TacticalBasisMotifError("motif specialist requires the deployed tactical_basis_v1 interface")
        values = generator._build_tactical_basis_vectors(own_state, target_state)
        return values[:3]

    def get_deterministic_action(self, obs: np.ndarray) -> np.ndarray:
        if self.env is None:
            raise TacticalBasisMotifError("motif specialist requires set_env() before evaluation")
        self._step_count += 1
        reference_action = self.policy.get_deterministic_action(obs)
        local_step = self._step_count - self.config.warmup_high_level_steps
        active = 1 <= local_step <= self.config.intervention_high_level_steps
        receipt: Optional[dict[str, Any]] = None
        action = reference_action
        phase = "warmup" if local_step <= 0 else "post_intervention"
        if active:
            own_state, target_state = self.env._get_current_states()
            receipt = compile_tactical_basis_motif_action(
                motif=self.motif,
                intervention_step=local_step,
                reference_action=reference_action,
                basis_vectors=self._active_basis(own_state, target_state),
                own_state=own_state,
                target_state=target_state,
                config=self.config,
            )
            action = np.asarray(receipt["applied_action"], dtype=np.float32)
            phase = "motif_intervention"
        if hasattr(self.env, "set_runtime_specialist_context"):
            self.env.set_runtime_specialist_context(
                specialist_key=self.specialist_key,
                specialist_profile=self.motif if active else None,
                specialist_mode_name=self.specialist_mode_name,
                specialist_reason="plan2_nonlearning_tactical_basis_motif",
            )
        self._last_step_metadata = {
            "motif_probe_phase": phase,
            "motif_probe_name": self.motif,
            "motif_probe_global_step": self._step_count,
            "motif_probe_intervention_step": int(local_step) if active else None,
            "motif_probe_uses_future_state": False,
            "motif_probe_receipt": receipt,
        }
        return np.asarray(action, dtype=np.float32)

    def get_last_step_metadata(self) -> Dict[str, Any]:
        return dict(self._last_step_metadata)
