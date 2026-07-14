"""Pure validation helpers for the non-learning P4-v2 sampler preflight.

The module intentionally does not import an environment, a policy, or JSBSim.
It validates the design contract that must pass before a later, separately
authorized physical reachability probe may exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from uav_vpp_guidance.evaluation.engagement_geometry_taxonomy import (
    GeometryTaxonomyConfig,
    classify_relative_geometry,
)


REQUIRED_OPPONENTS = (
    "expert_rule_based",
    "end_to_end_neural",
    "independent_ppo_vpp",
)
REQUIRED_REFERENCE_CONTROLLERS = ("zero_vpp", "profile_compiled_fixed")
ALLOWED_PHASES = ("pre_merge", "post_merge", "re_entry")


class PreflightContractError(ValueError):
    """Raised when a P4-v2 design violates a fail-closed contract."""


@dataclass(frozen=True)
class SkillPhaseCell:
    """One required dynamic-state x phase coverage cell."""

    skill: str
    dynamic_state: str
    phase: str

    @property
    def key(self) -> str:
        return f"{self.skill}:{self.dynamic_state}:{self.phase}"


@dataclass(frozen=True)
class DynamicTaxonomySidecar:
    """Analysis-only label emitted beside, never inside, policy observations."""

    initial_class: str
    dynamic_class: str
    phase: str
    phase_entry_step: int
    own_to_target_los_angle_deg: float
    target_velocity_to_own_los_angle_deg: float


@dataclass(frozen=True)
class SamplerFeasibilityPlan:
    """Validated design plan; it contains no environment or training state."""

    declared_cells: tuple[SkillPhaseCell, ...]
    train_source_id: str
    dev_source_id: str
    train_signature_namespace: str
    dev_signature_namespace: str
    min_episodes_per_cell_per_opponent: int
    min_policy_steps_per_cell_per_opponent: int
    min_distinct_scenario_signatures: int


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PreflightContractError(f"{name} must be a mapping")
    return value


def _require_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PreflightContractError(f"{name} must be a non-empty string")
    return value


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise PreflightContractError(f"{name} must be a boolean")
    return value


def _require_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PreflightContractError(f"{name} must be a positive integer")
    return value


def _require_nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PreflightContractError(f"{name} must be a non-negative integer")
    return value


def classify_dynamic_state(
    own_to_target_los_angle_deg: Any,
    target_velocity_to_own_los_angle_deg: Any,
    *,
    hemisphere_boundary_deg: float = 90.0,
    transition_half_width_deg: float = 5.0,
) -> str:
    """Return one of the five dynamic states without changing policy inputs.

    The existing four-quadrant taxonomy keeps its angle convention. Its
    transition band is represented as ``crossing_entry`` for the v2 sidecar.
    Unknown/non-finite inputs remain ``unknown`` and are never coerced.
    """

    geometry = classify_relative_geometry(
        own_to_target_los_angle_deg,
        target_velocity_to_own_los_angle_deg,
        GeometryTaxonomyConfig(
            hemisphere_boundary_deg=float(hemisphere_boundary_deg),
            transition_half_width_deg=float(transition_half_width_deg),
        ),
    )
    return "crossing_entry" if geometry == "transition" else geometry


def build_dynamic_taxonomy_sidecar(
    record: Mapping[str, Any], *, hemisphere_boundary_deg: float = 90.0,
    transition_half_width_deg: float = 5.0,
) -> DynamicTaxonomySidecar:
    """Build an auditable dynamic label from one telemetry record."""

    initial_class = _require_str(record.get("initial_class"), "initial_class")
    phase = _require_str(record.get("phase"), "phase")
    if phase not in ALLOWED_PHASES:
        raise PreflightContractError(f"unsupported phase: {phase}")
    phase_entry_step = _require_nonnegative_int(
        record.get("phase_entry_step"), "phase_entry_step"
    )
    try:
        own_angle = float(record["own_to_target_los_angle_deg"])
        target_angle = float(record["target_velocity_to_own_los_angle_deg"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PreflightContractError("telemetry record has invalid taxonomy angles") from exc
    dynamic_class = classify_dynamic_state(
        own_angle,
        target_angle,
        hemisphere_boundary_deg=hemisphere_boundary_deg,
        transition_half_width_deg=transition_half_width_deg,
    )
    if dynamic_class == "unknown":
        raise PreflightContractError("dynamic taxonomy label is unknown")
    return DynamicTaxonomySidecar(
        initial_class=initial_class,
        dynamic_class=dynamic_class,
        phase=phase,
        phase_entry_step=phase_entry_step,
        own_to_target_los_angle_deg=own_angle,
        target_velocity_to_own_los_angle_deg=target_angle,
    )


def derive_declared_cells(skill_registry: Mapping[str, Any]) -> tuple[SkillPhaseCell, ...]:
    """Expand the frozen registry into the P4-v2 dynamic coverage matrix."""

    skills = _require_mapping(skill_registry.get("skills"), "registry.skills")
    cells: list[SkillPhaseCell] = []
    for skill_name, raw_definition in skills.items():
        definition = _require_mapping(raw_definition, f"skills.{skill_name}")
        coverage = _require_mapping(
            definition.get("geometry_phase_coverage"),
            f"skills.{skill_name}.geometry_phase_coverage",
        )
        states = coverage.get("geometry_states")
        phases = coverage.get("phases")
        if not isinstance(states, Sequence) or isinstance(states, (str, bytes)):
            raise PreflightContractError(f"skills.{skill_name}.geometry_states must be a list")
        if not isinstance(phases, Sequence) or isinstance(phases, (str, bytes)):
            raise PreflightContractError(f"skills.{skill_name}.phases must be a list")
        for state in states:
            state_name = _require_str(state, f"skills.{skill_name}.geometry_state")
            for phase in phases:
                phase_name = _require_str(phase, f"skills.{skill_name}.phase")
                if phase_name not in ALLOWED_PHASES:
                    raise PreflightContractError(
                        f"skills.{skill_name} declares unsupported phase {phase_name}"
                    )
                cells.append(SkillPhaseCell(str(skill_name), state_name, phase_name))
    if len({cell.key for cell in cells}) != len(cells):
        raise PreflightContractError("registry declares duplicate skill/state/phase cells")
    return tuple(cells)


def _seed_range(manifest: Mapping[str, Any]) -> tuple[int, int]:
    values = manifest.get("candidate_seed_range")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) != 2:
        raise PreflightContractError("candidate_seed_range must contain [start, end]")
    start = _require_positive_int(values[0], "candidate_seed_range[0]")
    end = _require_positive_int(values[1], "candidate_seed_range[1]")
    if start > end:
        raise PreflightContractError("candidate_seed_range start must not exceed end")
    return start, end


def _validate_candidate_manifest(
    manifest: Mapping[str, Any], *, expected_split: str
) -> tuple[str, tuple[int, int], set[str]]:
    if _require_str(manifest.get("split"), "manifest.split") != expected_split:
        raise PreflightContractError(f"manifest split must be {expected_split}")
    if _require_bool(manifest.get("execution_permitted"), "manifest.execution_permitted"):
        raise PreflightContractError("candidate manifest must remain execution_permitted: false")
    if _require_str(manifest.get("coverage_source"), "manifest.coverage_source") != "registry_dynamic_state_matrix":
        raise PreflightContractError("candidate manifest must derive coverage from registry_dynamic_state_matrix")
    opponents = tuple(manifest.get("opponents", ()))
    if opponents != REQUIRED_OPPONENTS:
        raise PreflightContractError("candidate manifest must use the frozen three-opponent order")
    controllers = tuple(manifest.get("reference_controllers", ()))
    if controllers != REQUIRED_REFERENCE_CONTROLLERS:
        raise PreflightContractError("candidate manifest must use only frozen non-learning references")
    namespace = _require_str(manifest.get("scenario_signature_namespace"), "scenario_signature_namespace")
    packages = manifest.get("distance_speed_package_ids")
    if not isinstance(packages, Sequence) or isinstance(packages, (str, bytes)) or not packages:
        raise PreflightContractError("distance_speed_package_ids must be a non-empty list")
    return namespace, _seed_range(manifest), {str(item) for item in packages}


def build_sampler_feasibility_plan(
    config: Mapping[str, Any], skill_registry: Mapping[str, Any],
    train_manifest: Mapping[str, Any], dev_manifest: Mapping[str, Any],
) -> SamplerFeasibilityPlan:
    """Validate the non-executing P4-v2 design and return its coverage plan."""

    authorization = _require_mapping(config.get("authorization"), "authorization")
    for key in ("execution_permitted", "jsbsim_probe_permitted", "training_permitted"):
        if _require_bool(authorization.get(key), f"authorization.{key}"):
            raise PreflightContractError(f"authorization.{key} must remain false")

    contract = _require_mapping(config.get("observation_action_contract"), "observation_action_contract")
    if contract.get("observation_dim") != 66 or contract.get("action_dim") != 3:
        raise PreflightContractError("P4-v2 preflight must preserve the 66-D to 3-D contract")
    if contract.get("policy_input_mutation") != "prohibited":
        raise PreflightContractError("dynamic taxonomy must remain an analysis-only sidecar")

    support = _require_mapping(config.get("coverage_support_gate"), "coverage_support_gate")
    min_episodes = _require_positive_int(
        support.get("min_episodes_per_cell_per_opponent"),
        "coverage_support_gate.min_episodes_per_cell_per_opponent",
    )
    min_steps = _require_positive_int(
        support.get("min_policy_steps_per_cell_per_opponent"),
        "coverage_support_gate.min_policy_steps_per_cell_per_opponent",
    )
    min_signatures = _require_positive_int(
        support.get("min_distinct_scenario_signatures"),
        "coverage_support_gate.min_distinct_scenario_signatures",
    )

    train_id = _require_str(train_manifest.get("source_id"), "train.source_id")
    dev_id = _require_str(dev_manifest.get("source_id"), "dev.source_id")
    train_namespace, train_range, train_packages = _validate_candidate_manifest(
        train_manifest, expected_split="train"
    )
    dev_namespace, dev_range, dev_packages = _validate_candidate_manifest(
        dev_manifest, expected_split="dev"
    )
    if train_namespace == dev_namespace:
        raise PreflightContractError("train and dev signature namespaces must differ")
    if not (train_range[1] < dev_range[0] or dev_range[1] < train_range[0]):
        raise PreflightContractError("train and dev candidate seed ranges must be disjoint")
    if train_packages.intersection(dev_packages):
        raise PreflightContractError("train and dev distance-speed packages must be disjoint")

    cells = derive_declared_cells(skill_registry)
    if len(cells) != 26:
        raise PreflightContractError(
            f"P4-v2 design expects 26 declared cells, found {len(cells)}"
        )
    return SamplerFeasibilityPlan(
        declared_cells=cells,
        train_source_id=train_id,
        dev_source_id=dev_id,
        train_signature_namespace=train_namespace,
        dev_signature_namespace=dev_namespace,
        min_episodes_per_cell_per_opponent=min_episodes,
        min_policy_steps_per_cell_per_opponent=min_steps,
        min_distinct_scenario_signatures=min_signatures,
    )
