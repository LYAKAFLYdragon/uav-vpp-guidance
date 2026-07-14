"""Fail-closed contracts for a future physical run-in to phase-sampler handoff."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


class PhaseReachabilityContractError(ValueError):
    """Raised when a v2 design or a handoff ledger violates its frozen boundary."""


@dataclass(frozen=True)
class PhaseReachabilityPlan:
    source_id: str
    run_in_controller: str
    handoff_mode: str
    opponents: tuple[str, ...]
    post_merge_steps: int
    re_entry_steps: int
    qualifying_fraction: float
    action_replacement_permitted: bool


REQUIRED_OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
REQUIRED_CONTINUITY_FIELDS = (
    "episode_lineage_id",
    "jsbsim_state_lineage_id",
    "observation_schema_sha256",
    "history_sha256",
    "prediction_checkpoint_sha256",
    "prediction_valid",
    "vpp_action_sha256",
    "guidance_state_sha256",
    "pid_state_sha256",
)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PhaseReachabilityContractError(f"{name} must be a mapping")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PhaseReachabilityContractError(f"{name} must be a positive integer")
    return value


def _fraction(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not 0.0 < float(value) <= 1.0:
        raise PhaseReachabilityContractError(f"{name} must be in (0, 1]")
    return float(value)


def build_phase_reachability_plan(config: Mapping[str, Any]) -> PhaseReachabilityPlan:
    """Validate a design-only v2 contract without running a simulator."""

    protocol = _mapping(config.get("phase_reachability_v2"), "phase_reachability_v2")
    authorization = _mapping(protocol.get("authorization"), "authorization")
    for key in ("execution_permitted", "training_permitted", "action_replacement_permitted"):
        if authorization.get(key) is not False:
            raise PhaseReachabilityContractError(f"authorization.{key} must remain false")
    if protocol.get("source_id") != "THESIS-PHASE-REACHABILITY-V2-DESIGN":
        raise PhaseReachabilityContractError("unexpected source_id")
    run_in = _mapping(protocol.get("run_in"), "run_in")
    if run_in.get("controller") != "legacy_static_oracle_task_gate" or run_in.get("learning") is not False:
        raise PhaseReachabilityContractError("run-in must use the frozen non-learning reference controller")
    handoff = _mapping(protocol.get("handoff"), "handoff")
    if handoff.get("mode") != "physical_continuation_sidecar":
        raise PhaseReachabilityContractError("handoff.mode must be physical_continuation_sidecar")
    for key in ("reset_after_first_pass", "snapshot_restore", "future_state_injection"):
        if handoff.get(key) is not False:
            raise PhaseReachabilityContractError(f"handoff.{key} must remain false")
    if handoff.get("sampler_action_mode") != "observe_only":
        raise PhaseReachabilityContractError("handoff.sampler_action_mode must be observe_only")
    opponents = tuple(protocol.get("opponents", ()))
    if opponents != REQUIRED_OPPONENTS:
        raise PhaseReachabilityContractError("v2 must preserve the frozen three-opponent order")
    gate = _mapping(protocol.get("phase_gate"), "phase_gate")
    return PhaseReachabilityPlan(
        source_id=str(protocol["source_id"]),
        run_in_controller=str(run_in["controller"]),
        handoff_mode=str(handoff["mode"]),
        opponents=opponents,
        post_merge_steps=_positive_int(gate.get("post_merge_step_minimum_per_episode"), "phase_gate.post_merge_step_minimum_per_episode"),
        re_entry_steps=_positive_int(gate.get("re_entry_step_minimum_per_episode"), "phase_gate.re_entry_step_minimum_per_episode"),
        qualifying_fraction=_fraction(gate.get("minimum_qualifying_episode_fraction_per_opponent"), "phase_gate.minimum_qualifying_episode_fraction_per_opponent"),
        action_replacement_permitted=False,
    )


def validate_handoff_continuity(run_in: Mapping[str, Any], sampler: Mapping[str, Any]) -> None:
    """Reject reset, future-state leakage, or state/history discontinuity at handoff."""

    for name, record in (("run_in", run_in), ("sampler", sampler)):
        if record.get("backend") != "jsbsim" or bool(record.get("backend_fallback_occurred", False)):
            raise PhaseReachabilityContractError(f"{name} must retain strict JSBSim without fallback")
        if bool(record.get("reset_occurred", False)) or bool(record.get("future_state_injected", False)):
            raise PhaseReachabilityContractError(f"{name} reports an illegal reset or future-state injection")
    if not bool(run_in.get("first_pass_complete", False)):
        raise PhaseReachabilityContractError("run-in has not reached first pass")
    if int(sampler.get("step", -1)) != int(run_in.get("step", -1)) + 1:
        raise PhaseReachabilityContractError("handoff step must immediately follow run-in")
    for field in REQUIRED_CONTINUITY_FIELDS:
        if not run_in.get(field) or sampler.get(field) != run_in.get(field):
            raise PhaseReachabilityContractError(f"handoff continuity mismatch: {field}")
