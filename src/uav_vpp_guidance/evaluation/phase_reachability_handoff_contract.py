"""Fail-closed contracts for a future physical run-in to phase-sampler handoff."""

from __future__ import annotations

import hashlib
import json
import math
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
    execution_permitted: bool
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

STATIC_CONTINUITY_FIELDS = (
    "episode_lineage_id",
    "jsbsim_state_lineage_id",
    "observation_schema_sha256",
    "prediction_checkpoint_sha256",
)
PARENT_HASH_FIELDS = (
    "history_sha256",
    "vpp_action_sha256",
    "guidance_state_sha256",
    "pid_state_sha256",
)


def _json_safe(value: Any) -> Any:
    """Normalise telemetry into a deterministic, compact hash payload."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def telemetry_sha256(value: Any) -> str:
    """Hash a telemetry fragment without retaining its potentially large payload."""

    payload = json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
    """Validate either the frozen design or the separately authorised v2 probe."""

    protocol = _mapping(config.get("phase_reachability_v2"), "phase_reachability_v2")
    authorization = _mapping(protocol.get("authorization"), "authorization")
    source_id = str(protocol.get("source_id", ""))
    execution_permitted = authorization.get("execution_permitted")
    if authorization.get("training_permitted") is not False:
        raise PhaseReachabilityContractError("authorization.training_permitted must remain false")
    if authorization.get("action_replacement_permitted") is not False:
        raise PhaseReachabilityContractError("authorization.action_replacement_permitted must remain false")
    if source_id == "THESIS-PHASE-REACHABILITY-V2-DESIGN":
        if execution_permitted is not False:
            raise PhaseReachabilityContractError("design execution_permitted must remain false")
    elif source_id == "THESIS-PHASE-REACHABILITY-V2-RUN-IN-HANDOFF-R1":
        if execution_permitted is not True:
            raise PhaseReachabilityContractError("authorised probe execution_permitted must be true")
    else:
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
        source_id=source_id,
        run_in_controller=str(run_in["controller"]),
        handoff_mode=str(handoff["mode"]),
        opponents=opponents,
        post_merge_steps=_positive_int(gate.get("post_merge_step_minimum_per_episode"), "phase_gate.post_merge_step_minimum_per_episode"),
        re_entry_steps=_positive_int(gate.get("re_entry_step_minimum_per_episode"), "phase_gate.re_entry_step_minimum_per_episode"),
        qualifying_fraction=_fraction(gate.get("minimum_qualifying_episode_fraction_per_opponent"), "phase_gate.minimum_qualifying_episode_fraction_per_opponent"),
        execution_permitted=bool(execution_permitted),
        action_replacement_permitted=False,
    )


def validate_handoff_continuity(run_in: Mapping[str, Any], sampler: Mapping[str, Any]) -> None:
    """Reject reset/leakage while allowing the physical state to advance at k+1.

    The sampler record must retain *parent* hashes from the first-pass boundary
    at step k.  Its current history, VPP, guidance, and PID telemetry at k+1
    are expected to differ after one real JSBSim step and are therefore not
    required to equal the boundary values.
    """

    for name, record in (("run_in", run_in), ("sampler", sampler)):
        if record.get("backend") != "jsbsim" or bool(record.get("backend_fallback_occurred", False)):
            raise PhaseReachabilityContractError(f"{name} must retain strict JSBSim without fallback")
        if bool(record.get("reset_occurred", False)) or bool(record.get("future_state_injected", False)):
            raise PhaseReachabilityContractError(f"{name} reports an illegal reset or future-state injection")
    if not bool(run_in.get("first_pass_complete", False)):
        raise PhaseReachabilityContractError("run-in has not reached first pass")
    if int(sampler.get("step", -1)) != int(run_in.get("step", -1)) + 1:
        raise PhaseReachabilityContractError("handoff step must immediately follow run-in")
    if int(sampler.get("parent_step", -1)) != int(run_in.get("step", -1)):
        raise PhaseReachabilityContractError("sampler parent_step must reference the run-in boundary")
    for field in STATIC_CONTINUITY_FIELDS:
        if not run_in.get(field) or sampler.get(field) != run_in.get(field):
            raise PhaseReachabilityContractError(f"handoff continuity mismatch: {field}")
    for field in PARENT_HASH_FIELDS:
        if not run_in.get(field) or not sampler.get(field):
            raise PhaseReachabilityContractError(f"handoff missing hash: {field}")
        if sampler.get(f"parent_{field}") != run_in.get(field):
            raise PhaseReachabilityContractError(f"handoff parent hash mismatch: {field}")


class PhaseReachabilityContinuitySidecar:
    """Observe-only ledger for one physical run-in to sampler continuation.

    This object never receives a mutable action reference and never calls an
    environment, policy, VPP, guidance, or PID method.  It hashes compact
    telemetry after each real step, records the first-pass boundary at k, and
    records the next physical observation at k+1 with explicit parent hashes.
    """

    def __init__(
        self,
        *,
        run_id: str,
        method_name: str,
        task_name: str,
        seed: int,
        episode: int,
        scenario_name: str,
        observation_schema: Mapping[str, Any],
        prediction_checkpoint_sha256: str,
        environment_episode: int,
    ) -> None:
        self._episode_lineage_id = telemetry_sha256(
            {
                "run_id": run_id,
                "method": method_name,
                "task": task_name,
                "seed": int(seed),
                "episode": int(episode),
                "scenario": scenario_name,
            }
        )
        self._jsbsim_state_lineage_id = telemetry_sha256(
            {
                "episode_lineage_id": self._episode_lineage_id,
                "backend": "jsbsim",
                "environment_episode": int(environment_episode),
            }
        )
        self._observation_schema_sha256 = telemetry_sha256(observation_schema)
        self._prediction_checkpoint_sha256 = str(prediction_checkpoint_sha256)
        self._environment_episode = int(environment_episode)
        self._boundary: dict[str, Any] | None = None
        self._sampler: dict[str, Any] | None = None
        self._error: str | None = None

    @staticmethod
    def _history_snapshot(agent: Any, observation_vector: Any) -> Mapping[str, Any]:
        history = getattr(agent, "history", None)
        if history is None:
            return {
                "observability": "not_exposed_memoryless_reference",
                "current_observation_sha256": telemetry_sha256(observation_vector),
            }
        return {"observability": "agent_history", "history": history}

    @staticmethod
    def _guidance_snapshot(info: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            key: info.get(key)
            for key in (
                "virtual_point",
                "vp_position_neu",
                "vp_offset",
                "vp_world_offset",
                "virtual_point_source",
                "effective_guidance_mode",
                "mode_switch_effective",
                "predicted_target_position",
            )
        }

    @staticmethod
    def _pid_snapshot(info: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            "observability": "command_response_telemetry",
            **{
                key: info.get(key)
                for key in (
                    "nz_cmd",
                    "roll_rate_cmd",
                    "throttle_cmd",
                    "nz_saturated",
                    "roll_rate_saturated",
                    "throttle_saturated",
                    "saturation_flag",
                    "pid_gain_deltas",
                )
            },
        }

    def _record(
        self,
        *,
        step: int,
        observation: Mapping[str, Any],
        action: Any,
        info: Mapping[str, Any],
        agent: Any,
        environment_episode: int,
    ) -> dict[str, Any]:
        if int(environment_episode) != self._environment_episode:
            raise PhaseReachabilityContractError("environment episode lineage changed after reset")
        if info.get("backend") != "jsbsim" or bool(info.get("backend_fallback_occurred", False)):
            raise PhaseReachabilityContractError("sidecar requires strict JSBSim without fallback")
        if int(info.get("current_step", step)) != int(step):
            raise PhaseReachabilityContractError("sidecar step does not match environment step")
        observation_vector = observation.get("observation_vector")
        record = {
            "backend": "jsbsim",
            "backend_fallback_occurred": False,
            "reset_occurred": False,
            "future_state_injected": False,
            "first_pass_complete": bool(info.get("first_pass_complete", False)),
            "step": int(step),
            "episode_lineage_id": self._episode_lineage_id,
            "jsbsim_state_lineage_id": self._jsbsim_state_lineage_id,
            "observation_schema_sha256": self._observation_schema_sha256,
            "observation_vector_sha256": telemetry_sha256(observation_vector),
            "history_sha256": telemetry_sha256(self._history_snapshot(agent, observation_vector)),
            "prediction_checkpoint_sha256": self._prediction_checkpoint_sha256,
            "prediction_valid": bool(info.get("prediction_valid", False)),
            "vpp_action_sha256": telemetry_sha256(action),
            "guidance_state_sha256": telemetry_sha256(self._guidance_snapshot(info)),
            "pid_state_sha256": telemetry_sha256(self._pid_snapshot(info)),
            "jsbsim_state_sha256": telemetry_sha256(
                {"own_state": info.get("own_state"), "target_state": info.get("target_state")}
            ),
            "pid_state_observability": "command_response_telemetry",
        }
        return record

    def observe_after_step(
        self,
        *,
        step: int,
        observation: Mapping[str, Any],
        action: Any,
        info: Mapping[str, Any],
        agent: Any,
        environment_episode: int,
    ) -> None:
        """Record telemetry only; no policy or environment value is mutated."""

        if self._error is not None or self._sampler is not None:
            return
        try:
            current = self._record(
                step=step,
                observation=observation,
                action=action,
                info=info,
                agent=agent,
                environment_episode=environment_episode,
            )
            if self._boundary is None:
                if current["first_pass_complete"]:
                    self._boundary = current
                return
            if int(current["step"]) != int(self._boundary["step"]) + 1:
                raise PhaseReachabilityContractError("sampler record is not the physical k+1 step")
            current["parent_step"] = self._boundary["step"]
            for field in PARENT_HASH_FIELDS:
                current[f"parent_{field}"] = self._boundary[field]
            validate_handoff_continuity(self._boundary, current)
            self._sampler = current
        except PhaseReachabilityContractError as error:
            self._error = str(error)

    def ledger(self) -> dict[str, Any]:
        return {
            "sidecar_mode": "observe_only",
            "action_replacement_permitted": False,
            "handoff_detected": self._boundary is not None,
            "continuity_valid": self._sampler is not None and self._error is None,
            "continuity_error": self._error,
            "run_in_boundary": self._boundary,
            "sampler_observation": self._sampler,
        }
