"""Disposable runtime-contract checks used before a new thesis Source ID."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
REQUIRED_TRACE_FIELDS = (
    "step",
    "phase",
    "raw_range_m",
    "raw_range_rate_mps",
    "vp_world_offset_m",
    "guidance_command_normalized",
    "aoa_rad",
    "actual_nz_g",
    "runtime_vpp_anchor_mode",
)


class RuntimeSmokeContractError(RuntimeError):
    """Raised when a disposable smoke rollout lacks prerequisite telemetry."""


def _finite_scalar(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeSmokeContractError(f"runtime smoke lacks finite {name}") from error
    if not math.isfinite(result):
        raise RuntimeSmokeContractError(f"runtime smoke lacks finite {name}")
    return result


def _finite_vector(value: Any, name: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise RuntimeSmokeContractError(f"runtime smoke lacks 3-D {name}")
    return [_finite_scalar(component, name) for component in value]


def validate_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one full episode without inferring any performance conclusion."""
    if str(record.get("opponent")) not in OPPONENTS:
        raise RuntimeSmokeContractError("runtime smoke opponent is outside the fixed registry")
    if not bool(record.get("handoff_reached")) or not str(record.get("handoff_state_hash") or ""):
        raise RuntimeSmokeContractError("runtime smoke did not produce a handoff state hash")
    observer_trace = record.get("observer_trace")
    if not isinstance(observer_trace, list) or not observer_trace:
        raise RuntimeSmokeContractError("runtime smoke observer trace is empty")
    switches = record.get("anchor_switches")
    if not isinstance(switches, list) or not switches:
        raise RuntimeSmokeContractError("runtime smoke anchor-switch trace is empty")
    if not bool(record.get("released")):
        raise RuntimeSmokeContractError("runtime smoke observer protocol did not release")
    predicted_switches = [
        item
        for item in switches
        if item.get("mode") == "predicted_target"
        and item.get("reason") == "v22_first_valid_observer_receipt"
    ]
    if len(predicted_switches) != 1:
        raise RuntimeSmokeContractError("runtime smoke must contain exactly one valid observer release")
    if not bool(record.get("warmup_control_isolated")):
        raise RuntimeSmokeContractError("runtime smoke observer prediction entered control during warm-up")
    if bool(record.get("post_release_prediction_fallback")):
        raise RuntimeSmokeContractError("runtime smoke prediction fell back after release")

    phases = record.get("phase_trace")
    if not isinstance(phases, list) or not phases:
        raise RuntimeSmokeContractError("runtime smoke phase trace is empty")
    reentry_indices = [index for index, item in enumerate(phases) if item.get("phase") == "re_entry"]
    if not reentry_indices:
        raise RuntimeSmokeContractError("runtime smoke did not reach raw-SI re-entry")
    first_reentry = reentry_indices[0]
    if not any(bool(item.get("first_pass_complete")) for item in phases[: first_reentry + 1]):
        raise RuntimeSmokeContractError("runtime smoke re-entry precedes raw-SI first pass")

    trace = record.get("runtime_trace")
    if not isinstance(trace, list) or not trace:
        raise RuntimeSmokeContractError("runtime smoke telemetry trace is empty")
    for step in trace:
        if not isinstance(step, Mapping):
            raise RuntimeSmokeContractError("runtime smoke telemetry row is malformed")
        if any(field not in step for field in REQUIRED_TRACE_FIELDS):
            raise RuntimeSmokeContractError("runtime smoke telemetry fields are incomplete")
        _finite_scalar(step["raw_range_m"], "raw_range_m")
        _finite_scalar(step["raw_range_rate_mps"], "raw_range_rate_mps")
        _finite_scalar(step["aoa_rad"], "aoa_rad")
        _finite_scalar(step["actual_nz_g"], "actual_nz_g")
        _finite_vector(step["vp_world_offset_m"], "vp_world_offset_m")
        _finite_vector(step["guidance_command_normalized"], "guidance_command_normalized")
        if step["runtime_vpp_anchor_mode"] not in ("current_target", "predicted_target"):
            raise RuntimeSmokeContractError("runtime smoke anchor mode is unreadable")
    return {
        "opponent": str(record["opponent"]),
        "handoff_state_hash": str(record["handoff_state_hash"]),
        "runtime_steps": len(trace),
        "observer_steps": len(observer_trace),
        "release_step": int(predicted_switches[0]["step"]),
    }


def summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Require exactly one valid full episode for each fixed opponent."""
    if len(records) != len(OPPONENTS):
        raise RuntimeSmokeContractError("runtime smoke must contain exactly three records")
    rows = [validate_record(record) for record in records]
    opponents = [row["opponent"] for row in rows]
    if tuple(sorted(opponents)) != tuple(sorted(OPPONENTS)):
        raise RuntimeSmokeContractError("runtime smoke opponent universe is incomplete or duplicated")
    return {"records": len(rows), "opponents": rows, "passes": True}
