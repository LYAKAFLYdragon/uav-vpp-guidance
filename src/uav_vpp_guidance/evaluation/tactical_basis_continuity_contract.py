"""Pure contract checks for tactical-basis continuous baseline trajectories."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from typing import Any, Mapping, Sequence


OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
REQUIRED_PHASES = ("post_merge", "re_entry")
REQUIRED_ACTION_SEMANTICS = "tactical_basis_v1"


class TacticalBasisContinuityContractError(ValueError):
    """Raised when a collector record violates the frozen data contract."""


def stable_hash(payload: Mapping[str, Any]) -> str:
    """Hash plain JSON telemetry deterministically for the continuity ledger."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _finite_scalar(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _finite_vector(value: Any, size: int) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != size:
        return False
    return all(_finite_scalar(item) for item in value)


def step_passes(step: Mapping[str, Any], gates: Mapping[str, Any]) -> bool:
    """Check one post-handoff baseline step without importing runtime code."""

    if (
        step.get("active_action_semantics") != REQUIRED_ACTION_SEMANTICS
        or step.get("configured_action_semantics") != REQUIRED_ACTION_SEMANTICS
        or step.get("tactical_basis_enabled") is not True
        or step.get("backend") != "jsbsim"
        or bool(step.get("backend_fallback_occurred", False))
        or bool(step.get("prediction_fallback", False))
        or bool(step.get("history_padding_used", False))
        or step.get("checkpoint_fallback") is not False
        or str(step.get("phase")) not in REQUIRED_PHASES
        or str(step.get("dynamic_state")) != "disadvantage"
        or int(step.get("history_length", -1)) != 10
        or not _finite_vector(step.get("base_observation_16d"), 16)
        or not _finite_vector(step.get("observation_66d"), 66)
        or not _finite_vector(step.get("reference_action"), 3)
        or not _finite_vector(step.get("tactical_basis_ll_world"), 3)
        or not _finite_vector(step.get("tactical_basis_io_world"), 3)
        or not _finite_vector(step.get("tactical_basis_cd_world"), 3)
        or not _finite_vector(step.get("vpp_world_offset_m"), 3)
        or not _finite_scalar(step.get("basis_condition_number"))
        or int(step.get("basis_rank", -1)) != 3
    ):
        return False
    return float(step["basis_condition_number"]) <= float(gates["max_basis_condition_number"])


def record_passes(record: Mapping[str, Any], gates: Mapping[str, Any]) -> bool:
    """Apply the full no-reset, 40-step continuity contract to one episode."""

    result = record.get("result")
    window = record.get("handoff_window")
    if not isinstance(result, Mapping) or not isinstance(window, Sequence):
        return False
    required_steps = int(gates["required_window_steps"])
    return bool(
        result.get("strict_jsbsim")
        and result.get("handoff_reached")
        and result.get("window_complete")
        and result.get("no_reset_after_initialization")
        and result.get("no_backend_fallback")
        and result.get("no_prediction_fallback")
        and result.get("no_history_padding")
        and result.get("no_checkpoint_fallback")
        and result.get("continuity_hash_chain_complete")
        and len(window) == required_steps
        and all(isinstance(step, Mapping) and step_passes(step, gates) for step in window)
    )


def summarize_records(records: Sequence[Mapping[str, Any]], gates: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize unpooled per-opponent collection coverage and its single gate."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        opponent = str(record.get("opponent") or "")
        if opponent not in OPPONENTS:
            raise TacticalBasisContinuityContractError("record has an unknown opponent")
        grouped[opponent].append(record)

    per_opponent: dict[str, dict[str, Any]] = {}
    passed_opponents: list[str] = []
    for opponent in OPPONENTS:
        attempted = grouped[opponent]
        qualifying = [record for record in attempted if record_passes(record, gates)]
        signatures = {str(record.get("scenario_signature") or "") for record in qualifying}
        signatures.discard("")
        mirrors = {str(record.get("mirror_sign") or "") for record in qualifying}
        mirrors.discard("")
        cell_passes = bool(
            len(qualifying) >= int(gates["min_qualifying_handoffs_per_opponent"])
            and len(signatures) >= int(gates["min_distinct_signatures_per_opponent"])
            and {"negative", "positive"}.issubset(mirrors)
        )
        if cell_passes:
            passed_opponents.append(opponent)
        per_opponent[opponent] = {
            "episodes_attempted": len(attempted),
            "qualifying_handoffs": len(qualifying),
            "distinct_scenario_signatures": sorted(signatures),
            "mirror_signs": sorted(mirrors),
            "passes": cell_passes,
        }

    passes = len(passed_opponents) >= int(gates["opponents_required"])
    return {
        "records": len(records),
        "per_opponent": per_opponent,
        "passed_opponents": passed_opponents,
        "passes": passes,
        "decision": (
            "phase_and_contract_gate_passed"
            if passes
            else "phase_or_continuity_contract_not_established"
        ),
        "next_action": (
            "new_independent_p2_design_permitted"
            if passes
            else "stop_temporal_reference_and_shared_skill_training_line"
        ),
    }
