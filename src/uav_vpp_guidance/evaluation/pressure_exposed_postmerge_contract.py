"""Pure gates for pressure-exposed post-merge evidence collection.

The contract deliberately separates two questions that were previously easy
to conflate: whether a post-merge trajectory is physically observable, and
whether it contains a sustained defensive-pressure endpoint.  It is used by
the read-only Taxonomy60 audit and by the future non-learning collector.
"""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Mapping, Sequence

SOURCE_ID = "THESIS-DISADVANTAGE-PRESSURE-EXPOSED-POSTMERGE-COLLECTOR-V1"
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
REQUIRED_DYNAMIC_STATE = "disadvantage"
REQUIRED_PHASES = ("post_merge", "re_entry")


class PressureExposedPostMergeContractError(ValueError):
    """Raised when a pressure-exposed cohort violates its frozen contract."""


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _as_score(value: Any) -> float | None:
    return float(value) if _finite(value) else None


def _finite_vector(value: Any, dimension: int) -> bool:
    return bool(
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == dimension
        and all(_finite(item) for item in value)
    )


def _pressure_step_passes(step: Mapping[str, Any], gates: Mapping[str, Any]) -> bool:
    """Validate one collector step in the pressure-qualified window."""

    score = _as_score(step.get("target_attack_score"))
    return bool(
        step.get("active_action_semantics") == "tactical_basis_v1"
        and step.get("configured_action_semantics") == "tactical_basis_v1"
        and step.get("tactical_basis_enabled") is True
        and step.get("backend") == "jsbsim"
        and not bool(step.get("backend_fallback_occurred", False))
        and not bool(step.get("prediction_fallback", False))
        and not bool(step.get("history_padding_used", False))
        and step.get("checkpoint_fallback") is False
        and str(step.get("phase")) in REQUIRED_PHASES
        and int(step.get("history_length", -1)) == 10
        and _finite_vector(step.get("base_observation_16d"), 16)
        and _finite_vector(step.get("observation_66d"), 66)
        and _finite_vector(step.get("reference_action"), 3)
        and score is not None
        and score >= 0.0
        and _finite(step.get("range_m"))
        and _finite(step.get("range_rate_mps"))
        and _finite(step.get("aa_deg"))
        and _finite(step.get("ata_deg"))
        and _finite(step.get("own_specific_energy_m"))
        and _finite(step.get("altitude_m"))
    )


def pressure_window_passes(window: Sequence[Mapping[str, Any]], gates: Mapping[str, Any]) -> bool:
    """Require a continuous 40-step post-merge interval with real pressure."""

    required_steps = int(gates["required_window_steps"])
    if len(window) != required_steps or not all(isinstance(step, Mapping) for step in window):
        return False
    if not all(_pressure_step_passes(step, gates) for step in window):
        return False
    if str(window[0].get("dynamic_state")) != REQUIRED_DYNAMIC_STATE:
        return False
    pressured = sum(
        float(step["target_attack_score"]) >= float(gates["target_attack_score_min"])
        for step in window
    )
    return pressured >= int(gates["min_pressure_steps_per_window"])


def record_passes(record: Mapping[str, Any], gates: Mapping[str, Any]) -> bool:
    """Apply continuity and pressure requirements to one collected episode."""

    result = record.get("result")
    if not isinstance(result, Mapping):
        return False
    required_true = (
        "strict_jsbsim",
        "handoff_reached",
        "window_complete",
        "no_reset_after_initialization",
        "no_backend_fallback",
        "no_prediction_fallback",
        "no_history_padding",
        "no_checkpoint_fallback",
        "continuity_hash_chain_complete",
    )
    if any(result.get(key) is not True for key in required_true):
        return False
    window = record.get("handoff_window")
    return isinstance(window, Sequence) and pressure_window_passes(window, gates)


def find_first_pressure_window(
    steps: Sequence[Mapping[str, Any]], gates: Mapping[str, Any]
) -> list[Mapping[str, Any]] | None:
    """Return the first valid fixed-length post-merge pressure window.

    This helper has no simulator dependency.  It permits a dynamic taxonomy
    transition *after* the first disadvantage state, because escaping pressure
    can legitimately move the aircraft toward neutral geometry; the starting
    state remains the causal task contract.
    """

    required_steps = int(gates["required_window_steps"])
    candidate: list[Mapping[str, Any]] = []
    for step in steps:
        if not isinstance(step, Mapping) or str(step.get("phase")) not in REQUIRED_PHASES:
            candidate.clear()
            continue
        candidate.append(step)
        if len(candidate) > required_steps:
            candidate.pop(0)
        if len(candidate) == required_steps and pressure_window_passes(candidate, gates):
            return list(candidate)
    return None


def summarize_records(records: Sequence[Mapping[str, Any]], gates: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize the pressure cohort without pooling opponents."""

    required_opponents = tuple(gates.get("required_opponents", OPPONENTS))
    if any(opponent not in OPPONENTS for opponent in required_opponents):
        raise PressureExposedPostMergeContractError("unknown required opponent")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        opponent = str(record.get("opponent") or "")
        if opponent not in OPPONENTS:
            raise PressureExposedPostMergeContractError("record has an unknown opponent")
        grouped[opponent].append(record)

    per_opponent: dict[str, dict[str, Any]] = {}
    passed_required: list[str] = []
    for opponent in OPPONENTS:
        attempted = grouped[opponent]
        qualifying = [record for record in attempted if record_passes(record, gates)]
        signatures = sorted(
            {str(record.get("scenario_signature")) for record in qualifying if record.get("scenario_signature")}
        )
        mirrors = sorted(
            {str(record.get("mirror_sign")) for record in qualifying if record.get("mirror_sign")}
        )
        passes = bool(
            len(qualifying) >= int(gates["min_qualifying_episodes_per_opponent"])
            and len(signatures) >= int(gates["min_distinct_signatures_per_opponent"])
            and {"negative", "positive"}.issubset(mirrors)
        )
        if opponent in required_opponents and passes:
            passed_required.append(opponent)
        per_opponent[opponent] = {
            "episodes_attempted": len(attempted),
            "qualifying_pressure_windows": len(qualifying),
            "distinct_scenario_signatures": signatures,
            "mirror_signs": mirrors,
            "passes": passes,
        }

    gate_passes = len(passed_required) == len(required_opponents)
    return {
        "records": len(records),
        "required_opponents": list(required_opponents),
        "per_opponent": per_opponent,
        "passed_required_opponents": passed_required,
        "passes": gate_passes,
        "decision": "pressure_exposed_postmerge_cohort_established" if gate_passes else "pressure_exposed_postmerge_cohort_not_established",
        "next_action": "capability_cards_and_compact_ablation_permitted" if gate_passes else "freeze_source_do_not_train_or_ablate",
    }
