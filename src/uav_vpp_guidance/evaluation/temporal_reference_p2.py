"""Pure P2 compilation-envelope summaries for temporal VPP references."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence


class TemporalReferenceP2Error(ValueError):
    """Raised when an offline P2 row does not satisfy its fixed schema."""


def _finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TemporalReferenceP2Error(f"P2 row lacks finite {name}") from error
    if result != result or result in (float("inf"), float("-inf")):
        raise TemporalReferenceP2Error(f"P2 row lacks finite {name}")
    return result


def state_passes(row: Mapping[str, Any], gates: Mapping[str, Any]) -> bool:
    """Apply only the predeclared compiler-envelope gate to one replayed state."""

    return bool(
        row.get("replay_matched")
        and row.get("baseline_complete")
        and row.get("strict_jsbsim")
        and not row.get("backend_fallback")
        and not row.get("prediction_fallback")
        and int(row.get("baseline_steps", -1)) >= int(gates["min_baseline_steps_per_state"])
        and _finite(row.get("action_box_limited_fraction"), "action_box_limited_fraction")
        <= float(gates["max_action_box_limited_fraction"])
        and _finite(row.get("mean_projection_residual_m"), "mean_projection_residual_m")
        <= float(gates["max_mean_projection_residual_m"])
        and _finite(row.get("peak_actual_reference_delta_m"), "peak_actual_reference_delta_m")
        >= float(gates["min_peak_actual_reference_delta_m"])
    )


def summarize_rows(rows: Sequence[Mapping[str, Any]], gates: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize P2 separately by template variant and opponent, never pooling them."""

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        variant = str(row.get("candidate_variant") or "")
        opponent = str(row.get("opponent") or "")
        if not variant or opponent not in {"expert", "end_to_end", "independent_ppo_vpp"}:
            raise TemporalReferenceP2Error("P2 row has an invalid candidate variant or opponent")
        grouped[(variant, opponent)].append(row)

    matrix: dict[str, dict[str, Any]] = {}
    candidates: list[str] = []
    for variant in sorted({key[0] for key in grouped}):
        per_opponent: dict[str, Any] = {}
        for opponent in ("expert", "end_to_end", "independent_ppo_vpp"):
            subset = grouped.get((variant, opponent), [])
            passed = [row for row in subset if state_passes(row, gates)]
            signatures = {str(row.get("scenario_signature")) for row in passed}
            mirrors = {str(row.get("mirror_sign")) for row in passed}
            cell = {
                "states_total": len(subset),
                "states_compiler_feasible": len(passed),
                "distinct_signatures": len(signatures),
                "mirror_signs": sorted(mirrors),
                "passes": bool(
                    len(passed) >= int(gates["min_states_per_opponent"])
                    and len(signatures) >= int(gates["min_distinct_signatures_per_opponent"])
                    and len(mirrors) >= int(gates["min_mirror_signs_per_opponent"])
                ),
            }
            per_opponent[opponent] = cell
        matrix[variant] = per_opponent
        if sum(bool(cell["passes"]) for cell in per_opponent.values()) >= int(
            gates["candidate_opponents_required"]
        ):
            candidates.append(variant)
    return {
        "rows": len(rows),
        "matrix": matrix,
        "compiler_feasible_variants": candidates,
        "passes": bool(candidates),
        "interpretation": (
            "compiler_envelope_feasible_only" if candidates else "compiler_envelope_not_established"
        ),
    }
