"""Pure post-merge tactical-basis P2 envelope calculations."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from uav_vpp_guidance.evaluation.temporal_reference_compiler import (
    TemporalReferenceConfig,
    TemporalReferenceTemplate,
    compile_temporal_reference_program,
)


OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")


class TacticalBasisPostmergeP2Error(ValueError):
    """Raised when a P2 input row or frozen gate is malformed."""


def _finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TacticalBasisPostmergeP2Error(f"{name} must be finite") from error
    if not np.isfinite(result):
        raise TacticalBasisPostmergeP2Error(f"{name} must be finite")
    return result


def candidate_variants(reference: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand only the signs frozen in the P1 temporal-reference design."""

    templates = reference.get("templates")
    if not isinstance(templates, Mapping):
        raise TacticalBasisPostmergeP2Error("temporal-reference templates are required")
    variants: list[dict[str, Any]] = []
    for name in sorted(templates):
        value = templates[name]
        if not isinstance(value, Mapping):
            raise TacticalBasisPostmergeP2Error(f"template {name} is malformed")
        template = TemporalReferenceTemplate.from_mapping(str(name), value)
        lateral = value.get("lateral_sign_candidates")
        vertical = value.get("vertical_sign_candidates")
        if not isinstance(lateral, Sequence) or isinstance(lateral, (str, bytes)):
            raise TacticalBasisPostmergeP2Error(f"template {name} lacks lateral signs")
        if not isinstance(vertical, Sequence) or isinstance(vertical, (str, bytes)):
            raise TacticalBasisPostmergeP2Error(f"template {name} lacks vertical signs")
        for lateral_sign in lateral:
            for vertical_sign in vertical:
                if int(lateral_sign) not in (-1, 1) or int(vertical_sign) not in (-1, 1):
                    raise TacticalBasisPostmergeP2Error(f"template {name} has an invalid sign")
                variants.append(
                    {
                        "candidate_variant": f"{name}:lat{int(lateral_sign):+d}:vert{int(vertical_sign):+d}",
                        "template": template,
                        "lateral_sign": int(lateral_sign),
                        "vertical_sign": int(vertical_sign),
                    }
                )
    if not variants:
        raise TacticalBasisPostmergeP2Error("no P2 candidate variants were declared")
    return variants


def compile_candidate_row(
    record: Mapping[str, Any],
    candidate: Mapping[str, Any],
    config: TemporalReferenceConfig,
    *,
    source_record_sha256: str,
) -> dict[str, Any]:
    """Compile one stored 40-step baseline window without simulator execution."""

    window = record.get("handoff_window")
    if not isinstance(window, Sequence) or len(window) != config.horizon_steps:
        raise TacticalBasisPostmergeP2Error("record lacks the frozen complete baseline window")
    try:
        actions = [step["reference_action"] for step in window]
        bases = [
            [
                step["tactical_basis_ll_world"],
                step["tactical_basis_io_world"],
                step["tactical_basis_cd_world"],
            ]
            for step in window
        ]
    except (KeyError, TypeError) as error:
        raise TacticalBasisPostmergeP2Error("record lacks action or tactical-basis telemetry") from error
    receipts = compile_temporal_reference_program(
        template=candidate["template"],
        reference_actions=actions,
        basis_vectors_by_step=bases,
        config=config,
        lateral_sign=int(candidate["lateral_sign"]),
        vertical_sign=int(candidate["vertical_sign"]),
    )
    actual_norms = [
        float(np.linalg.norm(np.asarray(receipt["actual_reference_world_delta_m"], dtype=np.float64)))
        for receipt in receipts
    ]
    row = {
        "candidate_variant": str(candidate["candidate_variant"]),
        "opponent": str(record["opponent"]),
        "scenario_name": str(record["scenario_name"]),
        "scenario_signature": str(record["scenario_signature"]),
        "mirror_sign": str(record["mirror_sign"]),
        "handoff_state_hash": record.get("result", {}).get("handoff_state_hash"),
        "source_record_sha256": str(source_record_sha256),
        "data_contract_qualified": True,
        "strict_jsbsim": True,
        "baseline_steps": len(window),
        "backend_fallback": False,
        "prediction_fallback": False,
        "action_box_limited_fraction": float(
            np.mean([bool(receipt["action_box_limited"]) for receipt in receipts])
        ),
        "slew_limited_fraction": float(
            np.mean([bool(receipt["slew_limited"]) for receipt in receipts])
        ),
        "mean_projection_residual_m": float(
            np.mean([float(receipt["projection_residual_norm_m"]) for receipt in receipts])
        ),
        "peak_actual_reference_delta_m": float(max(actual_norms)),
        "max_basis_condition_number": float(
            max(float(receipt["basis_condition_number"]) for receipt in receipts)
        ),
        "receipts": receipts,
    }
    return row


def state_passes(row: Mapping[str, Any], gates: Mapping[str, Any]) -> bool:
    """Apply the frozen physical envelope gate to one trajectory-template row."""

    return bool(
        row.get("data_contract_qualified")
        and row.get("strict_jsbsim")
        and not bool(row.get("backend_fallback", True))
        and not bool(row.get("prediction_fallback", True))
        and int(row.get("baseline_steps", -1)) >= int(gates["required_window_steps"])
        and _finite(row.get("action_box_limited_fraction"), "action_box_limited_fraction")
        <= float(gates["max_action_box_limited_fraction"])
        and _finite(row.get("slew_limited_fraction"), "slew_limited_fraction")
        <= float(gates["max_slew_limited_fraction"])
        and _finite(row.get("mean_projection_residual_m"), "mean_projection_residual_m")
        <= float(gates["max_mean_projection_residual_m"])
        and _finite(row.get("peak_actual_reference_delta_m"), "peak_actual_reference_delta_m")
        >= float(gates["min_peak_actual_reference_delta_m"])
        and _finite(row.get("max_basis_condition_number"), "max_basis_condition_number")
        <= float(gates["max_basis_condition_number"])
    )


def summarize_rows(rows: Sequence[Mapping[str, Any]], gates: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize template feasibility by opponent without pooled statistics."""

    required_opponents = tuple(gates["required_opponents"])
    if set(required_opponents) != {"end_to_end", "independent_ppo_vpp"}:
        raise TacticalBasisPostmergeP2Error("P2 requires exactly the two V3-qualified opponents")
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        opponent = str(row.get("opponent") or "")
        variant = str(row.get("candidate_variant") or "")
        if opponent not in OPPONENTS or not variant:
            raise TacticalBasisPostmergeP2Error("row has an invalid opponent or candidate variant")
        grouped[(variant, opponent)].append(row)

    matrix: dict[str, dict[str, Any]] = {}
    passing_variants: list[str] = []
    ranking: list[tuple[tuple[float, float, float, str], str]] = []
    for variant in sorted({key[0] for key in grouped}):
        per_opponent: dict[str, Any] = {}
        for opponent in OPPONENTS:
            subset = grouped.get((variant, opponent), [])
            passed = [row for row in subset if state_passes(row, gates)]
            signatures = {str(row.get("scenario_signature") or "") for row in passed}
            signatures.discard("")
            mirrors = {str(row.get("mirror_sign") or "") for row in passed}
            mirrors.discard("")
            gate_role = "required" if opponent in required_opponents else "diagnostic_only"
            cell_passes = bool(
                gate_role == "required"
                and len(passed) >= int(gates["min_states_per_required_opponent"])
                and len(signatures) >= int(gates["min_distinct_signatures_per_required_opponent"])
                and {"negative", "positive"}.issubset(mirrors)
            )
            per_opponent[opponent] = {
                "gate_role": gate_role,
                "states_total": len(subset),
                "states_physically_feasible": len(passed),
                "distinct_signatures": sorted(signatures),
                "mirror_signs": sorted(mirrors),
                "mean_projection_residual_m": (
                    None if not passed else float(np.mean([row["mean_projection_residual_m"] for row in passed]))
                ),
                "minimum_peak_actual_reference_delta_m": (
                    None if not passed else float(min(row["peak_actual_reference_delta_m"] for row in passed))
                ),
                "passes": cell_passes,
            }
        matrix[variant] = per_opponent
        if all(per_opponent[opponent]["passes"] for opponent in required_opponents):
            passing_variants.append(variant)
            required_cells = [per_opponent[opponent] for opponent in required_opponents]
            ranking.append(
                (
                    (
                        -float(min(cell["states_physically_feasible"] for cell in required_cells)),
                        -float(min(cell["minimum_peak_actual_reference_delta_m"] for cell in required_cells)),
                        float(max(cell["mean_projection_residual_m"] for cell in required_cells)),
                        variant,
                    ),
                    variant,
                )
            )
    selected_variant = None if not ranking else sorted(ranking)[0][1]
    return {
        "rows": len(rows),
        "matrix": matrix,
        "physically_feasible_variants": passing_variants,
        "selected_variant": selected_variant,
        "passes": bool(passing_variants),
        "decision": (
            "postmerge_physical_envelope_established"
            if passing_variants
            else "postmerge_physical_envelope_not_established"
        ),
        "next_action": (
            "one_selected_variant_runtime_probe_design_permitted"
            if passing_variants
            else "stop_temporal_reference_and_shared_skill_training_line"
        ),
    }
