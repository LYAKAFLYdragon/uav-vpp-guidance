"""Read-only episode-level attribution for the defensive-extension pilot."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from statistics import mean
from typing import Any


SOURCE_ID = "THESIS-DEFEXT-RANGEEXT-PILOT-V1-ATTRIBUTION-R1"
CANDIDATE = "candidate_fixed_defensive_extension"
HEAD_ON = "frozen_fixed_head_on"
CROSSING = "frozen_fixed_crossing"
METHODS = (CANDIDATE, HEAD_ON, CROSSING)
EXPECTED_STEP_TELEMETRY = (
    "normalized_vpp_action",
    "phase",
    "dynamic_state",
    "vp_forward_bias_m",
    "vp_lateral_bias_m",
    "vp_vertical_bias_m",
    "nz_cmd",
    "actual_nz_g",
    "ego_attack_aoa_deg",
    "own_speed_mps",
    "own_altitude_m",
    "range_m",
    "range_rate_mps",
    "specific_energy_height_delta_m",
    "altitude_delta_m",
    "target_in_attack_zone",
    "ego_in_attack_zone",
    "saturation_flag",
)


class AttributionContractError(RuntimeError):
    """Raised when frozen evidence cannot support the declared audit."""


def qualification_reason(record: Mapping[str, Any]) -> str:
    if not bool(record.get("handoff_reached")):
        return "no_handoff"
    if bool(record.get("qualifying")):
        return "qualifying"
    if int(record.get("valid_target_steps", 0)) < 20:
        return "insufficient_valid_target_steps"
    return "nonqualifying_unspecified"


def handoff_metadata_equivalent(*records: Mapping[str, Any]) -> bool:
    reached = {bool(record.get("handoff_reached")) for record in records}
    if len(reached) != 1:
        return False
    if reached == {False}:
        return len({record.get("terminal_reason") for record in records}) == 1
    return len({record.get("handoff_step") for record in records}) == 1


def _index(records: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for record in records:
        signature = str(record.get("scenario_signature") or "")
        if not signature:
            raise AttributionContractError("record is missing scenario_signature")
        if signature in indexed:
            raise AttributionContractError(f"duplicate scenario_signature: {signature}")
        indexed[signature] = record
    return indexed


def _paired_delta(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    require_handoff_equivalence: bool,
) -> float | None:
    if not bool(candidate.get("qualifying")) or not bool(baseline.get("qualifying")):
        return None
    if require_handoff_equivalence and not handoff_metadata_equivalent(
        candidate, baseline
    ):
        return None
    candidate_loss = candidate.get("intent_loss_auc20")
    baseline_loss = baseline.get("intent_loss_auc20")
    if candidate_loss is None or baseline_loss is None:
        return None
    return float(candidate_loss) - float(baseline_loss)


def _scenario_fields(
    scenario_signature: str,
    metadata: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, Any]:
    values = dict((metadata or {}).get(scenario_signature, {}))
    parts = scenario_signature.split("|")
    return {
        "distance_speed_package": values.get(
            "distance_speed_package", parts[0] if parts else None
        ),
        "height_condition": values.get(
            "height_condition", parts[1] if len(parts) > 1 else None
        ),
        "mirror_sign": values.get(
            "mirror_sign", parts[2] if len(parts) > 2 else None
        ),
        "initial_range_m": values.get("initial_range_m"),
        "own_speed_mps": values.get("own_speed_mps"),
        "target_speed_mps": values.get("target_speed_mps"),
    }


def _method_fields(prefix: str, record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_handoff_reached": bool(record.get("handoff_reached")),
        f"{prefix}_handoff_step": record.get("handoff_step"),
        f"{prefix}_valid_target_steps": int(record.get("valid_target_steps", 0)),
        f"{prefix}_qualification_reason": qualification_reason(record),
        f"{prefix}_intent_loss_auc20": record.get("intent_loss_auc20"),
        f"{prefix}_range_opening_fraction_auc20": record.get(
            "range_opening_fraction_auc20"
        ),
        f"{prefix}_specific_energy_height_delta_m_at_20": record.get(
            "specific_energy_height_delta_m_at_20"
        ),
        f"{prefix}_target_attack_zone_exposure_fraction_auc20": record.get(
            "target_attack_zone_exposure_fraction_auc20"
        ),
        f"{prefix}_ego_attack_zone_exposure_fraction_auc20": record.get(
            "ego_attack_zone_exposure_fraction_auc20"
        ),
        f"{prefix}_terminal_reason": record.get("terminal_reason"),
        f"{prefix}_ego_failure": bool(record.get("ego_failure")),
        f"{prefix}_ego_hp": record.get("ego_hp"),
        f"{prefix}_target_hp": record.get("target_hp"),
    }


def build_attribution_rows(
    records: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    scenario_metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for opponent, method_records in records.items():
        missing = set(METHODS) - set(method_records)
        if missing:
            raise AttributionContractError(
                f"{opponent}: missing methods {sorted(missing)}"
            )
        indexed = {method: _index(method_records[method]) for method in METHODS}
        signatures = set(indexed[CANDIDATE])
        for method in METHODS[1:]:
            if set(indexed[method]) != signatures:
                raise AttributionContractError(
                    f"{opponent}: scenario set differs for {method}"
                )
        for signature in sorted(signatures):
            candidate = indexed[CANDIDATE][signature]
            head = indexed[HEAD_ON][signature]
            crossing = indexed[CROSSING][signature]
            candidate_head_pair = bool(
                candidate.get("qualifying") and head.get("qualifying")
            )
            candidate_cross_pair = bool(
                candidate.get("qualifying") and crossing.get("qualifying")
            )
            candidate_head_handoff_equivalent = handoff_metadata_equivalent(
                candidate, head
            )
            candidate_cross_handoff_equivalent = handoff_metadata_equivalent(
                candidate, crossing
            )
            all_handoff_equivalent = handoff_metadata_equivalent(
                candidate, head, crossing
            )
            candidate_specific_failure = bool(
                candidate.get("ego_failure")
                and not head.get("ego_failure")
                and not crossing.get("ego_failure")
            )
            candidate_specific_failure_same_handoff = bool(
                candidate_specific_failure and all_handoff_equivalent
            )
            loose_head_delta = _paired_delta(
                candidate, head, require_handoff_equivalence=False
            )
            loose_cross_delta = _paired_delta(
                candidate, crossing, require_handoff_equivalence=False
            )
            strict_head_delta = _paired_delta(
                candidate, head, require_handoff_equivalence=True
            )
            strict_cross_delta = _paired_delta(
                candidate, crossing, require_handoff_equivalence=True
            )
            tags: list[str] = []
            if candidate_specific_failure_same_handoff:
                tags.append("candidate_specific_safety_signal")
            elif candidate_specific_failure:
                tags.append("candidate_failure_without_common_handoff")
            if not all_handoff_equivalent:
                tags.append("pre_handoff_pairing_mismatch")
            if loose_cross_delta is not None and abs(loose_cross_delta) < 0.05:
                tags.append("existing_crossing_geometry_overlap")
            if not candidate_head_pair and not candidate_cross_pair:
                tags.append("non_discriminative_or_unpaired")
            row = {
                "source_id": SOURCE_ID,
                "opponent": opponent,
                "scenario_signature": signature,
                "scenario_seed": candidate.get("scenario_seed"),
                **_scenario_fields(signature, scenario_metadata),
                **_method_fields("candidate", candidate),
                **_method_fields("head_on", head),
                **_method_fields("crossing", crossing),
                "all_method_handoff_metadata_equivalent": all_handoff_equivalent,
                "candidate_head_pairable": candidate_head_pair,
                "candidate_crossing_pairable": candidate_cross_pair,
                "candidate_head_handoff_metadata_equivalent": (
                    candidate_head_handoff_equivalent
                ),
                "candidate_crossing_handoff_metadata_equivalent": (
                    candidate_cross_handoff_equivalent
                ),
                "candidate_head_strict_pairable": bool(
                    candidate_head_pair and candidate_head_handoff_equivalent
                ),
                "candidate_crossing_strict_pairable": bool(
                    candidate_cross_pair and candidate_cross_handoff_equivalent
                ),
                "candidate_minus_head_on_delta_loose": loose_head_delta,
                "candidate_minus_crossing_delta_loose": loose_cross_delta,
                "candidate_minus_head_on_delta_strict": strict_head_delta,
                "candidate_minus_crossing_delta_strict": strict_cross_delta,
                "candidate_specific_ego_failure": candidate_specific_failure,
                "candidate_specific_failure_same_handoff": (
                    candidate_specific_failure_same_handoff
                ),
                "attribution_tags": ";".join(tags) if tags else "none",
            }
            rows.append(row)
    return rows


def _mean_or_none(values: Sequence[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return mean(present) if present else None


def summarize_attribution(
    rows: Sequence[Mapping[str, Any]],
    record_field_names: Sequence[str],
) -> dict[str, Any]:
    opponents = sorted({str(row["opponent"]) for row in rows})
    per_opponent: dict[str, Any] = {}
    for opponent in opponents:
        subset = [row for row in rows if row["opponent"] == opponent]
        candidate_reasons = Counter(
            str(row["candidate_qualification_reason"]) for row in subset
        )
        per_opponent[opponent] = {
            "scenarios": len(subset),
            "pre_handoff_mismatch_scenarios": sum(
                not bool(row["all_method_handoff_metadata_equivalent"])
                for row in subset
            ),
            "candidate_head_loose_pairs": sum(
                bool(row["candidate_head_pairable"]) for row in subset
            ),
            "candidate_head_strict_pairs": sum(
                bool(row["candidate_head_strict_pairable"]) for row in subset
            ),
            "candidate_crossing_loose_pairs": sum(
                bool(row["candidate_crossing_pairable"]) for row in subset
            ),
            "candidate_crossing_strict_pairs": sum(
                bool(row["candidate_crossing_strict_pairable"]) for row in subset
            ),
            "candidate_specific_ego_failures": sum(
                bool(row["candidate_specific_ego_failure"]) for row in subset
            ),
            "candidate_specific_failures_same_handoff": sum(
                bool(row["candidate_specific_failure_same_handoff"])
                for row in subset
            ),
            "candidate_qualification_reasons": dict(candidate_reasons),
            "mean_candidate_minus_head_on_delta_loose": _mean_or_none(
                [row["candidate_minus_head_on_delta_loose"] for row in subset]
            ),
            "mean_candidate_minus_head_on_delta_strict": _mean_or_none(
                [row["candidate_minus_head_on_delta_strict"] for row in subset]
            ),
            "mean_candidate_minus_crossing_delta_loose": _mean_or_none(
                [row["candidate_minus_crossing_delta_loose"] for row in subset]
            ),
            "mean_candidate_minus_crossing_delta_strict": _mean_or_none(
                [row["candidate_minus_crossing_delta_strict"] for row in subset]
            ),
        }
    missing_telemetry = [
        field for field in EXPECTED_STEP_TELEMETRY if field not in record_field_names
    ]
    safety_cases = [
        dict(row) for row in rows if row["candidate_specific_failure_same_handoff"]
    ]
    return {
        "source_id": SOURCE_ID,
        "audit_mode": "read_only_no_rerun_no_training",
        "row_count": len(rows),
        "per_opponent": per_opponent,
        "candidate_specific_safety_cases": safety_cases,
        "step_telemetry_available": not missing_telemetry,
        "missing_step_telemetry_fields": missing_telemetry,
        "formal_verdict_unchanged": "safety_or_contract_no_go",
        "new_skill_gap_supported": False,
        "composition_hypothesis": "plausible_but_not_causally_established",
        "execution_safety_hypothesis": (
            "candidate_specific_terminal_signal_observed_mechanism_unresolved"
        ),
        "posthoc_attribution_verdict": (
            "causal_mechanism_not_identifiable_from_frozen_artifacts"
        ),
        "required_next_protocol_fix": (
            "prove deterministic pre-handoff equivalence and persist per-step "
            "VPP/guidance/PID telemetry before any new performance pilot"
        ),
    }


def analyze_records(
    records: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    scenario_metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    rows = build_attribution_rows(records, scenario_metadata)
    first_record = next(
        iter(next(iter(records.values())).values())
    )[0]
    summary = summarize_attribution(rows, list(first_record))
    return {"summary": summary, "rows": rows}
