"""Pure design-contract checks for the selected post-merge runtime probe."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Iterable, Mapping, Sequence


SOURCE_ID = "THESIS-DISADVANTAGE-TACTICAL-BASIS-POSTMERGE-LATERAL-RUNTIME-PROBE-V1"
P2_SOURCE_ID = "THESIS-DISADVANTAGE-TACTICAL-BASIS-POSTMERGE-ENVELOPE-P2-V1"
P2_SELECTED_VARIANT = "lateral_extension:lat+1:vert+1"
METHODS = ("fixed_head_on", "lateral_extension_lat_pos_vert_pos")
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
REQUIRED_OPPONENTS = ("end_to_end", "independent_ppo_vpp")


class TacticalBasisLateralRuntimeProbeDesignError(ValueError):
    """Raised when the future runtime probe drifts from the frozen P2 result."""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TacticalBasisLateralRuntimeProbeDesignError(f"{name} must be a mapping")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TacticalBasisLateralRuntimeProbeDesignError(f"{name} must be a sequence")
    return value


def validate_design_spec(
    config: Mapping[str, Any],
    p2_summary: Mapping[str, Any],
    p2_verification: Mapping[str, Any],
    probe_manifest: Mapping[str, Any],
    parent_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the design before any runtime policy adapter is implemented."""

    if config.get("source_id") != SOURCE_ID:
        raise TacticalBasisLateralRuntimeProbeDesignError("runtime-probe source ID drifted")
    if config.get("stage") != "paired_nonlearning_postmerge_runtime_probe":
        raise TacticalBasisLateralRuntimeProbeDesignError("runtime-probe stage drifted")
    authorization = _mapping(config.get("authorization"), "authorization")
    for key in (
        "execution_permitted",
        "training_permitted",
        "tuning_permitted",
        "reward_change_permitted",
        "high_level_ppo_training_permitted",
        "shared_skill_training_permitted",
        "guidance_pid_change_permitted",
        "action_envelope_expansion_permitted",
        "snapshot_restore_permitted",
        "future_state_injection_permitted",
        "history_padding_permitted",
    ):
        if authorization.get(key) is not False:
            raise TacticalBasisLateralRuntimeProbeDesignError(f"runtime probe must prohibit {key}")

    candidate = _mapping(config.get("candidate"), "candidate")
    if (
        candidate.get("p2_source_id") != P2_SOURCE_ID
        or candidate.get("p2_selected_variant") != P2_SELECTED_VARIANT
        or candidate.get("method_id") != METHODS[1]
        or int(candidate.get("program_steps", -1)) != 40
        or candidate.get("post_program_controller") != "frozen_fixed_head_on"
    ):
        raise TacticalBasisLateralRuntimeProbeDesignError("candidate is not the unique P2-selected program")
    if tuple(_sequence(_mapping(config.get("methods"), "methods").get("order"), "methods.order")) != METHODS:
        raise TacticalBasisLateralRuntimeProbeDesignError("method matrix is not the fixed paired comparison")
    opponents = _mapping(config.get("opponents"), "opponents")
    if tuple(_sequence(opponents.get("order"), "opponents.order")) != OPPONENTS:
        raise TacticalBasisLateralRuntimeProbeDesignError("opponent registry drifted")
    if tuple(_sequence(opponents.get("required_for_gate"), "opponents.required_for_gate")) != REQUIRED_OPPONENTS:
        raise TacticalBasisLateralRuntimeProbeDesignError("required opponents drifted")
    if opponents.get("pooled_gate") != "prohibited":
        raise TacticalBasisLateralRuntimeProbeDesignError("pooled gate must remain prohibited")

    if (
        p2_summary.get("source_id") != P2_SOURCE_ID
        or p2_summary.get("passes") is not True
        or p2_summary.get("selected_variant") != P2_SELECTED_VARIANT
        or p2_summary.get("next_action") != "one_selected_variant_runtime_probe_design_permitted"
        or p2_verification.get("source_id") != P2_SOURCE_ID
        or p2_verification.get("passes") is not True
        or int(p2_verification.get("rows_verified", -1)) != 136
    ):
        raise TacticalBasisLateralRuntimeProbeDesignError("P2 selection provenance is not valid")

    if probe_manifest.get("source_id") != SOURCE_ID + "-MANIFEST12":
        raise TacticalBasisLateralRuntimeProbeDesignError("runtime-probe manifest source ID drifted")
    if probe_manifest.get("parent_manifest_sha256") != config["inputs"]["parent_manifest_sha256"]:
        raise TacticalBasisLateralRuntimeProbeDesignError("runtime-probe manifest parent hash drifted")
    if parent_manifest.get("source_id") != probe_manifest.get("parent_manifest_source_id"):
        raise TacticalBasisLateralRuntimeProbeDesignError("runtime-probe manifest parent source drifted")
    parent_by_name = {
        str(item.get("name")): item
        for item in _sequence(parent_manifest.get("scenarios"), "parent_manifest.scenarios")
        if isinstance(item, Mapping)
    }
    scenarios = _sequence(probe_manifest.get("scenarios"), "probe_manifest.scenarios")
    if len(scenarios) != 12 or len({str(item.get("name")) for item in scenarios if isinstance(item, Mapping)}) != 12:
        raise TacticalBasisLateralRuntimeProbeDesignError("runtime-probe manifest must reference exactly 12 scenarios")
    for item in scenarios:
        entry = _mapping(item, "probe_manifest.scenario")
        name = str(entry.get("name") or "")
        parent = parent_by_name.get(name)
        if parent is None:
            raise TacticalBasisLateralRuntimeProbeDesignError("runtime-probe scenario is absent from the frozen parent")
        parent_metadata = _mapping(parent.get("metadata"), "parent_manifest.metadata")
        if int(entry.get("scenario_seed", -1)) != int(parent_metadata.get("scenario_seed", -2)):
            raise TacticalBasisLateralRuntimeProbeDesignError("runtime-probe scenario seed differs from the frozen parent")

    contract = _mapping(config.get("contract"), "contract")
    if (
        contract.get("backend") != "jsbsim"
        or contract.get("strict_backend") is not True
        or contract.get("action_semantics") != "tactical_basis_v1"
        or contract.get("reset_at_handoff") is not False
        or contract.get("snapshot_restore") is not False
        or contract.get("future_state_injection") is not False
        or contract.get("history_padding") is not False
        or contract.get("routing_enabled") is not False
        or tuple(_sequence(contract.get("target_phases"), "contract.target_phases")) != ("post_merge",)
        or contract.get("target_dynamic_state") != "disadvantage"
    ):
        raise TacticalBasisLateralRuntimeProbeDesignError("runtime causal contract drifted")
    metrics = _mapping(config.get("metrics"), "metrics")
    if int(metrics.get("response_window_steps", -1)) != 40:
        raise TacticalBasisLateralRuntimeProbeDesignError("response window must equal the selected program")
    if float(_mapping(metrics.get("primary_pressure"), "metrics.primary_pressure").get("candidate_minus_baseline_max", 1.0)) >= 0.0:
        raise TacticalBasisLateralRuntimeProbeDesignError("primary pressure metric must require a reduction")
    secondary = _mapping(metrics.get("secondary_geometry"), "metrics.secondary_geometry")
    if not all(float(secondary.get(key, 0.0)) > 0.0 for key in ("range_rate_mean_min_mps", "end_range_min_m", "end_specific_energy_min_m")):
        raise TacticalBasisLateralRuntimeProbeDesignError("secondary geometry MDEs must be positive")
    gates = _mapping(config.get("gates"), "gates")
    if int(gates.get("min_qualifying_pairs_per_required_opponent", -1)) < 4 or int(gates.get("min_distinct_signatures_per_required_opponent", -1)) < 4:
        raise TacticalBasisLateralRuntimeProbeDesignError("pair coverage gate is weaker than the P2 provenance gate")

    return {
        "source_id": SOURCE_ID,
        "candidate_variant": P2_SELECTED_VARIANT,
        "methods": list(METHODS),
        "opponents": list(OPPONENTS),
        "required_opponents": list(REQUIRED_OPPONENTS),
        "scenarios": len(scenarios),
        "planned_pairs": len(OPPONENTS) * len(scenarios),
        "planned_episodes": len(METHODS) * len(OPPONENTS) * len(scenarios),
        "execution_permitted": False,
    }


def is_ego_failure(reason: Any) -> bool:
    """Return whether a terminal event is an ego crash or out-of-bounds event."""

    normalized = str(reason or "").lower()
    return not normalized.startswith("target_") and ("crash" in normalized or "out_of_bounds" in normalized)


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _mean(values: Iterable[float]) -> float | None:
    finite = [float(value) for value in values if _finite(value) is not None]
    return None if not finite else float(sum(finite) / len(finite))


def _window_metrics(record: Mapping[str, Any], *, response_steps: int) -> dict[str, float] | None:
    """Compute post-handoff physical metrics from a complete runtime window."""

    window = record.get("program_window")
    if not isinstance(window, Sequence) or len(window) != response_steps:
        return None
    target_attack = []
    range_rate = []
    range_values = []
    energy_values = []
    for item in window:
        if not isinstance(item, Mapping):
            return None
        values = {
            "target_attack_score": _finite(item.get("target_attack_score")),
            "range_rate_mps": _finite(item.get("range_rate_mps")),
            "range_m": _finite(item.get("range_m")),
            "own_specific_energy_m": _finite(item.get("own_specific_energy_m")),
        }
        if any(value is None for value in values.values()):
            return None
        target_attack.append(float(values["target_attack_score"]))
        range_rate.append(float(values["range_rate_mps"]))
        range_values.append(float(values["range_m"]))
        energy_values.append(float(values["own_specific_energy_m"]))
    return {
        "mean_target_attack_score": float(sum(target_attack) / len(target_attack)),
        "mean_range_rate_mps": float(sum(range_rate) / len(range_rate)),
        "end_range_m": float(range_values[-1]),
        "end_specific_energy_m": float(energy_values[-1]),
    }


def _runtime_window_valid(record: Mapping[str, Any], method: str, *, response_steps: int) -> bool:
    result = record.get("result")
    if not isinstance(result, Mapping):
        return False
    required_true = (
        "strict_jsbsim",
        "handoff_reached",
        "program_complete",
        "no_reset_after_initialization",
        "no_backend_fallback",
        "no_prediction_fallback_in_program",
        "no_history_padding",
        "no_checkpoint_fallback",
        "continuity_hash_chain_complete",
        "finite_program_telemetry",
    )
    if any(result.get(key) is not True for key in required_true):
        return False
    if int(result.get("program_steps", -1)) != response_steps:
        return False
    if method == METHODS[1] and (
        int(result.get("candidate_box_limited_steps", -1)) != 0
        or int(result.get("candidate_slew_limited_steps", -1)) != 0
        or int(result.get("candidate_basis_violation_steps", -1)) != 0
    ):
        return False
    return _window_metrics(record, response_steps=response_steps) is not None


def _runtime_contract_violation(record: Mapping[str, Any], method: str) -> bool:
    """Identify failure modes that invalidate rather than merely fail a pair."""

    result = record.get("result")
    if not isinstance(result, Mapping):
        return True
    if bool(result.get("fatal_contract_violation", False)):
        return True
    if method == METHODS[1] and any(
        int(result.get(key, 0)) > 0
        for key in (
            "candidate_nonfinite_action_steps",
            "candidate_box_limited_steps",
            "candidate_slew_limited_steps",
            "candidate_basis_violation_steps",
        )
    ):
        return True
    return False


def summarize_runtime_records(
    records: Sequence[Mapping[str, Any]],
    *,
    scenarios_per_opponent: int,
    metrics: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute the unpooled paired runtime decision from raw episode records."""

    if scenarios_per_opponent != 12:
        raise TacticalBasisLateralRuntimeProbeDesignError("runtime probe requires the frozen 12-scenario manifest")
    response_steps = int(metrics["response_window_steps"])
    pressure_cfg = _mapping(metrics["primary_pressure"], "metrics.primary_pressure")
    secondary_cfg = _mapping(metrics["secondary_geometry"], "metrics.secondary_geometry")
    safety_cfg = _mapping(metrics["execution_and_safety"], "metrics.execution_and_safety")
    expected = len(METHODS) * len(OPPONENTS) * scenarios_per_opponent
    keyed: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    contract_violations: list[str] = []
    for record in records:
        method = str(record.get("method") or "")
        opponent = str(record.get("opponent") or "")
        scenario = str(record.get("scenario_name") or "")
        if method not in METHODS or opponent not in OPPONENTS or not scenario:
            raise TacticalBasisLateralRuntimeProbeDesignError("runtime record identity is invalid")
        key = (method, opponent, scenario)
        if key in keyed:
            raise TacticalBasisLateralRuntimeProbeDesignError("runtime record matrix has a duplicate key")
        keyed[key] = record
        if _runtime_contract_violation(record, method):
            contract_violations.append("/".join(key))

    pair_rows: list[dict[str, Any]] = []
    matrix: dict[str, dict[str, Any]] = {}
    passed_required: list[str] = []
    for opponent in OPPONENTS:
        baseline_records = [
            row for (method, row_opponent, _scenario), row in keyed.items()
            if method == METHODS[0] and row_opponent == opponent
        ]
        candidate_records = [
            row for (method, row_opponent, _scenario), row in keyed.items()
            if method == METHODS[1] and row_opponent == opponent
        ]
        baseline_by_scenario = {str(row["scenario_name"]): row for row in baseline_records}
        candidate_by_scenario = {str(row["scenario_name"]): row for row in candidate_records}
        valid_pairs: list[dict[str, Any]] = []
        pressure_pairs: list[dict[str, Any]] = []
        handoff_hashes_match = True
        handoff_hash_pairs = 0
        for scenario in sorted(set(baseline_by_scenario) | set(candidate_by_scenario)):
            baseline = baseline_by_scenario.get(scenario)
            candidate = candidate_by_scenario.get(scenario)
            pair: dict[str, Any] = {
                "opponent": opponent,
                "scenario_name": scenario,
                "scenario_signature": None if baseline is None else baseline.get("scenario_signature"),
                "mirror_sign": None if baseline is None else baseline.get("mirror_sign"),
                "baseline_present": baseline is not None,
                "candidate_present": candidate is not None,
                "baseline_window_valid": False if baseline is None else _runtime_window_valid(baseline, METHODS[0], response_steps=response_steps),
                "candidate_window_valid": False if candidate is None else _runtime_window_valid(candidate, METHODS[1], response_steps=response_steps),
                "handoff_hash_match": False,
                "pressure_exposed": False,
                "primary_pressure_delta": None,
                "mean_range_rate_delta_mps": None,
                "end_range_delta_m": None,
                "end_specific_energy_delta_m": None,
            }
            if baseline is not None and candidate is not None:
                baseline_handoff_hash = baseline.get("result", {}).get("handoff_state_hash")
                candidate_handoff_hash = candidate.get("result", {}).get("handoff_state_hash")
                pair["handoff_hash_match"] = (
                    baseline_handoff_hash == candidate_handoff_hash
                    and baseline_handoff_hash is not None
                )
                # Non-handoff episodes are not candidate interventions and cannot
                # invalidate the causal hash check for qualifying pairs.
                if baseline_handoff_hash is not None and candidate_handoff_hash is not None:
                    handoff_hash_pairs += 1
                    handoff_hashes_match &= bool(pair["handoff_hash_match"])
            if pair["baseline_window_valid"] and pair["candidate_window_valid"] and pair["handoff_hash_match"]:
                baseline_metrics = _window_metrics(baseline, response_steps=response_steps)
                candidate_metrics = _window_metrics(candidate, response_steps=response_steps)
                assert baseline_metrics is not None and candidate_metrics is not None
                pair.update(
                    {
                        "pressure_exposed": baseline_metrics["mean_target_attack_score"]
                        >= float(pressure_cfg["baseline_min_mean_target_attack_score"]),
                        "primary_pressure_delta": candidate_metrics["mean_target_attack_score"] - baseline_metrics["mean_target_attack_score"],
                        "mean_range_rate_delta_mps": candidate_metrics["mean_range_rate_mps"] - baseline_metrics["mean_range_rate_mps"],
                        "end_range_delta_m": candidate_metrics["end_range_m"] - baseline_metrics["end_range_m"],
                        "end_specific_energy_delta_m": candidate_metrics["end_specific_energy_m"] - baseline_metrics["end_specific_energy_m"],
                    }
                )
                valid_pairs.append(pair)
                if pair["pressure_exposed"]:
                    pressure_pairs.append(pair)
            pair_rows.append(pair)

        signatures = {str(pair["scenario_signature"]) for pair in valid_pairs if pair.get("scenario_signature")}
        mirrors = {str(pair["mirror_sign"]) for pair in valid_pairs if pair.get("mirror_sign")}
        primary_delta = _mean(pair["primary_pressure_delta"] for pair in pressure_pairs)
        range_rate_delta = _mean(pair["mean_range_rate_delta_mps"] for pair in valid_pairs)
        end_range_delta = _mean(pair["end_range_delta_m"] for pair in valid_pairs)
        end_energy_delta = _mean(pair["end_specific_energy_delta_m"] for pair in valid_pairs)
        candidate_failures = sum(is_ego_failure(row.get("result", {}).get("terminal_reason")) for row in candidate_records)
        baseline_failures = sum(is_ego_failure(row.get("result", {}).get("terminal_reason")) for row in baseline_records)
        coverage_pass = bool(
            len(valid_pairs) >= int(gates["min_qualifying_pairs_per_required_opponent"])
            and len(signatures) >= int(gates["min_distinct_signatures_per_required_opponent"])
            and {"negative", "positive"}.issubset(mirrors)
        )
        pressure_coverage_pass = len(pressure_pairs) >= int(gates["min_pressure_exposed_pairs_per_required_opponent"])
        primary_pass = primary_delta is not None and primary_delta <= float(pressure_cfg["candidate_minus_baseline_max"])
        secondary_passes = {
            "range_rate": range_rate_delta is not None and range_rate_delta >= float(secondary_cfg["range_rate_mean_min_mps"]),
            "range": end_range_delta is not None and end_range_delta >= float(secondary_cfg["end_range_min_m"]),
            "specific_energy": end_energy_delta is not None and end_energy_delta >= float(secondary_cfg["end_specific_energy_min_m"]),
        }
        safety_pass = candidate_failures - baseline_failures <= int(safety_cfg["ego_crash_oob_count_delta_vs_baseline_max"])
        gate_role = "required" if opponent in REQUIRED_OPPONENTS else "diagnostic_only"
        opponent_passes = bool(
            gate_role == "required"
            and coverage_pass
            and pressure_coverage_pass
            and primary_pass
            and any(secondary_passes.values())
            and safety_pass
            and handoff_hashes_match
        )
        if opponent_passes:
            passed_required.append(opponent)
        matrix[opponent] = {
            "gate_role": gate_role,
            "baseline_records": len(baseline_records),
            "candidate_records": len(candidate_records),
            "valid_pairs": len(valid_pairs),
            "pressure_exposed_pairs": len(pressure_pairs),
            "distinct_signatures": sorted(signatures),
            "mirror_signs": sorted(mirrors),
            "handoff_hashes_match": handoff_hashes_match,
            "handoff_hash_pairs": handoff_hash_pairs,
            "mean_primary_pressure_delta": primary_delta,
            "mean_range_rate_delta_mps": range_rate_delta,
            "mean_end_range_delta_m": end_range_delta,
            "mean_end_specific_energy_delta_m": end_energy_delta,
            "secondary_passes": secondary_passes,
            "candidate_ego_crash_oob": candidate_failures,
            "baseline_ego_crash_oob": baseline_failures,
            "safety_pass": safety_pass,
            "coverage_pass": coverage_pass,
            "pressure_coverage_pass": pressure_coverage_pass,
            "primary_pressure_pass": primary_pass,
            "passes": opponent_passes,
        }

    required_passes = [opponent for opponent in REQUIRED_OPPONENTS if matrix[opponent]["passes"]]
    if contract_violations:
        decision = "runtime_contract_failure"
        next_action = "freeze_source_do_not_train_or_rerun"
    elif len(records) != expected:
        decision = "runtime_matrix_incomplete"
        next_action = "freeze_source_do_not_train_or_rerun"
    elif len(required_passes) == len(REQUIRED_OPPONENTS):
        decision = "cross_opponent_authority_established"
        next_action = "single_skill_pilot_preregistration_design_permitted"
    elif required_passes:
        decision = "opponent_conditional_authority"
        next_action = "report_condition_boundary_do_not_train_shared_skills"
    elif any(
        not matrix[opponent]["coverage_pass"] or not matrix[opponent]["pressure_coverage_pass"]
        for opponent in REQUIRED_OPPONENTS
    ):
        decision = "runtime_not_evaluable"
        next_action = "freeze_source_do_not_train_or_rerun"
    else:
        decision = "behavioral_or_control_authority_not_established"
        next_action = "stop_temporal_reference_and_shared_skill_training_line"
    return {
        "records": len(records),
        "expected_records": expected,
        "matrix": matrix,
        "pair_rows": pair_rows,
        "contract_violations": contract_violations,
        "required_opponents_passed": required_passes,
        "passes": decision == "cross_opponent_authority_established",
        "decision": decision,
        "next_action": next_action,
    }
