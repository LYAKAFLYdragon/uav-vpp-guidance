#!/usr/bin/env python3
"""Independently verify the Plan 2 non-learning motif-capacity experiment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import run_jsbsim_hrl_comparison as comparison


SOURCE_ID = "THESIS-DISADVANTAGE-CAPABILITY-PLAN2-V1"
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
REFERENCE_METHOD = "fixed_head_on"
CONTROL_METHOD = "fixed_crossing"
MOTIFS = ("lateral_break_same_side", "range_extension_los", "vertical_separation")
METHODS = (REFERENCE_METHOD, CONTROL_METHOD, *MOTIFS)
GRAVITY_MPS2 = 9.80665


class Plan2VerificationError(RuntimeError):
    """Raised when a Plan 2 artifact violates its preregistered contract."""


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(values: Sequence[float]) -> float | None:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else None


def _vector(point: Mapping[str, Any], prefix: str) -> np.ndarray:
    values = [point.get(f"{prefix}_{axis}") for axis in ("x", "y", "z")]
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise Plan2VerificationError(f"trajectory point has invalid {prefix} vector")
    return vector


def _guidance_vector(point: Mapping[str, Any]) -> np.ndarray:
    vector = np.asarray(
        [point.get("nz_cmd"), point.get("roll_rate_cmd"), point.get("throttle_cmd")],
        dtype=np.float64,
    )
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise Plan2VerificationError("trajectory point has invalid guidance receipt")
    return vector


def _specific_energy_height(point: Mapping[str, Any]) -> float:
    values = np.asarray(
        [
            point.get("speed_mps"),
            point.get("target_speed_mps"),
            point.get("altitude_m"),
            point.get("target_altitude_m"),
        ],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise Plan2VerificationError("trajectory point lacks target speed/altitude telemetry")
    own_speed, target_speed, own_altitude, target_altitude = values
    return float(
        own_altitude
        + own_speed**2 / (2.0 * GRAVITY_MPS2)
        - target_altitude
        - target_speed**2 / (2.0 * GRAVITY_MPS2)
    )


def _ego_failure(record: Mapping[str, Any]) -> bool:
    reason = str(record.get("combat_reason") or record.get("terminal_reason") or "").lower()
    return "ego" in reason and ("crash" in reason or "out_of_bounds" in reason or "oob" in reason)


def _warmup_hash(record: Mapping[str, Any], warmup_steps: int) -> str:
    trajectory = record.get("trajectory", [])
    if not isinstance(trajectory, list) or len(trajectory) < warmup_steps:
        raise Plan2VerificationError("record does not contain a complete warm-up trajectory")
    payload = []
    for point in trajectory[:warmup_steps]:
        if not isinstance(point, Mapping):
            raise Plan2VerificationError("trajectory point is invalid")
        payload.append(
            {
                "own": _vector(point, "ego_pos").tolist(),
                "target": _vector(point, "target_pos").tolist(),
                "action": np.asarray(point.get("vpp_action"), dtype=np.float64).tolist(),
                "prediction_valid": bool(point.get("prediction_valid")),
                "prediction_fallback": bool(point.get("prediction_fallback")),
            }
        )
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _candidate_window(
    record: Mapping[str, Any],
    motif: str,
    intervention_steps: int,
) -> list[Mapping[str, Any]]:
    trajectory = record.get("trajectory", [])
    if not isinstance(trajectory, list):
        raise Plan2VerificationError("record trajectory is invalid")
    window = []
    for point in trajectory:
        metadata = point.get("agent_step_metadata")
        if not isinstance(metadata, Mapping):
            continue
        if metadata.get("motif_probe_phase") != "motif_intervention":
            continue
        if metadata.get("motif_probe_name") != motif or metadata.get("motif_probe_uses_future_state") is not False:
            raise Plan2VerificationError("motif metadata contract drifted")
        receipt = metadata.get("motif_probe_receipt")
        if not isinstance(receipt, Mapping) or receipt.get("motif") != motif:
            raise Plan2VerificationError("motif receipt is missing or inconsistent")
        window.append(point)
    if len(window) != intervention_steps:
        raise Plan2VerificationError(
            f"motif window incomplete for {motif}: expected {intervention_steps}, found {len(window)}"
        )
    return window


def _step_receipt(point: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = point.get("agent_step_metadata")
    if not isinstance(metadata, Mapping) or not isinstance(metadata.get("motif_probe_receipt"), Mapping):
        raise Plan2VerificationError("motif receipt is missing")
    return metadata["motif_probe_receipt"]


def _load_config(path: Path) -> dict[str, Any]:
    config = comparison._load_config_with_includes(path)
    protocol = config.get("disadvantage_capability_plan2")
    if not isinstance(protocol, dict) or protocol.get("source_id") != SOURCE_ID:
        raise Plan2VerificationError("Plan 2 source ID is invalid")
    forbidden = (
        "training_permitted",
        "tuning_permitted",
        "reward_change_permitted",
        "vpp_change_permitted",
        "guidance_change_permitted",
        "pid_change_permitted",
        "future_state_or_outcome_routing_permitted",
    )
    if not protocol.get("evaluation_only") or any(protocol.get(key) is not False for key in forbidden):
        raise Plan2VerificationError("Plan 2 authorization contract drifted")
    if tuple(protocol.get("opponents", ())) != OPPONENTS or tuple(protocol.get("methods", ())) != METHODS:
        raise Plan2VerificationError("Plan 2 method/opponent matrix drifted")
    return config


def _validate_manifest(config: Mapping[str, Any]) -> set[str]:
    protocol = config["disadvantage_capability_plan2"]
    manifest_path = ROOT / str(protocol["manifest"])
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    if manifest.get("source_id") != SOURCE_ID or len(manifest.get("scenarios", [])) != 12:
        raise Plan2VerificationError("Plan 2 manifest is invalid")
    if str(protocol.get("manifest_sha256")) != comparison._scenario_manifest_payload_sha256(manifest):
        raise Plan2VerificationError("Plan 2 manifest hash drifted")
    protocol_manifest = manifest.get("protocol", {})
    if protocol_manifest.get("training_permitted") is not False or protocol_manifest.get("future_state_or_outcome_routing") is not False:
        raise Plan2VerificationError("Plan 2 manifest is not evaluation-only")
    return {str(item["name"]) for item in manifest["scenarios"]}


def _summarize_candidate(
    *,
    motif: str,
    candidate_records: Mapping[str, Mapping[str, Any]],
    reference_records: Mapping[str, Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    execution_gate = protocol["gates"]["execution"]
    geometry_gate = protocol["gates"]["geometry"][motif]
    warmup = int(protocol["warmup_high_level_steps"])
    intervention = int(protocol["intervention_high_level_steps"])
    ordered_pairs = [(candidate_records[key], reference_records[key]) for key in sorted(candidate_records)]
    if len(ordered_pairs) != 12:
        raise Plan2VerificationError("candidate universe is incomplete")
    warmups_match = all(_warmup_hash(candidate, warmup) == _warmup_hash(reference, warmup) for candidate, reference in ordered_pairs)
    candidate_windows = [_candidate_window(candidate, motif, intervention) for candidate, _ in ordered_pairs]
    reference_windows = [
        list(reference["trajectory"])[warmup : warmup + intervention]
        for _, reference in ordered_pairs
    ]
    if any(len(window) != intervention for window in reference_windows):
        raise Plan2VerificationError("reference window is incomplete")
    receipts = [_step_receipt(point) for window in candidate_windows for point in window]
    basis_ranks = [int(receipt.get("basis_rank", -1)) for receipt in receipts]
    basis_conditions = [float(receipt.get("basis_condition_number", np.nan)) for receipt in receipts]
    actual_norms = [float(np.linalg.norm(np.asarray(receipt.get("actual_world_delta_m"), dtype=np.float64))) for receipt in receipts]
    residuals = [float(receipt.get("projection_residual_norm_m", np.nan)) for receipt in receipts]
    clip_fraction = _mean([float(bool(receipt.get("action_clipped", False))) for receipt in receipts])
    vpp_deltas = [
        float(np.linalg.norm(_vector(candidate, "world_offset") - _vector(reference, "world_offset")))
        for candidate_window, reference_window in zip(candidate_windows, reference_windows)
        for candidate, reference in zip(candidate_window, reference_window)
    ]
    guidance_deltas = [
        float(np.linalg.norm(_guidance_vector(candidate) - _guidance_vector(reference)))
        for candidate_window, reference_window in zip(candidate_windows, reference_windows)
        for candidate, reference in zip(candidate_window, reference_window)
    ]
    threat_delta = _mean(
        [
            float(bool(candidate.get("target_in_attack_zone", False)))
            - float(bool(reference.get("target_in_attack_zone", False)))
            for candidate_window, reference_window in zip(candidate_windows, reference_windows)
            for candidate, reference in zip(candidate_window, reference_window)
        ]
    )
    range_rate_delta = _mean(
        [
            float(candidate.get("range_rate_mps", np.nan)) - float(reference.get("range_rate_mps", np.nan))
            for candidate_window, reference_window in zip(candidate_windows, reference_windows)
            for candidate, reference in zip(candidate_window, reference_window)
        ]
    )
    altitude_delta = _mean(
        [
            abs(float(candidate.get("altitude_m", np.nan)) - float(candidate.get("target_altitude_m", np.nan)))
            - abs(float(reference.get("altitude_m", np.nan)) - float(reference.get("target_altitude_m", np.nan)))
            for candidate_window, reference_window in zip(candidate_windows, reference_windows)
            for candidate, reference in zip(candidate_window, reference_window)
        ]
    )
    energy_delta = _mean(
        [
            _specific_energy_height(candidate) - _specific_energy_height(reference)
            for candidate_window, reference_window in zip(candidate_windows, reference_windows)
            for candidate, reference in zip(candidate_window, reference_window)
        ]
    )
    candidate_ego_rate = _mean([float(_ego_failure(item)) for item in candidate_records.values()])
    reference_ego_rate = _mean([float(_ego_failure(item)) for item in reference_records.values()])
    candidate_saturation = _mean(
        [
            float(
                bool(point.get("nz_saturated"))
                or bool(point.get("roll_rate_saturated"))
                or bool(point.get("throttle_saturated"))
            )
            for window in candidate_windows
            for point in window
        ]
    )
    reference_saturation = _mean(
        [
            float(
                bool(point.get("nz_saturated"))
                or bool(point.get("roll_rate_saturated"))
                or bool(point.get("throttle_saturated"))
            )
            for window in reference_windows
            for point in window
        ]
    )
    no_fallback = all(
        bool(point.get("prediction_valid")) and not bool(point.get("prediction_fallback"))
        for window in candidate_windows
        for point in window
    ) and all(not bool(record.get("backend_fallback_occurred", False)) for record in candidate_records.values())
    execution_pass = bool(
        warmups_match
        and no_fallback
        and all(rank >= int(execution_gate["min_basis_rank"]) for rank in basis_ranks)
        and all(np.isfinite(value) and value <= float(execution_gate["max_basis_condition_number"]) for value in basis_conditions)
        and (_mean(actual_norms) or 0.0) >= float(execution_gate["min_actual_world_delta_norm_m"])
        and (_mean(vpp_deltas) or 0.0) >= float(execution_gate["min_paired_vpp_delta_norm_m"])
        and (_mean(guidance_deltas) or 0.0) >= float(execution_gate["min_paired_guidance_delta_norm"])
        and (_mean(residuals) is not None and _mean(residuals) <= float(execution_gate["max_projection_residual_norm_m"]))
        and (clip_fraction is not None and clip_fraction <= float(execution_gate["max_action_clip_fraction"]))
        and candidate_ego_rate is not None
        and reference_ego_rate is not None
        and candidate_ego_rate - reference_ego_rate <= float(execution_gate["max_ego_crash_oob_delta_vs_reference"])
        and candidate_saturation is not None
        and reference_saturation is not None
        and candidate_saturation - reference_saturation <= float(execution_gate["max_saturation_fraction_delta_vs_reference"])
    )
    if motif == "lateral_break_same_side":
        geometry_pass = threat_delta is not None and threat_delta <= float(
            geometry_gate["max_target_attack_zone_fraction_delta_vs_reference"]
        )
    elif motif == "range_extension_los":
        geometry_pass = range_rate_delta is not None and range_rate_delta >= float(
            geometry_gate["min_range_rate_delta_mps_vs_reference"]
        )
    else:
        geometry_pass = bool(
            altitude_delta is not None
            and energy_delta is not None
            and altitude_delta >= float(geometry_gate["min_abs_altitude_delta_increase_m_vs_reference"])
            and energy_delta >= float(geometry_gate["min_specific_energy_delta_m_vs_reference"])
        )
    return {
        "warmup_hashes_match": warmups_match,
        "qualified_pairs": len(ordered_pairs),
        "basis_rank_min": min(basis_ranks),
        "basis_condition_number_max": max(basis_conditions),
        "prediction_and_backend_contract_pass": no_fallback,
        "mean_actual_world_delta_norm_m": _mean(actual_norms),
        "mean_paired_vpp_delta_norm_m": _mean(vpp_deltas),
        "mean_paired_guidance_delta_norm": _mean(guidance_deltas),
        "mean_projection_residual_norm_m": _mean(residuals),
        "action_clip_fraction": clip_fraction,
        "target_attack_zone_fraction_delta_vs_reference": threat_delta,
        "range_rate_delta_mps_vs_reference": range_rate_delta,
        "abs_altitude_delta_increase_m_vs_reference": altitude_delta,
        "specific_energy_delta_m_vs_reference": energy_delta,
        "ego_crash_oob_delta_vs_reference": candidate_ego_rate - reference_ego_rate,
        "saturation_fraction_delta_vs_reference": candidate_saturation - reference_saturation,
        "execution_pass": execution_pass,
        "motif_geometry_pass": geometry_pass,
        "motif_pass": bool(execution_pass and geometry_pass),
    }


def verify(config_path: Path, output_root: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    protocol = config["disadvantage_capability_plan2"]
    scenarios = _validate_manifest(config)
    expected = {(scenario, method) for scenario in scenarios for method in METHODS}
    matrix: dict[str, dict[str, Any]] = {motif: {} for motif in MOTIFS}
    control_summary: dict[str, dict[str, Any]] = {}
    for opponent in OPPONENTS:
        run_dir = output_root / f"{SOURCE_ID}-{opponent}"
        failures = _read_json(run_dir / "failures.json").get("failures", [])
        if failures:
            raise Plan2VerificationError(f"Plan 2 run has failures: {opponent}")
        payload = _read_json(run_dir / "aggregate" / "episode_records.json")
        records = payload.get("episodes") if isinstance(payload, Mapping) else None
        if not isinstance(records, list):
            raise Plan2VerificationError(f"Plan 2 records are invalid: {opponent}")
        index: dict[tuple[str, str], Mapping[str, Any]] = {}
        for record in records:
            if not isinstance(record, Mapping):
                raise Plan2VerificationError("Plan 2 record is not a mapping")
            scenario = str(record.get("scenario"))
            method = str(record.get("controller") or record.get("method"))
            if scenario not in scenarios or method not in METHODS:
                raise Plan2VerificationError(f"unexpected Plan 2 record: {opponent}/{scenario}/{method}")
            if record.get("opponent_stage") != opponent or record.get("backend") != "jsbsim" or not bool(record.get("strict_backend")):
                raise Plan2VerificationError(f"backend/opponent contract failed: {opponent}/{scenario}/{method}")
            key = (scenario, method)
            if key in index:
                raise Plan2VerificationError("Plan 2 record universe contains a duplicate")
            index[key] = record
        if set(index) != expected:
            raise Plan2VerificationError(f"Plan 2 record universe is incomplete: {opponent}")
        references = {scenario: index[(scenario, REFERENCE_METHOD)] for scenario in scenarios}
        controls = {scenario: index[(scenario, CONTROL_METHOD)] for scenario in scenarios}
        control_summary[opponent] = {
            "resolved_wins": sum(bool(record.get("win", False)) for record in controls.values()),
            "ego_crash_oob_rate": _mean([float(_ego_failure(record)) for record in controls.values()]),
        }
        for motif in MOTIFS:
            candidates = {scenario: index[(scenario, motif)] for scenario in scenarios}
            matrix[motif][opponent] = _summarize_candidate(
                motif=motif,
                candidate_records=candidates,
                reference_records=references,
                protocol=protocol,
            )
    required = int(protocol["gates"]["geometry"]["candidates_must_pass_opponents"])
    candidates = [
        motif
        for motif in MOTIFS
        if sum(bool(matrix[motif][opponent]["motif_pass"]) for opponent in OPPONENTS) >= required
    ]
    execution_supported = {
        motif: [
            opponent for opponent in OPPONENTS if bool(matrix[motif][opponent]["execution_pass"])
        ]
        for motif in MOTIFS
    }
    if candidates:
        decision = "candidate_low_level_motif_exists"
    elif any(len(items) >= required for items in execution_supported.values()):
        single_opponent_geometry = any(
            any(bool(matrix[motif][opponent]["motif_geometry_pass"]) for opponent in OPPONENTS)
            for motif in MOTIFS
        )
        decision = "opponent_dependent_sensitivity" if single_opponent_geometry else "current_envelope_physical_or_behavioral_reachability_gap"
    else:
        decision = "vpp_interface_transmission_not_established"
    return {
        "source_id": SOURCE_ID,
        "manifest_sha256": protocol["manifest_sha256"],
        "total_records": len(OPPONENTS) * len(METHODS) * len(scenarios),
        "matrix": matrix,
        "fixed_crossing_control": control_summary,
        "candidate_motifs": candidates,
        "execution_supported_opponents": execution_supported,
        "decision": decision,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = verify(args.config.resolve(), args.output_root.resolve())
    if args.write:
        target = args.output_root.resolve() / "plan2_verification.json"
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
