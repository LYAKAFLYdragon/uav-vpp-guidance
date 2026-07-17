#!/usr/bin/env python3
"""Independently verify the Plan 1 frozen routing-capacity experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import run_jsbsim_hrl_comparison as comparison


SOURCE_ID = "THESIS-DISADVANTAGE-CAPABILITY-PLAN1-V1"
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
PRIMARY_OPPONENTS = ("expert", "end_to_end")
METHODS = ("canonical_ppo", "fixed_head_on", "fixed_crossing", "phase_rule_head_to_cross")
FIXED_METHODS = ("fixed_head_on", "fixed_crossing")


class Plan1VerificationError(RuntimeError):
    """Raised when a frozen Plan 1 artifact violates its contract."""


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _outcome(record: Mapping[str, Any]) -> str:
    if bool(record.get("win")):
        return "win"
    if bool(record.get("loss")):
        return "loss"
    if bool(record.get("draw")):
        return "draw"
    return "unresolved"


def _ordinal(record: Mapping[str, Any]) -> int:
    return {"loss": 0, "draw": 1, "unresolved": 1, "win": 2}[_outcome(record)]


def _target_attack_zone_seconds(record: Mapping[str, Any]) -> float:
    trajectory = [step for step in record.get("trajectory", []) if isinstance(step, Mapping)]
    if not trajectory:
        raise Plan1VerificationError("raw trajectory is missing")
    duration = float(record.get("total_time_s", 0.0))
    return duration * sum(bool(step.get("target_in_attack_zone", False)) for step in trajectory) / len(trajectory)


def _ego_crash_or_oob(record: Mapping[str, Any]) -> bool:
    reason = str(record.get("combat_reason") or record.get("termination_reason") or "").lower()
    return "ego" in reason and ("crash" in reason or "out_of_bounds" in reason or "oob" in reason)


def _validate_phase_rule_record(record: Mapping[str, Any]) -> dict[str, Any]:
    trajectory = [step for step in record.get("trajectory", []) if isinstance(step, Mapping)]
    if not trajectory:
        raise Plan1VerificationError("phase-rule raw trajectory is missing")
    phases = []
    selections = []
    for step in trajectory:
        observed_first_pass = step.get("phase_rule_first_pass_complete")
        selected = step.get("phase_rule_selected_specialist")
        if observed_first_pass is None or selected is None:
            raise Plan1VerificationError("phase-rule telemetry is incomplete")
        if step.get("phase_rule_uses_future_state") is not False:
            raise Plan1VerificationError("phase-rule future-state contract is violated")
        expected = "crossing_feasible" if bool(observed_first_pass) else "head_on"
        if str(selected) != expected:
            raise Plan1VerificationError("phase-rule selected a specialist inconsistent with observed phase")
        phases.append(bool(observed_first_pass))
        selections.append(str(selected))
    switches = sum(left != right for left, right in zip(selections, selections[1:]))
    if switches > 1:
        raise Plan1VerificationError("phase-rule switched more than once")
    if bool(phases[0]) and selections[0] != "crossing_feasible":
        raise Plan1VerificationError("phase-rule initial post-merge selection is invalid")
    return {
        "switches": switches,
        "first_pass_observed": any(phases),
        "trajectory_steps": len(trajectory),
    }


def _validate_temporal_schedule_record(record: Mapping[str, Any], initial_steps: int) -> dict[str, Any]:
    trajectory = [step for step in record.get("trajectory", []) if isinstance(step, Mapping)]
    if not trajectory:
        raise Plan1VerificationError("temporal-schedule raw trajectory is missing")
    selections = []
    for index, step in enumerate(trajectory, start=1):
        selected = step.get("temporal_schedule_selected_specialist")
        if selected is None or step.get("temporal_schedule_uses_future_state") is not False:
            raise Plan1VerificationError("temporal-schedule telemetry is incomplete")
        expected = "head_on" if index <= initial_steps else "crossing_feasible"
        if str(selected) != expected:
            raise Plan1VerificationError("temporal schedule selected a specialist inconsistent with its predeclared step")
        if int(step.get("temporal_schedule_initial_high_level_steps", -1)) != initial_steps:
            raise Plan1VerificationError("temporal schedule warm-up contract drifted")
        selections.append(str(selected))
    switches = sum(left != right for left, right in zip(selections, selections[1:]))
    if switches > 1:
        raise Plan1VerificationError("temporal schedule switched more than once")
    return {"switches": switches, "trajectory_steps": len(trajectory), "switch_observed": switches == 1}


def _best_fixed(records: Mapping[str, Mapping[str, Any]]) -> tuple[str, Mapping[str, Any]]:
    candidates = [(method, records[method]) for method in FIXED_METHODS]
    return max(
        candidates,
        key=lambda item: (
            _ordinal(item[1]),
            -_target_attack_zone_seconds(item[1]),
            item[0],
        ),
    )


def _load_config(path: Path) -> dict[str, Any]:
    config = comparison._load_config_with_includes(path)
    protocol = config.get("disadvantage_capability_plan1", {})
    source_id = str(protocol.get("source_id") or "")
    if source_id not in {SOURCE_ID, "THESIS-DISADVANTAGE-CAPABILITY-PLAN1-V2"}:
        raise Plan1VerificationError("Plan 1 source ID is not recognized")
    if not protocol.get("evaluation_only") or any(
        protocol.get(key) is not False
        for key in (
            "training_permitted",
            "tuning_permitted",
            "reward_change_permitted",
            "vpp_change_permitted",
            "guidance_change_permitted",
            "pid_change_permitted",
            "future_state_or_outcome_routing_permitted",
        )
    ):
        raise Plan1VerificationError("Plan 1 authorization contract drifted")
    if tuple(protocol.get("opponents", ())) != OPPONENTS:
        raise Plan1VerificationError("Plan 1 opponent matrix drifted")
    methods = tuple(protocol.get("methods", ()))
    schedule_method = str(protocol.get("route_schedule_method") or "phase_rule_head_to_cross")
    if len(methods) != 4 or not {"canonical_ppo", "fixed_head_on", "fixed_crossing", schedule_method}.issubset(methods):
        raise Plan1VerificationError("Plan 1 method/opponent matrix drifted")
    return config


def verify(config_path: Path, output_root: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    protocol = config["disadvantage_capability_plan1"]
    manifest_path = ROOT / str(protocol["manifest"])
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    source_id = str(protocol["source_id"])
    methods = tuple(str(item) for item in protocol["methods"])
    schedule_method = str(protocol.get("route_schedule_method") or "phase_rule_head_to_cross")
    schedule_type = str(protocol.get("schedule_type") or "phase_first_pass")
    if manifest.get("source_id") != source_id or len(manifest.get("scenarios", [])) != 12:
        raise Plan1VerificationError("Plan 1 manifest is invalid")
    if str(protocol.get("manifest_sha256")) != comparison._scenario_manifest_payload_sha256(manifest):
        raise Plan1VerificationError("Plan 1 manifest hash drifted")
    scenarios = {str(item["name"]) for item in manifest["scenarios"]}
    expected = {(scenario, method) for scenario in scenarios for method in methods}
    by_opponent: dict[str, dict[tuple[str, str], Mapping[str, Any]]] = {}
    phase_rule_checks: dict[str, list[dict[str, Any]]] = {}
    for opponent in OPPONENTS:
        run_dir = output_root / f"{source_id}-{opponent}"
        failures = _read_json(run_dir / "failures.json").get("failures", [])
        if failures:
            raise Plan1VerificationError(f"Plan 1 run has failures: {opponent}")
        payload = _read_json(run_dir / "aggregate" / "episode_records.json")
        records = payload.get("episodes") if isinstance(payload, Mapping) else None
        if not isinstance(records, list):
            raise Plan1VerificationError(f"Plan 1 records are invalid: {opponent}")
        index: dict[tuple[str, str], Mapping[str, Any]] = {}
        checks: list[dict[str, Any]] = []
        for record in records:
            if not isinstance(record, Mapping):
                raise Plan1VerificationError("Plan 1 record is not a mapping")
            scenario = str(record.get("scenario"))
            method = str(record.get("controller") or record.get("method"))
            if scenario not in scenarios or method not in methods:
                raise Plan1VerificationError(f"Plan 1 unexpected record: {opponent}/{scenario}/{method}")
            if record.get("opponent_stage") != opponent or record.get("backend") != "jsbsim" or not bool(record.get("strict_backend")):
                raise Plan1VerificationError(f"Plan 1 backend/opponent contract failed: {opponent}/{scenario}/{method}")
            key = (scenario, method)
            if key in index:
                raise Plan1VerificationError(f"Plan 1 duplicate record: {opponent}/{key}")
            index[key] = record
            if method == schedule_method:
                if schedule_type == "phase_first_pass":
                    checks.append({"scenario": scenario, **_validate_phase_rule_record(record)})
                elif schedule_type == "temporal_warmup":
                    warmup = int(protocol["temporal_schedule_contract"]["initial_high_level_steps"])
                    checks.append({"scenario": scenario, **_validate_temporal_schedule_record(record, warmup)})
                else:
                    raise Plan1VerificationError("Plan 1 schedule type is unknown")
        if set(index) != expected:
            raise Plan1VerificationError(f"Plan 1 record universe incomplete: {opponent}")
        by_opponent[opponent] = index
        phase_rule_checks[opponent] = checks

    gate_cfg = protocol["gates"]["routing_explanation"]
    opponent_summary: dict[str, dict[str, Any]] = {}
    for opponent, records in by_opponent.items():
        improvements = 0
        exposure_deltas = []
        fixed_ego_failures = []
        phase_ego_failures = []
        paired_rows = []
        for scenario in sorted(scenarios):
            row = {method: records[(scenario, method)] for method in methods}
            fixed_name, fixed = _best_fixed(row)
            schedule = row[schedule_method]
            improvement = _ordinal(schedule) > _ordinal(fixed) and _outcome(schedule) == "win"
            improvements += int(improvement)
            exposure_delta = _target_attack_zone_seconds(fixed) - _target_attack_zone_seconds(schedule)
            exposure_deltas.append(exposure_delta)
            fixed_ego_failures.append(float(_ego_crash_or_oob(fixed)))
            phase_ego_failures.append(float(_ego_crash_or_oob(schedule)))
            paired_rows.append(
                {
                    "scenario": scenario,
                    "selected_fixed_baseline": fixed_name,
                    "route_schedule_outcome": _outcome(schedule),
                    "best_fixed_outcome": _outcome(fixed),
                    "route_schedule_resolved_improvement": improvement,
                    "opponent_attack_zone_delta_s_schedule_minus_best_fixed": -exposure_delta,
                }
            )
        mean_exposure_reduction = sum(exposure_deltas) / len(exposure_deltas)
        ego_delta = sum(phase_ego_failures) / len(phase_ego_failures) - sum(fixed_ego_failures) / len(fixed_ego_failures)
        minimum_improvements = int(
            gate_cfg.get("min_schedule_resolved_improvements_per_primary")
            if "min_schedule_resolved_improvements_per_primary" in gate_cfg
            else gate_cfg["min_phase_rule_resolved_improvements_per_primary"]
        )
        minimum_exposure_reduction = float(
            gate_cfg.get("min_opponent_attack_zone_time_reduction_s_per_primary")
            if "min_opponent_attack_zone_time_reduction_s_per_primary" in gate_cfg
            else gate_cfg["min_attack_zone_time_reduction_s_per_primary"]
        )
        passes = (
            improvements >= minimum_improvements
            and mean_exposure_reduction >= minimum_exposure_reduction
            and ego_delta <= float(gate_cfg["max_ego_crash_oob_delta_vs_best_fixed"])
        )
        opponent_summary[opponent] = {
            "records": len(records),
            "route_schedule_resolved_improvements_vs_best_fixed": improvements,
            "mean_opponent_attack_zone_time_reduction_s": mean_exposure_reduction,
            "ego_crash_oob_delta_vs_best_fixed": ego_delta,
            "passes_practical_routing_gate": passes,
            "paired_rows": paired_rows,
        }
    primary_passes = [opponent_summary[item]["passes_practical_routing_gate"] for item in PRIMARY_OPPONENTS]
    routing_supported = all(primary_passes)
    return {
        "source_id": source_id,
        "manifest_sha256": protocol["manifest_sha256"],
        "total_records": sum(len(rows) for rows in by_opponent.values()),
        "route_schedule_checks": phase_rule_checks,
        "route_schedule_method": schedule_method,
        "opponents": opponent_summary,
        "primary_opponents": list(PRIMARY_OPPONENTS),
        "routing_explanation_supported": routing_supported,
        "next_action": (
            "stop_new_low_level_skill_line; routing/composition is the supported explanation"
            if routing_supported
            else "proceed_to_nonlearning_motif_feasibility_only; do_not_patch_routing"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = verify(args.config.resolve(), args.output_root.resolve())
    if args.write:
        target = args.output_root.resolve() / "plan1_verification.json"
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
