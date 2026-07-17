#!/usr/bin/env python3
"""Build a read-only decision-discriminability ledger for Taxonomy60."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


SOURCE_ID = "THESIS-DUALSKILL-TAXONOMY60-3OPP-V1"
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
METHODS = (
    "fixed_head_on",
    "fixed_crossing",
    "static_oracle_gate",
    "canonical_ppo",
    "ppo_unbootstrapped_crossing",
)
COMPARISONS = {
    "M0_fixed_head_on_vs_M1_fixed_crossing": ("fixed_head_on", "fixed_crossing"),
    "M0_fixed_head_on_vs_M2_static_gate": ("fixed_head_on", "static_oracle_gate"),
    "M1_fixed_crossing_vs_M2_static_gate": ("fixed_crossing", "static_oracle_gate"),
    "M2_static_gate_vs_M3_canonical_ppo": ("static_oracle_gate", "canonical_ppo"),
    "M3_canonical_ppo_vs_M4_no_bootstrap": ("canonical_ppo", "ppo_unbootstrapped_crossing"),
}


class DiscriminabilityError(RuntimeError):
    pass


def _finite_vector(value: Any, size: int = 3) -> bool:
    return (
        isinstance(value, list)
        and len(value) == size
        and all(isinstance(item, (int, float)) and math.isfinite(float(item)) for item in value)
    )


def _frame_decision(record: Mapping[str, Any], frame: Mapping[str, Any]) -> tuple[str, list[float]]:
    action = frame.get("vpp_action")
    if not _finite_vector(action):
        raise DiscriminabilityError("record has an invalid first VPP action")
    mode_id = frame.get("commander_mode_id")
    if isinstance(mode_id, int):
        mode_by_id = {0: "head_on", 1: "crossing", 2: "recovery"}
        if mode_id not in mode_by_id:
            raise DiscriminabilityError("record has an unknown commander mode")
        mode = mode_by_id[mode_id]
    else:
        runtime_key = frame.get("runtime_specialist_key")
        if runtime_key in {"head_on", "head_on_specialist"}:
            mode = "head_on"
        elif runtime_key in {"crossing", "crossing_feasible", "crossing_specialist"}:
            mode = "crossing"
        elif runtime_key in {"recovery", "recovery_specialist"}:
            mode = "recovery"
        elif record.get("controller") == "fixed_head_on":
            mode = "head_on"
        elif record.get("controller") == "fixed_crossing":
            mode = "crossing"
        else:
            raise DiscriminabilityError("record has no effective-mode receipt")
    return mode, [float(value) for value in action]


def _first_decision(record: Mapping[str, Any]) -> tuple[str, list[float]]:
    trajectory = record.get("trajectory")
    if not isinstance(trajectory, list) or not trajectory or not isinstance(trajectory[0], Mapping):
        raise DiscriminabilityError("record lacks first telemetry frame")
    return _frame_decision(record, trajectory[0])


def _different_action(left: Sequence[float], right: Sequence[float]) -> bool:
    return any(abs(a - b) > 1e-9 for a, b in zip(left, right))


def _first_episode_decision_difference(left: Mapping[str, Any], right: Mapping[str, Any]) -> int | None:
    left_frames, right_frames = left.get("trajectory"), right.get("trajectory")
    if not isinstance(left_frames, list) or not isinstance(right_frames, list):
        raise DiscriminabilityError("record lacks trajectory telemetry")
    for step, (left_frame, right_frame) in enumerate(zip(left_frames, right_frames), start=1):
        if not isinstance(left_frame, Mapping) or not isinstance(right_frame, Mapping):
            raise DiscriminabilityError("record has a malformed telemetry frame")
        left_mode, left_action = _frame_decision(left, left_frame)
        right_mode, right_action = _frame_decision(right, right_frame)
        if left_mode != right_mode or _different_action(left_action, right_action):
            return step
    return None


def pair_row(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    comparison: str,
    left_guarded: bool,
    right_guarded: bool,
) -> dict[str, Any]:
    """Return the preregistered first-decision evidence for one method pair."""
    if comparison not in COMPARISONS:
        raise DiscriminabilityError("unknown comparison")
    left_mode, left_action = _first_decision(left)
    right_mode, right_action = _first_decision(right)
    mode_different = left_mode != right_mode
    action_different = _different_action(left_action, right_action)
    distinct = mode_different or action_different
    episode_difference_step = _first_episode_decision_difference(left, right)
    episode_distinct = episode_difference_step is not None
    guarded = left_guarded or right_guarded
    if comparison == "M3_canonical_ppo_vs_M4_no_bootstrap":
        disposition = (
            "excluded_treatment_identical"
            if not distinct
            else "unexpected_distinguishable_requires_new_source"
        )
        routing_eligible = False
    elif comparison == "M2_static_gate_vs_M3_canonical_ppo":
        routing_eligible = episode_distinct and not guarded
        disposition = (
            "routing_attribution_eligible"
            if routing_eligible
            else "deployed_interface_only"
            if guarded
            else "excluded_not_distinguishable"
        )
    else:
        routing_eligible = False
        disposition = "included_distinguishable" if distinct else "included_non_discriminative"
    metadata = left.get("scenario_metadata")
    if not isinstance(metadata, Mapping):
        raise DiscriminabilityError("record lacks scenario metadata")
    return {
        "scenario_id": left.get("scenario"),
        "initial_class": metadata.get("initial_class"),
        "opponent": left.get("opponent_stage"),
        "comparison": comparison,
        "left_method": left.get("controller"),
        "right_method": right.get("controller"),
        "left_outcome": left.get("outcome"),
        "right_outcome": right.get("outcome"),
        "left_terminal_reason": left.get("terminal_reason"),
        "right_terminal_reason": right.get("terminal_reason"),
        "left_first_effective_mode": left_mode,
        "right_first_effective_mode": right_mode,
        "left_first_vpp_action": json.dumps(left_action),
        "right_first_vpp_action": json.dumps(right_action),
        "first_mode_different": mode_different,
        "first_vpp_action_different": action_different,
        "first_decision_distinguishable": distinct,
        "episode_decision_distinguishable": episode_distinct,
        "first_episode_decision_difference_step": episode_difference_step,
        "left_guard_confounded": left_guarded,
        "right_guard_confounded": right_guarded,
        "routing_attribution_eligible": routing_eligible,
        "paper_matrix_disposition": disposition,
    }


def _guard_keys(route_decision: Mapping[str, Any]) -> set[tuple[str, str, str]]:
    values = route_decision.get("guard_confounded_episode_ids", [])
    if not isinstance(values, list):
        raise DiscriminabilityError("guard receipt is malformed")
    keys: set[tuple[str, str, str]] = set()
    for value in values:
        parts = str(value).split("|")
        if len(parts) != 3:
            raise DiscriminabilityError("guard receipt contains an invalid identity")
        keys.add((parts[0], parts[1], parts[2]))
    return keys


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DiscriminabilityError("{} is not a JSON object".format(path.name))
    return value


def load_formal_records(root: Path) -> tuple[dict[tuple[str, str, str], dict[str, Any]], set[tuple[str, str, str]], str]:
    verification = _read_json(root / "formal_verification.json")
    route_decision = _read_json(root / "taxonomy60_route_decision.json")
    execution = _read_json(root / "formal_execution_manifest.json")
    if (
        verification.get("source_id") != SOURCE_ID
        or verification.get("verdict") != "pass"
        or int(verification.get("records", 0)) != 900
        or route_decision.get("source_id") != SOURCE_ID
        or route_decision.get("formal_validity_verdict") != "pass"
        or execution.get("source_id") != SOURCE_ID
        or execution.get("execution_status") != "completed"
    ):
        raise DiscriminabilityError("formal source receipts are not valid")
    guard_keys = _guard_keys(route_decision)
    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    digest = hashlib.sha256()
    for opponent in OPPONENTS:
        raw_root = root / "{}-{}".format(SOURCE_ID, opponent) / "raw"
        paths = sorted(raw_root.rglob("episode_*.json")) if raw_root.is_dir() else []
        if len(paths) != 300:
            raise DiscriminabilityError("{} has {} raw records rather than 300".format(opponent, len(paths)))
        for path in paths:
            payload = path.read_bytes()
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(hashlib.sha256(payload).hexdigest().encode("ascii"))
            record = json.loads(payload)
            if not isinstance(record, dict):
                raise DiscriminabilityError("raw episode is not an object")
            scenario = str(record.get("scenario", ""))
            method = str(record.get("controller", ""))
            key = (scenario, opponent, method)
            if (
                not scenario
                or method not in METHODS
                or record.get("opponent_stage") != opponent
                or record.get("backend") != "jsbsim"
                or record.get("strict_backend") is not True
                or record.get("backend_fallback_occurred") is not False
                or key in records
            ):
                raise DiscriminabilityError("raw record violates the frozen evidence contract")
            _first_decision(record)
            records[key] = record
    if len(records) != 900:
        raise DiscriminabilityError("raw records are not the exact 900-record universe")
    return records, guard_keys, digest.hexdigest()


def build_pair_rows(
    records: Mapping[tuple[str, str, str], Mapping[str, Any]], guard_keys: set[tuple[str, str, str]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for opponent in OPPONENTS:
        scenarios = sorted({key[0] for key in records if key[1] == opponent})
        if len(scenarios) != 60:
            raise DiscriminabilityError("{} lacks the 60-scenario frozen envelope".format(opponent))
        for scenario in scenarios:
            for comparison, (left_method, right_method) in COMPARISONS.items():
                left_key = (scenario, opponent, left_method)
                right_key = (scenario, opponent, right_method)
                if left_key not in records or right_key not in records:
                    raise DiscriminabilityError("method pair is incomplete")
                rows.append(
                    pair_row(
                        records[left_key],
                        records[right_key],
                        comparison=comparison,
                        left_guarded=left_key in guard_keys,
                        right_guarded=right_key in guard_keys,
                    )
                )
    return rows


def summary_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["opponent"]), str(row["comparison"]))].append(row)
    result: list[dict[str, Any]] = []
    for (opponent, comparison), group in sorted(grouped.items()):
        result.append(
            {
                "opponent": opponent,
                "comparison": comparison,
                "paired_scenarios": len(group),
                "first_mode_different_pairs": sum(bool(row["first_mode_different"]) for row in group),
                "first_vpp_action_different_pairs": sum(bool(row["first_vpp_action_different"]) for row in group),
                "first_decision_distinguishable_pairs": sum(bool(row["first_decision_distinguishable"]) for row in group),
                "episode_decision_distinguishable_pairs": sum(bool(row["episode_decision_distinguishable"]) for row in group),
                "guard_confounded_pairs": sum(
                    bool(row["left_guard_confounded"]) or bool(row["right_guard_confounded"]) for row in group
                ),
                "routing_attribution_eligible_pairs": sum(bool(row["routing_attribution_eligible"]) for row in group),
                "excluded_treatment_identical_pairs": sum(
                    row["paper_matrix_disposition"] == "excluded_treatment_identical" for row in group
                ),
            }
        )
    return result


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise DiscriminabilityError("refusing to write an empty evidence file")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_analysis(root: Path, analysis_dir: Path, *, overwrite: bool = False) -> dict[str, Any]:
    if analysis_dir.exists() and any(analysis_dir.iterdir()) and not overwrite:
        raise DiscriminabilityError("analysis directory already exists; use --overwrite only for derived outputs")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    records, guard_keys, raw_digest = load_formal_records(root)
    rows = build_pair_rows(records, guard_keys)
    summary = summary_rows(rows)
    if len(rows) != 900 or len(summary) != 15:
        raise DiscriminabilityError("derived pair matrix is incomplete")
    m3_m4 = [row for row in summary if row["comparison"] == "M3_canonical_ppo_vs_M4_no_bootstrap"]
    m2_m3 = [row for row in summary if row["comparison"] == "M2_static_gate_vs_M3_canonical_ppo"]
    m2_m3_eligible = {row["opponent"]: row["routing_attribution_eligible_pairs"] for row in m2_m3}
    decision = {
        "derived_analysis_id": "THESIS-DUALSKILL-TAXONOMY60-DISCRIMINABILITY-CLOSURE-R1",
        "source_id": SOURCE_ID,
        "analysis_kind": "read_only_first_decision_discriminability",
        "raw_record_count": len(records),
        "raw_joint_digest": raw_digest,
        "guard_confounded_record_count": len(guard_keys),
        "m0_m2_status": "retain_as_frozen_specialist_and_static_interface_comparisons",
        "m2_m3_status": (
            "retain_only_guard_free_dynamic-decision pairs for routing attribution"
            if any(m2_m3_eligible.values())
            else "stop_pure_m2_m3_routing_attribution"
        ),
        "m3_m4_status": (
            "stop_m3_m4_formal_ablation"
            if all(row["first_decision_distinguishable_pairs"] == 0 for row in m3_m4)
            else "unexpected_difference_requires_new_independent_source"
        ),
        "m2_m3_guard_free_dynamic_decision_pairs": m2_m3_eligible,
        "boundary": "No policy, simulator, configuration, raw episode, metric, or outcome is changed by this derived analysis.",
    }
    _write_csv(analysis_dir / "pair_ledger.csv", rows)
    _write_csv(analysis_dir / "discriminability_summary.csv", summary)
    (analysis_dir / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        result = export_analysis(args.output_root.resolve(), args.analysis_dir.resolve(), overwrite=args.overwrite)
    except (OSError, ValueError, TypeError, KeyError, DiscriminabilityError) as error:
        print(json.dumps({"verdict": "fail", "error": str(error)}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
