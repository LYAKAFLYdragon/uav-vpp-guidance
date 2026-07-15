"""Read-only audit for the completed P2-B2 physical preflight."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping

import yaml


ROOT = Path(__file__).resolve().parent.parent
SOURCE_ID = "THESIS-GLOBAL-ADVANTAGE-V1-P2-PHYSICAL-PREFLIGHT-V1"
PHASES = ("pre_merge", "post_merge", "re_entry")
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
DEFAULT_RESULT_ROOT = Path(
    "E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/"
    "global_advantage_v1_p2_physical_preflight"
)
DEFAULT_MANIFEST = (
    ROOT
    / "config"
    / "experiment"
    / "manifests"
    / "thesis_global_advantage_v1_p2_physical_preflight60.yaml"
)
DEFAULT_JSON_OUT = ROOT / "reports" / "thesis_global_advantage_p2_physical_preflight_audit_20260715.json"
DEFAULT_CSV_OUT = ROOT / "reports" / "thesis_global_advantage_p2_physical_preflight_phase_matrix_20260715.csv"


class AuditError(ValueError):
    """Raised when a supposedly frozen P2-B2 artifact cannot be audited."""


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AuditError(f"expected JSON mapping: {path}")
    return payload


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise AuditError(f"expected YAML mapping: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _finite_vector(value: Any, size: int) -> bool:
    if not isinstance(value, list) or len(value) != size:
        return False
    try:
        return all(math.isfinite(float(item)) for item in value)
    except (TypeError, ValueError):
        return False


def _finite_state(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    try:
        return all(math.isfinite(float(value[key])) for key in ("speed_mps", "altitude_m", "nz_g"))
    except (KeyError, TypeError, ValueError):
        return False


def _records(result_root: Path) -> list[dict[str, Any]]:
    episodes = result_root / "episodes"
    paths = sorted(episodes.glob("*/*.json"))
    if not paths:
        raise AuditError(f"no episode artifacts found: {episodes}")
    records = [_load_json(path) for path in paths]
    if len({(record.get("opponent"), record.get("scenario_name")) for record in records}) != len(records):
        raise AuditError("duplicate opponent/scenario episode records")
    return records


def audit_records(manifest: Mapping[str, Any], records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Derive pre-registered physical and phase-observability gates from raw records."""

    scenarios = list(manifest.get("scenarios") or [])
    scenario_by_name = {str(item.get("name")): item for item in scenarios}
    if len(scenarios) != 60 or len(scenario_by_name) != 60:
        raise AuditError("P2-B2 manifest must contain 60 unique scenarios")
    records_list = [dict(record) for record in records]
    expected_records = len(scenarios) * len(OPPONENTS)
    if any(record.get("source_id") != SOURCE_ID for record in records_list):
        raise AuditError("episode source ID drifted")
    if any(str(record.get("opponent")) not in OPPONENTS for record in records_list):
        raise AuditError("episode opponent drifted")
    if any(str(record.get("scenario_name")) not in scenario_by_name for record in records_list):
        raise AuditError("episode scenario is absent from frozen manifest")

    phase_matrix: dict[str, Counter[str]] = {opponent: Counter() for opponent in OPPONENTS}
    state_phase_matrix: dict[tuple[str, str, str], int] = defaultdict(int)
    terminal_mix: dict[str, Counter[str]] = {opponent: Counter() for opponent in OPPONENTS}
    first_phase_counts: Counter[str] = Counter()
    expected_initial_phase_counts: Counter[str] = Counter()
    initial_phase_match_count = 0
    first_pass_count = 0
    valid_records = 0
    finite_steps = 0
    total_steps = 0
    backend_fallback_steps = 0
    prediction_fallback_steps = 0
    phase_predicate_observable_steps = 0
    minimum_step_count = math.inf
    maximum_step_count = 0

    for record in records_list:
        opponent = str(record["opponent"])
        scenario = scenario_by_name[str(record["scenario_name"])]
        metadata = scenario.get("metadata") or {}
        state = str(metadata.get("initial_class"))
        expected_initial_phase = str(metadata.get("phase_at_reset"))
        steps = list(record.get("steps") or [])
        total_steps += len(steps)
        minimum_step_count = min(minimum_step_count, len(steps))
        maximum_step_count = max(maximum_step_count, len(steps))
        valid_records += int(bool(record.get("valid")))
        first_pass_count += int(bool(record.get("first_pass_observed")))
        terminal_mix[opponent][str(record.get("terminal_reason") or "missing")] += 1
        expected_initial_phase_counts[expected_initial_phase] += 1
        if steps:
            first_phase = str(steps[0].get("phase") or "missing")
            first_phase_counts[first_phase] += 1
            initial_phase_match_count += int(first_phase == expected_initial_phase)
        for step in steps:
            phase = str(step.get("phase") or "missing")
            phase_matrix[opponent][phase] += 1
            state_phase_matrix[(state, opponent, phase)] += 1
            finite_steps += int(
                _finite_vector(step.get("action"), 3)
                and _finite_state(step.get("own"))
                and _finite_state(step.get("target"))
            )
            backend_fallback_steps += int(bool(step.get("backend_fallback")))
            prediction_fallback_steps += int(bool(step.get("prediction_fallback")))
            phase_predicate_observable_steps += int(
                "range_m" in step and "range_rate_mps" in step
            )

    phase_rows = [
        {
            "initial_state": state,
            "opponent": opponent,
            "phase": phase,
            "step_count": state_phase_matrix[(state, opponent, phase)],
        }
        for state in sorted({str((item.get("metadata") or {}).get("initial_class")) for item in scenarios})
        for opponent in OPPONENTS
        for phase in PHASES
    ]
    phase_gaps = {
        opponent: [phase for phase in PHASES if phase_matrix[opponent][phase] == 0]
        for opponent in OPPONENTS
    }
    physical_reachability_passed = (
        len(records_list) == expected_records
        and valid_records == expected_records
        and minimum_step_count >= 20
        and finite_steps == total_steps
        and backend_fallback_steps == 0
        and prediction_fallback_steps == 0
    )
    phase_observability_passed = (
        initial_phase_match_count == expected_records
        and phase_predicate_observable_steps == total_steps
        and not any(phase_gaps.values())
    )
    return {
        "source_id": SOURCE_ID,
        "expected_record_count": expected_records,
        "record_count": len(records_list),
        "physical_reachability": {
            "passed": physical_reachability_passed,
            "valid_record_count": valid_records,
            "minimum_step_count": 0 if minimum_step_count == math.inf else int(minimum_step_count),
            "maximum_step_count": maximum_step_count,
            "total_step_count": total_steps,
            "finite_step_count": finite_steps,
            "backend_fallback_step_count": backend_fallback_steps,
            "prediction_fallback_step_count": prediction_fallback_steps,
        },
        "phase_observability": {
            "passed": phase_observability_passed,
            "expected_initial_phase_counts": dict(sorted(expected_initial_phase_counts.items())),
            "first_observed_phase_counts": dict(sorted(first_phase_counts.items())),
            "initial_phase_match_count": initial_phase_match_count,
            "phase_predicate_observable_step_count": phase_predicate_observable_steps,
            "total_step_count": total_steps,
            "phase_matrix": {
                opponent: {phase: phase_matrix[opponent][phase] for phase in PHASES}
                for opponent in OPPONENTS
            },
            "phase_gaps": phase_gaps,
            "first_pass_observed_episode_count": first_pass_count,
            "training_unlocked": False,
        },
        "terminal_mix": {opponent: dict(terminal_mix[opponent]) for opponent in OPPONENTS},
        "state_opponent_phase_rows": phase_rows,
        "decision": (
            "physical_pass_phase_observability_no_go"
            if physical_reachability_passed and not phase_observability_passed
            else "physical_and_phase_pass"
            if physical_reachability_passed and phase_observability_passed
            else "physical_contract_no_go"
        ),
    }


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("initial_state", "opponent", "phase", "step_count")
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV_OUT)
    args = parser.parse_args()

    manifest = _load_yaml(args.manifest)
    result_root = args.result_root
    audit = audit_records(manifest, _records(result_root))
    audit["provenance"] = {
        "result_root": str(result_root),
        "manifest_path": str(args.manifest),
        "manifest_sha256": _sha256(args.manifest),
        "gate_sha256": _sha256(result_root / "physical_preflight_gate.json"),
        "run_manifest_sha256": _sha256(result_root / "run_manifest.json"),
    }
    _write_json(args.json_out, audit)
    _write_csv(args.csv_out, audit["state_opponent_phase_rows"])
    print(json.dumps({key: audit[key] for key in ("source_id", "decision", "physical_reachability", "phase_observability")}, indent=2))
    return 0 if audit["decision"] == "physical_and_phase_pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
