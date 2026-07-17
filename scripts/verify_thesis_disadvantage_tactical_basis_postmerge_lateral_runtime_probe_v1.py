#!/usr/bin/env python3
"""Independently verify the paired post-merge lateral runtime probe outputs."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from uav_vpp_guidance.evaluation.tactical_basis_continuity_contract import stable_hash
from uav_vpp_guidance.evaluation.tactical_basis_lateral_runtime_probe_contract import (
    METHODS,
    OPPONENTS,
    SOURCE_ID,
    summarize_runtime_records,
)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise RuntimeError(f"YAML mapping required: {path}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON mapping required: {path}")
    return value


def _verify_trace(record: Mapping[str, Any]) -> None:
    trace = record.get("continuity_trace")
    if not isinstance(trace, list) or not trace:
        raise RuntimeError("runtime continuity trace is missing")
    previous = record.get("initial_continuity_hash")
    if not isinstance(previous, str) or not previous:
        raise RuntimeError("runtime initial continuity hash is missing")
    by_step: dict[int, Mapping[str, Any]] = {}
    for item in trace:
        if not isinstance(item, Mapping) or item.get("continuity_prev_hash") != previous:
            raise RuntimeError("runtime continuity hash chain is discontinuous")
        step = int(item.get("step", -1))
        if step <= 0 or step in by_step or not item.get("continuity_hash"):
            raise RuntimeError("runtime continuity trace has invalid indices")
        by_step[step] = item
        previous = str(item["continuity_hash"])
    for payload in record.get("program_window", []):
        if not isinstance(payload, Mapping):
            raise RuntimeError("runtime program payload is malformed")
        trace_item = by_step.get(int(payload.get("step", -1)))
        if trace_item is None or trace_item.get("continuity_hash") != payload.get("continuity_hash"):
            raise RuntimeError("runtime program payload is not anchored in the continuity trace")
        source = dict(payload)
        actual_hash = source.pop("continuity_hash", None)
        source.pop("method", None)
        # The runtime chain hash is formed before these post-step analysis
        # fields are appended. The complete record file remains SHA-256-bound
        # by records_index.json, so both control and analysis telemetry are
        # independently integrity protected.
        for key in (
            "range_m",
            "range_rate_mps",
            "ata_deg",
            "aa_deg",
            "own_speed_mps",
            "own_altitude_m",
            "own_specific_energy_m",
            "target_attack_score",
            "ego_attack_score",
            "target_in_attack_zone",
            "ego_in_attack_zone",
        ):
            source.pop(key, None)
        if actual_hash != stable_hash(source):
            raise RuntimeError("runtime program payload hash mismatch")


def _verify_historical_gate(
    reported: Mapping[str, Any], corrected: Mapping[str, Any]
) -> bool:
    """Accept only the documented post-run aggregation correction.

    The original matrix was produced before the summary rule excluded
    non-handoff episodes from the causal hash aggregate and classified a
    zero-pressure cohort as not evaluable. Raw pair rows and all physical
    metrics must still match exactly.
    """

    if reported == {"source_id": SOURCE_ID, **corrected}:
        return False
    legacy = copy.deepcopy(dict(reported))
    expected = {"source_id": SOURCE_ID, **copy.deepcopy(dict(corrected))}
    for value in (legacy, expected):
        value.pop("decision", None)
        value.pop("next_action", None)
        value.pop("passes", None)
    for opponent in OPPONENTS:
        legacy_cell = legacy.get("matrix", {}).get(opponent)
        expected_cell = expected.get("matrix", {}).get(opponent)
        if not isinstance(legacy_cell, dict) or not isinstance(expected_cell, dict):
            raise RuntimeError("runtime-probe historical gate matrix is malformed")
        legacy_cell.pop("handoff_hashes_match", None)
        expected_cell.pop("handoff_hashes_match", None)
        expected_cell.pop("handoff_hash_pairs", None)
    if legacy != expected:
        raise RuntimeError("runtime-probe historical gate differs beyond the documented aggregation correction")
    return True


def verify(config_path: Path, output: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    if config.get("source_id") != SOURCE_ID:
        raise RuntimeError("runtime-probe config source ID drifted")
    expected_output = _resolve(str(config["outputs"]["root"]))
    if output.resolve() != expected_output.resolve():
        raise RuntimeError("runtime-probe output root differs from frozen config")
    for key in ("p2_summary", "p2_verification", "probe_manifest", "parent_manifest", "temporal_reference_compiler"):
        path = _resolve(str(config["inputs"][key]))
        if not path.is_file() or _hash(path) != str(config["inputs"][key + "_sha256"]):
            raise RuntimeError(f"runtime-probe input hash mismatch: {key}")
    index = _load_json(output / "records_index.json")
    if index.get("source_id") != SOURCE_ID or not isinstance(index.get("records"), list):
        raise RuntimeError("runtime-probe records index is invalid")
    expected_records = len(METHODS) * len(OPPONENTS) * 12
    if len(index["records"]) != expected_records:
        raise RuntimeError("runtime-probe record count drifted")
    records: list[dict[str, Any]] = []
    for item in index["records"]:
        path = output / str(item.get("path") or "")
        if not path.is_file() or _hash(path) != str(item.get("sha256") or ""):
            raise RuntimeError("runtime-probe episode record hash mismatch")
        record = _load_json(path).get("record")
        if not isinstance(record, dict) or record.get("source_id") != SOURCE_ID:
            raise RuntimeError("runtime-probe episode record identity drifted")
        if record.get("method") not in METHODS or record.get("opponent") not in OPPONENTS:
            raise RuntimeError("runtime-probe method or opponent drifted")
        _verify_trace(record)
        records.append(record)
    summary = summarize_runtime_records(records, scenarios_per_opponent=12, metrics=config["metrics"], gates=config["gates"])
    expected_gate = {key: value for key, value in summary.items() if key != "pair_rows"}
    historical_gate_correction = _verify_historical_gate(
        _load_json(output / "runtime_probe_gate.json"), expected_gate
    )
    pairing = _load_json(output / "pairing_audit.json")
    if pairing != {"source_id": SOURCE_ID, "pairs": summary["pair_rows"]}:
        raise RuntimeError("runtime-probe pairing audit differs from independent recomputation")
    with (output / "paired_geometry_deltas.csv").open("r", newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    if len(csv_rows) != len(summary["pair_rows"]):
        raise RuntimeError("runtime-probe paired CSV row count differs from audit")
    manifest = _load_json(output / "run_manifest.json")
    if (
        manifest.get("source_id") != SOURCE_ID
        or int(manifest.get("records", -1)) != expected_records
        or manifest.get("config_sha256") != _hash(config_path)
    ):
        raise RuntimeError("runtime-probe run manifest is incomplete")
    _write_correction = {
        "source_id": SOURCE_ID,
        "kind": "post_run_analysis_correction",
        "raw_records_modified": False,
        "historical_gate_correction_applied": historical_gate_correction,
        "reason": (
            "Causal handoff hashes apply only to pairs that reached handoff; "
            "zero pressure exposure is not evidence of absent control authority."
        ),
        "corrected_gate": {"source_id": SOURCE_ID, **expected_gate},
    }
    (output / "runtime_probe_analysis_correction.json").write_text(
        json.dumps(_write_correction, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    result = {
        "source_id": SOURCE_ID,
        "records_verified": len(records),
        "corrected_decision": summary["decision"],
        "corrected_next_action": summary["next_action"],
        "historical_gate_correction_applied": historical_gate_correction,
        "passes": True,
    }
    (output / "runtime_probe_verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.config, args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
