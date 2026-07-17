#!/usr/bin/env python3
"""Independently verify the paired post-merge lateral runtime probe outputs."""

from __future__ import annotations

import argparse
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
        if actual_hash != stable_hash(source):
            raise RuntimeError("runtime program payload hash mismatch")


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
    if _load_json(output / "runtime_probe_gate.json") != {"source_id": SOURCE_ID, **expected_gate}:
        raise RuntimeError("runtime-probe gate differs from independent recomputation")
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
    result = {
        "source_id": SOURCE_ID,
        "records_verified": len(records),
        "decision": summary["decision"],
        "next_action": summary["next_action"],
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
