#!/usr/bin/env python3
"""Independently verify tactical-basis continuity collector artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from uav_vpp_guidance.evaluation.tactical_basis_continuity_contract import (
    OPPONENTS,
    stable_hash,
    summarize_records,
)


SOURCE_ID = "THESIS-DISADVANTAGE-TACTICAL-BASIS-CONTINUITY-COLLECTOR-V1"


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


def _verify_chain(record: Mapping[str, Any]) -> None:
    trace = record.get("continuity_trace")
    if not isinstance(trace, list) or not trace:
        raise RuntimeError("continuity trace is missing")
    previous = record.get("initial_continuity_hash")
    if not isinstance(previous, str) or not previous:
        raise RuntimeError("initial continuity hash is missing")
    seen_steps: set[int] = set()
    for item in trace:
        if not isinstance(item, Mapping) or item.get("continuity_prev_hash") != previous:
            raise RuntimeError("continuity hash chain is discontinuous")
        step = int(item.get("step", -1))
        if step in seen_steps or step <= 0:
            raise RuntimeError("continuity trace has invalid step indices")
        seen_steps.add(step)
        previous = str(item.get("continuity_hash") or "")
        if not previous:
            raise RuntimeError("continuity trace lacks hash")
    trace_by_step = {int(item["step"]): item for item in trace}
    for item in record.get("handoff_window", []):
        if not isinstance(item, Mapping):
            raise RuntimeError("handoff window entry is malformed")
        trace_item = trace_by_step.get(int(item.get("step", -1)))
        if trace_item is None or trace_item.get("continuity_hash") != item.get("continuity_hash"):
            raise RuntimeError("handoff window is not anchored in the continuity trace")
        payload = dict(item)
        actual_hash = payload.pop("continuity_hash", None)
        if actual_hash != stable_hash(payload):
            raise RuntimeError("handoff window telemetry hash mismatch")


def verify(config_path: Path, output: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    if config.get("source_id") != SOURCE_ID:
        raise RuntimeError("collector config source ID drifted")
    expected_output = Path(str(config["outputs"]["root"]))
    if output.resolve() != expected_output.resolve():
        raise RuntimeError("verifier output root differs from frozen config")
    index = _load_json(output / "records_index.json")
    if index.get("source_id") != SOURCE_ID or not isinstance(index.get("records"), list):
        raise RuntimeError("records index is invalid")
    expected_records = len(OPPONENTS) * 12
    if len(index["records"]) != expected_records:
        raise RuntimeError("collector record count drifted")
    records: list[dict[str, Any]] = []
    for item in index["records"]:
        path = output / str(item.get("path", ""))
        if not path.is_file() or _hash(path) != str(item.get("sha256", "")):
            raise RuntimeError("record hash mismatch")
        payload = _load_json(path)
        record = payload.get("record")
        if not isinstance(record, dict) or record.get("source_id") != SOURCE_ID:
            raise RuntimeError("record identity drifted")
        if record.get("opponent") not in OPPONENTS:
            raise RuntimeError("record opponent drifted")
        if record.get("continuity_trace"):
            _verify_chain(record)
        records.append(record)
    summary = summarize_records(records, config["gates"])
    reported = _load_json(output / "continuity_gate.json")
    if reported != {"source_id": SOURCE_ID, **summary}:
        raise RuntimeError("reported continuity gate differs from independent recomputation")
    manifest = _load_json(output / "run_manifest.json")
    if manifest.get("source_id") != SOURCE_ID or manifest.get("records") != expected_records:
        raise RuntimeError("run manifest is incomplete")
    result = {
        "source_id": SOURCE_ID,
        "records_verified": len(records),
        "gate_recomputed": summary,
        "passes": True,
    }
    (output / "continuity_verification.json").write_text(
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
