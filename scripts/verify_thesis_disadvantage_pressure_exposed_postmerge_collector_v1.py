#!/usr/bin/env python3
"""Independently verify the frozen pressure-exposed post-merge collector."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parent.parent
RUNNER_PATH = ROOT / "scripts" / "run_thesis_disadvantage_pressure_exposed_postmerge_collector_v1.py"


class VerificationError(RuntimeError):
    """Raised when the collected matrix cannot be independently reconciled."""


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runner():
    spec = importlib.util.spec_from_file_location("pressure_collector_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise VerificationError("cannot import collector runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify(output: Path, config_path: Path) -> dict[str, Any]:
    runner = _runner()
    design = runner.validate_design(config_path, check_external_inputs=True)
    output = output.resolve()
    manifest_path = output / "run_manifest.json"
    gate_path = output / "pressure_gate.json"
    index_path = output / "records_index.json"
    if not all(path.is_file() for path in (manifest_path, gate_path, index_path)):
        raise VerificationError("collector output lacks a required receipt")
    run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stored_gate = json.loads(gate_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if (
        run_manifest.get("source_id") != runner.SOURCE_ID
        or run_manifest.get("config_sha256") != _hash(config_path)
        or int(run_manifest.get("records", -1)) != int(design["planned_records"])
        or stored_gate.get("source_id") != runner.SOURCE_ID
    ):
        raise VerificationError("run manifest provenance does not match the authorized config")
    entries = index.get("records")
    if not isinstance(entries, list) or len(entries) != int(design["planned_records"]):
        raise VerificationError("records index does not cover the complete formal matrix")
    records = []
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        path = output / str(entry.get("path") or "")
        if not path.is_file() or _hash(path) != entry.get("sha256"):
            raise VerificationError(f"record hash mismatch: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        record = payload.get("record")
        if not isinstance(record, Mapping):
            raise VerificationError(f"record payload malformed: {path}")
        identity = (str(record.get("opponent")), str(record.get("scenario_name")))
        if identity in seen:
            raise VerificationError(f"duplicate matrix identity: {identity}")
        seen.add(identity)
        if record.get("source_id") != runner.SOURCE_ID or record.get("initial_state") != "disadvantage":
            raise VerificationError(f"record source/state drift: {identity}")
        records.append(record)
    expected = {
        (opponent, scenario["name"])
        for opponent in runner.OPPONENTS
        for scenario in runner._load_yaml(runner._resolve(str(runner.load_config(config_path)["inputs"]["scenario_manifest"]))) ["scenarios"]
    }
    if seen != expected:
        raise VerificationError("record identities do not match the frozen scenario-opponent matrix")
    recomputed = runner.summarize_records(records, runner.load_config(config_path)["gates"])
    stored_without_source = {key: value for key, value in stored_gate.items() if key != "source_id"}
    if recomputed != stored_without_source:
        raise VerificationError("stored gate does not match independent recomputation")
    return {
        "source_id": runner.SOURCE_ID,
        "verification_kind": "independent_output_hash_and_gate_recomputation",
        "records_verified": len(records),
        "config_sha256": _hash(config_path),
        "gate_recomputed": recomputed,
        "passes": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify(args.output_root, args.config.resolve())
    except VerificationError as error:
        parser.error(str(error))
    result_path = args.output_root / "pressure_gate_verification.json"
    if result_path.exists():
        parser.error("refusing to overwrite existing verification receipt")
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
