#!/usr/bin/env python3
"""Verify the isolated source-trace bundle without running JSBSim."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

SOURCE_ID = "THESIS-MILD-DISADVANTAGE-HANDOFF-SOURCE-TRACES-ISOLATED-V1-20260904"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    errors: list[str] = []
    manifest_path = root / "run_manifest.json"
    replay_path = root / "replay_verification.json"
    source_manifest_path = root / "source_trace_manifest.json"
    if not manifest_path.is_file() or not replay_path.is_file() or not source_manifest_path.is_file():
        errors.append("required manifest is missing")
        records = []
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        records = source_manifest.get("records", [])
        if manifest.get("source_id") != SOURCE_ID:
            errors.append("run manifest Source ID mismatch")
        if replay.get("source_id") != SOURCE_ID or not replay.get("passed"):
            errors.append("replay verification did not pass")
        if len(records) != 12:
            errors.append(f"expected 12 records, found {len(records)}")
    keys = set()
    for record in records:
        key = (record.get("opponent"), record.get("candidate"), record.get("repetition"))
        if key in keys:
            errors.append(f"duplicate job {key}")
        keys.add(key)
        path = root / record["opponent"] / "trajectories" / f"{record['candidate']}_{int(record['repetition']):02d}.csv"
        if not path.is_file():
            errors.append(f"missing registered trace {path}")
            continue
        if file_sha256(path) != record.get("registered_trace", {}).get("sha256"):
            errors.append(f"registered trace hash mismatch {path}")
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            errors.append(f"empty registered trace {path}")
            continue
        if any(row.get("backend") != "jsbsim" for row in rows):
            errors.append(f"non-JSBSim row in {path}")
        if any(row.get("backend_fallback_occurred", "False").lower() == "true" for row in rows):
            errors.append(f"backend fallback in {path}")
        previous = rows[0].get("mode", "")
        transition = None
        for index, row in enumerate(rows[1:], start=1):
            if row.get("mode", "") != previous:
                transition = (index, row.get("state_hash_before_action", ""))
                break
            previous = row.get("mode", "")
        if transition is None:
            errors.append(f"no handoff transition in {path}")
        else:
            if transition[0] != int(record.get("handoff_index", -1)):
                errors.append(f"handoff index mismatch in {path}")
            if transition[1] != record.get("handoff_hash"):
                errors.append(f"handoff hash mismatch in {path}")

    result = {
        "ok": not errors,
        "source_id": SOURCE_ID,
        "record_count": len(records),
        "errors": errors,
        "gate": "isolated_source_trace_replay",
    }
    write_json(root / "verifier_result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
