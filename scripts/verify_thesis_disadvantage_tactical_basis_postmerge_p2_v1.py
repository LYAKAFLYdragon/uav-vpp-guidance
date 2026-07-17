#!/usr/bin/env python3
"""Independently recompute the post-merge tactical-basis P2 envelope screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from uav_vpp_guidance.evaluation.tactical_basis_continuity_contract import record_passes
from uav_vpp_guidance.evaluation.tactical_basis_postmerge_p2 import (
    candidate_variants,
    compile_candidate_row,
    summarize_rows,
)
from uav_vpp_guidance.evaluation.temporal_reference_compiler import TemporalReferenceConfig


SOURCE_ID = "THESIS-DISADVANTAGE-TACTICAL-BASIS-POSTMERGE-ENVELOPE-P2-V1"
CONTINUITY_SOURCE_ID = "THESIS-DISADVANTAGE-TACTICAL-BASIS-CONTINUITY-COLLECTOR-V3"
CONTINUITY_GATES = {
    "required_window_steps": 40,
    "max_basis_condition_number": 100.0,
    "min_qualifying_handoffs_per_opponent": 4,
    "min_distinct_signatures_per_opponent": 4,
    "opponents_required": 2,
}


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise RuntimeError("YAML mapping required")
    return value


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("JSON mapping required")
    return value


def _compiler(reference: dict[str, Any]) -> TemporalReferenceConfig:
    return TemporalReferenceConfig(
        horizon_steps=int(reference["horizon_steps"]),
        high_level_dt_s=float(reference["high_level_dt_s"]),
        max_action_delta=float(reference["max_action_delta"]),
        max_action_slew=float(reference["max_action_slew"]),
        max_basis_condition_number=float(reference["max_basis_condition_number"]),
    )


def verify(config_path: Path, output: Path) -> dict[str, Any]:
    config = _yaml(config_path)
    if config.get("source_id") != SOURCE_ID:
        raise RuntimeError("P2 source ID drifted")
    if output.resolve() != _resolve(str(config["outputs"]["root"])).resolve():
        raise RuntimeError("P2 output root differs from config")
    inputs = config["inputs"]
    for key in (
        "continuity_gate",
        "continuity_verification",
        "continuity_records_index",
        "continuity_config",
        "temporal_reference_design",
        "temporal_reference_compiler",
    ):
        path = _resolve(str(inputs[key]))
        if not path.is_file() or _hash(path) != str(inputs[key + "_sha256"]):
            raise RuntimeError(f"P2 input hash mismatch: {key}")
    gate = _json(_resolve(str(inputs["continuity_gate"])))
    verification = _json(_resolve(str(inputs["continuity_verification"])))
    if gate.get("source_id") != CONTINUITY_SOURCE_ID or gate.get("passes") is not True or verification.get("passes") is not True:
        raise RuntimeError("V3 evidence gate is not valid")
    index = _json(_resolve(str(inputs["continuity_records_index"])))
    root = _resolve(str(inputs["continuity_root"]))
    records = []
    hashes: dict[str, str] = {}
    for item in index.get("records", []):
        path = root / str(item.get("path", ""))
        if not path.is_file() or _hash(path) != str(item.get("sha256", "")):
            raise RuntimeError("V3 source record hash mismatch")
        record = _json(path).get("record")
        if isinstance(record, dict) and record.get("source_id") == CONTINUITY_SOURCE_ID and record_passes(record, CONTINUITY_GATES):
            records.append(record)
            hashes[str(path)] = str(item["sha256"])
    expected = config["input_cohort"]["qualifying_records_per_opponent"]
    actual = {opponent: sum(row["opponent"] == opponent for row in records) for opponent in ("expert", "end_to_end", "independent_ppo_vpp")}
    if actual != expected or len(records) != int(config["input_cohort"]["qualifying_records_total"]):
        raise RuntimeError("V3 qualified cohort drifted")
    reference = _yaml(_resolve(str(inputs["temporal_reference_design"])))['temporal_reference']
    variants = candidate_variants(reference)
    compiler = _compiler(reference)
    rows = []
    for candidate in variants:
        for record in sorted(records, key=lambda item: (str(item["opponent"]), str(item["scenario_name"]))):
            record_path = root / "episodes" / str(record["opponent"]) / (str(record["scenario_name"]) + ".json")
            rows.append(compile_candidate_row(record, candidate, compiler, source_record_sha256=hashes[str(record_path)]))
    matrix = _json(output / "offline_envelope_matrix.json")
    if matrix != {"source_id": SOURCE_ID, "rows": rows}:
        raise RuntimeError("P2 matrix differs from independent recomputation")
    summary = summarize_rows(rows, config["gates"])
    reported = _json(output / "offline_envelope_summary.json")
    if reported != {"source_id": SOURCE_ID, **summary}:
        raise RuntimeError("P2 summary differs from independent recomputation")
    manifest = _json(output / "run_manifest.json")
    if manifest.get("source_id") != SOURCE_ID or manifest.get("rows") != len(rows):
        raise RuntimeError("P2 run manifest is incomplete")
    result = {"source_id": SOURCE_ID, "rows_verified": len(rows), "summary": summary, "passes": True}
    (output / "p2_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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
