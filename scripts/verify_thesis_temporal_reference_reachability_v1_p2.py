#!/usr/bin/env python3
"""Independently recompute the P2 compiler-envelope decision from its matrix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from uav_vpp_guidance.evaluation.temporal_reference_p2 import summarize_rows


def _load(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise RuntimeError("P2 config must be a mapping")
    return value


def _convert(row: dict[str, str]) -> dict[str, Any]:
    boolean = {"replay_matched", "strict_jsbsim", "baseline_complete", "backend_fallback", "prediction_fallback"}
    integer = {"baseline_steps"}
    numeric = {"action_box_limited_fraction", "slew_limited_fraction", "mean_projection_residual_m", "peak_actual_reference_delta_m", "max_basis_condition_number"}
    result: dict[str, Any] = dict(row)
    for key in boolean:
        result[key] = str(row[key]).lower() == "true"
    for key in integer:
        result[key] = int(row[key])
    for key in numeric:
        result[key] = float(row[key])
    return result


def verify(config_path: Path, output: Path) -> dict[str, Any]:
    config = _load(config_path)
    matrix = output / "offline_envelope_matrix.csv"
    if not matrix.is_file():
        raise RuntimeError("P2 matrix is missing")
    with matrix.open(newline="", encoding="utf-8") as handle:
        rows = [_convert(dict(row)) for row in csv.DictReader(handle)]
    summary = summarize_rows(rows, config["gates"])
    reported = json.loads((output / "p2_verification.json").read_text(encoding="utf-8"))
    expected = {"source_id": config["source_id"], "stage": config["stage"], **summary}
    if reported != expected:
        raise RuntimeError("P2 reported verification does not match the independent recomputation")
    return {"source_id": config["source_id"], "rows": len(rows), "reported_gate_matches_recomputed": True, "passes": summary["passes"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.config, args.output), ensure_ascii=False, indent=2))
