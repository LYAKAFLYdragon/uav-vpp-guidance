#!/usr/bin/env python3
"""Independently recompute the V9 action-authority matrix from raw records."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from uav_vpp_guidance.evaluation.raw_si_first_pass_event import RawSIFirstPassConfig
from uav_vpp_guidance.evaluation.raw_si_reentry_phase import RawSIReentryPhaseConfig, RawSIReentryPhaseTracker
from uav_vpp_guidance.evaluation.thesis_reentry_v9_action_authority import (
    OPPONENTS,
    REFERENCE_METHOD,
    apply_perturbation,
    summarize_records,
)


SOURCE_ID = "THESIS-FIVE-STATE-REENTRY-VPP-ACTION-AUTHORITY-V9"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON mapping: {path}")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return value


def _replay_phase_trace(trace: list[Mapping[str, Any]], first_pass: Mapping[str, Any]) -> None:
    tracker = RawSIReentryPhaseTracker(RawSIReentryPhaseConfig(first_pass=RawSIFirstPassConfig(**dict(first_pass))))
    previous_step = -1
    for item in trace:
        step = int(item["step"])
        if step <= previous_step:
            raise ValueError("V9 phase trace is not strictly increasing")
        previous_step = step
        phase = tracker.update(float(item["raw_range_m"]), float(item["raw_range_rate_mps"]), step_index=step)
        if phase != item["phase"] or tracker.first_pass_complete != bool(item["first_pass_complete"]):
            raise ValueError("V9 raw-SI phase replay mismatch")


def _verify_record(record: Mapping[str, Any], *, config: Mapping[str, Any]) -> None:
    method = str(record.get("method"))
    if record.get("source_id") != SOURCE_ID or method not in config["methods"]:
        raise ValueError("V9 record identity mismatch")
    trace = record.get("phase_trace")
    if not isinstance(trace, list) or not trace:
        raise ValueError("V9 record lacks phase trace")
    _replay_phase_trace(trace, config["observability"]["raw_si_first_pass_event"])
    if not bool(record.get("handoff_reached")):
        if int(record.get("window_steps", 0)) != 0:
            raise ValueError("V9 non-handoff record contains a window")
        return
    handoff_step = int(record["handoff_step"])
    handoff = [item for item in trace if int(item["step"]) == handoff_step]
    if len(handoff) != 1 or handoff[0]["phase"] != "re_entry" or not bool(handoff[0]["first_pass_complete"]):
        raise ValueError("V9 handoff is not a physical re-entry state")
    window = record.get("window")
    if not isinstance(window, list) or len(window) != int(record.get("window_steps", -1)):
        raise ValueError("V9 window length mismatch")
    if len(window) == int(config["observability"]["local_window_steps"]):
        mean_loss = sum(float(item["after_intent_loss"]) for item in window) / len(window)
        if not math.isclose(mean_loss, float(record["intent_loss_auc20"]), abs_tol=1e-12, rel_tol=0.0):
            raise ValueError("V9 intent-loss AUC mismatch")
    delta = config["methods"][method]
    for item in window:
        receipt = apply_perturbation(item["reference_action"], delta)
        if not np.allclose(receipt["applied_action"], item["applied_action"], rtol=0.0, atol=1e-6):
            raise ValueError("V9 applied action mismatch")
        if not math.isclose(receipt["applied_delta_norm"], float(item["applied_delta_norm"]), abs_tol=1e-6, rel_tol=0.0):
            raise ValueError("V9 applied delta norm mismatch")
        vector = np.asarray(item["observation_66d"], dtype=np.float64)
        if vector.shape != (66,) or not np.isfinite(vector).all():
            raise ValueError("V9 saved invalid 66-D observation")


def verify(output_root: Path) -> dict[str, Any]:
    config = _load_yaml(output_root / "authorized_config.yaml")
    if config.get("source_id") != SOURCE_ID:
        raise ValueError("V9 authorized config identity mismatch")
    records = []
    methods = tuple(config["methods"])
    for method in methods:
        for opponent in OPPONENTS:
            paths = sorted((output_root / "episodes" / method / opponent).glob("*.json"))
            if len(paths) != 12:
                raise ValueError(f"V9 record universe incomplete: {method}/{opponent}")
            for path in paths:
                payload = _load_json(path)
                record = payload.get("record")
                if not isinstance(record, Mapping) or record.get("opponent") != opponent or record.get("method") != method:
                    raise ValueError("V9 episode identity mismatch")
                _verify_record(record, config=config)
                records.append(record)
    gates = config["gates"]
    summary = summarize_records(
        records,
        method_deltas=config["methods"],
        min_window_steps=int(config["observability"]["local_window_steps"]),
        min_pairs=int(gates["min_qualifying_pairs_per_opponent"]),
        min_applied_delta_norm=float(gates["min_applied_delta_norm"]),
        practical_delta=float(gates["practical_intent_loss_delta_vs_reference_max"]),
        safety_delta_max=float(gates["ego_crash_oob_delta_vs_reference_max"]),
    )
    gate = _load_json(output_root / "action_authority_gate.json")
    expected = {"source_id": SOURCE_ID, **summary}
    if gate != expected:
        raise ValueError("V9 runner and independent gate differ")
    manifest = _load_json(output_root / "run_manifest.json")
    if int(manifest.get("records", -1)) != len(records) or manifest.get("training") != "prohibited":
        raise ValueError("V9 run manifest drift")
    return {"source_id": SOURCE_ID, "verification_status": "pass", **summary}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        result, code = verify(args.output_root), 0
    except Exception as error:
        result, code = {"source_id": SOURCE_ID, "verification_status": "fail", "error": str(error)}, 2
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.report:
        args.report.write_text(rendered + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
