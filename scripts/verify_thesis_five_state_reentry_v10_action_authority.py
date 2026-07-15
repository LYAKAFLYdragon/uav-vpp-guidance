#!/usr/bin/env python3
"""Independent verifier for V10's telemetry-hardened action-authority probe."""
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
    apply_perturbation,
    summarize_records,
)


SOURCE_ID = "THESIS-FIVE-STATE-REENTRY-VPP-ACTION-AUTHORITY-V10"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping: {path}")
    return value


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping: {path}")
    return value


def _finite(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"V10 required finite field is absent: {name}")
    return float(value)


def _replay_phase(trace: list[Mapping[str, Any]], event: Mapping[str, Any]) -> None:
    tracker = RawSIReentryPhaseTracker(RawSIReentryPhaseConfig(first_pass=RawSIFirstPassConfig(**dict(event))))
    previous = -1
    for item in trace:
        step = int(item["step"])
        if step <= previous:
            raise ValueError("V10 phase trace is not strictly increasing")
        previous = step
        phase = tracker.update(float(item["raw_range_m"]), float(item["raw_range_rate_mps"]), step_index=step)
        if phase != item["phase"] or tracker.first_pass_complete != bool(item["first_pass_complete"]):
            raise ValueError("V10 raw-SI phase replay mismatch")


def _verify_record(record: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    if record.get("source_id") != SOURCE_ID:
        raise ValueError("V10 record source mismatch")
    trace = record.get("phase_trace")
    if not isinstance(trace, list) or not trace:
        raise ValueError("V10 record lacks phase trace")
    _replay_phase(trace, config["observability"]["raw_si_first_pass_event"])
    if not bool(record.get("handoff_reached")):
        if int(record.get("window_steps", 0)) != 0:
            raise ValueError("V10 non-handoff record contains local-window telemetry")
        return
    handoff = [item for item in trace if int(item["step"]) == int(record["handoff_step"])]
    if len(handoff) != 1 or handoff[0]["phase"] != "re_entry" or not bool(handoff[0]["first_pass_complete"]):
        raise ValueError("V10 handoff does not meet physical re-entry contract")
    window = record.get("window")
    if not isinstance(window, list) or len(window) != int(record.get("window_steps", -1)):
        raise ValueError("V10 local-window length mismatch")
    if len(window) == int(config["observability"]["local_window_steps"]):
        mean_loss = sum(_finite(step.get("after_intent_loss"), "after_intent_loss") for step in window) / len(window)
        if not math.isclose(mean_loss, _finite(record.get("intent_loss_auc20"), "intent_loss_auc20"), abs_tol=1e-12, rel_tol=0.0):
            raise ValueError("V10 intent-loss AUC mismatch")
    required = tuple(config["telemetry_contract"]["required_finite"])
    guidance_keys = tuple(config["telemetry_contract"]["guidance_command_required_finite"])
    for step in window:
        receipt = apply_perturbation(step["reference_action"], config["methods"][record["method"]])
        if not np.allclose(receipt["applied_action"], step["applied_action"], rtol=0.0, atol=1e-6):
            raise ValueError("V10 action receipt mismatch")
        for key in required:
            _finite(step.get(key), key)
        guidance = step.get("guidance_command")
        if not isinstance(guidance, Mapping):
            raise ValueError("V10 guidance receipt is missing")
        for key in guidance_keys:
            _finite(guidance.get(key), "guidance_command." + key)
        vertical = step.get("vp_vertical_bias_m")
        if vertical is not None:
            _finite(vertical, "vp_vertical_bias_m")
        if bool(step.get("vp_vertical_bias_observed")) != (vertical is not None):
            raise ValueError("V10 vertical observability receipt mismatch")
        proxy = step.get("vertical_response_proxy")
        if not isinstance(proxy, Mapping):
            raise ValueError("V10 vertical response proxy is missing")
        for key in config["telemetry_contract"]["vertical_response_proxies"]:
            _finite(proxy.get(key), "vertical_response_proxy." + key)
        vector = np.asarray(step.get("observation_66d"), dtype=np.float64)
        if vector.shape != (66,) or not np.isfinite(vector).all():
            raise ValueError("V10 66-D telemetry mismatch")


def verify(output_root: Path) -> dict[str, Any]:
    config = _yaml(output_root / "authorized_config.yaml")
    if config.get("source_id") != SOURCE_ID:
        raise ValueError("V10 authorized config source mismatch")
    records = []
    for method in config["methods"]:
        for opponent in OPPONENTS:
            paths = sorted((output_root / "episodes" / method / opponent).glob("*.json"))
            if len(paths) != 12:
                raise ValueError(f"V10 record universe incomplete: {method}/{opponent}")
            for path in paths:
                record = _json(path).get("record")
                if not isinstance(record, Mapping) or record.get("method") != method or record.get("opponent") != opponent:
                    raise ValueError("V10 record identity mismatch")
                _verify_record(record, config)
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
    expected_gate = {"source_id": SOURCE_ID, **summary}
    if _json(output_root / "action_authority_gate.json") != expected_gate:
        raise ValueError("V10 runner and verifier gates differ")
    manifest = _json(output_root / "run_manifest.json")
    if manifest.get("training") != "prohibited" or int(manifest.get("records", -1)) != len(records):
        raise ValueError("V10 run manifest mismatch")
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
