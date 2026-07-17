#!/usr/bin/env python3
"""Apply the formal `(AA, ATA)` taxonomy convention to frozen V1 telemetry.

This is a read-only correction: it reads hashed V1 episode records, leaves
them unchanged, and writes a separate receipt with its own input hashes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from uav_vpp_guidance.evaluation.engagement_geometry_taxonomy import classify_relative_geometry  # noqa: E402


RUNNER_PATH = ROOT / "scripts" / "run_thesis_disadvantage_pressure_exposed_postmerge_collector_v1.py"
CORRECTION_ID = "THESIS-DISADVANTAGE-PRESSURE-EXPOSED-POSTMERGE-TAXONOMY-CORRECTION-R1"


class CorrectionError(RuntimeError):
    """Raised when V1 raw evidence cannot support a read-only correction."""


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runner():
    spec = importlib.util.spec_from_file_location("pressure_collector", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise CorrectionError("cannot import collector runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def correct(output_root: Path, config_path: Path) -> dict[str, Any]:
    runner = _runner()
    config = runner.load_config(config_path)
    index_path = output_root / "records_index.json"
    verification_path = output_root / "pressure_gate_verification.json"
    if not index_path.is_file() or not verification_path.is_file():
        raise CorrectionError("V1 output must first pass the independent verifier")
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if verification.get("passes") is not True or int(verification.get("records_verified", 0)) != 48:
        raise CorrectionError("V1 independent verification is not a complete pass")
    index = json.loads(index_path.read_text(encoding="utf-8")).get("records", [])
    records = []
    corrections = []
    for entry in index:
        path = output_root / str(entry.get("path") or "")
        if not path.is_file() or _hash(path) != entry.get("sha256"):
            raise CorrectionError(f"V1 record hash mismatch: {path}")
        record = json.loads(path.read_text(encoding="utf-8")).get("record")
        if not isinstance(record, Mapping):
            raise CorrectionError(f"malformed V1 record: {path}")
        corrected = copy.deepcopy(record)
        trace = corrected.get("phase_trace")
        if not isinstance(trace, list) or not trace:
            raise CorrectionError(f"missing V1 phase trace: {path}")
        original_initial = str(trace[0].get("dynamic_state"))
        for step in trace:
            if not isinstance(step, Mapping):
                raise CorrectionError(f"malformed V1 step: {path}")
            step["dynamic_state"] = classify_relative_geometry(step.get("aa_deg"), step.get("ata_deg"))
        window = runner.find_first_pressure_window(trace, config["gates"])
        corrected["handoff_window"] = [] if window is None else window
        result = corrected.get("result", {})
        if not isinstance(result, dict):
            raise CorrectionError(f"malformed V1 result: {path}")
        result["handoff_reached"] = window is not None
        result["handoff_step"] = None if window is None else window[0]["step"]
        result["window_complete"] = window is not None and len(window) == int(config["gates"]["required_window_steps"])
        result["no_prediction_fallback"] = window is not None and not any(step["prediction_fallback"] for step in window)
        corrected["result"] = result
        records.append(corrected)
        corrections.append(
            {
                "record_path": str(entry["path"]),
                "record_sha256": entry["sha256"],
                "opponent": corrected["opponent"],
                "scenario_name": corrected["scenario_name"],
                "initial_dynamic_state_original": original_initial,
                "initial_dynamic_state_corrected": trace[0]["dynamic_state"],
                "qualifying_window_corrected": window is not None,
                "qualifying_window_start_step_corrected": None if window is None else window[0]["step"],
            }
        )
    summary = runner.summarize_records(records, config["gates"])
    return {
        "correction_id": CORRECTION_ID,
        "source_id": runner.SOURCE_ID,
        "kind": "read_only_dynamic_taxonomy_parameter_order_correction",
        "taxonomy_argument_order": ["AA", "ATA"],
        "source_records_index_sha256": _hash(index_path),
        "source_verification_sha256": _hash(verification_path),
        "source_records": len(records),
        "corrections": corrections,
        "corrected_gate": summary,
        "interpretation_boundary": "No simulator, action, policy, reward, scenario, or original episode file was changed. This receipt corrects an analysis-only label contract and is not the original V1 gate receipt.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result_path = args.output_root / "pressure_gate_taxonomy_correction_r1.json"
    if result_path.exists():
        parser.error("refusing to overwrite an existing correction receipt")
    try:
        result = correct(args.output_root.resolve(), args.config.resolve())
    except CorrectionError as error:
        parser.error(str(error))
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "corrections"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
