#!/usr/bin/env python3
"""Independent recomputation of V8's physical first-pass-aware phase gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from uav_vpp_guidance.evaluation.raw_si_first_pass_event import RawSIFirstPassConfig
from uav_vpp_guidance.evaluation.raw_si_reentry_phase import RawSIReentryPhaseConfig, RawSIReentryPhaseTracker


SOURCE_ID = "THESIS-FIVE-STATE-REENTRY-PHASE-CONTRACT-PROBE-V8"
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")


def _replay(ledger: dict) -> dict[str, int]:
    contract = ledger.get("v8_phase_contract", {})
    tracker = RawSIReentryPhaseTracker(
        RawSIReentryPhaseConfig(first_pass=RawSIFirstPassConfig(**contract["first_pass_event"]))
    )
    reentry_steps = 0
    reentry_66d_steps = 0
    first_pass_steps = 0
    for index, step in enumerate(ledger["steps"]):
        base = np.asarray(step["base_observation_16d"], dtype=np.float64)
        raw_range = float(base[0] * float(contract["range_reference_m"]))
        raw_rate = float(base[1] * float(contract["range_rate_reference_mps"]))
        phase = tracker.update(raw_range, raw_rate, step_index=index)
        if phase != step.get("v8_phase") or tracker.first_pass_complete != bool(step.get("v8_first_pass_latched")):
            raise ValueError("phase replay mismatch")
        if abs(raw_range - float(step["v8_raw_range_m"])) > 1e-8 or abs(raw_rate - float(step["v8_raw_range_rate_mps"])) > 1e-8:
            raise ValueError("normalized-to-physical unit mismatch")
        if phase == "re_entry":
            reentry_steps += 1
            reentry_66d_steps += int(step.get("observation_66d") is not None)
        first_pass_steps += int(tracker.first_pass_complete)
    return {"first_pass_steps": first_pass_steps, "reentry_steps": reentry_steps, "reentry_66d_steps": reentry_66d_steps}


def verify(root: Path) -> dict:
    rows = []
    for opponent in OPPONENTS:
        paths = sorted((root / "episodes" / opponent).glob("*.json"))
        if len(paths) != 12:
            raise ValueError(f"missing V8 records for {opponent}")
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            record, ledger = payload["record"], payload["ledger"]
            if record.get("source_id") != SOURCE_ID or record.get("opponent") != opponent:
                raise ValueError("record identity mismatch")
            replay = _replay(ledger)
            for key, value in replay.items():
                if int(record["result"][key]) != value:
                    raise ValueError(f"{key} mismatch")
            rows.append(record)
    matrix = {}
    for opponent in OPPONENTS:
        group = [row for row in rows if row["opponent"] == opponent]
        qualifying = [row for row in group if int(row["result"]["reentry_66d_steps"]) > 0]
        cell = {
            "records": len(group),
            "qualifying_episodes": len(qualifying),
            "reentry_66d_steps": sum(int(row["result"]["reentry_66d_steps"]) for row in group),
            "mirrors": len({row["metadata"]["mirror_sign"] for row in qualifying}),
            "signatures": len({row["metadata"]["geometry_cell_id"] for row in qualifying}),
        }
        cell["passes"] = cell["records"] == 12 and cell["qualifying_episodes"] >= 2 and cell["reentry_66d_steps"] >= 20 and cell["mirrors"] == 2 and cell["signatures"] >= 2
        matrix[opponent] = cell
    gate = json.loads((root / "phase_contract_gate.json").read_text(encoding="utf-8"))
    if gate.get("matrix") != matrix:
        raise ValueError("gate matrix mismatch")
    return {"source_id": SOURCE_ID, "verification_status": "pass", "records": len(rows), "matrix": matrix, "passes": all(cell["passes"] for cell in matrix.values())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        result, code = verify(args.output_dir), 0
    except Exception as error:
        result, code = {"source_id": SOURCE_ID, "verification_status": "fail", "error": str(error)}, 2
    text = json.dumps(result, indent=2)
    print(text)
    if args.report:
        args.report.write_text(text + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
