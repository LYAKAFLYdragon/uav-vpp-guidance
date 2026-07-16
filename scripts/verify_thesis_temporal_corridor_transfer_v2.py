#!/usr/bin/env python3
"""Independently recompute the V2 temporal-corridor gate from episode records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional


ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import run_thesis_temporal_corridor_transfer_v2 as runner
from uav_vpp_guidance.evaluation.thesis_temporal_corridor_transfer_v2 import (
    METHODS,
    OPPONENTS,
    SOURCE_ID,
    summarize_records,
)


def verify(config_path: Path, output: Optional[Path] = None) -> dict:
    config = runner._config(config_path)
    manifest = runner._load(runner._resolve(str(config["inputs"]["manifest"])))
    root = output or runner._resolve(str(config["outputs"]["root"]))
    records = []
    for method in METHODS:
        for opponent in OPPONENTS:
            for scenario in manifest["scenarios"]:
                path = root / "episodes" / method / opponent / f"{scenario['name']}.json"
                row = json.loads(path.read_text(encoding="utf-8"))["record"]
                if (
                    row.get("source_id") != SOURCE_ID
                    or row.get("method") != method
                    or row.get("opponent") != opponent
                    or row.get("pair_key") != scenario["metadata"]["pair_key"]
                ):
                    raise RuntimeError("V2 record identity mismatch")
                for index, step in enumerate(row.get("window", [])):
                    if int(step.get("local_step", -1)) != index:
                        raise RuntimeError("V2 local-step mismatch")
                    if method == "frozen_fixed_head_on" and step.get("temporal_corridor_receipt") is not None:
                        raise RuntimeError("V2 baseline unexpectedly used the corridor compiler")
                    if method != "frozen_fixed_head_on":
                        receipt = step.get("temporal_corridor_receipt")
                        if not isinstance(receipt, dict) or any(abs(float(value)) > 1.0 + 1e-8 for value in receipt.get("applied_action", [])):
                            raise RuntimeError("V2 temporal-corridor receipt is invalid")
                records.append(row)
    gate = summarize_records(
        records,
        scenarios_per_opponent=len(manifest["scenarios"]),
        window_steps=int(config["observability"]["local_window_steps"]),
        gates=config["gates"],
    )
    reported = json.loads((root / "gate.json").read_text(encoding="utf-8"))
    if reported != {"source_id": SOURCE_ID, **gate}:
        raise RuntimeError("V2 reported gate mismatch")
    return {
        "source_id": SOURCE_ID,
        "records": len(records),
        "reported_gate_matches_recomputed": True,
        "passes": gate["passes"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.config, args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
