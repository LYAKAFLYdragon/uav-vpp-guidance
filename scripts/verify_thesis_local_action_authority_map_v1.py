#!/usr/bin/env python3
"""Independently recompute the local-action-authority map from raw records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import run_thesis_local_action_authority_map_v1 as runner
from uav_vpp_guidance.evaluation.thesis_local_action_authority_map import METHODS, SOURCE_ID, summarize


def verify(config_path: Path, output: Path | None = None) -> dict:
    config = runner._config(config_path); cohort = runner._cohort(config); root = output or runner._resolve(config["outputs"]["root"]); rows = []
    for entry in cohort["entries"]:
        safe = entry["cohort_key"].replace(":", "__").replace("/", "_")
        for method in METHODS:
            row = json.loads((root / "records" / method / entry["opponent"] / f"{safe}.json").read_text(encoding="utf-8"))["record"]
            if row.get("source_id") != SOURCE_ID or row.get("cohort_key") != entry["cohort_key"] or row.get("method") != method:
                raise RuntimeError("local-authority record identity mismatch")
            rows.append(row)
    gate = summarize(rows, config["gates"]); reported = json.loads((root / "gate.json").read_text(encoding="utf-8"))
    if reported != {"source_id": SOURCE_ID, **gate}: raise RuntimeError("local-authority gate mismatch")
    return {"source_id": SOURCE_ID, "records": len(rows), "reported_gate_matches_recomputed": True, "passes": gate["passes"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--output", type=Path); args = parser.parse_args(); print(json.dumps(verify(args.config, args.output), ensure_ascii=False, indent=2))
