#!/usr/bin/env python3
"""Independently recompute the replay-valid V2 local-authority map."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import run_thesis_local_action_authority_map_v2 as runner
from uav_vpp_guidance.evaluation.thesis_local_action_authority_map import METHODS, summarize

def verify(config_path: Path, output: Path | None = None) -> dict:
    config = runner.base._config(config_path); cohort = runner.base._cohort(config); root = output or runner.base._resolve(config["outputs"]["root"]); rows = []
    for entry in cohort["entries"]:
        safe = entry["cohort_key"].replace(":", "__").replace("/", "_")
        for method in METHODS:
            row = json.loads((root / "records" / method / entry["opponent"] / f"{safe}.json").read_text(encoding="utf-8"))["record"]
            if row.get("source_id") != runner.SOURCE_ID or row.get("cohort_key") != entry["cohort_key"] or row.get("method") != method: raise RuntimeError("V2 local-authority record identity mismatch")
            rows.append(row)
    gate = summarize(rows, config["gates"]); reported = json.loads((root / "gate.json").read_text(encoding="utf-8"))
    if reported != {"source_id": runner.SOURCE_ID, **gate}: raise RuntimeError("V2 local-authority gate mismatch")
    return {"source_id": runner.SOURCE_ID, "records": len(rows), "reported_gate_matches_recomputed": True, "passes": gate["passes"]}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--output", type=Path); args = parser.parse_args(); print(json.dumps(verify(args.config, args.output), ensure_ascii=False, indent=2))
