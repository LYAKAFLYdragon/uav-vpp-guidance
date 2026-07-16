#!/usr/bin/env python3
"""Derive the replay-valid V2-only cohort without reselecting outcomes."""
from __future__ import annotations
import argparse, copy, hashlib, json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent
SOURCE_ID = "THESIS-FIVE-STATE-LOCAL-ACTION-AUTHORITY-MAP-V2"

def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=ROOT / "config/experiment/manifests/thesis_local_action_authority_map_v2_cohort12.yaml"); args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("refusing to overwrite V2 replay-valid cohort")
    parent = yaml.safe_load((ROOT / "config/experiment/manifests/thesis_local_action_authority_map_v1_cohort28.yaml").read_text(encoding="utf-8"))
    entries = [copy.deepcopy(entry) for entry in parent["entries"] if entry["historical_source_kind"] == "v2"]
    counts = {opponent: sum(entry["opponent"] == opponent for entry in entries) for opponent in ("expert", "end_to_end", "independent_ppo_vpp")}
    if len(entries) != 12 or counts != {"expert": 5, "end_to_end": 2, "independent_ppo_vpp": 5}: raise RuntimeError(f"unexpected V2 replay-valid cohort: {counts}")
    payload = {"source_id": SOURCE_ID + "-COHORT12", "status": "preregistered_design_only", "selection": "v1_artifact_cohort_filtered_only_by_engineering_replay_compatibility", "parent_cohort": "config/experiment/manifests/thesis_local_action_authority_map_v1_cohort28.yaml", "opponents_reported_separately": True, "counts_by_opponent": counts, "entries": entries}
    payload["payload_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle: yaml.safe_dump(payload, handle, sort_keys=False)
    print(payload["payload_sha256"]); return 0
if __name__ == "__main__": raise SystemExit(main())
