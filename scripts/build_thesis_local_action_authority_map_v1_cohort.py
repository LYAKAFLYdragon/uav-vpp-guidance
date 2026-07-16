#!/usr/bin/env python3
"""Freeze V15/V2 artifact-qualified handoffs for the local-authority map."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent.parent
SOURCE_ID = "THESIS-FIVE-STATE-LOCAL-ACTION-AUTHORITY-MAP-V1"
HISTORICAL = {
    "v15": ROOT.parent / "uav-vpp-guidance-five-state-heldout-envelope-v1-results" / "five_state_reentry_v15_fixed_encoder_temporal_corridor",
    "v2": ROOT.parent / "uav-vpp-guidance-five-state-heldout-envelope-v1-results" / "temporal_corridor_transfer_v2",
}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build() -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for source_kind, root in HISTORICAL.items():
        config_path = root / "authorized_config.json"
        config = _load_json(config_path)["config"]
        for path in sorted((root / "episodes" / "frozen_fixed_head_on").rglob("*.json")):
            record = _load_json(path)["record"]
            if not (record.get("handoff_reached") and record.get("contract_passed") and int(record.get("window_steps", 0)) == 20):
                continue
            entries.append(
                {
                    "cohort_key": f"{source_kind}:{record['opponent']}:{record['pair_key']}",
                    "historical_source_kind": source_kind,
                    "historical_source_id": record["source_id"],
                    "historical_artifact_root": str(root),
                    "historical_authorized_config": str(config_path),
                    "historical_authorized_config_sha256": _hash(config_path),
                    "historical_record": str(path),
                    "historical_record_sha256": _hash(path),
                    "historical_handoff_hash": record["handoff_state_hash"],
                    "historical_handoff_step": record.get("handoff_step"),
                    "opponent": record["opponent"],
                    "pair_key": record["pair_key"],
                    "scenario_signature": next(
                        item["metadata"]["scenario_signature"]
                        for item in yaml.safe_load((ROOT / config["inputs"]["manifest"]).read_text(encoding="utf-8"))["scenarios"]
                        if item["metadata"]["pair_key"] == record["pair_key"]
                    ),
                }
            )
    if len(entries) != 28 or len({entry["cohort_key"] for entry in entries}) != 28:
        raise RuntimeError("local-authority cohort must contain 28 unique qualified handoffs")
    counts = {opponent: sum(entry["opponent"] == opponent for entry in entries) for opponent in ("expert", "end_to_end", "independent_ppo_vpp")}
    if counts != {"expert": 12, "end_to_end": 5, "independent_ppo_vpp": 11}:
        raise RuntimeError(f"unexpected cohort opponent counts: {counts}")
    payload = {
        "source_id": SOURCE_ID + "-COHORT28",
        "status": "preregistered_design_only",
        "selection": "historical_fixed_head_on_handoff_reached_and_contract_passed_and_window20",
        "opponents_reported_separately": True,
        "counts_by_opponent": counts,
        "entries": entries,
    }
    payload["payload_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "config/experiment/manifests/thesis_local_action_authority_map_v1_cohort28.yaml")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite the frozen local-authority cohort")
    payload = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(yaml.safe_dump(payload, sort_keys=False))
    print(payload["payload_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
