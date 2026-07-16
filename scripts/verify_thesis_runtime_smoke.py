#!/usr/bin/env python3
"""Independently verify a disposable three-opponent runtime smoke output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from uav_vpp_guidance.evaluation.thesis_runtime_smoke import OPPONENTS, summarize_records


def verify(output: Path) -> dict:
    records = []
    for opponent in OPPONENTS:
        payload = json.loads((output / "episodes" / f"{opponent}.json").read_text(encoding="utf-8"))
        records.append(payload["record"])
    summary = summarize_records(records)
    reported = json.loads((output / "smoke_gate.json").read_text(encoding="utf-8"))
    expected = {"smoke_id": reported.get("smoke_id"), **summary}
    if reported != expected:
        raise RuntimeError("runtime smoke reported gate differs from independent recomputation")
    manifest = json.loads((output / "smoke_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("scientific_source_id") is not None or manifest.get("not_paper_safe") is not True:
        raise RuntimeError("runtime smoke provenance boundary is missing")
    return {"records": len(records), "reported_gate_matches_recomputed": True, "passes": summary["passes"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.output), ensure_ascii=False, indent=2))
