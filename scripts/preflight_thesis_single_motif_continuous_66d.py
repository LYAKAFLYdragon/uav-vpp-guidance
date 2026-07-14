#!/usr/bin/env python3
"""Fail-closed, non-executing preflight for the single-motif collector."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from uav_vpp_guidance.evaluation.single_motif_continuous_66d import (  # noqa: E402
    build_single_motif_plan,
)


DEFAULT_CONFIG = ROOT / "config" / "experiment" / "jsbsim_hrl_thesis_single_motif_continuous_66d_r1.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    plan = build_single_motif_plan(config)
    checkpoint_exists = plan.p3_checkpoint.is_file()
    checkpoint_sha = _sha256(plan.p3_checkpoint) if checkpoint_exists else None
    result = {
        "source_id": plan.source_id,
        "mode": "preflight_only_no_jsbsim_no_training",
        "passed": checkpoint_exists and checkpoint_sha == plan.p3_checkpoint_sha256,
        "p3_checkpoint": str(plan.p3_checkpoint),
        "p3_checkpoint_exists": checkpoint_exists,
        "p3_checkpoint_sha256": checkpoint_sha,
        "expected_p3_checkpoint_sha256": plan.p3_checkpoint_sha256,
        "opponents": list(plan.opponents),
        "target": {
            "state": plan.target_state,
            "phases": list(plan.target_phases),
            "skill": plan.target_skill,
            "profile": plan.target_profile,
        },
        "gate": {
            "minimum_qualifying_episodes_per_opponent": plan.min_episodes_per_opponent,
            "minimum_valid_target_steps_per_opponent": plan.min_target_steps_per_opponent,
            "minimum_distinct_scenario_signatures_per_opponent": plan.min_signatures_per_opponent,
            "minimum_distinct_mirror_signs_per_opponent": plan.min_mirror_signs_per_opponent,
        },
        "training_authorized": False,
    }
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
