#!/usr/bin/env python3
"""Run one independent matched-telemetry source episode.

This wrapper keeps the existing non-learning matched telemetry collector and
its contracts, but restricts it to one opponent, one candidate, and one
repetition.  A fresh process per episode prevents JSBSim/FCS reset drift from
being mistaken for a physical response.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MATCHED = ROOT / "scripts/run_mild_disadvantage_envelope_recovery_matched_telemetry_v1.py"
DEFAULT_CONFIG = ROOT / "config/experiment/mild_disadvantage_envelope_recovery_matched_telemetry_v1.yaml"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--opponent", choices=("expert", "end_to_end"), required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--execute", action="store_true", help="Required to execute JSBSim")
    args = parser.parse_args()

    module = load_module(MATCHED, "matched_single_episode_runner")
    spec = module._load(args.config.resolve())
    if not args.execute:
        print("LOCKED: no JSBSim execution")
        return 0

    candidates = spec["probe"]["candidates"]
    if args.candidate not in candidates:
        raise SystemExit(f"unknown candidate {args.candidate!r}")
    spec["probe"]["candidates"] = {args.candidate: candidates[args.candidate]}
    spec["probe"]["repetitions_per_candidate_and_opponent"] = 1
    spec["probe"]["evaluation_seed_base"] = int(args.seed)

    root = args.output_dir.resolve()
    if root.exists() and any(root.rglob("*")):
        raise SystemExit(f"refusing to overwrite non-empty output directory: {root}")
    root.mkdir(parents=True, exist_ok=True)

    base = module._load(module._resolve(spec["base_config"]))
    matched = module._load_module(module.MATCHED_RUNNER, "matched_single_episode_helpers")
    legacy = matched._legacy_module()
    rows, manifest = module._run_opponent(
        spec,
        base,
        args.opponent,
        root,
        matched,
        legacy,
    )
    if len(rows) != 1:
        raise RuntimeError(f"expected one episode, got {len(rows)}")
    print(
        json.dumps(
            {
                "opponent": args.opponent,
                "candidate": args.candidate,
                "seed": args.seed,
                "episode_count": len(rows),
                "backend_effective": manifest.get("backend_effective"),
                "backend_fallback_occurred": manifest.get("backend_fallback_occurred"),
                "output_dir": str(root),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
