#!/usr/bin/env python3
"""Preflight or execute the authorized defensive-extension feasibility pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from uav_vpp_guidance.training.thesis_defext_rangeext_pilot import (  # noqa: E402
    load_authorized_config,
    run_pilot,
    validate_authorization,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.preflight == args.run:
        raise ValueError("select exactly one of --preflight or --run")
    config = load_authorized_config(args.config)
    if args.preflight:
        print(json.dumps(validate_authorization(config), indent=2, ensure_ascii=False))
        return 0
    result = run_pilot(config)
    print(json.dumps({
        "source_id": result["source_id"],
        "training_steps": result["training"]["training_steps"],
        "stopped": result["training"]["stopped"],
        "heldout_executed": result["heldout_executed"],
        "verdict": result["decision"]["verdict"] if result["decision"] else None,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
