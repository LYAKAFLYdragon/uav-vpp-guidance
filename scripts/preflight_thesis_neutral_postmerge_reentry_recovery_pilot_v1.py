#!/usr/bin/env python3
"""Validate the neutral post-merge pilot design without executing it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from uav_vpp_guidance.evaluation.thesis_neutral_postmerge_reentry_recovery_pilot import (  # noqa: E402
    NeutralPostMergePilotContractError,
    validate_design,
)


DEFAULT_CONFIG = ROOT / "config" / "experiment" / "thesis_neutral_postmerge_reentry_recovery_pilot_v1.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    try:
        print(json.dumps(validate_design(args.config), indent=2, ensure_ascii=False))
    except (OSError, ValueError, NeutralPostMergePilotContractError) as error:
        print(f"INVALID neutral post-merge pilot design: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
