#!/usr/bin/env python3
"""Run only a separately authorized V2 neutral/post-merge pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from uav_vpp_guidance.training.thesis_neutral_postmerge_reentry_recovery_pilot_v2 import (  # noqa: E402
    NeutralPostMergePilotV2Error,
    preflight,
    run,
)


DEFAULT_CONFIG = ROOT / "config" / "experiment" / "thesis_neutral_postmerge_reentry_recovery_pilot_v2_authorized.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        if args.preflight:
            print(json.dumps(preflight(args.config), indent=2, ensure_ascii=False))
            return 0
        if not args.execute:
            print(json.dumps({"source_id": "THESIS-NEUTRAL-POSTMERGE-REENTRY-RECOVERY-PILOT-V2", "mode": "requires_separate_authorization"}, indent=2))
            return 0
        print(json.dumps(run(args.config), indent=2, ensure_ascii=False))
        return 0
    except (OSError, ValueError, NeutralPostMergePilotV2Error) as error:
        print(f"Neutral post-merge V2 pilot failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
