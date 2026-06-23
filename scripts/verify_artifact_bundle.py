#!/usr/bin/env python3
"""Verify a portable artifact bundle produced by the Stage 6G/6H pipeline.

Example:
    python scripts/verify_artifact_bundle.py \
        --bundle outputs/pipeline/stage6g6h/bundles/stage6g
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uav_vpp_guidance.pipeline.artifact_bundle import ArtifactBundle


def main():
    parser = argparse.ArgumentParser(description="Verify a portable artifact bundle")
    parser.add_argument("--bundle", type=str, required=True, help="Path to bundle directory")
    parser.add_argument("--output", type=str, default=None, help="Path to write verification report JSON")
    args = parser.parse_args()

    bundle = ArtifactBundle(source_dir=args.bundle, bundle_dir=args.bundle)
    result = bundle.verify()

    report = {
        "bundle_dir": str(Path(args.bundle).resolve()),
        "valid": result["valid"],
        "missing": result["missing"],
        "mismatches": result["mismatches"],
        "contract_valid": result["contract_valid"],
        "contract_missing": result["contract_missing"],
    }

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    sys.exit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
