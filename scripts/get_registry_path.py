#!/usr/bin/env python3
"""Query checkpoint registry for training output directories or checkpoint paths.

Usage:
  # Get training output directory
  python scripts/get_registry_path.py --key p0a_vpp --field output_dir

  # Get checkpoint path (seed 0)
  python scripts/get_registry_path.py --key p0a_vpp --field checkpoint --seed 0

  # Get evaluation checkpoint for a stage/method
  python scripts/get_registry_path.py --stage p0a --method no_prediction --field checkpoint
"""

import argparse
import sys
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description="Query checkpoint registry")
    parser.add_argument("--registry", type=Path, default=Path("config/checkpoint_registry.yaml"))
    parser.add_argument("--key", type=str, default=None, help="Training registry key")
    parser.add_argument("--stage", type=str, default=None, help="Evaluation stage name")
    parser.add_argument("--method", type=str, default=None, help="Method name (with --stage)")
    parser.add_argument("--field", type=str, required=True, choices=["output_dir", "checkpoint", "gains"])
    parser.add_argument("--seed", type=int, default=0, help="Seed for template substitution")
    args = parser.parse_args()

    if not args.registry.exists():
        print(f"Registry not found: {args.registry}", file=sys.stderr)
        return 1

    registry = yaml.safe_load(args.registry.read_text(encoding="utf-8"))

    if args.key:
        training = registry.get("training", {})
        entry = training.get(args.key)
        if not entry:
            print(f"Key '{args.key}' not found in training entries", file=sys.stderr)
            return 1
        value = entry.get(args.field)
        if not value:
            print(f"Field '{args.field}' not found for key '{args.key}'", file=sys.stderr)
            return 1
        # Substitute seed template
        if "{seed}" in value:
            value = value.format(seed=args.seed)
        print(value)
        return 0

    if args.stage:
        if not args.method:
            print("--method is required with --stage", file=sys.stderr)
            return 1
        eval_methods = registry.get("evaluation_methods", {})
        stage_methods = eval_methods.get(args.stage)
        if not stage_methods:
            print(f"Stage '{args.stage}' not found", file=sys.stderr)
            return 1
        method_cfg = stage_methods.get(args.method)
        if not method_cfg:
            print(f"Method '{args.method}' not found in stage '{args.stage}'", file=sys.stderr)
            return 1
        value = method_cfg.get(args.field)
        if not value:
            # Try fallback to checkpoint field name
            if args.field == "checkpoint":
                print(f"Checkpoint not found for {args.stage}/{args.method}", file=sys.stderr)
                return 1
            # gains field is optional
            return 0
        print(value)
        return 0

    print("Either --key or --stage/--method must be provided", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
