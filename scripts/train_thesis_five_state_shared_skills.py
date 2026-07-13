#!/usr/bin/env python3
"""P4 shared-skill entry point with an explicit geometry-pretrain interlock.

The real geometry-pretraining loop is implemented, but the committed config
remains ``training_permitted: false``. This command therefore supports
readiness and planning now, and rejects any rollout until a later explicit
authorization change. Combat finetuning is intentionally not callable here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (Path(__file__).resolve().parent, REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from verify_thesis_five_state_shared_skills import (
    DEFAULT_COMBAT_CONFIG,
    DEFAULT_GEOMETRY_CONFIG,
    DEFAULT_REGISTRY,
    DEFAULT_REPORT,
    build_readiness_report,
)
from uav_vpp_guidance.training.thesis_shared_skill_geometry import (
    build_geometry_training_plan,
    run_geometry_pretrain,
)


def _load_yaml(path: Path):
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return payload


def _merge(base, overlay):
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_geometry_config(path: Path):
    """Resolve the one-time authorized overlay without mutating its base config."""

    config = _load_yaml(path)
    base_value = config.pop("base_config", None)
    if not base_value:
        return config
    base_path = Path(base_value)
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    base = _load_yaml(base_path)
    expected = str(config.get("authorization", {}).get("base_config_sha256", "")).lower()
    import hashlib

    actual = hashlib.sha256(base_path.read_bytes()).hexdigest()
    if not expected or actual != expected:
        raise ValueError("Authorized geometry config base SHA mismatch")
    return _merge(base, config)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--geometry-config", type=Path, default=DEFAULT_GEOMETRY_CONFIG)
    parser.add_argument("--combat-config", type=Path, default=DEFAULT_COMBAT_CONFIG)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--skip-jsbsim-probe", action="store_true")
    parser.add_argument("--geometry-plan", action="store_true", help="Render the geometry-pretrain plan without JSBSim rollouts or output writes.")
    parser.add_argument("--run-geometry-pretrain", action="store_true", help="Run P4 geometry pretraining only after a future config authorization.")
    parser.add_argument("--attempt-training", action="store_true", help="Deprecated alias for --run-geometry-pretrain; remains interlocked by config.")
    args = parser.parse_args()
    geometry_config = _load_geometry_config(args.geometry_config)
    registry = _load_yaml(args.registry)
    if args.geometry_plan:
        print(json.dumps(build_geometry_training_plan(geometry_config, registry), indent=2, sort_keys=True))
        return 0
    if args.run_geometry_pretrain or args.attempt_training:
        result = run_geometry_pretrain(
            geometry_config,
            registry,
            output_root=Path(geometry_config["outputs"]["root"]),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    report = build_readiness_report(
        args.registry,
        args.geometry_config,
        args.combat_config,
        run_jsbsim_probe=not args.skip_jsbsim_probe,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["static_readiness_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
