#!/usr/bin/env python3
"""Run exactly one isolated frozen-controller source episode."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
SOURCE_ID = "THESIS-MILD-DISADVANTAGE-HANDOFF-SOURCE-TRACES-ISOLATED-V1-20260904"
MATCHED_TELEMETRY = ROOT / "scripts/run_mild_disadvantage_envelope_recovery_matched_telemetry_v1.py"

from uav_vpp_guidance.common.provenance import record_config_override_if_changed


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def set_nested(config: dict[str, Any], key: str, value: Any, source: str) -> None:
    current: Any = config
    parts = key.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    old = current.get(parts[-1])
    current[parts[-1]] = value
    record_config_override_if_changed(config, key, value, old, source)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--opponent", choices=("expert", "end_to_end"), required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--execute", action="store_true", help="Required to execute JSBSim")
    args = parser.parse_args(argv)

    if not args.execute:
        print(f"LOCKED: no JSBSim execution. Planned source: {SOURCE_ID}")
        return 0

    root = args.output_dir.resolve()
    if root.exists() and any(root.rglob("*")):
        raise SystemExit(f"refusing to overwrite non-empty output directory: {root}")
    root.mkdir(parents=True, exist_ok=True)

    matched = load_module(MATCHED_TELEMETRY, "isolated_source_matched_telemetry")
    matched.SOURCE_ID = SOURCE_ID
    spec = matched._load(args.config.resolve())
    spec["experiment"]["source_id"] = SOURCE_ID
    candidates = spec["probe"]["candidates"]
    if args.candidate not in candidates:
        raise SystemExit(f"unknown candidate {args.candidate!r}")
    spec["probe"]["candidates"] = {args.candidate: candidates[args.candidate]}
    spec["probe"]["repetitions_per_candidate_and_opponent"] = 1
    offset = 50 if args.opponent == "end_to_end" else 0
    spec["probe"]["evaluation_seed_base"] = int(args.seed) - offset

    base = matched._load(matched._resolve(spec["base_config"]))
    set_nested(
        base,
        "env.legacy_project_root",
        str((ROOT.parent / "uav-vpp-guidance-artifacts/CAN-20260705/assets").resolve()),
        "run_mild_disadvantage_handoff_source_trace_worker_v1.py:jsbsim_root",
    )
    set_nested(base, "backend", "jsbsim", "run_mild_disadvantage_handoff_source_trace_worker_v1.py:backend")
    set_nested(base, "env.backend", "jsbsim", "run_mild_disadvantage_handoff_source_trace_worker_v1.py:env_backend")
    set_nested(base, "env.use_jsbsim", True, "run_mild_disadvantage_handoff_source_trace_worker_v1.py:use_jsbsim")
    set_nested(base, "env.strict_backend", True, "run_mild_disadvantage_handoff_source_trace_worker_v1.py:strict_backend")
    matched_helpers = matched._load_module(
        matched.MATCHED_RUNNER, "isolated_source_matched_probe_helpers"
    )
    legacy = matched_helpers._legacy_module()
    rows, manifest = matched._run_opponent(
        spec, base, args.opponent, root, matched_helpers, legacy
    )
    if len(rows) != 1 or int(rows[0]["seed"]) != int(args.seed):
        raise RuntimeError(f"worker seed contract failed: {rows}")

    trajectory_paths = sorted((root / "trajectories").glob("*.csv"))
    if len(trajectory_paths) != 1:
        raise RuntimeError(f"expected exactly one trajectory, found {trajectory_paths}")
    write_json(
        root / "worker_receipt.json",
        {
            "source_id": SOURCE_ID,
            "opponent": args.opponent,
            "candidate": args.candidate,
            "seed": int(args.seed),
            "episode_count": 1,
            "trajectory": str(trajectory_paths[0]),
            "run_manifest": str(root / "run_manifest.json"),
            "backend_effective": manifest.get("backend_effective"),
            "backend_fallback_occurred": manifest.get("backend_fallback_occurred"),
            "training_started": False,
        },
    )
    print(json.dumps({"source_id": SOURCE_ID, "output_dir": str(root), "seed": args.seed}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
