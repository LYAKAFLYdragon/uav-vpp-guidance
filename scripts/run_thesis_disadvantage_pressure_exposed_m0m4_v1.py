#!/usr/bin/env python3
"""Plan or execute the one-time pressure-exposed M0--M4 comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import preflight_thesis_disadvantage_pressure_exposed_m0m4_v1 as preflight  # noqa: E402
import verify_thesis_disadvantage_pressure_exposed_m0m4_v1 as verifier  # noqa: E402


SOURCE_ID = preflight.SOURCE_ID
OPPONENTS = preflight.OPPONENTS
METHODS = preflight.METHODS
DEFAULT_CONFIG = preflight.DEFAULT_CONFIG
DEFAULT_OUTPUT = preflight.DEFAULT_OUTPUT
MANIFEST = preflight.MANIFEST
DECISION = preflight.DECISION


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _clean_worktree() -> bool:
    result = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=False)
    return result.returncode == 0 and not result.stdout.strip()


def build_plan(*, config_path: Path = DEFAULT_CONFIG, output_root: Path = DEFAULT_OUTPUT, python_executable: str = sys.executable) -> list[dict[str, Any]]:
    protocol = _load_yaml(config_path)["pressure_exposed_m0m4_compact_ablation"]
    plan = []
    for opponent in OPPONENTS:
        plan.append(
            {
                "opponent": opponent,
                "run_id": "{}-{}".format(SOURCE_ID, opponent),
                "expected_records": int(protocol["expected_episodes_per_opponent"]),
                "command": [
                    python_executable, str(ROOT / "scripts" / "run_jsbsim_hrl_comparison.py"),
                    "--config", str(config_path.resolve()), "--run-id", "{}-{}".format(SOURCE_ID, opponent),
                    "--output-root", str(output_root), "--preset", "ablation", "--methods", *METHODS,
                    "--tasks", "head_on", "--seeds", "0", "--n-episodes", "1", "--backend", "jsbsim",
                    "--jsbsim-root", str(protocol["jsbsim_legacy_root"]), "--opponent-stage", opponent,
                    "--run-status", "formal", "--device", "cpu",
                ],
            }
        )
    if sum(item["expected_records"] for item in plan) != int(protocol["expected_episodes"]):
        raise RuntimeError("execution plan does not total the frozen 240-record universe")
    return plan


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _root_contract(plan: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "source_id": SOURCE_ID,
        "required_root_files": ["formal_execution_plan.json", "formal_execution_manifest.json", "formal_artifact_contract.json", "preflight_receipt.json", "formal_verification.json"],
        "required_child_runs": [{"run_id": item["run_id"], "opponent": item["opponent"], "expected_records": item["expected_records"], "required_files": ["run_manifest.json", "artifact_contract.json", "failures.json", "aggregate/episode_records.json", "observation_audit.json"]} for item in plan],
    }


def execute(*, config_path: Path, output_root: Path, python_executable: str) -> int:
    if output_root.exists():
        raise FileExistsError("refusing to overwrite the one-time output root: {}".format(output_root))
    if not _clean_worktree():
        raise RuntimeError("the one-time source requires a clean Git worktree")
    receipt = preflight.run_preflight(config_path=config_path, output_root=output_root)
    if receipt["verdict"] != "pass":
        raise RuntimeError("preflight did not authorize the compact ablation")
    plan = build_plan(config_path=config_path, output_root=output_root, python_executable=python_executable)
    output_root.mkdir(parents=True, exist_ok=False)
    _write_json(output_root / "preflight_receipt.json", receipt)
    _write_json(output_root / "formal_execution_plan.json", {"source_id": SOURCE_ID, "plan": plan})
    _write_json(output_root / "formal_artifact_contract.json", _root_contract(plan))
    executions = []
    for item in plan:
        completed = subprocess.run(item["command"], cwd=ROOT, check=False)
        executions.append({"opponent": item["opponent"], "run_id": item["run_id"], "returncode": completed.returncode})
        if completed.returncode != 0:
            break
    status = "completed" if len(executions) == len(plan) and all(item["returncode"] == 0 for item in executions) else "failed"
    execution = {
        "source_id": SOURCE_ID,
        "attempt_kind": "initial",
        "execution_status": status,
        "config_path": str(config_path),
        "config_sha256": _hash(config_path),
        "manifest_path": str(MANIFEST),
        "manifest_sha256": _hash(MANIFEST),
        "decision_contract_path": str(DECISION),
        "decision_contract_sha256": _hash(DECISION),
        "expected_records": 240,
        "plan": plan,
        "executions": executions,
    }
    _write_json(output_root / "formal_execution_manifest.json", execution)
    if status != "completed":
        return 1
    result = verifier.verify(output_root, config_path=config_path)
    _write_json(output_root / "formal_verification.json", result)
    return 0 if result["verdict"] == "pass" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge-source")
    args = parser.parse_args()
    config_path, output_root = args.config.resolve(), args.output_root.resolve()
    if not args.execute:
        print(json.dumps({"source_id": SOURCE_ID, "plan": build_plan(config_path=config_path, output_root=output_root)}, ensure_ascii=False, indent=2))
        return 0
    if args.acknowledge_source != SOURCE_ID:
        parser.error("--execute requires --acknowledge-source {}".format(SOURCE_ID))
    try:
        return execute(config_path=config_path, output_root=output_root, python_executable=sys.executable)
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
