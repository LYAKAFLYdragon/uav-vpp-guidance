#!/usr/bin/env python3
"""Run exactly one disposable 15-episode strict-JSBSim Taxonomy60 smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_ID = "THESIS-DUALSKILL-TAXONOMY60-3OPP-V1-P2-SMOKE"
FORMAL_SOURCE_ID = "THESIS-DUALSKILL-TAXONOMY60-3OPP-V1"
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
METHODS = (
    "fixed_head_on",
    "fixed_crossing",
    "static_oracle_gate",
    "canonical_ppo",
    "ppo_unbootstrapped_crossing",
)
DEFAULT_CONFIG = REPO_ROOT / "config" / "experiment" / "jsbsim_hrl_thesis_dualskill_taxonomy60_p2_smoke.yaml"
DEFAULT_OUTPUT_ROOT = Path("E:/uav-vpp-guidance-thesis-evidence") / SOURCE_ID


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_smoke_contract(config_path: Path) -> Dict[str, Any]:
    config = _load_yaml(config_path)
    contract = dict(config.get("thesis_dualskill_taxonomy60_p2_smoke") or {})
    if contract.get("source_id") != SOURCE_ID:
        raise ValueError("P2 smoke config source ID does not match")
    if contract.get("formal_source_id") != FORMAL_SOURCE_ID:
        raise ValueError("P2 smoke config formal Source ID does not match")
    if tuple(contract.get("required_opponents") or ()) != OPPONENTS:
        raise ValueError("P2 smoke config opponent matrix does not match")
    if tuple(contract.get("required_methods") or ()) != METHODS:
        raise ValueError("P2 smoke config method matrix does not match")
    if int(contract.get("expected_episodes", 0)) != len(OPPONENTS) * len(METHODS):
        raise ValueError("P2 smoke config episode count does not match")
    if contract.get("paper_evidence") is not False:
        raise ValueError("P2 smoke must be explicitly excluded from paper evidence")
    return contract


def build_smoke_plan(
    *,
    config_path: Path,
    output_root: Path,
    python_executable: str,
) -> List[Dict[str, Any]]:
    """Build, but do not execute, one five-method child run for each opponent."""

    _require_smoke_contract(config_path)
    runner = REPO_ROOT / "scripts" / "run_jsbsim_hrl_comparison.py"
    plan = []
    for opponent in OPPONENTS:
        run_id = "{}-{}".format(SOURCE_ID, opponent)
        command = [
            python_executable,
            str(runner),
            "--config",
            str(config_path),
            "--run-id",
            run_id,
            "--output-root",
            str(output_root),
            "--preset",
            "ablation",
            "--methods",
            *METHODS,
            "--tasks",
            "crossing_feasible",
            "--seeds",
            "0",
            "--n-episodes",
            "1",
            "--backend",
            "jsbsim",
            "--opponent-stage",
            opponent,
            "--run-status",
            "smoke",
            "--device",
            "cpu",
        ]
        plan.append(
            {
                "opponent": opponent,
                "run_id": run_id,
                "run_dir": str(output_root / run_id),
                "episodes": len(METHODS),
                "command": command,
            }
        )
    return plan


def _write_execution_manifest(
    output_root: Path,
    *,
    config_path: Path,
    plan: List[Dict[str, Any]],
    executions: List[Dict[str, Any]],
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_id": SOURCE_ID,
        "formal_source_id": FORMAL_SOURCE_ID,
        "paper_evidence": False,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "expected_records": len(OPPONENTS) * len(METHODS),
        "plan": plan,
        "executions": executions,
    }
    path = output_root / "smoke_execution_manifest.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    output_root = args.output_root.resolve()
    plan = build_smoke_plan(
        config_path=config_path,
        output_root=output_root,
        python_executable=args.python_executable,
    )
    if not args.execute:
        _write_execution_manifest(
            output_root,
            config_path=config_path,
            plan=plan,
            executions=[],
        )
        print(json.dumps({"source_id": SOURCE_ID, "plan": plan, "execute": False}, indent=2))
        return 0

    for item in plan:
        run_dir = Path(item["run_dir"])
        if run_dir.exists():
            raise FileExistsError("Refusing to overwrite disposable smoke run: {}".format(run_dir))

    executions = []
    for item in plan:
        completed = subprocess.run(item["command"], cwd=REPO_ROOT, check=False)
        executions.append({"opponent": item["opponent"], "returncode": completed.returncode})
        _write_execution_manifest(
            output_root,
            config_path=config_path,
            plan=plan,
            executions=executions,
        )
        if completed.returncode != 0:
            return completed.returncode

    verifier = REPO_ROOT / "scripts" / "verify_thesis_dualskill_taxonomy60_p2_smoke.py"
    completed = subprocess.run(
        [args.python_executable, str(verifier), "--output-root", str(output_root)],
        cwd=REPO_ROOT,
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
