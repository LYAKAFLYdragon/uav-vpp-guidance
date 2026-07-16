#!/usr/bin/env python3
"""Plan or execute the single frozen Taxonomy60 formal comparison universe.

The default mode is deliberately side-effect free: it only materializes the
three-child execution plan.  ``--execute`` is separately acknowledged because
it creates the one allowed 900-episode Source and must never be used as a
convenient follow-up experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import preflight_thesis_dualskill_taxonomy60_formal as preflight  # noqa: E402
import run_jsbsim_hrl_comparison as comparison  # noqa: E402


SOURCE_ID = "THESIS-DUALSKILL-TAXONOMY60-3OPP-V1"
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
METHODS = (
    "fixed_head_on",
    "fixed_crossing",
    "static_oracle_gate",
    "canonical_ppo",
    "ppo_unbootstrapped_crossing",
)
DEFAULT_CONFIG = REPO_ROOT / "config" / "experiment" / "jsbsim_hrl_thesis_dualskill_taxonomy60_formal.yaml"
DEFAULT_OUTPUT_ROOT = Path("E:/uav-vpp-guidance-thesis-evidence") / SOURCE_ID
MANIFEST_PATH = REPO_ROOT / "config" / "experiment" / "manifests" / "thesis_dualskill_taxonomy60_v1.yaml"
DECISION_CONTRACT_PATH = (
    REPO_ROOT
    / "config"
    / "experiment"
    / "decision_contracts"
    / "thesis_dualskill_taxonomy60_no_go_v3.yaml"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _require_formal_contract(config_path: Path) -> Dict[str, Any]:
    """Resolve and validate the frozen formal variables before plan creation."""

    config = comparison._load_config_with_includes(config_path)
    config = comparison._materialize_scenario_manifest(config, config_path)
    protocol = config.get("thesis_dualskill_taxonomy60_formal", {})
    if protocol.get("source_id") != SOURCE_ID:
        raise ValueError("formal config source ID does not match Taxonomy60")
    if protocol.get("evaluation_only") is not True:
        raise ValueError("formal config must remain evaluation-only")
    if protocol.get("training_permitted") is not False or protocol.get("tuning_permitted") is not False:
        raise ValueError("formal config permits training or tuning")
    if tuple(protocol.get("opponents") or ()) != OPPONENTS:
        raise ValueError("formal config opponent matrix does not match")
    if tuple(protocol.get("methods") or ()) != METHODS:
        raise ValueError("formal config method matrix does not match")
    if int(protocol.get("expected_episodes", 0)) != 900:
        raise ValueError("formal config does not declare 900 records")
    if config.get("env", {}).get("strict_backend") is not True:
        raise ValueError("formal config does not require strict JSBSim")

    task_counts = config.get("scenario_manifest_resolution", {}).get("task_counts")
    if task_counts != {"head_on": 48, "crossing_feasible": 12}:
        raise ValueError("formal scenario manifest task counts do not match")
    expected_per_opponent = len(METHODS) * sum(task_counts.values())
    if int(protocol.get("expected_episodes_per_opponent", 0)) != expected_per_opponent:
        raise ValueError("formal per-opponent episode count does not match")

    manifest = _load_yaml(MANIFEST_PATH)
    decision = _load_yaml(DECISION_CONTRACT_PATH)
    selection = config.get("scenario_manifest", {})
    if selection.get("payload_sha256") != manifest.get("integrity", {}).get("payload_sha256"):
        raise ValueError("formal config does not bind the current manifest payload")
    manifest_decision = manifest.get("decision_contract", {})
    if manifest_decision.get("id") != decision.get("contract_id"):
        raise ValueError("formal manifest does not bind the current decision contract")
    if manifest_decision.get("sha256") != _sha256(DECISION_CONTRACT_PATH):
        raise ValueError("formal manifest decision-contract hash does not match")
    return config


def build_formal_plan(
    *,
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    python_executable: str = sys.executable,
) -> List[Dict[str, Any]]:
    """Build the immutable 3 x 300 child-run plan without creating outputs."""

    config_path = config_path.resolve()
    config = _require_formal_contract(config_path)
    protocol = config["thesis_dualskill_taxonomy60_formal"]
    tasks = tuple(config["run_defaults"]["tasks"])
    jsbsim_root = str(protocol["jsbsim_legacy_root"])
    episodes = int(protocol["expected_episodes_per_opponent"])
    runner = REPO_ROOT / "scripts" / "run_jsbsim_hrl_comparison.py"
    plan: List[Dict[str, Any]] = []
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
            *tasks,
            "--seeds",
            "0",
            "--n-episodes",
            "1",
            "--backend",
            "jsbsim",
            "--jsbsim-root",
            jsbsim_root,
            "--opponent-stage",
            opponent,
            "--run-status",
            "formal",
            "--device",
            "cpu",
        ]
        plan.append(
            {
                "opponent": opponent,
                "run_id": run_id,
                "run_dir": str(output_root / run_id),
                "episodes": episodes,
                "command": command,
            }
        )
    if sum(item["episodes"] for item in plan) != int(protocol["expected_episodes"]):
        raise RuntimeError("formal execution plan does not total 900 records")
    return plan


def _git_is_clean() -> bool:
    completed = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0 and not completed.stdout.strip()


def _execution_payload(
    *,
    config_path: Path,
    output_root: Path,
    plan: List[Dict[str, Any]],
    status: str,
    executions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    manifest = _load_yaml(MANIFEST_PATH)
    decision = _load_yaml(DECISION_CONTRACT_PATH)
    protocol = comparison._load_config_with_includes(config_path)[
        "thesis_dualskill_taxonomy60_formal"
    ]
    return {
        "source_id": SOURCE_ID,
        "attempt_kind": "initial",
        "execution_status": status,
        "formal_config_path": str(config_path),
        "formal_config_sha256": _sha256(config_path),
        "manifest_path": str(MANIFEST_PATH),
        "manifest_payload_sha256": manifest["integrity"]["payload_sha256"],
        "decision_contract_path": str(DECISION_CONTRACT_PATH),
        "decision_contract_sha256": _sha256(DECISION_CONTRACT_PATH),
        "decision_contract_id": decision["contract_id"],
        "jsbsim_legacy_root": protocol["jsbsim_legacy_root"],
        "jsbsim_data_manifest_sha256": protocol["jsbsim_data_manifest_sha256"],
        "expected_records": int(protocol["expected_episodes"]),
        "expected_records_per_opponent": int(protocol["expected_episodes_per_opponent"]),
        "plan": plan,
        "executions": executions,
        "output_root": str(output_root),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_root_contract(output_root: Path, plan: List[Dict[str, Any]]) -> None:
    _write_json(
        output_root / "formal_artifact_contract.json",
        {
            "schema_version": "1.0.0",
            "required_root_files": [
                "formal_execution_manifest.json",
                "formal_execution_plan.json",
                "formal_artifact_contract.json",
                "formal_verification.json",
                "taxonomy60_route_decision.json",
            ],
            "required_child_runs": [
                {
                    "run_id": item["run_id"],
                    "opponent": item["opponent"],
                    "expected_records": item["episodes"],
                    "required_files": [
                        "run_manifest.json",
                        "artifact_contract.json",
                        "failures.json",
                        "aggregate/episode_records.json",
                        "observation_audit.json",
                    ],
                }
                for item in plan
            ],
        },
    )


def execute_formal_plan(
    *,
    config_path: Path,
    output_root: Path,
    python_executable: str,
) -> int:
    """Execute exactly one formal universe after all engineering gates pass."""

    if output_root.exists():
        raise FileExistsError("Refusing to overwrite formal output root: {}".format(output_root))
    if not _git_is_clean():
        raise RuntimeError("formal Source requires a clean Git worktree")
    receipt = preflight.run_preflight(
        formal_config_path=config_path,
        formal_output_root=output_root,
    )
    if receipt.get("verdict") != "pass":
        raise RuntimeError("P3 preflight did not pass for formal execution")

    plan = build_formal_plan(
        config_path=config_path,
        output_root=output_root,
        python_executable=python_executable,
    )
    output_root.mkdir(parents=True, exist_ok=False)
    _write_root_contract(output_root, plan)
    executions: List[Dict[str, Any]] = []
    execution_path = output_root / "formal_execution_manifest.json"
    plan_path = output_root / "formal_execution_plan.json"
    _write_json(plan_path, {"source_id": SOURCE_ID, "plan": plan})
    _write_json(
        execution_path,
        _execution_payload(
            config_path=config_path,
            output_root=output_root,
            plan=plan,
            status="running",
            executions=executions,
        ),
    )

    child_failure = False
    for item in plan:
        completed = subprocess.run(item["command"], cwd=REPO_ROOT, check=False)
        execution = {
            "opponent": item["opponent"],
            "run_id": item["run_id"],
            "returncode": int(completed.returncode),
        }
        executions.append(execution)
        child_failure = child_failure or completed.returncode != 0
        _write_json(
            execution_path,
            _execution_payload(
                config_path=config_path,
                output_root=output_root,
                plan=plan,
                status="failed" if child_failure else "running",
                executions=executions,
            ),
        )
        if child_failure:
            break

    final_status = "failed" if child_failure else "completed"
    _write_json(
        execution_path,
        _execution_payload(
            config_path=config_path,
            output_root=output_root,
            plan=plan,
            status=final_status,
            executions=executions,
        ),
    )
    verifier = REPO_ROOT / "scripts" / "verify_thesis_dualskill_taxonomy60_formal.py"
    verification = subprocess.run(
        [
            python_executable,
            str(verifier),
            "--output-root",
            str(output_root),
            "--formal-config",
            str(config_path),
        ],
        cwd=REPO_ROOT,
        check=False,
    )
    return 1 if child_failure or verification.returncode != 0 else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge-formal-source", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    output_root = args.output_root.resolve()
    plan = build_formal_plan(
        config_path=config_path,
        output_root=output_root,
        python_executable=args.python_executable,
    )
    if not args.execute:
        print(json.dumps({"source_id": SOURCE_ID, "execute": False, "plan": plan}, indent=2))
        return 0
    if not args.acknowledge_formal_source:
        parser.error("--execute requires --acknowledge-formal-source")
    return execute_formal_plan(
        config_path=config_path,
        output_root=output_root,
        python_executable=args.python_executable,
    )


if __name__ == "__main__":
    raise SystemExit(main())
