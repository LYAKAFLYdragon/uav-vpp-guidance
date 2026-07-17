#!/usr/bin/env python3
"""Execute the one authorized frozen Plan 1 routing-capacity matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import verify_thesis_disadvantage_capability_plan1 as verifier


DEFAULT_CONFIG = ROOT / "config" / "experiment" / "thesis_disadvantage_capability_plan1_v1.yaml"


def _git_clean() -> bool:
    completed = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True, check=False)
    return completed.returncode == 0 and not completed.stdout.strip()


def build_plan(config_path: Path, output_root: Path, python_executable: str) -> list[dict]:
    config = verifier._load_config(config_path)
    protocol = config["disadvantage_capability_plan1"]
    source_id = str(protocol["source_id"])
    if protocol.get("execution_permitted") is not True:
        raise RuntimeError("Plan 1 execution is not authorized by config")
    jsbsim_root = str(ROOT)
    runner = ROOT / "scripts" / "run_jsbsim_hrl_comparison.py"
    return [
        {
            "opponent": opponent,
            "run_id": f"{source_id}-{opponent}",
            "command": [
                python_executable,
                str(runner),
                "--config", str(config_path),
                "--run-id", f"{source_id}-{opponent}",
                "--output-root", str(output_root),
                "--preset", "ablation",
                "--methods", *protocol["methods"],
                "--tasks", "head_on",
                "--seeds", "0",
                "--n-episodes", "1",
                "--backend", "jsbsim",
                "--jsbsim-root", jsbsim_root,
                "--opponent-stage", opponent,
                "--run-status", "formal-small",
                "--device", "cpu",
            ],
        }
        for opponent in protocol["opponents"]
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    output_root = args.output_root.resolve()
    source_id = str(verifier._load_config(config_path)["disadvantage_capability_plan1"]["source_id"])
    plan = build_plan(config_path, output_root, args.python_executable)
    if not args.execute:
        print(json.dumps({"source_id": source_id, "execute": False, "plan": plan}, ensure_ascii=False, indent=2))
        return 0
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite Plan 1 output root: {output_root}")
    if not _git_clean():
        raise RuntimeError("Plan 1 requires a clean Git worktree")
    output_root.mkdir(parents=True)
    receipt = {"source_id": source_id, "plan": plan, "executions": []}
    for item in plan:
        completed = subprocess.run(item["command"], cwd=ROOT, check=False)
        receipt["executions"].append({"opponent": item["opponent"], "returncode": completed.returncode})
        (output_root / "execution_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if completed.returncode != 0:
            return int(completed.returncode)
    result = verifier.verify(config_path, output_root)
    (output_root / "plan1_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
