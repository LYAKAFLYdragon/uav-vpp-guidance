#!/usr/bin/env python3
"""Execute once the non-learning paired task-interface runtime probe."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import preflight_thesis_task_interface_discriminability_v1 as design_preflight  # noqa: E402
import run_jsbsim_hrl_comparison as comparison  # noqa: E402
import verify_thesis_task_interface_discriminability_runtime_probe_v1 as verifier  # noqa: E402


PROBE_ID = verifier.PROBE_ID
CONFIG = ROOT / "config" / "experiment" / "jsbsim_hrl_thesis_task_interface_discriminability_runtime_probe_v1_authorized.yaml"
METHODS = verifier.METHODS
OPPONENTS = verifier.OPPONENTS
CONTEXTS = verifier.CONTEXTS


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    return comparison._load_config_with_includes(path)


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _git_clean() -> bool:
    result = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=False)
    return result.returncode == 0 and not result.stdout.strip()


def _git_sha() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _authorized_config(path: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    raw = _load_config(path)
    protocol = raw.get("task_interface_discriminability_runtime_probe", {})
    if protocol.get("probe_id") != PROBE_ID or protocol.get("execution_permitted") is not True:
        raise ValueError("runtime probe is not explicitly authorized")
    if protocol.get("training_permitted") is not False or protocol.get("tuning_permitted") is not False or protocol.get("performance_comparison_permitted") is not False:
        raise ValueError("runtime probe authorization permits disallowed work")
    if tuple(protocol.get("opponents") or ()) != OPPONENTS or tuple(protocol.get("methods") or ()) != METHODS or tuple(protocol.get("contexts") or ()) != CONTEXTS:
        raise ValueError("runtime probe matrix differs from its preregistration")
    if int(protocol.get("expected_records", 0)) != 48 or int(protocol.get("expected_records_per_opponent", 0)) != 16:
        raise ValueError("runtime probe record-count contract mismatch")
    static = design_preflight.run_preflight()
    receipt_path = ROOT / str(protocol["required_static_receipt"])
    if not receipt_path.is_file():
        raise FileNotFoundError("required static preflight receipt is missing")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if static.get("verdict") != "pass" or receipt.get("verdict") != "pass" or receipt.get("config_sha256") != static.get("config_sha256") or receipt.get("manifest_sha256") != static.get("manifest_sha256"):
        raise ValueError("static preflight receipt no longer binds the current design")
    output_root = Path(str(protocol["output_root"]))
    return comparison._materialize_scenario_manifest(raw, path), protocol, output_root


def build_plan(config_path: Path = CONFIG) -> list[dict[str, Any]]:
    config, protocol, _ = _authorized_config(config_path.resolve())
    plan = []
    for opponent in OPPONENTS:
        for context in CONTEXTS:
            scenarios = config["tasks"][context]["task"]["scenarios"]
            for method in METHODS:
                for scenario in scenarios:
                    plan.append({"opponent": opponent, "context": context, "method": method, "scenario": scenario["name"], "physical_scenario_id": scenario["metadata"]["physical_scenario_id"], "seed": int(scenario["metadata"]["scenario_seed"])})
    if len(plan) != int(protocol["expected_records"]):
        raise RuntimeError("runtime plan does not total 48 records")
    return plan


def _capture_reset(env: Any, captured: list[dict[str, Any]]):
    original = env.reset

    def wrapped_reset(*args: Any, **kwargs: Any):
        observation = original(*args, **kwargs)
        own, target = env._get_current_states()
        schema = observation.get("observation_schema", {})
        vector = observation.get("observation_vector")
        captured.append(
            {
                "feature_names": list(schema.get("feature_names", [])),
                "vector": _json_value(vector),
                "physical_reset_receipt": {
                    "own_position_m": _json_value(own.get("position_m", own.get("position_neu"))),
                    "target_position_m": _json_value(target.get("position_m", target.get("position_neu"))),
                    "own_speed_mps": _json_value(own.get("speed_mps")),
                    "target_speed_mps": _json_value(target.get("speed_mps")),
                    "own_altitude_m": _json_value(own.get("altitude_m")),
                    "target_altitude_m": _json_value(target.get("altitude_m")),
                },
            }
        )
        return observation

    env.reset = wrapped_reset
    return original


def _run_records(config_path: Path, output_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config, protocol, _ = _authorized_config(config_path)
    records: list[dict[str, Any]] = []
    config_receipts: dict[str, Any] = {}
    git_sha = _git_sha()
    for opponent in OPPONENTS:
        for context in CONTEXTS:
            for method in METHODS:
                method_def = config["methods"][method]
                eval_config = comparison.build_eval_config(config, method, context, "jsbsim", opponent)
                scenarios = eval_config["task"]["scenarios"]
                env = comparison._build_env(eval_config)
                try:
                    sample = env.reset(scenario=scenarios[0], seed=int(scenarios[0]["metadata"]["scenario_seed"]))
                    checkpoint = None
                    if comparison._uses_single_policy_checkpoint(method_def.get("agent_type", "ppo")):
                        checkpoint = comparison._repo_path(method_def.get("checkpoint"))
                    agent = comparison._build_agent(method_def, eval_config, checkpoint or Path("unused"), sample, "cpu")
                    if hasattr(agent, "set_env"):
                        agent.set_env(env)
                    if hasattr(agent, "set_task_name"):
                        agent.set_task_name(context)
                    config_receipts["{}/{}/{}".format(opponent, context, method)] = {
                        "config_sha256": comparison.config_sha256(eval_config),
                        "config_overrides": comparison.get_config_overrides(eval_config),
                    }
                    for episode, scenario in enumerate(scenarios):
                        if hasattr(agent, "reset_episode"):
                            agent.reset_episode()
                        captured: list[dict[str, Any]] = []
                        original_reset = _capture_reset(env, captured)
                        try:
                            record = comparison._run_episode(
                                env=env, agent=agent, method_name=method, method_def=method_def,
                                task_name=context, scenario=scenario, seed=int(scenario["metadata"]["scenario_seed"]),
                                episode=episode, run_id=PROBE_ID, config=eval_config, git_commit=git_sha,
                                run_status="engineering_preflight", save_full=True,
                            )
                        finally:
                            env.reset = original_reset
                        if len(captured) != 1:
                            raise RuntimeError("episode did not yield exactly one reset receipt")
                        record.update(captured[0])
                        record["probe_id"] = PROBE_ID
                        record["probe_identity"] = {
                            "opponent": opponent,
                            "context": context,
                            "method": method,
                            "physical_scenario_id": scenario["metadata"]["physical_scenario_id"],
                            "scenario_alias": scenario["name"],
                            "seed": int(scenario["metadata"]["scenario_seed"]),
                        }
                        records.append(_json_value(record))
                finally:
                    env.close()
    if len(records) != int(protocol["expected_records"]):
        raise RuntimeError("runtime probe did not produce all 48 records")
    return records, config_receipts


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8")


def execute(config_path: Path, output_root: Path) -> int:
    if output_root.exists():
        raise FileExistsError("refusing to overwrite the one-time runtime probe root")
    if not _git_clean():
        raise RuntimeError("runtime probe requires a clean worktree")
    plan = build_plan(config_path)
    output_root.mkdir(parents=True)
    _write_json(output_root / "probe_plan.json", {"probe_id": PROBE_ID, "plan": plan})
    try:
        records, receipts = _run_records(config_path, output_root)
    except Exception as error:
        _write_json(output_root / "execution_manifest.json", {"probe_id": PROBE_ID, "status": "failed", "records": 0, "error": str(error)})
        raise
    _write_json(output_root / "records.json", {"probe_id": PROBE_ID, "records": records})
    _write_json(output_root / "config_receipts.json", receipts)
    _write_json(output_root / "execution_manifest.json", {"probe_id": PROBE_ID, "status": "completed", "records": len(records), "config_sha256": _hash(config_path), "git_sha": _git_sha()})
    result = verifier.verify(output_root, config_path)
    _write_json(output_root / "verification.json", result)
    return 0 if result["verdict"] == "pass" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge-probe")
    args = parser.parse_args()
    config_path = args.config.resolve()
    _, protocol, default_output = _authorized_config(config_path)
    output_root = args.output_root.resolve() if args.output_root else default_output
    if not args.execute:
        print(json.dumps({"probe_id": PROBE_ID, "execution_authorized": False, "plan": build_plan(config_path)}, ensure_ascii=False, indent=2))
        return 0
    if args.acknowledge_probe != PROBE_ID:
        parser.error("--execute requires --acknowledge-probe {}".format(PROBE_ID))
    try:
        return execute(config_path, output_root)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
