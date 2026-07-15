#!/usr/bin/env python3
"""One-time P2-B5 continuous sampler; execution remains separately authorised."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "src"))

import run_thesis_global_advantage_p2_b3_phase_observability as b3
from uav_vpp_guidance.evaluation.global_advantage_p2_b5_phase_feasible_sampler import (
    MANIFEST_SOURCE_ID,
    SOURCE_ID,
    PhaseFeasibleSamplerCollector,
    PhaseFeasibleSamplerError,
    build_phase_feasible_sampler_plan,
    evaluate_gate,
    raw_si_phase_input,
    sha256_file,
)
from uav_vpp_guidance.training.thesis_shared_skill_geometry import PhaseTracker


DEFAULT_CONFIG = ROOT / "config" / "experiment" / "thesis_global_advantage_v1_p2_b5_phase_feasible_sampler.yaml"
AUTHORIZATION_DELTA_PATHS = {
    "config/experiment/thesis_global_advantage_v1_p2_b5_phase_feasible_sampler.yaml",
    "reports/thesis_global_advantage_p2_b5_phase_feasible_sampler_execution_authorization_20260715_zh.md",
    "reports/thesis_five_state_global_advantage_goal_checklist_v1_20260714_zh.md",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise PhaseFeasibleSamplerError(f"expected YAML mapping: {path}")
    return payload


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _require_hash(path: Path, expected: Any, label: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64:
        raise PhaseFeasibleSamplerError(f"{label} must declare SHA-256")
    if not path.is_file() or sha256_file(path) != expected.lower():
        raise PhaseFeasibleSamplerError(f"{label} SHA-256 mismatch")


def _manifest_payload_sha256(manifest: Mapping[str, Any]) -> str:
    prepared = copy.deepcopy(dict(manifest))
    prepared.pop("integrity", None)
    return hashlib.sha256(
        json.dumps(prepared, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_authorization(config_path: Path, authorization: Mapping[str, Any]) -> None:
    if authorization.get("execution_permitted") is not True:
        raise PhaseFeasibleSamplerError("B5 execution is not authorised")
    if config_path.resolve() != DEFAULT_CONFIG.resolve():
        raise PhaseFeasibleSamplerError("B5 execution requires the canonical config path")
    required_sha = authorization.get("required_implementation_git_sha")
    if (
        not isinstance(required_sha, str)
        or len(required_sha) != 40
        or not b3._git_is_ancestor(required_sha)
        or b3._git_value("status", "--porcelain") != ""
    ):
        raise PhaseFeasibleSamplerError("B5 execution requires a clean frozen implementation SHA")
    unexpected = b3._git_changed_paths_since(required_sha) - AUTHORIZATION_DELTA_PATHS
    if unexpected:
        raise PhaseFeasibleSamplerError(f"B5 has unauthorised post-freeze changes: {sorted(unexpected)}")
    files = authorization.get("authorized_code_files")
    if not isinstance(files, Sequence) or not files:
        raise PhaseFeasibleSamplerError("B5 execution requires exact authorised code hashes")
    for entry in files:
        if not isinstance(entry, Mapping):
            raise PhaseFeasibleSamplerError("B5 authorised code entry must be a mapping")
        _require_hash(_repo_path(str(entry.get("path", ""))), entry.get("sha256"), "B5 authorised code")


def validate_sources(
    config_path: Path = DEFAULT_CONFIG, *, require_execution: bool = False
) -> tuple[dict[str, Any], Any, dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Validate all B5 inputs; only ``require_execution`` activates the run guard."""

    config = _load_yaml(config_path)
    plan = build_phase_feasible_sampler_plan(config)
    if require_execution:
        _validate_authorization(config_path, config["authorization"])
    inputs = config.get("inputs")
    if not isinstance(inputs, Mapping):
        raise PhaseFeasibleSamplerError("B5 inputs must be a mapping")
    manifest_path = _repo_path(str(inputs.get("manifest", "")))
    runtime_path = _repo_path(str(inputs.get("runtime_config", "")))
    registry_path = _repo_path(str(inputs.get("runtime_registry", "")))
    _require_hash(manifest_path, inputs.get("manifest_sha256"), "B5 manifest")
    _require_hash(runtime_path, inputs.get("runtime_config_sha256"), "B5 runtime config")
    _require_hash(registry_path, inputs.get("runtime_registry_sha256"), "B5 runtime registry")
    _require_hash(plan.p3_checkpoint, plan.p3_checkpoint_sha256, "B5 frozen P3 encoder")
    manifest = _load_yaml(manifest_path)
    scenarios = list(manifest.get("scenarios") or [])
    if (
        manifest.get("source_id") != MANIFEST_SOURCE_ID
        or manifest.get("source_id") != inputs.get("manifest_source_id")
        or len(scenarios) != 12
        or _manifest_payload_sha256(manifest) != manifest.get("integrity", {}).get("payload_sha256")
    ):
        raise PhaseFeasibleSamplerError("B5 manifest identity/integrity drifted")
    if not all(
        (scenario.get("metadata") or {}).get("taxonomy_geometry_state") == "disadvantage"
        and (scenario.get("metadata") or {}).get("phase_at_reset") == "pre_merge"
        for scenario in scenarios
    ):
        raise PhaseFeasibleSamplerError("B5 scenario is outside its preregistered motif")
    for label, entry in (manifest.get("disjointness") or {}).items():
        _require_hash(
            _repo_path(str(entry.get("path", ""))),
            entry.get("sha256"),
            f"B5 disjointness source {label}",
        )
    runtime = _load_yaml(runtime_path)
    registry = _load_yaml(registry_path)
    b3._validate_registry_assets(registry, plan.reference_specialist)
    specialist = registry["specialists"][plan.reference_specialist]
    if specialist.get("checkpoint_sha256") != plan.reference_specialist_sha256:
        raise PhaseFeasibleSamplerError("B5 frozen head-on checkpoint SHA drifted")
    return config, plan, runtime, registry, scenarios


def _phase_replay(steps: Sequence[Mapping[str, Any]]) -> bool:
    tracker = PhaseTracker()
    for record in steps:
        raw = record.get("raw_si_phase_input")
        if not isinstance(raw, Mapping):
            return False
        replayed = tracker.update(float(raw["range_m"]), float(raw["range_rate_mps"]))
        if replayed != record.get("phase"):
            return False
    return True


def _episode(
    runtime: Mapping[str, Any],
    registry: Mapping[str, Any],
    scenario: Mapping[str, Any],
    opponent: str,
    plan: Any,
    tolerances: Mapping[str, Any],
) -> dict[str, Any]:
    env = b3.pilot._build_env(runtime, registry, opponent)
    reference = b3.pilot._specialist(registry, plan.reference_specialist)
    try:
        observation = env.reset(scenario=dict(scenario), seed=int(scenario["metadata"]["scenario_seed"]))
        if getattr(env, "_backend", None) != "jsbsim":
            raise PhaseFeasibleSamplerError("B5 strict JSBSim backend was not selected")
        receipt = b3.scenario_application_receipt(scenario, env.jsbsim_env.get_state(), tolerances)
        step0_tracker = PhaseTracker()
        step0_raw = raw_si_phase_input(observation)
        step0_phase = step0_tracker.update(step0_raw["range_m"], step0_raw["range_rate_mps"])
        collector = PhaseFeasibleSamplerCollector(
            plan=plan,
            opponent=opponent,
            scenario_name=str(scenario["name"]),
            scenario_metadata=scenario["metadata"],
            seed=int(scenario["metadata"]["scenario_seed"]),
            environment_episode=int(getattr(env, "episode_count", 0)),
            observation_schema=observation["observation_schema"],
        )
        terminal_reason = "horizon"
        for step in range(1, plan.max_high_level_steps + 1):
            action = b3._finite_vector(
                reference.get_deterministic_action(observation["observation_vector"]),
                3,
                "frozen head-on action",
            )
            collector.begin_step(step=step, observation=observation, action=action)
            next_observation, _reward, terminated, truncated, info = env.step(action)
            collector.finish_step(step=step, action=action, info=info)
            observation = next_observation
            if terminated or truncated:
                terminal_reason = str(info.get("reason") or "terminated")
                break
        ledger = collector.ledger()
        summary = ledger["summary"]
        summary["scenario_application_passed"] = bool(receipt["passed"])
        summary["raw_si_phase_replay_passed"] = _phase_replay(ledger["steps"])
        summary["initial_pre_merge_semantics_passed"] = step0_phase == "pre_merge"
        return {
            "source_id": SOURCE_ID,
            "opponent": opponent,
            "scenario_name": scenario["name"],
            "scenario_metadata": scenario["metadata"],
            "terminal_reason": terminal_reason,
            "reset_receipt": receipt,
            "step0_raw_si_phase_input": step0_raw,
            "step0_phase": step0_phase,
            "ledger": ledger,
        }
    finally:
        env.close()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _episode_path(output: Path, opponent: str, scenario: Mapping[str, Any]) -> Path:
    return output / "episodes" / opponent / f"{scenario['name']}.json"


def run(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Run B5 once after a future, separately reviewed authorisation commit."""

    config, plan, runtime, registry, scenarios = validate_sources(config_path, require_execution=True)
    output = Path(str(config["outputs"]["root"]))
    if output.exists():
        raise PhaseFeasibleSamplerError("B5 requires a new empty output root")
    if shutil.disk_usage(output.parent).free / (1024**3) < plan.min_free_disk_gb:
        raise PhaseFeasibleSamplerError("B5 disk gate failed")
    output.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    try:
        tolerances = config["contract"]["scenario_application_tolerances"]
        for opponent in plan.opponents:
            for scenario in scenarios:
                record = _episode(runtime, registry, scenario, opponent, plan, tolerances)
                path = _episode_path(output, opponent, scenario)
                _write_json(path, record)
                records.append(
                    {
                        "opponent": opponent,
                        "scenario": scenario["name"],
                        "path": str(path.relative_to(output)).replace("\\", "/"),
                        "sha256": sha256_file(path),
                        "ledger": record["ledger"],
                    }
                )
        gate = evaluate_gate([record["ledger"] for record in records], plan)
        gate["scenario_application_passed"] = all(
            item["ledger"]["summary"].get("scenario_application_passed", False) for item in records
        )
        gate["raw_si_phase_replay_passed"] = all(
            item["ledger"]["summary"].get("raw_si_phase_replay_passed", False) for item in records
        )
        gate["initial_pre_merge_semantics_passed"] = all(
            item["ledger"]["summary"].get("initial_pre_merge_semantics_passed", False) for item in records
        )
        gate["training_unlocked"] = False
        _write_json(output / "phase_feasible_sampler_gate.json", gate)
        _write_json(
            output / "run_manifest.json",
            {
                "source_id": SOURCE_ID,
                "paper_safe": False,
                "mode": "noncanonical_evaluation_only",
                "provenance": {
                    "git_sha": b3._git_value("rev-parse", "HEAD"),
                    "config_sha256": sha256_file(config_path),
                    "manifest_sha256": config["inputs"]["manifest_sha256"],
                    "runtime_config_sha256": config["inputs"]["runtime_config_sha256"],
                    "runtime_registry_sha256": config["inputs"]["runtime_registry_sha256"],
                    "p3_checkpoint_sha256": plan.p3_checkpoint_sha256,
                    "reference_specialist_sha256": plan.reference_specialist_sha256,
                },
                "records": records,
                "gate": gate,
            },
        )
        return gate
    except Exception:
        _write_json(
            output / "runner_failure_manifest.json",
            {"source_id": SOURCE_ID, "completed_record_count": len(records)},
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        if not args.execute:
            config, plan, _runtime, _registry, scenarios = validate_sources(args.config)
            print(
                json.dumps(
                    {
                        "source_id": config["source_id"],
                        "mode": "validated_not_executed",
                        "execution_permitted": plan.execution_permitted,
                        "planned_records": len(scenarios) * len(plan.opponents),
                        "training_permitted": False,
                    },
                    indent=2,
                )
            )
            return 0
        gate = run(args.config)
    except (OSError, ValueError, yaml.YAMLError, PhaseFeasibleSamplerError) as error:
        print(f"INVALID B5 phase-feasible sampler: {error}", file=sys.stderr)
        return 2
    print(json.dumps(gate, indent=2, ensure_ascii=False))
    return 0 if gate["gate_passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
