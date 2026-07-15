"""Strict, non-learning JSBSim reachability preflight for P2-B2."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from uav_vpp_guidance.evaluation.global_advantage_runin_contract import base_observation
from uav_vpp_guidance.training import thesis_defext_rangeext_pilot as pilot
from uav_vpp_guidance.training.thesis_shared_skill_geometry import PhaseTracker


SOURCE_ID = "THESIS-GLOBAL-ADVANTAGE-V1-P2-PHYSICAL-PREFLIGHT-V1"
DEFAULT_CONFIG = ROOT / "config" / "experiment" / "thesis_global_advantage_v1_p2_physical_preflight.yaml"
_PREFLIGHT_MANIFEST_SOURCE_ID = "THESIS-GLOBAL-ADVANTAGE-V1-P2-PHYSICAL-PREFLIGHT60-V1"
_OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
_PHASES = ("pre_merge", "post_merge", "re_entry")
_AUTHORIZATION_DELTA_PATHS = {
    "config/experiment/thesis_global_advantage_v1_p2_physical_preflight.yaml",
    "reports/thesis_global_advantage_p2_physical_preflight_execution_authorization_20260715_zh.md",
    "reports/thesis_five_state_global_advantage_goal_checklist_v1_20260714_zh.md",
}


class PreflightError(ValueError):
    pass


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise PreflightError(f"Expected YAML mapping: {path}")
    return payload


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _repo_path(value: str) -> Path:
    path = _path(value).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise PreflightError(f"repository path escapes worktree: {path}") from error
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _git_is_ancestor(commit: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )


def _git_changed_paths_since(commit: str) -> set[str]:
    output = subprocess.check_output(
        ["git", "diff", "--name-only", f"{commit}..HEAD"],
        cwd=ROOT,
        text=True,
        stderr=subprocess.DEVNULL,
    )
    return {line.replace("\\", "/") for line in output.splitlines() if line}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _finite_vector(value: Any, name: str) -> list[float]:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise PreflightError(f"{name} must be finite 3-D")
    return [float(item) for item in vector]


def _finite_state(state: Mapping[str, Any], name: str) -> dict[str, float]:
    result = {key: float(state[key]) for key in ("speed_mps", "altitude_m", "nz_g")}
    if not all(math.isfinite(value) for value in result.values()):
        raise PreflightError(f"{name} contains non-finite values")
    return result


def build_preflight_scenarios(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    if manifest.get("source_id") != _PREFLIGHT_MANIFEST_SOURCE_ID:
        raise PreflightError("unexpected preflight manifest source")
    scenarios = list(manifest.get("scenarios") or [])
    if len(scenarios) != 60 or len({item["metadata"]["geometry_cell_id"] for item in scenarios}) != 60:
        raise PreflightError("preflight manifest must contain 60 unique geometry cells")
    return scenarios


def _require_sha256(path: Path, expected: Any, label: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64:
        raise PreflightError(f"{label} must declare a SHA-256")
    if not path.is_file() or _sha256(path) != expected.lower():
        raise PreflightError(f"{label} SHA mismatch")


def _validate_authorization(config_path: Path, authorization: Mapping[str, Any]) -> None:
    permitted = authorization.get("execution_permitted")
    if not isinstance(permitted, bool):
        raise PreflightError("execution_permitted must be explicit")
    if not permitted:
        return
    if config_path.resolve() != DEFAULT_CONFIG.resolve():
        raise PreflightError("execution requires the canonical in-repository config")
    required_sha = authorization.get("required_implementation_git_sha")
    if (
        not isinstance(required_sha, str)
        or len(required_sha) != 40
        or not _git_is_ancestor(required_sha)
        or _git_value("status", "--porcelain") != ""
    ):
        raise PreflightError(
            "execution requires a clean worktree and frozen implementation ancestor"
        )
    unexpected_changes = _git_changed_paths_since(required_sha) - _AUTHORIZATION_DELTA_PATHS
    if unexpected_changes:
        raise PreflightError(
            "execution contains non-authorization changes after the implementation freeze: "
            f"{sorted(unexpected_changes)}"
        )
    authorized_files = authorization.get("authorized_code_files")
    if not isinstance(authorized_files, Sequence) or not authorized_files:
        raise PreflightError("execution requires explicit authorized code-file hashes")
    for entry in authorized_files:
        if not isinstance(entry, Mapping):
            raise PreflightError("authorized code-file entry must be a mapping")
        path = _repo_path(str(entry.get("path", "")))
        _require_sha256(path, entry.get("sha256"), f"authorized code file {path}")


def _validate_registry_assets(registry: Mapping[str, Any], reference_specialist: str) -> None:
    opponents = registry.get("opponents")
    specialists = registry.get("specialists")
    if not isinstance(opponents, Mapping) or not isinstance(specialists, Mapping):
        raise PreflightError("runtime registry lacks opponents or specialists")
    for opponent in _OPPONENTS:
        entry = opponents.get(opponent)
        if not isinstance(entry, Mapping):
            raise PreflightError(f"runtime registry lacks opponent: {opponent}")
        checkpoint = entry.get("checkpoint")
        if opponent == "expert":
            if checkpoint is not None:
                raise PreflightError("expert opponent must not load a checkpoint")
            continue
        if not isinstance(checkpoint, str):
            raise PreflightError(f"opponent checkpoint is missing: {opponent}")
        _require_sha256(
            Path(checkpoint), entry.get("checkpoint_sha256"), f"opponent checkpoint {opponent}"
        )
    specialist = specialists.get(reference_specialist)
    if not isinstance(specialist, Mapping):
        raise PreflightError(f"runtime registry lacks reference specialist: {reference_specialist}")
    checkpoint = specialist.get("checkpoint")
    if not isinstance(checkpoint, str):
        raise PreflightError("reference specialist checkpoint is missing")
    _require_sha256(
        Path(checkpoint), specialist.get("checkpoint_sha256"), "reference specialist checkpoint"
    )


def _validate_sources(
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    config = _load_yaml(config_path)
    if config.get("source_id") != SOURCE_ID:
        raise PreflightError("unexpected P2 physical preflight source")
    authorization = config.get("authorization")
    if not isinstance(authorization, Mapping):
        raise PreflightError("authorization must be a mapping")
    _validate_authorization(config_path, authorization)
    if any(
        authorization.get(key) is not False
        for key in (
            "training_permitted",
            "tuning_permitted",
            "candidate_policy_loading_permitted",
            "heldout_claims_permitted",
        )
    ):
        raise PreflightError("P2 physical preflight authorization drifted")
    contract = config.get("contract")
    if not isinstance(contract, Mapping):
        raise PreflightError("contract must be a mapping")
    if (
        contract.get("backend") != "jsbsim"
        or contract.get("strict_backend") is not True
        or contract.get("fresh_environment_per_episode") is not True
    ):
        raise PreflightError("P2 physical contract drifted")
    if tuple(contract.get("opponents", ())) != _OPPONENTS:
        raise PreflightError("opponent contract drifted")
    if not isinstance(contract.get("min_free_disk_gb"), (int, float)):
        raise PreflightError("min_free_disk_gb must be declared")
    inputs = config.get("inputs")
    if not isinstance(inputs, Mapping):
        raise PreflightError("inputs must be a mapping")
    manifest_path = _repo_path(str(inputs.get("manifest", "")))
    runtime_path = _repo_path(str(inputs.get("runtime_config", "")))
    registry_path = _repo_path(str(inputs.get("runtime_registry", "")))
    _require_sha256(manifest_path, inputs.get("manifest_sha256"), "preflight manifest")
    _require_sha256(runtime_path, inputs.get("runtime_config_sha256"), "runtime config")
    _require_sha256(registry_path, inputs.get("runtime_registry_sha256"), "runtime registry")
    manifest = _load_yaml(manifest_path)
    if manifest.get("source_id") != inputs.get("manifest_source_id"):
        raise PreflightError("preflight manifest source ID drifted")
    scenarios = build_preflight_scenarios(manifest)
    runtime = _load_yaml(runtime_path)
    registry = _load_yaml(registry_path)
    _validate_registry_assets(registry, str(inputs.get("reference_specialist", "")))
    return config, manifest, runtime, registry, scenarios


def _provenance(config_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    inputs = config["inputs"]
    authorization = config["authorization"]
    return {
        "git_sha": _git_value("rev-parse", "HEAD"),
        "git_dirty": _git_value("status", "--porcelain") != "",
        "config_sha256": _sha256(config_path),
        "manifest_sha256": _sha256(_repo_path(str(inputs["manifest"]))),
        "runtime_config_sha256": _sha256(_repo_path(str(inputs["runtime_config"]))),
        "runtime_registry_sha256": _sha256(_repo_path(str(inputs["runtime_registry"]))),
        "required_implementation_git_sha": authorization.get("required_implementation_git_sha"),
        "authorized_code_files": list(authorization.get("authorized_code_files") or []),
    }


def _episode(
    runtime: Mapping[str, Any], registry: Mapping[str, Any], scenario: Mapping[str, Any], opponent: str, maximum_steps: int
) -> dict[str, Any]:
    env = pilot._build_env(runtime, registry, opponent)
    reference = pilot._specialist(registry, "run_in_head_on")
    try:
        observation = env.reset(scenario=dict(scenario), seed=int(scenario["metadata"]["scenario_seed"]))
        if getattr(env, "_backend", None) != "jsbsim":
            raise PreflightError("strict JSBSim backend was not selected")
        tracker = PhaseTracker()
        steps: list[dict[str, Any]] = []
        terminal_reason = "horizon"
        for index in range(1, maximum_steps + 1):
            action = _finite_vector(reference.get_deterministic_action(observation["observation_vector"]), "reference action")
            next_observation, _reward, terminated, truncated, info = env.step(action)
            own = _finite_state(info.get("own_state", {}), "own state")
            target = _finite_state(info.get("target_state", {}), "target state")
            base = base_observation(next_observation)
            phase = tracker.update(base["range_m"], base["range_rate_mps"])
            fallback = bool(info.get("prediction_fallback", False)) and info.get("prediction_fallback_phase") != "warmup"
            record = {
                "step": index,
                "phase": phase,
                "first_pass_complete": bool(info.get("first_pass_complete", False)),
                "backend": info.get("backend"),
                "backend_fallback": bool(info.get("backend_fallback_occurred", False)),
                "prediction_fallback": fallback,
                "action": action,
                "own": own,
                "target": target,
                "terminal_reason": info.get("reason"),
            }
            steps.append(record)
            observation = next_observation
            if terminated or truncated:
                terminal_reason = str(info.get("reason") or "terminated")
                break
        return {
            "source_id": SOURCE_ID,
            "opponent": opponent,
            "scenario_name": scenario["name"],
            "geometry_cell_id": scenario["metadata"]["geometry_cell_id"],
            "step_count": len(steps),
            "terminal_reason": terminal_reason,
            "first_pass_observed": any(item["first_pass_complete"] for item in steps),
            "phase_counts": dict(Counter(item["phase"] for item in steps)),
            "valid": bool(steps)
            and all(item["backend"] == "jsbsim" and not item["backend_fallback"] and not item["prediction_fallback"] for item in steps),
            "steps": steps,
        }
    finally:
        env.close()


def run(config_path: Path) -> dict[str, Any]:
    config, manifest, runtime, registry, scenarios = _validate_sources(config_path)
    if config["authorization"].get("execution_permitted") is not True:
        raise PreflightError("P2 physical preflight is not authorised")
    contract = config["contract"]
    opponents = _OPPONENTS
    output = Path(config["outputs"]["root"])
    if output.exists():
        raise PreflightError("output root must be fresh")
    free_gb = shutil.disk_usage(output.parent).free / (1024**3)
    if free_gb < float(contract["min_free_disk_gb"]):
        raise PreflightError(
            f"disk gate failed: {free_gb:.1f} GB < {float(contract['min_free_disk_gb']):.1f} GB"
        )
    output.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    try:
        for opponent in opponents:
            for scenario in scenarios:
                record = _episode(runtime, registry, scenario, opponent, int(contract["max_high_level_steps"]))
                _write_json(output / "episodes" / opponent / f"{scenario['name']}.json", record)
                records.append(record)
        minimum = int(contract["min_valid_steps"])
        expected = len(scenarios) * len(opponents)
        phase_matrix = {
            opponent: {
                phase: sum(record["phase_counts"].get(phase, 0) for record in records if record["opponent"] == opponent)
                for phase in _PHASES
            }
            for opponent in opponents
        }
        physical_reachability_passed = len(records) == expected and all(
            record["valid"] and record["step_count"] >= minimum for record in records
        )
        phase_gaps = {
            opponent: [phase for phase, count in values.items() if count == 0]
            for opponent, values in phase_matrix.items()
        }
        gate = {
            "source_id": SOURCE_ID,
            "record_count": len(records),
            "expected_record_count": expected,
            "physical_reachability_passed": physical_reachability_passed,
            "phase_matrix": phase_matrix,
            "phase_gaps": phase_gaps,
            "phase_coverage_complete": not any(phase_gaps.values()),
            "training_unlocked": False,
        }
        _write_json(output / "physical_preflight_gate.json", gate)
        _write_json(
            output / "run_manifest.json",
            {
                "source_id": SOURCE_ID,
                "records": records,
                "provenance": _provenance(config_path, config),
                "gate": gate,
            },
        )
        return gate
    except Exception:
        _write_json(output / "runner_failure_manifest.json", {"source_id": SOURCE_ID, "completed_record_count": len(records)})
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        config, _manifest, _runtime, _registry, scenarios = _validate_sources(args.config)
        print(
            json.dumps(
                {
                    "source_id": config.get("source_id"),
                    "mode": "validated_not_executed",
                    "execution_permitted": config["authorization"]["execution_permitted"],
                    "planned_records": len(scenarios) * len(_OPPONENTS),
                },
                indent=2,
            )
        )
        return 0
    gate = run(args.config)
    print(json.dumps(gate, indent=2))
    return 0 if gate["physical_reachability_passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
