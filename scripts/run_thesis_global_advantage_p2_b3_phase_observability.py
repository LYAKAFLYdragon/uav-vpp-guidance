"""One-shot, non-learning P2-B3 JSBSim phase-observability preflight."""

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


SOURCE_ID = "THESIS-GLOBAL-ADVANTAGE-V1-P2-PHASE-OBSERVABILITY-PREFLIGHT-V1"
MANIFEST_SOURCE_ID = "THESIS-GLOBAL-ADVANTAGE-V1-P2-PHASE-OBSERVABILITY-MANIFEST30-V1"
DEFAULT_CONFIG = ROOT / "config" / "experiment" / "thesis_global_advantage_v1_p2_b3_phase_observability.yaml"
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
PHASES = ("pre_merge", "post_merge", "re_entry")
AUTHORIZATION_DELTA_PATHS = {
    "config/experiment/thesis_global_advantage_v1_p2_b3_phase_observability.yaml",
    "reports/thesis_global_advantage_p2_b3_phase_observability_execution_authorization_20260715_zh.md",
    "reports/thesis_five_state_global_advantage_goal_checklist_v1_20260714_zh.md",
}


class PhaseObservabilityError(ValueError):
    """Raised if P2-B3 cannot establish its read-only measurement contract."""


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise PhaseObservabilityError(f"expected YAML mapping: {path}")
    return payload


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _repo_path(value: str) -> Path:
    path = _path(value).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise PhaseObservabilityError(f"repository path escapes worktree: {path}") from error
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


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise PhaseObservabilityError(f"{label} must be numeric") from error
    if not math.isfinite(result):
        raise PhaseObservabilityError(f"{label} must be finite")
    return result


def _finite_vector(value: Any, size: int, label: str) -> list[float]:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape != (size,) or not np.isfinite(vector).all():
        raise PhaseObservabilityError(f"{label} must be finite {size}-D")
    return [float(item) for item in vector]


def _state_kinematics(state: Mapping[str, Any], label: str) -> dict[str, Any]:
    position = state.get("position_neu", state.get("position_m", state.get("position")))
    velocity = state.get("velocity_vector_mps", state.get("velocity_neu"))
    position_vector = _finite_vector(position, 3, f"{label}.position")
    velocity_vector = _finite_vector(velocity, 3, f"{label}.velocity")
    speed = _finite_float(state.get("speed_mps"), f"{label}.speed_mps")
    altitude = _finite_float(state.get("altitude_m"), f"{label}.altitude_m")
    horizontal_speed = math.hypot(velocity_vector[0], velocity_vector[1])
    if horizontal_speed <= 1e-6:
        raise PhaseObservabilityError(f"{label}.velocity has no heading")
    return {
        "position_neu_m": position_vector,
        "velocity_neu_mps": velocity_vector,
        "speed_mps": speed,
        "altitude_m": altitude,
        "heading_deg": math.degrees(math.atan2(velocity_vector[1], velocity_vector[0])) % 360.0,
    }


def _requested_kinematics(initial: Mapping[str, Any], label: str) -> dict[str, Any]:
    position = _finite_vector(initial.get("position_m"), 3, f"{label}.position_m")
    speed = _finite_float(initial.get("velocity_mps"), f"{label}.velocity_mps")
    heading_deg = _finite_float(initial.get("heading_deg"), f"{label}.heading_deg") % 360.0
    heading_rad = math.radians(heading_deg)
    return {
        "position_neu_m": position,
        "velocity_neu_mps": [speed * math.cos(heading_rad), speed * math.sin(heading_rad), 0.0],
        "speed_mps": speed,
        "altitude_m": position[2],
        "heading_deg": heading_deg,
    }


def _heading_error_deg(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def scenario_application_receipt(
    scenario: Mapping[str, Any], raw_states: Mapping[str, Mapping[str, Any]], tolerances: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare requested NEU geometry against the raw JSBSim state immediately after reset."""

    own_requested = _requested_kinematics(scenario["own_init"], "requested.own")
    target_requested = _requested_kinematics(scenario["target_init"], "requested.target")
    own_applied = _state_kinematics(raw_states["own"], "applied.own")
    target_applied = _state_kinematics(raw_states["target"], "applied.target")
    checks: dict[str, dict[str, float | bool]] = {}
    for label, requested, applied in (
        ("own", own_requested, own_applied),
        ("target", target_requested, target_applied),
    ):
        position_error = float(
            np.linalg.norm(
                np.asarray(requested["position_neu_m"]) - np.asarray(applied["position_neu_m"])
            )
        )
        velocity_error = float(
            np.linalg.norm(
                np.asarray(requested["velocity_neu_mps"]) - np.asarray(applied["velocity_neu_mps"])
            )
        )
        heading_error = _heading_error_deg(
            float(requested["heading_deg"]), float(applied["heading_deg"])
        )
        altitude_error = abs(float(requested["altitude_m"]) - float(applied["altitude_m"]))
        checks[label] = {
            "position_error_m": position_error,
            "velocity_error_mps": velocity_error,
            "heading_error_deg": heading_error,
            "altitude_error_m": altitude_error,
            "passed": (
                position_error <= _finite_float(tolerances["position_error_m"], "position tolerance")
                and velocity_error <= _finite_float(tolerances["velocity_error_mps"], "velocity tolerance")
                and heading_error <= _finite_float(tolerances["heading_error_deg"], "heading tolerance")
                and altitude_error <= _finite_float(tolerances["altitude_error_m"], "altitude tolerance")
            ),
        }
    requested_relative = np.asarray(target_requested["position_neu_m"]) - np.asarray(
        own_requested["position_neu_m"]
    )
    requested_range = float(np.linalg.norm(requested_relative))
    requested_range_rate = float(
        np.dot(
            np.asarray(target_requested["velocity_neu_mps"]) - np.asarray(own_requested["velocity_neu_mps"]),
            requested_relative / max(requested_range, 1e-9),
        )
    )
    return {
        "requested": {"own": own_requested, "target": target_requested},
        "applied": {"own": own_applied, "target": target_applied},
        "checks": checks,
        "requested_range_m": requested_range,
        "requested_range_rate_mps": requested_range_rate,
        "passed": bool(checks["own"]["passed"] and checks["target"]["passed"]),
    }


def _phase_tracker_state(tracker: PhaseTracker) -> dict[str, int | bool]:
    return {
        "seen_merge": bool(tracker._seen_merge),
        "post_merge_steps": int(tracker._post_merge_steps),
    }


def phase_receipt(tracker: PhaseTracker, base: Mapping[str, Any]) -> dict[str, Any]:
    """Persist all mutable tracker inputs and outputs needed for an exact replay."""

    range_m = _finite_float(base["range_m"], "base.range_m")
    range_rate_mps = _finite_float(base["range_rate_mps"], "base.range_rate_mps")
    before = _phase_tracker_state(tracker)
    phase = tracker.update(range_m, range_rate_mps)
    after = _phase_tracker_state(tracker)
    return {
        "range_m": range_m,
        "range_rate_mps": range_rate_mps,
        "merge_range_m": float(tracker.merge_range_m),
        "reentry_rate_mps": float(tracker.reentry_rate_mps),
        "tracker_before": before,
        "phase_after_update": phase,
        "tracker_after": after,
    }


def replay_phase_receipts(receipts: Sequence[Mapping[str, Any]]) -> bool:
    """Re-run PhaseTracker solely from persisted receipts; never infer missing inputs."""

    if not receipts:
        return False
    first = receipts[0]
    tracker = PhaseTracker(
        merge_range_m=_finite_float(first["merge_range_m"], "receipt.merge_range_m"),
        reentry_rate_mps=_finite_float(first["reentry_rate_mps"], "receipt.reentry_rate_mps"),
    )
    for receipt in receipts:
        if (
            float(receipt["merge_range_m"]) != tracker.merge_range_m
            or float(receipt["reentry_rate_mps"]) != tracker.reentry_rate_mps
            or dict(receipt["tracker_before"]) != _phase_tracker_state(tracker)
        ):
            return False
        phase = tracker.update(float(receipt["range_m"]), float(receipt["range_rate_mps"]))
        if phase != receipt["phase_after_update"] or dict(receipt["tracker_after"]) != _phase_tracker_state(tracker):
            return False
    return True


def _require_sha256(path: Path, expected: Any, label: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64:
        raise PhaseObservabilityError(f"{label} must declare SHA-256")
    if not path.is_file() or _sha256(path) != expected.lower():
        raise PhaseObservabilityError(f"{label} SHA mismatch")


def _validate_authorization(config_path: Path, authorization: Mapping[str, Any]) -> None:
    permitted = authorization.get("execution_permitted")
    if not isinstance(permitted, bool):
        raise PhaseObservabilityError("execution_permitted must be explicit")
    if not permitted:
        return
    if config_path.resolve() != DEFAULT_CONFIG.resolve():
        raise PhaseObservabilityError("execution requires the canonical B3 config")
    required_sha = authorization.get("required_implementation_git_sha")
    if (
        not isinstance(required_sha, str)
        or len(required_sha) != 40
        or not _git_is_ancestor(required_sha)
        or _git_value("status", "--porcelain") != ""
    ):
        raise PhaseObservabilityError(
            "execution requires a clean worktree and frozen implementation ancestor"
        )
    unexpected = _git_changed_paths_since(required_sha) - AUTHORIZATION_DELTA_PATHS
    if unexpected:
        raise PhaseObservabilityError(
            f"execution contains non-authorization changes after freeze: {sorted(unexpected)}"
        )
    authorized_files = authorization.get("authorized_code_files")
    if not isinstance(authorized_files, Sequence) or not authorized_files:
        raise PhaseObservabilityError("execution requires authorized code-file hashes")
    for entry in authorized_files:
        if not isinstance(entry, Mapping):
            raise PhaseObservabilityError("authorized code-file entry must be a mapping")
        path = _repo_path(str(entry.get("path", "")))
        _require_sha256(path, entry.get("sha256"), f"authorized code file {path}")


def _validate_registry_assets(registry: Mapping[str, Any], specialist_key: str) -> None:
    opponents = registry.get("opponents")
    specialists = registry.get("specialists")
    if not isinstance(opponents, Mapping) or not isinstance(specialists, Mapping):
        raise PhaseObservabilityError("runtime registry lacks opponents or specialists")
    for opponent in OPPONENTS:
        entry = opponents.get(opponent)
        if not isinstance(entry, Mapping):
            raise PhaseObservabilityError(f"runtime registry lacks opponent: {opponent}")
        checkpoint = entry.get("checkpoint")
        if opponent == "expert":
            if checkpoint is not None:
                raise PhaseObservabilityError("expert opponent must not load a checkpoint")
            continue
        _require_sha256(
            Path(str(checkpoint)), entry.get("checkpoint_sha256"), f"opponent checkpoint {opponent}"
        )
    specialist = specialists.get(specialist_key)
    if not isinstance(specialist, Mapping):
        raise PhaseObservabilityError("reference specialist missing from registry")
    _require_sha256(
        Path(str(specialist.get("checkpoint"))),
        specialist.get("checkpoint_sha256"),
        "reference specialist checkpoint",
    )


def _validate_sources(
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    config = _load_yaml(config_path)
    if config.get("source_id") != SOURCE_ID:
        raise PhaseObservabilityError("unexpected B3 source ID")
    authorization = config.get("authorization")
    if not isinstance(authorization, Mapping):
        raise PhaseObservabilityError("authorization must be a mapping")
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
        raise PhaseObservabilityError("B3 authorization drifted")
    contract = config.get("contract")
    if not isinstance(contract, Mapping):
        raise PhaseObservabilityError("contract must be a mapping")
    if (
        contract.get("backend") != "jsbsim"
        or contract.get("strict_backend") is not True
        or contract.get("fresh_environment_per_episode") is not True
        or tuple(contract.get("opponents", ())) != OPPONENTS
    ):
        raise PhaseObservabilityError("B3 runtime contract drifted")
    inputs = config.get("inputs")
    if not isinstance(inputs, Mapping):
        raise PhaseObservabilityError("inputs must be a mapping")
    manifest_path = _repo_path(str(inputs.get("manifest", "")))
    runtime_path = _repo_path(str(inputs.get("runtime_config", "")))
    registry_path = _repo_path(str(inputs.get("runtime_registry", "")))
    _require_sha256(manifest_path, inputs.get("manifest_sha256"), "B3 manifest")
    _require_sha256(runtime_path, inputs.get("runtime_config_sha256"), "B3 runtime config")
    _require_sha256(registry_path, inputs.get("runtime_registry_sha256"), "B3 runtime registry")
    manifest = _load_yaml(manifest_path)
    scenarios = list(manifest.get("scenarios") or [])
    if (
        manifest.get("source_id") != inputs.get("manifest_source_id")
        or manifest.get("source_id") != MANIFEST_SOURCE_ID
        or len(scenarios) != 30
        or len({item.get("name") for item in scenarios}) != 30
    ):
        raise PhaseObservabilityError("B3 manifest contract drifted")
    if not all((item.get("metadata") or {}).get("phase_at_reset") == "pre_merge" for item in scenarios):
        raise PhaseObservabilityError("B3 manifest loses its pre-merge reset contract")
    runtime = _load_yaml(runtime_path)
    registry = _load_yaml(registry_path)
    _validate_registry_assets(registry, str(inputs.get("reference_specialist", "")))
    return config, manifest, runtime, registry, scenarios


def _episode_path(output: Path, opponent: str, scenario: Mapping[str, Any]) -> Path:
    return output / "episodes" / opponent / f"{str(scenario['name']).replace('/', '_')}.json"


def _episode(
    runtime: Mapping[str, Any],
    registry: Mapping[str, Any],
    scenario: Mapping[str, Any],
    opponent: str,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    env = pilot._build_env(runtime, registry, opponent)
    reference = pilot._specialist(registry, "run_in_head_on")
    try:
        observation = env.reset(scenario=dict(scenario), seed=int(scenario["metadata"]["scenario_seed"]))
        if getattr(env, "_backend", None) != "jsbsim":
            raise PhaseObservabilityError("strict JSBSim backend was not selected")
        raw_states = env.jsbsim_env.get_state()
        reset_receipt = scenario_application_receipt(
            scenario, raw_states, contract["scenario_application_tolerances"]
        )
        reset_base = base_observation(observation)
        tracker = PhaseTracker()
        step0_phase = phase_receipt(tracker, reset_base)
        expected_phase = str(scenario["metadata"]["phase_at_reset"])
        steps: list[dict[str, Any]] = []
        terminal_reason = "horizon"
        phase_before_action = str(step0_phase["phase_after_update"])
        for index in range(1, int(contract["max_high_level_steps"]) + 1):
            action = _finite_vector(
                reference.get_deterministic_action(observation["observation_vector"]),
                3,
                "reference action",
            )
            next_observation, _reward, terminated, truncated, info = env.step(action)
            own = _state_kinematics(info.get("own_state", {}), "step.own")
            target = _state_kinematics(info.get("target_state", {}), "step.target")
            next_base = base_observation(next_observation)
            phase_after_step = phase_receipt(tracker, next_base)
            fallback = bool(info.get("prediction_fallback", False)) and info.get(
                "prediction_fallback_phase"
            ) != "warmup"
            steps.append(
                {
                    "step": index,
                    "base_observation": next_base,
                    "phase_before_action": phase_before_action,
                    "phase_after_step": phase_after_step,
                    "first_pass_complete": bool(info.get("first_pass_complete", False)),
                    "backend": info.get("backend"),
                    "backend_fallback": bool(info.get("backend_fallback_occurred", False)),
                    "prediction_fallback": fallback,
                    "action": action,
                    "own": own,
                    "target": target,
                    "terminal_reason": info.get("reason"),
                }
            )
            phase_before_action = str(phase_after_step["phase_after_update"])
            observation = next_observation
            if terminated or truncated:
                terminal_reason = str(info.get("reason") or "terminated")
                break
        phase_replay = replay_phase_receipts(
            [step0_phase, *[item["phase_after_step"] for item in steps]]
        )
        valid = (
            bool(steps)
            and reset_receipt["passed"]
            and phase_replay
            and all(
                item["backend"] == "jsbsim"
                and not item["backend_fallback"]
                and not item["prediction_fallback"]
                for item in steps
            )
        )
        return {
            "source_id": SOURCE_ID,
            "opponent": opponent,
            "scenario_name": scenario["name"],
            "scenario_metadata": scenario["metadata"],
            "step_count": len(steps),
            "terminal_reason": terminal_reason,
            "first_pass_observed": any(item["first_pass_complete"] for item in steps),
            "reset_receipt": reset_receipt,
            "step0": {"base_observation": reset_base, "phase": step0_phase},
            "phase_replay_consistent": phase_replay,
            "phase_counts": dict(
                Counter(item["phase_after_step"]["phase_after_update"] for item in steps)
            ),
            "valid": valid,
            "steps": steps,
        }
    finally:
        env.close()


def _provenance(config_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    inputs = config["inputs"]
    return {
        "git_sha": _git_value("rev-parse", "HEAD"),
        "git_dirty": _git_value("status", "--porcelain") != "",
        "config_sha256": _sha256(config_path),
        "manifest_sha256": _sha256(_repo_path(str(inputs["manifest"]))),
        "runtime_config_sha256": _sha256(_repo_path(str(inputs["runtime_config"]))),
        "runtime_registry_sha256": _sha256(_repo_path(str(inputs["runtime_registry"]))),
        "required_implementation_git_sha": config["authorization"].get(
            "required_implementation_git_sha"
        ),
        "authorized_code_files": list(config["authorization"].get("authorized_code_files") or []),
    }


def run(config_path: Path) -> dict[str, Any]:
    config, _manifest, runtime, registry, scenarios = _validate_sources(config_path)
    if config["authorization"].get("execution_permitted") is not True:
        raise PhaseObservabilityError("P2-B3 is not authorised")
    contract = config["contract"]
    output = Path(config["outputs"]["root"])
    if output.exists():
        raise PhaseObservabilityError("B3 output root must be fresh")
    free_gb = shutil.disk_usage(output.parent).free / (1024**3)
    if free_gb < float(contract["min_free_disk_gb"]):
        raise PhaseObservabilityError("B3 disk gate failed")
    output.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    try:
        for opponent in OPPONENTS:
            for scenario in scenarios:
                record = _episode(runtime, registry, scenario, opponent, contract)
                _write_json(_episode_path(output, opponent, scenario), record)
                records.append(record)
        expected = len(scenarios) * len(OPPONENTS)
        phase_matrix = {
            opponent: {
                phase: sum(
                    record["phase_counts"].get(phase, 0)
                    for record in records
                    if record["opponent"] == opponent
                )
                for phase in PHASES
            }
            for opponent in OPPONENTS
        }
        structural_passed = len(records) == expected and all(
            record["valid"] and record["step_count"] >= int(contract["min_valid_steps"])
            for record in records
        )
        scenario_application_passed = all(record["reset_receipt"]["passed"] for record in records)
        phase_replay_passed = all(record["phase_replay_consistent"] for record in records)
        initial_pre_merge_passed = all(
            record["step0"]["phase"]["phase_after_update"]
            == record["scenario_metadata"]["phase_at_reset"]
            for record in records
        )
        gate = {
            "source_id": SOURCE_ID,
            "record_count": len(records),
            "expected_record_count": expected,
            "structural_physical_contract_passed": structural_passed,
            "scenario_application_passed": scenario_application_passed,
            "phase_replay_passed": phase_replay_passed,
            "initial_pre_merge_semantics_passed": initial_pre_merge_passed,
            "phase_observability_passed": (
                structural_passed
                and scenario_application_passed
                and phase_replay_passed
                and initial_pre_merge_passed
            ),
            "phase_matrix": phase_matrix,
            "training_unlocked": False,
        }
        _write_json(output / "phase_observability_gate.json", gate)
        _write_json(
            output / "run_manifest.json",
            {"source_id": SOURCE_ID, "records": records, "provenance": _provenance(config_path, config), "gate": gate},
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
    if not args.execute:
        config, _manifest, _runtime, _registry, scenarios = _validate_sources(args.config)
        print(
            json.dumps(
                {
                    "source_id": config["source_id"],
                    "mode": "validated_not_executed",
                    "execution_permitted": config["authorization"]["execution_permitted"],
                    "planned_records": len(scenarios) * len(OPPONENTS),
                },
                indent=2,
            )
        )
        return 0
    gate = run(args.config)
    print(json.dumps(gate, indent=2))
    return 0 if gate["phase_observability_passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
