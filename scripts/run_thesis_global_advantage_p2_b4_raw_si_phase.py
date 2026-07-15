"""One-shot P2-B4 preflight: phase tracking uses raw SI relative geometry."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_thesis_global_advantage_p2_b3_phase_observability as b3


SOURCE_ID = "THESIS-GLOBAL-ADVANTAGE-V1-P2-RAW-SI-PHASE-PREFLIGHT-V1"
MANIFEST_SOURCE_ID = "THESIS-GLOBAL-ADVANTAGE-V1-P2-RAW-SI-PHASE-MANIFEST30-V1"
DEFAULT_CONFIG = ROOT / "config" / "experiment" / "thesis_global_advantage_v1_p2_b4_raw_si_phase.yaml"
AUTHORIZATION_DELTA_PATHS = {
    "config/experiment/thesis_global_advantage_v1_p2_b4_raw_si_phase.yaml",
    "reports/thesis_global_advantage_p2_b4_raw_si_phase_execution_authorization_20260715_zh.md",
    "reports/thesis_five_state_global_advantage_goal_checklist_v1_20260714_zh.md",
}


class RawSIPhaseError(ValueError):
    """Raised when the B4 raw-SI phase contract drifts or cannot be measured."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_si_phase_input(observation: Mapping[str, Any]) -> dict[str, float | str]:
    """Use unnormalised relative geometry; policy-vector names are not SI values."""

    relative = observation.get("relative_state")
    if not isinstance(relative, Mapping):
        raise RawSIPhaseError("observation is missing raw relative_state")
    return {
        "range_m": b3._finite_float(relative.get("range_m"), "relative_state.range_m"),
        "range_rate_mps": b3._finite_float(
            relative.get("range_rate_mps"), "relative_state.range_rate_mps"
        ),
        "unit_contract": "raw_SI_from_observation.relative_state",
    }


def normalized_policy_diagnostic(observation: Mapping[str, Any]) -> dict[str, float]:
    base = b3.base_observation(observation)
    return {
        "range_m_normalized": b3._finite_float(base["range_m"], "policy.range_m"),
        "range_rate_normalized": b3._finite_float(
            base["range_rate_mps"], "policy.range_rate_mps"
        ),
    }


def _phase_receipt(tracker: b3.PhaseTracker, raw_input: Mapping[str, Any]) -> dict[str, Any]:
    return b3.phase_receipt(
        tracker,
        {
            "range_m": raw_input["range_m"],
            "range_rate_mps": raw_input["range_rate_mps"],
        },
    )


def _normalization_consistent(
    raw_input: Mapping[str, Any], policy: Mapping[str, Any], normalization: Mapping[str, Any]
) -> bool:
    range_scale = b3._finite_float(normalization["range_m_scale"], "range_m_scale")
    rate_scale = b3._finite_float(normalization["range_rate_mps_scale"], "range_rate_mps_scale")
    range_tolerance = b3._finite_float(normalization["range_tolerance_m"], "range_tolerance_m")
    rate_tolerance = b3._finite_float(normalization["range_rate_tolerance_mps"], "rate tolerance")
    return (
        abs(float(policy["range_m_normalized"]) * range_scale - float(raw_input["range_m"]))
        <= range_tolerance
        and abs(
            float(policy["range_rate_normalized"]) * rate_scale
            - float(raw_input["range_rate_mps"])
        )
        <= rate_tolerance
    )


def _require_sha256(path: Path, expected: Any, label: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64:
        raise RawSIPhaseError(f"{label} must declare SHA-256")
    if not path.is_file() or _sha256(path) != expected.lower():
        raise RawSIPhaseError(f"{label} SHA mismatch")


def _validate_authorization(config_path: Path, authorization: Mapping[str, Any]) -> None:
    permitted = authorization.get("execution_permitted")
    if not isinstance(permitted, bool):
        raise RawSIPhaseError("execution_permitted must be explicit")
    if not permitted:
        return
    if config_path.resolve() != DEFAULT_CONFIG.resolve():
        raise RawSIPhaseError("execution requires canonical B4 config")
    required_sha = authorization.get("required_implementation_git_sha")
    if (
        not isinstance(required_sha, str)
        or len(required_sha) != 40
        or not b3._git_is_ancestor(required_sha)
        or b3._git_value("status", "--porcelain") != ""
    ):
        raise RawSIPhaseError("execution requires clean worktree and frozen implementation")
    unexpected = b3._git_changed_paths_since(required_sha) - AUTHORIZATION_DELTA_PATHS
    if unexpected:
        raise RawSIPhaseError(f"non-authorization changes after freeze: {sorted(unexpected)}")
    authorized_files = authorization.get("authorized_code_files")
    if not isinstance(authorized_files, Sequence) or not authorized_files:
        raise RawSIPhaseError("execution requires authorised code hashes")
    for entry in authorized_files:
        if not isinstance(entry, Mapping):
            raise RawSIPhaseError("authorised code file entry must be a mapping")
        _require_sha256(
            b3._repo_path(str(entry.get("path", ""))),
            entry.get("sha256"),
            f"authorised code file {entry.get('path')}",
        )


def _validate_sources(
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    config = b3._load_yaml(config_path)
    if config.get("source_id") != SOURCE_ID:
        raise RawSIPhaseError("unexpected B4 source ID")
    authorization = config.get("authorization")
    if not isinstance(authorization, Mapping):
        raise RawSIPhaseError("authorization must be a mapping")
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
        raise RawSIPhaseError("B4 authorization drifted")
    contract = config.get("contract")
    if not isinstance(contract, Mapping):
        raise RawSIPhaseError("contract must be a mapping")
    if (
        contract.get("backend") != "jsbsim"
        or contract.get("strict_backend") is not True
        or contract.get("fresh_environment_per_episode") is not True
        or tuple(contract.get("opponents", ())) != b3.OPPONENTS
        or contract.get("phase_input_unit_contract") != "raw_SI_from_observation.relative_state"
    ):
        raise RawSIPhaseError("B4 contract drifted")
    inputs = config.get("inputs")
    if not isinstance(inputs, Mapping):
        raise RawSIPhaseError("inputs must be a mapping")
    manifest_path = b3._repo_path(str(inputs.get("manifest", "")))
    runtime_path = b3._repo_path(str(inputs.get("runtime_config", "")))
    registry_path = b3._repo_path(str(inputs.get("runtime_registry", "")))
    _require_sha256(manifest_path, inputs.get("manifest_sha256"), "B4 manifest")
    _require_sha256(runtime_path, inputs.get("runtime_config_sha256"), "B4 runtime config")
    _require_sha256(registry_path, inputs.get("runtime_registry_sha256"), "B4 runtime registry")
    manifest = b3._load_yaml(manifest_path)
    scenarios = list(manifest.get("scenarios") or [])
    if (
        manifest.get("source_id") != MANIFEST_SOURCE_ID
        or manifest.get("source_id") != inputs.get("manifest_source_id")
        or len(scenarios) != 30
        or not all((item.get("metadata") or {}).get("phase_at_reset") == "pre_merge" for item in scenarios)
    ):
        raise RawSIPhaseError("B4 manifest contract drifted")
    runtime = b3._load_yaml(runtime_path)
    registry = b3._load_yaml(registry_path)
    b3._validate_registry_assets(registry, str(inputs.get("reference_specialist", "")))
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
    env = b3.pilot._build_env(runtime, registry, opponent)
    reference = b3.pilot._specialist(registry, "run_in_head_on")
    try:
        observation = env.reset(scenario=dict(scenario), seed=int(scenario["metadata"]["scenario_seed"]))
        if getattr(env, "_backend", None) != "jsbsim":
            raise RawSIPhaseError("strict JSBSim backend was not selected")
        receipt = b3.scenario_application_receipt(
            scenario, env.jsbsim_env.get_state(), contract["scenario_application_tolerances"]
        )
        tracker = b3.PhaseTracker()
        step0_raw = raw_si_phase_input(observation)
        step0_policy = normalized_policy_diagnostic(observation)
        step0_phase = _phase_receipt(tracker, step0_raw)
        normalization = contract["policy_normalization_diagnostic"]
        step0_normalization_consistent = _normalization_consistent(
            step0_raw, step0_policy, normalization
        )
        phase_before_action = str(step0_phase["phase_after_update"])
        steps: list[dict[str, Any]] = []
        terminal_reason = "horizon"
        for index in range(1, int(contract["max_high_level_steps"]) + 1):
            action = b3._finite_vector(
                reference.get_deterministic_action(observation["observation_vector"]), 3, "action"
            )
            next_observation, _reward, terminated, truncated, info = env.step(action)
            raw_input = raw_si_phase_input(next_observation)
            policy = normalized_policy_diagnostic(next_observation)
            phase_after_step = _phase_receipt(tracker, raw_input)
            fallback = bool(info.get("prediction_fallback", False)) and info.get(
                "prediction_fallback_phase"
            ) != "warmup"
            steps.append(
                {
                    "step": index,
                    "phase_before_action": phase_before_action,
                    "phase_after_step": phase_after_step,
                    "phase_input_raw_si": raw_input,
                    "policy_normalized_diagnostic": policy,
                    "normalization_consistent": _normalization_consistent(
                        raw_input, policy, normalization
                    ),
                    "first_pass_complete": bool(info.get("first_pass_complete", False)),
                    "backend": info.get("backend"),
                    "backend_fallback": bool(info.get("backend_fallback_occurred", False)),
                    "prediction_fallback": fallback,
                    "action": action,
                    "terminal_reason": info.get("reason"),
                }
            )
            phase_before_action = str(phase_after_step["phase_after_update"])
            observation = next_observation
            if terminated or truncated:
                terminal_reason = str(info.get("reason") or "terminated")
                break
        replay = b3.replay_phase_receipts(
            [step0_phase, *[item["phase_after_step"] for item in steps]]
        )
        valid = (
            bool(steps)
            and receipt["passed"]
            and replay
            and step0_normalization_consistent
            and all(
                item["backend"] == "jsbsim"
                and not item["backend_fallback"]
                and not item["prediction_fallback"]
                and item["normalization_consistent"]
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
            "reset_receipt": receipt,
            "step0": {
                "phase_input_raw_si": step0_raw,
                "policy_normalized_diagnostic": step0_policy,
                "normalization_consistent": step0_normalization_consistent,
                "phase": step0_phase,
            },
            "phase_replay_consistent": replay,
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
        "git_sha": b3._git_value("rev-parse", "HEAD"),
        "git_dirty": b3._git_value("status", "--porcelain") != "",
        "config_sha256": _sha256(config_path),
        "manifest_sha256": _sha256(b3._repo_path(str(inputs["manifest"]))),
        "runtime_config_sha256": _sha256(b3._repo_path(str(inputs["runtime_config"]))),
        "runtime_registry_sha256": _sha256(b3._repo_path(str(inputs["runtime_registry"]))),
        "required_implementation_git_sha": config["authorization"].get(
            "required_implementation_git_sha"
        ),
        "authorized_code_files": list(config["authorization"].get("authorized_code_files") or []),
    }


def run(config_path: Path) -> dict[str, Any]:
    config, _manifest, runtime, registry, scenarios = _validate_sources(config_path)
    if config["authorization"].get("execution_permitted") is not True:
        raise RawSIPhaseError("B4 is not authorised")
    contract = config["contract"]
    output = Path(config["outputs"]["root"])
    if output.exists():
        raise RawSIPhaseError("B4 output root must be fresh")
    if shutil.disk_usage(output.parent).free / (1024**3) < float(contract["min_free_disk_gb"]):
        raise RawSIPhaseError("B4 disk gate failed")
    output.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    try:
        for opponent in b3.OPPONENTS:
            for scenario in scenarios:
                record = _episode(runtime, registry, scenario, opponent, contract)
                b3._write_json(_episode_path(output, opponent, scenario), record)
                records.append(record)
        expected = len(scenarios) * len(b3.OPPONENTS)
        phase_matrix = {
            opponent: {
                phase: sum(
                    record["phase_counts"].get(phase, 0)
                    for record in records
                    if record["opponent"] == opponent
                )
                for phase in b3.PHASES
            }
            for opponent in b3.OPPONENTS
        }
        structural = len(records) == expected and all(
            record["valid"] and record["step_count"] >= int(contract["min_valid_steps"])
            for record in records
        )
        scenario_application = all(record["reset_receipt"]["passed"] for record in records)
        replay = all(record["phase_replay_consistent"] for record in records)
        pre_merge = all(
            record["step0"]["phase"]["phase_after_update"]
            == record["scenario_metadata"]["phase_at_reset"]
            for record in records
        )
        gate = {
            "source_id": SOURCE_ID,
            "record_count": len(records),
            "expected_record_count": expected,
            "structural_physical_contract_passed": structural,
            "scenario_application_passed": scenario_application,
            "phase_replay_passed": replay,
            "initial_pre_merge_semantics_passed": pre_merge,
            "raw_si_phase_contract_passed": structural and scenario_application and replay and pre_merge,
            "phase_matrix": phase_matrix,
            "training_unlocked": False,
        }
        b3._write_json(output / "raw_si_phase_gate.json", gate)
        b3._write_json(
            output / "run_manifest.json",
            {"source_id": SOURCE_ID, "records": records, "provenance": _provenance(config_path, config), "gate": gate},
        )
        return gate
    except Exception:
        b3._write_json(
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
        print(json.dumps({"source_id": config["source_id"], "mode": "validated_not_executed", "execution_permitted": config["authorization"]["execution_permitted"], "planned_records": len(scenarios) * len(b3.OPPONENTS)}, indent=2))
        return 0
    gate = run(args.config)
    print(json.dumps(gate, indent=2))
    return 0 if gate["raw_si_phase_contract_passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
