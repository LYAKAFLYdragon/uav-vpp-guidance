#!/usr/bin/env python3
"""Run the V5 raw-SI observability collector after a separate frozen overlay."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scripts.preflight_thesis_five_state_raw_si_v5_observability import (
    DEFAULT_CONFIG,
    ROOT as PREFLIGHT_ROOT,
    load_yaml,
    sha256_file,
    validate_design,
)
from uav_vpp_guidance.evaluation.thesis_five_state_raw_si_v5_contract import (
    SOURCE_ID,
    FiveStateRawSIV5ContractError,
    build_plan,
    evaluate_observability,
    expected_pairs_by_state,
    validate_manifest,
)
from uav_vpp_guidance.evaluation.thesis_five_state_raw_si_v5_record_only import (
    FiveStateRawSIV5Episode,
    build_record_only_env,
    runtime_record,
)


if ROOT != PREFLIGHT_ROOT:
    raise RuntimeError("V5 runner and preflight disagree on repository root")


REQUIRED_AUTHORIZED_CODE_PATHS = (
    "scripts/build_thesis_five_state_raw_si_v5_manifest.py",
    "scripts/preflight_thesis_five_state_raw_si_v5_observability.py",
    "scripts/run_thesis_five_state_raw_si_v5_observability.py",
    "src/uav_vpp_guidance/evaluation/raw_si_first_pass_event.py",
    "src/uav_vpp_guidance/evaluation/single_motif_continuous_66d.py",
    "src/uav_vpp_guidance/evaluation/thesis_five_state_raw_si_v5_contract.py",
    "src/uav_vpp_guidance/evaluation/thesis_five_state_raw_si_v5_record_only.py",
    "src/uav_vpp_guidance/evaluation/p4_v2_sampler_feasibility.py",
    "src/uav_vpp_guidance/envs/tracking_env.py",
    "src/uav_vpp_guidance/envs/expert_opponent.py",
    "src/uav_vpp_guidance/envs/end_to_end_opponent.py",
    "src/uav_vpp_guidance/envs/vpp_ppo_opponent.py",
    "src/uav_vpp_guidance/evaluation/feature_name_projected_opponent.py",
    "src/uav_vpp_guidance/hierarchy/five_state_observation_contract.py",
    "src/uav_vpp_guidance/hierarchy/shared_intent_profiles.py",
    "src/uav_vpp_guidance/hierarchy/specialist_policy.py",
    "src/uav_vpp_guidance/hierarchy/temporal_geometry_features.py",
    "src/uav_vpp_guidance/training/thesis_defext_rangeext_pilot.py",
    "src/uav_vpp_guidance/training/thesis_neutral_postmerge_reentry_recovery_pilot.py",
    "src/uav_vpp_guidance/training/thesis_shared_skill_geometry.py",
)


def _merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _repo_path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _require_hash(path: Path, expected: Any, label: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64 or not path.is_file() or sha256_file(path) != expected.lower():
        raise RuntimeError("{0} SHA-256 mismatch".format(label))


def _git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _load_authorized_config(path: Path) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    overlay = load_yaml(path)
    base_value = overlay.get("base_config")
    if not base_value:
        raise RuntimeError("V5 execution requires a separate authorized overlay")
    base_path = _repo_path(str(base_value))
    execution = overlay.get("execution") if isinstance(overlay.get("execution"), Mapping) else {}
    _require_hash(base_path, execution.get("base_config_sha256"), "V5 base config")
    merge_overlay = dict(overlay)
    merge_overlay.pop("base_config", None)
    return _merge(load_yaml(base_path), merge_overlay), base_path, overlay


def _validate_overlay_scope(overlay: Mapping[str, Any], base_path: Path) -> None:
    allowed = {"base_config", "status", "authorization", "execution", "outputs"}
    unexpected = set(overlay) - allowed
    if unexpected:
        raise RuntimeError("V5 authorization overlay cannot alter frozen method fields: {0}".format(sorted(unexpected)))
    outputs = overlay.get("outputs", {})
    if not isinstance(outputs, Mapping) or set(outputs) - {"creation_permitted_by_this_config"}:
        raise RuntimeError("V5 overlay may only permit the predeclared output root")
    if outputs.get("creation_permitted_by_this_config") is not True:
        raise RuntimeError("V5 overlay must explicitly permit output creation")
    execution = overlay.get("execution", {})
    permitted_execution = {
        "base_config_sha256",
        "required_implementation_git_sha",
        "authorization_delta_paths",
        "authorized_code_files",
    }
    if not isinstance(execution, Mapping) or set(execution) - permitted_execution:
        raise RuntimeError("V5 overlay cannot alter frozen execution inputs")
    if overlay.get("base_config") != str(base_path.relative_to(ROOT)).replace("\\", "/"):
        raise RuntimeError("V5 overlay must reference the canonical design config")


def _validate_authorization(config: Mapping[str, Any], base_path: Path, overlay: Mapping[str, Any]) -> None:
    _validate_overlay_scope(overlay, base_path)
    if config.get("source_id") != SOURCE_ID or config.get("status") != "authorized_one_time_execution":
        raise RuntimeError("unexpected V5 execution source or status")
    if "proposed_training" in config:
        raise RuntimeError("V5 execution must not define proposed_training")
    if config.get("observability", {}).get("training_reward") != "prohibited":
        raise RuntimeError("V5 execution must remain reward-unobserved")
    authorization = config.get("authorization") or {}
    if authorization.get("execution_permitted") is not True:
        raise RuntimeError("V5 execution is not authorized")
    for key, value in authorization.items():
        if key != "execution_permitted" and value is not False:
            raise RuntimeError("V5 authorization.{0} must remain false".format(key))
    execution = config.get("execution") or {}
    required_sha = str(execution.get("required_implementation_git_sha", ""))
    if not required_sha or subprocess.run(["git", "merge-base", "--is-ancestor", required_sha, "HEAD"], cwd=ROOT, check=False).returncode != 0:
        raise RuntimeError("V5 execution requires its frozen implementation SHA")
    if _git_value("status", "--porcelain"):
        raise RuntimeError("V5 execution requires a clean worktree")
    allowed_delta = {str(item) for item in execution.get("authorization_delta_paths", [])}
    changed = {
        line.replace("\\", "/") for line in _git_value("diff", "--name-only", "{0}..HEAD".format(required_sha)).splitlines() if line.strip()
    }
    if changed - allowed_delta:
        raise RuntimeError("V5 has unauthorized post-freeze changes: {0}".format(sorted(changed - allowed_delta)))
    files = execution.get("authorized_code_files")
    if not isinstance(files, list) or len(files) != len(REQUIRED_AUTHORIZED_CODE_PATHS):
        raise RuntimeError("V5 authorized code whitelist is incomplete")
    declared = {str(item.get("path", "")).replace("\\", "/") for item in files if isinstance(item, Mapping)}
    required = set(REQUIRED_AUTHORIZED_CODE_PATHS)
    if declared != required:
        raise RuntimeError("V5 authorized code whitelist drifted")
    for entry in files:
        if not isinstance(entry, Mapping):
            raise RuntimeError("V5 authorized code entry must be a mapping")
        _require_hash(_repo_path(str(entry.get("path", ""))), entry.get("sha256"), "V5 authorized code")
    validate_design(base_path)


def _write_json(path: Path, payload: Any) -> None:
    from uav_vpp_guidance.training.thesis_neutral_postmerge_reentry_recovery_pilot import write_json

    write_json(path, payload)


def _run_episode(
    config: Mapping[str, Any], registry: Mapping[str, Any], scenario: Mapping[str, Any], opponent: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    from scripts.run_thesis_global_advantage_p2_b3_phase_observability import scenario_application_receipt
    from uav_vpp_guidance.training import thesis_neutral_postmerge_reentry_recovery_pilot as engine

    env = build_record_only_env(config, registry, opponent)
    reset_capture: dict[str, Any] = {}
    original_reset = env.reset

    def _capturing_reset(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
        observation = original_reset(*args, **kwargs)
        reset_capture["raw_state"] = env.jsbsim_env.get_state()
        return observation

    env.reset = _capturing_reset  # type: ignore[method-assign]
    try:
        fixed = engine._specialist(registry, "frozen_fixed_head_on")
        encoder = engine.FrozenP3Encoder(
            Path(config["inputs"]["frozen_p3_encoder"]),
            str(config["inputs"]["frozen_p3_encoder_sha256"]),
        )
        episode = FiveStateRawSIV5Episode(env, fixed, encoder, config)
        try:
            result, ledger = episode.run(scenario, int(scenario["metadata"]["scenario_seed"]))
        finally:
            episode.close()
        raw_state = reset_capture.get("raw_state")
        if not isinstance(raw_state, Mapping):
            raise RuntimeError("V5 did not capture the immediate JSBSim reset state")
        receipt = scenario_application_receipt(
            scenario, raw_state, config["contract"]["scenario_application_tolerances"]
        )
        evidence = {
            "scenario_application_passed": bool(receipt.get("passed")),
            "scenario_application_receipt": receipt,
            "raw_si_event_replay_passed": bool(ledger["summary"].get("raw_si_event_replay_passed")),
            "fixed_head_on_action_only": True,
        }
        return result, ledger, evidence
    finally:
        # FiveStateRawSIV5Episode closes the environment in the normal path.
        # This fallback covers failures before the episode is constructed.
        try:
            env.close()
        except Exception:
            pass


def authorized_preflight(config_path: Path) -> dict[str, Any]:
    config, base_path, overlay = _load_authorized_config(config_path)
    _validate_authorization(config, base_path, overlay)
    plan = build_plan(config)
    manifest_path = _repo_path(str(config["inputs"]["observability_manifest"]))
    _require_hash(manifest_path, config["inputs"].get("observability_manifest_sha256"), "V5 observability manifest")
    manifest = load_yaml(manifest_path)
    validate_manifest(manifest)
    output = Path(str(config["outputs"]["root"]))
    if config["outputs"].get("creation_permitted_by_this_config") is not True:
        raise RuntimeError("V5 output creation is not authorized")
    free_gb = shutil.disk_usage(output.parent).free / (1024 ** 3)
    if not output.is_absolute() or output.exists() or free_gb < plan.min_free_disk_gb:
        raise RuntimeError("V5 output root failed fresh-output or disk gate")
    return {
        "source_id": SOURCE_ID,
        "authorized": True,
        "planned_records": 180,
        "output_root": str(output),
        "free_disk_gb": round(free_gb, 3),
        "reference_controller": "frozen_fixed_head_on",
        "training_permitted": False,
    }


def run(config_path: Path) -> dict[str, Any]:
    config, base_path, overlay = _load_authorized_config(config_path)
    preflight = authorized_preflight(config_path)
    plan = build_plan(config)
    manifest_path = _repo_path(str(config["inputs"]["observability_manifest"]))
    manifest = load_yaml(manifest_path)
    registry = load_yaml(_repo_path(str(config["inputs"]["runtime_registry"])))
    output = Path(str(config["outputs"]["root"]))
    output.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    try:
        shutil.copy2(config_path, output / "authorized_overlay.yaml")
        shutil.copy2(base_path, output / "frozen_base_config.yaml")
        shutil.copy2(manifest_path, output / "frozen_observability_manifest.yaml")
        _write_json(output / "authorization_preflight.json", {
            **preflight,
            "git_sha": _git_value("rev-parse", "HEAD"),
            "authorized_overlay_sha256": sha256_file(config_path),
            "frozen_base_config_sha256": sha256_file(base_path),
        })
        _write_json(output / "resolved_config.json", config)
        _write_json(output / "artifact_provenance.json", {
            "source_id": SOURCE_ID,
            "git_sha": _git_value("rev-parse", "HEAD"),
            "frozen_observability_manifest_sha256": sha256_file(manifest_path),
            "runtime_template_sha256": config["inputs"]["runtime_template_sha256"],
            "runtime_registry_sha256": config["inputs"]["runtime_registry_sha256"],
            "frozen_p3_encoder_sha256": config["inputs"]["frozen_p3_encoder_sha256"],
            "frozen_fixed_head_on_sha256": config["inputs"]["frozen_fixed_head_on_sha256"],
            "training_reward": "prohibited",
            "environment_reward_recording": "prohibited",
        })
        for opponent in plan.opponents:
            for scenario in manifest["scenarios"]:
                result, ledger, evidence = _run_episode(config, registry, scenario, opponent)
                record = runtime_record(
                    opponent=opponent, scenario=scenario, result=result, ledger=ledger, runtime_evidence=evidence
                )
                records.append(record)
                _write_json(output / "episodes" / opponent / "{0}.json".format(scenario["name"]), {
                    "record": record, "ledger": ledger,
                })
        gate = evaluate_observability(records, plan, expected_pairs_by_state(manifest))
        gate["training_unlocked"] = False
        _write_json(output / "raw_si_observability_gate.json", gate)
        _write_json(output / "run_manifest.json", {
            "source_id": SOURCE_ID,
            "paper_safe": False,
            "mode": "nonlearning_raw_si_phase_observability",
            "record_count": len(records),
            "training_reward": "prohibited",
            "environment_reward_recording": "prohibited",
            "gate": gate,
        })
        return gate
    except Exception as error:
        _write_json(output / "runner_failure_manifest.json", {
            "source_id": SOURCE_ID,
            "completed_record_count": len(records),
            "error": repr(error),
            "training_reward": "prohibited",
        })
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.preflight == args.execute:
        print("choose exactly one of --preflight or --execute", file=sys.stderr)
        return 2
    try:
        rendered = authorized_preflight(args.config) if args.preflight else run(args.config)
        print(json.dumps(rendered, indent=2, ensure_ascii=False))
        return 0
    except (OSError, RuntimeError, ValueError, yaml.YAMLError, FiveStateRawSIV5ContractError) as error:
        print("V5 observability failed: {0}".format(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
