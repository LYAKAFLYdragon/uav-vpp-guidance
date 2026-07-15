#!/usr/bin/env python3
"""Run V4 reward-unobserved feasibility only after separate authorization."""

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

from scripts.preflight_thesis_neutral_postmerge_reentry_recovery_v4_runtime_feasibility import (
    DEFAULT_CONFIG,
    ROOT as PREFLIGHT_ROOT,
    load_yaml,
    sha256_file,
    validate_design,
)
from uav_vpp_guidance.evaluation.thesis_neutral_postmerge_v4_contract import (
    SOURCE_ID,
    NeutralPostMergeV4ContractError,
    build_plan,
    evaluate_family_coverage,
    runtime_identity_index,
    validate_feasibility_manifest,
)
from uav_vpp_guidance.evaluation.thesis_neutral_postmerge_v4_record_only import (
    RewardFreeNeutralPostMergeEpisode,
    build_record_only_env,
    record_only_result,
    runtime_record,
)


if ROOT != PREFLIGHT_ROOT:
    raise RuntimeError("V4 runner and preflight disagree on repository root")


def _merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _require_hash(path: Path, expected: Any, label: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64 or not path.is_file() or sha256_file(path) != expected.lower():
        raise RuntimeError(f"{label} SHA-256 mismatch")


def _git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _load_authorized_config(path: Path) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    overlay = load_yaml(path)
    base_value = overlay.get("base_config")
    if not base_value:
        raise RuntimeError("V4 execution requires a separate authorised overlay with base_config")
    base_path = _repo_path(str(base_value))
    execution = overlay.get("execution") if isinstance(overlay.get("execution"), Mapping) else {}
    _require_hash(base_path, execution.get("base_config_sha256"), "V4 base config")
    merge_overlay = dict(overlay)
    merge_overlay.pop("base_config", None)
    return _merge(load_yaml(base_path), merge_overlay), base_path, overlay


def _validate_overlay_scope(overlay: Mapping[str, Any], base_path: Path) -> None:
    allowed = {"base_config", "status", "authorization", "execution", "outputs"}
    unexpected = set(overlay) - allowed
    if unexpected:
        raise RuntimeError(f"V4 authorisation overlay cannot alter frozen method fields: {sorted(unexpected)}")
    outputs = overlay.get("outputs", {})
    if not isinstance(outputs, Mapping) or set(outputs) - {"creation_permitted_by_this_config"}:
        raise RuntimeError("V4 authorisation overlay may only permit the predeclared output root")
    if outputs.get("creation_permitted_by_this_config") is not True:
        raise RuntimeError("V4 authorisation overlay must explicitly permit only output-root creation")
    if overlay.get("base_config") != str(base_path.relative_to(ROOT)).replace("\\", "/"):
        raise RuntimeError("V4 authorisation overlay must reference the canonical design base config")


def _validate_authorization(config: Mapping[str, Any], base_path: Path, overlay: Mapping[str, Any]) -> None:
    _validate_overlay_scope(overlay, base_path)
    if config.get("source_id") != SOURCE_ID or config.get("status") != "authorized_one_time_execution":
        raise RuntimeError("unexpected V4 execution source or status")
    feasibility = config.get("runtime_feasibility", {})
    if (
        "proposed_training" in config
        or feasibility.get("training_reward") != "prohibited"
        or feasibility.get("environment_reward_recording") != "prohibited"
    ):
        raise RuntimeError("V4 authorization must retain the reward-unobserved record-only contract")
    authorization = config.get("authorization") or {}
    if authorization.get("execution_permitted") is not True:
        raise RuntimeError("V4 execution is not authorised")
    for key in (
        "training_permitted",
        "tuning_permitted",
        "candidate_policy_loading_permitted",
        "high_level_ppo_training_permitted",
        "four_skill_training_permitted",
        "combat_finetune_permitted",
        "vpp_change_permitted",
        "guidance_change_permitted",
        "pid_change_permitted",
        "snapshot_restore",
        "future_state_injection",
        "history_padding_permitted",
        "heldout_claims_permitted",
    ):
        if authorization.get(key) is not False:
            raise RuntimeError(f"V4 authorization.{key} must remain false")
    execution = config.get("execution") or {}
    required_sha = str(execution.get("required_implementation_git_sha", ""))
    if not required_sha or subprocess.run(["git", "merge-base", "--is-ancestor", required_sha, "HEAD"], cwd=ROOT, check=False).returncode != 0 or _git_value("status", "--porcelain"):
        raise RuntimeError("V4 execution requires a clean frozen implementation SHA")
    allowed = {str(item) for item in execution.get("authorization_delta_paths", [])}
    changed = {line.replace("\\", "/") for line in _git_value("diff", "--name-only", f"{required_sha}..HEAD").splitlines() if line.strip()}
    if changed - allowed:
        raise RuntimeError(f"V4 has unauthorised post-freeze changes: {sorted(changed - allowed)}")
    files = execution.get("authorized_code_files")
    if not isinstance(files, list) or not files:
        raise RuntimeError("V4 execution requires authorised code hashes")
    for entry in files:
        if not isinstance(entry, Mapping):
            raise RuntimeError("V4 authorised code entry must be a mapping")
        _require_hash(_repo_path(str(entry.get("path", ""))), entry.get("sha256"), "V4 authorised code")
    validate_design(base_path)


def _write_json(path: Path, payload: Any) -> None:
    from uav_vpp_guidance.training.thesis_neutral_postmerge_reentry_recovery_pilot import write_json

    write_json(path, payload)


def _phase_replay(ledger: Mapping[str, Any]) -> bool:
    from uav_vpp_guidance.training import thesis_neutral_postmerge_reentry_recovery_pilot as engine

    tracker = engine.legacy.PhaseTracker()
    for step in ledger.get("steps", []):
        raw = step.get("raw_si_phase_input") if isinstance(step, Mapping) else None
        if not isinstance(raw, Mapping):
            return False
        if tracker.update(float(raw["range_m"]), float(raw["range_rate_mps"])) != step.get("phase"):
            return False
    return True


def _run_episode(
    config: Mapping[str, Any], registry: Mapping[str, Any], scenario: Mapping[str, Any], opponent: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, bool]]:
    from uav_vpp_guidance.training import thesis_neutral_postmerge_reentry_recovery_pilot as engine
    from scripts.run_thesis_global_advantage_p2_b3_phase_observability import scenario_application_receipt

    with engine._legacy_target_contract():
        encoder = engine.FrozenP3Encoder(
            Path(config["fixed_contract"]["encoder"]["checkpoint"]),
            str(config["fixed_contract"]["encoder"]["sha256"]),
        )
        env = build_record_only_env(config, registry, opponent)
        reset_capture: dict[str, Any] = {}
        original_reset = env.reset

        def _capturing_reset(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
            observation = original_reset(*args, **kwargs)
            reset_capture["observation"] = observation
            reset_capture["raw_state"] = env.jsbsim_env.get_state()
            return observation

        env.reset = _capturing_reset  # type: ignore[method-assign]
        episode = RewardFreeNeutralPostMergeEpisode(
            env,
            engine._specialist(registry, "run_in_head_on"),
            encoder,
            config,
        )
        baseline = engine._specialist(registry, "frozen_fixed_head_on")
        try:
            observation, handoff = episode.reset_to_handoff(scenario, int(scenario["metadata"]["scenario_seed"]))
            reset_observation = reset_capture.get("observation")
            raw_state = reset_capture.get("raw_state")
            if not isinstance(reset_observation, Mapping) or not isinstance(raw_state, Mapping):
                raise RuntimeError("V4 runtime could not capture the immediate reset state")
            receipt = scenario_application_receipt(
                scenario,
                raw_state,
                config["contract"]["scenario_application_tolerances"],
            )
            raw_reset = reset_observation.get("relative_state")
            if not isinstance(raw_reset, Mapping):
                raise RuntimeError("V4 reset observation lacks raw-SI relative state")
            reset_phase = engine.legacy.PhaseTracker().update(
                float(raw_reset["range_m"]), float(raw_reset["range_rate_mps"])
            )
            metrics: list[dict[str, Any]] = []
            if observation is not None:
                episode.set_handoff_context(
                    specialist_key="head_on",
                    specialist_profile=None,
                    specialist_mode_name="head_on_specialist",
                )
                done = False
                while not done:
                    if episode.observation is None:
                        raise RuntimeError("V4 frozen baseline lost its source observation")
                    action = baseline.get_deterministic_action(episode.observation["observation_vector"])
                    next_observation, done, metric = episode.record_only_step(action)
                    metrics.append(metric)
                    if not done:
                        if next_observation is None:
                            raise RuntimeError("V4 unfinished episode lost its 66-D observation")
                        observation = next_observation
            ledger = episode.ledger(
                opponent=opponent,
                method="frozen_fixed_head_on",
                scenario=scenario,
                handoff=handoff,
            )
            result = record_only_result(
                opponent=opponent,
                scenario=scenario,
                handoff=handoff,
                metrics=metrics,
                ledger=ledger,
            )
            return result, ledger, {
                "scenario_application_passed": bool(receipt.get("passed")),
                "raw_si_phase_replay_passed": _phase_replay(ledger),
                "initial_pre_merge_semantics_passed": reset_phase == "pre_merge",
                "record_only_no_reward_terms": "proposed_training" not in config and all("reward" not in metric for metric in metrics),
            }
        finally:
            episode.close()


def run(config_path: Path) -> dict[str, Any]:
    config, base_path, overlay = _load_authorized_config(config_path)
    _validate_authorization(config, base_path, overlay)
    plan = build_plan(config)
    manifest_path = _repo_path(str(config["inputs"]["feasibility_manifest"]))
    _require_hash(manifest_path, config["inputs"].get("feasibility_manifest_sha256"), "V4 feasibility manifest")
    manifest = load_yaml(manifest_path)
    validate_feasibility_manifest(manifest)
    registry_path = _repo_path(str(config["inputs"]["runtime_registry"]))
    registry = load_yaml(registry_path)
    output = Path(str(config["outputs"]["root"]))
    if config["outputs"].get("creation_permitted_by_this_config") is not True:
        raise RuntimeError("V4 output creation is not authorised")
    if not output.is_absolute() or output.exists() or shutil.disk_usage(output.parent).free / (1024**3) < plan.min_free_disk_gb:
        raise RuntimeError("V4 output root failed fresh-output or disk gate")
    output.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    try:
        _write_json(output / "authorization_preflight.json", {
            "source_id": SOURCE_ID,
            "git_sha": _git_value("rev-parse", "HEAD"),
            "config_sha256": sha256_file(config_path),
            "base_config_sha256": sha256_file(base_path),
            "execution_permitted": True,
            "training_reward": "prohibited",
            "environment_reward_recording": "prohibited",
        })
        _write_json(output / "resolved_config.json", config)
        for opponent in plan.opponents:
            for scenario in manifest["scenarios"]:
                result, ledger, runtime_evidence = _run_episode(config, registry, scenario, opponent)
                ledger = copy.deepcopy(ledger)
                engine_source = ledger.get("source_id")
                ledger["source_id"] = SOURCE_ID
                ledger["engine_provenance"] = {
                    "reused_engine_module": "thesis_neutral_postmerge_reentry_recovery_pilot.runin_only",
                    "engine_source_id": engine_source,
                    "post_handoff_engine": "thesis_neutral_postmerge_v4_record_only",
                    "training_reward": "prohibited",
                    "environment_reward_recording": "prohibited",
                }
                record = runtime_record(
                    source_id=SOURCE_ID,
                    opponent=opponent,
                    scenario=scenario,
                    result=result,
                    ledger=ledger,
                    runtime_evidence=runtime_evidence,
                )
                records.append(record)
                _write_json(output / "episodes" / opponent / f"{scenario['name']}.json", {"record": record, "ledger": ledger})
        gate = evaluate_family_coverage(records, plan, runtime_identity_index(manifest))
        gate["training_unlocked"] = False
        _write_json(output / "runtime_feasibility_gate.json", gate)
        _write_json(output / "run_manifest.json", {
            "source_id": SOURCE_ID,
            "paper_safe": False,
            "mode": "nonlearning_reward_unobserved_runtime_feasibility",
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
            "training_reward": "prohibited",
            "environment_reward_recording": "prohibited",
            "error": repr(error),
        })
        raise


def authorized_preflight(config_path: Path) -> dict[str, Any]:
    """Validate a one-time overlay without instantiating JSBSim or creating output."""

    config, base_path, overlay = _load_authorized_config(config_path)
    _validate_authorization(config, base_path, overlay)
    plan = build_plan(config)
    manifest_path = _repo_path(str(config["inputs"]["feasibility_manifest"]))
    _require_hash(manifest_path, config["inputs"].get("feasibility_manifest_sha256"), "V4 feasibility manifest")
    manifest = load_yaml(manifest_path)
    validate_feasibility_manifest(manifest)
    output = Path(str(config["outputs"]["root"]))
    if config["outputs"].get("creation_permitted_by_this_config") is not True:
        raise RuntimeError("V4 output creation is not authorised")
    if not output.is_absolute() or output.exists() or shutil.disk_usage(output.parent).free / (1024**3) < plan.min_free_disk_gb:
        raise RuntimeError("V4 output root failed fresh-output or disk gate")
    return {
        "source_id": SOURCE_ID,
        "mode": "authorized_preflight_no_jsbsim_no_output_creation",
        "execution_permitted": True,
        "training_permitted": False,
        "training_reward": "prohibited",
        "environment_reward_recording": "prohibited",
        "git_sha": _git_value("rev-parse", "HEAD"),
        "scenario_count": len(manifest["scenarios"]),
        "planned_records": len(manifest["scenarios"]) * len(plan.opponents),
        "output_root": str(output),
        "output_root_absent": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        if args.preflight and args.execute:
            raise RuntimeError("choose either --preflight or --execute")
        if args.preflight:
            result = validate_design(args.config) if args.config == DEFAULT_CONFIG else authorized_preflight(args.config)
        elif args.execute:
            result = run(args.config)
        else:
            result = validate_design(args.config)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"V4 runtime-feasibility runner refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
