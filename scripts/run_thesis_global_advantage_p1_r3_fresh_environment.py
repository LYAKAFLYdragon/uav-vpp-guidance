#!/usr/bin/env python3
"""Implement, but do not implicitly authorise, the P1 R3 fresh-process probe."""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from uav_vpp_guidance.evaluation.global_advantage_p1_r3_contract import (  # noqa: E402
    GlobalAdvantageP1R3ContractError,
    GlobalAdvantageP1R3Plan,
    build_p1_r3_plan,
    capture_runtime_envelope,
    compare_r3_repeats,
    finite_action,
    snapshot_hash,
)
from uav_vpp_guidance.evaluation.global_advantage_runin_contract import (  # noqa: E402
    base_observation,
    classify_step,
    telemetry_record,
)
from uav_vpp_guidance.training import thesis_defext_rangeext_pilot as pilot  # noqa: E402
from uav_vpp_guidance.training.thesis_shared_skill_geometry import (  # noqa: E402
    FrozenP3Encoder,
    PhaseTracker,
)


DEFAULT_CONFIG = (
    ROOT
    / "config"
    / "experiment"
    / "thesis_global_advantage_v1_p1_r3_fresh_environment.yaml"
)
_AUTHORIZATION_DELTA_PATHS = {
    "config/experiment/thesis_global_advantage_v1_p1_r3_fresh_environment.yaml",
    "reports/thesis_global_advantage_p1_r3_execution_authorization_request_20260715_zh.md",
    "reports/thesis_five_state_global_advantage_goal_checklist_v1_20260714_zh.md",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise GlobalAdvantageP1R3ContractError(f"YAML root must be a mapping: {path}")
    return data


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


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


def _variant_order(
    scenarios: Sequence[Mapping[str, Any]], variant: str
) -> list[Mapping[str, Any]]:
    if variant == "forward":
        return list(scenarios)
    if variant == "reverse":
        return list(reversed(scenarios))
    if variant != "mirror_interleaved":
        raise GlobalAdvantageP1R3ContractError(f"unknown order variant: {variant}")
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for scenario in scenarios:
        metadata = scenario.get("metadata", {})
        key = str(metadata.get("mirror_pair_id") or scenario.get("name"))
        groups.setdefault(key, []).append(scenario)
    ordered: list[Mapping[str, Any]] = []
    for index, key in enumerate(sorted(groups)):
        group = sorted(
            groups[key],
            key=lambda item: str(item.get("metadata", {}).get("mirror_sign", "")),
        )
        if index % 2:
            group.reverse()
        ordered.extend(group)
    return ordered


def _history_embedding(
    history: deque[np.ndarray], encoder: FrozenP3Encoder
) -> list[float] | None:
    if len(history) != history.maxlen:
        return None
    embedding = np.asarray(encoder.encode(np.stack(tuple(history), axis=0))).reshape(-1)
    if embedding.shape != (32,) or not np.isfinite(embedding).all():
        raise GlobalAdvantageP1R3ContractError("P3 temporal embedding is malformed")
    return [float(value) for value in embedding]


def _history_payload(
    history: deque[np.ndarray], base_names: Sequence[str]
) -> list[dict[str, float]]:
    return [
        {name: float(value) for name, value in zip(base_names, frame)}
        for frame in history
    ]


def _step_state(
    observation: Mapping[str, Any],
    history: deque[np.ndarray],
    phase_tracker: PhaseTracker,
    encoder: FrozenP3Encoder,
) -> tuple[dict[str, float], str, str, list[float] | None]:
    base = base_observation(observation)
    history.append(np.asarray([base[name] for name in base], dtype=np.float32))
    phase = phase_tracker.update(base["range_m"], base["range_rate_mps"])
    dynamic_state = classify_step(base, phase)
    return base, phase, dynamic_state, _history_embedding(history, encoder)


def _episode_path(root: Path, launch: Mapping[str, Any]) -> Path:
    scenario = str(launch["scenario"]["name"]).replace("/", "_").replace("\\", "_")
    return (
        root
        / "episodes"
        / str(launch["opponent"])
        / f"repeat_{int(launch['repeat_index'])}_{launch['order_variant']}"
        / f"{scenario}.json"
    )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".tmp", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)


def _validate_sources(
    config_path: Path,
) -> tuple[
    GlobalAdvantageP1R3Plan,
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    Path,
]:
    config = _load_yaml(config_path)
    plan = build_p1_r3_plan(config)
    protocol = config["global_advantage_p1_r3"]
    if plan.execution_permitted:
        if config_path.resolve() != DEFAULT_CONFIG.resolve():
            raise GlobalAdvantageP1R3ContractError(
                "R3 execution requires the canonical in-repository config"
            )
        required_sha = protocol["authorization"].get("required_implementation_git_sha")
        if (
            not isinstance(required_sha, str)
            or len(required_sha) != 40
            or not _git_is_ancestor(required_sha)
            or _git_value("status", "--porcelain") != ""
        ):
            raise GlobalAdvantageP1R3ContractError(
                "R3 execution requires a clean worktree and frozen implementation ancestor"
            )
        unexpected_changes = _git_changed_paths_since(required_sha) - _AUTHORIZATION_DELTA_PATHS
        if unexpected_changes:
            raise GlobalAdvantageP1R3ContractError(
                "R3 execution contains non-authorization changes after the "
                f"implementation freeze: {sorted(unexpected_changes)}"
            )
        authorized_files = protocol["authorization"].get("authorized_code_files")
        if not isinstance(authorized_files, Sequence) or not authorized_files:
            raise GlobalAdvantageP1R3ContractError(
                "R3 execution requires explicit authorized code-file hashes"
            )
        for entry in authorized_files:
            if not isinstance(entry, Mapping):
                raise GlobalAdvantageP1R3ContractError(
                    "authorized code-file entry must be a mapping"
                )
            path = _repo_path(str(entry.get("path", ""))).resolve()
            try:
                path.relative_to(ROOT.resolve())
            except ValueError as error:
                raise GlobalAdvantageP1R3ContractError(
                    f"authorized code path escapes the repository: {path}"
                ) from error
            expected_hash = entry.get("sha256")
            if (
                not isinstance(expected_hash, str)
                or not path.is_file()
                or _sha256(path) != expected_hash.lower()
            ):
                raise GlobalAdvantageP1R3ContractError(
                    f"authorized code-file SHA mismatch: {path}"
                )
    source = protocol["sources"]
    runtime_config_path = _repo_path(str(source["runtime_config"]))
    registry_path = _repo_path(str(source["runtime_registry"]))
    manifest_path = _repo_path(str(source["scenario_manifest"]))
    expected_hashes = (
        (runtime_config_path, "runtime_config_sha256"),
        (registry_path, "runtime_registry_sha256"),
        (manifest_path, "scenario_manifest_sha256"),
    )
    for path, key in expected_hashes:
        if not path.is_file() or _sha256(path) != str(source[key]).lower():
            raise GlobalAdvantageP1R3ContractError(f"R3 source hash mismatch: {key}")
    runtime_config = _load_yaml(runtime_config_path)
    registry = _load_yaml(registry_path)
    for opponent in plan.opponents:
        entry = registry.get("opponents", {}).get(opponent)
        if not isinstance(entry, Mapping):
            raise GlobalAdvantageP1R3ContractError(
                f"R3 opponent registry entry is missing: {opponent}"
            )
        checkpoint = entry.get("checkpoint")
        if checkpoint is None:
            if opponent != "expert":
                raise GlobalAdvantageP1R3ContractError(
                    f"R3 opponent checkpoint is missing: {opponent}"
                )
            continue
        expected_checkpoint_hash = entry.get("checkpoint_sha256")
        checkpoint_path = Path(str(checkpoint))
        if (
            not isinstance(expected_checkpoint_hash, str)
            or not checkpoint_path.is_file()
            or _sha256(checkpoint_path) != expected_checkpoint_hash.lower()
        ):
            raise GlobalAdvantageP1R3ContractError(
                f"R3 opponent checkpoint hash mismatch: {opponent}"
            )
    manifest = _load_yaml(manifest_path)
    frozen_p3 = runtime_config["fixed_contract"]["encoder"]
    p3_path = Path(str(frozen_p3["checkpoint"]))
    if (
        p3_path != Path(str(source["frozen_p3_checkpoint"]))
        or str(frozen_p3["sha256"]).lower()
        != str(source["frozen_p3_checkpoint_sha256"]).lower()
        or not p3_path.is_file()
        or _sha256(p3_path) != str(source["frozen_p3_checkpoint_sha256"]).lower()
    ):
        raise GlobalAdvantageP1R3ContractError("R3 frozen P3 checkpoint contract drifted")
    scenarios = list(manifest.get("scenarios") or [])
    if len(scenarios) != plan.scenario_count:
        raise GlobalAdvantageP1R3ContractError("R3 scenario count drifted")
    if manifest.get("source_id") != source["scenario_manifest_source_id"]:
        raise GlobalAdvantageP1R3ContractError("R3 scenario manifest source ID drifted")
    return plan, runtime_config, registry, scenarios, Path(protocol["outputs"]["root"])


def _telemetry_complete(steps: Sequence[Mapping[str, Any]]) -> bool:
    required = (
        "reference_action",
        "opponent_action",
        "runtime_envelope_sha256",
        "normalized_vpp_action",
        "vp_forward_bias_m",
        "vp_lateral_bias_m",
        "vpp_vertical_component",
        "nz_cmd",
        "actual_nz_g",
        "ego_attack_aoa_deg",
        "own_speed_mps",
        "own_altitude_m",
    )
    return bool(steps) and all(
        all(step.get(field) is not None for field in required) for step in steps
    )


def _fallback_free(steps: Sequence[Mapping[str, Any]]) -> bool:
    return all(
        step.get("backend") == "jsbsim"
        and not bool(step.get("backend_fallback_occurred", False))
        and not (
            bool(step.get("prediction_fallback", False))
            and step.get("prediction_fallback_phase") != "warmup"
        )
        for step in steps
    )


def _run_child_episode(
    *,
    env: Any,
    reference: Any,
    encoder: FrozenP3Encoder,
    launch: Mapping[str, Any],
    plan: GlobalAdvantageP1R3Plan,
) -> dict[str, Any]:
    scenario = dict(launch["scenario"])
    metadata = dict(scenario.get("metadata") or {})
    seed = int(metadata["scenario_seed"])
    observation = env.reset(scenario=scenario, seed=seed)
    if getattr(env, "_backend", None) != "jsbsim":
        raise GlobalAdvantageP1R3ContractError("R3 backend is not JSBSim")
    env.set_runtime_specialist_context(
        specialist_key="head_on",
        specialist_profile=None,
        specialist_mode_name="run_in_head_on_specialist",
        specialist_reason="global_advantage_p1_r3_reference_only",
    )
    history: deque[np.ndarray] = deque(maxlen=plan.history_window_steps)
    phase_tracker = PhaseTracker()
    base, phase, dynamic_state, embedding = _step_state(
        observation, history, phase_tracker, encoder
    )
    history_payload = _history_payload(history, tuple(base))
    reference_metadata = reference.checkpoint_metadata()
    reset_envelope = capture_runtime_envelope(
        env=env,
        observation=observation,
        history=history_payload,
        phase_tracker=phase_tracker,
        reference_metadata=reference_metadata,
    )
    header = {
        "source_id": plan.source_id,
        "opponent": str(launch["opponent"]),
        "scenario_name": str(scenario["name"]),
        "scenario_signature": snapshot_hash(scenario, plan.hash_decimal_places),
        "scenario_metadata": metadata,
        "scenario_seed": seed,
        "repeat_index": int(launch["repeat_index"]),
        "order_variant": str(launch["order_variant"]),
        "process_id": os.getpid(),
    }
    reset = {
        "base_observation": base,
        "phase": phase,
        "dynamic_state": dynamic_state,
        "temporal_embedding": embedding,
        "runtime_envelope": reset_envelope,
        "runtime_envelope_sha256": snapshot_hash(
            reset_envelope, plan.hash_decimal_places
        ),
    }
    steps: list[dict[str, Any]] = []
    terminal_reason = "horizon"
    for step in range(1, plan.max_high_level_steps + 1):
        reference_action = finite_action(
            reference.get_deterministic_action(observation["observation_vector"]),
            name="reference_action",
        )
        next_observation, _reward, terminated, truncated, info = env.step(reference_action)
        opponent_action = finite_action(info.get("opponent_action"), name="opponent_action")
        base, phase, dynamic_state, embedding = _step_state(
            next_observation, history, phase_tracker, encoder
        )
        history_payload = _history_payload(history, tuple(base))
        envelope = capture_runtime_envelope(
            env=env,
            observation=next_observation,
            history=history_payload,
            phase_tracker=phase_tracker,
            reference_metadata=reference_metadata,
        )
        record = telemetry_record(
            step=step,
            base=base,
            phase=phase,
            dynamic_state=dynamic_state,
            history=history_payload,
            temporal_embedding=embedding,
            action=reference_action,
            info=info,
            decimal_places=plan.hash_decimal_places,
        )
        record["reference_action"] = reference_action
        record["opponent_action"] = opponent_action
        record["reference_action_sha256"] = snapshot_hash(
            {"action": reference_action}, plan.hash_decimal_places
        )
        record["opponent_action_sha256"] = snapshot_hash(
            {"action": opponent_action}, plan.hash_decimal_places
        )
        record["runtime_envelope_sha256"] = snapshot_hash(
            envelope, plan.hash_decimal_places
        )
        record["step_sha256"] = snapshot_hash(
            {key: value for key, value in record.items() if key != "step_sha256"},
            plan.hash_decimal_places,
        )
        steps.append(record)
        observation = next_observation
        if terminated or truncated:
            terminal_reason = str(info.get("reason") or "terminated")
            break
    if not steps:
        raise GlobalAdvantageP1R3ContractError("R3 episode has no recorded step")
    boundary = next(
        (record for record in steps if bool(record.get("first_pass_complete", False))),
        steps[-1],
    )
    summary = {
        "source_id": plan.source_id,
        "opponent": header["opponent"],
        "scenario_signature": header["scenario_signature"],
        "scenario_name": header["scenario_name"],
        "repeat_index": header["repeat_index"],
        "order_variant": header["order_variant"],
        "process_id": header["process_id"],
        "reset_envelope_sha256": reset["runtime_envelope_sha256"],
        "first_reference_action_sha256": steps[0]["reference_action_sha256"],
        "first_opponent_action_sha256": steps[0]["opponent_action_sha256"],
        "trajectory_sha256": snapshot_hash(
            {"step_hashes": [record["step_sha256"] for record in steps]},
            plan.hash_decimal_places,
        ),
        "boundary_envelope_sha256": boundary["runtime_envelope_sha256"],
        "boundary_step": int(boundary["step"]),
        "terminal_reason": terminal_reason,
        "step_count": len(steps),
        "telemetry_complete": _telemetry_complete(steps),
        "no_backend_or_prediction_fallback": _fallback_free(steps),
    }
    return {"header": header, "reset": reset, "steps": steps, "summary": summary}


def _run_child(config_path: Path, launch_path: Path) -> dict[str, Any]:
    plan, runtime_config, registry, scenarios, output_root = _validate_sources(config_path)
    if not plan.execution_permitted:
        raise GlobalAdvantageP1R3ContractError("R3 execution is not authorised")
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    if launch.get("source_id") != plan.source_id:
        raise GlobalAdvantageP1R3ContractError("child launch source ID mismatch")
    scenario = launch.get("scenario")
    if not isinstance(scenario, Mapping):
        raise GlobalAdvantageP1R3ContractError("child launch is missing scenario")
    signatures = {
        snapshot_hash(item, plan.hash_decimal_places): item for item in scenarios
    }
    signature = snapshot_hash(scenario, plan.hash_decimal_places)
    if signature not in signatures or dict(signatures[signature]) != dict(scenario):
        raise GlobalAdvantageP1R3ContractError("child scenario is not in the frozen manifest")
    opponent = str(launch.get("opponent"))
    if opponent not in plan.opponents:
        raise GlobalAdvantageP1R3ContractError("child launch opponent is invalid")
    expected_path = _episode_path(output_root, launch)
    if expected_path.exists():
        raise GlobalAdvantageP1R3ContractError("child output artifact already exists")
    fixed = runtime_config["fixed_contract"]["encoder"]
    encoder = FrozenP3Encoder(Path(fixed["checkpoint"]), str(fixed["sha256"]))
    env = pilot._build_env(runtime_config, registry, opponent)
    reference = pilot._specialist(registry, "run_in_head_on")
    try:
        episode = _run_child_episode(
            env=env,
            reference=reference,
            encoder=encoder,
            launch=launch,
            plan=plan,
        )
        _write_json_atomic(expected_path, episode)
    finally:
        env.close()
    return {
        "artifact_path": str(expected_path),
        "artifact_sha256": _sha256(expected_path),
        "process_id": episode["header"]["process_id"],
    }


def _provenance(config_path: Path, protocol: Mapping[str, Any]) -> Mapping[str, Any]:
    source = protocol["sources"]
    return {
        "git_sha": _git_value("rev-parse", "HEAD"),
        "git_dirty": _git_value("status", "--porcelain") != "",
        "python": sys.version,
        "platform": platform.platform(),
        "config_sha256": _sha256(config_path),
        "runtime_config_sha256": _sha256(_repo_path(source["runtime_config"])),
        "runtime_registry_sha256": _sha256(_repo_path(source["runtime_registry"])),
        "scenario_manifest_sha256": _sha256(_repo_path(source["scenario_manifest"])),
    }


def run(config_path: Path) -> Mapping[str, Any]:
    plan, _runtime_config, _registry, scenarios, output_root = _validate_sources(config_path)
    if not plan.execution_permitted:
        raise GlobalAdvantageP1R3ContractError("R3 execution is not authorised")
    if output_root.exists():
        raise GlobalAdvantageP1R3ContractError("R3 output root must be fresh and absent")
    free_gb = shutil.disk_usage(output_root.parent).free / (1024**3)
    if free_gb < plan.min_free_disk_gb:
        raise GlobalAdvantageP1R3ContractError(
            f"R3 disk gate failed: {free_gb:.1f} GB < {plan.min_free_disk_gb:.1f} GB"
        )
    output_root.mkdir(parents=True, exist_ok=False)
    config = _load_yaml(config_path)
    protocol = config["global_advantage_p1_r3"]
    summaries: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    try:
        for repeat_index, variant in enumerate(plan.order_variants):
            for opponent in plan.opponents:
                for scenario in _variant_order(scenarios, variant):
                    launch = {
                        "source_id": plan.source_id,
                        "opponent": opponent,
                        "scenario": dict(scenario),
                        "repeat_index": repeat_index,
                        "order_variant": variant,
                    }
                    launch_name = (
                        f"{opponent}_r{repeat_index}_{scenario['name']}.json".replace("/", "_")
                    )
                    launch_path = output_root / "launches" / launch_name
                    _write_json_atomic(launch_path, launch)
                    try:
                        child = subprocess.run(
                            [
                                sys.executable,
                                str(Path(__file__).resolve()),
                                "--config",
                                str(config_path),
                                "--child-launch",
                                str(launch_path),
                            ],
                            cwd=ROOT,
                            text=True,
                            capture_output=True,
                            check=False,
                            timeout=plan.child_timeout_seconds,
                        )
                    except subprocess.TimeoutExpired as error:
                        raise GlobalAdvantageP1R3ContractError(
                            f"R3 child timed out after {plan.child_timeout_seconds}s for "
                            f"{opponent}/{scenario['name']}"
                        ) from error
                    if child.returncode != 0:
                        raise GlobalAdvantageP1R3ContractError(
                            f"R3 child failed for {opponent}/{scenario['name']}: {child.stderr.strip()}"
                        )
                    artifact_path = _episode_path(output_root, launch)
                    raw = json.loads(artifact_path.read_text(encoding="utf-8"))
                    summary = dict(raw["summary"])
                    if int(summary["process_id"]) == os.getpid():
                        raise GlobalAdvantageP1R3ContractError("R3 child reused the parent process")
                    summary["artifact_path"] = str(artifact_path)
                    summary["artifact_sha256"] = _sha256(artifact_path)
                    summaries.append(summary)
                    artifacts.append(
                        {
                            "path": str(artifact_path),
                            "sha256": summary["artifact_sha256"],
                            "size_bytes": artifact_path.stat().st_size,
                        }
                    )
        gate = compare_r3_repeats(summaries)
        provenance = _provenance(config_path, protocol)
        _write_json_atomic(output_root / "p1_r3_gate_summary.json", gate)
        _write_json_atomic(
            output_root / "run_manifest.json",
            {
                "source_id": plan.source_id,
                "mode": "non_learning_fresh_process_runin_reproducibility_preflight",
                "training_permitted": False,
                "heldout_evaluation_permitted": False,
                "episodes": summaries,
                "artifacts": artifacts,
                "gate": gate,
                "provenance": provenance,
            },
        )
        return {"output_root": str(output_root), "gate": gate, "provenance": provenance}
    except Exception:
        _write_json_atomic(
            output_root / "runner_failure_manifest.json",
            {
                "source_id": plan.source_id,
                "status": "runner_exception_output_preserved_for_audit",
                "completed_episode_count": len(summaries),
            },
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--child-launch", type=Path)
    args = parser.parse_args()
    try:
        plan, _runtime, _registry, scenarios, output = _validate_sources(args.config)
        if args.child_launch is not None:
            result = _run_child(args.config, args.child_launch)
            print(json.dumps(result, sort_keys=True))
            return 0
        if not args.execute:
            print(
                json.dumps(
                    {
                        "source_id": plan.source_id,
                        "mode": "validated_not_executed",
                        "execution_permitted": plan.execution_permitted,
                        "training_permitted": False,
                        "scenario_count": len(scenarios),
                        "episode_count": len(scenarios)
                        * len(plan.opponents)
                        * len(plan.order_variants),
                        "output_root": str(output),
                    },
                    indent=2,
                )
            )
            return 0
        result = run(args.config)
        print(json.dumps(result, indent=2))
        return 0 if result["gate"]["passed"] else 3
    except (OSError, ValueError, yaml.YAMLError, GlobalAdvantageP1R3ContractError) as error:
        print(f"P1 R3 preflight failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
