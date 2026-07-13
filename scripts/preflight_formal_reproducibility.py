#!/usr/bin/env python3
"""Preflight reproducibility checks for formal held-out comparison runs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import yaml


CORE_AGENT_DEPENDENCIES = {
    "hierarchical_commander": [
        "scripts/run_jsbsim_hrl_comparison.py",
        "src/uav_vpp_guidance/evaluation/hierarchical_commander_policy.py",
        "src/uav_vpp_guidance/evaluation/recorders.py",
        "src/uav_vpp_guidance/agents/commander_ppo_agent.py",
        "src/uav_vpp_guidance/agents/commander_double_dqn_agent.py",
        "src/uav_vpp_guidance/agents/policy_network.py",
        "src/uav_vpp_guidance/agents/replay_buffer.py",
        "src/uav_vpp_guidance/envs/hierarchical_commander_env.py",
        "src/uav_vpp_guidance/envs/tracking_env.py",
        "src/uav_vpp_guidance/hierarchy/specialist_policy.py",
        "src/uav_vpp_guidance/hierarchy/commander_mode_constraints.py",
    ],
    "oracle_task_gate": [
        "src/uav_vpp_guidance/evaluation/oracle_task_gate_policy.py",
    ],
    "random_task_gate": [
        "src/uav_vpp_guidance/evaluation/random_task_gate_policy.py",
    ],
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _resolve_include_path(config_path: Path, include_value: str) -> Path:
    candidate = (config_path.parent / include_value).resolve()
    if candidate.exists():
        return candidate
    fallback = (config_path.parent / ".." / Path(include_value).name).resolve()
    return fallback


def _load_yaml_with_dependencies(
    config_path: Path,
    *,
    yaml_dependencies: Set[Path],
) -> Dict[str, Any]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    yaml_dependencies.add(config_path.resolve())
    includes = payload.pop("includes", []) or []
    merged: Dict[str, Any] = {}
    for include_value in includes:
        include_path = _resolve_include_path(config_path, str(include_value))
        if not include_path.exists():
            raise FileNotFoundError(
                f"Included config not found: {include_value} (from {config_path})"
            )
        merged = _deep_merge(
            merged,
            _load_yaml_with_dependencies(
                include_path,
                yaml_dependencies=yaml_dependencies,
            ),
        )
    return _deep_merge(merged, payload)


def _git_is_tracked(repo_root: Path, target: Path) -> bool:
    try:
        relative = target.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return False
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(relative)],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _checkpoint_dimension_info(checkpoint_path: Path) -> Dict[str, Any]:
    if not checkpoint_path.exists():
        return {
            "checkpoint_obs_dim": None,
            "checkpoint_action_dim": None,
            "checkpoint_load_error": None,
        }
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    except Exception as exc:  # pragma: no cover - defensive guard
        return {
            "checkpoint_obs_dim": None,
            "checkpoint_action_dim": None,
            "checkpoint_load_error": str(exc),
        }
    return {
        "checkpoint_obs_dim": checkpoint.get("obs_dim"),
        "checkpoint_action_dim": checkpoint.get("action_dim"),
        "checkpoint_load_error": None,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_manifest_sha256(path: Path) -> str:
    """Hash a directory from sorted relative paths and per-file SHA-256 values."""

    digest = hashlib.sha256()
    for candidate in sorted(item for item in path.rglob("*") if item.is_file()):
        relative_path = candidate.relative_to(path).as_posix()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(candidate).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _collect_frozen_assets(
    *,
    repo_root: Path,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Collect immutable external assets declared by an evaluation config."""

    contract = config.get("frozen_assets", {}) or {}
    source_worktree_value = contract.get("source_worktree")
    source_worktree = Path(str(source_worktree_value)) if source_worktree_value else None
    source_git_sha = contract.get("source_git_sha")
    source_git_actual = None
    source_git_error = None
    if source_worktree is not None and source_worktree.exists():
        try:
            source_git_actual = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=source_worktree, text=True
            ).strip()
        except Exception as exc:  # pragma: no cover - defensive guard
            source_git_error = str(exc)

    artifacts: List[Dict[str, Any]] = []
    for declared in contract.get("artifacts", []) or []:
        if not isinstance(declared, dict):
            raise TypeError("frozen_assets.artifacts entries must be mappings")
        path_value = declared.get("path")
        resolved_path = _resolve_repo_relative_path(repo_root, path_value)
        if resolved_path is None:
            raise ValueError("frozen asset is missing path")
        kind = str(declared.get("kind", "file"))
        if kind not in {"file", "directory"}:
            raise ValueError(f"Unsupported frozen asset kind: {kind}")
        exists = resolved_path.exists()
        dimensions = _checkpoint_dimension_info(resolved_path) if exists and kind == "file" else {
            "checkpoint_obs_dim": None,
            "checkpoint_action_dim": None,
            "checkpoint_load_error": None,
        }
        artifacts.append(
            {
                "id": declared.get("id", str(path_value)),
                "kind": kind,
                "path": str(resolved_path),
                "exists": exists,
                "expected_sha256": declared.get("sha256"),
                "actual_sha256": (
                    _directory_manifest_sha256(resolved_path)
                    if exists and kind == "directory"
                    else _sha256_file(resolved_path)
                    if exists
                    else None
                ),
                "expected_obs_dim": declared.get("expected_obs_dim"),
                "expected_action_dim": declared.get("expected_action_dim"),
                **dimensions,
            }
        )

    return {
        "source_worktree": str(source_worktree) if source_worktree else None,
        "source_worktree_exists": bool(source_worktree and source_worktree.exists()),
        "expected_source_git_sha": source_git_sha,
        "actual_source_git_sha": source_git_actual,
        "source_git_error": source_git_error,
        "artifacts": artifacts,
    }


def _collect_scenario_manifest(
    *,
    config_path: Path,
    config: Dict[str, Any],
    yaml_dependencies: Set[Path],
) -> Dict[str, Any]:
    """Validate and register the immutable explicit-scenario manifest, if used."""

    spec = config.get("scenario_manifest")
    if not isinstance(spec, dict):
        return {"configured": False}
    raw_path = spec.get("path")
    if not raw_path:
        return {"configured": True, "path": None, "error": "missing scenario_manifest.path"}
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    if not path.exists():
        return {"configured": True, "path": str(path), "exists": False}

    yaml_dependencies.add(path)
    manifest = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    integrity = copy.deepcopy(manifest.get("integrity", {}))
    recorded_hash = integrity.pop("payload_sha256", None)
    manifest_for_hash = copy.deepcopy(manifest)
    manifest_for_hash["integrity"] = integrity
    actual_hash = hashlib.sha256(
        json.dumps(manifest_for_hash, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    task_key_field = str(spec.get("task_key_field", "task_registry_key"))
    task_counts: Dict[str, int] = {}
    for scenario in manifest.get("scenarios", []) or []:
        metadata = scenario.get("metadata", {}) if isinstance(scenario, dict) else {}
        task_name = metadata.get(task_key_field) if isinstance(metadata, dict) else None
        if task_name:
            task_counts[str(task_name)] = task_counts.get(str(task_name), 0) + 1
    return {
        "configured": True,
        "path": str(path),
        "exists": True,
        "source_id": manifest.get("source_id"),
        "expected_source_id": spec.get("source_id"),
        "recorded_payload_sha256": recorded_hash,
        "expected_payload_sha256": spec.get("payload_sha256"),
        "actual_payload_sha256": actual_hash,
        "scenario_count": len(manifest.get("scenarios", []) or []),
        "expected_task_counts": spec.get("expected_task_counts", {}),
        "actual_task_counts": task_counts,
    }


def _resolve_repo_relative_path(repo_root: Path, path_value: Optional[str]) -> Optional[Path]:
    if not path_value:
        return None
    path = Path(str(path_value))
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def _expected_action_dim(agent_type: str, config: Dict[str, Any]) -> Optional[int]:
    if agent_type == "hierarchical_commander":
        commander_cfg = config.get("commander", {})
        configured_num_modes = commander_cfg.get("num_modes")
        if configured_num_modes is not None:
            return int(configured_num_modes)
        modes = commander_cfg.get("modes", [])
        if isinstance(modes, list) and modes:
            return int(len(modes))
        return None
    policy_cfg = config.get("policy", {})
    action_dim = policy_cfg.get("action_dim")
    return int(action_dim) if action_dim is not None else None


def _checkpoint_action_dim_is_compatible(
    *,
    agent_type: str,
    expected_action_dim: Optional[int],
    checkpoint_action_dim: Optional[int],
) -> bool:
    if expected_action_dim is None or checkpoint_action_dim is None:
        return True
    if agent_type == "hierarchical_commander":
        return int(checkpoint_action_dim) <= int(expected_action_dim)
    return int(expected_action_dim) == int(checkpoint_action_dim)


def _add_runtime_artifact(
    runtime_artifacts: Dict[Tuple[str, str], Dict[str, Any]],
    *,
    repo_root: Path,
    owner: str,
    kind: str,
    path_value: Optional[str],
    tracked_required: bool = False,
    expected_action_dim: Optional[int] = None,
) -> Optional[Path]:
    checkpoint_kinds = {
        "checkpoint",
        "oracle_specialist_checkpoint",
        "commander_mode_checkpoint",
    }
    resolved_path = _resolve_repo_relative_path(repo_root, path_value)
    if resolved_path is None:
        return None

    key = (kind, str(resolved_path))
    artifact = runtime_artifacts.get(key)
    if artifact is None:
        artifact = {
            "kind": kind,
            "path": str(resolved_path),
            "exists": resolved_path.exists(),
            "tracked_required": tracked_required,
            "owners": [owner],
        }
        if kind in checkpoint_kinds:
            artifact.update(_checkpoint_dimension_info(resolved_path))
        else:
            artifact.update(
                {
                    "checkpoint_obs_dim": None,
                    "checkpoint_action_dim": None,
                    "checkpoint_load_error": None,
                }
            )
        runtime_artifacts[key] = artifact
    else:
        artifact["tracked_required"] = bool(
            artifact.get("tracked_required", False) or tracked_required
        )
        owners = list(artifact.get("owners", []))
        if owner not in owners:
            owners.append(owner)
            artifact["owners"] = owners

    if expected_action_dim is not None:
        artifact["expected_action_dim"] = expected_action_dim
    return resolved_path


def _collect_predictor_runtime_artifacts(
    runtime_artifacts: Dict[Tuple[str, str], Dict[str, Any]],
    *,
    repo_root: Path,
    owner: str,
    config: Dict[str, Any],
) -> None:
    trajectory_prediction = config.get("trajectory_prediction", {})
    if not isinstance(trajectory_prediction, dict):
        return
    if not trajectory_prediction.get("enabled", False):
        return
    checkpoint_path = trajectory_prediction.get("checkpoint_path")
    if checkpoint_path:
        _add_runtime_artifact(
            runtime_artifacts,
            repo_root=repo_root,
            owner=owner,
            kind="trajectory_prediction_checkpoint",
            path_value=str(checkpoint_path),
            tracked_required=False,
        )


def collect_formal_dependencies(
    *,
    repo_root: Path,
    config_path: Path,
) -> Dict[str, Any]:
    yaml_dependencies: Set[Path] = set()
    resolved_root = _load_yaml_with_dependencies(
        config_path,
        yaml_dependencies=yaml_dependencies,
    )
    runtime_artifacts: Dict[Tuple[str, str], Dict[str, Any]] = {}
    _collect_predictor_runtime_artifacts(
        runtime_artifacts,
        repo_root=repo_root,
        owner="comparison_config",
        config=resolved_root,
    )
    scenario_manifest = _collect_scenario_manifest(
        config_path=config_path,
        config=resolved_root,
        yaml_dependencies=yaml_dependencies,
    )
    frozen_assets = _collect_frozen_assets(repo_root=repo_root, config=resolved_root)

    methods = resolved_root.get("methods", {})
    selected_methods = list(
        resolved_root.get("run_defaults", {}).get("main_methods", methods.keys())
    )
    method_reports: List[Dict[str, Any]] = []
    code_dependencies: Set[Path] = set()

    for method_name in selected_methods:
        method_def = methods.get(method_name, {})
        agent_type = str(method_def.get("agent_type", "ppo"))
        method_cfg_path = method_def.get("config_path")
        resolved_method_cfg: Dict[str, Any] = {}
        resolved_method_cfg_path: Optional[Path] = None
        if method_cfg_path:
            resolved_method_cfg_path = (repo_root / str(method_cfg_path)).resolve()
            if resolved_method_cfg_path.exists():
                resolved_method_cfg = _load_yaml_with_dependencies(
                    resolved_method_cfg_path,
                    yaml_dependencies=yaml_dependencies,
                )
                _collect_predictor_runtime_artifacts(
                    runtime_artifacts,
                    repo_root=repo_root,
                    owner=f"{method_name}.config",
                    config=resolved_method_cfg,
                )
        for rel_path in CORE_AGENT_DEPENDENCIES.get(agent_type, []):
            code_dependencies.add((repo_root / rel_path).resolve())

        checkpoint_info: Dict[str, Any] = {"exists": False}
        checkpoint_path = method_def.get("checkpoint")
        if checkpoint_path:
            resolved_checkpoint_path = (repo_root / str(checkpoint_path)).resolve()
            checkpoint_info = {
                "path": str(resolved_checkpoint_path),
                "exists": resolved_checkpoint_path.exists(),
            }
            if resolved_checkpoint_path.exists():
                checkpoint_info.update(_checkpoint_dimension_info(resolved_checkpoint_path))

        expected_action_dim = _expected_action_dim(agent_type, resolved_method_cfg)

        if agent_type == "oracle_task_gate":
            specialists = method_def.get("specialists", {}) or {}
            for specialist_key, specialist_cfg in specialists.items():
                owner = f"{method_name}.specialists.{specialist_key}"
                specialist_config_path = _add_runtime_artifact(
                    runtime_artifacts,
                    repo_root=repo_root,
                    owner=owner,
                    kind="oracle_specialist_config",
                    path_value=specialist_cfg.get("config_path"),
                    tracked_required=True,
                )
                if specialist_config_path is not None and specialist_config_path.exists():
                    specialist_config = _load_yaml_with_dependencies(
                        specialist_config_path,
                        yaml_dependencies=yaml_dependencies,
                    )
                    _collect_predictor_runtime_artifacts(
                        runtime_artifacts,
                        repo_root=repo_root,
                        owner=owner,
                        config=specialist_config,
                    )
                _add_runtime_artifact(
                    runtime_artifacts,
                    repo_root=repo_root,
                    owner=owner,
                    kind="oracle_specialist_checkpoint",
                    path_value=specialist_cfg.get("checkpoint"),
                    tracked_required=False,
                    expected_action_dim=_expected_action_dim("ppo", {}),
                )
        elif agent_type == "hierarchical_commander":
            commander_cfg = resolved_method_cfg.get("commander", {})
            for mode_cfg in list(commander_cfg.get("modes", [])):
                mode_id = int(mode_cfg.get("id", -1))
                mode_name = str(mode_cfg.get("name", f"mode_{mode_id}"))
                owner = f"{method_name}.commander.mode[{mode_id}:{mode_name}]"
                mode_config_path = _add_runtime_artifact(
                    runtime_artifacts,
                    repo_root=repo_root,
                    owner=owner,
                    kind="commander_mode_config",
                    path_value=mode_cfg.get("config_path"),
                    tracked_required=True,
                )
                if mode_config_path is not None and mode_config_path.exists():
                    mode_config = _load_yaml_with_dependencies(
                        mode_config_path,
                        yaml_dependencies=yaml_dependencies,
                    )
                    _collect_predictor_runtime_artifacts(
                        runtime_artifacts,
                        repo_root=repo_root,
                        owner=owner,
                        config=mode_config,
                    )
                _add_runtime_artifact(
                    runtime_artifacts,
                    repo_root=repo_root,
                    owner=owner,
                    kind="commander_mode_checkpoint",
                    path_value=mode_cfg.get("checkpoint"),
                    tracked_required=False,
                    expected_action_dim=_expected_action_dim("ppo", {}),
                )

        method_reports.append(
            {
                "method": method_name,
                "agent_type": agent_type,
                "config_path": str(resolved_method_cfg_path) if resolved_method_cfg_path else None,
                "config_exists": bool(
                    resolved_method_cfg_path and resolved_method_cfg_path.exists()
                ),
                "expected_action_dim": expected_action_dim,
                "checkpoint": checkpoint_info,
            }
        )

    return {
        "resolved_root": resolved_root,
        "yaml_dependencies": sorted(str(path) for path in yaml_dependencies),
        "code_dependencies": sorted(str(path) for path in code_dependencies),
        "methods": method_reports,
        "runtime_artifacts": sorted(
            runtime_artifacts.values(),
            key=lambda item: (str(item["kind"]), str(item["path"])),
        ),
        "scenario_manifest": scenario_manifest,
        "frozen_assets": frozen_assets,
    }


def run_preflight(*, repo_root: Path, config_path: Path) -> Tuple[Dict[str, Any], List[str]]:
    report = collect_formal_dependencies(repo_root=repo_root, config_path=config_path)
    issues: List[str] = []

    for dep in report["yaml_dependencies"]:
        dep_path = Path(dep)
        if not dep_path.exists():
            issues.append(f"missing YAML dependency: {dep_path}")
            continue
        if not _git_is_tracked(repo_root, dep_path):
            issues.append(f"untracked YAML dependency: {dep_path}")

    for dep in report["code_dependencies"]:
        dep_path = Path(dep)
        if not dep_path.exists():
            issues.append(f"missing code dependency: {dep_path}")
            continue
        if not _git_is_tracked(repo_root, dep_path):
            issues.append(f"untracked code dependency: {dep_path}")

    scenario_manifest = report.get("scenario_manifest", {})
    if scenario_manifest.get("configured", False):
        if scenario_manifest.get("error"):
            issues.append(f"scenario manifest: {scenario_manifest['error']}")
        elif not scenario_manifest.get("exists", False):
            issues.append(f"scenario manifest missing: {scenario_manifest.get('path')}")
        elif (
            scenario_manifest.get("expected_source_id")
            and scenario_manifest.get("source_id")
            != scenario_manifest.get("expected_source_id")
        ):
            issues.append(
                "scenario manifest source_id mismatch: "
                f"expected={scenario_manifest.get('expected_source_id')}, "
                f"actual={scenario_manifest.get('source_id')}"
            )
        elif (
            scenario_manifest.get("recorded_payload_sha256")
            != scenario_manifest.get("actual_payload_sha256")
            or scenario_manifest.get("expected_payload_sha256")
            != scenario_manifest.get("actual_payload_sha256")
        ):
            issues.append(
                "scenario manifest payload SHA-256 mismatch: "
                f"recorded={scenario_manifest.get('recorded_payload_sha256')}, "
                f"expected={scenario_manifest.get('expected_payload_sha256')}, "
                f"actual={scenario_manifest.get('actual_payload_sha256')}"
            )
        elif scenario_manifest.get("expected_task_counts") != scenario_manifest.get(
            "actual_task_counts"
        ):
            issues.append(
                "scenario manifest task counts mismatch: "
                f"expected={scenario_manifest.get('expected_task_counts')}, "
                f"actual={scenario_manifest.get('actual_task_counts')}"
            )

    for method_report in report["methods"]:
        if method_report["config_path"] and not method_report["config_exists"]:
            issues.append(
                f"{method_report['method']}: method config missing: {method_report['config_path']}"
            )
        checkpoint = method_report["checkpoint"]
        if checkpoint.get("path") and not checkpoint.get("exists", False):
            issues.append(f"{method_report['method']}: checkpoint missing: {checkpoint['path']}")
        expected_action_dim = method_report.get("expected_action_dim")
        checkpoint_action_dim = checkpoint.get("checkpoint_action_dim")
        if not _checkpoint_action_dim_is_compatible(
            agent_type=str(method_report["agent_type"]),
            expected_action_dim=expected_action_dim,
            checkpoint_action_dim=checkpoint_action_dim,
        ):
            issues.append(
                f"{method_report['method']}: checkpoint action dim mismatch "
                f"(checkpoint={checkpoint_action_dim}, expected={expected_action_dim})"
            )

    for artifact in report.get("runtime_artifacts", []):
        artifact_path = Path(str(artifact["path"]))
        owner_str = ", ".join(sorted(artifact.get("owners", [])))
        artifact_label = f"{artifact['kind']} ({owner_str})"
        if not artifact.get("exists", False):
            issues.append(f"{artifact_label}: missing: {artifact_path}")
            continue
        if artifact.get("tracked_required", False) and not _git_is_tracked(
            repo_root, artifact_path
        ):
            issues.append(f"{artifact_label}: untracked: {artifact_path}")
        expected_action_dim = artifact.get("expected_action_dim")
        checkpoint_action_dim = artifact.get("checkpoint_action_dim")
        checkpoint_load_error = artifact.get("checkpoint_load_error")
        if checkpoint_load_error:
            issues.append(
                f"{artifact_label}: checkpoint unreadable: {artifact_path} ({checkpoint_load_error})"
            )
        if not _checkpoint_action_dim_is_compatible(
            agent_type="ppo",
            expected_action_dim=expected_action_dim,
            checkpoint_action_dim=checkpoint_action_dim,
        ):
            issues.append(
                f"{artifact_label}: checkpoint action dim mismatch "
                f"(checkpoint={checkpoint_action_dim}, expected={expected_action_dim})"
            )

    frozen_assets = report.get("frozen_assets", {})
    if frozen_assets.get("source_worktree"):
        if not frozen_assets.get("source_worktree_exists", False):
            issues.append(
                "frozen asset source worktree missing: "
                f"{frozen_assets.get('source_worktree')}"
            )
        elif frozen_assets.get("source_git_error"):
            issues.append(
                "frozen asset source git inspection failed: "
                f"{frozen_assets.get('source_git_error')}"
            )
        elif (
            frozen_assets.get("expected_source_git_sha")
            and frozen_assets.get("actual_source_git_sha")
            != frozen_assets.get("expected_source_git_sha")
        ):
            issues.append(
                "frozen asset source SHA mismatch: "
                f"expected={frozen_assets.get('expected_source_git_sha')}, "
                f"actual={frozen_assets.get('actual_source_git_sha')}"
            )

    for artifact in frozen_assets.get("artifacts", []):
        label = f"frozen asset {artifact['id']}"
        if not artifact.get("exists", False):
            issues.append(f"{label}: missing: {artifact['path']}")
            continue
        if artifact.get("expected_sha256") and artifact.get("actual_sha256") != artifact.get(
            "expected_sha256"
        ):
            issues.append(
                f"{label}: SHA-256 mismatch "
                f"(expected={artifact.get('expected_sha256')}, "
                f"actual={artifact.get('actual_sha256')})"
            )
        expected_obs_dim = artifact.get("expected_obs_dim")
        if expected_obs_dim is not None and artifact.get("checkpoint_obs_dim") != int(
            expected_obs_dim
        ):
            issues.append(
                f"{label}: checkpoint obs dim mismatch "
                f"(expected={expected_obs_dim}, actual={artifact.get('checkpoint_obs_dim')})"
            )
        expected_action_dim = artifact.get("expected_action_dim")
        if expected_action_dim is not None and artifact.get("checkpoint_action_dim") != int(
            expected_action_dim
        ):
            issues.append(
                f"{label}: checkpoint action dim mismatch "
                f"(expected={expected_action_dim}, actual={artifact.get('checkpoint_action_dim')})"
            )

    return report, issues


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preflight reproducibility checks for formal held-out runs."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        required=True,
        help="Repository root to validate (for example a clean worktree).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Comparison config path to validate, absolute or relative to --repo-root.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full report as JSON.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = (repo_root / config_path).resolve()

    if not repo_root.exists():
        print(f"ERROR: repo root does not exist: {repo_root}", file=sys.stderr)
        return 2
    if not config_path.exists():
        print(f"ERROR: config does not exist: {config_path}", file=sys.stderr)
        return 2

    try:
        report, issues = run_preflight(repo_root=repo_root, config_path=config_path)
    except FileNotFoundError as exc:
        report = {"yaml_dependencies": [], "code_dependencies": [], "methods": []}
        issues = [str(exc)]

    if args.json:
        print(json.dumps({"report": report, "issues": issues}, indent=2, ensure_ascii=False))
    else:
        print("Formal Reproducibility Preflight")
        print(f"Repo root: {repo_root}")
        print(f"Config: {config_path}")
        print(f"YAML deps: {len(report['yaml_dependencies'])}")
        print(f"Code deps: {len(report['code_dependencies'])}")
        print(f"Methods: {len(report['methods'])}")
        if issues:
            print("Issues:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("Issues: none")

    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
