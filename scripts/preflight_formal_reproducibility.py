#!/usr/bin/env python3
"""Preflight reproducibility checks for formal held-out comparison runs."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import yaml


CORE_AGENT_DEPENDENCIES = {
    "hierarchical_commander": [
        "src/uav_vpp_guidance/evaluation/hierarchical_commander_policy.py",
        "src/uav_vpp_guidance/agents/commander_ppo_agent.py",
        "src/uav_vpp_guidance/agents/commander_double_dqn_agent.py",
        "src/uav_vpp_guidance/envs/hierarchical_commander_env.py",
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
        return {"checkpoint_obs_dim": None, "checkpoint_action_dim": None}
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    return {
        "checkpoint_obs_dim": checkpoint.get("obs_dim"),
        "checkpoint_action_dim": checkpoint.get("action_dim"),
    }


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
        if (
            expected_action_dim is not None
            and checkpoint_action_dim is not None
            and int(expected_action_dim) != int(checkpoint_action_dim)
        ):
            issues.append(
                f"{method_report['method']}: checkpoint action dim mismatch "
                f"(checkpoint={checkpoint_action_dim}, expected={expected_action_dim})"
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
