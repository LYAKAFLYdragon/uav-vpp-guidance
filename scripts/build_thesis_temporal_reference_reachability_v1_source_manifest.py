#!/usr/bin/env python3
"""Freeze and verify the P0/P1 temporal-reference input snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from uav_vpp_guidance.common.git import get_git_info
from uav_vpp_guidance.evaluation.temporal_reference_compiler import (
    TEMPLATE_NAMES,
    TemporalReferenceConfig,
    TemporalReferenceTemplate,
)


SOURCE_ID = "THESIS-DISADVANTAGE-TEMPORAL-REFERENCE-REACHABILITY-V1"
DEFAULT_CONFIG = ROOT / "config" / "experiment" / "thesis_disadvantage_temporal_reference_reachability_v1_design.yaml"
DEFAULT_OUTPUT = ROOT / "reports" / "temporal_reference_reachability_v1" / "source_manifest.json"


class TemporalReferenceSourceManifestError(ValueError):
    """Raised when P0/P1 provenance cannot be frozen or independently verified."""


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path_value: Any) -> Path:
    path = Path(str(path_value))
    return path if path.is_absolute() else ROOT / path


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TemporalReferenceSourceManifestError("design config must be a mapping")
    return payload


def _yaml_include_closure(path: Path) -> list[Path]:
    """Return one config and its YAML include closure without resolving overrides."""

    visited: set[Path] = set()
    ordered: list[Path] = []

    def visit(current: Path) -> None:
        resolved = current.resolve()
        if resolved in visited:
            return
        if not resolved.is_file():
            raise TemporalReferenceSourceManifestError(f"included runtime config is missing: {resolved}")
        visited.add(resolved)
        ordered.append(resolved)
        payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
        includes = payload.get("includes", []) if isinstance(payload, Mapping) else []
        if not isinstance(includes, list):
            raise TemporalReferenceSourceManifestError(f"includes must be a list: {resolved}")
        for include in includes:
            include_path = Path(str(include))
            visit(include_path if include_path.is_absolute() else resolved.parent / include_path)

    visit(path)
    return ordered


def _runtime_provenance(config: Mapping[str, Any]) -> dict[str, str]:
    runtime = config.get("runtime_contract")
    if not isinstance(runtime, Mapping):
        raise TemporalReferenceSourceManifestError("runtime_contract is missing")
    executable = _resolve(runtime.get("python_executable"))
    if not executable.is_file():
        raise TemporalReferenceSourceManifestError(f"planned Python executable is missing: {executable}")
    probe = (
        "import json, platform, sys; import jsbsim; "
        "print(json.dumps({'python_version': platform.python_version(), "
        "'platform': platform.platform(), 'jsbsim_version': getattr(jsbsim, '__version__', None), "
        "'jsbsim_file': jsbsim.__file__}, sort_keys=True))"
    )
    try:
        actual = json.loads(subprocess.check_output([str(executable), "-c", probe], text=True).strip())
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        raise TemporalReferenceSourceManifestError("could not inspect the planned JSBSim runtime") from error
    if str(actual.get("python_version")) != str(runtime.get("python_version")):
        raise TemporalReferenceSourceManifestError("planned Python version drifted")
    if str(actual.get("jsbsim_version")) != str(runtime.get("jsbsim_version")):
        raise TemporalReferenceSourceManifestError("planned JSBSim version drifted")
    return {
        "python_executable": str(executable),
        "python_version": str(actual["python_version"]),
        "platform": str(actual["platform"]),
        "jsbsim_version": str(actual["jsbsim_version"]),
        "jsbsim_file": str(actual["jsbsim_file"]),
    }


def _validate_config(config: Mapping[str, Any]) -> TemporalReferenceConfig:
    if config.get("source_id") != SOURCE_ID:
        raise TemporalReferenceSourceManifestError("design config source ID drifted")
    authorization = config.get("authorization")
    if not isinstance(authorization, Mapping):
        raise TemporalReferenceSourceManifestError("design config authorization is missing")
    prohibited = (
        "execution_permitted",
        "training_permitted",
        "tuning_permitted",
        "reward_change_permitted",
        "high_level_ppo_training_permitted",
        "shared_skill_training_permitted",
        "guidance_pid_change_permitted",
        "action_envelope_expansion_permitted",
        "snapshot_restore_permitted",
        "future_state_injection_permitted",
        "history_padding_permitted",
    )
    if any(authorization.get(key) is not False for key in prohibited):
        raise TemporalReferenceSourceManifestError("P0/P1 authorization must prohibit execution and training")
    contract = config.get("evaluation_contract")
    if not isinstance(contract, Mapping) or contract.get("action_semantics") != "tactical_basis_v1":
        raise TemporalReferenceSourceManifestError("P1 must use the deployed tactical_basis_v1 semantics")
    if contract.get("formal_runs_allowed") != 1 or contract.get("pooled_gate_permitted") is not False:
        raise TemporalReferenceSourceManifestError("formal-run or opponent-reporting contract drifted")
    manifest_path = _resolve(contract.get("formal_manifest"))
    if not manifest_path.is_file():
        raise TemporalReferenceSourceManifestError("frozen formal manifest is missing")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    recorded_payload_sha = str(manifest.get("integrity", {}).get("payload_sha256", "")).lower()
    if recorded_payload_sha != str(contract.get("formal_manifest_sha256", "")).lower():
        raise TemporalReferenceSourceManifestError("formal manifest payload hash drifted")
    reference = config.get("temporal_reference")
    if not isinstance(reference, Mapping):
        raise TemporalReferenceSourceManifestError("temporal_reference section is missing")
    compiler_config = TemporalReferenceConfig(
        horizon_steps=int(reference["horizon_steps"]),
        high_level_dt_s=float(reference["high_level_dt_s"]),
        max_action_delta=float(reference["max_action_delta"]),
        max_action_slew=float(reference["max_action_slew"]),
        max_basis_condition_number=float(reference["max_basis_condition_number"]),
    )
    if float(reference.get("duration_s", -1.0)) != compiler_config.horizon_steps * compiler_config.high_level_dt_s:
        raise TemporalReferenceSourceManifestError("temporal duration must equal horizon_steps * high_level_dt_s")
    if reference.get("temporal_profile") != "sin_squared_c1":
        raise TemporalReferenceSourceManifestError("P1 temporal profile drifted")
    templates = reference.get("templates")
    if not isinstance(templates, Mapping) or tuple(templates) != TEMPLATE_NAMES:
        raise TemporalReferenceSourceManifestError("P1 template library must match the frozen three-template contract")
    for name, value in templates.items():
        template = TemporalReferenceTemplate.from_mapping(str(name), value)
        if max(abs(component) for component in template.peak_action_delta) > compiler_config.max_action_delta:
            raise TemporalReferenceSourceManifestError(f"template {name} expands the action envelope")
    return compiler_config


def _snapshot_git_info(*, output_path: Path, require_clean_git: bool) -> dict[str, Any]:
    """Require a clean source tree while allowing the manifest artifact itself.

    The snapshot is intentionally generated inside the repository.  Its own
    untracked output must not make the code/config snapshot appear dirty, but
    every other tracked or untracked change remains a hard P0 failure.
    """

    try:
        lines = subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True
        ).splitlines()
    except (OSError, subprocess.CalledProcessError) as error:
        raise TemporalReferenceSourceManifestError("could not inspect Git worktree status") from error
    try:
        allowed = output_path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        # Temporary verifier outputs outside the repository never appear in
        # ``git status`` and therefore cannot need a self-output exemption.
        allowed = None
    unexpected = [
        line for line in lines if allowed is None or line[3:].replace("\\", "/") != allowed
    ]
    if require_clean_git and unexpected:
        raise TemporalReferenceSourceManifestError(
            "P0 source snapshot requires a clean Git worktree; unexpected changes: "
            + ", ".join(unexpected)
        )
    git_info = get_git_info(str(ROOT))
    if not unexpected:
        # ``get_git_info`` sees the generated output as untracked; normalize
        # that self-output away only after the explicit status check above.
        git_info["dirty"] = False
        git_info["description"] = f"{git_info.get('branch') or 'HEAD'}@{git_info.get('short_commit')}"
    return git_info


def build_source_manifest(
    config_path: Path = DEFAULT_CONFIG,
    *,
    output_path: Path = DEFAULT_OUTPUT,
    require_clean_git: bool = True,
) -> dict[str, Any]:
    config = _load_config(config_path)
    compiler_config = _validate_config(config)
    git_info = _snapshot_git_info(output_path=output_path, require_clean_git=require_clean_git)
    inputs = config.get("frozen_inputs")
    if not isinstance(inputs, list) or not inputs:
        raise TemporalReferenceSourceManifestError("frozen_inputs must be a non-empty list")
    entries: list[dict[str, Any]] = []
    for item in inputs:
        if not isinstance(item, Mapping) or not item.get("id") or not item.get("path"):
            raise TemporalReferenceSourceManifestError("each frozen input requires id and path")
        path = _resolve(item["path"])
        if not path.is_file():
            raise TemporalReferenceSourceManifestError(f"frozen input is missing: {item['id']} -> {path}")
        actual_sha = _hash(path)
        expected_sha = str(item.get("sha256", "")).lower() or None
        if expected_sha and expected_sha != actual_sha:
            raise TemporalReferenceSourceManifestError(
                f"frozen input SHA mismatch for {item['id']}: expected {expected_sha}, got {actual_sha}"
            )
        entries.append(
            {
                "id": str(item["id"]),
                "path": str(item["path"]),
                "resolved_path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": actual_sha,
                "declared_sha256": expected_sha,
            }
        )
        if item.get("include_closure"):
            for included in _yaml_include_closure(path)[1:]:
                entries.append(
                    {
                        "id": f"{item['id']}::include::{included.name}",
                        "path": str(included.relative_to(ROOT)),
                        "resolved_path": str(included),
                        "size_bytes": included.stat().st_size,
                        "sha256": _hash(included),
                        "declared_sha256": None,
                    }
                )
    return {
        "schema_version": 1,
        "source_id": SOURCE_ID,
        "status": "p0_p1_frozen_design_execution_not_authorized",
        "snapshot_git_info": git_info,
        "runtime_provenance": _runtime_provenance(config),
        "authorization": dict(config["authorization"]),
        "evaluation_contract": dict(config["evaluation_contract"]),
        "temporal_reference_contract": {
            "action_semantics": "tactical_basis_v1",
            "horizon_steps": compiler_config.horizon_steps,
            "high_level_dt_s": compiler_config.high_level_dt_s,
            "duration_s": compiler_config.horizon_steps * compiler_config.high_level_dt_s,
            "max_action_delta": compiler_config.max_action_delta,
            "max_action_slew": compiler_config.max_action_slew,
            "max_basis_condition_number": compiler_config.max_basis_condition_number,
            "template_names": list(TEMPLATE_NAMES),
        },
        "research_boundary": dict(config["research_boundary"]),
        "inputs": entries,
    }


def verify_source_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("source_id") != SOURCE_ID:
        raise TemporalReferenceSourceManifestError("source manifest schema or ID drifted")
    authorization = payload.get("authorization", {})
    if authorization.get("execution_permitted") is not False or authorization.get("training_permitted") is not False:
        raise TemporalReferenceSourceManifestError("source manifest authorization drifted")
    failures: list[str] = []
    for item in payload.get("inputs", []):
        current = Path(str(item.get("resolved_path", "")))
        if not current.is_file():
            failures.append(f"missing:{item.get('id')}")
            continue
        if _hash(current) != item.get("sha256"):
            failures.append(f"sha256_mismatch:{item.get('id')}")
    if failures:
        raise TemporalReferenceSourceManifestError("source manifest verification failed: " + ", ".join(failures))
    return {"source_id": SOURCE_ID, "input_count": len(payload.get("inputs", [])), "passes": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    if args.verify:
        result = verify_source_manifest(args.output)
    else:
        result = build_source_manifest(
            args.config,
            output_path=args.output,
            require_clean_git=not args.allow_dirty,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
