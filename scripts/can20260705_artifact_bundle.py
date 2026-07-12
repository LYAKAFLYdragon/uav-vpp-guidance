#!/usr/bin/env python3
"""Create and verify the immutable CAN-20260705 reproduction bundle.

The formal raw episodes were generated from commit 9a9f9bf6.  The worktree
that currently hosts those episodes has since advanced, so this tool keeps the
code snapshot, external assets, and raw formal outputs separately auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch


CANONICAL_COMMIT = "9a9f9bf6d68560afa81560f5260b78f318dcd9b1"
RUN_IDS = (
    "oracle_vs_commander_post_merge_recovery_manifest60_jointgate_fwdneg_formal_expert_20260705",
    "oracle_vs_commander_post_merge_recovery_manifest60_jointgate_fwdneg_formal_end_to_end_20260705",
)
HIGH_LEVEL_CHECKPOINT = Path(
    "outputs/diagnostics/"
    "hierarchical_commander_mvp_2mode_task_type_mode_dominance_gated_alignment_"
    "shaped_v2_oracle_imitation_warmstart_balanced_tail_checkpoint_retrospective_"
    "expert_10seed_20260701_checkpoint_snapshots/best.pt"
)
HEAD_ON_CHECKPOINT = Path(
    "outputs/experiments/"
    "prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_"
    "longscale00_tactical_basis_headon_mvp_combat_finetune_mixed_opponent_"
    "aware_32k_task_type_headon_weighted/checkpoints/best.pt"
)
CROSSING_CHECKPOINT = Path(
    "outputs/experiments/prediction_vpp_ppo_crossing_v3/checkpoints/last.pt"
)
PREDICTION_CHECKPOINT = Path("outputs/trajectory_prediction/best_model.pt")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {"bundle_manifest.json", "bundle_verification.json"}:
            yield path


def git_commit(worktree: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=worktree,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def checkpoint_shape(path: Path) -> Dict[str, Any]:
    payload = torch.load(str(path), map_location="cpu")
    return {
        "obs_dim": payload.get("obs_dim"),
        "action_dim": payload.get("action_dim"),
    }


def copy_payload(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"required source is missing: {source}")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing destination: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)


def asset_specs(asset_source_root: Path, formal_source_root: Path) -> List[Tuple[str, Path, Path, Path]]:
    """Return name, source, bundle-relative path, worktree-relative path."""
    return [
        (
            "jsbsim_data",
            formal_source_root / "envs/JSBSim",
            Path("assets/envs/JSBSim"),
            Path("envs/JSBSim"),
        ),
        (
            "experiment_checkpoints",
            asset_source_root / "outputs/experiments",
            Path("assets/outputs/experiments"),
            Path("outputs/experiments"),
        ),
        (
            "trajectory_prediction",
            asset_source_root / "outputs/trajectory_prediction",
            Path("assets/outputs/trajectory_prediction"),
            Path("outputs/trajectory_prediction"),
        ),
        (
            "high_level_checkpoint",
            asset_source_root / HIGH_LEVEL_CHECKPOINT,
            Path("assets") / HIGH_LEVEL_CHECKPOINT,
            HIGH_LEVEL_CHECKPOINT,
        ),
    ]


def run_specs(formal_source_root: Path) -> List[Tuple[str, Path, Path]]:
    return [
        (
            run_id,
            formal_source_root / "outputs/jsbsim_hrl_comparison" / run_id,
            Path("formal_runs") / run_id,
        )
        for run_id in RUN_IDS
    ]


def build_file_index(bundle_root: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in payload_files(bundle_root):
        records.append(
            {
                "path": path.relative_to(bundle_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        try:
            path.chmod(stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
        except OSError:
            pass
    try:
        root.chmod(stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    except OSError:
        pass


def create_bundle(
    *,
    canonical_worktree: Path,
    asset_source_root: Path,
    formal_source_root: Path,
    bundle_root: Path,
    make_bundle_read_only: bool,
) -> Dict[str, Any]:
    if git_commit(canonical_worktree) != CANONICAL_COMMIT:
        raise RuntimeError(
            f"canonical worktree commit mismatch: expected {CANONICAL_COMMIT}, "
            f"got {git_commit(canonical_worktree)}"
        )
    if bundle_root.exists():
        raise FileExistsError(f"bundle already exists: {bundle_root}")

    bundle_root.mkdir(parents=True)
    copied_assets: Dict[str, Dict[str, str]] = {}
    copied_runs: Dict[str, str] = {}
    for name, source, bundle_relative, worktree_relative in asset_specs(
        asset_source_root, formal_source_root
    ):
        bundle_target = bundle_root / bundle_relative
        worktree_target = canonical_worktree / worktree_relative
        copy_payload(source, bundle_target)
        copy_payload(bundle_target, worktree_target)
        copied_assets[name] = {
            "bundle_path": bundle_relative.as_posix(),
            "worktree_path": worktree_relative.as_posix(),
            "source_path": str(source),
        }

    for run_id, source, bundle_relative in run_specs(formal_source_root):
        copy_payload(source, bundle_root / bundle_relative)
        copied_runs[run_id] = bundle_relative.as_posix()

    checkpoint_shapes = {
        "high_level": checkpoint_shape(bundle_root / "assets" / HIGH_LEVEL_CHECKPOINT),
        "head_on": checkpoint_shape(bundle_root / "assets" / HEAD_ON_CHECKPOINT),
        "crossing": checkpoint_shape(bundle_root / "assets" / CROSSING_CHECKPOINT),
    }
    manifest = {
        "schema_version": "1.0",
        "bundle_id": "CAN-20260705",
        "canonical_commit": CANONICAL_COMMIT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_worktree": str(canonical_worktree),
        "formal_source_root": str(formal_source_root),
        "asset_source_root": str(asset_source_root),
        "assets": copied_assets,
        "formal_runs": copied_runs,
        "checkpoint_shapes": checkpoint_shapes,
        "required_run_files": [
            "run_manifest.json",
            "artifact_contract.json",
            "resolved_config.yaml",
            "summary.csv",
            "aggregate",
            "raw",
        ],
        "files": build_file_index(bundle_root),
    }
    (bundle_root / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (bundle_root / "README.md").write_text(
        "# CAN-20260705 immutable reproduction bundle\n\n"
        f"Canonical code commit: `{CANONICAL_COMMIT}`.\n\n"
        "This directory stores copied external assets and formal raw outputs. "
        "Use `scripts/can20260705_artifact_bundle.py verify` before relying on it.\n",
        encoding="utf-8",
    )
    result = verify_bundle(bundle_root=bundle_root, canonical_worktree=canonical_worktree)
    (bundle_root / "bundle_verification.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    if not result["valid"]:
        raise RuntimeError(f"new bundle verification failed: {result['issues']}")
    if make_bundle_read_only:
        make_read_only(bundle_root)
    return result


def verify_bundle(*, bundle_root: Path, canonical_worktree: Path | None) -> Dict[str, Any]:
    manifest_path = bundle_root / "bundle_manifest.json"
    if not manifest_path.exists():
        return {"valid": False, "issues": ["bundle_manifest.json is missing"]}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    issues: List[str] = []
    if manifest.get("canonical_commit") != CANONICAL_COMMIT:
        issues.append("canonical commit recorded in bundle manifest is incorrect")
    for record in manifest.get("files", []):
        path = bundle_root / record["path"]
        if not path.exists():
            issues.append(f"missing payload file: {record['path']}")
            continue
        if path.stat().st_size != record["size_bytes"]:
            issues.append(f"size mismatch: {record['path']}")
        elif sha256_file(path) != record["sha256"]:
            issues.append(f"hash mismatch: {record['path']}")
    for run_id, run_relative in manifest.get("formal_runs", {}).items():
        run_root = bundle_root / run_relative
        for required in manifest.get("required_run_files", []):
            if not (run_root / required).exists():
                issues.append(f"{run_id}: missing required run artifact {required}")
        run_manifest_path = run_root / "run_manifest.json"
        if run_manifest_path.exists():
            run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
            if run_manifest.get("status") != "completed":
                issues.append(f"{run_id}: run status is not completed")
            if run_manifest.get("paper_safe") is not True:
                issues.append(f"{run_id}: paper_safe is not true")
            if run_manifest.get("git_info", {}).get("commit") != CANONICAL_COMMIT:
                issues.append(f"{run_id}: git commit does not match CAN-20260705")
    for name, expected in manifest.get("checkpoint_shapes", {}).items():
        relative = {
            "high_level": Path("assets") / HIGH_LEVEL_CHECKPOINT,
            "head_on": Path("assets") / HEAD_ON_CHECKPOINT,
            "crossing": Path("assets") / CROSSING_CHECKPOINT,
        }[name]
        path = bundle_root / relative
        if not path.exists():
            issues.append(f"{name}: checkpoint is missing")
        elif checkpoint_shape(path) != expected:
            issues.append(f"{name}: checkpoint shape mismatch")
    if canonical_worktree is not None:
        if git_commit(canonical_worktree) != CANONICAL_COMMIT:
            issues.append("canonical worktree is not at CAN-20260705 commit")
        for asset in manifest.get("assets", {}).values():
            bundle_path = bundle_root / asset["bundle_path"]
            worktree_path = canonical_worktree / asset["worktree_path"]
            if not worktree_path.exists():
                issues.append(f"worktree asset missing: {asset['worktree_path']}")
                continue
            if bundle_path.is_file():
                if sha256_file(bundle_path) != sha256_file(worktree_path):
                    issues.append(f"worktree asset hash mismatch: {asset['worktree_path']}")
    return {"valid": not issues, "issues": issues, "bundle_id": manifest.get("bundle_id")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create", help="copy and verify an immutable bundle")
    create.add_argument("--canonical-worktree", type=Path, required=True)
    create.add_argument("--asset-source-root", type=Path, required=True)
    create.add_argument("--formal-source-root", type=Path, required=True)
    create.add_argument("--bundle-root", type=Path, required=True)
    create.add_argument("--writable", action="store_true", help="do not mark the finished bundle read-only")
    verify = subparsers.add_parser("verify", help="verify hashes, shapes, and formal run contracts")
    verify.add_argument("--bundle-root", type=Path, required=True)
    verify.add_argument("--canonical-worktree", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "create":
        result = create_bundle(
            canonical_worktree=args.canonical_worktree.resolve(),
            asset_source_root=args.asset_source_root.resolve(),
            formal_source_root=args.formal_source_root.resolve(),
            bundle_root=args.bundle_root.resolve(),
            make_bundle_read_only=not args.writable,
        )
    else:
        result = verify_bundle(
            bundle_root=args.bundle_root.resolve(),
            canonical_worktree=args.canonical_worktree.resolve() if args.canonical_worktree else None,
        )
    print(json.dumps(result, indent=2))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
