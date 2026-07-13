#!/usr/bin/env python3
"""Verify frozen assets and disk guards before five-state v1 training."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ASSET_MANIFEST = (
    REPO_ROOT / "reports" / "thesis_five_state_shared_intent_v1_asset_manifest.yaml"
)
DEFAULT_RETENTION_POLICY = (
    REPO_ROOT
    / "config"
    / "experiment"
    / "thesis_five_state_shared_intent_v1_output_retention.yaml"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping in {path}")
    return payload


def _checkpoint_dimensions(path: Path) -> Dict[str, Optional[int]]:
    checkpoint = torch.load(path, map_location="cpu")
    return {
        "obs_dim": checkpoint.get("obs_dim"),
        "action_dim": checkpoint.get("action_dim"),
    }


def _asset_report(asset: Dict[str, Any]) -> Dict[str, Any]:
    path = Path(str(asset["frozen_path"]))
    item: Dict[str, Any] = {
        "id": str(asset["id"]),
        "path": str(path),
        "exists": path.is_file(),
        "expected_sha256": str(asset["sha256"]).lower(),
        "actual_sha256": None,
        "expected_obs_dim": asset.get("expected_obs_dim"),
        "expected_action_dim": asset.get("expected_action_dim"),
        "obs_dim": None,
        "action_dim": None,
        "errors": [],
    }
    if not item["exists"]:
        item["errors"].append("missing")
        return item

    item["actual_sha256"] = _sha256_file(path)
    if item["actual_sha256"] != item["expected_sha256"]:
        item["errors"].append("sha256_mismatch")

    if item["expected_obs_dim"] is not None or item["expected_action_dim"] is not None:
        try:
            dimensions = _checkpoint_dimensions(path)
            item.update(dimensions)
        except Exception as exc:  # pragma: no cover - defensive reporting
            item["errors"].append(f"checkpoint_load_error:{exc}")
            return item
        if item["expected_obs_dim"] != item["obs_dim"]:
            item["errors"].append("obs_dim_mismatch")
        if item["expected_action_dim"] != item["action_dim"]:
            item["errors"].append("action_dim_mismatch")
    return item


def _retention_report(
    policy: Dict[str, Any],
    disk_usage: Callable[[str], shutil._ntuple_diskusage],
) -> Dict[str, Any]:
    output_root = Path(str(policy["output_root"]))
    minimum_free_gb = float(policy["minimum_free_gb_before_run"])
    top_k = int(policy.get("checkpoint_retention", {}).get("top_k", 0))
    allowed = set(policy.get("raw_telemetry", {}).get("full_allowed_splits", []) or [])
    prohibited = set(policy.get("raw_telemetry", {}).get("prohibited_splits", []) or [])
    usage = disk_usage(output_root.anchor)
    free_gb = usage.free / (1024**3)
    errors: List[str] = []
    if minimum_free_gb < 100:
        errors.append("minimum_free_gb_below_100")
    if free_gb < minimum_free_gb:
        errors.append("insufficient_free_space")
    if top_k != 3:
        errors.append("checkpoint_top_k_must_equal_3")
    if "train" in allowed or "train" not in prohibited:
        errors.append("training_raw_telemetry_must_be_prohibited")
    if output_root.exists() and any(output_root.iterdir()):
        errors.append("output_root_is_not_empty")
    return {
        "output_root": str(output_root),
        "output_root_exists": output_root.exists(),
        "minimum_free_gb": minimum_free_gb,
        "actual_free_gb": round(free_gb, 2),
        "checkpoint_top_k": top_k,
        "full_raw_telemetry_allowed_splits": sorted(allowed),
        "full_raw_telemetry_prohibited_splits": sorted(prohibited),
        "errors": errors,
    }


def build_preflight_report(
    asset_manifest_path: Path,
    retention_policy_path: Path,
    *,
    disk_usage: Callable[[str], shutil._ntuple_diskusage] = shutil.disk_usage,
) -> Dict[str, Any]:
    """Build a pure preflight result for scripts and tests."""

    manifest = _load_yaml(asset_manifest_path)
    policy = _load_yaml(retention_policy_path)
    assets = [_asset_report(item) for item in manifest.get("assets", [])]
    retention = _retention_report(policy, disk_usage)
    asset_errors = [
        f"{asset['id']}:{error}"
        for asset in assets
        for error in asset["errors"]
    ]
    third = manifest.get("third_ppo_vpp_opponent", {}) or {}
    third_ready = bool(third.get("eligible", False))
    blockers = [] if third_ready else [str(third.get("status", "third_opponent_unready"))]
    integrity_pass = not asset_errors and not retention["errors"]
    return {
        "lane": manifest.get("lane", {}),
        "asset_manifest": str(asset_manifest_path),
        "retention_policy": str(retention_policy_path),
        "assets": assets,
        "retention": retention,
        "asset_integrity_pass": integrity_pass,
        "third_opponent_ready": third_ready,
        "training_ready": integrity_pass and third_ready,
        "errors": asset_errors + list(retention["errors"]),
        "blockers": blockers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-manifest", type=Path, default=DEFAULT_ASSET_MANIFEST)
    parser.add_argument("--retention-policy", type=Path, default=DEFAULT_RETENTION_POLICY)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--require-training-ready",
        action="store_true",
        help="Return nonzero until an eligible independent PPO/VPP opponent is frozen.",
    )
    args = parser.parse_args()
    report = build_preflight_report(args.asset_manifest, args.retention_policy)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    if report["errors"]:
        return 1
    if args.require_training_ready and not report["training_ready"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
