#!/usr/bin/env python
"""Independently verify the rebased H0 and end-to-end policy assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from uav_vpp_guidance.common.hash import file_sha256
from uav_vpp_guidance.evaluation.lstm_retraining_v2 import (
    inspect_policy_checkpoint,
    strict_load_predictor,
)


SOURCE_ID = "THESIS-PREDICTED-GEOMETRY-H0-CURRENT-AND-E2E-ASSETS-V1-20260903"
PREDICTOR_SOURCE_ID = "THESIS-PREDICTED-GEOMETRY-LSTM-VPP-RETRAIN-V2-20260903"
EXPECTED_H0_SEEDS = (9201, 9202, 9203)
EXPECTED_E2E_SEED = 9204


def _receipt(path: Path) -> dict[str, Any]:
    exists = path.is_file()
    return {
        "path": str(path.resolve()),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else None,
        "sha256": file_sha256(path) if exists else None,
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _verify_run(
    run_dir: Path,
    *,
    expected_role: str,
    expected_seed: int,
    errors: list[str],
) -> dict[str, Any]:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        errors.append(f"missing run manifest: {manifest_path}")
        return {"role": expected_role, "seed": expected_seed, "ok": False}
    manifest = _load_json(manifest_path)
    if manifest.get("source_id") != SOURCE_ID:
        errors.append(f"{run_dir}: wrong asset source_id")
    if manifest.get("role") != expected_role:
        errors.append(f"{run_dir}: role mismatch")
    if int(manifest.get("seed", -1)) != expected_seed:
        errors.append(f"{run_dir}: seed mismatch")
    if manifest.get("stage") != "full_training":
        errors.append(f"{run_dir}: not marked full_training")
    if manifest.get("status") != "completed":
        errors.append(f"{run_dir}: training did not complete")
    if not manifest.get("training_started"):
        errors.append(f"{run_dir}: training_started is false")
    if int(manifest.get("training_requested_timesteps", 0)) < 200000:
        errors.append(f"{run_dir}: training timesteps are below 200000")

    checkpoint = run_dir / "checkpoints" / "last.pt"
    actual = _receipt(checkpoint)
    recorded = manifest.get("checkpoint_receipt") or {}
    if not actual["exists"]:
        errors.append(f"{run_dir}: checkpoint is missing")
    if recorded.get("sha256") != actual.get("sha256"):
        errors.append(f"{run_dir}: recorded checkpoint SHA-256 mismatch")
    if recorded.get("size_bytes") != actual.get("size_bytes"):
        errors.append(f"{run_dir}: recorded checkpoint size mismatch")
    if manifest.get("checkpoint", {}).get("receipt", {}).get("sha256") != actual.get("sha256"):
        errors.append(f"{run_dir}: nested checkpoint contract mismatch")

    try:
        contract = inspect_policy_checkpoint(checkpoint, min_timesteps=200000)
    except Exception as exc:
        errors.append(f"{run_dir}: checkpoint contract failed: {type(exc).__name__}: {exc}")
        contract = {}
    if contract and (contract.get("obs_dim"), contract.get("action_dim")) != (16, 3):
        errors.append(f"{run_dir}: policy dimensions are not 16 x 3")

    resolved = run_dir / "resolved_config.yaml"
    if not resolved.is_file():
        errors.append(f"{run_dir}: resolved_config.yaml is missing")
    else:
        recorded_config = manifest.get("config", {})
        if recorded_config.get("sha256") != file_sha256(resolved):
            errors.append(f"{run_dir}: resolved config SHA-256 mismatch")
    command = manifest.get("command_result", {}).get("command")
    if not command:
        errors.append(f"{run_dir}: training command is missing")
    git_info = manifest.get("git_info", {})
    if not git_info.get("commit") or not git_info.get("branch"):
        errors.append(f"{run_dir}: Git provenance is incomplete")
    return {
        "role": expected_role,
        "seed": expected_seed,
        "run_dir": str(run_dir.resolve()),
        "checkpoint": actual,
        "contract": contract,
        "config": _receipt(resolved),
        "ok": not any(str(run_dir) in item for item in errors),
    }


def verify(asset_root: Path, predictor_path: Path) -> dict[str, Any]:
    errors: list[str] = []
    runs: list[dict[str, Any]] = []
    for seed in EXPECTED_H0_SEEDS:
        runs.append(_verify_run(
            asset_root / "h0_current_geometry" / f"seed_{seed}",
            expected_role="h0_current_geometry",
            expected_seed=seed,
            errors=errors,
        ))
    runs.append(_verify_run(
        asset_root / "end_to_end_opponent" / f"seed_{EXPECTED_E2E_SEED}",
        expected_role="end_to_end_opponent",
        expected_seed=EXPECTED_E2E_SEED,
        errors=errors,
    ))

    predictor_result: dict[str, Any] = {"path": str(predictor_path.resolve())}
    try:
        predictor_result["strict_load"] = strict_load_predictor(predictor_path)
    except Exception as exc:
        errors.append(f"predictor strict load failed: {type(exc).__name__}: {exc}")
    predictor_manifest_path = predictor_path.parent / "run_manifest.json"
    if not predictor_manifest_path.is_file():
        errors.append("predictor run_manifest.json is missing")
    else:
        predictor_manifest = _load_json(predictor_manifest_path)
        if predictor_manifest.get("source_id") != PREDICTOR_SOURCE_ID:
            errors.append("predictor source_id is not the audited V2 source")
        expected_sha = predictor_manifest.get("checkpoint", {}).get("receipt", {}).get("sha256")
        if expected_sha != file_sha256(predictor_path):
            errors.append("predictor manifest SHA-256 mismatch")

    asset_manifest_path = asset_root / "asset_manifest.json"
    if not asset_manifest_path.is_file():
        errors.append("asset_manifest.json is missing")
    else:
        asset_manifest = _load_json(asset_manifest_path)
        if asset_manifest.get("source_id") != SOURCE_ID:
            errors.append("asset_manifest.json has the wrong source_id")
        if len(asset_manifest.get("assets", [])) != 4:
            errors.append("asset_manifest.json does not list exactly four assets")

    return {
        "source_id": SOURCE_ID,
        "asset_root": str(asset_root.resolve()),
        "predictor": predictor_result,
        "runs": runs,
        "errors": errors,
        "ok": not errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=Path("outputs/prediction_highlevel_h0_asset_retraining_v1"))
    parser.add_argument("--predictor", type=Path, default=Path("outputs/predicted_geometry_lstm_vpp_retraining_v2/predictor_training/best_model.pt"))
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    result = verify(args.asset_root.resolve(), args.predictor.resolve())
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
