#!/usr/bin/env python
"""Independently verify the V2 retraining source and portable bundle."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from uav_vpp_guidance.evaluation.lstm_retraining_v2 import (
    PREDICTOR_CONTRACT,
    SOURCE_ID,
    file_sha256,
    inspect_policy_checkpoint,
    strict_load_predictor,
    verify_corpus,
    verify_formal_metrics,
)

EXPECTED_POLICY_SEEDS = (7101, 7102, 7103)


def _read_json(path: Path, errors: list[str]) -> dict:
    if not path.is_file():
        errors.append(f"missing JSON: {path}")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"invalid JSON {path}: {type(exc).__name__}: {exc}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"JSON root is not an object: {path}")
        return {}
    return value


def _bundle_path(root: Path, value: str) -> Path:
    """Accept bundle paths written by either Windows or POSIX producers."""
    return root / Path(value.replace("\\", os.sep).replace("/", os.sep))


def _require_git_info(value: object, label: str, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append(f"{label}: git_info is missing")
        return
    for key in ("commit", "branch", "dirty"):
        if key not in value or value[key] in (None, ""):
            errors.append(f"{label}: git_info.{key} is missing")


def _verify_policy_entry(root: Path, entry: dict, errors: list[str]) -> dict | None:
    seed = entry.get("seed")
    label = f"policy seed {seed}"
    if seed not in EXPECTED_POLICY_SEEDS:
        errors.append(f"unexpected policy seed: {seed}")
        return None
    training_dir = root / "policy_training" / f"seed_{seed}"
    manifest = _read_json(training_dir / "run_manifest.json", errors)
    if manifest.get("stage") != "policy_training":
        errors.append(f"{label}: run manifest stage is not policy_training")
    if int(manifest.get("seed", -1)) != int(seed):
        errors.append(f"{label}: run manifest seed mismatch")
    if manifest.get("config_sha256") != entry.get("config_sha256"):
        errors.append(f"{label}: source and child config hashes differ")
    command_result = manifest.get("command_result")
    if not isinstance(command_result, dict) or not command_result.get("command"):
        errors.append(f"{label}: training command is missing")
    elif int(command_result.get("returncode", -1)) != 0:
        errors.append(f"{label}: training command did not succeed")
    _require_git_info(manifest.get("git_info"), label, errors)

    checkpoint = training_dir / "checkpoints" / "last.pt"
    try:
        info = inspect_policy_checkpoint(checkpoint)
    except Exception as exc:
        errors.append(f"{label}: {type(exc).__name__}: {exc}")
        return None
    if info["total_timesteps"] != int(entry.get("checkpoint", {}).get("total_timesteps", -1)):
        errors.append(f"{label}: checkpoint timesteps differ from source manifest")
    if info["total_timesteps"] < 100_000:
        errors.append(f"{label}: checkpoint is not a full training result")
    receipt = entry.get("checkpoint", {}).get("receipt", {})
    if receipt.get("sha256") != info["receipt"].get("sha256"):
        errors.append(f"{label}: checkpoint hash differs from source manifest")
    return info


def verify_source(root: Path) -> dict:
    """Verify every required provenance and runtime gate for the V2 source."""
    errors: list[str] = []
    source = _read_json(root / "source_run_manifest.json", errors)
    if source.get("source_id") != SOURCE_ID:
        errors.append("source_run_manifest has the wrong source_id")
    if source.get("status") != "completed":
        errors.append("source_run_manifest is not completed")

    predictor = source.get("predictor", {})
    predictor_checkpoint = predictor.get("checkpoint", {})
    if predictor_checkpoint.get("architecture") != PREDICTOR_CONTRACT:
        errors.append("predictor architecture does not match the strict contract")
    if predictor_checkpoint.get("strict_load") is not True:
        errors.append("predictor strict_load is not true")
    try:
        predictor_info = strict_load_predictor(root / "predictor_training" / "best_model.pt")
    except Exception as exc:
        predictor_info = {"error": f"{type(exc).__name__}: {exc}"}
        errors.append(predictor_info["error"])
    else:
        expected_sha = predictor_checkpoint.get("receipt", {}).get("sha256")
        if expected_sha != predictor_info["receipt"].get("sha256"):
            errors.append("predictor hash differs from source manifest")

    corpus_info = verify_corpus(
        root / "corpus" / "trajectories" / "no_prediction_vpp" / "jsbsim",
        min_files=6,
        min_rows=15,
    )
    errors.extend(corpus_info["errors"])

    policies = source.get("policies", [])
    by_seed = {entry.get("seed"): entry for entry in policies if isinstance(entry, dict)}
    if set(by_seed) != set(EXPECTED_POLICY_SEEDS):
        errors.append(f"policy seed set is {sorted(by_seed)}, expected {list(EXPECTED_POLICY_SEEDS)}")
    policy_info = []
    for seed in EXPECTED_POLICY_SEEDS:
        entry = by_seed.get(seed)
        if entry is None:
            continue
        info = _verify_policy_entry(root, entry, errors)
        if info is not None:
            policy_info.append(info)

    asset_manifest = _read_json(root / "asset_manifest.json", errors)
    if asset_manifest.get("source_id") != SOURCE_ID:
        errors.append("asset_manifest has the wrong source_id")
    asset_entries = asset_manifest.get("assets", [])
    asset_keys = {(item.get("role"), item.get("seed")) for item in asset_entries if isinstance(item, dict)}
    expected_keys = {("trajectory_predictor", None)} | {("lstm_vpp_policy", seed) for seed in EXPECTED_POLICY_SEEDS}
    if asset_keys != expected_keys:
        errors.append(f"asset manifest entries are {sorted(asset_keys)}, expected {sorted(expected_keys)}")
    for item in asset_entries:
        if not isinstance(item, dict):
            continue
        receipt = item.get("contract", {}).get("receipt", {})
        path = Path(receipt.get("path", ""))
        if not path.is_file():
            errors.append(f"asset manifest file is missing: {path}")
        elif file_sha256(path) != receipt.get("sha256"):
            errors.append(f"asset manifest hash mismatch: {path}")

    eval_manifests = sorted((root / "heldout").glob("seed_*/expert/run_manifest.json"))
    if not eval_manifests:
        errors.append("strict heldout evaluation manifest is missing")
    evaluation_info = []
    for manifest_path in eval_manifests:
        evaluation = _read_json(manifest_path, errors)
        if evaluation.get("source_id") != SOURCE_ID:
            errors.append(f"{manifest_path}: wrong source_id")
        if evaluation.get("backend_requested") != "jsbsim" or evaluation.get("backend_effective") != "jsbsim":
            errors.append(f"{manifest_path}: backend is not strict JSBSim")
        if evaluation.get("backend_fallback_occurred") is not False:
            errors.append(f"{manifest_path}: backend fallback occurred")
        verification = evaluation.get("verification", {})
        episode_count = int(verification.get("episode_count", 0) or 0)
        if not verification.get("ok", False):
            errors.append(f"{manifest_path}: strict verification is not passing")
        if int(evaluation.get("reset_count", -1)) != episode_count:
            errors.append(f"{manifest_path}: reset count is not one initial reset per episode")
        scenarios = evaluation.get("scenarios", [])
        eval_seeds = evaluation.get("evaluation_seeds", [])
        episodes_per_scenario = int(evaluation.get("episodes_per_scenario", 0) or 0)
        expected_episodes = len(scenarios) * len(eval_seeds) * episodes_per_scenario
        if expected_episodes != episode_count:
            errors.append(f"{manifest_path}: episode denominator is inconsistent")
        metrics_path = Path(evaluation.get("metrics_path", ""))
        metrics = _read_json(metrics_path, errors)
        formal = verify_formal_metrics(metrics, episode_count)
        errors.extend(f"{manifest_path}: {error}" for error in formal["errors"])
        evaluation_info.append({"path": str(manifest_path), "episode_count": episode_count, "scenarios": scenarios})

    bundle = _read_json(root / "portable_bundle" / "bundle_info.json", errors)
    bundle_files = bundle.get("files", [])
    if not bundle_files:
        errors.append("portable bundle has no file list")
    for item in bundle_files:
        if not isinstance(item, dict):
            errors.append("portable bundle contains a malformed file entry")
            continue
        path = _bundle_path(root / "portable_bundle", str(item.get("path", "")))
        if not path.is_file():
            errors.append(f"bundle file missing: {path}")
        elif file_sha256(path) != item.get("sha256"):
            errors.append(f"bundle hash mismatch: {path}")

    return {
        "source_id": SOURCE_ID,
        "ok": not errors,
        "errors": errors,
        "predictor": predictor_info,
        "corpus": corpus_info,
        "policies": policy_info,
        "evaluations": evaluation_info,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    result = verify_source(root)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
