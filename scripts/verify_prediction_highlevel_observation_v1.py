#!/usr/bin/env python
"""Verify H0/H1/H2 artifacts without running a simulator."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from uav_vpp_guidance.evaluation.prediction_highlevel_observation_v1 import (
    EXPECTED_METHODS,
    EXPECTED_OPPONENTS,
    EXPECTED_SCENARIOS,
    SOURCE_ID,
    finite_numeric,
    file_sha256,
)


OPTIONAL_NAN_FIELDS = {
    # These summaries are legitimately unavailable when prediction is disabled
    # or when an episode never enters the corresponding event state.
    "mean_env_prediction_error_m",
    "median_env_prediction_error_m",
    "mean_offline_aligned_error_m",
    "median_offline_aligned_error_m",
    "median_prediction_error_m",
    "mean_anchor_shift_m",
    "time_to_first_advantage_s",
}


def _load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _verify_file_receipt(receipt: dict[str, Any], label: str, errors: list[str]) -> None:
    if not isinstance(receipt, dict) or not receipt.get("exists"):
        errors.append(f"{label} is missing")
        return
    path = receipt.get("path")
    actual_hash = file_sha256(path) if path else None
    if actual_hash is None:
        errors.append(f"{label} path is missing: {path}")
    elif receipt.get("sha256") != actual_hash:
        errors.append(f"{label} hash does not match the current file")
    if path and Path(path).is_file() and receipt.get("size_bytes") != Path(path).stat().st_size:
        errors.append(f"{label} size does not match the recorded receipt")


def _finite_episode_value(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(_finite_episode_value(item) for item in value)
    if isinstance(value, dict):
        return all(_finite_episode_value(item) for item in value.values())
    return True


def _verify_run(run_dir: Path, source_id: str = SOURCE_ID) -> dict[str, Any]:
    errors: list[str] = []
    manifest_path = run_dir / "run_manifest.json"
    metrics_path = run_dir / "metrics.json"
    if not manifest_path.is_file():
        errors.append("missing run_manifest.json")
        return {"path": str(run_dir), "ok": False, "errors": errors}
    manifest = _load(manifest_path)
    if manifest.get("source_id") != source_id:
        errors.append("wrong source_id")
    if "prediction_decision_bridge_v1" in json.dumps(manifest):
        errors.append("manifest references forbidden prediction_decision_bridge_v1 output")
    method = manifest.get("method")
    opponent = manifest.get("opponent")
    if method not in EXPECTED_METHODS:
        errors.append(f"unknown method: {method}")
    if opponent not in EXPECTED_OPPONENTS:
        errors.append(f"unknown opponent: {opponent}")
    _verify_file_receipt(manifest.get("checkpoint") or {}, "checkpoint", errors)
    config_path = manifest.get("config_path")
    if not config_path:
        errors.append("config_path is missing")
    elif manifest.get("config_sha256") != file_sha256(config_path):
        errors.append("resolved config hash does not match the current file")
    if not metrics_path.is_file():
        errors.append("missing metrics.json")
        return {"path": str(run_dir), "ok": not errors, "errors": errors}

    metrics = _load(metrics_path)
    episodes = metrics.get("raw_episodes", [])
    if not isinstance(episodes, list) or not episodes:
        errors.append("metrics.json has no raw_episodes")
    for index, episode in enumerate(episodes):
        if episode.get("method") not in (None, method):
            errors.append(f"episode {index} method mismatch")
        if episode.get("opponent") not in (None, opponent):
            errors.append(f"episode {index} opponent mismatch")
        if episode.get("scenario") not in EXPECTED_SCENARIOS:
            errors.append(f"episode {index} has an unknown scenario")
        for key, value in episode.items():
            if not _finite_episode_value(value):
                # These metrics can be unavailable when an episode terminates
                # before a final geometry sample exists.
                if key not in OPTIONAL_NAN_FIELDS | {"mean_prediction_error_m", "final_range_m", "final_ata_deg"}:
                    errors.append(f"episode {index} field {key} is non-finite")
    schema = manifest.get("observation_schema", {})
    if manifest.get("obs_dim") != 16:
        errors.append(f"expected 16-D observation, got {manifest.get('obs_dim')}")
    loaded_dims = manifest.get("checkpoint_load") or {}
    if loaded_dims.get("obs_dim") not in (None, manifest.get("obs_dim")):
        errors.append("checkpoint-loaded observation dimension disagrees with manifest")
    if loaded_dims.get("action_dim") not in (None, manifest.get("action_dim")):
        errors.append("checkpoint-loaded action dimension disagrees with manifest")
    if manifest.get("backend_requested") == "jsbsim" and manifest.get("backend_effective") != "jsbsim":
        errors.append("requested JSBSim but effective backend is not JSBSim")
    if manifest.get("backend_requested") == "jsbsim" and manifest.get("backend_fallback_occurred", False):
        errors.append("strict JSBSim run recorded backend fallback")
    if method in {"h0_current_geometry", "h1_frozen_predicted_geometry"}:
        if manifest.get("training_started"):
            errors.append(f"{method} must not start training")
        if manifest.get("ppo_update_permitted"):
            errors.append(f"{method} must not permit PPO updates")
        if manifest.get("training_scope") != "none":
            errors.append(f"{method} training_scope must be none")
    if method == "h2_retrained_predicted_geometry":
        if manifest.get("training_scope") != "high_level_ppo_only":
            errors.append("H2 evaluation must record high_level_ppo_only scope")
        if not manifest.get("low_level_frozen", False):
            errors.append("H2 low-level policy is not marked frozen")
        if not manifest.get("low_level_execution_chain_frozen", False):
            errors.append("H2 execution chain is not marked frozen")
    if method != "h0_current_geometry":
        predictor = manifest.get("predictor_checkpoint") or {}
        _verify_file_receipt(predictor, "predictor checkpoint", errors)
        if not manifest.get("predictor_frozen", False):
            errors.append("predicted-geometry method does not mark predictor frozen")
    expected_replace = method != "h0_current_geometry"
    if bool(schema.get("replace_geometry_with_prediction", False)) != expected_replace:
        errors.append("observation replacement flag does not match method")
    if method == "h1_frozen_predicted_geometry" and manifest.get("training_started"):
        errors.append("H1 must not start training")
    scenario_counts: dict[str, int] = {}
    for episode in episodes:
        scenario = str(episode.get("scenario"))
        scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
    return {
        "path": str(run_dir),
        "ok": not errors,
        "errors": errors,
        "episode_count": len(episodes),
        "method": method,
        "opponent": opponent,
        "training_seed": manifest.get("training_seed"),
        "evaluation_seeds": manifest.get("evaluation_seeds", []),
        "scenario_counts": scenario_counts,
    }


def _verify_pilot_root(root: Path, source_id: str = SOURCE_ID) -> dict[str, Any]:
    """Verify the runner-level pilot receipt before it can authorize formal work."""
    errors: list[str] = []
    manifest_path = root / "run_manifest.json"
    status_path = root / "pilot_status.json"
    if not manifest_path.is_file():
        errors.append("pilot run_manifest.json is missing")
    else:
        manifest = _load(manifest_path)
        if manifest.get("source_id") != source_id:
            errors.append("pilot manifest has wrong source_id")
        if manifest.get("phase") != "pilot":
            errors.append("pilot manifest has wrong phase")
        if manifest.get("status") != "completed":
            errors.append("pilot runner did not complete")
        if not manifest.get("execution_started", False):
            errors.append("pilot execution was not recorded")
    if not status_path.is_file():
        errors.append("pilot_status.json is missing")
    else:
        status = _load(status_path)
        if status.get("source_id") != source_id:
            errors.append("pilot status has wrong source_id")
        if status.get("formal_authorized") is not False:
            errors.append("pilot status must remain unauthorized until independent verification")
        if status.get("status") != "pending_independent_verification":
            errors.append("pilot status is not pending independent verification")
    run_results = [
        _verify_run(path.parent, source_id)
        for path in root.rglob("metrics.json")
        if path.parent.name in EXPECTED_OPPONENTS
    ]
    if not run_results:
        errors.append("pilot has no method/opponent run artifacts")
    errors.extend(error for item in run_results for error in item.get("errors", []))
    return {
        "source_id": source_id,
        "phase": "pilot",
        "root": str(root),
        "ok": not errors,
        "errors": errors,
        "run_count": len(run_results),
        "runs": run_results,
    }


def verify(root: Path) -> dict[str, Any]:
    run_dirs = [path.parent for path in root.rglob("metrics.json") if path.parent.name in EXPECTED_OPPONENTS]
    source_id = SOURCE_ID
    for candidate in (root / "formal_manifest.json", root / "run_manifest.json"):
        if candidate.is_file():
            try:
                source_id = str(_load(candidate).get("source_id") or SOURCE_ID)
                break
            except (OSError, json.JSONDecodeError):
                pass
    results = [_verify_run(path, source_id) for path in sorted(run_dirs)]
    errors: list[str] = []
    if not results:
        errors.append("no method/opponent run artifacts found")
    root_manifest_path = root / "run_manifest.json"
    if root_manifest_path.is_file():
        root_manifest = _load(root_manifest_path)
        if root_manifest.get("phase") == "pilot":
            pilot_result = _verify_pilot_root(root, source_id)
            errors.extend(pilot_result["errors"])
    formal_manifest_path = root / "formal_manifest.json"
    if formal_manifest_path.is_file():
        formal_manifest = _load(formal_manifest_path)
        if formal_manifest.get("source_id") != source_id:
            errors.append("formal manifest has wrong source_id")
        if formal_manifest.get("phase") != "formal":
            errors.append("formal manifest has wrong phase")
        formal_methods = tuple(formal_manifest.get("methods", []))
        formal_opponents = tuple(formal_manifest.get("opponents", []))
        formal_scenarios = tuple(formal_manifest.get("scenarios", []))
        if formal_methods != EXPECTED_METHODS:
            errors.append("formal manifest methods are not exactly H0/H1/H2")
        if formal_opponents != EXPECTED_OPPONENTS:
            errors.append("formal manifest opponents are not exactly expert/end_to_end")
        if formal_scenarios != EXPECTED_SCENARIOS:
            errors.append("formal manifest scenarios are not the five locked scenarios")
        train_seeds = tuple(int(seed) for seed in formal_manifest.get("train_seeds", []))
        if len(train_seeds) < 3:
            errors.append("formal manifest has fewer than three training seeds")
        pilot_receipt_path = formal_manifest.get("pilot_verification_path")
        if not pilot_receipt_path:
            errors.append("formal manifest is missing pilot_verification_path")
        else:
            try:
                pilot_receipt = _load(Path(pilot_receipt_path))
                if pilot_receipt.get("source_id") != source_id:
                    errors.append("pilot verification receipt has wrong source_id")
                if pilot_receipt.get("ok") is not True:
                    errors.append("pilot verification receipt is not passing")
            except (OSError, json.JSONDecodeError):
                errors.append("pilot verification receipt is missing or invalid")
        expected_keys = {
            (method, seed, opponent)
            for method in formal_methods
            for seed in train_seeds
            for opponent in EXPECTED_OPPONENTS
        }
        found_keys = set()
        for item in results:
            try:
                found_keys.add(
                    (
                        item.get("method"),
                        int(item.get("training_seed")),
                        item.get("opponent"),
                    )
                )
            except (TypeError, ValueError):
                errors.append(f"formal run has invalid training seed: {item.get('path')}")
                continue
            expected_per_scenario = int(formal_manifest.get("episodes_per_scenario", 0)) * len(
                item.get("evaluation_seeds", [])
            )
            for scenario in EXPECTED_SCENARIOS:
                actual = item.get("scenario_counts", {}).get(scenario, 0)
                if actual != expected_per_scenario:
                    errors.append(
                        f"formal run {item['path']} has invalid {scenario} count: "
                        f"{actual} != {expected_per_scenario}"
                    )
        if found_keys != expected_keys:
            errors.append("formal artifacts do not exactly contain all method x seed x opponent runs")
        if (root / "formal_analysis_manifest.json").is_file() and not (
            root / "metrics_per_seed.csv"
        ).is_file():
            errors.append("formal analysis manifest exists but metrics_per_seed.csv is missing")
    return {
        "source_id": source_id,
        "root": str(root),
        "run_count": len(results),
        "passed_run_count": sum(item["ok"] for item in results),
        "ok": not errors and all(item["ok"] for item in results),
        "errors": errors,
        "runs": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    result = verify(args.root.resolve())
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
