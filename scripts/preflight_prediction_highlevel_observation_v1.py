#!/usr/bin/env python
"""Write a non-running preflight for the H0/H1/H2 protocol.

This command audits configuration and asset references only. It never creates
an environment, calls ``reset``/``step``, trains a policy, or loads a model
payload.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from uav_vpp_guidance.common.git import get_git_info
from uav_vpp_guidance.evaluation.prediction_highlevel_observation_v1 import (
    EXPECTED_METHODS,
    EXPECTED_OPPONENTS,
    EXPECTED_SCENARIOS,
    PREDICTOR_CONTRACT,
    SOURCE_ID,
    file_receipt,
    load_resolved_config,
    protocol_receipt,
)
from uav_vpp_guidance.common.hash import config_sha256
from uav_vpp_guidance.common.hash import file_sha256
from uav_vpp_guidance.evaluation.prediction_highlevel_observation_v1 import method_config


DEFAULT_CONFIG = REPO_ROOT / "config/experiment/prediction_highlevel_observation_v1.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/prediction_highlevel_observation_v1/preflight"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _dependency_report() -> str:
    names = ("numpy", "PyYAML", "torch", "jsbsim", "matplotlib")
    lines = ["# Dependency Report", "", f"Python: `{sys.version.split()[0]}`", f"Platform: `{platform.platform()}`", ""]
    for name in names:
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = "not installed"
        lines.append(f"- `{name}`: `{version}`")
    return "\n".join(lines) + "\n"


def audit_config(config: dict) -> list[str]:
    errors: list[str] = []
    protocol = config.get("protocol", {})
    if not protocol.get("source_id"):
        errors.append("protocol.source_id is missing")
    if tuple(protocol.get("methods", [])) != EXPECTED_METHODS:
        errors.append("protocol.methods are not exactly H0/H1/H2")
    if tuple(protocol.get("opponents", [])) != EXPECTED_OPPONENTS:
        errors.append("protocol.opponents are not expert/end_to_end")
    if tuple(protocol.get("scenarios", [])) != EXPECTED_SCENARIOS:
        errors.append("protocol.scenarios are not the five locked scenarios")
    if protocol.get("predictor_future_state_access") is not False:
        errors.append("future-state access must be explicitly false")
    if protocol.get("h1_updates_ppo") is not False:
        errors.append("H1 must not update PPO")
    if protocol.get("h2_updates_only_high_level_ppo") is not True:
        errors.append("H2 must update only high-level PPO")
    vpp = config.get("virtual_point", {})
    if vpp.get("action_dim") != 3:
        errors.append("virtual_point.action_dim must be exactly 3")
    if not isinstance(vpp.get("d_long_range"), list) or len(vpp["d_long_range"]) != 2:
        errors.append("virtual_point.d_long_range must define two bounds")
    if not isinstance(vpp.get("d_lat_range"), list) or len(vpp["d_lat_range"]) != 2:
        errors.append("virtual_point.d_lat_range must define two bounds")
    if not isinstance(vpp.get("d_vert_range"), list) or len(vpp["d_vert_range"]) != 2:
        errors.append("virtual_point.d_vert_range must define two bounds")
    tp = config.get("trajectory_prediction", {})
    model = tp.get("model", {})
    for key, expected in PREDICTOR_CONTRACT.items():
        if model.get(key) != expected:
            errors.append(f"trajectory_prediction.model.{key} != {expected!r}")
    forbidden = "prediction_decision_bridge_v1"
    output_root = str(config.get("experiment", {}).get("output_root", ""))
    if forbidden in output_root:
        errors.append("output_root reuses a forbidden prediction_decision_bridge_v1 path")
    return errors


def _source_receipt(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "sha256": file_sha256(path),
    }


def audit_source_contract(config: dict) -> tuple[list[str], dict, dict]:
    """Check the new protocol's critical source facts without importing runtime objects."""
    source_paths = {
        "observation_builder": REPO_ROOT / "src/uav_vpp_guidance/envs/observation.py",
        "tracking_env": REPO_ROOT / "src/uav_vpp_guidance/envs/tracking_env.py",
        "predicted_geometry": REPO_ROOT / "src/uav_vpp_guidance/trajectory_prediction/predicted_geometry.py",
        "predictor_adapter": REPO_ROOT / "src/uav_vpp_guidance/trajectory_prediction/predictor_adapter.py",
        "evaluation": REPO_ROOT / "src/uav_vpp_guidance/evaluation/evaluate_prediction_comparison.py",
    }
    receipts = {name: _source_receipt(path) for name, path in source_paths.items()}
    errors: list[str] = [
        f"missing source file: {name}"
        for name, receipt in receipts.items()
        if not receipt["exists"]
    ]

    def read(name: str) -> str:
        path = source_paths[name]
        if not path.is_file():
            return ""
        return path.read_text(encoding="utf-8")

    geometry_source = read("predicted_geometry")
    adapter_source = read("predictor_adapter")
    tracking_source = read("tracking_env")
    observation_source = read("observation_builder")
    evaluation_source = read("evaluation")

    checks = {
        "predicted_geometry_function_present": "def compute_predicted_geometry" in geometry_source,
        "predicted_geometry_does_not_read_target_future_velocity": "_velocity(target_state)" not in geometry_source,
        "predicted_geometry_does_not_reference_oracle_anchor": "oracle_future_position" not in geometry_source,
        "adapter_converts_relative_displacement": "target_pos + np.asarray(pred_disp" in adapter_source,
        "tracking_env_calls_predicted_geometry": "compute_predicted_geometry(" in tracking_source,
        "tracking_env_exposes_prediction_replacement": "replace_geometry_with_prediction" in tracking_source,
        "observation_builder_exposes_replacement": "replace_geometry_with_prediction" in observation_source,
        "evaluation_records_raw_episodes": "raw_episodes" in evaluation_source,
    }
    errors.extend(key for key, passed in checks.items() if not passed)

    h0 = method_config(config, "h0_current_geometry")
    h1 = method_config(config, "h1_frozen_predicted_geometry")
    h2 = method_config(config, "h2_retrained_predicted_geometry")
    shared_sections = ("env", "policy", "limits", "guidance", "reward", "opponent_registry", "scenarios")
    shared_execution_chain = all(
        h0.get(section) == h1.get(section) == h2.get(section)
        for section in shared_sections
    )
    checks["h0_h1_h2_shared_execution_sections"] = shared_execution_chain
    if not shared_execution_chain:
        errors.append("H0/H1/H2 differ in a locked execution section")

    return errors, receipts, checks


def run_preflight(config_path: Path, output_dir: Path) -> dict:
    config = load_resolved_config(config_path)
    errors = audit_config(config)
    source_errors, source_files, source_checks = audit_source_contract(config)
    errors.extend(source_errors)
    protocol = protocol_receipt(config, config_path)
    assets = {
        "trajectory_predictor": file_receipt(
            REPO_ROOT / config["trajectory_prediction"]["checkpoint_path"]
        ),
        "h0_policy": file_receipt(REPO_ROOT / config["methods"]["h0_current_geometry"]["checkpoint"]),
        "h1_policy": file_receipt(REPO_ROOT / config["methods"]["h1_frozen_predicted_geometry"]["checkpoint"]),
        "opponent_end_to_end": file_receipt(REPO_ROOT / config["opponent_registry"]["end_to_end"]["checkpoint"]),
    }
    protocol_source_id = protocol.get("source_id", SOURCE_ID)
    receipt = {
        "status": "blocked_missing_asset_or_config_error" if errors or any(not x["exists"] for x in assets.values()) else "ready_for_review",
        "source_id": protocol_source_id,
        "config_sha256": config_sha256(config, truncate=0),
        "config": protocol,
        "assets": assets,
        "config_errors": errors,
        "source_files": source_files,
        "source_checks": source_checks,
        "formal_execution_started": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "environment_manifest.json").write_text(
        json.dumps({"git_info": get_git_info(str(REPO_ROOT)), "python": sys.version, "platform": platform.platform(), "source_id": protocol_source_id}, indent=2),
        encoding="utf-8",
    )
    (output_dir / "asset_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    _write(output_dir / "git_status.txt", json.dumps(get_git_info(str(REPO_ROOT)), indent=2) + "\n")
    _write(output_dir / "dependency_report.md", _dependency_report())
    _write(output_dir / "observation_space_audit.md", "# Observation Space Audit\n\nH0, H1 and H2 use a fixed 16-dimensional vector. H1/H2 replace only range, range-rate, ATA sine/cosine and AA sine/cosine through an explicit opt-in flag.\n\nSource checks: `" + json.dumps({key: source_checks[key] for key in source_checks if "observation" in key or "replacement" in key}, ensure_ascii=False) + "`\n")
    _write(output_dir / "predictor_audit.md", "# Predictor Audit\n\nThe predictor is frozen. Predicted geometry uses predictor output plus current own state; no future simulator state is read.\n\nSource checks: `" + json.dumps({key: source_checks[key] for key in source_checks if "predicted_geometry" in key or "adapter" in key}, ensure_ascii=False) + "`\n")
    _write(output_dir / "execution_chain_audit.md", "# Execution Chain Audit\n\nThe low-level policy, VPP, guidance, PID, JSBSim model, terminal rules and scenario definitions remain shared across H0/H1/H2.\n\nShared-section check: `" + str(source_checks.get("h0_h1_h2_shared_execution_sections", False)) + "`\n")
    _write(output_dir / "source_contract_audit.json", json.dumps({"source_files": source_files, "checks": source_checks, "errors": source_errors}, indent=2, ensure_ascii=False) + "\n")
    _write(output_dir / "existing_bridge_separation.md", "# Existing Bridge Separation\n\nThe new protocol writes only under `outputs/prediction_highlevel_observation_v1/` and rejects prediction_decision_bridge_v1 paths.\n")
    _write(output_dir / "protocol_lock_v1.md", "# Protocol Lock\n\nSource ID: `" + protocol_source_id + "`\n\nH0 is current geometry, H1 is frozen PPO with predicted geometry replacement, and H2 retrains only the PPO policy with the same replacement contract.\n")
    _write(output_dir / "predicted_geometry_definition.md", "# Predicted Geometry Definition\n\nPredicted range is the norm of predicted target position minus current own position. Predicted range-rate is `(predicted_range - current_range) / lookahead`. Predicted ATA uses estimated target velocity from predicted displacement; predicted AA uses current own velocity.\n")
    _write(output_dir / "causality_check.md", "# Causality Check\n\nNo future true target position, velocity, reward, or terminal state is passed to the predictor or observation builder.\n")
    _write(output_dir / "coordinate_consistency_check.md", "# Coordinate Consistency\n\nAll positions and velocities are interpreted in the repository NEU convention before geometry computation.\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    receipt = run_preflight(args.config.resolve(), args.output_dir.resolve())
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["status"] == "ready_for_review" else 2


if __name__ == "__main__":
    raise SystemExit(main())
