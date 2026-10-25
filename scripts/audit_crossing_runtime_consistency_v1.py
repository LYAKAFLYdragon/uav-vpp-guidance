#!/usr/bin/env python
"""Read-only audit of crossing checkpoint semantics versus recovery runtime config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_CONFIG = Path(
    "E:/uav-vpp-guidance-artifacts/CAN-20260705/assets/outputs/experiments/"
    "prediction_vpp_ppo_crossing_v3/config_snapshot.yaml"
)
DEFAULT_CHECKPOINT = Path(
    "E:/uav-vpp-guidance-artifacts/CAN-20260705/assets/outputs/experiments/"
    "prediction_vpp_ppo_crossing_v3/checkpoints/best.pt"
)
DEFAULT_RUNTIME_CONFIG = REPO_ROOT / (
    "outputs/mild_disadvantage_envelope_recovery_v1/physical_probe_v1/"
    "expert/resolved_config.yaml"
)
DEFAULT_OUTPUT = REPO_ROOT / "reports/crossing_runtime_consistency_audit_v1.json"


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _get(mapping: dict[str, Any], path: str, default: Any = None) -> Any:
    value: Any = mapping
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _check(name: str, train: Any, runtime: Any, status: str, note: str) -> dict[str, Any]:
    return {"name": name, "training": train, "runtime": runtime, "status": status, "note": note}


def audit(checkpoint_config: Path, checkpoint: Path, runtime_config: Path) -> dict[str, Any]:
    train = _load(checkpoint_config)
    runtime = _load(runtime_config)
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    train_vpp = train.get("virtual_point", {})
    runtime_vpp = runtime.get("virtual_point", {})
    train_pred = train.get("trajectory_prediction", {})
    runtime_pred = runtime.get("trajectory_prediction", {})
    train_env = train.get("env", {})
    runtime_env = runtime.get("env", {})

    checks = [
        _check(
            "checkpoint_obs_dim",
            payload.get("obs_dim"),
            runtime.get("observation", {}).get("include_opponent_stage"),
            "pass" if payload.get("obs_dim") == 18 and runtime.get("observation", {}).get("include_opponent_stage") is True else "mismatch",
            "runtime observation is 16 base features plus 2 opponent-stage flags",
        ),
        _check(
            "checkpoint_action_dim",
            payload.get("action_dim"),
            _get(runtime, "policy.action_dim"),
            "pass" if payload.get("action_dim") == _get(runtime, "policy.action_dim") == 3 else "mismatch",
            "three-dimensional normalized VPP action shape is aligned",
        ),
        _check(
            "action_semantics",
            train_vpp.get("action_semantics"),
            runtime_vpp.get("action_semantics"),
            "mismatch" if train_vpp.get("action_semantics") != runtime_vpp.get("action_semantics") else "pass",
            "the learned action coordinates depend on this contract",
        ),
        _check(
            "anchor_mode",
            train_vpp.get("anchor_mode"),
            runtime_vpp.get("anchor_mode"),
            "mismatch" if train_vpp.get("anchor_mode") != runtime_vpp.get("anchor_mode") else "pass",
            "training used predicted target; probe used current target",
        ),
        _check(
            "trajectory_prediction_enabled",
            train_pred.get("enabled"),
            runtime_pred.get("enabled"),
            "mismatch" if train_pred.get("enabled") != runtime_pred.get("enabled") else "pass",
            "the probe disabled the predictor used by the crossing training lane",
        ),
        _check(
            "prediction_anchor_mode",
            _get(train_pred, "integration.anchor_mode"),
            _get(runtime_pred, "integration.anchor_mode"),
            "mismatch" if _get(train_pred, "integration.anchor_mode") != _get(runtime_pred, "integration.anchor_mode") else "pass",
            "prediction anchor affects the VPP reference even when the policy input stays 18-D",
        ),
        _check(
            "task_context",
            train.get("task"),
            runtime.get("task"),
            "mismatch" if (
                runtime.get("task") is None
                and any(key in train_vpp for key in (
                    "offset_frame_by_task",
                    "close_range_anchor_mode_by_task",
                    "post_merge_predicted_target_blend_by_task",
                ))
            ) else "pass",
            "runtime has no explicit task context for task-conditioned tactical-basis rules",
        ),
        _check(
            "offset_frame",
            _get(train_vpp, "offset_frame_by_task.crossing_feasible"),
            _get(runtime_vpp, "offset_frame_by_task.crossing_feasible"),
            "mismatch" if _get(train_vpp, "offset_frame_by_task.crossing_feasible") != _get(runtime_vpp, "offset_frame_by_task.crossing_feasible") else "pass",
            "crossing training used an explicit world/encounter frame mapping",
        ),
        _check(
            "tactical_basis_parameters",
            {
                "dynamics_aware": train_vpp.get("dynamics_aware"),
                "longitudinal_frame": train_vpp.get("tactical_basis_longitudinal_frame"),
                "lateral_frame": train_vpp.get("tactical_basis_lateral_frame"),
                "vertical_frame": train_vpp.get("tactical_basis_vertical_frame"),
            },
            {
                "dynamics_aware": runtime_vpp.get("dynamics_aware"),
                "longitudinal_frame": runtime_vpp.get("tactical_basis_longitudinal_frame"),
                "lateral_frame": runtime_vpp.get("tactical_basis_lateral_frame"),
                "vertical_frame": runtime_vpp.get("tactical_basis_vertical_frame"),
            },
            "mismatch" if any(
                train_vpp.get(key) != runtime_vpp.get(key)
                for key in ("dynamics_aware", "tactical_basis_longitudinal_frame", "tactical_basis_lateral_frame", "tactical_basis_vertical_frame")
            ) else "pass",
            "runtime omitted the trained tactical-basis compiler settings",
        ),
        _check(
            "smoothing_alpha",
            train_vpp.get("smoothing_alpha"),
            runtime_vpp.get("smoothing_alpha"),
            "pass" if train_vpp.get("smoothing_alpha") == runtime_vpp.get("smoothing_alpha") == 0.3 else "mismatch",
            "this shared filter value is aligned",
        ),
        _check(
            "simulation_timing",
            {key: train_env.get(key) for key in ("sim_freq", "decision_freq", "high_level_dt", "action_repeat")},
            {key: runtime_env.get(key) for key in ("sim_freq", "decision_freq", "high_level_dt", "action_repeat")},
            "pass" if all(train_env.get(key) == runtime_env.get(key) for key in ("sim_freq", "decision_freq", "high_level_dt", "action_repeat")) else "mismatch",
            "integration timing is aligned",
        ),
        _check(
            "action_ranges",
            {key: train_vpp.get(key) for key in ("d_long_range", "d_lat_range", "d_vert_range")},
            {key: runtime_vpp.get(key) for key in ("d_long_range", "d_lat_range", "d_vert_range")},
            "pass" if all(train_vpp.get(key) == runtime_vpp.get(key) for key in ("d_long_range", "d_lat_range", "d_vert_range")) else "mismatch",
            "physical VPP ranges are aligned",
        ),
    ]
    mismatches = [item["name"] for item in checks if item["status"] == "mismatch"]
    return {
        "audit_id": "CROSSING-RUNTIME-CONSISTENCY-AUDIT-V1-20260903",
        "checkpoint_config": str(checkpoint_config.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "runtime_config": str(runtime_config.resolve()),
        "checkpoint_payload": {
            "obs_dim": payload.get("obs_dim"),
            "action_dim": payload.get("action_dim"),
            "total_timesteps": payload.get("total_timesteps"),
        },
        "checks": checks,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "decision": "not_comparable_as_trained_crossing_semantics" if mismatches else "comparable",
        "implication": (
            "The 12-episode recovery result is valid for the implemented runtime tactic, "
            "but it cannot reject crossing specialist capability under its trained tactical_basis_v1/predicted_target contract."
            if mismatches
            else "The runtime matches the crossing training contract."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-config", type=Path, default=DEFAULT_CHECKPOINT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--runtime-config", type=Path, default=DEFAULT_RUNTIME_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = audit(args.checkpoint_config.resolve(), args.checkpoint.resolve(), args.runtime_config.resolve())
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
