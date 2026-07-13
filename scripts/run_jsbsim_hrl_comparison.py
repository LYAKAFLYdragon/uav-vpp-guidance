#!/usr/bin/env python3
"""Run the minimal JSBSim HRL/VPP comparison matrix.

This runner is intentionally research-facing rather than controller-facing:

* main preset: prediction VPP vs no-prediction VPP vs end-to-end RL
* ablation preset: perfect vs learned vs no vs noisy prediction
* frozen task set: head-on, feasible crossing, break-turn, sustained-turn

It writes a portable manifest, artifact contract, resolved config snapshot, raw
episode JSONs, and a compact summary CSV.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import yaml

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from uav_vpp_guidance.agents.end_to_end_ppo_agent import EndToEndPPOAgent
from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.common.artifact_contract import ArtifactContract
from uav_vpp_guidance.common.hash import config_sha256
from uav_vpp_guidance.common.manifest import RunManifest
from uav_vpp_guidance.common.provenance import (
    get_config_overrides,
    record_config_override_if_changed,
)
from uav_vpp_guidance.evaluation.recorders import EpisodeRecorder, RunRecorder
from uav_vpp_guidance.metrics.combat_evaluator import compute_combat_metrics
from uav_vpp_guidance.utils.config import load_yaml_config, merge_config
from uav_vpp_guidance.virtual_point.coordinate_transform import VALID_OFFSET_FRAMES
from uav_vpp_guidance.virtual_point.generator import (
    VALID_OFFENSIVE_ANCHOR_LATERAL_SIGN_MODES,
)


SCRIPT_NAME = Path(__file__).name
STAGE_NAME = "jsbsim_hrl_comparison"
STAGE_VERSION = "0.1.0"

TASK_METRIC_POLICY = {
    "head_on": {
        "primary_success_metric": "success_rate",
        "success_semantics": "Default proximity success from CloseRangeTrackingEnv.",
        "tracking_metrics": ["final_range_m", "mean_range_m", "time_in_envelope_s"],
    },
    "crossing_feasible": {
        "primary_success_metric": "success_rate",
        "success_semantics": "Default proximity success from CloseRangeTrackingEnv.",
        "tracking_metrics": ["final_range_m", "mean_range_m", "time_in_envelope_s"],
    },
    "break_turn": {
        "primary_success_metric": None,
        "success_semantics": "BreakTurnEnv disables proximity success; success_rate is not a valid primary metric.",
        "tracking_metrics": ["final_range_m", "mean_range_m", "min_range_m", "time_in_envelope_s"],
    },
    "sustained_turn": {
        "primary_success_metric": None,
        "success_semantics": "SustainedTurnEnv disables proximity success; success_rate is not a valid primary metric.",
        "tracking_metrics": ["completed_orbits", "mean_range_m", "time_in_envelope_s", "mean_abs_ata_deg"],
    },
}

PREDICTION_VARIANT_SEMANTICS = {
    "none": "No trajectory prediction; VPP anchors on the current target state.",
    "learned": "Frozen learned predictor supplies the predicted target anchor for VPP; prediction features are not added to policy observation by default.",
    "noisy": "Learned prediction anchor with seeded Gaussian position noise injected before VPP generation.",
    "perfect": "Constant-velocity kinematic oracle using the current true target velocity; this is not a true future-trajectory oracle for maneuvering targets.",
}

DESIGN_NOTES = {
    "main_claim_scope": "Prediction assists VPP anchor placement. Current main preset does not add prediction features to the high-level policy observation.",
    "prediction_features_in_observation": False,
    "task_metric_policy": TASK_METRIC_POLICY,
    "prediction_variant_semantics": PREDICTION_VARIANT_SEMANTICS,
    "formal_policy": "formal-small runs are pilot evidence; paper-safe formal evidence requires a separately chosen full budget after pilot analysis.",
    "curriculum_opponent_eval_policy": "When --opponent-stage=curriculum, this runner evaluates a single resolved opponent. The YAML active_stage selects it; if omitted, the final listed stage is used.",
    "survival_rate_semantics": "survival_rate means the ego aircraft was not physically destroyed, shot down, crashed, or out-of-bounds; HP-timeout losses can still count as survived.",
    "attack_zone_angle_policy": "close_range_max_aoa_deg is the preferred explicit close-range attack-zone angle knob; close_range_max_aoa_rad remains supported for legacy configs.",
}

COMBAT_GEOMETRY_DIAGNOSTIC_METRICS = (
    "first_ego_attack_time_s",
    "first_target_attack_time_s",
    "first_pass_step",
    "post_merge_attack_zone_advantage_s",
    "close_range_anchor_mode_active_fraction",
    "close_range_anchor_mode_first_active_step",
    "post_merge_anchor_mode_active_fraction",
    "post_merge_anchor_mode_condition_met_fraction",
    "post_merge_anchor_mode_first_active_step",
    "post_merge_anchor_mode_lateral_world_offset_latch_active_fraction",
    "post_merge_anchor_mode_lateral_world_offset_latch_first_active_step",
    "post_merge_anchor_mode_recovery_active_fraction",
    "post_merge_anchor_mode_recovery_first_step",
    "post_merge_anchor_mode_released_fraction",
    "post_merge_anchor_mode_release_first_step",
    "post_merge_offensive_anchor_condition_met_fraction",
    "post_merge_offensive_anchor_alignment_disadvantage_fraction",
    "post_merge_offensive_anchor_blend_active_fraction",
    "post_merge_offensive_anchor_first_active_step",
    "post_merge_offensive_anchor_blend_release_blend_active_fraction",
    "post_merge_offensive_anchor_blend_released_fraction",
    "post_merge_offensive_anchor_blend_release_first_step",
    "post_merge_offensive_anchor_blend_release_direct_track_active_fraction",
    "post_merge_offensive_anchor_blend_release_direct_track_first_step",
    "post_merge_offensive_anchor_blend_release_recovery_active_fraction",
    "post_merge_offensive_anchor_blend_release_recovery_first_step",
    "post_merge_offensive_anchor_blend_release_recovery_altitude_trigger_active_fraction",
    "post_merge_offensive_anchor_blend_release_recovery_altitude_trigger_first_step",
    "post_merge_offensive_anchor_blend_release_recovery_forward_bias_trigger_active_fraction",
    "post_merge_offensive_anchor_blend_release_recovery_forward_bias_trigger_first_step",
    "post_merge_offensive_anchor_blend_release_vp_forward_bias_clamp_active_fraction",
    "post_merge_offensive_anchor_blend_release_vp_forward_bias_clamp_first_step",
    "post_merge_offensive_anchor_lateral_world_offset_latch_active_fraction",
    "post_merge_offensive_anchor_lateral_world_offset_latch_first_active_step",
    "post_merge_offensive_anchor_blend_release_lateral_only_active_fraction",
    "post_merge_offensive_anchor_blend_release_lateral_only_first_step",
    "post_merge_tactical_basis_recovery_profile_active_fraction",
    "post_merge_tactical_basis_recovery_profile_first_step",
    "post_merge_predicted_target_forward_scale_active_fraction",
    "post_merge_predicted_target_forward_scale_first_active_step",
    "post_merge_predicted_target_forward_scale_release_scale_active_fraction",
    "post_merge_predicted_target_forward_scale_released_fraction",
    "post_merge_predicted_target_forward_scale_release_first_step",
    "post_merge_offensive_anchor_active_longitudinal_scale",
    "post_merge_offensive_anchor_active_lateral_scale",
    "post_merge_active_vp_longitudinal_scale",
    "post_merge_active_vp_lateral_scale",
    "post_merge_offensive_anchor_active_override_longitudinal_scale",
    "post_merge_offensive_anchor_active_override_lateral_scale",
    "pre_merge_vp_forward_bias_m",
    "post_merge_anchor_mode_active_vp_forward_bias_m",
    "post_merge_anchor_mode_active_vp_lateral_bias_m",
    "post_merge_offensive_anchor_active_vp_forward_bias_m",
    "post_merge_offensive_anchor_active_vp_lateral_bias_m",
    "post_merge_predicted_target_forward_scale_active_vp_forward_bias_m",
    "post_merge_predicted_target_forward_scale_active_vp_lateral_bias_m",
    "pre_merge_mean_tactical_basis_action_ll",
    "pre_merge_mean_tactical_basis_action_io",
    "pre_merge_mean_tactical_basis_action_cd",
    "pre_merge_positive_tactical_basis_io_fraction",
    "post_merge_mean_tactical_basis_action_ll",
    "post_merge_mean_tactical_basis_action_io",
    "vp_forward_bias_m",
    "vp_lateral_bias_m",
    "merge_min_range_m",
    "damage_margin",
)

VALID_VPP_ANCHOR_MODES = (
    "current_target",
    "constant_velocity",
    "oracle_future_position",
    "rule_based_pursuit",
    "predicted_target",
    "offensive_position",
)


def _load_config_with_includes(config_path: Path) -> Dict[str, Any]:
    config = load_yaml_config(str(config_path))
    includes = config.pop("includes", [])
    merged: Dict[str, Any] = {}
    for inc in includes:
        inc_path = (config_path.parent / inc).resolve()
        if not inc_path.exists():
            inc_path = (config_path.parent / ".." / Path(inc).name).resolve()
        if not inc_path.exists():
            raise FileNotFoundError(f"Included config not found: {inc}")
        merged = merge_config(merged, _load_config_with_includes(inc_path))
    return merge_config(merged, config)


def _deep_update(target: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(target)
    for key, value in source.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _get_path(config: Dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    node: Any = config
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def _set_path(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    node = config
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = copy.deepcopy(value)


def _record_set(
    config: Dict[str, Any],
    dotted_key: str,
    value: Any,
    source: str,
) -> None:
    old_value = _get_path(config, dotted_key)
    new_value = value
    # CLI task-scoped overrides should patch the loaded YAML mapping rather than
    # replace it wholesale, so head_on-only edits preserve baseline crossing
    # entries and other task defaults.
    if dotted_key.endswith("_by_task") and isinstance(value, dict):
        if isinstance(old_value, dict):
            new_value = _deep_update(copy.deepcopy(old_value), value)
        else:
            new_value = copy.deepcopy(value)
    _set_path(config, dotted_key, new_value)
    record_config_override_if_changed(
        config,
        key=dotted_key,
        new_value=new_value,
        old_value=old_value,
        source=source,
    )


def _record_block(
    config: Dict[str, Any],
    dotted_key: str,
    value: Dict[str, Any],
    source: str,
) -> None:
    old_value = _get_path(config, dotted_key)
    merged = _deep_update(old_value if isinstance(old_value, dict) else {}, value)
    _set_path(config, dotted_key, merged)
    record_config_override_if_changed(
        config,
        key=dotted_key,
        new_value=merged,
        old_value=old_value,
        source=source,
    )


def _repo_path(path_value: Optional[str]) -> Optional[Path]:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _resolve_task_def(task_def: Dict[str, Any]) -> Dict[str, Any]:
    task_def = copy.deepcopy(task_def)
    task_config = task_def.get("config_path")
    if not task_config:
        return task_def

    config_path = _repo_path(str(task_config))
    if config_path is None:
        raise RuntimeError("Task config path is required")
    if not config_path.exists():
        raise FileNotFoundError(f"Task config not found: {config_path}")

    loaded = _load_config_with_includes(config_path)
    inline_overrides = {
        key: value
        for key, value in task_def.items()
        if key not in {"config_path", "label"}
    }
    resolved = merge_config(loaded, inline_overrides)
    if "label" in task_def:
        resolved["label"] = task_def["label"]
    resolved["config_path"] = str(task_config)
    return resolved


def _scenario_manifest_payload_sha256(manifest: Dict[str, Any]) -> str:
    """Return the builder-compatible hash while excluding its self-reference."""

    payload = copy.deepcopy(manifest)
    integrity = payload.get("integrity")
    if isinstance(integrity, dict):
        integrity.pop("payload_sha256", None)
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _materialize_scenario_manifest(
    comparison_config: Dict[str, Any],
    comparison_config_path: Path,
) -> Dict[str, Any]:
    """Expand a frozen scenario manifest into task groups for the comparison runner.

    The taxonomy ablation stores all 30 physical scenarios in one immutable
    manifest. Grouping by the preregistered task registry key is necessary so
    the static oracle sees the intended legacy task label for every episode.
    """

    manifest_spec = comparison_config.get("scenario_manifest")
    if not isinstance(manifest_spec, dict):
        return comparison_config

    raw_path = manifest_spec.get("path")
    if not raw_path:
        raise ValueError("scenario_manifest.path is required")
    manifest_path = Path(str(raw_path))
    if not manifest_path.is_absolute():
        manifest_path = (comparison_config_path.parent / manifest_path).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Scenario manifest not found: {manifest_path}")

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    expected_source_id = manifest_spec.get("source_id")
    if expected_source_id and manifest.get("source_id") != expected_source_id:
        raise ValueError(
            "Scenario manifest source_id mismatch: "
            f"expected={expected_source_id}, actual={manifest.get('source_id')}"
        )
    expected_hash = manifest_spec.get("payload_sha256")
    actual_hash = _scenario_manifest_payload_sha256(manifest)
    if expected_hash and actual_hash != expected_hash:
        raise ValueError(
            "Scenario manifest payload hash mismatch: "
            f"expected={expected_hash}, actual={actual_hash}"
        )

    scenarios = manifest.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("Scenario manifest must contain a non-empty scenarios list")

    task_field = str(manifest_spec.get("task_key_field", "task_registry_key"))
    expected_task_counts = manifest_spec.get("expected_task_counts", {}) or {}
    active_task_groups = manifest_spec.get("active_task_groups")
    if active_task_groups is not None:
        active_task_groups = {str(task_name) for task_name in active_task_groups}
        if not active_task_groups:
            raise ValueError("scenario_manifest.active_task_groups must not be empty")
        unknown_task_groups = active_task_groups.difference(expected_task_counts)
        if unknown_task_groups:
            raise ValueError(
                "scenario_manifest.active_task_groups are not declared in "
                f"expected_task_counts: {sorted(unknown_task_groups)}"
            )
        expected_task_counts = {
            task_name: count
            for task_name, count in expected_task_counts.items()
            if str(task_name) in active_task_groups
        }
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            raise TypeError("Scenario manifest entries must be mappings")
        metadata = scenario.get("metadata", {})
        task_name = metadata.get(task_field) if isinstance(metadata, dict) else None
        if not task_name:
            raise ValueError(
                f"Scenario {scenario.get('name', '<unnamed>')} is missing metadata.{task_field}"
            )
        grouped.setdefault(str(task_name), []).append(copy.deepcopy(scenario))

    if set(grouped) != set(expected_task_counts):
        raise ValueError(
            "Scenario manifest task groups mismatch: "
            f"expected={sorted(expected_task_counts)}, actual={sorted(grouped)}"
        )
    for task_name, expected_count in expected_task_counts.items():
        actual_count = len(grouped.get(str(task_name), []))
        if actual_count != int(expected_count):
            raise ValueError(
                f"Scenario manifest count mismatch for {task_name}: "
                f"expected={expected_count}, actual={actual_count}"
            )

    materialized = copy.deepcopy(comparison_config)
    base_tasks = materialized.get("tasks", {})
    if not isinstance(base_tasks, dict):
        raise TypeError("Comparison config tasks must be a mapping")
    task_labels = manifest_spec.get("task_labels", {}) or {}
    resolved_tasks: Dict[str, Dict[str, Any]] = {}
    for task_name, task_scenarios in grouped.items():
        if task_name not in base_tasks:
            raise KeyError(
                f"Scenario manifest references task {task_name!r} absent from the comparison config"
            )
        task_def = copy.deepcopy(base_tasks[task_name])
        task_block = task_def.setdefault("task", {})
        task_block["name"] = task_name
        task_block["scenarios"] = task_scenarios
        if task_name in task_labels:
            task_def["label"] = str(task_labels[task_name])
        resolved_tasks[task_name] = task_def

    materialized["tasks"] = resolved_tasks
    materialized["scenario_manifest_resolution"] = {
        "path": str(manifest_path),
        "source_id": manifest.get("source_id"),
        "payload_sha256": actual_hash,
        "task_counts": {task_name: len(items) for task_name, items in grouped.items()},
        "use_scenario_seed": bool(manifest_spec.get("use_scenario_seed", False)),
    }
    return materialized


def _import_class(class_path: str):
    normalized = str(class_path)
    if normalized.startswith("src."):
        normalized = normalized[len("src.") :]
    module_name, class_name = normalized.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _apply_paper_safety_guards(
    manifest: RunManifest,
    *,
    run_status: str,
    backend: str,
    dry_run: bool,
) -> bool:
    """Apply reproducibility guards before computing the final paper-safe bit."""

    if manifest.git_info.get("dirty"):
        manifest.add_invalid_for_paper_reason(
            "git working tree is dirty; formal evidence must be generated from a clean tree"
        )

    return run_status == "formal" and backend == "jsbsim" and not dry_run


def _resolve_opponent_entry(
    comparison_config: Dict[str, Any],
    opponent_stage: str,
) -> Dict[str, Any]:
    stage = str(opponent_stage or "none")
    if stage == "none":
        return {"stage": "none", "type": "none"}
    registry = comparison_config.get("opponent_registry", {})
    if stage not in registry:
        raise KeyError(f"Unknown opponent_stage: {stage}")
    entry = copy.deepcopy(registry[stage])
    entry["stage"] = stage
    if entry.get("type") == "curriculum":
        stages = list(entry.get("stages", []))
        if not stages:
            raise ValueError("curriculum opponent requires at least one stage")
        resolved_stage = entry.get("active_stage") or stages[-1]
        resolved = _resolve_opponent_entry(comparison_config, resolved_stage)
        resolved["stage"] = stage
        resolved["resolved_stage"] = resolved_stage
        resolved["curriculum_config"] = entry
        return resolved
    return entry


def _apply_opponent_stage(
    config: Dict[str, Any],
    comparison_config: Dict[str, Any],
    opponent_stage: str,
) -> Dict[str, Any]:
    source = f"{SCRIPT_NAME}:opponent_stage"
    opponent_entry = _resolve_opponent_entry(comparison_config, opponent_stage)
    _record_set(config, "opponent_stage", opponent_stage, source)
    _record_set(config, "opponent", opponent_entry, source)
    if opponent_stage != "none":
        _record_set(config, "attack_zone.enabled", True, source)
    return opponent_entry


def _build_opponent_policy(config: Dict[str, Any]):
    opponent_entry = copy.deepcopy(config.get("opponent", {}))
    if not opponent_entry or opponent_entry.get("stage") == "none":
        return None

    class_path = opponent_entry.get("class")
    if not class_path:
        typ = opponent_entry.get("type")
        if typ == "rule_based":
            class_path = "uav_vpp_guidance.envs.expert_opponent.ExpertOpponent"
        elif typ == "neural":
            class_path = "uav_vpp_guidance.envs.end_to_end_opponent.EndToEndOpponent"
        else:
            raise ValueError(f"Opponent entry missing class: {opponent_entry}")

    cls = _import_class(class_path)
    checkpoint = opponent_entry.get("checkpoint")
    if checkpoint is not None:
        checkpoint_path = _repo_path(str(checkpoint))
        opponent_entry["checkpoint"] = str(checkpoint_path)
    else:
        checkpoint_path = None

    kwargs = {}
    if checkpoint_path is not None:
        kwargs["checkpoint_path"] = str(checkpoint_path)
    if "config" in opponent_entry:
        kwargs["config"] = opponent_entry.get("config")
    if "device" in opponent_entry:
        kwargs["device"] = opponent_entry.get("device")
    if "invert_observation" in opponent_entry:
        kwargs["invert_observation"] = opponent_entry.get("invert_observation")
    return cls(**kwargs)


def _load_checkpoint_config(
    checkpoint_path: Optional[Path],
    fallback_config_path: Optional[Path],
) -> Dict[str, Any]:
    if checkpoint_path is not None and checkpoint_path.exists():
        if checkpoint_path.suffix.lower() != ".zip":
            checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
            ckpt_config = checkpoint.get("config", {})
            if ckpt_config:
                return copy.deepcopy(ckpt_config)

    if fallback_config_path is not None and fallback_config_path.exists():
        return _load_config_with_includes(fallback_config_path)

    missing = checkpoint_path if checkpoint_path is not None else fallback_config_path
    raise FileNotFoundError(f"No checkpoint config or fallback config found: {missing}")


def _build_env(config: Dict[str, Any]):
    opponent_policy = _build_opponent_policy(config)
    opponent_config = copy.deepcopy(config.get("opponent", {}))
    class_name = config.get("task", {}).get("env_class", "CloseRangeTrackingEnv")
    if class_name == "SustainedTurnEnv":
        from uav_vpp_guidance.envs.sustained_turn_env import SustainedTurnEnv

        return SustainedTurnEnv(
            config,
            opponent_policy=opponent_policy,
            opponent_config=opponent_config,
        )
    if class_name == "BreakTurnEnv":
        from uav_vpp_guidance.envs.break_turn_env import BreakTurnEnv

        return BreakTurnEnv(
            config,
            opponent_policy=opponent_policy,
            opponent_config=opponent_config,
        )
    if class_name == "MultiWaypointTrackingEnv":
        from uav_vpp_guidance.envs.multi_waypoint_tracking_env import (
            MultiWaypointTrackingEnv,
        )

        return MultiWaypointTrackingEnv(
            config,
            opponent_policy=opponent_policy,
            opponent_config=opponent_config,
        )

    from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv

    return CloseRangeTrackingEnv(
        config,
        opponent_policy=opponent_policy,
        opponent_config=opponent_config,
    )


def _apply_backend(config: Dict[str, Any], backend: str, source: str) -> None:
    backend = str(backend).lower()
    use_jsbsim = backend == "jsbsim"
    _record_set(config, "backend", backend, source)
    _record_set(config, "env.backend", backend, source)
    _record_set(config, "env.use_jsbsim", use_jsbsim, source)
    _record_set(config, "env.strict_backend", use_jsbsim, source)


def _apply_task_and_common_overrides(
    config: Dict[str, Any],
    comparison_config: Dict[str, Any],
    task_name: str,
    task_def: Dict[str, Any],
    backend: str,
    max_steps_override: Optional[int],
    jsbsim_root: Optional[str],
) -> None:
    source = f"{SCRIPT_NAME}:{task_name}_eval_config"
    common_env = comparison_config.get("env", {})
    if common_env:
        _record_block(config, "env", common_env, source)
    if task_def.get("env"):
        _record_block(config, "env", task_def["env"], source)
    if comparison_config.get("low_level_controller"):
        _record_set(
            config,
            "low_level_controller",
            comparison_config["low_level_controller"],
            source,
        )
    if task_def.get("low_level_controller"):
        _record_block(config, "low_level_controller", task_def["low_level_controller"], source)
    if comparison_config.get("guidance"):
        _record_block(config, "guidance", comparison_config["guidance"], source)
    if task_def.get("guidance"):
        _record_block(config, "guidance", task_def["guidance"], source)
    for block_name in ("limits", "reward", "observation", "attack_zone"):
        if comparison_config.get(block_name):
            _record_block(config, block_name, comparison_config[block_name], source)
        if task_def.get(block_name):
            _record_block(config, block_name, task_def[block_name], source)
    _record_set(config, "task", task_def["task"], source)
    if max_steps_override is not None:
        _record_set(config, "env.max_high_level_steps", int(max_steps_override), source)
    if jsbsim_root and str(backend).lower() == "jsbsim":
        _record_set(config, "env.legacy_project_root", jsbsim_root, source)
    _apply_backend(config, backend, source)


def _apply_attack_zone_cli_overrides(
    config: Dict[str, Any],
    overrides: Dict[str, Any],
) -> None:
    if not overrides:
        return
    source = f"{SCRIPT_NAME}:attack_zone_cli"
    for key, value in sorted(overrides.items()):
        _record_set(config, f"attack_zone.{key}", value, source)


def _apply_prediction_cli_overrides(
    config: Dict[str, Any],
    method_def: Dict[str, Any],
    overrides: Dict[str, Any],
) -> None:
    if not overrides:
        return
    if method_def.get("agent_type") == "end_to_end":
        return
    if str(method_def.get("prediction_variant", "none")).lower() == "none":
        return
    source = f"{SCRIPT_NAME}:prediction_cli"
    if "lookahead_time_s" in overrides:
        _record_set(
            config,
            "trajectory_prediction.prediction.lookahead_time_s",
            float(overrides["lookahead_time_s"]),
            source,
        )


def _apply_vpp_cli_overrides(
    config: Dict[str, Any],
    method_def: Dict[str, Any],
    overrides: Dict[str, Any],
) -> None:
    if not overrides:
        return
    if method_def.get("agent_type") == "end_to_end":
        raise ValueError("--vpp-offset-frame cannot be applied to end-to-end methods")
    source = f"{SCRIPT_NAME}:vpp_cli"
    if "offset_frame" in overrides:
        _record_set(
            config,
            "virtual_point.offset_frame",
            str(overrides["offset_frame"]),
            source,
        )
    if "offset_frame_by_task" in overrides:
        _record_set(
            config,
            "virtual_point.offset_frame_by_task",
            dict(overrides["offset_frame_by_task"]),
            source,
        )
    if "predicted_target_blend" in overrides:
        _record_set(
            config,
            "virtual_point.predicted_target_blend",
            float(overrides["predicted_target_blend"]),
            source,
        )
    if "predicted_target_blend_by_task" in overrides:
        _record_set(
            config,
            "virtual_point.predicted_target_blend_by_task",
            dict(overrides["predicted_target_blend_by_task"]),
            source,
        )
    if "predicted_target_forward_scale" in overrides:
        _record_set(
            config,
            "virtual_point.predicted_target_forward_scale",
            float(overrides["predicted_target_forward_scale"]),
            source,
        )
    if "predicted_target_forward_scale_by_task" in overrides:
        _record_set(
            config,
            "virtual_point.predicted_target_forward_scale_by_task",
            dict(overrides["predicted_target_forward_scale_by_task"]),
            source,
        )
    if "post_merge_predicted_target_blend" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_predicted_target_blend",
            float(overrides["post_merge_predicted_target_blend"]),
            source,
        )
    if "post_merge_predicted_target_blend_by_task" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_predicted_target_blend_by_task",
            dict(overrides["post_merge_predicted_target_blend_by_task"]),
            source,
        )
    if "post_merge_offensive_anchor_blend" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_offensive_anchor_blend",
            float(overrides["post_merge_offensive_anchor_blend"]),
            source,
        )
    if "post_merge_offensive_anchor_blend_by_task" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_offensive_anchor_blend_by_task",
            dict(overrides["post_merge_offensive_anchor_blend_by_task"]),
            source,
        )
    if "post_merge_offensive_anchor_longitudinal_scale" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_offensive_anchor_longitudinal_scale",
            float(overrides["post_merge_offensive_anchor_longitudinal_scale"]),
            source,
        )
    if "post_merge_offensive_anchor_longitudinal_scale_by_task" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_offensive_anchor_longitudinal_scale_by_task",
            dict(overrides["post_merge_offensive_anchor_longitudinal_scale_by_task"]),
            source,
        )
    if "post_merge_offensive_anchor_lateral_scale" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_offensive_anchor_lateral_scale",
            float(overrides["post_merge_offensive_anchor_lateral_scale"]),
            source,
        )
    if "post_merge_offensive_anchor_lateral_scale_by_task" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_offensive_anchor_lateral_scale_by_task",
            dict(overrides["post_merge_offensive_anchor_lateral_scale_by_task"]),
            source,
        )
    if "post_merge_offensive_anchor_blend_requires_geometry_disadvantage" in overrides:
        _record_set(
            config,
            (
                "virtual_point."
                "post_merge_offensive_anchor_blend_requires_geometry_disadvantage"
            ),
            bool(
                overrides[
                    "post_merge_offensive_anchor_blend_requires_geometry_disadvantage"
                ]
            ),
            source,
        )
    if (
        "post_merge_offensive_anchor_blend_requires_geometry_disadvantage_by_task"
        in overrides
    ):
        _record_set(
            config,
            (
                "virtual_point."
                "post_merge_offensive_anchor_blend_requires_geometry_disadvantage_by_task"
            ),
            dict(
                overrides[
                    "post_merge_offensive_anchor_blend_requires_geometry_disadvantage_by_task"
                ]
            ),
            source,
        )
    if "post_merge_offensive_anchor_geometry_disadvantage_aa_deg_min" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_offensive_anchor_geometry_disadvantage_aa_deg_min",
            float(
                overrides[
                    "post_merge_offensive_anchor_geometry_disadvantage_aa_deg_min"
                ]
            ),
            source,
        )
    if "post_merge_offensive_anchor_geometry_disadvantage_aa_deg_min_by_task" in overrides:
        _record_set(
            config,
            (
                "virtual_point."
                "post_merge_offensive_anchor_geometry_disadvantage_aa_deg_min_by_task"
            ),
            dict(
                overrides[
                    "post_merge_offensive_anchor_geometry_disadvantage_aa_deg_min_by_task"
                ]
            ),
            source,
        )
    if "post_merge_offensive_anchor_blend_release_blend" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_offensive_anchor_blend_release_blend",
            float(overrides["post_merge_offensive_anchor_blend_release_blend"]),
            source,
        )
    if "post_merge_offensive_anchor_blend_release_blend_by_task" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_offensive_anchor_blend_release_blend_by_task",
            dict(overrides["post_merge_offensive_anchor_blend_release_blend_by_task"]),
            source,
        )
    if "post_merge_offensive_anchor_blend_release_ego_only_streak_steps" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_offensive_anchor_blend_release_ego_only_streak_steps",
            int(
                overrides[
                    "post_merge_offensive_anchor_blend_release_ego_only_streak_steps"
                ]
            ),
            source,
        )
    if (
        "post_merge_offensive_anchor_blend_release_ego_only_streak_steps_by_task"
        in overrides
    ):
        _record_set(
            config,
            "virtual_point.post_merge_offensive_anchor_blend_release_ego_only_streak_steps_by_task",
            dict(
                overrides[
                    "post_merge_offensive_anchor_blend_release_ego_only_streak_steps_by_task"
                ]
            ),
            source,
        )
    if "post_merge_offensive_anchor_blend_release_reset_on_streak_break" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_offensive_anchor_blend_release_reset_on_streak_break",
            bool(
                overrides[
                    "post_merge_offensive_anchor_blend_release_reset_on_streak_break"
                ]
            ),
            source,
        )
    if (
        "post_merge_offensive_anchor_blend_release_reset_on_streak_break_by_task"
        in overrides
    ):
        _record_set(
            config,
            "virtual_point.post_merge_offensive_anchor_blend_release_reset_on_streak_break_by_task",
            dict(
                overrides[
                    "post_merge_offensive_anchor_blend_release_reset_on_streak_break_by_task"
                ]
            ),
            source,
        )
    if "post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m",
            float(
                overrides[
                    "post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m"
                ]
            ),
            source,
        )
    if (
        "post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m_by_task"
        in overrides
    ):
        _record_set(
            config,
            "virtual_point.post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m_by_task",
            dict(
                overrides[
                    "post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m_by_task"
                ]
            ),
            source,
        )
    if "post_merge_offensive_anchor_blend_release_direct_track_enabled" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_offensive_anchor_blend_release_direct_track_enabled",
            bool(overrides["post_merge_offensive_anchor_blend_release_direct_track_enabled"]),
            source,
        )
    if (
        "post_merge_offensive_anchor_blend_release_direct_track_enabled_by_task"
        in overrides
    ):
        _record_set(
            config,
            "virtual_point.post_merge_offensive_anchor_blend_release_direct_track_enabled_by_task",
            dict(
                overrides[
                    "post_merge_offensive_anchor_blend_release_direct_track_enabled_by_task"
                ]
            ),
            source,
        )
    if "post_merge_offensive_anchor_blend_release_recovery_below_altitude_m" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_offensive_anchor_blend_release_recovery_below_altitude_m",
            float(
                overrides[
                    "post_merge_offensive_anchor_blend_release_recovery_below_altitude_m"
                ]
            ),
            source,
        )
    if (
        "post_merge_offensive_anchor_blend_release_recovery_below_altitude_m_by_task"
        in overrides
    ):
        _record_set(
            config,
            "virtual_point.post_merge_offensive_anchor_blend_release_recovery_below_altitude_m_by_task",
            dict(
                overrides[
                    "post_merge_offensive_anchor_blend_release_recovery_below_altitude_m_by_task"
                ]
            ),
            source,
        )
    if (
        "post_merge_offensive_anchor_blend_release_recovery_longitudinal_blend"
        in overrides
    ):
        _record_set(
            config,
            "virtual_point.post_merge_offensive_anchor_blend_release_recovery_longitudinal_blend",
            float(
                overrides[
                    "post_merge_offensive_anchor_blend_release_recovery_longitudinal_blend"
                ]
            ),
            source,
        )
    if (
        "post_merge_offensive_anchor_blend_release_recovery_longitudinal_blend_by_task"
        in overrides
    ):
        _record_set(
            config,
            "virtual_point.post_merge_offensive_anchor_blend_release_recovery_longitudinal_blend_by_task",
            dict(
                overrides[
                    "post_merge_offensive_anchor_blend_release_recovery_longitudinal_blend_by_task"
                ]
            ),
            source,
        )
    if "post_merge_offensive_anchor_blend_release_recovery_lateral_blend" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_offensive_anchor_blend_release_recovery_lateral_blend",
            float(
                overrides[
                    "post_merge_offensive_anchor_blend_release_recovery_lateral_blend"
                ]
            ),
            source,
        )
    if (
        "post_merge_offensive_anchor_blend_release_recovery_lateral_blend_by_task"
        in overrides
    ):
        _record_set(
            config,
            "virtual_point.post_merge_offensive_anchor_blend_release_recovery_lateral_blend_by_task",
            dict(
                overrides[
                    "post_merge_offensive_anchor_blend_release_recovery_lateral_blend_by_task"
                ]
            ),
            source,
        )
    if (
        "post_merge_offensive_anchor_blend_release_recovery_forward_bias_m_max"
        in overrides
    ):
        _record_set(
            config,
            "virtual_point.post_merge_offensive_anchor_blend_release_recovery_forward_bias_m_max",
            float(
                overrides[
                    "post_merge_offensive_anchor_blend_release_recovery_forward_bias_m_max"
                ]
            ),
            source,
        )
    if (
        "post_merge_offensive_anchor_blend_release_recovery_forward_bias_m_max_by_task"
        in overrides
    ):
        _record_set(
            config,
            "virtual_point.post_merge_offensive_anchor_blend_release_recovery_forward_bias_m_max_by_task",
            dict(
                overrides[
                    "post_merge_offensive_anchor_blend_release_recovery_forward_bias_m_max_by_task"
                ]
            ),
            source,
        )
    if "post_merge_offensive_anchor_blend_release_vp_forward_bias_m_min" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_offensive_anchor_blend_release_vp_forward_bias_m_min",
            float(
                overrides[
                    "post_merge_offensive_anchor_blend_release_vp_forward_bias_m_min"
                ]
            ),
            source,
        )
    if (
        "post_merge_offensive_anchor_blend_release_vp_forward_bias_m_min_by_task"
        in overrides
    ):
        _record_set(
            config,
            "virtual_point.post_merge_offensive_anchor_blend_release_vp_forward_bias_m_min_by_task",
            dict(
                overrides[
                    "post_merge_offensive_anchor_blend_release_vp_forward_bias_m_min_by_task"
                ]
            ),
            source,
        )
    if (
        "post_merge_offensive_anchor_blend_release_vp_forward_bias_clamp_negative_lateral_below_altitude_m"
        in overrides
    ):
        _record_set(
            config,
            (
                "virtual_point."
                "post_merge_offensive_anchor_blend_release_"
                "vp_forward_bias_clamp_negative_lateral_below_altitude_m"
            ),
            float(
                overrides[
                    "post_merge_offensive_anchor_blend_release_"
                    "vp_forward_bias_clamp_negative_lateral_below_altitude_m"
                ]
            ),
            source,
        )
    if (
        "post_merge_offensive_anchor_blend_release_vp_forward_bias_clamp_negative_lateral_below_altitude_m_by_task"
        in overrides
    ):
        _record_set(
            config,
            (
                "virtual_point."
                "post_merge_offensive_anchor_blend_release_"
                "vp_forward_bias_clamp_negative_lateral_below_altitude_m_by_task"
            ),
            dict(
                overrides[
                    "post_merge_offensive_anchor_blend_release_"
                    "vp_forward_bias_clamp_negative_lateral_below_altitude_m_by_task"
                ]
            ),
            source,
        )
    if "post_merge_offensive_anchor_lateral_world_offset_latch_on_activation" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_offensive_anchor_lateral_world_offset_latch_on_activation",
            bool(
                overrides[
                    "post_merge_offensive_anchor_lateral_world_offset_latch_on_activation"
                ]
            ),
            source,
        )
    if (
        "post_merge_offensive_anchor_lateral_world_offset_latch_on_activation_by_task"
        in overrides
    ):
        _record_set(
            config,
            "virtual_point.post_merge_offensive_anchor_lateral_world_offset_latch_on_activation_by_task",
            dict(
                overrides[
                    "post_merge_offensive_anchor_lateral_world_offset_latch_on_activation_by_task"
                ]
            ),
            source,
        )
    if "post_merge_offensive_anchor_blend_release_lateral_only" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_offensive_anchor_blend_release_lateral_only",
            bool(
                overrides[
                    "post_merge_offensive_anchor_blend_release_lateral_only"
                ]
            ),
            source,
        )
    if "post_merge_offensive_anchor_blend_release_lateral_only_by_task" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_offensive_anchor_blend_release_lateral_only_by_task",
            dict(
                overrides[
                    "post_merge_offensive_anchor_blend_release_lateral_only_by_task"
                ]
            ),
            source,
        )
    if "post_merge_offensive_anchor_blend_release_lateral_only_hold_steps" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_offensive_anchor_blend_release_lateral_only_hold_steps",
            int(overrides["post_merge_offensive_anchor_blend_release_lateral_only_hold_steps"]),
            source,
        )
    if (
        "post_merge_offensive_anchor_blend_release_lateral_only_hold_steps_by_task"
        in overrides
    ):
        _record_set(
            config,
            "virtual_point.post_merge_offensive_anchor_blend_release_lateral_only_hold_steps_by_task",
            dict(
                overrides[
                    "post_merge_offensive_anchor_blend_release_lateral_only_hold_steps_by_task"
                ]
            ),
            source,
        )
    if "offensive_anchor_blend" in overrides:
        _record_set(
            config,
            "virtual_point.offensive_anchor_blend",
            float(overrides["offensive_anchor_blend"]),
            source,
        )
    if "offensive_anchor_blend_by_task" in overrides:
        _record_set(
            config,
            "virtual_point.offensive_anchor_blend_by_task",
            dict(overrides["offensive_anchor_blend_by_task"]),
            source,
        )
    if "post_merge_predicted_target_forward_scale" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_predicted_target_forward_scale",
            float(overrides["post_merge_predicted_target_forward_scale"]),
            source,
        )
    if "post_merge_predicted_target_forward_scale_by_task" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_predicted_target_forward_scale_by_task",
            dict(overrides["post_merge_predicted_target_forward_scale_by_task"]),
            source,
        )
    if "post_merge_predicted_target_forward_scale_release_scale" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_predicted_target_forward_scale_release_scale",
            float(overrides["post_merge_predicted_target_forward_scale_release_scale"]),
            source,
        )
    if "post_merge_predicted_target_forward_scale_release_scale_by_task" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_predicted_target_forward_scale_release_scale_by_task",
            dict(overrides["post_merge_predicted_target_forward_scale_release_scale_by_task"]),
            source,
        )
    if "post_merge_predicted_target_forward_scale_release_ego_only_streak_steps" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_predicted_target_forward_scale_release_ego_only_streak_steps",
            int(
                overrides[
                    "post_merge_predicted_target_forward_scale_release_ego_only_streak_steps"
                ]
            ),
            source,
        )
    if (
        "post_merge_predicted_target_forward_scale_release_ego_only_streak_steps_by_task"
        in overrides
    ):
        _record_set(
            config,
            "virtual_point.post_merge_predicted_target_forward_scale_release_ego_only_streak_steps_by_task",
            dict(
                overrides[
                    "post_merge_predicted_target_forward_scale_release_ego_only_streak_steps_by_task"
                ]
            ),
            source,
        )
    if "post_merge_predicted_target_forward_scale_release_reset_on_streak_break" in overrides:
        _record_set(
            config,
            (
                "virtual_point."
                "post_merge_predicted_target_forward_scale_release_reset_on_streak_break"
            ),
            bool(
                overrides[
                    "post_merge_predicted_target_forward_scale_release_reset_on_streak_break"
                ]
            ),
            source,
        )
    if (
        "post_merge_predicted_target_forward_scale_release_reset_on_streak_break_by_task"
        in overrides
    ):
        _record_set(
            config,
            (
                "virtual_point."
                "post_merge_predicted_target_forward_scale_release_reset_on_streak_break_by_task"
            ),
            dict(
                overrides[
                    "post_merge_predicted_target_forward_scale_release_reset_on_streak_break_by_task"
                ]
            ),
            source,
        )
    if "post_merge_predicted_target_forward_scale_hold_steps" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_predicted_target_forward_scale_hold_steps",
            int(overrides["post_merge_predicted_target_forward_scale_hold_steps"]),
            source,
        )
    if "post_merge_predicted_target_forward_scale_hold_steps_by_task" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_predicted_target_forward_scale_hold_steps_by_task",
            dict(overrides["post_merge_predicted_target_forward_scale_hold_steps_by_task"]),
            source,
        )
    if "post_merge_predicted_target_blend_release_on_attack_zone" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_predicted_target_blend_release_on_attack_zone",
            bool(overrides["post_merge_predicted_target_blend_release_on_attack_zone"]),
            source,
        )
    if "post_merge_predicted_target_blend_release_on_attack_zone_by_task" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_predicted_target_blend_release_on_attack_zone_by_task",
            dict(overrides["post_merge_predicted_target_blend_release_on_attack_zone_by_task"]),
            source,
        )
    if "post_merge_predicted_target_blend_release_requires_target_attack_zone" in overrides:
        _record_set(
            config,
            (
                "virtual_point."
                "post_merge_predicted_target_blend_release_requires_target_attack_zone"
            ),
            bool(
                overrides[
                    "post_merge_predicted_target_blend_release_requires_target_attack_zone"
                ]
            ),
            source,
        )
    if (
        "post_merge_predicted_target_blend_release_requires_target_attack_zone_by_task"
        in overrides
    ):
        _record_set(
            config,
            (
                "virtual_point."
                "post_merge_predicted_target_blend_release_requires_target_attack_zone_by_task"
            ),
            dict(
                overrides[
                    "post_merge_predicted_target_blend_release_requires_target_attack_zone_by_task"
                ]
            ),
            source,
        )
    if "post_merge_predicted_target_blend_release_below_altitude_m" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_predicted_target_blend_release_below_altitude_m",
            float(overrides["post_merge_predicted_target_blend_release_below_altitude_m"]),
            source,
        )
    if "post_merge_predicted_target_blend_release_below_altitude_m_by_task" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_predicted_target_blend_release_below_altitude_m_by_task",
            dict(overrides["post_merge_predicted_target_blend_release_below_altitude_m_by_task"]),
            source,
        )
    if "post_merge_predicted_target_blend_hold_steps" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_predicted_target_blend_hold_steps",
            int(overrides["post_merge_predicted_target_blend_hold_steps"]),
            source,
        )
    if "post_merge_predicted_target_blend_hold_steps_by_task" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_predicted_target_blend_hold_steps_by_task",
            dict(overrides["post_merge_predicted_target_blend_hold_steps_by_task"]),
            source,
        )
    if "post_merge_predicted_target_blend_release_blend" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_predicted_target_blend_release_blend",
            float(overrides["post_merge_predicted_target_blend_release_blend"]),
            source,
        )
    if "post_merge_predicted_target_blend_release_blend_by_task" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_predicted_target_blend_release_blend_by_task",
            dict(overrides["post_merge_predicted_target_blend_release_blend_by_task"]),
            source,
        )
    if "longitudinal_scale" in overrides:
        _record_set(
            config,
            "virtual_point.longitudinal_scale",
            float(overrides["longitudinal_scale"]),
            source,
        )
    if "longitudinal_scale_by_task" in overrides:
        _record_set(
            config,
            "virtual_point.longitudinal_scale_by_task",
            dict(overrides["longitudinal_scale_by_task"]),
            source,
        )
    if "lateral_scale" in overrides:
        _record_set(
            config,
            "virtual_point.lateral_scale",
            float(overrides["lateral_scale"]),
            source,
        )
    if "lateral_scale_by_task" in overrides:
        _record_set(
            config,
            "virtual_point.lateral_scale_by_task",
            dict(overrides["lateral_scale_by_task"]),
            source,
        )
    if "offensive_anchor_longitudinal_m" in overrides:
        _record_set(
            config,
            "virtual_point.offensive_anchor_longitudinal_m",
            float(overrides["offensive_anchor_longitudinal_m"]),
            source,
        )
    if "offensive_anchor_longitudinal_m_by_task" in overrides:
        _record_set(
            config,
            "virtual_point.offensive_anchor_longitudinal_m_by_task",
            dict(overrides["offensive_anchor_longitudinal_m_by_task"]),
            source,
        )
    if "offensive_anchor_frame" in overrides:
        _record_set(
            config,
            "virtual_point.offensive_anchor_frame",
            str(overrides["offensive_anchor_frame"]),
            source,
        )
    if "offensive_anchor_frame_by_task" in overrides:
        _record_set(
            config,
            "virtual_point.offensive_anchor_frame_by_task",
            dict(overrides["offensive_anchor_frame_by_task"]),
            source,
        )
    if "offensive_anchor_encounter_stable_max_heading_delta_deg" in overrides:
        _record_set(
            config,
            "virtual_point.offensive_anchor_encounter_stable_max_heading_delta_deg",
            float(
                overrides[
                    "offensive_anchor_encounter_stable_max_heading_delta_deg"
                ]
            ),
            source,
        )
    if (
        "offensive_anchor_encounter_stable_max_heading_delta_deg_by_task"
        in overrides
    ):
        _record_set(
            config,
            (
                "virtual_point."
                "offensive_anchor_encounter_stable_max_heading_delta_deg_by_task"
            ),
            dict(
                overrides[
                    "offensive_anchor_encounter_stable_max_heading_delta_deg_by_task"
                ]
            ),
            source,
        )
    if "offensive_anchor_lateral_frame" in overrides:
        _record_set(
            config,
            "virtual_point.offensive_anchor_lateral_frame",
            str(overrides["offensive_anchor_lateral_frame"]),
            source,
        )
    if "offensive_anchor_lateral_frame_by_task" in overrides:
        _record_set(
            config,
            "virtual_point.offensive_anchor_lateral_frame_by_task",
            dict(overrides["offensive_anchor_lateral_frame_by_task"]),
            source,
        )
    if "offensive_anchor_lateral_sign_mode" in overrides:
        _record_set(
            config,
            "virtual_point.offensive_anchor_lateral_sign_mode",
            str(overrides["offensive_anchor_lateral_sign_mode"]),
            source,
        )
    if "offensive_anchor_lateral_sign_mode_by_task" in overrides:
        _record_set(
            config,
            "virtual_point.offensive_anchor_lateral_sign_mode_by_task",
            dict(overrides["offensive_anchor_lateral_sign_mode_by_task"]),
            source,
        )
    if "offensive_anchor_lateral_m" in overrides:
        _record_set(
            config,
            "virtual_point.offensive_anchor_lateral_m",
            float(overrides["offensive_anchor_lateral_m"]),
            source,
        )
    if "offensive_anchor_lateral_m_by_task" in overrides:
        _record_set(
            config,
            "virtual_point.offensive_anchor_lateral_m_by_task",
            dict(overrides["offensive_anchor_lateral_m_by_task"]),
            source,
        )
    if "offensive_anchor_vertical_m" in overrides:
        _record_set(
            config,
            "virtual_point.offensive_anchor_vertical_m",
            float(overrides["offensive_anchor_vertical_m"]),
            source,
        )
    if "offensive_anchor_vertical_m_by_task" in overrides:
        _record_set(
            config,
            "virtual_point.offensive_anchor_vertical_m_by_task",
            dict(overrides["offensive_anchor_vertical_m_by_task"]),
            source,
        )
    if "post_merge_anchor_mode" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_anchor_mode",
            str(overrides["post_merge_anchor_mode"]),
            source,
        )
    if "post_merge_anchor_mode_by_task" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_anchor_mode_by_task",
            dict(overrides["post_merge_anchor_mode_by_task"]),
            source,
        )
    if "post_merge_anchor_mode_requires_geometry_disadvantage" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_anchor_mode_requires_geometry_disadvantage",
            bool(overrides["post_merge_anchor_mode_requires_geometry_disadvantage"]),
            source,
        )
    if "post_merge_anchor_mode_requires_geometry_disadvantage_by_task" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_anchor_mode_requires_geometry_disadvantage_by_task",
            dict(
                overrides[
                    "post_merge_anchor_mode_requires_geometry_disadvantage_by_task"
                ]
            ),
            source,
        )
    if "post_merge_anchor_mode_recovery_below_altitude_m" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_anchor_mode_recovery_below_altitude_m",
            float(overrides["post_merge_anchor_mode_recovery_below_altitude_m"]),
            source,
        )
    if "post_merge_anchor_mode_recovery_below_altitude_m_by_task" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_anchor_mode_recovery_below_altitude_m_by_task",
            dict(overrides["post_merge_anchor_mode_recovery_below_altitude_m_by_task"]),
            source,
        )
    if "post_merge_anchor_mode_release_ego_only_streak_steps" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_anchor_mode_release_ego_only_streak_steps",
            int(overrides["post_merge_anchor_mode_release_ego_only_streak_steps"]),
            source,
        )
    if "post_merge_anchor_mode_release_ego_only_streak_steps_by_task" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_anchor_mode_release_ego_only_streak_steps_by_task",
            dict(overrides["post_merge_anchor_mode_release_ego_only_streak_steps_by_task"]),
            source,
        )
    if "post_merge_anchor_mode_release_reset_on_streak_break" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_anchor_mode_release_reset_on_streak_break",
            bool(overrides["post_merge_anchor_mode_release_reset_on_streak_break"]),
            source,
        )
    if "post_merge_anchor_mode_release_reset_on_streak_break_by_task" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_anchor_mode_release_reset_on_streak_break_by_task",
            dict(
                overrides[
                    "post_merge_anchor_mode_release_reset_on_streak_break_by_task"
                ]
            ),
            source,
        )
    if "post_merge_anchor_mode_offensive_anchor_blend" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_anchor_mode_offensive_anchor_blend",
            float(overrides["post_merge_anchor_mode_offensive_anchor_blend"]),
            source,
        )
    if "post_merge_anchor_mode_offensive_anchor_blend_by_task" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_anchor_mode_offensive_anchor_blend_by_task",
            dict(overrides["post_merge_anchor_mode_offensive_anchor_blend_by_task"]),
            source,
        )
    if "post_merge_anchor_mode_offensive_anchor_longitudinal_blend" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_anchor_mode_offensive_anchor_longitudinal_blend",
            float(overrides["post_merge_anchor_mode_offensive_anchor_longitudinal_blend"]),
            source,
        )
    if "post_merge_anchor_mode_offensive_anchor_longitudinal_blend_by_task" in overrides:
        _record_set(
            config,
            (
                "virtual_point."
                "post_merge_anchor_mode_offensive_anchor_longitudinal_blend_by_task"
            ),
            dict(
                overrides[
                    "post_merge_anchor_mode_offensive_anchor_longitudinal_blend_by_task"
                ]
            ),
            source,
        )
    if "post_merge_anchor_mode_offensive_anchor_lateral_blend" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_anchor_mode_offensive_anchor_lateral_blend",
            float(overrides["post_merge_anchor_mode_offensive_anchor_lateral_blend"]),
            source,
        )
    if "post_merge_anchor_mode_offensive_anchor_lateral_blend_by_task" in overrides:
        _record_set(
            config,
            (
                "virtual_point."
                "post_merge_anchor_mode_offensive_anchor_lateral_blend_by_task"
            ),
            dict(
                overrides[
                    "post_merge_anchor_mode_offensive_anchor_lateral_blend_by_task"
                ]
            ),
            source,
        )
    if "post_merge_anchor_mode_lateral_world_offset_latch_on_activation" in overrides:
        _record_set(
            config,
            (
                "virtual_point."
                "post_merge_anchor_mode_lateral_world_offset_latch_on_activation"
            ),
            bool(
                overrides[
                    "post_merge_anchor_mode_lateral_world_offset_latch_on_activation"
                ]
            ),
            source,
        )
    if (
        "post_merge_anchor_mode_lateral_world_offset_latch_on_activation_by_task"
        in overrides
    ):
        _record_set(
            config,
            (
                "virtual_point."
                "post_merge_anchor_mode_lateral_world_offset_latch_on_activation_by_task"
            ),
            dict(
                overrides[
                    "post_merge_anchor_mode_lateral_world_offset_latch_on_activation_by_task"
                ]
            ),
            source,
        )
    if "post_merge_anchor_mode_longitudinal_scale" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_anchor_mode_longitudinal_scale",
            float(overrides["post_merge_anchor_mode_longitudinal_scale"]),
            source,
        )
    if "post_merge_anchor_mode_longitudinal_scale_by_task" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_anchor_mode_longitudinal_scale_by_task",
            dict(overrides["post_merge_anchor_mode_longitudinal_scale_by_task"]),
            source,
        )
    if "post_merge_anchor_mode_lateral_scale" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_anchor_mode_lateral_scale",
            float(overrides["post_merge_anchor_mode_lateral_scale"]),
            source,
        )
    if "post_merge_anchor_mode_lateral_scale_by_task" in overrides:
        _record_set(
            config,
            "virtual_point.post_merge_anchor_mode_lateral_scale_by_task",
            dict(overrides["post_merge_anchor_mode_lateral_scale_by_task"]),
            source,
        )
    if "close_range_anchor_mode" in overrides:
        _record_set(
            config,
            "virtual_point.close_range_anchor_mode",
            str(overrides["close_range_anchor_mode"]),
            source,
        )
    if "close_range_anchor_mode_by_task" in overrides:
        _record_set(
            config,
            "virtual_point.close_range_anchor_mode_by_task",
            dict(overrides["close_range_anchor_mode_by_task"]),
            source,
        )
    if "close_range_anchor_trigger_range_m" in overrides:
        _record_set(
            config,
            "virtual_point.close_range_anchor_trigger_range_m",
            float(overrides["close_range_anchor_trigger_range_m"]),
            source,
        )
    if "close_range_anchor_trigger_range_m_by_task" in overrides:
        _record_set(
            config,
            "virtual_point.close_range_anchor_trigger_range_m_by_task",
            dict(overrides["close_range_anchor_trigger_range_m_by_task"]),
            source,
        )
    if "close_range_anchor_alignment_angle_deg_max" in overrides:
        _record_set(
            config,
            "virtual_point.close_range_anchor_alignment_angle_deg_max",
            float(overrides["close_range_anchor_alignment_angle_deg_max"]),
            source,
        )
    if "close_range_anchor_alignment_angle_deg_max_by_task" in overrides:
        _record_set(
            config,
            "virtual_point.close_range_anchor_alignment_angle_deg_max_by_task",
            dict(overrides["close_range_anchor_alignment_angle_deg_max_by_task"]),
            source,
        )
    if "close_range_anchor_release_on_post_merge" in overrides:
        _record_set(
            config,
            "virtual_point.close_range_anchor_release_on_post_merge",
            bool(overrides["close_range_anchor_release_on_post_merge"]),
            source,
        )
    if "close_range_anchor_release_on_post_merge_by_task" in overrides:
        _record_set(
            config,
            "virtual_point.close_range_anchor_release_on_post_merge_by_task",
            dict(overrides["close_range_anchor_release_on_post_merge_by_task"]),
            source,
        )
    if "close_range_anchor_requires_first_pass" in overrides:
        _record_set(
            config,
            "virtual_point.close_range_anchor_requires_first_pass",
            bool(overrides["close_range_anchor_requires_first_pass"]),
            source,
        )
    if "close_range_anchor_requires_first_pass_by_task" in overrides:
        _record_set(
            config,
            "virtual_point.close_range_anchor_requires_first_pass_by_task",
            dict(overrides["close_range_anchor_requires_first_pass_by_task"]),
            source,
        )
    if "close_range_anchor_offensive_anchor_blend" in overrides:
        _record_set(
            config,
            "virtual_point.close_range_anchor_offensive_anchor_blend",
            float(overrides["close_range_anchor_offensive_anchor_blend"]),
            source,
        )
    if "close_range_anchor_offensive_anchor_blend_by_task" in overrides:
        _record_set(
            config,
            "virtual_point.close_range_anchor_offensive_anchor_blend_by_task",
            dict(overrides["close_range_anchor_offensive_anchor_blend_by_task"]),
            source,
        )
    if "close_range_anchor_release_alignment_angle_deg_max" in overrides:
        _record_set(
            config,
            "virtual_point.close_range_anchor_release_alignment_angle_deg_max",
            float(overrides["close_range_anchor_release_alignment_angle_deg_max"]),
            source,
        )
    if "close_range_anchor_release_alignment_angle_deg_max_by_task" in overrides:
        _record_set(
            config,
            "virtual_point.close_range_anchor_release_alignment_angle_deg_max_by_task",
            dict(overrides["close_range_anchor_release_alignment_angle_deg_max_by_task"]),
            source,
        )
    if "close_range_anchor_post_merge_hold_steps" in overrides:
        _record_set(
            config,
            "virtual_point.close_range_anchor_post_merge_hold_steps",
            int(overrides["close_range_anchor_post_merge_hold_steps"]),
            source,
        )
    if "close_range_anchor_post_merge_hold_steps_by_task" in overrides:
        _record_set(
            config,
            "virtual_point.close_range_anchor_post_merge_hold_steps_by_task",
            dict(overrides["close_range_anchor_post_merge_hold_steps_by_task"]),
            source,
        )


def _apply_prediction_variant(
    config: Dict[str, Any],
    comparison_config: Dict[str, Any],
    method_name: str,
    method_def: Dict[str, Any],
) -> None:
    source = f"{SCRIPT_NAME}:{method_name}_{method_def.get('prediction_variant', 'none')}"
    variant = str(method_def.get("prediction_variant", "none")).lower()
    pred_defaults = comparison_config.get("prediction", {})
    predictor_checkpoint = method_def.get(
        "predictor_checkpoint", pred_defaults.get("learned_predictor_checkpoint")
    )
    lookahead_time_s = float(pred_defaults.get("lookahead_time_s", 1.0))

    if method_def.get("agent_type") == "end_to_end":
        _record_set(config, "end_to_end.enabled", True, source)
        _record_set(config, "virtual_point.enabled", False, source)
        _record_set(config, "trajectory_prediction.enabled", False, source)
        _record_set(config, "policy.action_dim", 3, source)
        return

    _record_set(config, "end_to_end.enabled", False, source)
    _record_set(config, "virtual_point.enabled", True, source)
    _record_set(config, "virtual_point.mode", "normal", source)

    if variant == "none":
        _record_set(config, "trajectory_prediction.enabled", False, source)
        _record_set(config, "virtual_point.anchor_mode", "current_target", source)
        _record_set(
            config,
            "trajectory_prediction.integration.prediction_noise_std_m",
            0.0,
            source,
        )
        return

    if variant == "perfect":
        _record_set(config, "trajectory_prediction.enabled", False, source)
        _record_set(config, "virtual_point.anchor_mode", "oracle_future_position", source)
        _record_set(
            config,
            "trajectory_prediction.prediction.lookahead_time_s",
            lookahead_time_s,
            source,
        )
        _record_set(
            config,
            "trajectory_prediction.integration.prediction_noise_std_m",
            0.0,
            source,
        )
        return

    if variant not in {"learned", "noisy"}:
        raise ValueError(f"Unknown prediction_variant for {method_name}: {variant}")

    _record_set(config, "trajectory_prediction.enabled", True, source)
    _record_set(config, "trajectory_prediction.predictor_type", "lstm", source)
    _record_set(config, "trajectory_prediction.checkpoint_path", predictor_checkpoint, source)
    _record_set(config, "trajectory_prediction.strict_predictor_init", True, source)
    _record_set(config, "trajectory_prediction.device", "cpu", source)
    _record_set(config, "trajectory_prediction.freeze_predictor_during_rl", True, source)
    _record_set(
        config,
        "trajectory_prediction.prediction.lookahead_time_s",
        lookahead_time_s,
        source,
    )
    _record_set(
        config,
        "trajectory_prediction.prediction.output_mode",
        "relative_displacement",
        source,
    )
    _record_set(
        config,
        "trajectory_prediction.prediction.fallback_mode",
        "constant_velocity",
        source,
    )
    _record_set(config, "trajectory_prediction.integration.anchor_mode", "predicted_target", source)
    _record_set(
        config,
        "trajectory_prediction.integration.add_prediction_to_observation",
        False,
        source,
    )
    _record_set(config, "virtual_point.anchor_mode", "predicted_target", source)

    noise_std_m = 0.0
    if variant == "noisy":
        noise_std_m = float(pred_defaults.get("noisy_prediction_std_m", 100.0))
    _record_set(
        config,
        "trajectory_prediction.integration.prediction_noise_std_m",
        noise_std_m,
        source,
    )


def build_eval_config(
    comparison_config: Dict[str, Any],
    method_name: str,
    task_name: str,
    backend: str,
    opponent_stage: str = "none",
    max_steps_override: Optional[int] = None,
    jsbsim_root: Optional[str] = None,
    attack_zone_overrides: Optional[Dict[str, Any]] = None,
    prediction_overrides: Optional[Dict[str, Any]] = None,
    vpp_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    methods = comparison_config.get("methods", {})
    tasks = comparison_config.get("tasks", {})
    if method_name not in methods:
        raise KeyError(f"Unknown method: {method_name}")
    if task_name not in tasks:
        raise KeyError(f"Unknown task: {task_name}")

    method_def = methods[method_name]
    task_def = _resolve_task_def(tasks[task_name])
    agent_type = method_def.get("agent_type", "ppo")
    if agent_type in {"oracle_task_gate", "random_task_gate", "hierarchical_commander"}:
        # oracle/random gate and hierarchical commander use the comparison config
        # as the evaluation environment base.
        config = copy.deepcopy(comparison_config)
    else:
        checkpoint_path = _repo_path(method_def.get("checkpoint"))
        fallback_config_path = _repo_path(method_def.get("config_path"))
        config = _load_checkpoint_config(checkpoint_path, fallback_config_path)
    _apply_task_and_common_overrides(
        config,
        comparison_config,
        task_name,
        task_def,
        backend=backend,
        max_steps_override=max_steps_override,
        jsbsim_root=jsbsim_root,
    )
    _apply_prediction_variant(config, comparison_config, method_name, method_def)
    _apply_prediction_cli_overrides(config, method_def, prediction_overrides or {})
    _apply_vpp_cli_overrides(config, method_def, vpp_overrides or {})
    _apply_opponent_stage(config, comparison_config, opponent_stage)
    _apply_attack_zone_cli_overrides(config, attack_zone_overrides or {})
    method_config_overrides = method_def.get("config_overrides", {})
    if method_config_overrides:
        for key, value in method_config_overrides.items():
            _record_set(
                config,
                key,
                value,
                f"{SCRIPT_NAME}:method_def_override",
            )
    return config


def _build_agent(
    method_def: Dict[str, Any],
    config: Dict[str, Any],
    checkpoint_path: Path,
    sample_obs: Dict[str, Any],
    device: str,
):
    agent_type = method_def.get("agent_type", "ppo")
    if agent_type == "end_to_end":
        obs_dim = int(sample_obs["observation_vector"].shape[0])
        action_dim = int(config.get("policy", {}).get("action_dim", 3))
        agent = EndToEndPPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=config, device=device)
        agent.load(str(checkpoint_path))
    elif agent_type == "sac":
        from uav_vpp_guidance.agents.sac_agent import SACAgent

        obs_dim = int(sample_obs["observation_vector"].shape[0])
        action_dim = int(config.get("policy", {}).get("action_dim", 3))
        agent = SACAgent(obs_dim=obs_dim, action_dim=action_dim, config=config, device=device)
        agent.load(str(checkpoint_path))
    elif agent_type == "rule_guidance":
        from uav_vpp_guidance.evaluation.rule_guidance_policy import RuleGuidancePolicy

        action_dim = int(config.get("policy", {}).get("action_dim", 3))
        agent = RuleGuidancePolicy(
            action_dim=action_dim,
            constant_action=method_def.get("constant_action"),
            action_by_task=method_def.get("action_by_task"),
        )
    elif agent_type == "legacy_hierarchical":
        from uav_vpp_guidance.evaluation.legacy_hierarchical_policy import (
            LegacyHierarchicalPolicy,
        )

        agent = LegacyHierarchicalPolicy(
            checkpoint_path=str(checkpoint_path),
            device=device,
            guidance_config=method_def.get("guidance_config"),
        )
    elif agent_type == "hierarchical_commander":
        from uav_vpp_guidance.evaluation.hierarchical_commander_policy import (
            HierarchicalCommanderPolicy,
        )

        obs_dim = int(sample_obs["observation_vector"].shape[0])
        agent = HierarchicalCommanderPolicy(
            checkpoint_path=str(checkpoint_path),
            config=config,
            obs_dim=obs_dim,
            device=device,
        )
    elif agent_type == "oracle_task_gate":
        from uav_vpp_guidance.evaluation.oracle_task_gate_policy import OracleTaskGatePolicy
        agent = OracleTaskGatePolicy(
            specialists_config=method_def.get("specialists", {}),
            device=device,
        )
    elif agent_type == "random_task_gate":
        from uav_vpp_guidance.evaluation.random_task_gate_policy import RandomTaskGatePolicy
        agent = RandomTaskGatePolicy(
            specialists_config=method_def.get("specialists", {}),
            device=device,
            macro_step=method_def.get("macro_step", 1),
        )
    else:
        obs_dim = int(sample_obs["observation_vector"].shape[0])
        action_dim = int(config.get("policy", {}).get("action_dim", 3))
        agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=config, device=device)
        agent.load(str(checkpoint_path))
    return agent


def _select_methods(config: Dict[str, Any], preset: str, methods: Optional[List[str]]) -> List[str]:
    if methods:
        return methods
    defaults = config.get("run_defaults", {})
    if preset == "ablation":
        return list(defaults.get("ablation_methods", []))
    return list(defaults.get("main_methods", []))


def _select_tasks(config: Dict[str, Any], tasks: Optional[List[str]]) -> List[str]:
    if tasks:
        return tasks
    return list(config.get("run_defaults", {}).get("tasks", []))


def _select_seeds(config: Dict[str, Any], args: argparse.Namespace) -> List[int]:
    if args.seeds is not None:
        return args.seeds
    defaults = config.get("run_defaults", {})
    if args.run_status == "smoke":
        return list(defaults.get("smoke_seeds", [0]))
    return list(defaults.get("formal_small_seeds", [0, 1, 2]))


def _task_scenarios(task_def: Dict[str, Any]) -> List[Optional[Dict[str, Any]]]:
    scenarios = task_def.get("task", {}).get("scenarios")
    if not scenarios:
        return [None]
    return [copy.deepcopy(s) for s in scenarios]


def _scenario_name(scenario: Optional[Dict[str, Any]], task_name: str) -> str:
    if scenario is None:
        return task_name
    return scenario.get("name", task_name)


def _finite_mean(values: Iterable[float]) -> float:
    clean = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(clean)) if clean else float("nan")


def _finite_min(values: Iterable[float]) -> float:
    clean = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.min(clean)) if clean else float("nan")


def _finite_max(values: Iterable[float]) -> float:
    clean = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.max(clean)) if clean else float("nan")


def _finite_median(values: Iterable[float]) -> float:
    clean = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.median(clean)) if clean else float("nan")


def _last_finite(values: Iterable[float]) -> float:
    clean = [float(v) for v in values if v is not None and np.isfinite(v)]
    return clean[-1] if clean else float("nan")


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _trajectory_dt_s(record: Dict[str, Any]) -> float:
    trajectory = record.get("trajectory", [])
    times = [
        _safe_float(point.get("time_s"))
        for point in trajectory
        if isinstance(point, dict)
    ]
    diffs = [
        b - a
        for a, b in zip(times, times[1:])
        if np.isfinite(a) and np.isfinite(b) and b > a
    ]
    if diffs:
        return _finite_median(diffs)
    steps = int(record.get("steps", 0) or 0)
    total_time_s = _safe_float(record.get("total_time_s"))
    if steps > 0 and np.isfinite(total_time_s) and total_time_s > 0.0:
        return total_time_s / steps
    return 0.2


def _first_true_time_s(trajectory: List[Dict[str, Any]], field: str) -> float:
    for point in trajectory:
        if not isinstance(point, dict) or not bool(point.get(field, False)):
            continue
        time_s = _safe_float(point.get("time_s"))
        if np.isfinite(time_s):
            return time_s
    return float("nan")


def _first_true_step(trajectory: List[Dict[str, Any]], field: str) -> float:
    for index, point in enumerate(trajectory, start=1):
        if not isinstance(point, dict) or not bool(point.get(field, False)):
            continue
        step = _safe_float(point.get("step"))
        if np.isfinite(step):
            return step
        return float(index)
    return float("nan")


def _first_post_merge_true_step(
    trajectory: List[Dict[str, Any]],
    field: str,
) -> float:
    for index, point in enumerate(trajectory, start=1):
        if not isinstance(point, dict):
            continue
        if not bool(point.get("post_merge", False)) or not bool(point.get(field, False)):
            continue
        step = _safe_float(point.get("step"))
        if np.isfinite(step):
            return step
        return float(index)
    return float("nan")


def _damage_margin_from_record(record: Dict[str, Any]) -> float:
    initial_hp = _safe_float(record.get("combat_initial_hp"), 100.0)
    ego_hp = _safe_float(record.get("ego_hp"))
    target_hp = _safe_float(record.get("target_hp"))
    if np.isfinite(initial_hp) and np.isfinite(ego_hp) and np.isfinite(target_hp):
        damage_dealt = initial_hp - target_hp
        damage_taken = initial_hp - ego_hp
        return float(damage_dealt - damage_taken)
    return _safe_float(record.get("hp_advantage"))


def summarize_episode_combat_geometry(record: Dict[str, Any]) -> Dict[str, float]:
    """Build episode-level VP/merge/attack-zone diagnostics from trajectory."""
    trajectory = [
        point for point in record.get("trajectory", []) if isinstance(point, dict)
    ]
    post_merge_trajectory = [
        point for point in trajectory if bool(point.get("post_merge", False))
    ]
    dt = _trajectory_dt_s(record)
    post_merge_advantage_s = 0.0
    for point in post_merge_trajectory:
        ego_window = 1.0 if bool(point.get("ego_in_attack_zone", False)) else 0.0
        target_window = 1.0 if bool(point.get("target_in_attack_zone", False)) else 0.0
        post_merge_advantage_s += (ego_window - target_window) * dt

    def _is_pre_merge(point: Dict[str, Any]) -> bool:
        if "pre_merge" in point:
            return bool(point.get("pre_merge", False))
        return not bool(point.get("post_merge", False))

    def _post_merge_fraction(field: str) -> float:
        return _finite_mean(
            1.0 if bool(point.get(field, False)) else 0.0
            for point in post_merge_trajectory
        )

    def _post_merge_active_mean(value_field: str, active_field: str) -> float:
        return _finite_mean(
            _safe_float(point.get(value_field))
            for point in post_merge_trajectory
            if bool(point.get(active_field, False))
        )

    return {
        "first_ego_attack_time_s": _first_true_time_s(
            trajectory,
            "ego_in_attack_zone",
        ),
        "first_target_attack_time_s": _first_true_time_s(
            trajectory,
            "target_in_attack_zone",
        ),
        "first_pass_step": _first_true_step(trajectory, "post_merge"),
        "post_merge_attack_zone_advantage_s": float(post_merge_advantage_s),
        "close_range_anchor_mode_active_fraction": _post_merge_fraction(
            "close_range_anchor_mode_active"
        ),
        "close_range_anchor_mode_first_active_step": _first_post_merge_true_step(
            trajectory,
            "close_range_anchor_mode_active",
        ),
        "post_merge_anchor_mode_active_fraction": _post_merge_fraction(
            "post_merge_anchor_mode_active"
        ),
        "post_merge_anchor_mode_condition_met_fraction": _post_merge_fraction(
            "post_merge_anchor_mode_condition_met"
        ),
        "post_merge_anchor_mode_first_active_step": _first_post_merge_true_step(
            trajectory,
            "post_merge_anchor_mode_active",
        ),
        "post_merge_anchor_mode_lateral_world_offset_latch_active_fraction": (
            _post_merge_fraction(
                "post_merge_anchor_mode_lateral_world_offset_latch_active"
            )
        ),
        "post_merge_anchor_mode_lateral_world_offset_latch_first_active_step": (
            _first_post_merge_true_step(
                trajectory,
                "post_merge_anchor_mode_lateral_world_offset_latch_active",
            )
        ),
        "post_merge_anchor_mode_recovery_active_fraction": _post_merge_fraction(
            "post_merge_anchor_mode_recovery_active"
        ),
        "post_merge_anchor_mode_recovery_first_step": _first_post_merge_true_step(
            trajectory,
            "post_merge_anchor_mode_recovery_active",
        ),
        "post_merge_anchor_mode_released_fraction": _post_merge_fraction(
            "post_merge_anchor_mode_released"
        ),
        "post_merge_anchor_mode_release_first_step": _first_post_merge_true_step(
            trajectory,
            "post_merge_anchor_mode_released",
        ),
        "post_merge_offensive_anchor_condition_met_fraction": _post_merge_fraction(
            "post_merge_offensive_anchor_condition_met"
        ),
        "post_merge_offensive_anchor_alignment_disadvantage_fraction": (
            _post_merge_fraction(
                "post_merge_offensive_anchor_alignment_disadvantage"
            )
        ),
        "post_merge_offensive_anchor_blend_active_fraction": _post_merge_fraction(
            "post_merge_offensive_anchor_blend_active"
        ),
        "post_merge_offensive_anchor_first_active_step": _first_post_merge_true_step(
            trajectory,
            "post_merge_offensive_anchor_blend_active",
        ),
        "post_merge_offensive_anchor_blend_release_blend_active_fraction": (
            _post_merge_fraction(
                "post_merge_offensive_anchor_blend_release_blend_active"
            )
        ),
        "post_merge_offensive_anchor_blend_released_fraction": _post_merge_fraction(
            "post_merge_offensive_anchor_blend_released"
        ),
        "post_merge_offensive_anchor_blend_release_first_step": (
            _first_post_merge_true_step(
                trajectory,
                "post_merge_offensive_anchor_blend_released",
            )
        ),
        "post_merge_offensive_anchor_blend_release_direct_track_active_fraction": (
            _post_merge_fraction(
                "post_merge_offensive_anchor_blend_release_direct_track_active"
            )
        ),
        "post_merge_offensive_anchor_blend_release_direct_track_first_step": (
            _first_post_merge_true_step(
                trajectory,
                "post_merge_offensive_anchor_blend_release_direct_track_active",
            )
        ),
        "post_merge_offensive_anchor_blend_release_recovery_active_fraction": (
            _post_merge_fraction(
                "post_merge_offensive_anchor_blend_release_recovery_active"
            )
        ),
        "post_merge_offensive_anchor_blend_release_recovery_first_step": (
            _first_post_merge_true_step(
                trajectory,
                "post_merge_offensive_anchor_blend_release_recovery_active",
            )
        ),
        "post_merge_offensive_anchor_blend_release_recovery_altitude_trigger_active_fraction": (
            _post_merge_fraction(
                "post_merge_offensive_anchor_blend_release_recovery_altitude_trigger_active"
            )
        ),
        "post_merge_offensive_anchor_blend_release_recovery_altitude_trigger_first_step": (
            _first_post_merge_true_step(
                trajectory,
                "post_merge_offensive_anchor_blend_release_recovery_altitude_trigger_active",
            )
        ),
        "post_merge_offensive_anchor_blend_release_recovery_forward_bias_trigger_active_fraction": (
            _post_merge_fraction(
                "post_merge_offensive_anchor_blend_release_recovery_forward_bias_trigger_active"
            )
        ),
        "post_merge_offensive_anchor_blend_release_recovery_forward_bias_trigger_first_step": (
            _first_post_merge_true_step(
                trajectory,
                "post_merge_offensive_anchor_blend_release_recovery_forward_bias_trigger_active",
            )
        ),
        "post_merge_offensive_anchor_blend_release_vp_forward_bias_clamp_active_fraction": (
            _post_merge_fraction(
                "post_merge_offensive_anchor_blend_release_vp_forward_bias_clamp_active"
            )
        ),
        "post_merge_offensive_anchor_blend_release_vp_forward_bias_clamp_first_step": (
            _first_post_merge_true_step(
                trajectory,
                "post_merge_offensive_anchor_blend_release_vp_forward_bias_clamp_active",
            )
        ),
        "post_merge_offensive_anchor_lateral_world_offset_latch_active_fraction": (
            _post_merge_fraction(
                "post_merge_offensive_anchor_lateral_world_offset_latch_active"
            )
        ),
        "post_merge_offensive_anchor_lateral_world_offset_latch_first_active_step": (
            _first_post_merge_true_step(
                trajectory,
                "post_merge_offensive_anchor_lateral_world_offset_latch_active",
            )
        ),
        "post_merge_offensive_anchor_blend_release_lateral_only_active_fraction": (
            _post_merge_fraction(
                "post_merge_offensive_anchor_blend_release_lateral_only_active"
            )
        ),
        "post_merge_offensive_anchor_blend_release_lateral_only_first_step": (
            _first_post_merge_true_step(
                trajectory,
                "post_merge_offensive_anchor_blend_release_lateral_only_active",
            )
        ),
        "post_merge_tactical_basis_recovery_profile_active_fraction": (
            _post_merge_fraction("post_merge_tactical_basis_recovery_profile_active")
        ),
        "post_merge_tactical_basis_recovery_profile_first_step": (
            _first_post_merge_true_step(
                trajectory,
                "post_merge_tactical_basis_recovery_profile_active",
            )
        ),
        "post_merge_predicted_target_forward_scale_active_fraction": (
            _post_merge_fraction(
                "post_merge_predicted_target_forward_scale_active"
            )
        ),
        "post_merge_predicted_target_forward_scale_first_active_step": (
            _first_post_merge_true_step(
                trajectory,
                "post_merge_predicted_target_forward_scale_active",
            )
        ),
        "post_merge_predicted_target_forward_scale_release_scale_active_fraction": (
            _post_merge_fraction(
                "post_merge_predicted_target_forward_scale_release_scale_active"
            )
        ),
        "post_merge_predicted_target_forward_scale_released_fraction": (
            _post_merge_fraction(
                "post_merge_predicted_target_forward_scale_released"
            )
        ),
        "post_merge_predicted_target_forward_scale_release_first_step": (
            _first_post_merge_true_step(
                trajectory,
                "post_merge_predicted_target_forward_scale_released",
            )
        ),
        "post_merge_offensive_anchor_active_longitudinal_scale": (
            _post_merge_active_mean(
                "longitudinal_scale",
                "post_merge_offensive_anchor_blend_active",
            )
        ),
        "post_merge_offensive_anchor_active_lateral_scale": _post_merge_active_mean(
            "lateral_scale",
            "post_merge_offensive_anchor_blend_active",
        ),
        "post_merge_active_vp_longitudinal_scale": _post_merge_active_mean(
            "longitudinal_scale",
            "post_merge_offensive_anchor_blend_active",
        ),
        "post_merge_active_vp_lateral_scale": _post_merge_active_mean(
            "lateral_scale",
            "post_merge_offensive_anchor_blend_active",
        ),
        "post_merge_offensive_anchor_active_override_longitudinal_scale": (
            _post_merge_active_mean(
                "post_merge_offensive_anchor_longitudinal_scale",
                "post_merge_offensive_anchor_blend_active",
            )
        ),
        "post_merge_offensive_anchor_active_override_lateral_scale": (
            _post_merge_active_mean(
                "post_merge_offensive_anchor_lateral_scale",
                "post_merge_offensive_anchor_blend_active",
            )
        ),
        "pre_merge_vp_forward_bias_m": _finite_mean(
            _safe_float(point.get("vp_forward_bias_m"))
            for point in trajectory
            if _is_pre_merge(point)
        ),
        "post_merge_anchor_mode_active_vp_forward_bias_m": _post_merge_active_mean(
            "vp_forward_bias_m",
            "post_merge_anchor_mode_active",
        ),
        "post_merge_anchor_mode_active_vp_lateral_bias_m": _post_merge_active_mean(
            "vp_lateral_bias_m",
            "post_merge_anchor_mode_active",
        ),
        "post_merge_offensive_anchor_active_vp_forward_bias_m": _post_merge_active_mean(
            "vp_forward_bias_m",
            "post_merge_offensive_anchor_blend_active",
        ),
        "post_merge_offensive_anchor_active_vp_lateral_bias_m": _post_merge_active_mean(
            "vp_lateral_bias_m",
            "post_merge_offensive_anchor_blend_active",
        ),
        "post_merge_predicted_target_forward_scale_active_vp_forward_bias_m": (
            _post_merge_active_mean(
                "vp_forward_bias_m",
                "post_merge_predicted_target_forward_scale_active",
            )
        ),
        "post_merge_predicted_target_forward_scale_active_vp_lateral_bias_m": (
            _post_merge_active_mean(
                "vp_lateral_bias_m",
                "post_merge_predicted_target_forward_scale_active",
            )
        ),
        "pre_merge_mean_tactical_basis_action_ll": _finite_mean(
            _safe_float(point.get("tactical_basis_action_ll"))
            for point in trajectory
            if _is_pre_merge(point)
        ),
        "pre_merge_mean_tactical_basis_action_io": _finite_mean(
            _safe_float(point.get("tactical_basis_action_io"))
            for point in trajectory
            if _is_pre_merge(point)
        ),
        "pre_merge_mean_tactical_basis_action_cd": _finite_mean(
            _safe_float(point.get("tactical_basis_action_cd"))
            for point in trajectory
            if _is_pre_merge(point)
        ),
        "pre_merge_positive_tactical_basis_io_fraction": _finite_mean(
            1.0 if _safe_float(point.get("tactical_basis_action_io")) > 0.0 else 0.0
            for point in trajectory
            if _is_pre_merge(point)
            and np.isfinite(_safe_float(point.get("tactical_basis_action_io")))
        ),
        "post_merge_mean_tactical_basis_action_ll": _finite_mean(
            _safe_float(point.get("tactical_basis_action_ll"))
            for point in post_merge_trajectory
        ),
        "post_merge_mean_tactical_basis_action_io": _finite_mean(
            _safe_float(point.get("tactical_basis_action_io"))
            for point in post_merge_trajectory
        ),
        "vp_forward_bias_m": _finite_mean(
            _safe_float(point.get("vp_forward_bias_m")) for point in trajectory
        ),
        "vp_lateral_bias_m": _finite_mean(
            _safe_float(point.get("vp_lateral_bias_m")) for point in trajectory
        ),
        "merge_min_range_m": _finite_min(
            _safe_float(point.get("range_m")) for point in trajectory
        ),
        "damage_margin": _damage_margin_from_record(record),
    }


def _checkpoint_dimension_info(checkpoint_path: Path) -> Dict[str, Any]:
    if checkpoint_path is None or not checkpoint_path.exists() or str(checkpoint_path) == "dummy":
        return {
            "checkpoint_obs_dim": None,
            "checkpoint_action_dim": None,
            "checkpoint_policy_action_dim": None,
        }
    if checkpoint_path.suffix.lower() == ".zip":
        return {
            "checkpoint_obs_dim": None,
            "checkpoint_action_dim": None,
            "checkpoint_policy_action_dim": None,
        }
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    ckpt_config = checkpoint.get("config", {})
    return {
        "checkpoint_obs_dim": checkpoint.get("obs_dim"),
        "checkpoint_action_dim": checkpoint.get("action_dim"),
        "checkpoint_policy_action_dim": ckpt_config.get("policy", {}).get("action_dim")
        if isinstance(ckpt_config, dict)
        else None,
    }


def _build_observation_audit(
    method_name: str,
    task_name: str,
    method_def: Dict[str, Any],
    config: Dict[str, Any],
    checkpoint_path: Path,
    sample_obs: Dict[str, Any],
) -> Dict[str, Any]:
    obs_vec = sample_obs["observation_vector"]
    schema = copy.deepcopy(sample_obs.get("observation_schema", {}))
    dim_info = _checkpoint_dimension_info(checkpoint_path)
    obs_dim = int(obs_vec.shape[0])
    schema_dim = schema.get("dim")
    low_level_policy_action_dim = int(config.get("policy", {}).get("action_dim", 3))
    policy_action_dim = low_level_policy_action_dim
    if method_def.get("agent_type", "ppo") == "hierarchical_commander":
        commander_cfg = config.get("commander", {})
        configured_num_modes = commander_cfg.get("num_modes")
        modes = commander_cfg.get("modes", [])
        if configured_num_modes is not None:
            policy_action_dim = int(configured_num_modes)
        elif modes:
            policy_action_dim = int(len(modes))
    ckpt_obs_dim = dim_info.get("checkpoint_obs_dim")
    ckpt_action_dim = dim_info.get("checkpoint_action_dim")
    ckpt_policy_action_dim = dim_info.get("checkpoint_policy_action_dim")
    action_dim_matches_checkpoint = ckpt_action_dim in (None, policy_action_dim)
    if (
        method_def.get("agent_type", "ppo") == "hierarchical_commander"
        and ckpt_action_dim is not None
    ):
        action_dim_matches_checkpoint = int(ckpt_action_dim) <= int(policy_action_dim)

    return {
        "method": method_name,
        "task": task_name,
        "agent_type": method_def.get("agent_type", "ppo"),
        "prediction_variant": method_def.get("prediction_variant", "none"),
        "sample_observation_dim": obs_dim,
        "schema_dim": schema_dim,
        "schema_dim_matches_vector": schema_dim in (None, obs_dim),
        "feature_count": len(schema.get("feature_names", [])),
        "checkpoint_obs_dim": ckpt_obs_dim,
        "obs_dim_matches_checkpoint": ckpt_obs_dim in (None, obs_dim),
        "policy_action_dim": policy_action_dim,
        "low_level_policy_action_dim": low_level_policy_action_dim,
        "checkpoint_action_dim": ckpt_action_dim,
        "checkpoint_policy_action_dim": ckpt_policy_action_dim,
        "action_dim_matches_checkpoint": action_dim_matches_checkpoint,
        "observation_schema": schema,
    }


def _run_episode(
    env,
    agent,
    method_name: str,
    method_def: Dict[str, Any],
    task_name: str,
    scenario: Optional[Dict[str, Any]],
    seed: int,
    episode: int,
    run_id: str,
    config: Dict[str, Any],
    git_commit: str,
    run_status: str,
    save_full: bool,
) -> Dict[str, Any]:
    config_hash = config_sha256(config)
    backend = config.get("backend", "jsbsim")
    strict_backend = bool(config.get("env", {}).get("strict_backend", backend == "jsbsim"))
    recorder = EpisodeRecorder(
        run_id=run_id,
        task=task_name,
        controller=method_name,
        seed=seed,
        episode=episode,
        config=config,
        config_sha256=config_hash,
        git_commit=git_commit,
        backend=backend,
        strict_backend=strict_backend,
        save_full=save_full,
    )

    reset_kwargs: Dict[str, Any] = {"seed": seed}
    if scenario is not None:
        reset_kwargs["scenario"] = scenario
    obs = env.reset(**reset_kwargs)
    reset_provenance = copy.deepcopy(obs.get("provenance", {}))
    observation_schema = copy.deepcopy(obs.get("observation_schema", {}))
    attack_zone_cfg = copy.deepcopy(config.get("attack_zone", {}))

    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    info: Dict[str, Any] = {}
    dt = float(env.env_config.get("high_level_dt", 0.2))
    prediction_valid_steps = 0
    prediction_fallback_steps = 0
    prediction_errors: List[float] = []
    ranges_m: List[float] = []
    abs_atas_deg: List[float] = []
    speeds_mps: List[float] = []
    altitudes_m: List[float] = []
    saturation_flags: List[float] = []
    completed_orbits_values: List[float] = []
    combat_time_to_kill = float("nan")
    envelope_steps = 0
    success_range_m = float(config.get("env", {}).get("success_range_m", 900.0))
    success_ata_deg = float(config.get("env", {}).get("success_ata_deg", 25.0))
    backend_fallback_occurred = False

    while not (terminated or truncated):
        # Macro-step: re-select specialist every macro_step steps (for random gate)
        if hasattr(agent, "macro_step") and hasattr(agent, "select_random_specialist") and hasattr(agent, "set_specialist"):
            if steps % agent.macro_step == 0 and steps > 0:
                selected = agent.select_random_specialist()
                agent.set_specialist(selected)
        obs_vec = obs["observation_vector"]
        step_kwargs: Dict[str, Any] = {}
        action = None
        if hasattr(agent, "get_env_step_kwargs"):
            maybe_step_kwargs = agent.get_env_step_kwargs(obs_vec)
            if maybe_step_kwargs is None:
                maybe_step_kwargs = {}
            if not isinstance(maybe_step_kwargs, dict):
                raise TypeError(
                    "agent.get_env_step_kwargs() must return a dict or None, "
                    f"got {type(maybe_step_kwargs).__name__}"
                )
            step_kwargs = dict(maybe_step_kwargs)
            action = step_kwargs.pop("action", None)
        if action is None and not step_kwargs:
            action = agent.get_deterministic_action(obs_vec)
        if (
            action is not None
            and method_def.get("agent_type") == "end_to_end"
            and hasattr(agent, "clip_action")
        ):
            action = agent.clip_action(action)

        if action is None:
            obs, reward, terminated, truncated, info = env.step(**step_kwargs)
        else:
            obs, reward, terminated, truncated, info = env.step(action, **step_kwargs)
        if hasattr(agent, "get_last_step_metadata"):
            step_metadata = agent.get_last_step_metadata()
            if step_metadata:
                info = {**info, **step_metadata}
        total_reward += float(reward)
        steps += 1
        own_state = info.get("own_state", {})
        target_state = info.get("target_state", {})
        recorder.record_step(steps, steps * dt, own_state, target_state, info, float(reward))

        range_m = _safe_float(info.get("range_m"))
        abs_ata_deg = abs(_safe_float(info.get("ata_deg")))
        speed_mps = _safe_float(own_state.get("speed_mps"))
        altitude_m = _safe_float(own_state.get("altitude_m"))
        completed_orbits = _safe_float(info.get("completed_orbits"))
        if np.isfinite(range_m):
            ranges_m.append(range_m)
        if np.isfinite(abs_ata_deg):
            abs_atas_deg.append(abs_ata_deg)
        if np.isfinite(speed_mps):
            speeds_mps.append(speed_mps)
        if np.isfinite(altitude_m):
            altitudes_m.append(altitude_m)
        if np.isfinite(completed_orbits):
            completed_orbits_values.append(completed_orbits)
        saturation_flags.append(1.0 if bool(info.get("saturation_flag", False)) else 0.0)
        if (
            np.isfinite(range_m)
            and np.isfinite(abs_ata_deg)
            and range_m <= success_range_m
            and abs_ata_deg <= success_ata_deg
        ):
            envelope_steps += 1

        if bool(info.get("prediction_valid", False)):
            prediction_valid_steps += 1
        if bool(info.get("prediction_fallback", False)):
            prediction_fallback_steps += 1
        pred_err = info.get("prediction_error_m")
        if pred_err is not None and np.isfinite(pred_err):
            prediction_errors.append(float(pred_err))
        ttk = info.get("combat_time_to_kill")
        if np.isfinite(_safe_float(ttk)) and not np.isfinite(combat_time_to_kill):
            combat_time_to_kill = float(ttk)
        backend_fallback_occurred = backend_fallback_occurred or bool(
            info.get("backend_fallback_occurred", False)
        )

        if steps >= env.max_steps:
            truncated = True
            break

    if not info:
        own_state, target_state = env._get_current_states()
        info = {
            "own_state": own_state,
            "target_state": target_state,
            "reason": "timeout",
            "is_success": False,
        }

    own_state = info.get("own_state", {})
    record = recorder.finalize(
        steps=steps,
        total_time_s=steps * dt,
        total_reward=total_reward,
        termination_reason=info.get("reason", "timeout"),
        success=bool(info.get("is_success", False)),
        final_position_m=own_state.get("position_m", own_state.get("position_neu")),
        final_speed_mps=float(own_state.get("speed_mps", 250.0)),
        final_altitude_m=float(own_state.get("altitude_m", 5000.0)),
    )
    denom = max(1, steps)
    record.update(
        {
            "method_label": method_def.get("label", method_name),
            "agent_type": method_def.get("agent_type", "ppo"),
            "prediction_variant": method_def.get("prediction_variant", "none"),
            "run_status": run_status,
            "mode": run_status,
            "opponent_stage": config.get("opponent_stage", "none"),
            "opponent_config": copy.deepcopy(config.get("opponent", {})),
            "scenario": _scenario_name(scenario, task_name),
            "scenario_metadata": copy.deepcopy(scenario.get("metadata", {})) if scenario else {},
            "observation_schema": observation_schema,
            "reset_provenance": reset_provenance,
            "config_overrides": get_config_overrides(config),
            "attack_zone_enabled": bool(attack_zone_cfg.get("enabled", False)),
            "combat_initial_hp": _safe_float(attack_zone_cfg.get("initial_hp"), 100.0),
            "damage_per_step": _safe_float(attack_zone_cfg.get("damage_per_step")),
            "close_range_max_km": _safe_float(attack_zone_cfg.get("close_range_max_km")),
            "close_range_max_aoa_deg": _safe_float(attack_zone_cfg.get("close_range_max_aoa_deg")),
            "prediction_valid_rate": prediction_valid_steps / denom,
            "prediction_fallback_rate": prediction_fallback_steps / denom,
            "mean_prediction_error_m": _finite_mean(prediction_errors),
            "final_range_m": _last_finite(ranges_m),
            "min_range_m": _finite_min(ranges_m),
            "mean_range_m": _finite_mean(ranges_m),
            "median_range_m": _finite_median(ranges_m),
            "final_abs_ata_deg": _last_finite(abs_atas_deg),
            "mean_abs_ata_deg": _finite_mean(abs_atas_deg),
            "max_abs_ata_deg": _finite_max(abs_atas_deg),
            "time_in_envelope_s": envelope_steps * dt,
            "envelope_fraction": envelope_steps / denom,
            "final_speed_mps": _last_finite(speeds_mps),
            "mean_speed_mps": _finite_mean(speeds_mps),
            "final_altitude_m": _last_finite(altitudes_m),
            "mean_altitude_m": _finite_mean(altitudes_m),
            "saturation_fraction": _finite_mean(saturation_flags),
            "completed_orbits": _last_finite(completed_orbits_values),
            "backend_fallback_occurred": backend_fallback_occurred,
            "backend_fallback_reason": info.get("backend_fallback_reason"),
            "combat_outcome": info.get("combat_outcome"),
            "combat_success": bool(info.get("combat_success", False)),
            "win": bool(info.get("win", False)),
            "loss": bool(info.get("loss", False)),
            "draw": bool(info.get("draw", False)),
            "ego_hp": _safe_float(info.get("ego_hp"), 100.0),
            "target_hp": _safe_float(info.get("target_hp"), 100.0),
            "hp_advantage": _safe_float(info.get("hp_advantage")),
            "combat_reason": info.get("combat_reason"),
            "combat_time_to_kill": combat_time_to_kill,
            "time_to_kill": combat_time_to_kill,
            "survived": not bool(
                info.get("combat_outcome") == "loss"
                and info.get("combat_reason") in {"ego_killed", "ego_crash_or_out_of_bounds"}
            ),
            "_deprecated_tracking_mean_range_m": _finite_mean(ranges_m),
            "_deprecated_tracking_mean_abs_ata_deg": _finite_mean(abs_atas_deg),
            "_deprecated_tracking_envelope_fraction": envelope_steps / denom,
            "final_info": {
                "anchor_mode": info.get("anchor_mode"),
                "guidance_mode": info.get("guidance_mode"),
                "prediction_enabled": info.get("prediction_enabled"),
                "prediction_valid": info.get("prediction_valid"),
                "prediction_fallback": info.get("prediction_fallback"),
                "prediction_noise_applied": info.get("prediction_noise_applied"),
                "prediction_noise_std_m": info.get("prediction_noise_std_m"),
            },
        }
    )
    record.update(summarize_episode_commander(record))
    record.update(summarize_episode_combat_geometry(record))
    return record


def _write_summary_csv(output_dir: Path, records: List[Dict[str, Any]]) -> Path:
    summary_path = output_dir / "summary.csv"
    fields = [
        "method",
        "method_label",
        "agent_type",
        "prediction_variant",
        "run_status",
        "mode",
        "opponent_stage",
        "task",
        "scenario",
        "seed",
        "episode",
        "backend",
        "strict_backend",
        "success",
        "combat_outcome",
        "combat_success",
        "win",
        "loss",
        "draw",
        "termination_reason",
        "attack_zone_enabled",
        "combat_initial_hp",
        "damage_per_step",
        "close_range_max_km",
        "close_range_max_aoa_deg",
        "steps",
        "total_time_s",
        "total_reward",
        "prediction_valid_rate",
        "prediction_fallback_rate",
        "mean_prediction_error_m",
        "final_range_m",
        "min_range_m",
        "mean_range_m",
        "median_range_m",
        "final_abs_ata_deg",
        "mean_abs_ata_deg",
        "max_abs_ata_deg",
        "time_in_envelope_s",
        "envelope_fraction",
        "final_speed_mps",
        "mean_speed_mps",
        "final_altitude_m",
        "mean_altitude_m",
        "saturation_fraction",
        "completed_orbits",
        "ego_hp",
        "target_hp",
        "hp_advantage",
        "combat_time_to_kill",
        "survived",
        "commander_mode_head_on_fraction",
        "commander_mode_crossing_fraction",
        "commander_mode_recovery_fraction",
        "commander_mean_switch_count",
        "commander_first_switch_step",
        "_deprecated_tracking_mean_range_m",
        "_deprecated_tracking_mean_abs_ata_deg",
        "_deprecated_tracking_envelope_fraction",
        "observation_dim",
        "checkpoint_obs_dim",
        "obs_dim_matches_checkpoint",
        "backend_fallback_occurred",
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            row = {
                "method": rec.get("controller"),
                "method_label": rec.get("method_label"),
                "agent_type": rec.get("agent_type"),
                "prediction_variant": rec.get("prediction_variant"),
                "run_status": rec.get("run_status"),
                "mode": rec.get("mode"),
                "opponent_stage": rec.get("opponent_stage"),
                "task": rec.get("task"),
                "scenario": rec.get("scenario"),
                "seed": rec.get("seed"),
                "episode": rec.get("episode"),
                "backend": rec.get("backend"),
                "strict_backend": rec.get("strict_backend"),
                "success": rec.get("success"),
                "combat_outcome": rec.get("combat_outcome"),
                "combat_success": rec.get("combat_success"),
                "win": rec.get("win"),
                "loss": rec.get("loss"),
                "draw": rec.get("draw"),
                "termination_reason": rec.get("termination_reason"),
                "attack_zone_enabled": rec.get("attack_zone_enabled"),
                "combat_initial_hp": rec.get("combat_initial_hp"),
                "damage_per_step": rec.get("damage_per_step"),
                "close_range_max_km": rec.get("close_range_max_km"),
                "close_range_max_aoa_deg": rec.get("close_range_max_aoa_deg"),
                "steps": rec.get("steps"),
                "total_time_s": rec.get("total_time_s"),
                "total_reward": rec.get("total_reward"),
                "prediction_valid_rate": rec.get("prediction_valid_rate"),
                "prediction_fallback_rate": rec.get("prediction_fallback_rate"),
                "mean_prediction_error_m": rec.get("mean_prediction_error_m"),
                "final_range_m": rec.get("final_range_m"),
                "min_range_m": rec.get("min_range_m"),
                "mean_range_m": rec.get("mean_range_m"),
                "median_range_m": rec.get("median_range_m"),
                "final_abs_ata_deg": rec.get("final_abs_ata_deg"),
                "mean_abs_ata_deg": rec.get("mean_abs_ata_deg"),
                "max_abs_ata_deg": rec.get("max_abs_ata_deg"),
                "time_in_envelope_s": rec.get("time_in_envelope_s"),
                "envelope_fraction": rec.get("envelope_fraction"),
                "final_speed_mps": rec.get("final_speed_mps"),
                "mean_speed_mps": rec.get("mean_speed_mps"),
                "final_altitude_m": rec.get("final_altitude_m"),
                "mean_altitude_m": rec.get("mean_altitude_m"),
                "saturation_fraction": rec.get("saturation_fraction"),
                "completed_orbits": rec.get("completed_orbits"),
                "ego_hp": rec.get("ego_hp"),
                "target_hp": rec.get("target_hp"),
                "hp_advantage": rec.get("hp_advantage"),
                "combat_time_to_kill": rec.get("combat_time_to_kill"),
                "survived": rec.get("survived"),
                "commander_mode_head_on_fraction": rec.get("commander_mode_head_on_fraction"),
                "commander_mode_crossing_fraction": rec.get("commander_mode_crossing_fraction"),
                "commander_mode_recovery_fraction": rec.get("commander_mode_recovery_fraction"),
                "commander_mean_switch_count": rec.get("commander_mean_switch_count"),
                "commander_first_switch_step": rec.get("commander_first_switch_step"),
                "_deprecated_tracking_mean_range_m": rec.get("_deprecated_tracking_mean_range_m"),
                "_deprecated_tracking_mean_abs_ata_deg": rec.get("_deprecated_tracking_mean_abs_ata_deg"),
                "_deprecated_tracking_envelope_fraction": rec.get("_deprecated_tracking_envelope_fraction"),
                "observation_dim": rec.get("observation_dim"),
                "checkpoint_obs_dim": rec.get("checkpoint_obs_dim"),
                "obs_dim_matches_checkpoint": rec.get("obs_dim_matches_checkpoint"),
                "backend_fallback_occurred": rec.get("backend_fallback_occurred"),
            }
            writer.writerow(row)
    return summary_path


def _write_aggregate_summary(output_dir: Path, records: List[Dict[str, Any]]) -> Path:
    aggregate: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for rec in records:
        opponent_stage = rec.get("opponent_stage", "none")
        key = (rec["controller"], rec["task"], opponent_stage)
        item = aggregate.setdefault(
            key,
            {
                "method": rec["controller"],
                "task": rec["task"],
                "run_status": rec.get("run_status"),
                "mode": rec.get("mode"),
                "opponent_stage": opponent_stage,
                "attack_zone_enabled": rec.get("attack_zone_enabled"),
                "combat_initial_hp": rec.get("combat_initial_hp"),
                "damage_per_step": rec.get("damage_per_step"),
                "close_range_max_km": rec.get("close_range_max_km"),
                "close_range_max_aoa_deg": rec.get("close_range_max_aoa_deg"),
                "episodes": 0,
                "records": [],
                "successes": 0,
                "total_rewards": [],
                "final_ranges_m": [],
                "min_ranges_m": [],
                "mean_ranges_m": [],
                "mean_abs_atas_deg": [],
                "time_in_envelope_s": [],
                "envelope_fractions": [],
                "saturation_fractions": [],
                "completed_orbits": [],
                "prediction_valid_rates": [],
                "prediction_fallback_rates": [],
                "commander_mode_head_on_fractions": [],
                "commander_mode_crossing_fractions": [],
                "commander_mode_recovery_fractions": [],
                "commander_switch_counts": [],
                "commander_first_switch_steps": [],
                "backend_fallbacks": 0,
                "ego_crashes": 0,
                "target_crash_or_oob": 0,
                "crashes": 0,
                "timeouts": 0,
            },
        )
        item["episodes"] += 1
        item["records"].append(rec)
        item["successes"] += int(bool(rec.get("success", False)))
        item["total_rewards"].append(float(rec.get("total_reward", 0.0)))
        for field, target in (
            ("final_range_m", "final_ranges_m"),
            ("min_range_m", "min_ranges_m"),
            ("mean_range_m", "mean_ranges_m"),
            ("mean_abs_ata_deg", "mean_abs_atas_deg"),
            ("time_in_envelope_s", "time_in_envelope_s"),
            ("envelope_fraction", "envelope_fractions"),
            ("saturation_fraction", "saturation_fractions"),
            ("completed_orbits", "completed_orbits"),
        ):
            item[target].append(_safe_float(rec.get(field)))
        item["prediction_valid_rates"].append(float(rec.get("prediction_valid_rate", 0.0)))
        item["prediction_fallback_rates"].append(float(rec.get("prediction_fallback_rate", 0.0)))
        item["commander_mode_head_on_fractions"].append(
            _safe_float(rec.get("commander_mode_head_on_fraction"))
        )
        item["commander_mode_crossing_fractions"].append(
            _safe_float(rec.get("commander_mode_crossing_fraction"))
        )
        item["commander_mode_recovery_fractions"].append(
            _safe_float(rec.get("commander_mode_recovery_fraction"))
        )
        item["commander_switch_counts"].append(
            _safe_float(rec.get("commander_mean_switch_count"))
        )
        item["commander_first_switch_steps"].append(
            _safe_float(rec.get("commander_first_switch_step"))
        )
        item["backend_fallbacks"] += int(bool(rec.get("backend_fallback_occurred", False)))
        item["ego_crashes"] += int(_record_has_ego_crash(rec))
        item["target_crash_or_oob"] += int(_record_has_target_crash_or_oob(rec))
        item["crashes"] = item["ego_crashes"] + item["target_crash_or_oob"]
        item["timeouts"] += int(_record_has_timeout(rec))

    rows = []
    for item in aggregate.values():
        episodes = max(1, int(item["episodes"]))
        combat_metrics = compute_combat_metrics(item["records"])
        group = _summary_group(
            task=item["task"],
            method=item["method"],
            opponent_stage=item["opponent_stage"],
            damage_per_step=item["damage_per_step"],
            close_range_max_km=item["close_range_max_km"],
            close_range_max_aoa_deg=item["close_range_max_aoa_deg"],
        )
        rows.append(
            {
                "group": group,
                "method": item["method"],
                "task": item["task"],
                "run_status": item["run_status"],
                "mode": item["mode"],
                "opponent_stage": item["opponent_stage"],
                "attack_zone_enabled": item["attack_zone_enabled"],
                "combat_initial_hp": item["combat_initial_hp"],
                "damage_per_step": item["damage_per_step"],
                "close_range_max_km": item["close_range_max_km"],
                "close_range_max_aoa_deg": item["close_range_max_aoa_deg"],
                "episodes": item["episodes"],
                "success_rate": item["successes"] / episodes,
                "combat_success_rate": combat_metrics["combat_success_rate"],
                "win_rate": combat_metrics["win_rate"],
                "survival_rate": combat_metrics["survival_rate"],
                "hp_advantage": combat_metrics["hp_advantage"],
                "mean_time_to_kill": combat_metrics["mean_time_to_kill"],
                "damage_exchange_rate": combat_metrics["damage_exchange_rate"],
                "effective_engagement_rate": combat_metrics["effective_engagement_rate"],
                "damaging_win_rate": combat_metrics["damaging_win_rate"],
                "mean_damage_dealt": combat_metrics["mean_damage_dealt"],
                "mean_damage_taken": combat_metrics["mean_damage_taken"],
                "mean_total_reward": float(np.mean(item["total_rewards"])),
                "mean_final_range_m": _finite_mean(item["final_ranges_m"]),
                "mean_min_range_m": _finite_mean(item["min_ranges_m"]),
                "mean_episode_mean_range_m": _finite_mean(item["mean_ranges_m"]),
                "mean_abs_ata_deg": _finite_mean(item["mean_abs_atas_deg"]),
                "mean_time_in_envelope_s": _finite_mean(item["time_in_envelope_s"]),
                "mean_envelope_fraction": _finite_mean(item["envelope_fractions"]),
                "mean_saturation_fraction": _finite_mean(item["saturation_fractions"]),
                "mean_completed_orbits": _finite_mean(item["completed_orbits"]),
                "mean_prediction_valid_rate": float(np.mean(item["prediction_valid_rates"])),
                "mean_prediction_fallback_rate": float(np.mean(item["prediction_fallback_rates"])),
                "commander_mode_head_on_fraction": _finite_mean(
                    item["commander_mode_head_on_fractions"]
                ),
                "commander_mode_crossing_fraction": _finite_mean(
                    item["commander_mode_crossing_fractions"]
                ),
                "commander_mode_recovery_fraction": _finite_mean(
                    item["commander_mode_recovery_fractions"]
                ),
                "commander_mean_switch_count": _finite_mean(
                    item["commander_switch_counts"]
                ),
                "commander_first_switch_step": _finite_mean(
                    item["commander_first_switch_steps"]
                ),
                "backend_fallbacks": item["backend_fallbacks"],
                "ego_crashes": item["ego_crashes"],
                "target_crash_or_oob": item["target_crash_or_oob"],
                "crashes": item["crashes"],
                "timeouts": item["timeouts"],
            }
        )

    path = output_dir / "aggregate" / "method_task_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, indent=2, ensure_ascii=False)
    return path


def _write_combat_geometry_diagnostics(
    output_dir: Path,
    records: List[Dict[str, Any]],
) -> Path:
    episode_rows: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for rec in records:
        diagnostics = summarize_episode_combat_geometry(rec)
        rec.update(diagnostics)
        trajectory = rec.get("trajectory", [])
        first_step = trajectory[0] if trajectory else {}
        row = {
            "method": rec.get("controller"),
            "task": rec.get("task"),
            "opponent_stage": rec.get("opponent_stage", "none"),
            "run_status": rec.get("run_status"),
            "mode": rec.get("mode"),
            "seed": rec.get("seed"),
            "episode": rec.get("episode"),
            "scenario": rec.get("scenario"),
            "offset_frame": first_step.get("offset_frame"),
            "configured_offset_frame": first_step.get("configured_offset_frame"),
            "virtual_point_source": first_step.get("virtual_point_source"),
            "anchor_mode": first_step.get("anchor_mode"),
            "anchor_mode_requested": first_step.get("anchor_mode_requested"),
            "close_range_anchor_mode": first_step.get("close_range_anchor_mode"),
            "close_range_anchor_trigger_range_m": first_step.get(
                "close_range_anchor_trigger_range_m"
            ),
            "close_range_anchor_release_on_post_merge": bool(
                first_step.get("close_range_anchor_release_on_post_merge", False)
            ),
            "close_range_anchor_requires_first_pass": bool(
                first_step.get("close_range_anchor_requires_first_pass", False)
            ),
            "close_range_anchor_offensive_anchor_blend": first_step.get(
                "close_range_anchor_offensive_anchor_blend"
            ),
            "close_range_anchor_offensive_anchor_blend_active": bool(
                first_step.get("close_range_anchor_offensive_anchor_blend_active", False)
            ),
            "close_range_anchor_post_merge_hold_steps": first_step.get(
                "close_range_anchor_post_merge_hold_steps"
            ),
            "close_range_anchor_alignment_angle_deg_max": first_step.get(
                "close_range_anchor_alignment_angle_deg_max"
            ),
            "close_range_anchor_release_alignment_angle_deg_max": first_step.get(
                "close_range_anchor_release_alignment_angle_deg_max"
            ),
            "offensive_anchor_frame": first_step.get("offensive_anchor_frame"),
            "offensive_anchor_lateral_frame": first_step.get(
                "offensive_anchor_lateral_frame"
            ),
            "offensive_anchor_lateral_sign_mode": first_step.get(
                "offensive_anchor_lateral_sign_mode"
            ),
            "offensive_anchor_lateral_sign": first_step.get(
                "offensive_anchor_lateral_sign"
            ),
            "offensive_anchor_longitudinal_m": first_step.get(
                "offensive_anchor_longitudinal_m"
            ),
            "offensive_anchor_lateral_m": first_step.get("offensive_anchor_lateral_m"),
            "offensive_anchor_vertical_m": first_step.get("offensive_anchor_vertical_m"),
            "post_merge_predicted_target_forward_scale": first_step.get(
                "post_merge_predicted_target_forward_scale"
            ),
            "post_merge_predicted_target_forward_scale_release_scale": first_step.get(
                "post_merge_predicted_target_forward_scale_release_scale"
            ),
            "post_merge_predicted_target_forward_scale_release_ego_only_streak_steps": (
                first_step.get(
                    "post_merge_predicted_target_forward_scale_release_ego_only_streak_steps"
                )
            ),
            "post_merge_predicted_target_forward_scale_release_reset_on_streak_break": bool(
                first_step.get(
                    "post_merge_predicted_target_forward_scale_release_reset_on_streak_break",
                    False,
                )
            ),
            "post_merge_predicted_target_forward_scale_hold_steps": first_step.get(
                "post_merge_predicted_target_forward_scale_hold_steps"
            ),
            "post_merge_anchor_mode": first_step.get("post_merge_anchor_mode"),
            "post_merge_anchor_mode_requires_geometry_disadvantage": bool(
                first_step.get(
                    "post_merge_anchor_mode_requires_geometry_disadvantage",
                    False,
                )
            ),
            "post_merge_anchor_mode_release_ego_only_streak_steps": first_step.get(
                "post_merge_anchor_mode_release_ego_only_streak_steps"
            ),
            "post_merge_anchor_mode_recovery_below_altitude_m": first_step.get(
                "post_merge_anchor_mode_recovery_below_altitude_m"
            ),
            "post_merge_anchor_mode_release_reset_on_streak_break": bool(
                first_step.get(
                    "post_merge_anchor_mode_release_reset_on_streak_break",
                    False,
                )
            ),
            "post_merge_anchor_mode_offensive_anchor_blend": first_step.get(
                "post_merge_anchor_mode_offensive_anchor_blend"
            ),
            "post_merge_anchor_mode_offensive_anchor_longitudinal_blend": first_step.get(
                "post_merge_anchor_mode_offensive_anchor_longitudinal_blend"
            ),
            "post_merge_anchor_mode_offensive_anchor_lateral_blend": first_step.get(
                "post_merge_anchor_mode_offensive_anchor_lateral_blend"
            ),
            "post_merge_anchor_mode_offensive_anchor_blend_active": bool(
                first_step.get(
                    "post_merge_anchor_mode_offensive_anchor_blend_active",
                    False,
                )
            ),
            "post_merge_anchor_mode_offensive_anchor_component_blend_active": bool(
                first_step.get(
                    "post_merge_anchor_mode_offensive_anchor_component_blend_active",
                    False,
                )
            ),
            "post_merge_anchor_mode_lateral_world_offset_latch_on_activation": bool(
                first_step.get(
                    "post_merge_anchor_mode_lateral_world_offset_latch_on_activation",
                    False,
                )
            ),
            "post_merge_offensive_anchor_blend_requires_geometry_disadvantage": bool(
                first_step.get(
                    "post_merge_offensive_anchor_blend_requires_geometry_disadvantage",
                    False,
                )
            ),
            "post_merge_offensive_anchor_blend_release_blend": first_step.get(
                "post_merge_offensive_anchor_blend_release_blend"
            ),
            "post_merge_offensive_anchor_blend_release_ego_only_streak_steps": (
                first_step.get(
                    "post_merge_offensive_anchor_blend_release_ego_only_streak_steps"
                )
            ),
            "post_merge_offensive_anchor_blend_release_reset_on_streak_break": bool(
                first_step.get(
                    "post_merge_offensive_anchor_blend_release_reset_on_streak_break",
                    False,
                )
            ),
            "post_merge_offensive_anchor_blend_release_direct_track_enabled": bool(
                first_step.get(
                    "post_merge_offensive_anchor_blend_release_direct_track_enabled",
                    True,
                )
            ),
            "post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m": (
                first_step.get(
                    "post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m"
                )
            ),
            "post_merge_offensive_anchor_blend_release_recovery_below_altitude_m": (
                first_step.get(
                    "post_merge_offensive_anchor_blend_release_recovery_below_altitude_m"
                )
            ),
            "post_merge_offensive_anchor_blend_release_recovery_longitudinal_blend": (
                first_step.get(
                    "post_merge_offensive_anchor_blend_release_recovery_longitudinal_blend"
                )
            ),
            "post_merge_offensive_anchor_blend_release_recovery_lateral_blend": (
                first_step.get(
                    "post_merge_offensive_anchor_blend_release_recovery_lateral_blend"
                )
            ),
            "post_merge_offensive_anchor_blend_release_recovery_forward_bias_m_max": (
                first_step.get(
                    "post_merge_offensive_anchor_blend_release_recovery_forward_bias_m_max"
                )
            ),
            "post_merge_offensive_anchor_blend_release_vp_forward_bias_m_min": (
                first_step.get(
                    "post_merge_offensive_anchor_blend_release_vp_forward_bias_m_min"
                )
            ),
            "post_merge_offensive_anchor_blend_release_lateral_only_hold_steps": (
                first_step.get(
                    "post_merge_offensive_anchor_blend_release_lateral_only_hold_steps"
                )
            ),
            "effective_guidance_mode": first_step.get("effective_guidance_mode"),
            "mode_switch_effective": any(
                bool(step.get("mode_switch_effective", False))
                for step in trajectory
            ),
            "direct_track_mode_effective": any(
                bool(step.get("direct_track_mode_effective", False))
                for step in trajectory
            ),
            "commander_mode_head_on_fraction": rec.get("commander_mode_head_on_fraction"),
            "commander_mode_crossing_fraction": rec.get(
                "commander_mode_crossing_fraction"
            ),
            "commander_mode_recovery_fraction": rec.get(
                "commander_mode_recovery_fraction"
            ),
            "commander_mean_switch_count": rec.get("commander_mean_switch_count"),
            "commander_first_switch_step": rec.get("commander_first_switch_step"),
            **diagnostics,
        }
        episode_rows.append(row)
        key = (
            str(row["method"]),
            str(row["task"]),
            str(row["opponent_stage"]),
        )
        grouped.setdefault(key, []).append(row)

    rows: List[Dict[str, Any]] = []
    for (method, task, opponent_stage), items in sorted(grouped.items()):
        row = {
            "method": method,
            "task": task,
            "opponent_stage": opponent_stage,
            "episodes": len(items),
            "run_status": items[0].get("run_status"),
            "mode": items[0].get("mode"),
            "offset_frame": items[0].get("offset_frame"),
            "configured_offset_frame": items[0].get("configured_offset_frame"),
            "virtual_point_source": items[0].get("virtual_point_source"),
            "anchor_mode": items[0].get("anchor_mode"),
            "anchor_mode_requested": items[0].get("anchor_mode_requested"),
            "close_range_anchor_mode": items[0].get("close_range_anchor_mode"),
            "close_range_anchor_trigger_range_m": items[0].get(
                "close_range_anchor_trigger_range_m"
            ),
            "close_range_anchor_release_on_post_merge": items[0].get(
                "close_range_anchor_release_on_post_merge"
            ),
            "close_range_anchor_requires_first_pass": items[0].get(
                "close_range_anchor_requires_first_pass"
            ),
            "close_range_anchor_offensive_anchor_blend": items[0].get(
                "close_range_anchor_offensive_anchor_blend"
            ),
            "close_range_anchor_offensive_anchor_blend_active": items[0].get(
                "close_range_anchor_offensive_anchor_blend_active"
            ),
            "close_range_anchor_post_merge_hold_steps": items[0].get(
                "close_range_anchor_post_merge_hold_steps"
            ),
            "close_range_anchor_alignment_angle_deg_max": items[0].get(
                "close_range_anchor_alignment_angle_deg_max"
            ),
            "close_range_anchor_release_alignment_angle_deg_max": items[0].get(
                "close_range_anchor_release_alignment_angle_deg_max"
            ),
            "offensive_anchor_frame": items[0].get("offensive_anchor_frame"),
            "offensive_anchor_lateral_frame": items[0].get(
                "offensive_anchor_lateral_frame"
            ),
            "offensive_anchor_lateral_sign_mode": items[0].get(
                "offensive_anchor_lateral_sign_mode"
            ),
            "offensive_anchor_lateral_sign": items[0].get(
                "offensive_anchor_lateral_sign"
            ),
            "offensive_anchor_longitudinal_m": items[0].get(
                "offensive_anchor_longitudinal_m"
            ),
            "offensive_anchor_lateral_m": items[0].get("offensive_anchor_lateral_m"),
            "offensive_anchor_vertical_m": items[0].get("offensive_anchor_vertical_m"),
            "post_merge_predicted_target_forward_scale": items[0].get(
                "post_merge_predicted_target_forward_scale"
            ),
            "post_merge_predicted_target_forward_scale_release_scale": items[0].get(
                "post_merge_predicted_target_forward_scale_release_scale"
            ),
            "post_merge_predicted_target_forward_scale_release_ego_only_streak_steps": (
                items[0].get(
                    "post_merge_predicted_target_forward_scale_release_ego_only_streak_steps"
                )
            ),
            "post_merge_predicted_target_forward_scale_release_reset_on_streak_break": (
                items[0].get(
                    "post_merge_predicted_target_forward_scale_release_reset_on_streak_break"
                )
            ),
            "post_merge_predicted_target_forward_scale_hold_steps": items[0].get(
                "post_merge_predicted_target_forward_scale_hold_steps"
            ),
            "post_merge_anchor_mode": items[0].get("post_merge_anchor_mode"),
            "post_merge_anchor_mode_requires_geometry_disadvantage": items[0].get(
                "post_merge_anchor_mode_requires_geometry_disadvantage"
            ),
            "post_merge_anchor_mode_release_ego_only_streak_steps": items[0].get(
                "post_merge_anchor_mode_release_ego_only_streak_steps"
            ),
            "post_merge_anchor_mode_recovery_below_altitude_m": items[0].get(
                "post_merge_anchor_mode_recovery_below_altitude_m"
            ),
            "post_merge_anchor_mode_release_reset_on_streak_break": items[0].get(
                "post_merge_anchor_mode_release_reset_on_streak_break"
            ),
            "post_merge_anchor_mode_offensive_anchor_blend": items[0].get(
                "post_merge_anchor_mode_offensive_anchor_blend"
            ),
            "post_merge_anchor_mode_offensive_anchor_longitudinal_blend": items[0].get(
                "post_merge_anchor_mode_offensive_anchor_longitudinal_blend"
            ),
            "post_merge_anchor_mode_offensive_anchor_lateral_blend": items[0].get(
                "post_merge_anchor_mode_offensive_anchor_lateral_blend"
            ),
            "post_merge_anchor_mode_offensive_anchor_blend_active": items[0].get(
                "post_merge_anchor_mode_offensive_anchor_blend_active"
            ),
            "post_merge_anchor_mode_offensive_anchor_component_blend_active": items[0].get(
                "post_merge_anchor_mode_offensive_anchor_component_blend_active"
            ),
            "post_merge_anchor_mode_lateral_world_offset_latch_on_activation": items[
                0
            ].get("post_merge_anchor_mode_lateral_world_offset_latch_on_activation"),
            "post_merge_offensive_anchor_blend_requires_geometry_disadvantage": items[
                0
            ].get("post_merge_offensive_anchor_blend_requires_geometry_disadvantage"),
            "post_merge_offensive_anchor_blend_release_blend": items[0].get(
                "post_merge_offensive_anchor_blend_release_blend"
            ),
            "post_merge_offensive_anchor_blend_release_ego_only_streak_steps": items[
                0
            ].get("post_merge_offensive_anchor_blend_release_ego_only_streak_steps"),
            "post_merge_offensive_anchor_blend_release_reset_on_streak_break": items[
                0
            ].get("post_merge_offensive_anchor_blend_release_reset_on_streak_break"),
            "post_merge_offensive_anchor_blend_release_direct_track_enabled": items[
                0
            ].get("post_merge_offensive_anchor_blend_release_direct_track_enabled"),
            "post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m": (
                items[0].get(
                    "post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m"
                )
            ),
            "post_merge_offensive_anchor_blend_release_recovery_below_altitude_m": (
                items[0].get(
                    "post_merge_offensive_anchor_blend_release_recovery_below_altitude_m"
                )
            ),
            "post_merge_offensive_anchor_blend_release_recovery_longitudinal_blend": (
                items[0].get(
                    "post_merge_offensive_anchor_blend_release_recovery_longitudinal_blend"
                )
            ),
            "post_merge_offensive_anchor_blend_release_recovery_lateral_blend": (
                items[0].get(
                    "post_merge_offensive_anchor_blend_release_recovery_lateral_blend"
                )
            ),
            "post_merge_offensive_anchor_blend_release_recovery_forward_bias_m_max": (
                items[0].get(
                    "post_merge_offensive_anchor_blend_release_recovery_forward_bias_m_max"
                )
            ),
            "post_merge_offensive_anchor_blend_release_vp_forward_bias_m_min": (
                items[0].get(
                    "post_merge_offensive_anchor_blend_release_vp_forward_bias_m_min"
                )
            ),
            "post_merge_offensive_anchor_blend_release_lateral_only_hold_steps": items[
                0
            ].get("post_merge_offensive_anchor_blend_release_lateral_only_hold_steps"),
            "effective_guidance_mode": items[0].get("effective_guidance_mode"),
            "mode_switch_effective": any(
                bool(item.get("mode_switch_effective", False)) for item in items
            ),
            "direct_track_mode_effective": any(
                bool(item.get("direct_track_mode_effective", False))
                for item in items
            ),
            "commander_mode_head_on_fraction": _finite_mean(
                item.get("commander_mode_head_on_fraction") for item in items
            ),
            "commander_mode_crossing_fraction": _finite_mean(
                item.get("commander_mode_crossing_fraction") for item in items
            ),
            "commander_mode_recovery_fraction": _finite_mean(
                item.get("commander_mode_recovery_fraction") for item in items
            ),
            "commander_mean_switch_count": _finite_mean(
                item.get("commander_mean_switch_count") for item in items
            ),
            "commander_first_switch_step": _finite_mean(
                item.get("commander_first_switch_step") for item in items
            ),
        }
        for metric in COMBAT_GEOMETRY_DIAGNOSTIC_METRICS:
            row[metric] = _finite_mean(item.get(metric) for item in items)
        rows.append(row)

    path = output_dir / "aggregate" / "combat_geometry_diagnostics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": list(COMBAT_GEOMETRY_DIAGNOSTIC_METRICS),
                "rows": rows,
                "episode_rows": episode_rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    return path


def _summary_group_value(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, (int, float)) and not np.isfinite(float(value)):
        return "None"
    return str(value)


def _summary_group(
    *,
    task: str,
    method: str,
    opponent_stage: str,
    damage_per_step: Any,
    close_range_max_km: Any,
    close_range_max_aoa_deg: Any,
) -> str:
    return "::".join(
        [
            str(task),
            str(method),
            str(opponent_stage),
            f"d{_summary_group_value(damage_per_step)}",
            f"r{_summary_group_value(close_range_max_km)}",
            f"a{_summary_group_value(close_range_max_aoa_deg)}",
        ]
    )


def _record_reason_text(rec: Dict[str, Any]) -> str:
    return f"{rec.get('termination_reason', '')} {rec.get('combat_reason', '')}".lower()


def _record_has_target_crash_or_oob(rec: Dict[str, Any]) -> bool:
    reason = _record_reason_text(rec)
    return "target_crash_or_out_of_bounds" in reason


def _record_has_ego_crash(rec: Dict[str, Any]) -> bool:
    reason = _record_reason_text(rec)
    if _record_has_target_crash_or_oob(rec):
        return False
    return "crash" in reason or "out_of_bounds" in reason


def _record_has_crash(rec: Dict[str, Any]) -> bool:
    reason = f"{rec.get('termination_reason', '')} {rec.get('combat_reason', '')}".lower()
    return "crash" in reason or "out_of_bounds" in reason


def _record_has_timeout(rec: Dict[str, Any]) -> bool:
    reason = _record_reason_text(rec)
    return "timeout" in reason or bool(rec.get("is_timeout", False))


def summarize_episode_commander(record: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate per-step commander telemetry into episode-level metrics."""
    trajectory = record.get("trajectory", []) or []
    if not trajectory:
        return {
            "commander_mode_head_on_fraction": np.nan,
            "commander_mode_crossing_fraction": np.nan,
            "commander_mode_recovery_fraction": np.nan,
            "commander_mean_switch_count": np.nan,
            "commander_first_switch_step": np.nan,
        }

    selected_specialists = [
        str(step.get("commander_selected_specialist", "")).lower()
        for step in trajectory
        if step.get("commander_selected_specialist") is not None
    ]
    mode_names = [
        str(step.get("commander_mode_name", "")).lower()
        for step in trajectory
        if step.get("commander_mode_name") is not None
    ]
    switch_counts = [
        float(step.get("commander_switch_count"))
        for step in trajectory
        if step.get("commander_switch_count") is not None
    ]
    first_switch_steps = [
        float(step.get("commander_first_switch_step"))
        for step in trajectory
        if step.get("commander_first_switch_step") is not None
    ]
    labels = selected_specialists if selected_specialists else mode_names
    total = max(1, len(labels))
    return {
        "commander_mode_head_on_fraction": (
            sum("head_on" in name for name in labels) / total if labels else np.nan
        ),
        "commander_mode_crossing_fraction": (
            sum("crossing" in name for name in labels) / total if labels else np.nan
        ),
        "commander_mode_recovery_fraction": (
            sum("recovery" in name for name in labels) / total if labels else np.nan
        ),
        "commander_mean_switch_count": switch_counts[-1] if switch_counts else np.nan,
        "commander_first_switch_step": first_switch_steps[0] if first_switch_steps else np.nan,
    }


def _write_config_snapshots(
    output_dir: Path,
    configs: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    snapshots: Dict[str, Dict[str, Any]] = {}
    for key, cfg in sorted(configs.items()):
        safe_key = key.replace("/", "__")
        path = manifest_dir / f"config_{safe_key}.yaml"
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        snapshots[key] = {
            "path": str(path),
            "sha256": config_sha256(cfg),
            "config_overrides": get_config_overrides(cfg),
        }
    return snapshots


def _checkpoint_checks(
    comparison_config: Dict[str, Any],
    methods: List[str],
    opponent_stage: str,
    allow_missing: bool,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    checks = []
    errors = []
    for method in methods:
        method_def = comparison_config.get("methods", {}).get(method, {})
        agent_type = method_def.get("agent_type", "ppo")
        # oracle_task_gate, random_task_gate, and rule_guidance do not require a single policy checkpoint
        if agent_type in {"oracle_task_gate", "random_task_gate", "rule_guidance"}:
            continue
        paths = [("policy_checkpoint", method_def.get("checkpoint"))]
        variant = method_def.get("prediction_variant")
        if variant in ("learned", "noisy"):
            pred_path = method_def.get(
                "predictor_checkpoint",
                comparison_config.get("prediction", {}).get("learned_predictor_checkpoint"),
            )
            paths.append(("predictor_checkpoint", pred_path))

        for role, raw_path in paths:
            path = _repo_path(raw_path)
            exists = bool(path and path.exists())
            checks.append(
                {
                    "method": method,
                    "role": role,
                    "path": str(path) if path else None,
                    "exists": exists,
                }
            )
            if not exists and not allow_missing:
                errors.append(f"{method}:{role} missing: {path}")
    opponent_entry = _resolve_opponent_entry(comparison_config, opponent_stage)
    opponent_checkpoint = opponent_entry.get("checkpoint")
    if opponent_checkpoint:
        path = _repo_path(opponent_checkpoint)
        exists = bool(path and path.exists())
        checks.append(
            {
                "method": "__opponent__",
                "role": f"opponent_{opponent_stage}_checkpoint",
                "path": str(path) if path else None,
                "exists": exists,
            }
        )
        if not exists and not allow_missing:
            errors.append(f"opponent:{opponent_stage}:checkpoint missing: {path}")
    return checks, errors


def _contract_for_run(dry_run: bool) -> ArtifactContract:
    if dry_run:
        return ArtifactContract(
            required_files=[
                "resolved_config.yaml",
                "dry_run_checks.json",
                "design_notes.json",
                "artifact_contract.json",
                "run_manifest.json",
            ],
            required_directories=["manifests"],
        )
    return ArtifactContract(
        required_files=[
            "resolved_config.yaml",
            "summary.csv",
            "aggregate/episode_records.json",
            "aggregate/method_task_summary.json",
            "aggregate/combat_geometry_diagnostics.json",
            "design_notes.json",
            "observation_audit.json",
            "artifact_contract.json",
            "run_manifest.json",
        ],
        required_directories=["raw", "manifests"],
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal JSBSim HRL/VPP comparison runner")
    parser.add_argument("--config", default="config/experiment/jsbsim_hrl_comparison.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", default="outputs/jsbsim_hrl_comparison")
    parser.add_argument("--preset", choices=["main", "ablation"], default="main")
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--n-episodes", type=int, default=None)
    parser.add_argument("--backend", choices=["jsbsim", "simple"], default=None)
    parser.add_argument(
        "--opponent-stage",
        type=str,
        default="none",
        choices=["none", "expert", "end_to_end", "curriculum"],
    )
    parser.add_argument(
        "--jsbsim-root",
        default=None,
        help="Legacy project root containing envs/JSBSim/data; recorded as env.legacy_project_root.",
    )
    parser.add_argument(
        "--attack-zone-enabled",
        choices=["auto", "true", "false"],
        default="auto",
        help="Override attack_zone.enabled. auto keeps the config/opponent-stage behavior.",
    )
    parser.add_argument("--attack-zone-initial-hp", type=float, default=None)
    parser.add_argument("--attack-zone-damage-per-step", type=float, default=None)
    parser.add_argument("--attack-zone-close-range-max-km", type=float, default=None)
    parser.add_argument(
        "--attack-zone-close-range-max-aoa-deg",
        type=float,
        default=None,
        help="Explicit close-range attack-zone angle threshold in degrees.",
    )
    parser.add_argument(
        "--prediction-lookahead-time-s",
        type=float,
        default=None,
        help="Override trajectory_prediction.prediction.lookahead_time_s for prediction-enabled methods.",
    )
    parser.add_argument(
        "--vpp-offset-frame",
        choices=["world_neu", "target_velocity", "los_relative"],
        default=None,
        help="Explicitly set virtual_point.offset_frame for VPP methods.",
    )
    parser.add_argument(
        "--vpp-offset-frame-by-task",
        default=None,
        help=(
            "Task-conditioned virtual_point.offset_frame mapping, for example "
            "'head_on=target_velocity,crossing_feasible=world_neu'."
        ),
    )
    parser.add_argument(
        "--vpp-predicted-target-blend",
        type=float,
        default=None,
        help=(
            "Blend factor in [0, 1] for predicted_target anchors. "
            "1.0 keeps the full predicted anchor and 0.0 falls back to current_target."
        ),
    )
    parser.add_argument(
        "--vpp-predicted-target-blend-by-task",
        default=None,
        help=(
            "Task-conditioned predicted-target anchor blend mapping, for example "
            "'head_on=0.35,crossing_feasible=1.0'."
        ),
    )
    parser.add_argument(
        "--vpp-predicted-target-forward-scale",
        type=float,
        default=None,
        help=(
            "Scale in [0, 1] applied to the target-velocity-frame forward "
            "component of predicted-target anchors before post-merge logic."
        ),
    )
    parser.add_argument(
        "--vpp-predicted-target-forward-scale-by-task",
        default=None,
        help=(
            "Task-conditioned predicted-target forward-scale mapping, for example "
            "'head_on=0.0,crossing_feasible=1.0'."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-predicted-target-blend",
        type=float,
        default=None,
        help=(
            "Optional predicted-target blend in [0, 1] that only applies "
            "after first pass is complete."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-predicted-target-blend-by-task",
        default=None,
        help=(
            "Task-conditioned post-merge predicted-target blend mapping, for example "
            "'head_on=0.5,crossing_feasible=1.0'."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-offensive-anchor-blend",
        type=float,
        default=None,
        help=(
            "Optional blend in [0, 1] from the active post-merge anchor toward "
            "the offensive-position anchor."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-offensive-anchor-blend-by-task",
        default=None,
        help=(
            "Task-conditioned post-merge offensive-anchor blend mapping, for "
            "example 'head_on=0.25,crossing_feasible=0.0'."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-offensive-anchor-longitudinal-scale",
        type=float,
        default=None,
        help=(
            "Optional longitudinal action scale override that only applies "
            "while the post-merge offensive-anchor blend is active."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-offensive-anchor-longitudinal-scale-by-task",
        default=None,
        help=(
            "Task-conditioned longitudinal action scale override that only "
            "applies while the post-merge offensive-anchor blend is active, "
            "for example 'head_on=0.25,crossing_feasible=1.0'."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-offensive-anchor-lateral-scale",
        type=float,
        default=None,
        help=(
            "Optional lateral action scale override that only applies while "
            "the post-merge offensive-anchor blend is active."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-offensive-anchor-lateral-scale-by-task",
        default=None,
        help=(
            "Task-conditioned lateral action scale override that only applies "
            "while the post-merge offensive-anchor blend is active, for "
            "example 'head_on=1.0,crossing_feasible=1.0'."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-offensive-anchor-blend-requires-geometry-disadvantage",
        choices=["true", "false"],
        default=None,
        help=(
            "Whether post-merge offensive-anchor blending should activate only "
            "after target-side attack-zone / score disadvantage appears."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-offensive-anchor-blend-requires-geometry-disadvantage-by-task",
        default=None,
        help=(
            "Task-conditioned boolean mapping for gating post-merge offensive-"
            "anchor blending on target-side geometry disadvantage, for example "
            "'head_on=true,crossing_feasible=false'."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-offensive-anchor-geometry-disadvantage-aa-deg-min",
        type=float,
        default=None,
        help=(
            "Optional AA threshold for early post-merge offensive-anchor "
            "geometry-disadvantage activation."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-offensive-anchor-geometry-disadvantage-aa-deg-min-by-task",
        default=None,
        help=(
            "Task-conditioned AA threshold mapping for early post-merge "
            "offensive-anchor geometry disadvantage, for example "
            "'head_on=170.0'."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-offensive-anchor-blend-release-blend",
        type=float,
        default=None,
        help=(
            "Optional offensive-anchor blend in [0, 1] to use after the "
            "stronger post-merge offensive-anchor blend releases."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-offensive-anchor-blend-release-blend-by-task",
        default=None,
        help=(
            "Task-conditioned release-blend mapping for post-merge offensive-"
            "anchor blending, for example 'head_on=0.0,crossing_feasible=0.0'."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-offensive-anchor-blend-release-ego-only-streak-steps",
        type=int,
        default=None,
        help=(
            "Release the stronger post-merge offensive-anchor blend after this "
            "many consecutive post-merge ego-only attack-zone steps."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-offensive-anchor-blend-release-ego-only-streak-steps-by-task",
        default=None,
        help=(
            "Task-conditioned consecutive post-merge ego-only attack-zone streak "
            "needed to release the stronger offensive-anchor blend, for example "
            "'head_on=5,crossing_feasible=0'."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-offensive-anchor-blend-release-reset-on-streak-break",
        choices=["true", "false"],
        default=None,
        help=(
            "Whether to re-arm the stronger post-merge offensive-anchor blend "
            "once the post-merge ego-only attack-zone streak breaks after release."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-offensive-anchor-blend-release-reset-on-streak-break-by-task",
        default=None,
        help=(
            "Task-conditioned boolean mapping for re-arming the stronger post-"
            "merge offensive-anchor blend when the ego-only attack-zone streak "
            "breaks, for example 'head_on=true,crossing_feasible=false'."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-offensive-anchor-blend-release-direct-track-below-altitude-m",
        type=float,
        default=None,
        help=(
            "If set, then after the stronger post-merge offensive-anchor blend "
            "has released, request direct-track recovery once the ego altitude "
            "drops below this threshold while the fight is opening and neither "
            "aircraft is currently in the attack zone."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-offensive-anchor-blend-release-direct-track-below-altitude-m-by-task",
        default=None,
        help=(
            "Task-conditioned altitude threshold mapping for release-phase "
            "direct-track recovery after the stronger offensive-anchor blend "
            "has released, for example 'head_on=800,crossing_feasible=0'."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-offensive-anchor-blend-release-direct-track-enabled",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=None,
        help=(
            "Global switch to enable or disable the post-merge offensive-anchor "
            "blend release direct-track path. When false, the direct-track "
            "recovery is never requested even if the altitude threshold is set."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-offensive-anchor-blend-release-direct-track-enabled-by-task",
        default=None,
        help=(
            "Task-conditioned boolean mapping for the direct-track enabled switch, "
            "for example 'head_on=false,crossing_feasible=true'."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-offensive-anchor-blend-release-recovery-below-altitude-m",
        type=float,
        default=None,
        help=(
            "Optional altitude threshold for activating the shallow post-release "
            "rear-quarter recovery stage while the fight is opening and neither "
            "aircraft is currently in the attack zone."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-offensive-anchor-blend-release-recovery-below-altitude-m-by-task",
        default=None,
        help=(
            "Task-conditioned altitude threshold mapping for the post-release "
            "rear-quarter recovery stage, for example "
            "'head_on=4500,crossing_feasible=0'."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-offensive-anchor-blend-release-recovery-longitudinal-blend",
        type=float,
        default=None,
        help=(
            "Optional longitudinal offensive-anchor component blend in [0, 1] "
            "used by the post-release rear-quarter recovery stage."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-offensive-anchor-blend-release-recovery-longitudinal-blend-by-task",
        default=None,
        help=(
            "Task-conditioned longitudinal component blend mapping for the "
            "post-release rear-quarter recovery stage, for example "
            "'head_on=0.0,crossing_feasible=0.0'."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-offensive-anchor-blend-release-recovery-lateral-blend",
        type=float,
        default=None,
        help=(
            "Optional lateral offensive-anchor component blend in [0, 1] used "
            "by the post-release rear-quarter recovery stage."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-offensive-anchor-blend-release-recovery-lateral-blend-by-task",
        default=None,
        help=(
            "Task-conditioned lateral component blend mapping for the "
            "post-release rear-quarter recovery stage, for example "
            "'head_on=0.25,crossing_feasible=0.0'."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-offensive-anchor-blend-release-recovery-forward-bias-m-max",
        type=float,
        default=None,
        help=(
            "Optional target-velocity forward-bias threshold in meters for "
            "activating the post-release rear-quarter recovery stage before the "
            "altitude guard trips. More negative values require deeper overdrive."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-offensive-anchor-blend-release-recovery-forward-bias-m-max-by-task",
        default=None,
        help=(
            "Task-conditioned forward-bias threshold mapping for the post-release "
            "rear-quarter recovery stage, for example 'head_on=-4500'."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-offensive-anchor-blend-release-vp-forward-bias-m-min",
        type=float,
        default=None,
        help=(
            "Optional minimum allowed target-velocity forward bias in meters for "
            "the released post-merge offensive-anchor path. More negative values "
            "than this threshold are clamped without entering recovery."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-offensive-anchor-blend-release-vp-forward-bias-m-min-by-task",
        default=None,
        help=(
            "Task-conditioned minimum forward-bias clamp mapping for the released "
            "post-merge offensive-anchor path, for example 'head_on=-4500'."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-offensive-anchor-blend-release-vp-forward-bias-clamp-negative-lateral-below-altitude-m",
        type=float,
        default=None,
        help=(
            "Optional altitude gate in meters for the released forward-bias clamp. "
            "When the clamp candidate still has negative target-velocity lateral "
            "bias, clamp activation is deferred until own altitude is at or below "
            "this threshold."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-offensive-anchor-blend-release-vp-forward-bias-clamp-negative-lateral-below-altitude-m-by-task",
        default=None,
        help=(
            "Task-conditioned altitude gate mapping for deferring the released "
            "forward-bias clamp while target-velocity lateral bias is still "
            "negative, for example 'head_on=4500'."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-offensive-anchor-lateral-world-offset-latch-on-activation",
        choices=["true", "false"],
        default=None,
        help=(
            "Whether to latch the offensive-anchor lateral world offset once the "
            "post-merge offensive-anchor blend first activates."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-offensive-anchor-lateral-world-offset-latch-on-activation-by-task",
        default=None,
        help=(
            "Task-conditioned boolean mapping for latching the offensive-anchor "
            "lateral world offset when the post-merge offensive-anchor blend "
            "activates, for example 'head_on=true,crossing_feasible=false'."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-offensive-anchor-blend-release-lateral-only",
        choices=["true", "false"],
        default=None,
        help=(
            "Whether to keep only the latched offensive-anchor lateral body "
            "after the stronger post-merge offensive-anchor blend releases."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-offensive-anchor-blend-release-lateral-only-by-task",
        default=None,
        help=(
            "Task-conditioned boolean mapping for keeping only the latched "
            "offensive-anchor lateral body after the stronger post-merge "
            "offensive-anchor blend releases, for example "
            "'head_on=true,crossing_feasible=false'."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-offensive-anchor-blend-release-lateral-only-hold-steps",
        type=int,
        default=None,
        help=(
            "Optional nonnegative step hold window for keeping only the latched "
            "offensive-anchor lateral body after release before reverting to the "
            "neutral no-direct-track path."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-offensive-anchor-blend-release-lateral-only-hold-steps-by-task",
        default=None,
        help=(
            "Task-conditioned nonnegative integer mapping for the lateral-only "
            "post-release hold window, for example "
            "'head_on=60,crossing_feasible=0'."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-predicted-target-forward-scale",
        type=float,
        default=None,
        help=(
            "Optional post-merge scale in [0, 1] applied only to the "
            "target-velocity-frame forward component of the predicted-target anchor."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-predicted-target-forward-scale-by-task",
        default=None,
        help=(
            "Task-conditioned post-merge predicted-target forward-scale mapping, "
            "for example 'head_on=0.0,crossing_feasible=1.0'."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-predicted-target-forward-scale-release-scale",
        type=float,
        default=None,
        help=(
            "Optional predicted-target forward scale in [0, 1] to use after the "
            "reduced post-merge forward scale releases."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-predicted-target-forward-scale-release-scale-by-task",
        default=None,
        help=(
            "Task-conditioned release-scale mapping for post-merge predicted-target "
            "forward scale, for example 'head_on=0.5,crossing_feasible=1.0'."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-predicted-target-forward-scale-release-ego-only-streak-steps",
        type=int,
        default=None,
        help=(
            "Release the reduced post-merge predicted-target forward scale after "
            "this many consecutive post-merge ego-only attack-zone steps."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-predicted-target-forward-scale-release-ego-only-streak-steps-by-task",
        default=None,
        help=(
            "Task-conditioned consecutive post-merge ego-only attack-zone streak "
            "needed to release the reduced predicted-target forward scale, for "
            "example 'head_on=35,crossing_feasible=0'."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-predicted-target-forward-scale-release-reset-on-streak-break",
        choices=["true", "false"],
        default=None,
        help=(
            "Whether to re-arm the reduced post-merge predicted-target forward scale "
            "once the post-merge ego-only attack-zone streak breaks after release."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-predicted-target-forward-scale-release-reset-on-streak-break-by-task",
        default=None,
        help=(
            "Task-conditioned boolean mapping for re-arming the reduced post-merge "
            "predicted-target forward scale when the ego-only attack-zone streak "
            "breaks, for example 'head_on=true,crossing_feasible=false'."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-predicted-target-forward-scale-hold-steps",
        type=int,
        default=None,
        help=(
            "Limit the reduced post-merge predicted-target forward scale to this "
            "many high-level steps after first pass."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-predicted-target-forward-scale-hold-steps-by-task",
        default=None,
        help=(
            "Task-conditioned hold-step mapping for the reduced post-merge "
            "predicted-target forward scale, for example "
            "'head_on=60,crossing_feasible=0'."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-predicted-target-blend-release-on-attack-zone",
        choices=["true", "false"],
        default=None,
        help=(
            "Whether a reduced post-merge predicted-target blend should stop once "
            "either aircraft enters the attack zone after first pass."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-predicted-target-blend-release-on-attack-zone-by-task",
        default=None,
        help=(
            "Task-conditioned boolean mapping for releasing the reduced post-merge "
            "predicted-target blend on first post-merge attack-zone contact, for "
            "example 'head_on=true,crossing_feasible=false'."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-predicted-target-blend-release-requires-target-attack-zone",
        choices=["true", "false"],
        default=None,
        help=(
            "Whether post-merge blend release should wait for target attack-zone "
            "exposure instead of any attack-zone contact."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-predicted-target-blend-release-requires-target-attack-zone-by-task",
        default=None,
        help=(
            "Task-conditioned boolean mapping for waiting until the target enters "
            "the attack zone before releasing the reduced post-merge predicted-"
            "target blend, for example 'head_on=true,crossing_feasible=false'."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-predicted-target-blend-release-below-altitude-m",
        type=float,
        default=None,
        help=(
            "Release the reduced post-merge predicted-target blend once own "
            "altitude drops below this threshold."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-predicted-target-blend-release-below-altitude-m-by-task",
        default=None,
        help=(
            "Task-conditioned altitude threshold mapping for releasing the reduced "
            "post-merge predicted-target blend, for example "
            "'head_on=2000,crossing_feasible=0'."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-predicted-target-blend-hold-steps",
        type=int,
        default=None,
        help=(
            "Limit the reduced post-merge predicted-target blend to this many "
            "high-level steps after first pass."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-predicted-target-blend-hold-steps-by-task",
        default=None,
        help=(
            "Task-conditioned hold-step mapping for the reduced post-merge "
            "predicted-target blend, for example 'head_on=200,crossing_feasible=0'."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-predicted-target-blend-release-blend",
        type=float,
        default=None,
        help=(
            "Optional predicted-target blend in [0, 1] to use after a reduced "
            "post-merge predicted-target blend releases or expires."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-predicted-target-blend-release-blend-by-task",
        default=None,
        help=(
            "Task-conditioned post-release predicted-target blend mapping, for example "
            "'head_on=0.75,crossing_feasible=1.0'."
        ),
    )
    parser.add_argument(
        "--vpp-longitudinal-scale",
        type=float,
        default=None,
        help="Scale the local VPP longitudinal offset component. 1.0 keeps current behavior.",
    )
    parser.add_argument(
        "--vpp-longitudinal-scale-by-task",
        default=None,
        help=(
            "Task-conditioned longitudinal offset scale mapping, for example "
            "'head_on=0.0,crossing_feasible=1.0'."
        ),
    )
    parser.add_argument(
        "--vpp-lateral-scale",
        type=float,
        default=None,
        help="Scale the local VPP lateral offset component. 1.0 keeps current behavior.",
    )
    parser.add_argument(
        "--vpp-lateral-scale-by-task",
        default=None,
        help=(
            "Task-conditioned local VPP lateral offset scaling, for example "
            "'head_on=0.0,crossing_feasible=1.0'."
        ),
    )
    parser.add_argument(
        "--vpp-offensive-anchor-blend",
        type=float,
        default=None,
        help=(
            "Blend the current anchor toward an offensive behind-target anchor. "
            "0 keeps current behavior and 1 uses the offensive anchor fully."
        ),
    )
    parser.add_argument(
        "--vpp-offensive-anchor-blend-by-task",
        default=None,
        help=(
            "Task-conditioned offensive-anchor blend mapping, for example "
            "'head_on=0.25,crossing_feasible=0.0'."
        ),
    )
    parser.add_argument(
        "--vpp-offensive-anchor-longitudinal-m",
        type=float,
        default=None,
        help=(
            "Distance in meters to place the offensive-position anchor behind the "
            "target along the target-velocity axis."
        ),
    )
    parser.add_argument(
        "--vpp-offensive-anchor-longitudinal-m-by-task",
        default=None,
        help=(
            "Task-conditioned offensive anchor lag distance mapping in meters, for "
            "example 'head_on=800,crossing_feasible=0'."
        ),
    )
    parser.add_argument(
        "--vpp-offensive-anchor-frame",
        choices=list(VALID_OFFSET_FRAMES),
        default=None,
        help=(
            "Frame used to construct offensive_position anchors. Defaults to "
            "target_velocity for backward compatibility."
        ),
    )
    parser.add_argument(
        "--vpp-offensive-anchor-frame-by-task",
        default=None,
        help=(
            "Task-conditioned offensive anchor frame mapping, for example "
            "'head_on=encounter,crossing_feasible=target_velocity'."
        ),
    )
    parser.add_argument(
        "--vpp-offensive-anchor-encounter-stable-max-heading-delta-deg",
        type=float,
        default=None,
        help=(
            "Clamp angle in degrees used when offensive_anchor_frame is "
            "encounter_stable. Lower values keep the rear-quarter axis closer "
            "to target motion."
        ),
    )
    parser.add_argument(
        "--vpp-offensive-anchor-encounter-stable-max-heading-delta-deg-by-task",
        default=None,
        help=(
            "Task-conditioned encounter_stable clamp angle mapping in degrees, "
            "for example 'head_on=20,crossing_feasible=45'."
        ),
    )
    parser.add_argument(
        "--vpp-offensive-anchor-lateral-frame",
        choices=list(VALID_OFFSET_FRAMES),
        default=None,
        help=(
            "Optional frame used only for the lateral component of offensive "
            "anchors. Defaults to the offensive anchor frame."
        ),
    )
    parser.add_argument(
        "--vpp-offensive-anchor-lateral-frame-by-task",
        default=None,
        help=(
            "Task-conditioned offensive anchor lateral frame mapping, for example "
            "'head_on=encounter,crossing_feasible=target_velocity'."
        ),
    )
    parser.add_argument(
        "--vpp-offensive-anchor-lateral-sign-mode",
        choices=list(VALID_OFFENSIVE_ANCHOR_LATERAL_SIGN_MODES),
        default=None,
        help=(
            "How offensive-anchor lateral side is chosen: same_side keeps the "
            "legacy rear-quarter side, while fixed_positive/fixed_negative force "
            "a stable side sign."
        ),
    )
    parser.add_argument(
        "--vpp-offensive-anchor-lateral-sign-mode-by-task",
        default=None,
        help=(
            "Task-conditioned offensive anchor lateral sign mode mapping, for "
            "example 'head_on=fixed_positive,crossing_feasible=same_side'."
        ),
    )
    parser.add_argument(
        "--vpp-offensive-anchor-lateral-m",
        type=float,
        default=None,
        help=(
            "Same-side rear-quarter lateral offset in meters for offensive_position "
            "anchors. Zero keeps the legacy straight-behind anchor."
        ),
    )
    parser.add_argument(
        "--vpp-offensive-anchor-lateral-m-by-task",
        default=None,
        help=(
            "Task-conditioned offensive anchor lateral distance mapping in meters, "
            "for example 'head_on=400,crossing_feasible=0'."
        ),
    )
    parser.add_argument(
        "--vpp-offensive-anchor-vertical-m",
        type=float,
        default=None,
        help=(
            "Vertical high-side offset in meters for offensive_position anchors. "
            "Positive values place the anchor above the target."
        ),
    )
    parser.add_argument(
        "--vpp-offensive-anchor-vertical-m-by-task",
        default=None,
        help=(
            "Task-conditioned offensive anchor vertical distance mapping in meters, "
            "for example 'head_on=500,crossing_feasible=0'."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-anchor-mode",
        choices=list(VALID_VPP_ANCHOR_MODES),
        default=None,
        help="Anchor mode to activate after first pass is complete.",
    )
    parser.add_argument(
        "--vpp-post-merge-anchor-mode-by-task",
        default=None,
        help=(
            "Task-conditioned post-merge anchor mode mapping, for example "
            "'head_on=current_target,crossing_feasible=predicted_target'."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-anchor-mode-requires-geometry-disadvantage",
        default=None,
        help=(
            "Require the existing post-merge geometry gate before activating "
            "the configured post-merge anchor mode. Accepts true or false."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-anchor-mode-requires-geometry-disadvantage-by-task",
        default=None,
        help=(
            "Task-conditioned post-merge anchor-mode gate mapping, for example "
            "'head_on=true,crossing_feasible=false'."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-anchor-mode-recovery-below-altitude-m",
        type=float,
        default=None,
        help=(
            "If set, the tactical post-merge anchor mode only activates in a "
            "late reopening-recovery window: own altitude at or below this "
            "threshold, range opening, and neither aircraft currently in the "
            "attack zone."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-anchor-mode-recovery-below-altitude-m-by-task",
        default=None,
        help=(
            "Task-conditioned altitude threshold mapping for the tactical post-"
            "merge anchor-mode reopening-recovery window, for example "
            "'head_on=4500,crossing_feasible=0'."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-anchor-mode-release-ego-only-streak-steps",
        type=int,
        default=None,
        help=(
            "Release the tactical post-merge anchor mode after this many "
            "consecutive post-merge ego-only attack-zone steps."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-anchor-mode-release-ego-only-streak-steps-by-task",
        default=None,
        help=(
            "Task-conditioned consecutive post-merge ego-only attack-zone streak "
            "needed to release the tactical post-merge anchor mode, for example "
            "'head_on=1,crossing_feasible=0'."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-anchor-mode-release-reset-on-streak-break",
        choices=["true", "false"],
        default=None,
        help=(
            "Whether to re-arm the tactical post-merge anchor mode when the "
            "ego-only attack-zone streak breaks after release."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-anchor-mode-release-reset-on-streak-break-by-task",
        default=None,
        help=(
            "Task-conditioned boolean mapping for re-arming the tactical post-"
            "merge anchor mode when the ego-only attack-zone streak breaks, for "
            "example 'head_on=false,crossing_feasible=false'."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-anchor-mode-offensive-anchor-blend",
        type=float,
        default=None,
        help=(
            "Optional blend in [0, 1] for expressing the tactical post-merge "
            "anchor mode as a soft pull toward the configured offensive anchor "
            "while preserving the underlying predicted-target anchor."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-anchor-mode-offensive-anchor-blend-by-task",
        default=None,
        help=(
            "Task-conditioned blend mapping for soft tactical post-merge anchor "
            "mode activation, for example 'head_on=0.25,crossing_feasible=0'."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-anchor-mode-offensive-anchor-longitudinal-blend",
        type=float,
        default=None,
        help=(
            "Optional longitudinal-only blend in [0, 1] for the soft tactical "
            "post-merge anchor mode. Use this to control rear-lag depth "
            "independently from lateral beam offset."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-anchor-mode-offensive-anchor-longitudinal-blend-by-task",
        default=None,
        help=(
            "Task-conditioned longitudinal-only blend mapping for the soft "
            "tactical post-merge anchor mode, for example "
            "'head_on=1.0,crossing_feasible=0'."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-anchor-mode-offensive-anchor-lateral-blend",
        type=float,
        default=None,
        help=(
            "Optional lateral-only blend in [0, 1] for the soft tactical "
            "post-merge anchor mode. Use this to control beam/side offset "
            "independently from rear-lag depth."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-anchor-mode-offensive-anchor-lateral-blend-by-task",
        default=None,
        help=(
            "Task-conditioned lateral-only blend mapping for the soft tactical "
            "post-merge anchor mode, for example "
            "'head_on=0.25,crossing_feasible=0'."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-anchor-mode-lateral-world-offset-latch-on-activation",
        choices=("true", "false"),
        default=None,
        help=(
            "Whether to latch the offensive-anchor lateral world offset at the "
            "moment the tactical post-merge anchor mode first activates. This "
            "keeps the rear-quarter side/body from rotating with later frame "
            "updates."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-anchor-mode-lateral-world-offset-latch-on-activation-by-task",
        default=None,
        help=(
            "Task-conditioned boolean mapping for latching the tactical "
            "post-merge anchor-mode lateral world offset on first activation, "
            "for example 'head_on=true,crossing_feasible=false'."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-anchor-mode-longitudinal-scale",
        type=float,
        default=None,
        help=(
            "Optional longitudinal policy-offset scale to apply only while the "
            "tactical post-merge anchor mode is active. Use 0 to suppress "
            "forward/back action drift and let the anchor geometry take over."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-anchor-mode-longitudinal-scale-by-task",
        default=None,
        help=(
            "Task-conditioned longitudinal policy-offset scale mapping for the "
            "tactical post-merge anchor mode, for example "
            "'head_on=0,crossing_feasible=1'."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-anchor-mode-lateral-scale",
        type=float,
        default=None,
        help=(
            "Optional lateral policy-offset scale to apply only while the "
            "tactical post-merge anchor mode is active. Use 0 to suppress "
            "beam/bracket drift and keep the VP near the selected anchor."
        ),
    )
    parser.add_argument(
        "--vpp-post-merge-anchor-mode-lateral-scale-by-task",
        default=None,
        help=(
            "Task-conditioned lateral policy-offset scale mapping for the "
            "tactical post-merge anchor mode, for example "
            "'head_on=0,crossing_feasible=1'."
        ),
    )
    parser.add_argument(
        "--vpp-close-range-anchor-mode",
        choices=list(VALID_VPP_ANCHOR_MODES),
        default=None,
        help="Anchor mode to activate once the engagement enters the merge band.",
    )
    parser.add_argument(
        "--vpp-close-range-anchor-mode-by-task",
        default=None,
        help=(
            "Task-conditioned close-range anchor mode mapping, for example "
            "'head_on=current_target,crossing_feasible=predicted_target'."
        ),
    )
    parser.add_argument(
        "--vpp-close-range-anchor-trigger-range-m",
        type=float,
        default=None,
        help=(
            "Range threshold in meters that activates close-range anchor switching. "
            "Defaults to combat_diagnostics.merge_range_m."
        ),
    )
    parser.add_argument(
        "--vpp-close-range-anchor-trigger-range-m-by-task",
        default=None,
        help=(
            "Task-conditioned close-range anchor trigger range mapping in meters, "
            "for example 'head_on=150,crossing_feasible=1000'."
        ),
    )
    parser.add_argument(
        "--vpp-close-range-anchor-alignment-angle-deg-max",
        type=float,
        default=None,
        help=(
            "Optional max(|ATA|, |AA|) gate in degrees for latching close-range "
            "anchor switching. When omitted, no alignment gate is applied."
        ),
    )
    parser.add_argument(
        "--vpp-close-range-anchor-alignment-angle-deg-max-by-task",
        default=None,
        help=(
            "Task-conditioned close-range anchor alignment gate in degrees, "
            "for example 'head_on=10,crossing_feasible=180'."
        ),
    )
    parser.add_argument(
        "--vpp-close-range-anchor-release-on-post-merge",
        choices=["true", "false"],
        default=None,
        help=(
            "Whether a latched close-range anchor should release once first pass "
            "is complete."
        ),
    )
    parser.add_argument(
        "--vpp-close-range-anchor-release-on-post-merge-by-task",
        default=None,
        help=(
            "Task-conditioned boolean mapping for releasing close-range anchor "
            "after first pass, for example 'head_on=true,crossing_feasible=false'."
        ),
    )
    parser.add_argument(
        "--vpp-close-range-anchor-requires-first-pass",
        choices=["true", "false"],
        default=None,
        help=(
            "Whether a latched close-range anchor may only activate after first "
            "pass is complete."
        ),
    )
    parser.add_argument(
        "--vpp-close-range-anchor-requires-first-pass-by-task",
        default=None,
        help=(
            "Task-conditioned boolean mapping for post-first-pass-only close-range "
            "anchor activation, for example "
            "'head_on=true,crossing_feasible=false'."
        ),
    )
    parser.add_argument(
        "--vpp-close-range-anchor-offensive-anchor-blend",
        type=float,
        default=None,
        help=(
            "Optional close-range tactical blend toward the offensive anchor when "
            "the configured close-range anchor mode is offensive_position."
        ),
    )
    parser.add_argument(
        "--vpp-close-range-anchor-offensive-anchor-blend-by-task",
        default=None,
        help=(
            "Task-conditioned close-range offensive-anchor blend mapping, for "
            "example 'head_on=0.25,crossing_feasible=0.0'."
        ),
    )
    parser.add_argument(
        "--vpp-close-range-anchor-release-alignment-angle-deg-max",
        type=float,
        default=None,
        help=(
            "Optional post-merge max(|ATA|, |AA|) threshold in degrees that must "
            "be satisfied before a released close-range anchor actually falls back "
            "to the base anchor mode."
        ),
    )
    parser.add_argument(
        "--vpp-close-range-anchor-release-alignment-angle-deg-max-by-task",
        default=None,
        help=(
            "Task-conditioned post-merge release alignment gate in degrees, for "
            "example 'head_on=120,crossing_feasible=180'."
        ),
    )
    parser.add_argument(
        "--vpp-close-range-anchor-post-merge-hold-steps",
        type=int,
        default=None,
        help=(
            "Number of extra post-merge high-level steps to keep the close-range "
            "anchor active before releasing it."
        ),
    )
    parser.add_argument(
        "--vpp-close-range-anchor-post-merge-hold-steps-by-task",
        default=None,
        help=(
            "Task-conditioned post-merge hold-step mapping, for example "
            "'head_on=1,crossing_feasible=0'."
        ),
    )
    parser.add_argument("--run-status", choices=["smoke", "formal-small", "formal"], default="smoke")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--no-trajectory", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-missing-checkpoints",
        action="store_true",
        help="Only for dry-run planning on machines without trained artifacts.",
    )
    return parser.parse_args()


def _collect_attack_zone_cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    if args.attack_zone_enabled != "auto":
        overrides["enabled"] = args.attack_zone_enabled == "true"
    if args.attack_zone_initial_hp is not None:
        overrides["initial_hp"] = float(args.attack_zone_initial_hp)
    if args.attack_zone_damage_per_step is not None:
        overrides["damage_per_step"] = float(args.attack_zone_damage_per_step)
    if args.attack_zone_close_range_max_km is not None:
        overrides["close_range_max_km"] = float(args.attack_zone_close_range_max_km)
    if args.attack_zone_close_range_max_aoa_deg is not None:
        overrides["close_range_max_aoa_deg"] = float(args.attack_zone_close_range_max_aoa_deg)
    return overrides


def _collect_prediction_cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    if args.prediction_lookahead_time_s is not None:
        overrides["lookahead_time_s"] = float(args.prediction_lookahead_time_s)
    return overrides


def _parse_task_frame_overrides(
    raw_value: Optional[str],
    *,
    option_name: str = "--vpp-offset-frame-by-task",
) -> Dict[str, str]:
    if raw_value in (None, ""):
        return {}
    mapping: Dict[str, str] = {}
    for item in str(raw_value).split(","):
        entry = item.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(f"{option_name} entries must be task=frame pairs")
        task_name, frame = (part.strip() for part in entry.split("=", 1))
        if not task_name:
            raise ValueError(f"{option_name} entries must include a task name")
        if frame not in VALID_OFFSET_FRAMES:
            raise ValueError(
                f"Unknown VPP frame {frame!r} for {option_name}; "
                f"expected one of {sorted(VALID_OFFSET_FRAMES)}"
            )
        mapping[task_name] = frame
    return mapping


def _parse_task_offensive_anchor_lateral_sign_mode_overrides(
    raw_value: Optional[str],
    *,
    option_name: str = "--vpp-offensive-anchor-lateral-sign-mode-by-task",
) -> Dict[str, str]:
    if raw_value in (None, ""):
        return {}
    mapping: Dict[str, str] = {}
    for item in str(raw_value).split(","):
        entry = item.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(f"{option_name} entries must be task=mode pairs")
        task_name, mode = (part.strip() for part in entry.split("=", 1))
        if not task_name:
            raise ValueError(f"{option_name} entries must include a task name")
        if mode not in VALID_OFFENSIVE_ANCHOR_LATERAL_SIGN_MODES:
            raise ValueError(
                f"Unknown VPP lateral sign mode {mode!r} for {option_name}; "
                f"expected one of {sorted(VALID_OFFENSIVE_ANCHOR_LATERAL_SIGN_MODES)}"
            )
        mapping[task_name] = mode
    return mapping


def _parse_task_blend_overrides(
    raw_value: Optional[str],
    *,
    option_name: str = "--vpp-predicted-target-blend-by-task",
) -> Dict[str, float]:
    if raw_value in (None, ""):
        return {}
    mapping: Dict[str, float] = {}
    for item in str(raw_value).split(","):
        entry = item.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(f"{option_name} entries must be task=blend pairs")
        task_name, blend_raw = (part.strip() for part in entry.split("=", 1))
        if not task_name:
            raise ValueError(f"{option_name} entries must include a task name")
        blend = float(blend_raw)
        if not np.isfinite(blend) or blend < 0.0 or blend > 1.0:
            raise ValueError(
                f"Invalid {option_name} blend {blend_raw!r}; expected a finite value in [0, 1]"
            )
        mapping[task_name] = blend
    return mapping


def _parse_task_nonnegative_scale_overrides(
    raw_value: Optional[str],
    *,
    option_name: str,
) -> Dict[str, float]:
    if raw_value in (None, ""):
        return {}
    mapping: Dict[str, float] = {}
    for item in str(raw_value).split(","):
        entry = item.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(f"{option_name} entries must be task=scale pairs")
        task_name, scale_raw = (part.strip() for part in entry.split("=", 1))
        if not task_name:
            raise ValueError(f"{option_name} entries must include a task name")
        scale = float(scale_raw)
        if not np.isfinite(scale) or scale < 0.0:
            raise ValueError(
                f"Invalid VPP lateral scale {scale_raw!r}; expected a finite value >= 0"
            )
        mapping[task_name] = scale
    return mapping


def _parse_task_nonnegative_float_overrides(
    raw_value: Optional[str],
    *,
    option_name: str,
) -> Dict[str, float]:
    if raw_value in (None, ""):
        return {}
    mapping: Dict[str, float] = {}
    for item in str(raw_value).split(","):
        entry = item.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(f"{option_name} entries must be task=value pairs")
        task_name, value_raw = (part.strip() for part in entry.split("=", 1))
        if not task_name:
            raise ValueError(f"{option_name} entries must include a task name")
        value = float(value_raw)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(
                f"Invalid {option_name} value {value_raw!r}; expected a finite value >= 0"
            )
        mapping[task_name] = value
    return mapping


def _parse_task_float_overrides(
    raw_value: Optional[str],
    *,
    option_name: str,
) -> Dict[str, float]:
    if raw_value in (None, ""):
        return {}
    mapping: Dict[str, float] = {}
    for item in str(raw_value).split(","):
        entry = item.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(f"{option_name} entries must be task=value pairs")
        task_name, value_raw = (part.strip() for part in entry.split("=", 1))
        if not task_name:
            raise ValueError(f"{option_name} entries must include a task name")
        value = float(value_raw)
        if not np.isfinite(value):
            raise ValueError(
                f"Invalid {option_name} value {value_raw!r}; expected a finite value"
            )
        mapping[task_name] = value
    return mapping


def _parse_task_anchor_mode_overrides(
    raw_value: Optional[str],
    *,
    option_name: str,
) -> Dict[str, str]:
    if raw_value in (None, ""):
        return {}
    mapping: Dict[str, str] = {}
    for item in str(raw_value).split(","):
        entry = item.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(f"{option_name} entries must be task=anchor_mode pairs")
        task_name, mode = (part.strip() for part in entry.split("=", 1))
        if not task_name:
            raise ValueError(f"{option_name} entries must include a task name")
        if mode not in VALID_VPP_ANCHOR_MODES:
            raise ValueError(
                f"Unknown VPP anchor mode {mode!r}; expected one of {list(VALID_VPP_ANCHOR_MODES)}"
            )
        mapping[task_name] = mode
    return mapping


def _parse_boolean_literal(raw_value: str, *, option_name: str) -> bool:
    normalized = str(raw_value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{option_name} values must be true or false")


def _parse_task_boolean_overrides(
    raw_value: Optional[str],
    *,
    option_name: str,
) -> Dict[str, bool]:
    if raw_value in (None, ""):
        return {}
    mapping: Dict[str, bool] = {}
    for item in str(raw_value).split(","):
        entry = item.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(f"{option_name} entries must be task=bool pairs")
        task_name, bool_raw = (part.strip() for part in entry.split("=", 1))
        if not task_name:
            raise ValueError(f"{option_name} entries must include a task name")
        mapping[task_name] = _parse_boolean_literal(
            bool_raw,
            option_name=option_name,
        )
    return mapping


def _parse_task_nonnegative_int_overrides(
    raw_value: Optional[str],
    *,
    option_name: str,
) -> Dict[str, int]:
    if raw_value in (None, ""):
        return {}
    mapping: Dict[str, int] = {}
    for item in str(raw_value).split(","):
        entry = item.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(f"{option_name} entries must be task=int pairs")
        task_name, int_raw = (part.strip() for part in entry.split("=", 1))
        if not task_name:
            raise ValueError(f"{option_name} entries must include a task name")
        value = int(int_raw)
        if float(int_raw) != float(value) or value < 0:
            raise ValueError(
                f"Invalid {option_name} value {int_raw!r}; expected a nonnegative integer"
            )
        mapping[task_name] = value
    return mapping


def _collect_vpp_cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    if args.vpp_offset_frame is not None:
        overrides["offset_frame"] = str(args.vpp_offset_frame)
    if args.vpp_offset_frame_by_task is not None:
        overrides["offset_frame_by_task"] = _parse_task_frame_overrides(
            args.vpp_offset_frame_by_task
        )
    if args.vpp_predicted_target_blend is not None:
        blend = float(args.vpp_predicted_target_blend)
        if not np.isfinite(blend) or blend < 0.0 or blend > 1.0:
            raise ValueError(
                "--vpp-predicted-target-blend must be a finite value in [0, 1]"
            )
        overrides["predicted_target_blend"] = blend
    if args.vpp_predicted_target_blend_by_task is not None:
        overrides["predicted_target_blend_by_task"] = _parse_task_blend_overrides(
            args.vpp_predicted_target_blend_by_task
        )
    if args.vpp_predicted_target_forward_scale is not None:
        scale = float(args.vpp_predicted_target_forward_scale)
        if not np.isfinite(scale) or scale < 0.0 or scale > 1.0:
            raise ValueError(
                "--vpp-predicted-target-forward-scale must be a finite value in [0, 1]"
            )
        overrides["predicted_target_forward_scale"] = scale
    if args.vpp_predicted_target_forward_scale_by_task is not None:
        overrides["predicted_target_forward_scale_by_task"] = (
            _parse_task_blend_overrides(
                args.vpp_predicted_target_forward_scale_by_task
            )
        )
    if args.vpp_post_merge_predicted_target_blend is not None:
        blend = float(args.vpp_post_merge_predicted_target_blend)
        if not np.isfinite(blend) or blend < 0.0 or blend > 1.0:
            raise ValueError(
                "--vpp-post-merge-predicted-target-blend must be a finite value in [0, 1]"
            )
        overrides["post_merge_predicted_target_blend"] = blend
    if args.vpp_post_merge_predicted_target_blend_by_task is not None:
        overrides["post_merge_predicted_target_blend_by_task"] = (
            _parse_task_blend_overrides(
                args.vpp_post_merge_predicted_target_blend_by_task
            )
        )
    if args.vpp_post_merge_offensive_anchor_blend is not None:
        blend = float(args.vpp_post_merge_offensive_anchor_blend)
        if not np.isfinite(blend) or blend < 0.0 or blend > 1.0:
            raise ValueError(
                "--vpp-post-merge-offensive-anchor-blend must be a finite value in [0, 1]"
            )
        overrides["post_merge_offensive_anchor_blend"] = blend
    if args.vpp_post_merge_offensive_anchor_blend_by_task is not None:
        overrides["post_merge_offensive_anchor_blend_by_task"] = (
            _parse_task_blend_overrides(
                args.vpp_post_merge_offensive_anchor_blend_by_task
            )
        )
    if args.vpp_post_merge_offensive_anchor_longitudinal_scale is not None:
        scale = float(args.vpp_post_merge_offensive_anchor_longitudinal_scale)
        if not np.isfinite(scale) or scale < 0.0:
            raise ValueError(
                "--vpp-post-merge-offensive-anchor-longitudinal-scale must be a finite value >= 0"
            )
        overrides["post_merge_offensive_anchor_longitudinal_scale"] = scale
    if (
        args.vpp_post_merge_offensive_anchor_longitudinal_scale_by_task
        is not None
    ):
        overrides["post_merge_offensive_anchor_longitudinal_scale_by_task"] = (
            _parse_task_nonnegative_scale_overrides(
                args.vpp_post_merge_offensive_anchor_longitudinal_scale_by_task,
                option_name=(
                    "--vpp-post-merge-offensive-anchor-longitudinal-scale-by-task"
                ),
            )
        )
    if args.vpp_post_merge_offensive_anchor_lateral_scale is not None:
        scale = float(args.vpp_post_merge_offensive_anchor_lateral_scale)
        if not np.isfinite(scale) or scale < 0.0:
            raise ValueError(
                "--vpp-post-merge-offensive-anchor-lateral-scale must be a finite value >= 0"
            )
        overrides["post_merge_offensive_anchor_lateral_scale"] = scale
    if args.vpp_post_merge_offensive_anchor_lateral_scale_by_task is not None:
        overrides["post_merge_offensive_anchor_lateral_scale_by_task"] = (
            _parse_task_nonnegative_scale_overrides(
                args.vpp_post_merge_offensive_anchor_lateral_scale_by_task,
                option_name=(
                    "--vpp-post-merge-offensive-anchor-lateral-scale-by-task"
                ),
            )
        )
    if args.vpp_post_merge_offensive_anchor_blend_requires_geometry_disadvantage is not None:
        overrides[
            "post_merge_offensive_anchor_blend_requires_geometry_disadvantage"
        ] = _parse_boolean_literal(
            args.vpp_post_merge_offensive_anchor_blend_requires_geometry_disadvantage,
            option_name=(
                "--vpp-post-merge-offensive-anchor-blend-requires-geometry-disadvantage"
            ),
        )
    if (
        args.vpp_post_merge_offensive_anchor_blend_requires_geometry_disadvantage_by_task
        is not None
    ):
        overrides[
            "post_merge_offensive_anchor_blend_requires_geometry_disadvantage_by_task"
        ] = _parse_task_boolean_overrides(
            args.vpp_post_merge_offensive_anchor_blend_requires_geometry_disadvantage_by_task,
            option_name=(
                "--vpp-post-merge-offensive-anchor-blend-requires-geometry-disadvantage-by-task"
            ),
        )
    if args.vpp_post_merge_offensive_anchor_geometry_disadvantage_aa_deg_min is not None:
        aa_deg_min = float(
            args.vpp_post_merge_offensive_anchor_geometry_disadvantage_aa_deg_min
        )
        if not np.isfinite(aa_deg_min) or aa_deg_min < 0.0:
            raise ValueError(
                "--vpp-post-merge-offensive-anchor-geometry-disadvantage-aa-deg-min "
                "must be a finite value >= 0"
            )
        overrides["post_merge_offensive_anchor_geometry_disadvantage_aa_deg_min"] = (
            aa_deg_min
        )
    if (
        args.vpp_post_merge_offensive_anchor_geometry_disadvantage_aa_deg_min_by_task
        is not None
    ):
        overrides["post_merge_offensive_anchor_geometry_disadvantage_aa_deg_min_by_task"] = (
            _parse_task_nonnegative_float_overrides(
                args.vpp_post_merge_offensive_anchor_geometry_disadvantage_aa_deg_min_by_task,
                option_name=(
                    "--vpp-post-merge-offensive-anchor-geometry-disadvantage-aa-deg-min-by-task"
                ),
            )
        )
    if args.vpp_post_merge_offensive_anchor_blend_release_blend is not None:
        blend = float(args.vpp_post_merge_offensive_anchor_blend_release_blend)
        if not np.isfinite(blend) or blend < 0.0 or blend > 1.0:
            raise ValueError(
                "--vpp-post-merge-offensive-anchor-blend-release-blend must be "
                "a finite value in [0, 1]"
            )
        overrides["post_merge_offensive_anchor_blend_release_blend"] = blend
    if args.vpp_post_merge_offensive_anchor_blend_release_blend_by_task is not None:
        overrides["post_merge_offensive_anchor_blend_release_blend_by_task"] = (
            _parse_task_blend_overrides(
                args.vpp_post_merge_offensive_anchor_blend_release_blend_by_task
            )
        )
    if (
        args.vpp_post_merge_offensive_anchor_blend_release_ego_only_streak_steps
        is not None
    ):
        streak_steps = int(
            args.vpp_post_merge_offensive_anchor_blend_release_ego_only_streak_steps
        )
        if streak_steps < 0:
            raise ValueError(
                "--vpp-post-merge-offensive-anchor-blend-release-ego-only-streak-steps "
                "must be a nonnegative integer"
            )
        overrides[
            "post_merge_offensive_anchor_blend_release_ego_only_streak_steps"
        ] = streak_steps
    if (
        args.vpp_post_merge_offensive_anchor_blend_release_ego_only_streak_steps_by_task
        is not None
    ):
        overrides[
            "post_merge_offensive_anchor_blend_release_ego_only_streak_steps_by_task"
        ] = _parse_task_nonnegative_int_overrides(
            args.vpp_post_merge_offensive_anchor_blend_release_ego_only_streak_steps_by_task,
            option_name=(
                "--vpp-post-merge-offensive-anchor-blend-release-ego-only-streak-steps-by-task"
            ),
        )
    if (
        args.vpp_post_merge_offensive_anchor_blend_release_reset_on_streak_break
        is not None
    ):
        overrides[
            "post_merge_offensive_anchor_blend_release_reset_on_streak_break"
        ] = _parse_boolean_literal(
            args.vpp_post_merge_offensive_anchor_blend_release_reset_on_streak_break,
            option_name=(
                "--vpp-post-merge-offensive-anchor-blend-release-reset-on-streak-break"
            ),
        )
    if (
        args.vpp_post_merge_offensive_anchor_blend_release_reset_on_streak_break_by_task
        is not None
    ):
        overrides[
            "post_merge_offensive_anchor_blend_release_reset_on_streak_break_by_task"
        ] = _parse_task_boolean_overrides(
            args.vpp_post_merge_offensive_anchor_blend_release_reset_on_streak_break_by_task,
            option_name=(
                "--vpp-post-merge-offensive-anchor-blend-release-reset-on-streak-break-by-task"
            ),
        )
    if (
        args.vpp_post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m
        is not None
    ):
        threshold = float(
            args.vpp_post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m
        )
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError(
                "--vpp-post-merge-offensive-anchor-blend-release-direct-track-below-altitude-m "
                "must be a finite value >= 0"
            )
        overrides[
            "post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m"
        ] = threshold
    if (
        args.vpp_post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m_by_task
        is not None
    ):
        overrides[
            "post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m_by_task"
        ] = _parse_task_nonnegative_float_overrides(
            args.vpp_post_merge_offensive_anchor_blend_release_direct_track_below_altitude_m_by_task,
            option_name=(
                "--vpp-post-merge-offensive-anchor-blend-release-direct-track-below-altitude-m-by-task"
            ),
        )
    if args.vpp_post_merge_offensive_anchor_blend_release_direct_track_enabled is not None:
        overrides[
            "post_merge_offensive_anchor_blend_release_direct_track_enabled"
        ] = bool(args.vpp_post_merge_offensive_anchor_blend_release_direct_track_enabled)
    if (
        args.vpp_post_merge_offensive_anchor_blend_release_direct_track_enabled_by_task
        is not None
    ):
        overrides[
            "post_merge_offensive_anchor_blend_release_direct_track_enabled_by_task"
        ] = _parse_task_boolean_overrides(
            args.vpp_post_merge_offensive_anchor_blend_release_direct_track_enabled_by_task,
            option_name=(
                "--vpp-post-merge-offensive-anchor-blend-release-direct-track-enabled-by-task"
            ),
        )
    if (
        args.vpp_post_merge_offensive_anchor_blend_release_recovery_below_altitude_m
        is not None
    ):
        threshold = float(
            args.vpp_post_merge_offensive_anchor_blend_release_recovery_below_altitude_m
        )
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError(
                "--vpp-post-merge-offensive-anchor-blend-release-recovery-below-altitude-m "
                "must be a finite value >= 0"
            )
        overrides[
            "post_merge_offensive_anchor_blend_release_recovery_below_altitude_m"
        ] = threshold
    if (
        args.vpp_post_merge_offensive_anchor_blend_release_recovery_below_altitude_m_by_task
        is not None
    ):
        overrides[
            "post_merge_offensive_anchor_blend_release_recovery_below_altitude_m_by_task"
        ] = _parse_task_nonnegative_float_overrides(
            args.vpp_post_merge_offensive_anchor_blend_release_recovery_below_altitude_m_by_task,
            option_name=(
                "--vpp-post-merge-offensive-anchor-blend-release-recovery-below-altitude-m-by-task"
            ),
        )
    if (
        args.vpp_post_merge_offensive_anchor_blend_release_recovery_longitudinal_blend
        is not None
    ):
        blend = float(
            args.vpp_post_merge_offensive_anchor_blend_release_recovery_longitudinal_blend
        )
        if not np.isfinite(blend) or blend < 0.0 or blend > 1.0:
            raise ValueError(
                "--vpp-post-merge-offensive-anchor-blend-release-recovery-longitudinal-blend "
                "must be a finite value in [0, 1]"
            )
        overrides[
            "post_merge_offensive_anchor_blend_release_recovery_longitudinal_blend"
        ] = blend
    if (
        args.vpp_post_merge_offensive_anchor_blend_release_recovery_longitudinal_blend_by_task
        is not None
    ):
        overrides[
            "post_merge_offensive_anchor_blend_release_recovery_longitudinal_blend_by_task"
        ] = _parse_task_blend_overrides(
            args.vpp_post_merge_offensive_anchor_blend_release_recovery_longitudinal_blend_by_task
        )
    if (
        args.vpp_post_merge_offensive_anchor_blend_release_recovery_lateral_blend
        is not None
    ):
        blend = float(
            args.vpp_post_merge_offensive_anchor_blend_release_recovery_lateral_blend
        )
        if not np.isfinite(blend) or blend < 0.0 or blend > 1.0:
            raise ValueError(
                "--vpp-post-merge-offensive-anchor-blend-release-recovery-lateral-blend "
                "must be a finite value in [0, 1]"
            )
        overrides[
            "post_merge_offensive_anchor_blend_release_recovery_lateral_blend"
        ] = blend
    if (
        args.vpp_post_merge_offensive_anchor_blend_release_recovery_lateral_blend_by_task
        is not None
    ):
        overrides[
            "post_merge_offensive_anchor_blend_release_recovery_lateral_blend_by_task"
        ] = _parse_task_blend_overrides(
            args.vpp_post_merge_offensive_anchor_blend_release_recovery_lateral_blend_by_task
        )
    if (
        args.vpp_post_merge_offensive_anchor_blend_release_recovery_forward_bias_m_max
        is not None
    ):
        threshold = float(
            args.vpp_post_merge_offensive_anchor_blend_release_recovery_forward_bias_m_max
        )
        if not np.isfinite(threshold):
            raise ValueError(
                "--vpp-post-merge-offensive-anchor-blend-release-recovery-forward-bias-m-max "
                "must be a finite value"
            )
        overrides[
            "post_merge_offensive_anchor_blend_release_recovery_forward_bias_m_max"
        ] = threshold
    if (
        args.vpp_post_merge_offensive_anchor_blend_release_recovery_forward_bias_m_max_by_task
        is not None
    ):
        overrides[
            "post_merge_offensive_anchor_blend_release_recovery_forward_bias_m_max_by_task"
        ] = _parse_task_float_overrides(
            args.vpp_post_merge_offensive_anchor_blend_release_recovery_forward_bias_m_max_by_task,
            option_name=(
                "--vpp-post-merge-offensive-anchor-blend-release-recovery-forward-bias-m-max-by-task"
            ),
        )
    if (
        args.vpp_post_merge_offensive_anchor_blend_release_vp_forward_bias_m_min
        is not None
    ):
        threshold = float(
            args.vpp_post_merge_offensive_anchor_blend_release_vp_forward_bias_m_min
        )
        if not np.isfinite(threshold):
            raise ValueError(
                "--vpp-post-merge-offensive-anchor-blend-release-vp-forward-bias-m-min "
                "must be a finite value"
            )
        overrides[
            "post_merge_offensive_anchor_blend_release_vp_forward_bias_m_min"
        ] = threshold
    if (
        args.vpp_post_merge_offensive_anchor_blend_release_vp_forward_bias_m_min_by_task
        is not None
    ):
        overrides[
            "post_merge_offensive_anchor_blend_release_vp_forward_bias_m_min_by_task"
        ] = _parse_task_float_overrides(
            args.vpp_post_merge_offensive_anchor_blend_release_vp_forward_bias_m_min_by_task,
            option_name=(
                "--vpp-post-merge-offensive-anchor-blend-release-vp-forward-bias-m-min-by-task"
            ),
        )
    if (
        args.vpp_post_merge_offensive_anchor_blend_release_vp_forward_bias_clamp_negative_lateral_below_altitude_m
        is not None
    ):
        altitude_m = float(
            args.vpp_post_merge_offensive_anchor_blend_release_vp_forward_bias_clamp_negative_lateral_below_altitude_m
        )
        if not np.isfinite(altitude_m):
            raise ValueError(
                "--vpp-post-merge-offensive-anchor-blend-release-vp-forward-bias-clamp-negative-lateral-below-altitude-m "
                "must be a finite value"
            )
        overrides[
            "post_merge_offensive_anchor_blend_release_vp_forward_bias_clamp_negative_lateral_below_altitude_m"
        ] = altitude_m
    if (
        args.vpp_post_merge_offensive_anchor_blend_release_vp_forward_bias_clamp_negative_lateral_below_altitude_m_by_task
        is not None
    ):
        overrides[
            "post_merge_offensive_anchor_blend_release_vp_forward_bias_clamp_negative_lateral_below_altitude_m_by_task"
        ] = _parse_task_float_overrides(
            args.vpp_post_merge_offensive_anchor_blend_release_vp_forward_bias_clamp_negative_lateral_below_altitude_m_by_task,
            option_name=(
                "--vpp-post-merge-offensive-anchor-blend-release-vp-forward-bias-clamp-negative-lateral-below-altitude-m-by-task"
            ),
        )
    if (
        args.vpp_post_merge_offensive_anchor_lateral_world_offset_latch_on_activation
        is not None
    ):
        overrides[
            "post_merge_offensive_anchor_lateral_world_offset_latch_on_activation"
        ] = _parse_boolean_literal(
            args.vpp_post_merge_offensive_anchor_lateral_world_offset_latch_on_activation,
            option_name=(
                "--vpp-post-merge-offensive-anchor-lateral-world-offset-latch-on-activation"
            ),
        )
    if (
        args.vpp_post_merge_offensive_anchor_lateral_world_offset_latch_on_activation_by_task
        is not None
    ):
        overrides[
            "post_merge_offensive_anchor_lateral_world_offset_latch_on_activation_by_task"
        ] = _parse_task_boolean_overrides(
            args.vpp_post_merge_offensive_anchor_lateral_world_offset_latch_on_activation_by_task,
            option_name=(
                "--vpp-post-merge-offensive-anchor-lateral-world-offset-latch-on-activation-by-task"
            ),
        )
    if args.vpp_post_merge_offensive_anchor_blend_release_lateral_only is not None:
        overrides["post_merge_offensive_anchor_blend_release_lateral_only"] = (
            _parse_boolean_literal(
                args.vpp_post_merge_offensive_anchor_blend_release_lateral_only,
                option_name=(
                    "--vpp-post-merge-offensive-anchor-blend-release-lateral-only"
                ),
            )
        )
    if (
        args.vpp_post_merge_offensive_anchor_blend_release_lateral_only_by_task
        is not None
    ):
        overrides[
            "post_merge_offensive_anchor_blend_release_lateral_only_by_task"
        ] = _parse_task_boolean_overrides(
            args.vpp_post_merge_offensive_anchor_blend_release_lateral_only_by_task,
            option_name=(
                "--vpp-post-merge-offensive-anchor-blend-release-lateral-only-by-task"
            ),
        )
    if (
        args.vpp_post_merge_offensive_anchor_blend_release_lateral_only_hold_steps
        is not None
    ):
        hold_steps = int(
            args.vpp_post_merge_offensive_anchor_blend_release_lateral_only_hold_steps
        )
        if hold_steps < 0:
            raise ValueError(
                "--vpp-post-merge-offensive-anchor-blend-release-lateral-only-hold-steps must be >= 0"
            )
        overrides[
            "post_merge_offensive_anchor_blend_release_lateral_only_hold_steps"
        ] = hold_steps
    if (
        args.vpp_post_merge_offensive_anchor_blend_release_lateral_only_hold_steps_by_task
        is not None
    ):
        overrides[
            "post_merge_offensive_anchor_blend_release_lateral_only_hold_steps_by_task"
        ] = _parse_task_nonnegative_int_overrides(
            args.vpp_post_merge_offensive_anchor_blend_release_lateral_only_hold_steps_by_task,
            option_name=(
                "--vpp-post-merge-offensive-anchor-blend-release-lateral-only-hold-steps-by-task"
            ),
        )
    if args.vpp_post_merge_predicted_target_forward_scale is not None:
        scale = float(args.vpp_post_merge_predicted_target_forward_scale)
        if not np.isfinite(scale) or scale < 0.0 or scale > 1.0:
            raise ValueError(
                "--vpp-post-merge-predicted-target-forward-scale must be a finite value in [0, 1]"
            )
        overrides["post_merge_predicted_target_forward_scale"] = scale
    if args.vpp_post_merge_predicted_target_forward_scale_by_task is not None:
        overrides["post_merge_predicted_target_forward_scale_by_task"] = (
            _parse_task_blend_overrides(
                args.vpp_post_merge_predicted_target_forward_scale_by_task
            )
        )
    if args.vpp_post_merge_predicted_target_forward_scale_release_scale is not None:
        scale = float(args.vpp_post_merge_predicted_target_forward_scale_release_scale)
        if not np.isfinite(scale) or scale < 0.0 or scale > 1.0:
            raise ValueError(
                "--vpp-post-merge-predicted-target-forward-scale-release-scale must be "
                "a finite value in [0, 1]"
            )
        overrides["post_merge_predicted_target_forward_scale_release_scale"] = scale
    if (
        args.vpp_post_merge_predicted_target_forward_scale_release_scale_by_task
        is not None
    ):
        overrides["post_merge_predicted_target_forward_scale_release_scale_by_task"] = (
            _parse_task_blend_overrides(
                args.vpp_post_merge_predicted_target_forward_scale_release_scale_by_task
            )
        )
    if (
        args.vpp_post_merge_predicted_target_forward_scale_release_ego_only_streak_steps
        is not None
    ):
        streak_steps = int(
            args.vpp_post_merge_predicted_target_forward_scale_release_ego_only_streak_steps
        )
        if streak_steps < 0:
            raise ValueError(
                "--vpp-post-merge-predicted-target-forward-scale-release-ego-only-streak-steps "
                "must be a nonnegative integer"
            )
        overrides[
            "post_merge_predicted_target_forward_scale_release_ego_only_streak_steps"
        ] = streak_steps
    if (
        args.vpp_post_merge_predicted_target_forward_scale_release_ego_only_streak_steps_by_task
        is not None
    ):
        overrides[
            "post_merge_predicted_target_forward_scale_release_ego_only_streak_steps_by_task"
        ] = _parse_task_nonnegative_int_overrides(
            args.vpp_post_merge_predicted_target_forward_scale_release_ego_only_streak_steps_by_task,
            option_name=(
                "--vpp-post-merge-predicted-target-forward-scale-release-ego-only-streak-steps-by-task"
            ),
        )
    if (
        args.vpp_post_merge_predicted_target_forward_scale_release_reset_on_streak_break
        is not None
    ):
        overrides[
            "post_merge_predicted_target_forward_scale_release_reset_on_streak_break"
        ] = _parse_boolean_literal(
            args.vpp_post_merge_predicted_target_forward_scale_release_reset_on_streak_break,
            option_name=(
                "--vpp-post-merge-predicted-target-forward-scale-release-reset-on-streak-break"
            ),
        )
    if (
        args.vpp_post_merge_predicted_target_forward_scale_release_reset_on_streak_break_by_task
        is not None
    ):
        overrides[
            "post_merge_predicted_target_forward_scale_release_reset_on_streak_break_by_task"
        ] = _parse_task_boolean_overrides(
            args.vpp_post_merge_predicted_target_forward_scale_release_reset_on_streak_break_by_task,
            option_name=(
                "--vpp-post-merge-predicted-target-forward-scale-release-reset-on-streak-break-by-task"
            ),
        )
    if args.vpp_post_merge_predicted_target_forward_scale_hold_steps is not None:
        hold_steps = int(args.vpp_post_merge_predicted_target_forward_scale_hold_steps)
        if hold_steps < 0:
            raise ValueError(
                "--vpp-post-merge-predicted-target-forward-scale-hold-steps must be "
                "a nonnegative integer"
            )
        overrides["post_merge_predicted_target_forward_scale_hold_steps"] = hold_steps
    if (
        args.vpp_post_merge_predicted_target_forward_scale_hold_steps_by_task
        is not None
    ):
        overrides["post_merge_predicted_target_forward_scale_hold_steps_by_task"] = (
            _parse_task_nonnegative_int_overrides(
                args.vpp_post_merge_predicted_target_forward_scale_hold_steps_by_task,
                option_name=(
                    "--vpp-post-merge-predicted-target-forward-scale-hold-steps-by-task"
                ),
            )
        )
    if args.vpp_post_merge_predicted_target_blend_release_on_attack_zone is not None:
        overrides["post_merge_predicted_target_blend_release_on_attack_zone"] = (
            _parse_boolean_literal(
                args.vpp_post_merge_predicted_target_blend_release_on_attack_zone,
                option_name=(
                    "--vpp-post-merge-predicted-target-blend-release-on-attack-zone"
                ),
            )
        )
    if (
        args.vpp_post_merge_predicted_target_blend_release_on_attack_zone_by_task
        is not None
    ):
        overrides["post_merge_predicted_target_blend_release_on_attack_zone_by_task"] = (
            _parse_task_boolean_overrides(
                args.vpp_post_merge_predicted_target_blend_release_on_attack_zone_by_task,
                option_name=(
                    "--vpp-post-merge-predicted-target-blend-release-on-attack-zone-by-task"
                ),
            )
        )
    if (
        args.vpp_post_merge_predicted_target_blend_release_requires_target_attack_zone
        is not None
    ):
        overrides[
            "post_merge_predicted_target_blend_release_requires_target_attack_zone"
        ] = _parse_boolean_literal(
            args.vpp_post_merge_predicted_target_blend_release_requires_target_attack_zone,
            option_name=(
                "--vpp-post-merge-predicted-target-blend-release-requires-target-attack-zone"
            ),
        )
    if (
        args.vpp_post_merge_predicted_target_blend_release_requires_target_attack_zone_by_task
        is not None
    ):
        overrides[
            "post_merge_predicted_target_blend_release_requires_target_attack_zone_by_task"
        ] = _parse_task_boolean_overrides(
            args.vpp_post_merge_predicted_target_blend_release_requires_target_attack_zone_by_task,
            option_name=(
                "--vpp-post-merge-predicted-target-blend-release-requires-target-attack-zone-by-task"
            ),
        )
    if args.vpp_post_merge_predicted_target_blend_release_below_altitude_m is not None:
        threshold = float(args.vpp_post_merge_predicted_target_blend_release_below_altitude_m)
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError(
                "--vpp-post-merge-predicted-target-blend-release-below-altitude-m "
                "must be a finite value >= 0"
            )
        overrides["post_merge_predicted_target_blend_release_below_altitude_m"] = threshold
    if (
        args.vpp_post_merge_predicted_target_blend_release_below_altitude_m_by_task
        is not None
    ):
        overrides[
            "post_merge_predicted_target_blend_release_below_altitude_m_by_task"
        ] = _parse_task_nonnegative_scale_overrides(
            args.vpp_post_merge_predicted_target_blend_release_below_altitude_m_by_task,
            option_name=(
                "--vpp-post-merge-predicted-target-blend-release-below-altitude-m-by-task"
            ),
        )
    if args.vpp_post_merge_predicted_target_blend_hold_steps is not None:
        hold_steps = int(args.vpp_post_merge_predicted_target_blend_hold_steps)
        if hold_steps < 0:
            raise ValueError(
                "--vpp-post-merge-predicted-target-blend-hold-steps must be a nonnegative integer"
            )
        overrides["post_merge_predicted_target_blend_hold_steps"] = hold_steps
    if args.vpp_post_merge_predicted_target_blend_hold_steps_by_task is not None:
        overrides["post_merge_predicted_target_blend_hold_steps_by_task"] = (
            _parse_task_nonnegative_int_overrides(
                args.vpp_post_merge_predicted_target_blend_hold_steps_by_task,
                option_name="--vpp-post-merge-predicted-target-blend-hold-steps-by-task",
            )
        )
    if args.vpp_post_merge_predicted_target_blend_release_blend is not None:
        blend = float(args.vpp_post_merge_predicted_target_blend_release_blend)
        if not np.isfinite(blend) or blend < 0.0 or blend > 1.0:
            raise ValueError(
                "--vpp-post-merge-predicted-target-blend-release-blend must be "
                "a finite value in [0, 1]"
            )
        overrides["post_merge_predicted_target_blend_release_blend"] = blend
    if args.vpp_post_merge_predicted_target_blend_release_blend_by_task is not None:
        overrides["post_merge_predicted_target_blend_release_blend_by_task"] = (
            _parse_task_blend_overrides(
                args.vpp_post_merge_predicted_target_blend_release_blend_by_task
            )
        )
    if args.vpp_longitudinal_scale is not None:
        scale = float(args.vpp_longitudinal_scale)
        if not np.isfinite(scale) or scale < 0.0:
            raise ValueError("--vpp-longitudinal-scale must be a finite value >= 0")
        overrides["longitudinal_scale"] = scale
    if args.vpp_longitudinal_scale_by_task is not None:
        overrides["longitudinal_scale_by_task"] = (
            _parse_task_nonnegative_scale_overrides(
                args.vpp_longitudinal_scale_by_task,
                option_name="--vpp-longitudinal-scale-by-task",
            )
        )
    if args.vpp_lateral_scale is not None:
        scale = float(args.vpp_lateral_scale)
        if not np.isfinite(scale) or scale < 0.0:
            raise ValueError("--vpp-lateral-scale must be a finite value >= 0")
        overrides["lateral_scale"] = scale
    if args.vpp_lateral_scale_by_task is not None:
        overrides["lateral_scale_by_task"] = _parse_task_nonnegative_scale_overrides(
            args.vpp_lateral_scale_by_task,
            option_name="--vpp-lateral-scale-by-task",
        )
    if args.vpp_offensive_anchor_blend is not None:
        blend = float(args.vpp_offensive_anchor_blend)
        if not np.isfinite(blend) or blend < 0.0 or blend > 1.0:
            raise ValueError(
                "--vpp-offensive-anchor-blend must be a finite value in [0, 1]"
            )
        overrides["offensive_anchor_blend"] = blend
    if args.vpp_offensive_anchor_blend_by_task is not None:
        overrides["offensive_anchor_blend_by_task"] = _parse_task_blend_overrides(
            args.vpp_offensive_anchor_blend_by_task,
            option_name="--vpp-offensive-anchor-blend-by-task",
        )
    if args.vpp_offensive_anchor_longitudinal_m is not None:
        distance_m = float(args.vpp_offensive_anchor_longitudinal_m)
        if not np.isfinite(distance_m) or distance_m < 0.0:
            raise ValueError(
                "--vpp-offensive-anchor-longitudinal-m must be a finite value >= 0"
            )
        overrides["offensive_anchor_longitudinal_m"] = distance_m
    if args.vpp_offensive_anchor_longitudinal_m_by_task is not None:
        overrides["offensive_anchor_longitudinal_m_by_task"] = (
            _parse_task_nonnegative_scale_overrides(
                args.vpp_offensive_anchor_longitudinal_m_by_task,
                option_name="--vpp-offensive-anchor-longitudinal-m-by-task",
            )
        )
    if args.vpp_offensive_anchor_frame is not None:
        overrides["offensive_anchor_frame"] = str(args.vpp_offensive_anchor_frame)
    if args.vpp_offensive_anchor_frame_by_task is not None:
        overrides["offensive_anchor_frame_by_task"] = _parse_task_frame_overrides(
            args.vpp_offensive_anchor_frame_by_task,
            option_name="--vpp-offensive-anchor-frame-by-task",
        )
    if args.vpp_offensive_anchor_encounter_stable_max_heading_delta_deg is not None:
        value = float(args.vpp_offensive_anchor_encounter_stable_max_heading_delta_deg)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(
                "--vpp-offensive-anchor-encounter-stable-max-heading-delta-deg "
                "must be a finite value >= 0"
            )
        overrides["offensive_anchor_encounter_stable_max_heading_delta_deg"] = value
    if (
        args.vpp_offensive_anchor_encounter_stable_max_heading_delta_deg_by_task
        is not None
    ):
        overrides[
            "offensive_anchor_encounter_stable_max_heading_delta_deg_by_task"
        ] = _parse_task_nonnegative_float_overrides(
            args.vpp_offensive_anchor_encounter_stable_max_heading_delta_deg_by_task,
            option_name=(
                "--vpp-offensive-anchor-encounter-stable-max-heading-delta-deg-by-task"
            ),
        )
    if args.vpp_offensive_anchor_lateral_frame is not None:
        overrides["offensive_anchor_lateral_frame"] = str(
            args.vpp_offensive_anchor_lateral_frame
        )
    if args.vpp_offensive_anchor_lateral_frame_by_task is not None:
        overrides["offensive_anchor_lateral_frame_by_task"] = (
            _parse_task_frame_overrides(
                args.vpp_offensive_anchor_lateral_frame_by_task,
                option_name="--vpp-offensive-anchor-lateral-frame-by-task",
            )
        )
    if args.vpp_offensive_anchor_lateral_sign_mode is not None:
        overrides["offensive_anchor_lateral_sign_mode"] = str(
            args.vpp_offensive_anchor_lateral_sign_mode
        )
    if args.vpp_offensive_anchor_lateral_sign_mode_by_task is not None:
        overrides["offensive_anchor_lateral_sign_mode_by_task"] = (
            _parse_task_offensive_anchor_lateral_sign_mode_overrides(
                args.vpp_offensive_anchor_lateral_sign_mode_by_task,
                option_name="--vpp-offensive-anchor-lateral-sign-mode-by-task",
            )
        )
    if args.vpp_offensive_anchor_lateral_m is not None:
        distance_m = float(args.vpp_offensive_anchor_lateral_m)
        if not np.isfinite(distance_m) or distance_m < 0.0:
            raise ValueError(
                "--vpp-offensive-anchor-lateral-m must be a finite value >= 0"
            )
        overrides["offensive_anchor_lateral_m"] = distance_m
    if args.vpp_offensive_anchor_lateral_m_by_task is not None:
        overrides["offensive_anchor_lateral_m_by_task"] = (
            _parse_task_nonnegative_scale_overrides(
                args.vpp_offensive_anchor_lateral_m_by_task,
                option_name="--vpp-offensive-anchor-lateral-m-by-task",
            )
        )
    if args.vpp_offensive_anchor_vertical_m is not None:
        distance_m = float(args.vpp_offensive_anchor_vertical_m)
        if not np.isfinite(distance_m):
            raise ValueError(
                "--vpp-offensive-anchor-vertical-m must be a finite value"
            )
        overrides["offensive_anchor_vertical_m"] = distance_m
    if args.vpp_offensive_anchor_vertical_m_by_task is not None:
        overrides["offensive_anchor_vertical_m_by_task"] = (
            _parse_task_float_overrides(
                args.vpp_offensive_anchor_vertical_m_by_task,
                option_name="--vpp-offensive-anchor-vertical-m-by-task",
            )
        )
    if args.vpp_post_merge_anchor_mode is not None:
        overrides["post_merge_anchor_mode"] = str(args.vpp_post_merge_anchor_mode)
    if args.vpp_post_merge_anchor_mode_by_task is not None:
        overrides["post_merge_anchor_mode_by_task"] = _parse_task_anchor_mode_overrides(
            args.vpp_post_merge_anchor_mode_by_task,
            option_name="--vpp-post-merge-anchor-mode-by-task",
        )
    if args.vpp_post_merge_anchor_mode_requires_geometry_disadvantage is not None:
        overrides["post_merge_anchor_mode_requires_geometry_disadvantage"] = (
            _parse_boolean_literal(
                args.vpp_post_merge_anchor_mode_requires_geometry_disadvantage,
                option_name=(
                    "--vpp-post-merge-anchor-mode-requires-geometry-disadvantage"
                ),
            )
        )
    if (
        args.vpp_post_merge_anchor_mode_requires_geometry_disadvantage_by_task
        is not None
    ):
        overrides["post_merge_anchor_mode_requires_geometry_disadvantage_by_task"] = (
            _parse_task_boolean_overrides(
                args.vpp_post_merge_anchor_mode_requires_geometry_disadvantage_by_task,
                option_name=(
                    "--vpp-post-merge-anchor-mode-requires-geometry-disadvantage-by-task"
                ),
            )
        )
    if args.vpp_post_merge_anchor_mode_recovery_below_altitude_m is not None:
        altitude_m = float(args.vpp_post_merge_anchor_mode_recovery_below_altitude_m)
        if not np.isfinite(altitude_m) or altitude_m < 0.0:
            raise ValueError(
                "--vpp-post-merge-anchor-mode-recovery-below-altitude-m must "
                "be a finite value >= 0"
            )
        overrides["post_merge_anchor_mode_recovery_below_altitude_m"] = altitude_m
    if args.vpp_post_merge_anchor_mode_recovery_below_altitude_m_by_task is not None:
        overrides["post_merge_anchor_mode_recovery_below_altitude_m_by_task"] = (
            _parse_task_nonnegative_scale_overrides(
                args.vpp_post_merge_anchor_mode_recovery_below_altitude_m_by_task,
                option_name=(
                    "--vpp-post-merge-anchor-mode-recovery-below-altitude-m-by-task"
                ),
            )
        )
    if args.vpp_post_merge_anchor_mode_release_ego_only_streak_steps is not None:
        streak_steps = int(args.vpp_post_merge_anchor_mode_release_ego_only_streak_steps)
        if streak_steps < 0:
            raise ValueError(
                "--vpp-post-merge-anchor-mode-release-ego-only-streak-steps must "
                "be a nonnegative integer"
            )
        overrides["post_merge_anchor_mode_release_ego_only_streak_steps"] = (
            streak_steps
        )
    if (
        args.vpp_post_merge_anchor_mode_release_ego_only_streak_steps_by_task
        is not None
    ):
        overrides["post_merge_anchor_mode_release_ego_only_streak_steps_by_task"] = (
            _parse_task_nonnegative_int_overrides(
                args.vpp_post_merge_anchor_mode_release_ego_only_streak_steps_by_task,
                option_name=(
                    "--vpp-post-merge-anchor-mode-release-ego-only-streak-steps-by-task"
                ),
            )
        )
    if args.vpp_post_merge_anchor_mode_release_reset_on_streak_break is not None:
        overrides["post_merge_anchor_mode_release_reset_on_streak_break"] = (
            _parse_boolean_literal(
                args.vpp_post_merge_anchor_mode_release_reset_on_streak_break,
                option_name=(
                    "--vpp-post-merge-anchor-mode-release-reset-on-streak-break"
                ),
            )
        )
    if (
        args.vpp_post_merge_anchor_mode_release_reset_on_streak_break_by_task
        is not None
    ):
        overrides["post_merge_anchor_mode_release_reset_on_streak_break_by_task"] = (
            _parse_task_boolean_overrides(
                args.vpp_post_merge_anchor_mode_release_reset_on_streak_break_by_task,
                option_name=(
                    "--vpp-post-merge-anchor-mode-release-reset-on-streak-break-by-task"
                ),
            )
        )
    if args.vpp_post_merge_anchor_mode_offensive_anchor_blend is not None:
        blend = float(args.vpp_post_merge_anchor_mode_offensive_anchor_blend)
        if not np.isfinite(blend) or blend < 0.0 or blend > 1.0:
            raise ValueError(
                "--vpp-post-merge-anchor-mode-offensive-anchor-blend must be "
                "a finite value in [0, 1]"
            )
        overrides["post_merge_anchor_mode_offensive_anchor_blend"] = blend
    if args.vpp_post_merge_anchor_mode_offensive_anchor_blend_by_task is not None:
        overrides["post_merge_anchor_mode_offensive_anchor_blend_by_task"] = (
            _parse_task_blend_overrides(
                args.vpp_post_merge_anchor_mode_offensive_anchor_blend_by_task,
                option_name=(
                    "--vpp-post-merge-anchor-mode-offensive-anchor-blend-by-task"
                ),
            )
        )
    if args.vpp_post_merge_anchor_mode_offensive_anchor_longitudinal_blend is not None:
        blend = float(args.vpp_post_merge_anchor_mode_offensive_anchor_longitudinal_blend)
        if not np.isfinite(blend) or blend < 0.0 or blend > 1.0:
            raise ValueError(
                "--vpp-post-merge-anchor-mode-offensive-anchor-longitudinal-blend "
                "must be a finite value in [0, 1]"
            )
        overrides["post_merge_anchor_mode_offensive_anchor_longitudinal_blend"] = blend
    if (
        args.vpp_post_merge_anchor_mode_offensive_anchor_longitudinal_blend_by_task
        is not None
    ):
        overrides["post_merge_anchor_mode_offensive_anchor_longitudinal_blend_by_task"] = (
            _parse_task_blend_overrides(
                args.vpp_post_merge_anchor_mode_offensive_anchor_longitudinal_blend_by_task,
                option_name=(
                    "--vpp-post-merge-anchor-mode-offensive-anchor-longitudinal-blend-by-task"
                ),
            )
        )
    if args.vpp_post_merge_anchor_mode_offensive_anchor_lateral_blend is not None:
        blend = float(args.vpp_post_merge_anchor_mode_offensive_anchor_lateral_blend)
        if not np.isfinite(blend) or blend < 0.0 or blend > 1.0:
            raise ValueError(
                "--vpp-post-merge-anchor-mode-offensive-anchor-lateral-blend "
                "must be a finite value in [0, 1]"
            )
        overrides["post_merge_anchor_mode_offensive_anchor_lateral_blend"] = blend
    if (
        args.vpp_post_merge_anchor_mode_offensive_anchor_lateral_blend_by_task
        is not None
    ):
        overrides["post_merge_anchor_mode_offensive_anchor_lateral_blend_by_task"] = (
            _parse_task_blend_overrides(
                args.vpp_post_merge_anchor_mode_offensive_anchor_lateral_blend_by_task,
                option_name=(
                    "--vpp-post-merge-anchor-mode-offensive-anchor-lateral-blend-by-task"
                ),
            )
        )
    if (
        args.vpp_post_merge_anchor_mode_lateral_world_offset_latch_on_activation
        is not None
    ):
        overrides["post_merge_anchor_mode_lateral_world_offset_latch_on_activation"] = (
            _parse_boolean_literal(
                args.vpp_post_merge_anchor_mode_lateral_world_offset_latch_on_activation,
                option_name=(
                    "--vpp-post-merge-anchor-mode-lateral-world-offset-latch-on-activation"
                ),
            )
        )
    if (
        args.vpp_post_merge_anchor_mode_lateral_world_offset_latch_on_activation_by_task
        is not None
    ):
        overrides[
            "post_merge_anchor_mode_lateral_world_offset_latch_on_activation_by_task"
        ] = _parse_task_boolean_overrides(
            args.vpp_post_merge_anchor_mode_lateral_world_offset_latch_on_activation_by_task,
            option_name=(
                "--vpp-post-merge-anchor-mode-lateral-world-offset-latch-on-activation-by-task"
            ),
        )
    if args.vpp_post_merge_anchor_mode_longitudinal_scale is not None:
        scale = float(args.vpp_post_merge_anchor_mode_longitudinal_scale)
        if not np.isfinite(scale) or scale < 0.0:
            raise ValueError(
                "--vpp-post-merge-anchor-mode-longitudinal-scale must be a "
                "finite value >= 0"
            )
        overrides["post_merge_anchor_mode_longitudinal_scale"] = scale
    if args.vpp_post_merge_anchor_mode_longitudinal_scale_by_task is not None:
        overrides["post_merge_anchor_mode_longitudinal_scale_by_task"] = (
            _parse_task_nonnegative_scale_overrides(
                args.vpp_post_merge_anchor_mode_longitudinal_scale_by_task,
                option_name=(
                    "--vpp-post-merge-anchor-mode-longitudinal-scale-by-task"
                ),
            )
        )
    if args.vpp_post_merge_anchor_mode_lateral_scale is not None:
        scale = float(args.vpp_post_merge_anchor_mode_lateral_scale)
        if not np.isfinite(scale) or scale < 0.0:
            raise ValueError(
                "--vpp-post-merge-anchor-mode-lateral-scale must be a finite "
                "value >= 0"
            )
        overrides["post_merge_anchor_mode_lateral_scale"] = scale
    if args.vpp_post_merge_anchor_mode_lateral_scale_by_task is not None:
        overrides["post_merge_anchor_mode_lateral_scale_by_task"] = (
            _parse_task_nonnegative_scale_overrides(
                args.vpp_post_merge_anchor_mode_lateral_scale_by_task,
                option_name=(
                    "--vpp-post-merge-anchor-mode-lateral-scale-by-task"
                ),
            )
        )
    if args.vpp_close_range_anchor_mode is not None:
        overrides["close_range_anchor_mode"] = str(args.vpp_close_range_anchor_mode)
    if args.vpp_close_range_anchor_mode_by_task is not None:
        overrides["close_range_anchor_mode_by_task"] = _parse_task_anchor_mode_overrides(
            args.vpp_close_range_anchor_mode_by_task,
            option_name="--vpp-close-range-anchor-mode-by-task",
        )
    if args.vpp_close_range_anchor_trigger_range_m is not None:
        trigger_range_m = float(args.vpp_close_range_anchor_trigger_range_m)
        if not np.isfinite(trigger_range_m) or trigger_range_m < 0.0:
            raise ValueError(
                "--vpp-close-range-anchor-trigger-range-m must be a finite value >= 0"
            )
        overrides["close_range_anchor_trigger_range_m"] = trigger_range_m
    if args.vpp_close_range_anchor_trigger_range_m_by_task is not None:
        overrides["close_range_anchor_trigger_range_m_by_task"] = (
            _parse_task_nonnegative_scale_overrides(
                args.vpp_close_range_anchor_trigger_range_m_by_task,
                option_name="--vpp-close-range-anchor-trigger-range-m-by-task",
            )
        )
    if args.vpp_close_range_anchor_alignment_angle_deg_max is not None:
        angle_deg_max = float(args.vpp_close_range_anchor_alignment_angle_deg_max)
        if not np.isfinite(angle_deg_max) or angle_deg_max < 0.0:
            raise ValueError(
                "--vpp-close-range-anchor-alignment-angle-deg-max must be a finite value >= 0"
            )
        overrides["close_range_anchor_alignment_angle_deg_max"] = angle_deg_max
    if args.vpp_close_range_anchor_alignment_angle_deg_max_by_task is not None:
        overrides["close_range_anchor_alignment_angle_deg_max_by_task"] = (
            _parse_task_nonnegative_scale_overrides(
                args.vpp_close_range_anchor_alignment_angle_deg_max_by_task,
                option_name="--vpp-close-range-anchor-alignment-angle-deg-max-by-task",
            )
        )
    if args.vpp_close_range_anchor_release_on_post_merge is not None:
        overrides["close_range_anchor_release_on_post_merge"] = (
            _parse_boolean_literal(
                args.vpp_close_range_anchor_release_on_post_merge,
                option_name="--vpp-close-range-anchor-release-on-post-merge",
            )
        )
    if args.vpp_close_range_anchor_release_on_post_merge_by_task is not None:
        overrides["close_range_anchor_release_on_post_merge_by_task"] = (
            _parse_task_boolean_overrides(
                args.vpp_close_range_anchor_release_on_post_merge_by_task,
                option_name="--vpp-close-range-anchor-release-on-post-merge-by-task",
            )
        )
    if args.vpp_close_range_anchor_requires_first_pass is not None:
        overrides["close_range_anchor_requires_first_pass"] = _parse_boolean_literal(
            args.vpp_close_range_anchor_requires_first_pass,
            option_name="--vpp-close-range-anchor-requires-first-pass",
        )
    if args.vpp_close_range_anchor_requires_first_pass_by_task is not None:
        overrides["close_range_anchor_requires_first_pass_by_task"] = (
            _parse_task_boolean_overrides(
                args.vpp_close_range_anchor_requires_first_pass_by_task,
                option_name="--vpp-close-range-anchor-requires-first-pass-by-task",
            )
        )
    if args.vpp_close_range_anchor_offensive_anchor_blend is not None:
        blend = float(args.vpp_close_range_anchor_offensive_anchor_blend)
        if not np.isfinite(blend) or blend < 0.0 or blend > 1.0:
            raise ValueError(
                "--vpp-close-range-anchor-offensive-anchor-blend must be a finite value in [0, 1]"
            )
        overrides["close_range_anchor_offensive_anchor_blend"] = blend
    if args.vpp_close_range_anchor_offensive_anchor_blend_by_task is not None:
        overrides["close_range_anchor_offensive_anchor_blend_by_task"] = (
            _parse_task_blend_overrides(
                args.vpp_close_range_anchor_offensive_anchor_blend_by_task,
                option_name="--vpp-close-range-anchor-offensive-anchor-blend-by-task",
            )
        )
    if args.vpp_close_range_anchor_release_alignment_angle_deg_max is not None:
        angle_deg_max = float(
            args.vpp_close_range_anchor_release_alignment_angle_deg_max
        )
        if not np.isfinite(angle_deg_max) or angle_deg_max < 0.0:
            raise ValueError(
                "--vpp-close-range-anchor-release-alignment-angle-deg-max "
                "must be a finite value >= 0"
            )
        overrides["close_range_anchor_release_alignment_angle_deg_max"] = (
            angle_deg_max
        )
    if args.vpp_close_range_anchor_release_alignment_angle_deg_max_by_task is not None:
        overrides["close_range_anchor_release_alignment_angle_deg_max_by_task"] = (
            _parse_task_nonnegative_scale_overrides(
                args.vpp_close_range_anchor_release_alignment_angle_deg_max_by_task,
                option_name=(
                    "--vpp-close-range-anchor-release-alignment-angle-deg-max-by-task"
                ),
            )
        )
    if args.vpp_close_range_anchor_post_merge_hold_steps is not None:
        hold_steps = int(args.vpp_close_range_anchor_post_merge_hold_steps)
        if hold_steps < 0:
            raise ValueError(
                "--vpp-close-range-anchor-post-merge-hold-steps must be a nonnegative integer"
            )
        overrides["close_range_anchor_post_merge_hold_steps"] = hold_steps
    if args.vpp_close_range_anchor_post_merge_hold_steps_by_task is not None:
        overrides["close_range_anchor_post_merge_hold_steps_by_task"] = (
            _parse_task_nonnegative_int_overrides(
                args.vpp_close_range_anchor_post_merge_hold_steps_by_task,
                option_name="--vpp-close-range-anchor-post-merge-hold-steps-by-task",
            )
        )
    return overrides


def main() -> int:
    args = _parse_args()
    config_path = _repo_path(args.config)
    if config_path is None:
        raise RuntimeError("Config path is required")
    comparison_config = _load_config_with_includes(config_path)
    comparison_config = _materialize_scenario_manifest(comparison_config, config_path)

    methods = _select_methods(comparison_config, args.preset, args.methods)
    tasks = _select_tasks(comparison_config, args.tasks)
    seeds = _select_seeds(comparison_config, args)
    n_episodes = int(args.n_episodes or comparison_config.get("run_defaults", {}).get("n_episodes", 1))
    backend = args.backend or comparison_config.get("run_defaults", {}).get("backend", "jsbsim")
    opponent_stage = args.opponent_stage or comparison_config.get("opponent_stage", "none")
    attack_zone_overrides = _collect_attack_zone_cli_overrides(args)
    prediction_overrides = _collect_prediction_cli_overrides(args)
    vpp_overrides = _collect_vpp_cli_overrides(args)
    jsbsim_root = args.jsbsim_root
    if jsbsim_root is None and str(backend).lower() == "jsbsim":
        jsbsim_root = os.environ.get("JSBSIM_ROOT") or str(REPO_ROOT)
    max_steps_override = None
    if args.run_status == "smoke":
        max_steps_override = comparison_config.get("run_defaults", {}).get("smoke_max_high_level_steps")

    run_dir = _repo_path(args.output_root) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved_config = copy.deepcopy(comparison_config)
    resolved_config["selection"] = {
        "preset": args.preset,
        "methods": methods,
        "tasks": tasks,
        "seeds": seeds,
        "n_episodes": n_episodes,
        "backend": backend,
        "opponent_stage": opponent_stage,
        "jsbsim_root": jsbsim_root,
        "attack_zone_overrides": attack_zone_overrides,
        "prediction_overrides": prediction_overrides,
        "vpp_overrides": vpp_overrides,
        "run_status": args.run_status,
        "dry_run": args.dry_run,
    }
    resolved_config_path = run_dir / "resolved_config.yaml"
    with open(resolved_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            resolved_config,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
    design_notes = copy.deepcopy(DESIGN_NOTES)
    design_notes["selected_methods"] = methods
    design_notes["selected_tasks"] = tasks
    design_notes["selected_backend"] = backend
    design_notes["selected_opponent_stage"] = opponent_stage
    design_notes["resolved_jsbsim_root"] = jsbsim_root
    design_notes_path = run_dir / "design_notes.json"
    with open(design_notes_path, "w", encoding="utf-8") as f:
        json.dump(design_notes, f, indent=2, ensure_ascii=False)

    manifest = RunManifest(
        stage_name=STAGE_NAME,
        stage_version=STAGE_VERSION,
        command_line=sys.argv,
        output_dir=str(run_dir),
        config_path=str(config_path),
        resolved_config=resolved_config,
    )
    manifest.mark_started()
    manifest.compute_config_hash()
    manifest.record_input_file("comparison_config", config_path)
    scenario_manifest_resolution = comparison_config.get("scenario_manifest_resolution", {})
    scenario_manifest_path = scenario_manifest_resolution.get("path")
    if scenario_manifest_path:
        manifest.record_input_file("scenario_manifest", Path(str(scenario_manifest_path)))
    manifest.record_output_file("resolved_config.yaml", resolved_config_path)
    manifest.record_output_file("design_notes.json", design_notes_path)
    manifest.extra.update(
        {
            "run_id": args.run_id,
            "run_status": args.run_status,
            "preset": args.preset,
            "backend": backend,
            "opponent_stage": opponent_stage,
            "jsbsim_root": jsbsim_root,
            "attack_zone_overrides": attack_zone_overrides,
            "prediction_overrides": prediction_overrides,
            "vpp_overrides": vpp_overrides,
            "methods": methods,
            "tasks": tasks,
            "seeds": seeds,
            "n_episodes": n_episodes,
            "design_notes": design_notes,
        }
    )

    contract = _contract_for_run(args.dry_run)
    contract_path = contract.save(run_dir)
    manifest.record_output_file("artifact_contract.json", contract_path)

    config_snapshots: Dict[str, Dict[str, Any]] = {}
    eval_configs: Dict[str, Dict[str, Any]] = {}
    failures: List[Dict[str, Any]] = []

    checkpoint_checks, checkpoint_errors = _checkpoint_checks(
        comparison_config,
        methods,
        opponent_stage=opponent_stage,
        allow_missing=args.allow_missing_checkpoints and args.dry_run,
    )
    if checkpoint_errors:
        failures.extend({"phase": "checkpoint_check", "error": err} for err in checkpoint_errors)

    try:
        for method in methods:
            for task in tasks:
                cfg = build_eval_config(
                    comparison_config,
                    method,
                    task,
                    backend=backend,
                    opponent_stage=opponent_stage,
                    max_steps_override=max_steps_override,
                    jsbsim_root=jsbsim_root,
                    attack_zone_overrides=attack_zone_overrides,
                    prediction_overrides=prediction_overrides,
                    vpp_overrides=vpp_overrides,
                )
                eval_configs[f"{method}/{task}"] = cfg
        config_snapshots = _write_config_snapshots(run_dir, eval_configs)
        manifest.extra["config_snapshots"] = config_snapshots
    except Exception as exc:
        failures.append(
            {
                "phase": "config_build",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )

    if args.dry_run:
        dry_run_path = run_dir / "dry_run_checks.json"
        with open(dry_run_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "checkpoint_checks": checkpoint_checks,
                    "failures": failures,
                    "config_snapshots": config_snapshots,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        manifest.record_output_file("dry_run_checks.json", dry_run_path)
    else:
        all_records: List[Dict[str, Any]] = []
        observation_audits: Dict[str, Dict[str, Any]] = {}
        git_commit = manifest.git_info.get("short_commit") or "unknown"
        start = time.time()
        for method in methods:
            method_def = comparison_config["methods"][method]
            agent_type = method_def.get("agent_type", "ppo")
            # oracle_task_gate, random_task_gate, and rule_guidance do not require a single checkpoint
            if agent_type in {"oracle_task_gate", "random_task_gate", "rule_guidance"}:
                checkpoint_path = None
            else:
                checkpoint_path = _repo_path(method_def.get("checkpoint"))
                if checkpoint_path is None or not checkpoint_path.exists():
                    failures.append(
                        {
                            "phase": "evaluation",
                            "method": method,
                            "error": f"Checkpoint not found: {checkpoint_path}",
                        }
                    )
                    continue
            for task in tasks:
                task_def = comparison_config["tasks"][task]
                cfg = eval_configs.get(f"{method}/{task}")
                if cfg is None:
                    continue
                scenarios = _task_scenarios(task_def)
                try:
                    env = _build_env(cfg)
                    first_scenario = scenarios[0]
                    sample_kwargs: Dict[str, Any] = {"seed": seeds[0] if seeds else 0}
                    if first_scenario is not None:
                        sample_kwargs["scenario"] = first_scenario
                    sample_obs = env.reset(**sample_kwargs)
                    audit_key = f"{method}/{task}"
                    observation_audit = _build_observation_audit(
                        method_name=method,
                        task_name=task,
                        method_def=method_def,
                        config=cfg,
                        checkpoint_path=checkpoint_path or Path("dummy"),
                        sample_obs=sample_obs,
                    )
                    observation_audits[audit_key] = observation_audit
                    if not observation_audit["schema_dim_matches_vector"]:
                        raise RuntimeError(
                            f"Observation schema dim mismatch for {audit_key}: "
                            f"schema={observation_audit['schema_dim']} "
                            f"vector={observation_audit['sample_observation_dim']}"
                        )
                    if observation_audit["obs_dim_matches_checkpoint"] is False:
                        raise RuntimeError(
                            f"Checkpoint observation dim mismatch for {audit_key}: "
                            f"checkpoint={observation_audit['checkpoint_obs_dim']} "
                            f"vector={observation_audit['sample_observation_dim']}"
                        )
                    if observation_audit["action_dim_matches_checkpoint"] is False:
                        raise RuntimeError(
                            f"Checkpoint action dim mismatch for {audit_key}: "
                            f"checkpoint={observation_audit['checkpoint_action_dim']} "
                            f"policy={observation_audit['policy_action_dim']}"
                        )
                    agent = _build_agent(method_def, cfg, checkpoint_path or Path("dummy"), sample_obs, args.device)
                    if hasattr(agent, "set_env"):
                        agent.set_env(env)
                    # Notify agent of current task if it supports it
                    if hasattr(agent, "set_task_name"):
                        agent.set_task_name(task)
                    for seed in seeds:
                        for scenario_idx, scenario in enumerate(scenarios):
                            for ep in range(n_episodes):
                                if hasattr(agent, "reset_episode"):
                                    agent.reset_episode()
                                # Random gate: select a random specialist at episode start
                                if hasattr(agent, "select_random_specialist") and not hasattr(agent, "reset_episode"):
                                    selected = agent.select_random_specialist()
                                    agent.set_specialist(selected)
                                episode_id = scenario_idx * n_episodes + ep
                                scenario_seed = (
                                    scenario.get("metadata", {}).get("scenario_seed")
                                    if isinstance(scenario, dict)
                                    else None
                                )
                                use_scenario_seed = bool(
                                    comparison_config.get("scenario_manifest_resolution", {}).get(
                                        "use_scenario_seed", False
                                    )
                                )
                                if use_scenario_seed and scenario_seed is not None:
                                    ep_seed = int(scenario_seed) + ep
                                else:
                                    ep_seed = seed * 100000 + scenario_idx * 100 + ep
                                rec = _run_episode(
                                    env=env,
                                    agent=agent,
                                    method_name=method,
                                    method_def=method_def,
                                    task_name=task,
                                    scenario=scenario,
                                    seed=ep_seed,
                                    episode=episode_id,
                                    run_id=args.run_id,
                                    config=cfg,
                                    git_commit=git_commit,
                                    run_status=args.run_status,
                                    save_full=not args.no_trajectory,
                                )
                                rec["observation_audit"] = copy.deepcopy(observation_audit)
                                rec["observation_dim"] = observation_audit["sample_observation_dim"]
                                rec["checkpoint_obs_dim"] = observation_audit["checkpoint_obs_dim"]
                                rec["obs_dim_matches_checkpoint"] = observation_audit["obs_dim_matches_checkpoint"]
                                all_records.append(rec)
                    env.close()
                except Exception as exc:
                    failures.append(
                        {
                            "phase": "evaluation",
                            "method": method,
                            "task": task,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )

        recorder = RunRecorder(run_dir)
        for rec in all_records:
            recorder.add(rec)
        recorder.write()
        (run_dir / "raw").mkdir(parents=True, exist_ok=True)
        summary_path = _write_summary_csv(run_dir, all_records)
        aggregate_path = _write_aggregate_summary(run_dir, all_records)
        combat_diagnostics_path = _write_combat_geometry_diagnostics(
            run_dir,
            all_records,
        )
        observation_audit_path = run_dir / "observation_audit.json"
        with open(observation_audit_path, "w", encoding="utf-8") as f:
            json.dump({"audits": observation_audits}, f, indent=2, ensure_ascii=False)
        failures_path = run_dir / "failures.json"
        with open(failures_path, "w", encoding="utf-8") as f:
            json.dump({"failures": failures}, f, indent=2, ensure_ascii=False)

        manifest.record_output_file("summary.csv", summary_path)
        manifest.record_output_file("aggregate/episode_records.json", recorder.aggregate_dir / "episode_records.json")
        manifest.record_output_file("aggregate/method_task_summary.json", aggregate_path)
        manifest.record_output_file("aggregate/combat_geometry_diagnostics.json", combat_diagnostics_path)
        manifest.record_output_file("observation_audit.json", observation_audit_path)
        manifest.record_output_file("failures.json", failures_path)
        manifest.artifacts_present["raw"] = (run_dir / "raw").is_dir()
        manifest.artifacts_present["manifests"] = (run_dir / "manifests").is_dir()
        manifest.extra["elapsed_seconds"] = time.time() - start
        manifest.extra["total_episodes"] = len(all_records)
        manifest.extra["observation_audits"] = observation_audits

    if args.run_status != "formal":
        manifest.add_invalid_for_paper_reason(f"run_status={args.run_status} is not final paper evidence")
    if backend != "jsbsim":
        manifest.add_invalid_for_paper_reason("backend is not jsbsim")
    if args.dry_run:
        manifest.add_invalid_for_paper_reason("dry-run does not execute simulations")
    if failures:
        manifest.add_invalid_for_paper_reason("one or more checks/evaluations failed")

    manifest.config_overrides = [
        override
        for cfg in eval_configs.values()
        for override in get_config_overrides(cfg)
    ]
    manifest.artifacts_present["manifests"] = (run_dir / "manifests").is_dir()
    manifest.artifacts_present["run_manifest.json"] = True
    manifest.save(run_dir)
    validation = contract.validate(run_dir, manifest=manifest.to_dict())
    manifest.extra["artifact_validation"] = validation
    if not validation["valid"]:
        manifest.add_invalid_for_paper_reason("artifact contract validation failed")

    if failures:
        manifest.mark_failed("; ".join(str(f.get("error")) for f in failures[:3]))
    else:
        manifest.mark_completed(
            paper_safe=_apply_paper_safety_guards(
                manifest,
                run_status=args.run_status,
                backend=backend,
                dry_run=args.dry_run,
            )
        )
    manifest.save(run_dir)

    print("=" * 60)
    print("JSBSim HRL comparison workflow complete")
    print(f"Output dir: {run_dir}")
    print(f"Status: {manifest.status}")
    print(f"Paper-safe: {manifest.paper_safe}")
    print(f"Artifact contract valid: {validation['valid']}")
    if failures:
        print(f"Failures: {len(failures)}")
        for failure in failures[:5]:
            print(f"  - {failure.get('phase')}: {failure.get('error')}")
    print("=" * 60)
    return 1 if failures and not (args.dry_run and args.allow_missing_checkpoints) else 0


if __name__ == "__main__":
    raise SystemExit(main())
