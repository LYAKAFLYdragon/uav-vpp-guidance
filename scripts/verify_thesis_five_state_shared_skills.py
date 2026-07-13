#!/usr/bin/env python3
"""Verify P4 shared-skill readiness without training any policy weights.

The verifier is intentionally fail-closed.  It freezes and loads the P3 v2
encoder, checks the 66-D profile-conditioned interface, validates all three
opponents, and performs one JSBSim action-interface probe.  A passing report
means only that P4 may later be explicitly authorized for training; it never
means that any of the four skills has passed a maneuver-performance gate.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
from itertools import product
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
for _path in (SCRIPT_DIR, REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from uav_vpp_guidance.hierarchy.five_state_observation_contract import (  # noqa: E402
    BASE_GEOMETRY_FEATURES,
    EXPLICIT_HISTORY_FEATURES,
    INTENT_TARGET_FEATURES,
    INTENT_WEIGHT_FEATURES,
    FiveStateObservationContract,
)
from uav_vpp_guidance.hierarchy.shared_intent_profiles import (  # noqa: E402
    PROFILE_NAMES,
    compile_profile,
    get_profile,
)
from uav_vpp_guidance.hierarchy.shared_intent_validity import (  # noqa: E402
    SKILL_NAMES,
    TacticalContext,
    base_allowed_profiles,
    build_validity_mask,
)
from uav_vpp_guidance.training.thesis_shared_skill_policy import (  # noqa: E402
    OBSERVATION_DIM,
    VPP_ACTION_COMPONENTS,
    VPP_ACTION_DIM,
    ThesisSharedSkillPolicy,
)
from uav_vpp_guidance.training.thesis_temporal_geometry_encoder import (  # noqa: E402
    TemporalGeometryEncoder,
)


DEFAULT_REGISTRY = REPO_ROOT / "config" / "experiment" / "thesis_five_state_shared_skill_registry_v1.yaml"
DEFAULT_GEOMETRY_CONFIG = REPO_ROOT / "config" / "experiment" / "train_thesis_five_state_shared_skills_geometry_v1.yaml"
DEFAULT_COMBAT_CONFIG = REPO_ROOT / "config" / "experiment" / "train_thesis_five_state_shared_skills_combat_v1.yaml"
DEFAULT_REPORT = REPO_ROOT / "reports" / "thesis_five_state_shared_intent_v1_p4_readiness_20260713.json"


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return payload


def _resolve_repo_path(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _add_check(
    report: dict[str, Any],
    *,
    name: str,
    passed: bool,
    detail: Mapping[str, Any] | None = None,
) -> None:
    entry = {"name": name, "passed": bool(passed), "detail": dict(detail or {})}
    report["checks"].append(entry)
    if not passed:
        report["errors"].append(name)


def _import_class(class_path: str):
    module_name, class_name = str(class_path).replace("src.", "", 1).rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


def _instantiate_opponent(entry: Mapping[str, Any]) -> object:
    cls = _import_class(str(entry["class"]))
    kwargs: dict[str, Any] = {}
    if entry.get("checkpoint"):
        kwargs["checkpoint_path"] = str(entry["checkpoint"])
    if "config" in entry:
        kwargs["config"] = copy.deepcopy(entry["config"])
    if "device" in entry:
        kwargs["device"] = str(entry["device"])
    if "invert_observation" in entry:
        kwargs["invert_observation"] = bool(entry["invert_observation"])
    return cls(**kwargs)


def _verify_frozen_inputs(
    report: dict[str, Any], registry: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    frozen = registry["frozen_inputs"]
    hashes: dict[str, str] = {}
    for key in (
        "observation_contract",
        "global_intent_profiles",
        "validity_mask",
        "opponent_registry",
        "asset_manifest",
        "output_retention_policy",
    ):
        entry = frozen[key]
        path = _resolve_repo_path(str(entry["path"]))
        exists = path.is_file()
        actual = _sha256_file(path) if exists else None
        expected = str(entry["sha256"]).lower()
        hashes[key] = actual or ""
        _add_check(
            report,
            name=f"frozen_input:{key}",
            passed=exists and actual == expected,
            detail={"path": str(path), "expected_sha256": expected, "actual_sha256": actual},
        )

    p3 = frozen["p3_temporal_encoder"]
    checkpoint_path = Path(str(p3["checkpoint"]))
    checkpoint_exists = checkpoint_path.is_file()
    checkpoint_hash = _sha256_file(checkpoint_path) if checkpoint_exists else None
    _add_check(
        report,
        name="frozen_input:p3_checkpoint_sha256",
        passed=checkpoint_exists and checkpoint_hash == str(p3["sha256"]).lower(),
        detail={
            "path": str(checkpoint_path),
            "expected_sha256": str(p3["sha256"]).lower(),
            "actual_sha256": checkpoint_hash,
        },
    )
    source_config = _resolve_repo_path(str(p3["source_config"]))
    source_hash = _sha256_file(source_config) if source_config.is_file() else None
    _add_check(
        report,
        name="frozen_input:p3_source_config_sha256",
        passed=source_hash == str(p3["source_config_sha256"]).lower(),
        detail={
            "path": str(source_config),
            "expected_sha256": str(p3["source_config_sha256"]).lower(),
            "actual_sha256": source_hash,
        },
    )
    return frozen, {"p3_checkpoint": checkpoint_path, "p3_checkpoint_sha256": checkpoint_hash, **hashes}


def _verify_contract_profiles_and_mask(report: dict[str, Any], registry: Mapping[str, Any]) -> None:
    frozen = registry["frozen_inputs"]
    contract_config = _load_yaml(_resolve_repo_path(frozen["observation_contract"]["path"]))
    contract = FiveStateObservationContract(
        temporal_embedding_dim=int(contract_config["temporal_embedding"]["dimension"]),
        history_window_steps=int(contract_config["explicit_history"]["window_steps"]),
    )
    expected_partition = frozen["observation_contract"]["feature_partition"]
    contract_ok = (
        contract.obs_dim == OBSERVATION_DIM == int(frozen["observation_contract"]["observation_dim"])
        and len(BASE_GEOMETRY_FEATURES) == int(expected_partition["base_geometry"])
        and len(EXPLICIT_HISTORY_FEATURES) == int(expected_partition["explicit_history"])
        and len(INTENT_TARGET_FEATURES) == int(expected_partition["intent_targets"])
        and len(INTENT_WEIGHT_FEATURES) == int(expected_partition["intent_weights"])
        and int(contract_config["temporal_embedding"]["dimension"]) == int(expected_partition["temporal_embedding"])
        and contract_config["contract_rules"]["task_id_input"] == "prohibited"
        and contract_config["contract_rules"]["opponent_stage_input"] == "prohibited"
        and contract_config["contract_rules"]["implicit_padding_or_truncation"] == "prohibited"
    )
    _add_check(
        report,
        name="observation_contract:strict_66d",
        passed=contract_ok,
        detail={"observation_dim": contract.obs_dim, "feature_names": list(contract.feature_names)},
    )

    profile_config = _load_yaml(_resolve_repo_path(frozen["global_intent_profiles"]["path"]))
    profile_names = tuple(profile_config.get("profiles", {}).keys())
    profile_ok = profile_names == PROFILE_NAMES and len(profile_names) == int(frozen["global_intent_profiles"]["profile_count"])
    for name in PROFILE_NAMES:
        compiled = compile_profile(name)
        defined = profile_config.get("profiles", {}).get(name, {})
        expected = get_profile(name)
        profile_ok = profile_ok and tuple(defined.get("targets", ())) == expected.targets
        profile_ok = profile_ok and tuple(defined.get("weights", ())) == expected.weights
        profile_ok = profile_ok and compiled["target_vector"].shape == (6,)
        profile_ok = profile_ok and compiled["weight_vector"].shape == (6,)
    _add_check(
        report,
        name="profiles:seven_global_profile_contract",
        passed=profile_ok,
        detail={"profiles": list(profile_names), "profile_count": len(profile_names)},
    )

    known_states = set(frozen["validity_mask"]["taxonomy_states"])
    known_phases = set(frozen["validity_mask"]["phases"])
    skill_ok = set(registry["skills"]) == set(SKILL_NAMES)
    all_contexts = tuple(product(sorted(known_states), sorted(known_phases), ("advantage", "balanced", "disadvantage")))
    skill_details: dict[str, Any] = {}
    for expected_id, skill_name in enumerate(SKILL_NAMES):
        entry = registry["skills"].get(skill_name, {})
        allowed_profiles = tuple(entry.get("allowed_profiles", ()))
        profile_usable = {
            profile: any(
                bool(
                    build_validity_mask(TacticalContext(state, phase, energy))["mask"]
                    [expected_id, PROFILE_NAMES.index(profile)]
                )
                for state, phase, energy in all_contexts
            )
            for profile in allowed_profiles
            if profile in PROFILE_NAMES
        }
        coverage = entry.get("geometry_phase_coverage", {})
        readiness_gate = entry.get("readiness_gate", {})
        entry_ok = (
            int(entry.get("skill_id", -1)) == expected_id
            and entry.get("status") == "untrained_not_ready"
            and entry.get("checkpoint") is None
            and entry.get("checkpoint_fallback") == "prohibited"
            and int(entry.get("observation_dim", -1)) == OBSERVATION_DIM
            and int(entry.get("action_dim", -1)) == VPP_ACTION_DIM
            and tuple(entry.get("action_components", ())) == VPP_ACTION_COMPONENTS
            and allowed_profiles == base_allowed_profiles(skill_name)
            and all(profile_usable.values())
            and set(coverage.get("geometry_states", ())).issubset(known_states)
            and set(coverage.get("phases", ())).issubset(known_phases)
            and bool(coverage.get("geometry_states"))
            and bool(coverage.get("phases"))
            and readiness_gate.get("status_before_training") == "not_evaluated"
            and readiness_gate.get("strict_interface") == "66d_profile_conditioned_to_3d_normalized_vpp"
            and readiness_gate.get("profile_conditioning") == "every_allowed_profile_must_be_present"
            and readiness_gate.get("checkpoint_fallback") == "prohibited"
            and readiness_gate.get("simulator") == "JSBSim_strict_backend_only"
            and readiness_gate.get("coverage") == "declared_geometry_phase_coverage_must_be_observed"
            and readiness_gate.get("safety")
            == "ego_crash_oob_rate_must_not_exceed_two_skill_baseline_plus_0.05_per_opponent"
        )
        skill_ok = skill_ok and entry_ok
        skill_details[skill_name] = {
            "allowed_profiles": list(allowed_profiles),
            "profile_usable": profile_usable,
            "declared_geometry_phase_coverage": coverage,
            "readiness_gate": readiness_gate,
        }
    _add_check(report, name="skill_registry:declared_profiles_and_coverage", passed=skill_ok, detail=skill_details)


def _verify_p3_encoder(report: dict[str, Any], registry: Mapping[str, Any]) -> None:
    p3 = registry["frozen_inputs"]["p3_temporal_encoder"]
    checkpoint_path = Path(str(p3["checkpoint"]))
    if not checkpoint_path.is_file():
        _add_check(report, name="p3_encoder:frozen_load_and_determinism", passed=False, detail={"reason": "missing_checkpoint"})
        return
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = dict(checkpoint["model_config"])
        model = TemporalGeometryEncoder(
            input_dim=int(config["input_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            embedding_dim=int(config["embedding_dim"]),
            horizons=checkpoint["dataset"]["prediction_horizons_steps"],
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        history = np.linspace(-1.0, 1.0, num=10 * 16, dtype=np.float32).reshape(10, 16)
        first = model.encode_history(history)
        second = model.encode_history(history)
        passed = (
            bool(p3.get("frozen", False))
            and p3.get("finetune_in_p4") == "prohibited"
            and bool(checkpoint.get("frozen_for_high_level_ppo", False))
            and int(config["input_dim"]) == int(p3["base_geometry_dim"]) == 16
            and int(config["embedding_dim"]) == int(p3["embedding_dim"]) == 32
            and first.shape == (1, 32)
            and np.isfinite(first).all()
            and np.allclose(first, second, atol=1e-7)
        )
        detail = {
            "model_config": config,
            "embedding_shape": list(first.shape),
            "deterministic": bool(np.allclose(first, second, atol=1e-7)),
            "checkpoint_frozen_flag": bool(checkpoint.get("frozen_for_high_level_ppo", False)),
        }
    except Exception as exc:  # pragma: no cover - defensive report path
        passed = False
        detail = {"exception": f"{type(exc).__name__}: {exc}"}
    _add_check(report, name="p3_encoder:frozen_load_and_determinism", passed=passed, detail=detail)


def _verify_skill_interface(report: dict[str, Any]) -> None:
    observations = []
    deterministic = True
    for index, profile_name in enumerate(PROFILE_NAMES[:4]):
        profile = compile_profile(profile_name)
        contract = FiveStateObservationContract()
        observation = contract.compose(
            base_geometry=np.full(16, 0.01 * index, dtype=np.float32),
            explicit_history=np.zeros(6, dtype=np.float32),
            intent_targets=profile["target_vector"],
            intent_weights=profile["weight_vector"],
            temporal_embedding=np.zeros(32, dtype=np.float32),
        )
        observations.append(observation)
    torch.manual_seed(20260713)
    policy = ThesisSharedSkillPolicy()
    policy.eval()
    batch = torch.as_tensor(np.stack(observations), dtype=torch.float32)
    with torch.no_grad():
        first = policy(batch)
        second = policy(batch)
    deterministic = bool(torch.allclose(first, second, atol=1e-7))
    passed = (
        batch.shape == (4, OBSERVATION_DIM)
        and first.shape == (4, VPP_ACTION_DIM)
        and bool(torch.isfinite(first).all())
        and bool((first.abs() <= 1.0).all())
        and deterministic
    )
    _add_check(
        report,
        name="shared_skill_interface:profile_conditioned_66d_to_normalized_3d",
        passed=passed,
        detail={
            "input_shape": list(batch.shape),
            "output_shape": list(first.shape),
            "action_components": list(VPP_ACTION_COMPONENTS),
            "deterministic": deterministic,
        },
    )


def _verify_opponents(report: dict[str, Any], registry: Mapping[str, Any]) -> None:
    opponent_path = _resolve_repo_path(registry["frozen_inputs"]["opponent_registry"]["path"])
    opponent_registry = _load_yaml(opponent_path)
    expected_ids = tuple(registry["frozen_inputs"]["opponent_registry"]["ids"])
    entries = opponent_registry.get("opponents", {})
    instantiated: dict[str, Any] = {}
    passed = tuple(entries.keys()) == expected_ids
    for opponent_id in expected_ids:
        entry = entries.get(opponent_id, {})
        try:
            instance = _instantiate_opponent(entry)
            instantiated[opponent_id] = type(instance).__name__
        except Exception as exc:  # pragma: no cover - dependent runtime diagnostic
            instantiated[opponent_id] = f"ERROR: {type(exc).__name__}: {exc}"
            passed = False
    weights = opponent_registry.get("training_sampling", {}).get("weights", {})
    passed = passed and opponent_registry.get("training_sampling", {}).get("strategy") == "balanced_round_robin"
    passed = passed and opponent_registry.get("training_sampling", {}).get("missing_or_failed_opponent") == "fail_closed"
    passed = passed and all(float(weights.get(opponent_id, 0.0)) == 1.0 for opponent_id in expected_ids)
    _add_check(
        report,
        name="opponents:three_way_balanced_loadable",
        passed=passed,
        detail={"ids": list(expected_ids), "instances": instantiated, "weights": weights},
    )


def _verify_phase_outputs(
    report: dict[str, Any], registry: Mapping[str, Any], config_path: Path, stage: str
) -> dict[str, Any]:
    config = _load_yaml(config_path)
    retention = _load_yaml(_resolve_repo_path(registry["frozen_inputs"]["output_retention_policy"]["path"]))
    output_root = Path(str(config["outputs"]["root"]))
    policy_root = Path(str(retention["output_root"]))
    root_is_fresh = not output_root.exists() or not any(output_root.iterdir())
    free_gb = shutil.disk_usage(output_root.anchor).free / (1024**3)
    sources_exist = all(_resolve_repo_path(value).is_file() for value in config["sources"].values())
    encoder = config["encoder"]
    common = (
        config["experiment"]["execution_mode"] == "readiness_only"
        and config["experiment"]["training_permitted"] is False
        and config["experiment"]["high_level_ppo_training"] == "prohibited"
        and sources_exist
        and encoder["checkpoint"] == registry["frozen_inputs"]["p3_temporal_encoder"]["checkpoint"]
        and encoder["sha256"] == registry["frozen_inputs"]["p3_temporal_encoder"]["sha256"]
        and encoder["trainable"] is False
        and encoder["fallback"] == "prohibited"
        and int(config["observation_action_contract"]["observation_dim"]) == OBSERVATION_DIM
        and int(config["observation_action_contract"]["action_dim"]) == VPP_ACTION_DIM
        and tuple(config["observation_action_contract"]["action_components"]) == VPP_ACTION_COMPONENTS
        and config["observation_action_contract"]["checkpoint_fallback"] == "prohibited"
        and config["runtime"]["simulator"] == "JSBSim"
        and config["runtime"]["strict_backend"] is True
        and config["runtime"]["predicted_target_vpp_interface"] == "frozen"
        and config["runtime"]["guidance_pid"] == "frozen_existing_chain"
        and config["runtime"]["prediction_reward"] == "prohibited"
        and config["runtime"]["trajectory_prediction"]["enabled"] is True
        and config["runtime"]["trajectory_prediction"]["freeze_predictor_during_rl"] is True
        and config["runtime"]["trajectory_prediction"]["checkpoint_strict"] is True
        and config["runtime"]["trajectory_prediction"]["integration"]["anchor_mode"] == "predicted_target"
        and config["runtime"]["virtual_point"]["anchor_mode"] == "predicted_target"
        and int(config["runtime"]["virtual_point"]["action_dim"]) == VPP_ACTION_DIM
        and config["outputs"]["must_be_fresh_and_empty"] is True
        and output_root.parent == policy_root
        and output_root.name.startswith("p4_")
        and root_is_fresh
        and int(config["outputs"]["checkpoint_retention_top_k"]) == int(retention["checkpoint_retention"]["top_k"]) == 3
        and config["outputs"]["train_full_raw_telemetry"] == "prohibited"
        and free_gb >= float(retention["minimum_free_gb_before_run"])
    )
    if stage == "geometry":
        common = common and tuple(config["skills"]["train_all"]) == SKILL_NAMES
        common = common and config["skills"]["require_profile_conditioning"] is True
    else:
        combat = config["combat_sampling"]
        common = common and tuple(combat["opponents"]) == tuple(registry["frozen_inputs"]["opponent_registry"]["ids"])
        common = common and combat["strategy"] == "balanced_round_robin"
        common = common and combat["missing_or_failed_opponent"] == "fail_closed"
        common = common and combat["aggregate_opponents"] == "prohibited"
        common = common and config["skills"]["require_geometry_pretrain_checkpoint_for_each_skill"] is True
    _add_check(
        report,
        name=f"{stage}_config:readiness_only_and_isolated_output",
        passed=common,
        detail={
            "config": str(config_path),
            "output_root": str(output_root),
            "output_root_fresh": root_is_fresh,
            "free_gb": round(free_gb, 2),
            "sources_exist": sources_exist,
        },
    )
    return config


def _run_jsbsim_probe(report: dict[str, Any], geometry_config: Mapping[str, Any]) -> None:
    """Run one strict reset/step using a bounded profile-conditioned VPP action."""

    try:
        from uav_vpp_guidance.envs.expert_opponent import ExpertOpponent
        from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv

        base = _load_yaml(REPO_ROOT / "config" / "experiment" / "jsbsim_hrl_comparison.yaml")
        runtime = geometry_config["runtime"]
        base["backend"] = "jsbsim"
        base.setdefault("env", {}).update(
            {
                "use_jsbsim": True,
                "strict_backend": True,
                "legacy_project_root": str(runtime["legacy_project_root"]),
                "aircraft_model": str(runtime["aircraft_model"]),
                "high_level_dt": float(runtime["high_level_dt_s"]),
                "max_high_level_steps": 2,
                "success_range_m": 0.0,
                "success_ata_deg": 0.0,
                "success_hold_time_s": 1.0,
                "max_range_m": 12000.0,
            }
        )
        base["trajectory_prediction"] = copy.deepcopy(runtime["trajectory_prediction"])
        base["virtual_point"] = copy.deepcopy(runtime["virtual_point"])
        env = CloseRangeTrackingEnv(
            base,
            opponent_policy=ExpertOpponent({}),
            opponent_config={"stage": "expert_rule_based"},
        )
        scenario = {
            "name": "p4_strict_interface_probe",
            "own_init": {"position_m": [0.0, 0.0, 5000.0], "velocity_mps": 200.0, "heading_deg": 0.0},
            "target_init": {"position_m": [2000.0, 0.0, 5000.0], "velocity_mps": 200.0, "heading_deg": 180.0},
            "metadata": {"initial_class": "head_on", "height_condition": "co_altitude", "mirror_sign": "positive"},
        }
        try:
            observation = env.reset(scenario=scenario, seed=20260713)
            if env._backend != "jsbsim":
                raise RuntimeError(f"backend fallback is prohibited, got {env._backend}")
            profile = compile_profile("front_intercept")
            contract_observation = FiveStateObservationContract().compose(
                base_geometry=np.asarray(observation["observation_vector"], dtype=np.float32),
                explicit_history=np.zeros(6, dtype=np.float32),
                intent_targets=profile["target_vector"],
                intent_weights=profile["weight_vector"],
                temporal_embedding=np.zeros(32, dtype=np.float32),
            )
            torch.manual_seed(20260713)
            policy = ThesisSharedSkillPolicy().eval()
            with torch.no_grad():
                action = policy(torch.as_tensor(contract_observation[None, :], dtype=torch.float32))[0].cpu().numpy()
            next_observation, _reward, _terminated, _truncated, info = env.step(action)
            passed = (
                np.asarray(observation["observation_vector"]).shape == (16,)
                and contract_observation.shape == (66,)
                and action.shape == (3,)
                and np.isfinite(action).all()
                and np.all(np.abs(action) <= 1.0)
                and np.asarray(next_observation["observation_vector"]).shape == (16,)
            )
            detail = {
                "backend": env._backend,
                "base_observation_dim": int(np.asarray(observation["observation_vector"]).size),
                "profile_conditioned_observation_dim": int(contract_observation.size),
                "vpp_action_dim": int(action.size),
                "step_reason": info.get("reason"),
            }
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()
    except Exception as exc:  # pragma: no cover - external simulator diagnostics
        passed = False
        detail = {"exception": f"{type(exc).__name__}: {exc}"}
    _add_check(report, name="jsbsim:strict_66d_to_3d_vpp_interface_probe", passed=passed, detail=detail)


def build_readiness_report(
    registry_path: Path = DEFAULT_REGISTRY,
    geometry_config_path: Path = DEFAULT_GEOMETRY_CONFIG,
    combat_config_path: Path = DEFAULT_COMBAT_CONFIG,
    *,
    run_jsbsim_probe: bool = True,
) -> dict[str, Any]:
    """Build a non-mutating P4 readiness report for scripts and tests."""

    registry = _load_yaml(registry_path)
    report: dict[str, Any] = {
        "schema_version": 1,
        "family": registry.get("lane", {}).get("family"),
        "stage": "p4_readiness_only",
        "training_launched": False,
        "training_launch_allowed_by_configs": False,
        "skill_readiness_status": {
            skill: "not_evaluated_untrained" for skill in SKILL_NAMES
        },
        "checks": [],
        "errors": [],
    }
    _verify_frozen_inputs(report, registry)
    _verify_contract_profiles_and_mask(report, registry)
    _verify_p3_encoder(report, registry)
    _verify_skill_interface(report)
    _verify_opponents(report, registry)
    geometry_config = _verify_phase_outputs(report, registry, geometry_config_path, "geometry")
    _verify_phase_outputs(report, registry, combat_config_path, "combat")
    if run_jsbsim_probe:
        _run_jsbsim_probe(report, geometry_config)
    else:
        report["checks"].append({"name": "jsbsim:strict_66d_to_3d_vpp_interface_probe", "passed": None, "detail": {"skipped": True}})
    report["static_readiness_pass"] = not report["errors"]
    report["status"] = (
        "passed_ready_for_explicit_future_authorization_only"
        if report["static_readiness_pass"]
        else "failed_readiness_do_not_train"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--geometry-config", type=Path, default=DEFAULT_GEOMETRY_CONFIG)
    parser.add_argument("--combat-config", type=Path, default=DEFAULT_COMBAT_CONFIG)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--skip-jsbsim-probe", action="store_true")
    args = parser.parse_args()
    report = build_readiness_report(
        args.registry,
        args.geometry_config,
        args.combat_config,
        run_jsbsim_probe=not args.skip_jsbsim_probe,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["static_readiness_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
