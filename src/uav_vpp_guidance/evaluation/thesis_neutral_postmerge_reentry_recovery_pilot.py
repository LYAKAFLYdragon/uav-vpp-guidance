"""Fail-closed design contract for the neutral post-merge single-skill pilot."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import yaml

from uav_vpp_guidance.hierarchy.shared_intent_profiles import PROFILE_NAMES, compile_profile
from uav_vpp_guidance.hierarchy.shared_intent_validity import SKILL_NAMES, TacticalContext, build_validity_mask


ROOT = Path(__file__).resolve().parents[3]
SOURCE_ID = "THESIS-NEUTRAL-POSTMERGE-REENTRY-RECOVERY-PILOT-V1"
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
METHODS = (
    "candidate_fixed_reentry_recovery",
    "frozen_fixed_head_on",
    "frozen_fixed_crossing",
)
TARGET_SKILL = "reentry_recovery"
TARGET_PROFILE = "reentry_preparation"
TARGET_STATE = "neutral"
TARGET_PHASE = "post_merge"
B6_FROZEN_PACKAGES = {
    (3400.0, 240.0, 350.0),
    (4200.0, 265.0, 370.0),
}


class NeutralPostMergePilotContractError(ValueError):
    """Raised when a design-only pilot input leaves its preregistered contract."""


@dataclass(frozen=True)
class NeutralPostMergePilotPlan:
    source_id: str
    execution_permitted: bool
    training_permitted: bool
    skill: str
    profile: str
    observation_dim: int
    action_dim: int
    dev_scenario_count: int
    heldout_scenario_count: int
    min_free_disk_gb: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise NeutralPostMergePilotContractError(f"expected YAML mapping: {path}")
    return payload


def manifest_payload_sha256(payload: Mapping[str, Any]) -> str:
    prepared = copy.deepcopy(dict(payload))
    integrity = prepared.get("integrity")
    if isinstance(integrity, dict):
        integrity.pop("payload_sha256", None)
    return hashlib.sha256(
        json.dumps(prepared, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NeutralPostMergePilotContractError(f"{label} must be a mapping")
    return value


def _require_hash(path: Path, expected: Any, label: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64:
        raise NeutralPostMergePilotContractError(f"{label} must declare a SHA-256")
    if not path.is_file() or sha256_file(path) != expected.lower():
        raise NeutralPostMergePilotContractError(f"{label} SHA-256 mismatch")


def _physical_signature(scenario: Mapping[str, Any]) -> tuple[Any, ...]:
    own = _require_mapping(scenario.get("own_init"), "scenario.own_init")
    target = _require_mapping(scenario.get("target_init"), "scenario.target_init")

    def rounded(values: Sequence[Any]) -> tuple[float, ...]:
        return tuple(round(float(value), 6) for value in values)

    return (
        rounded(own["position_m"]),
        round(float(own["velocity_mps"]), 6),
        round(float(own["heading_deg"]) % 360.0, 6),
        rounded(target["position_m"]),
        round(float(target["velocity_mps"]), 6),
        round(float(target["heading_deg"]) % 360.0, 6),
    )


def _scenario_seeds(scenarios: Sequence[Mapping[str, Any]]) -> set[int]:
    return {
        int(_require_mapping(item.get("metadata"), "scenario.metadata")["scenario_seed"])
        for item in scenarios
    }


def _validate_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_source_id: str,
    expected_count: int,
) -> list[Mapping[str, Any]]:
    scenarios = list(manifest.get("scenarios") or [])
    if manifest.get("source_id") != expected_source_id or manifest.get("status") != "preregistered_design_only":
        raise NeutralPostMergePilotContractError("pilot manifest identity or status drifted")
    if len(scenarios) != expected_count or len({item.get("name") for item in scenarios}) != expected_count:
        raise NeutralPostMergePilotContractError("pilot manifest scenario count or uniqueness drifted")
    if manifest_payload_sha256(manifest) != _require_mapping(manifest.get("integrity"), "manifest.integrity").get("payload_sha256"):
        raise NeutralPostMergePilotContractError("pilot manifest payload SHA drifted")
    signatures = {_physical_signature(item) for item in scenarios}
    if len(signatures) != expected_count or len(_scenario_seeds(scenarios)) != expected_count:
        raise NeutralPostMergePilotContractError("pilot manifest reuses a physical signature or seed")
    for scenario in scenarios:
        metadata = _require_mapping(scenario.get("metadata"), "scenario.metadata")
        if (
            metadata.get("initial_class") != TARGET_STATE
            or metadata.get("taxonomy_geometry_state") != TARGET_STATE
            or metadata.get("phase_at_reset") != "pre_merge"
            or metadata.get("continuous_run_in") is not True
            or metadata.get("routing_enabled") is not False
            or metadata.get("fixed_skill") != TARGET_SKILL
            or metadata.get("fixed_profile") != TARGET_PROFILE
            or float(metadata.get("initial_range_m", 0.0)) <= 1000.0
        ):
            raise NeutralPostMergePilotContractError("pilot scenario violates the neutral continuous-run-in contract")
    return scenarios


def _validate_disjoint_sources(manifest: Mapping[str, Any], scenarios: Sequence[Mapping[str, Any]]) -> None:
    pilot_signatures = {_physical_signature(item) for item in scenarios}
    pilot_seeds = _scenario_seeds(scenarios)
    sources = _require_mapping(manifest.get("disjointness"), "manifest.disjointness")
    if len(sources) < 9:
        raise NeutralPostMergePilotContractError("pilot manifest omitted a frozen disjointness source")
    for label, source in sources.items():
        entry = _require_mapping(source, f"manifest.disjointness.{label}")
        path = _repo_path(str(entry.get("path", "")))
        _require_hash(path, entry.get("sha256"), f"disjoint source {label}")
        frozen = load_yaml(path)
        frozen_scenarios = list(frozen.get("scenarios") or [])
        if pilot_signatures & {_physical_signature(item) for item in frozen_scenarios}:
            raise NeutralPostMergePilotContractError(f"pilot manifest overlaps frozen source: {label}")
        if pilot_seeds & _scenario_seeds(frozen_scenarios):
            raise NeutralPostMergePilotContractError(f"pilot manifest reuses frozen source seed: {label}")
        if entry.get("physical_intersection_count") != 0 or entry.get("seed_intersection_count") != 0:
            raise NeutralPostMergePilotContractError(f"pilot manifest recorded a nonzero overlap: {label}")


def _validate_profile_contract(fixed: Mapping[str, Any]) -> None:
    profile = compile_profile(TARGET_PROFILE)
    expected_targets = [30.0, 130.0, 2200.0, -30.0, 100.0, 0.0]
    expected_weights = [0.8, 0.8, 1.0, 0.8, 0.7, 0.4]
    if (
        fixed.get("profile_targets") != expected_targets
        or fixed.get("profile_weights") != expected_weights
        or list(profile["physical_targets"]) != expected_targets
        or any(
            abs(float(actual) - expected) > 1e-6
            for actual, expected in zip(profile["weight_vector"], expected_weights)
        )
    ):
        raise NeutralPostMergePilotContractError("reentry_preparation profile target or weight drifted")
    context = TacticalContext(geometry_state=TARGET_STATE, phase=TARGET_PHASE)
    mask = build_validity_mask(context)["mask"]
    if not bool(mask[SKILL_NAMES.index(TARGET_SKILL), PROFILE_NAMES.index(TARGET_PROFILE)]):
        raise NeutralPostMergePilotContractError("neutral/post_merge validity mask blocks the pilot action")


def build_design_plan(config: Mapping[str, Any]) -> NeutralPostMergePilotPlan:
    if config.get("source_id") != SOURCE_ID:
        raise NeutralPostMergePilotContractError("unexpected pilot source ID")
    if config.get("status") != "preregistered_design_only_not_authorised":
        raise NeutralPostMergePilotContractError("pilot status must remain design-only")
    authorization = _require_mapping(config.get("authorization"), "authorization")
    for key in (
        "execution_permitted",
        "training_permitted",
        "baseline_evaluation_permitted",
        "heldout_evaluation_permitted",
        "high_level_ppo_training_permitted",
        "four_skill_training_permitted",
        "combat_finetune_permitted",
        "tuning_permitted",
        "vpp_change_permitted",
        "guidance_change_permitted",
        "pid_change_permitted",
        "snapshot_restore",
        "future_state_injection",
        "history_padding_permitted",
    ):
        if authorization.get(key) is not False:
            raise NeutralPostMergePilotContractError(f"authorization.{key} must remain false")
    fixed = _require_mapping(config.get("fixed_contract"), "fixed_contract")
    if (
        fixed.get("skill") != TARGET_SKILL
        or fixed.get("profile") != TARGET_PROFILE
        or fixed.get("routing_enabled") is not False
        or fixed.get("high_level_policy_present") is not False
        or fixed.get("observation_dim") != 66
        or fixed.get("action_dim") != 3
        or _require_mapping(fixed.get("encoder"), "fixed_contract.encoder").get("trainable") is not False
        or fixed["encoder"].get("fallback") != "prohibited"
    ):
        raise NeutralPostMergePilotContractError("fixed skill/profile/encoder/routing contract drifted")
    _validate_profile_contract(fixed)
    if tuple(_require_mapping(config.get("opponents"), "opponents").get("order", ())) != OPPONENTS:
        raise NeutralPostMergePilotContractError("opponent order drifted")
    if set(_require_mapping(config.get("methods"), "methods")) != set(METHODS):
        raise NeutralPostMergePilotContractError("minimal method matrix drifted")
    if config["methods"]["candidate_fixed_reentry_recovery"].get("checkpoint") is not None:
        raise NeutralPostMergePilotContractError("candidate checkpoint must stay absent before authorization")
    return NeutralPostMergePilotPlan(
        source_id=SOURCE_ID,
        execution_permitted=False,
        training_permitted=False,
        skill=TARGET_SKILL,
        profile=TARGET_PROFILE,
        observation_dim=66,
        action_dim=3,
        dev_scenario_count=12,
        heldout_scenario_count=24,
        min_free_disk_gb=float(_require_mapping(config.get("contract"), "contract")["min_free_disk_gb"]),
    )


def validate_design(config_path: Path) -> dict[str, Any]:
    """Validate design inputs without loading a policy or starting JSBSim."""

    config = load_yaml(config_path)
    plan = build_design_plan(config)
    sources = _require_mapping(config.get("sources"), "sources")
    hashed_inputs = {
        "train_distribution": sources.get("train_distribution"),
        "dev_manifest": sources.get("dev_manifest"),
        "heldout_manifest": sources.get("heldout_manifest"),
        "runtime_registry": sources.get("runtime_registry"),
        "artifact_schema": sources.get("artifact_schema"),
    }
    for label, value in hashed_inputs.items():
        path = _repo_path(str(value))
        _require_hash(path, sources.get(f"{label}_sha256"), label)
    for label in ("manifest_builder", "preflight", "pilot_contract"):
        path = _repo_path(str(sources.get(label, "")))
        _require_hash(path, sources.get(f"{label}_sha256"), label)
    p3 = _require_mapping(_require_mapping(config["fixed_contract"], "fixed_contract").get("encoder"), "encoder")
    _require_hash(Path(str(p3.get("checkpoint", ""))), p3.get("sha256"), "frozen P3 encoder")
    methods = _require_mapping(config.get("methods"), "methods")
    for name in ("frozen_fixed_head_on", "frozen_fixed_crossing"):
        entry = _require_mapping(methods.get(name), f"methods.{name}")
        _require_hash(Path(str(entry.get("checkpoint", ""))), entry.get("checkpoint_sha256"), name)
    runtime_registry = load_yaml(_repo_path(str(sources["runtime_registry"])))
    opponents = _require_mapping(runtime_registry.get("opponents"), "runtime_registry.opponents")
    if tuple(opponents) != OPPONENTS:
        raise NeutralPostMergePilotContractError("runtime registry opponent order drifted")
    for name in ("end_to_end", "independent_ppo_vpp"):
        entry = _require_mapping(opponents.get(name), f"runtime_registry.opponents.{name}")
        _require_hash(Path(str(entry.get("checkpoint", ""))), entry.get("checkpoint_sha256"), f"opponent {name}")
    dev_manifest = load_yaml(_repo_path(str(sources["dev_manifest"])))
    heldout_manifest = load_yaml(_repo_path(str(sources["heldout_manifest"])))
    dev = _validate_manifest(
        dev_manifest,
        expected_source_id=f"{SOURCE_ID}-DEV12",
        expected_count=plan.dev_scenario_count,
    )
    heldout = _validate_manifest(
        heldout_manifest,
        expected_source_id=f"{SOURCE_ID}-HELDOUT24",
        expected_count=plan.heldout_scenario_count,
    )
    _validate_disjoint_sources(dev_manifest, dev)
    _validate_disjoint_sources(heldout_manifest, heldout)
    if {_physical_signature(item) for item in dev} & {_physical_signature(item) for item in heldout}:
        raise NeutralPostMergePilotContractError("dev and heldout manifests overlap physically")
    if _scenario_seeds(dev) & _scenario_seeds(heldout):
        raise NeutralPostMergePilotContractError("dev and heldout manifests reuse a seed")
    training = load_yaml(_repo_path(str(sources["train_distribution"])))
    support = _require_mapping(training.get("sampling_contract"), "train_distribution.sampling_contract")
    if (
        training.get("source_id") != f"{SOURCE_ID}-TRAIN"
        or training.get("training_permitted") is not False
        or support.get("initial_class") != TARGET_STATE
        or support.get("candidate_skill") != TARGET_SKILL
        or support.get("fixed_profile") != TARGET_PROFILE
        or support.get("continuous_physical_run_in") is not True
    ):
        raise NeutralPostMergePilotContractError("train distribution contract drifted")
    fixed_packages = {
        tuple(float(item[key]) for key in ("initial_range_m", "own_speed_mps", "target_speed_mps"))
        for manifest in (dev_manifest, heldout_manifest)
        for item in manifest["generation_contract"]["packages"]
    }
    banned = {tuple(float(value) for value in item) for item in support.get("forbidden_fixed_packages", [])}
    if not fixed_packages.issubset(banned):
        raise NeutralPostMergePilotContractError("train distribution did not exclude dev/heldout packages")
    if not B6_FROZEN_PACKAGES.issubset(banned):
        raise NeutralPostMergePilotContractError("train distribution did not exclude B6 packages")
    schema = load_yaml(_repo_path(str(sources["artifact_schema"])))
    if (
        schema.get("source_id") != f"{SOURCE_ID}-ARTIFACT-SCHEMA"
        or schema.get("execution_artifacts_created_by_design") is not False
        or schema.get("paper_safe") is not False
    ):
        raise NeutralPostMergePilotContractError("artifact schema must remain design-only")
    output = Path(str(_require_mapping(config.get("outputs"), "outputs").get("root", "")))
    if not output.is_absolute() or output.exists():
        raise NeutralPostMergePilotContractError("pilot output root must be absolute, new, and absent")
    free_gb = shutil.disk_usage(output.parent).free / (1024**3)
    return {
        "source_id": plan.source_id,
        "mode": "design_preflight_only_no_jsbsim_no_training",
        "execution_permitted": plan.execution_permitted,
        "training_permitted": plan.training_permitted,
        "skill": plan.skill,
        "profile": plan.profile,
        "target_window": f"{TARGET_STATE}->{TARGET_PHASE}",
        "dev_scenario_count": plan.dev_scenario_count,
        "heldout_scenario_count": plan.heldout_scenario_count,
        "planned_future_evaluation_records": (plan.dev_scenario_count + plan.heldout_scenario_count) * len(OPPONENTS) * len(METHODS),
        "opponents": list(OPPONENTS),
        "output_root_absent": True,
        "free_disk_gb": round(free_gb, 3),
        "disk_gate_would_pass": free_gb >= plan.min_free_disk_gb,
        "finite_profile_targets": all(math.isfinite(float(value)) for value in config["fixed_contract"]["profile_targets"]),
    }
