from __future__ import annotations

from collections import deque
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

from uav_vpp_guidance.evaluation.thesis_neutral_postmerge_v4_contract import (
    NeutralPostMergeV4ContractError,
    build_plan,
    evaluate_family_coverage,
    runtime_identity_index,
    validate_feasibility_manifest,
)
from uav_vpp_guidance.evaluation.thesis_neutral_postmerge_v4_record_only import (
    RewardFreeNeutralPostMergeEpisode,
    record_only_result,
    runtime_record,
)


ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "experiment" / "thesis_neutral_postmerge_reentry_recovery_v4_runtime_feasibility.yaml"
MANIFEST = ROOT / "config" / "experiment" / "manifests" / "thesis_neutral_postmerge_reentry_recovery_v4_runtime_feasibility48.yaml"
BUILDER = ROOT / "scripts" / "build_thesis_neutral_postmerge_reentry_recovery_v4_feasibility_manifest.py"
PREFLIGHT = ROOT / "scripts" / "preflight_thesis_neutral_postmerge_reentry_recovery_v4_runtime_feasibility.py"
RUNNER = ROOT / "scripts" / "run_thesis_neutral_postmerge_reentry_recovery_v4_runtime_feasibility.py"
VERIFIER = ROOT / "scripts" / "verify_thesis_neutral_postmerge_reentry_recovery_v4_runtime_feasibility.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _payload(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_v4_manifest_is_disjoint_from_v3_and_preserves_the_full_48_cell_identity_contract():
    builder = _load(BUILDER, "v4_builder")
    sources = {label: builder._load_yaml(path) for label, path in builder.DISJOINT_MANIFESTS}
    rebuilt = builder.build_manifest(sources)
    tracked = _payload(MANIFEST)

    assert rebuilt["integrity"]["payload_sha256"] == tracked["integrity"]["payload_sha256"]
    assert len(validate_feasibility_manifest(tracked)) == 48
    assert {item["metadata"]["feasibility_family_id"] for item in tracked["scenarios"]} == set(builder.FAMILY_ORDER)
    assert "v3_runtime_design" in tracked["disjointness"]
    assert all(item["physical_intersection_count"] == 0 for item in tracked["disjointness"].values())
    assert all(item["seed_intersection_count"] == 0 for item in tracked["disjointness"].values())


def test_v4_design_has_no_training_or_reward_contract_and_preflight_is_nonexecuting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    config = _payload(CONFIG)
    plan = build_plan(config)
    preflight = _load(PREFLIGHT, "v4_preflight")
    runner = _load(RUNNER, "v4_runner")

    assert "proposed_training" not in config
    assert config["runtime_feasibility"]["training_reward"] == "prohibited"
    assert config["runtime_feasibility"]["environment_reward_recording"] == "prohibited"
    assert plan.execution_permitted is False
    design = copy.deepcopy(config)
    design["outputs"]["root"] = str(tmp_path / "v4_design_preflight_output")
    design_path = tmp_path / "v4_design_preflight.yaml"
    design_path.write_text(yaml.safe_dump(design, sort_keys=False), encoding="utf-8")
    result = preflight.validate_design(design_path)
    assert result["planned_records"] == 144
    assert result["training_reward"] == "prohibited"
    assert result["environment_reward_recording"] == "prohibited"
    monkeypatch.setattr(sys, "argv", ["v4-runner", "--execute"])
    assert runner.main() == 2


def test_v4_contract_rejects_a_hidden_training_or_reward_block():
    config = _payload(CONFIG)
    config["proposed_training"] = {"geometry_reward": {"intent_error_clip": 2.0}}
    with pytest.raises(NeutralPostMergeV4ContractError, match="must not define proposed_training"):
        build_plan(config)
    config = _payload(CONFIG)
    config["contract"]["environment_reward_recording"] = "allowed"
    with pytest.raises(NeutralPostMergeV4ContractError, match="runtime contract must prohibit"):
        build_plan(config)


class _FakeRecordEnv:
    max_steps = 20

    def step(self, action: np.ndarray):
        assert action.shape == (3,)
        info = {
            "first_pass_complete": True,
            "target_in_attack_zone": False,
            "ego_in_attack_zone": False,
            "ego_hp": 100.0,
            "target_hp": 100.0,
            "vp_forward_bias_m": 1.0,
            "vp_lateral_bias_m": 2.0,
            "vp_vertical_bias_m": 3.0,
            "nz_cmd": 0.2,
            "nz_g": 0.1,
            "ego_attack_aoa_deg": 10.0,
            "reason": "timeout",
            "backend_fallback_occurred": False,
        }
        return {"synthetic": True}, 999.0, False, False, info


class _FakeRecordEpisode:
    def __init__(self) -> None:
        self.observation = {"synthetic": True}
        self.base = np.ones(16, dtype=np.float32)
        self.handoff_step = 3
        self.phase = "post_merge"
        self.dynamic_state = "neutral"
        self.first_pass_complete = True
        self.last_info = {"prediction_valid": True, "prediction_fallback": False}
        self.history = deque([np.ones(16, dtype=np.float32) for _ in range(10)], maxlen=10)
        self.total_steps = 3
        self.env = _FakeRecordEnv()
        self.post_handoff_steps: list[dict] = []

    def _target_pair_allowed(self) -> bool:
        return True

    def policy_observation(self) -> np.ndarray:
        return np.ones(66, dtype=np.float32)

    def _observe(self, observation) -> None:
        self.observation = observation

    def _step_record(self, **kwargs):
        return {"observation_66d": kwargs["observation_66d"].tolist(), "raw_si_phase_input": {"range_m": 1.0, "range_rate_mps": 1.0}}


def test_v4_record_only_step_ignores_environment_reward_and_never_requires_geometry_reward():
    fake = _FakeRecordEpisode()
    _next_observation, done, metric = RewardFreeNeutralPostMergeEpisode.record_only_step(fake, [0.0, 0.0, 0.0])

    assert done is False
    assert "reward" not in metric
    assert metric["valid_target_step"] is True
    assert len(fake.post_handoff_steps) == 1


def test_v4_serializer_and_gate_exclude_reward_and_paired_scores():
    manifest = _payload(MANIFEST)
    scenario = manifest["scenarios"][0]
    result = {
        "handoff_reached": True,
        "valid_target_steps": 20,
        "qualifying": True,
        "telemetry_complete": True,
        "terminal_reason": "timeout",
        "handoff_state_sha256": "handoff",
        "training_reward": "prohibited",
        "environment_reward_recording": "prohibited",
    }
    ledger = {"summary": {"no_backend_fallback": True, "full_post_handoff_telemetry": True}}
    evidence = {
        "scenario_application_passed": True,
        "raw_si_phase_replay_passed": True,
        "initial_pre_merge_semantics_passed": True,
        "record_only_no_reward_terms": True,
    }
    record = runtime_record(
        source_id=_payload(CONFIG)["source_id"],
        opponent="expert",
        scenario=scenario,
        result=result,
        ledger=ledger,
        runtime_evidence=evidence,
    )

    assert record["paired_delta_eligible"] is False
    assert record["role"] == "nonlearning_reward_unobserved_runtime_feasibility"
    assert record["result"]["training_reward"] == "prohibited"
    assert record["result"]["environment_reward_recording"] == "prohibited"
    assert "reward" not in record["result"]
    with pytest.raises(Exception, match="reward-bearing"):
        record_only_result(
            opponent="expert",
            scenario=scenario,
            handoff={"handoff_reached": True},
            metrics=[{"reward": 1.0}],
            ledger=ledger,
        )

    config = _payload(CONFIG)
    plan = build_plan(config)
    records = []
    for item in manifest["scenarios"]:
        for opponent in plan.opponents:
            records.append(
                {
                    "opponent": opponent,
                    "family_id": item["metadata"]["feasibility_family_id"],
                    "pair_key": item["metadata"]["pair_key"],
                    "metadata": item["metadata"],
                    "result": result,
                }
            )
    gate = evaluate_family_coverage(records, plan, runtime_identity_index(manifest))
    assert gate["selected_family"]["family_id"] == "close_balanced"
    assert gate["training_reward"] == "prohibited"
    assert gate["environment_reward_recording"] == "prohibited"


def test_v4_future_authorized_overlay_must_hash_and_reject_method_changes(tmp_path: Path):
    runner = _load(RUNNER, "v4_runner_overlay")
    base = ROOT / "config" / "experiment" / "thesis_neutral_postmerge_reentry_recovery_v4_runtime_feasibility.yaml"
    overlay = {
        "base_config": str(base.relative_to(ROOT)).replace("\\", "/"),
        "status": "authorized_one_time_execution",
        "authorization": {},
        "execution": {"base_config_sha256": hashlib.sha256(base.read_bytes()).hexdigest()},
        "outputs": {"creation_permitted_by_this_config": True},
    }
    runner._validate_overlay_scope(overlay, base)
    overlay["fixed_contract"] = {"action_dim": 9}
    with pytest.raises(RuntimeError, match="cannot alter frozen method fields"):
        runner._validate_overlay_scope(overlay, base)


def test_v4_authorization_requires_the_complete_whitelist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    runner = _load(RUNNER, "v4_runner_authorized_preflight")
    config = _payload(CONFIG)
    base_sha = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    head = runner._git_value("rev-parse", "HEAD")
    original_git_value = runner._git_value

    def _clean_git_value(*args: str) -> str:
        if args == ("status", "--porcelain") or args[0] == "diff":
            return ""
        return original_git_value(*args)

    monkeypatch.setattr(runner, "_git_value", _clean_git_value)
    entries = [
        {"path": path, "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest()}
        for path in runner.REQUIRED_AUTHORIZED_CODE_PATHS
    ]
    authorization = dict(config["authorization"])
    authorization["execution_permitted"] = True
    overlay = {
        "base_config": str(CONFIG.relative_to(ROOT)).replace("\\", "/"),
        "status": "authorized_one_time_execution",
        "authorization": authorization,
        "execution": {
            "base_config_sha256": base_sha,
            "required_implementation_git_sha": head,
            "authorization_delta_paths": [],
            "authorized_code_files": entries,
        },
        "outputs": {"creation_permitted_by_this_config": True},
    }
    overlay_path = tmp_path / "v4_authorized_preflight.yaml"
    overlay_path.write_text(yaml.safe_dump(overlay, sort_keys=False), encoding="utf-8")

    merged, base_path, loaded_overlay = runner._load_authorized_config(overlay_path)
    monkeypatch.setattr(runner, "validate_design", lambda _path: {"source_id": config["source_id"]})
    runner._validate_authorization(merged, base_path, loaded_overlay)

    overlay["execution"]["authorized_code_files"] = entries[:-1]
    overlay_path.write_text(yaml.safe_dump(overlay, sort_keys=False), encoding="utf-8")
    merged, base_path, loaded_overlay = runner._load_authorized_config(overlay_path)
    with pytest.raises(RuntimeError, match="whitelist drifted"):
        runner._validate_authorization(merged, base_path, loaded_overlay)


def _runner_json_value(value):
    if isinstance(value, dict):
        return {key: _runner_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_runner_json_value(item) for item in value]
    if isinstance(value, bool):
        return int(value)
    return value


def _write_runner_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_runner_json_value(payload), indent=2) + "\n", encoding="utf-8")


def _build_synthetic_v4_bundle(tmp_path: Path) -> Path:
    from uav_vpp_guidance.training import thesis_neutral_postmerge_reentry_recovery_pilot as engine

    output = tmp_path / "v4_complete_bundle"
    output.mkdir()
    config = _payload(CONFIG)
    manifest = _payload(MANIFEST)
    resolved = copy.deepcopy(config)
    resolved["status"] = "authorized_one_time_execution"
    resolved["authorization"]["execution_permitted"] = True
    resolved["outputs"]["creation_permitted_by_this_config"] = True
    overlay = {
        "base_config": str(CONFIG.relative_to(ROOT)).replace("\\", "/"),
        "status": "authorized_one_time_execution",
        "authorization": resolved["authorization"],
        "execution": {"base_config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest()},
        "outputs": {"creation_permitted_by_this_config": True},
    }
    overlay_path = output / "authorized_overlay.yaml"
    overlay_path.write_text(yaml.safe_dump(overlay, sort_keys=False), encoding="utf-8")
    (output / "frozen_base_config.yaml").write_bytes(CONFIG.read_bytes())
    (output / "frozen_feasibility_manifest.yaml").write_bytes(MANIFEST.read_bytes())
    overlay_sha = hashlib.sha256(overlay_path.read_bytes()).hexdigest()
    base_sha = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    manifest_sha = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    _write_runner_json(
        output / "authorization_preflight.json",
        {
            "source_id": config["source_id"],
            "git_sha": "synthetic-test",
            "config_sha256": overlay_sha,
            "base_config_sha256": base_sha,
            "execution_permitted": True,
            "training_reward": "prohibited",
            "environment_reward_recording": "prohibited",
        },
    )
    _write_runner_json(output / "resolved_config.json", resolved)
    _write_runner_json(
        output / "artifact_provenance.json",
        {
            "source_id": config["source_id"],
            "git_sha": "synthetic-test",
            "authorized_overlay_sha256": overlay_sha,
            "frozen_base_config_sha256": base_sha,
            "frozen_feasibility_manifest_sha256": manifest_sha,
            "runtime_template_sha256": config["inputs"]["runtime_template_sha256"],
            "runtime_registry_sha256": config["inputs"]["runtime_registry_sha256"],
            "frozen_p3_encoder_sha256": config["inputs"]["frozen_p3_encoder_sha256"],
            "frozen_run_in_head_on_specialist_sha256": config["inputs"]["frozen_run_in_head_on_specialist_sha256"],
            "training_reward": "prohibited",
            "environment_reward_recording": "prohibited",
        },
    )
    plan = build_plan(resolved)
    records = []
    for opponent in plan.opponents:
        for scenario in manifest["scenarios"]:
            tracker = engine.legacy.PhaseTracker()
            steps = []
            for index in range(20):
                raw = {"range_m": 4000.0 - index, "range_rate_mps": -100.0}
                steps.append(
                    {
                        "stage": "post_handoff",
                        "phase": tracker.update(raw["range_m"], raw["range_rate_mps"]),
                        "raw_si_phase_input": raw,
                        "observation_66d": [0.0] * 66,
                        "normalized_vpp_action": [0.0, 0.0, 0.0],
                        "backend_fallback_occurred": False,
                    }
                )
            ledger = {
                "summary": {"no_backend_fallback": True, "full_post_handoff_telemetry": True},
                "steps": steps,
            }
            result = {
                "handoff_reached": True,
                "valid_target_steps": 20,
                "qualifying": True,
                "telemetry_complete": True,
                "terminal_reason": "timeout",
                "handoff_state_sha256": "synthetic-handoff",
                "training_reward": "prohibited",
                "environment_reward_recording": "prohibited",
            }
            evidence = {
                "scenario_application_passed": True,
                "raw_si_phase_replay_passed": True,
                "initial_pre_merge_semantics_passed": True,
                "record_only_no_reward_terms": True,
            }
            record = runtime_record(
                source_id=config["source_id"],
                opponent=opponent,
                scenario=scenario,
                result=result,
                ledger=ledger,
                runtime_evidence=evidence,
            )
            records.append(record)
            _write_runner_json(output / "episodes" / opponent / f"{scenario['name']}.json", {"record": record, "ledger": ledger})
    gate = evaluate_family_coverage(records, plan, runtime_identity_index(manifest))
    gate["training_unlocked"] = False
    _write_runner_json(output / "runtime_feasibility_gate.json", gate)
    _write_runner_json(
        output / "run_manifest.json",
        {
            "source_id": config["source_id"],
            "paper_safe": False,
            "mode": "nonlearning_reward_unobserved_runtime_feasibility",
            "record_count": len(records),
            "training_reward": "prohibited",
            "environment_reward_recording": "prohibited",
            "gate": gate,
        },
    )
    return output


def test_v4_independent_artifact_verifier_accepts_complete_bundle_and_rejects_reward_leak(tmp_path: Path):
    verifier = _load(VERIFIER, "v4_artifact_verifier")
    output = _build_synthetic_v4_bundle(tmp_path)

    report = verifier.verify_bundle(output)
    assert report["verification_status"] == "pass"
    assert report["record_count"] == 144
    assert report["recomputed_gate"]["selected_family"]["family_id"] == "close_balanced"

    episode = next((output / "episodes").rglob("*.json"))
    payload = json.loads(episode.read_text(encoding="utf-8"))
    payload["record"]["result"]["reward"] = 1.0
    episode.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(verifier.V4ArtifactVerificationError, match="reward-unobserved boundary"):
        verifier.verify_bundle(output)
