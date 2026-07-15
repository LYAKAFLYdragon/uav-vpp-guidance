from __future__ import annotations

from collections import deque
import hashlib
import importlib.util
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


def test_v4_design_has_no_training_or_reward_contract_and_preflight_is_nonexecuting(monkeypatch: pytest.MonkeyPatch):
    config = _payload(CONFIG)
    plan = build_plan(config)
    preflight = _load(PREFLIGHT, "v4_preflight")
    runner = _load(RUNNER, "v4_runner")

    assert "proposed_training" not in config
    assert config["runtime_feasibility"]["training_reward"] == "prohibited"
    assert config["runtime_feasibility"]["environment_reward_recording"] == "prohibited"
    assert plan.execution_permitted is False
    result = preflight.validate_design()
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
