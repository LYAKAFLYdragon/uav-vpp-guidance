from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parent.parent
FORMAL_CONFIG = ROOT / "config" / "experiment" / "jsbsim_hrl_thesis_dualskill_taxonomy60_formal.yaml"
MANIFEST = ROOT / "config" / "experiment" / "manifests" / "thesis_dualskill_taxonomy60_v1.yaml"
DECISION = ROOT / "config" / "experiment" / "decision_contracts" / "thesis_dualskill_taxonomy60_no_go_v3.yaml"
RUNNER = ROOT / "scripts" / "run_thesis_dualskill_taxonomy60_formal.py"
VERIFIER = ROOT / "scripts" / "verify_thesis_dualskill_taxonomy60_formal.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _step(method: str, task: str):
    is_ppo = method in {"canonical_ppo", "ppo_unbootstrapped_crossing"}
    bootstrap_enabled = method == "canonical_ppo"
    return {
        "vpp_action": [0.1, -0.2, 0.3],
        "aoa_deg": 4.0,
        "nz_cmd": 1.2,
        "nz_actual": 1.1,
        "guidance_telemetry": {
            "effective_guidance_mode": "los_rate",
            "nz_cmd": 1.2,
            "roll_rate_cmd": 0.1,
            "throttle_cmd": 0.7,
            "nz_actual": 1.1,
        },
        "clipping": {
            "nz_saturated": False,
            "roll_rate_saturated": False,
            "throttle_saturated": False,
            "any_saturated": False,
        },
        "prediction_valid": False,
        "prediction_fallback": True,
        "commander_requested_mode_id": 0 if is_ppo else None,
        "commander_mode_id": 0 if is_ppo else None,
        "commander_mode_constraint_triggered": False,
        "commander_mode_constraint_reason": None,
        "commander_crossing_initial_task_bootstrap_enabled": bootstrap_enabled,
        "commander_crossing_initial_task_bootstrap_active": bootstrap_enabled
        and task == "crossing_feasible",
        "commander_crossing_pre_merge_mode_lock_active": False,
    }


def _record(scenario: dict, opponent: str, method: str):
    task = scenario["metadata"]["task_registry_key"]
    return {
        "controller": method,
        "task": task,
        "opponent_stage": opponent,
        "scenario": scenario["name"],
        "seed": scenario["metadata"]["scenario_seed"],
        "backend": "jsbsim",
        "strict_backend": True,
        "backend_fallback_occurred": False,
        "checkpoint_fallback_occurred": False,
        "run_status": "formal",
        "terminal_reason": "timeout",
        "outcome": "draw",
        "trajectory": [_step(method, task)],
    }


def _write_complete_formal_root(root: Path):
    verifier = _load_module("taxonomy60_formal_verifier_fixture", VERIFIER)
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    root.mkdir(parents=True, exist_ok=True)
    plan = [
        {
            "opponent": opponent,
            "run_id": "{}-{}".format(verifier.SOURCE_ID, opponent),
            "episodes": 300,
        }
        for opponent in verifier.OPPONENTS
    ]
    execution = {
        "source_id": verifier.SOURCE_ID,
        "attempt_kind": "initial",
        "execution_status": "completed",
        "formal_config_sha256": _sha256(FORMAL_CONFIG),
        "manifest_payload_sha256": manifest["integrity"]["payload_sha256"],
        "decision_contract_sha256": _sha256(DECISION),
        "decision_contract_id": "THESIS-DUALSKILL-TAXONOMY60-NO-GO-V3",
        "expected_records": 900,
        "executions": [
            {"opponent": opponent, "returncode": 0} for opponent in verifier.OPPONENTS
        ],
    }
    (root / "formal_execution_manifest.json").write_text(json.dumps(execution), encoding="utf-8")
    (root / "formal_execution_plan.json").write_text(json.dumps({"plan": plan}), encoding="utf-8")
    (root / "formal_artifact_contract.json").write_text(
        json.dumps({"required_child_runs": plan}), encoding="utf-8"
    )
    for opponent in verifier.OPPONENTS:
        run_dir = root / "{}-{}".format(verifier.SOURCE_ID, opponent)
        (run_dir / "aggregate").mkdir(parents=True)
        (run_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "paper_safe": True,
                    "extra": {
                        "backend": "jsbsim",
                        "opponent_stage": opponent,
                        "total_episodes": 300,
                        "artifact_validation": {"valid": True},
                        "checkpoint_checks": [{"exists": True}],
                    },
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "artifact_contract.json").write_text(
            json.dumps({"required_files": ["aggregate/episode_records.json"]}), encoding="utf-8"
        )
        (run_dir / "failures.json").write_text('{"failures": []}', encoding="utf-8")
        (run_dir / "observation_audit.json").write_text('{"audits": {}}', encoding="utf-8")
        records = [
            _record(scenario, opponent, method)
            for scenario in manifest["scenarios"]
            for method in verifier.METHODS
        ]
        assert len(records) == 300
        (run_dir / "aggregate" / "episode_records.json").write_text(
            json.dumps({"n_episodes": 300, "episodes": records}), encoding="utf-8"
        )
    return verifier


def test_formal_plan_is_exactly_three_strict_jsbsim_child_runs_without_outputs(tmp_path):
    runner = _load_module("taxonomy60_formal_runner_plan", RUNNER)
    output_root = tmp_path / "formal"
    plan = runner.build_formal_plan(
        config_path=FORMAL_CONFIG,
        output_root=output_root,
        python_executable="python",
    )

    assert len(plan) == 3
    assert sum(item["episodes"] for item in plan) == 900
    assert not output_root.exists()
    assert {item["opponent"] for item in plan} == {"expert", "end_to_end", "independent_ppo_vpp"}
    for item in plan:
        command = item["command"]
        assert command[command.index("--run-status") + 1] == "formal"
        assert command[command.index("--backend") + 1] == "jsbsim"
        assert command[command.index("--n-episodes") + 1] == "1"


def test_formal_execute_refuses_a_dirty_worktree_before_creating_outputs(tmp_path, monkeypatch):
    runner = _load_module("taxonomy60_formal_runner_dirty_gate", RUNNER)
    monkeypatch.setattr(runner, "_git_is_clean", lambda: False)

    with pytest.raises(RuntimeError, match="clean Git"):
        runner.execute_formal_plan(
            config_path=FORMAL_CONFIG,
            output_root=tmp_path / "formal",
            python_executable="python",
        )
    assert not (tmp_path / "formal").exists()


def test_independent_formal_verifier_accepts_the_complete_900_record_universe(tmp_path):
    verifier = _write_complete_formal_root(tmp_path)
    result = verifier.verify_formal_root(tmp_path, formal_config_path=FORMAL_CONFIG)

    assert result["verdict"] == "pass"
    assert result["records"] == 900
    decision = result["route_decision"]
    assert decision["selected_study_route"] == "paper_closure"
    assert len(decision["cell_classifications"]) == 180
    assert len(decision["route_candidates_in_precedence_order"]) == 10


def test_independent_formal_verifier_rejects_missing_vpp_telemetry_without_reading_results(tmp_path):
    verifier = _write_complete_formal_root(tmp_path)
    record_path = (
        tmp_path
        / "{}-expert".format(verifier.SOURCE_ID)
        / "aggregate"
        / "episode_records.json"
    )
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    payload["episodes"][0]["trajectory"][0].pop("vpp_action")
    record_path.write_text(json.dumps(payload), encoding="utf-8")

    result = verifier.verify_formal_root(tmp_path, formal_config_path=FORMAL_CONFIG)
    assert result["verdict"] == "fail"
    assert result["route_decision"]["selected_study_route"] == "implementation_invalid"
    assert result["route_decision"]["cell_classifications"] == []


def test_independent_formal_verifier_rejects_bootstrap_contract_drift(tmp_path):
    verifier = _write_complete_formal_root(tmp_path)
    record_path = (
        tmp_path
        / "{}-expert".format(verifier.SOURCE_ID)
        / "aggregate"
        / "episode_records.json"
    )
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    target = next(
        record
        for record in payload["episodes"]
        if record["controller"] == "canonical_ppo" and record["task"] == "crossing_feasible"
    )
    target["trajectory"][0]["commander_crossing_initial_task_bootstrap_active"] = False
    record_path.write_text(json.dumps(payload), encoding="utf-8")

    result = verifier.verify_formal_root(tmp_path, formal_config_path=FORMAL_CONFIG)
    assert result["verdict"] == "fail"
    assert "bootstrap" in result["error"]
