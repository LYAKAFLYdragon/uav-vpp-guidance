from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "experiment" / "jsbsim_hrl_thesis_dualskill_taxonomy60_p2_smoke.yaml"
SMOKE_MANIFEST = ROOT / "config" / "experiment" / "manifests" / "thesis_dualskill_taxonomy60_engineering_smoke_v1.yaml"
FORMAL_MANIFEST = ROOT / "config" / "experiment" / "manifests" / "thesis_dualskill_taxonomy60_v1.yaml"
COMPARISON_RUNNER = ROOT / "scripts" / "run_jsbsim_hrl_comparison.py"
SMOKE_RUNNER = ROOT / "scripts" / "run_thesis_dualskill_taxonomy60_p2_smoke.py"
SMOKE_VERIFIER = ROOT / "scripts" / "verify_thesis_dualskill_taxonomy60_p2_smoke.py"


def _merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_resolved(path: Path):
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    resolved = {}
    for include in payload.pop("includes", []) or []:
        resolved = _merge(resolved, _load_resolved((path.parent / include).resolve()))
    return _merge(resolved, payload)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest_hash(payload):
    copied = copy.deepcopy(payload)
    copied["integrity"].pop("payload_sha256", None)
    raw = json.dumps(copied, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def test_smoke_config_materializes_one_nonformal_crossing_episode_per_method():
    runner = _load_module("taxonomy60_p2_comparison_runner", COMPARISON_RUNNER)
    config = _load_resolved(CONFIG)
    materialized = runner._materialize_scenario_manifest(config, CONFIG)

    assert materialized["thesis_dualskill_taxonomy60_p2_smoke"]["expected_episodes"] == 15
    assert materialized["env"]["strict_backend"] is True
    assert (
        materialized["thesis_dualskill_taxonomy60_p2_smoke"]["jsbsim_legacy_root"]
        == "E:/uav-vpp-guidance-can20260705"
    )
    assert materialized["scenario_manifest_resolution"]["task_counts"] == {
        "crossing_feasible": 1
    }
    assert materialized["run_defaults"]["main_methods"] == list(
        _load_module("taxonomy60_p2_smoke_runner", SMOKE_RUNNER).METHODS
    )
    smoke_scenario = materialized["tasks"]["crossing_feasible"]["task"]["scenarios"][0]
    formal_names = {
        scenario["name"]
        for scenario in yaml.safe_load(FORMAL_MANIFEST.read_text(encoding="utf-8"))["scenarios"]
    }
    assert smoke_scenario["name"] not in formal_names
    assert smoke_scenario["metadata"]["disposable_engineering_only"] is True

    canonical = runner.build_eval_config(
        materialized, "canonical_ppo", "crossing_feasible", "jsbsim", "expert"
    )
    unbootstrapped = runner.build_eval_config(
        materialized,
        "ppo_unbootstrapped_crossing",
        "crossing_feasible",
        "jsbsim",
        "expert",
    )
    assert canonical["commander"]["crossing_initial_task_bootstrap"]["enabled"] is True
    assert unbootstrapped["commander"]["crossing_initial_task_bootstrap"]["enabled"] is False
    assert canonical["commander"]["crossing_pre_merge_mode_lock"]["enabled"] is False
    assert unbootstrapped["commander"]["crossing_pre_merge_mode_lock"]["enabled"] is False


def test_smoke_manifest_payload_hash_is_frozen_and_matches_config():
    manifest = yaml.safe_load(SMOKE_MANIFEST.read_text(encoding="utf-8"))
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    expected = _manifest_hash(manifest)

    assert manifest["integrity"]["payload_sha256"] == expected
    assert config["scenario_manifest"]["payload_sha256"] == expected
    assert config["scenario_manifest"]["source_id"] == manifest["source_id"]


def test_smoke_plan_is_exactly_fifteen_disposable_jsbsim_episodes(tmp_path):
    smoke_runner = _load_module("taxonomy60_p2_plan_runner", SMOKE_RUNNER)
    plan = smoke_runner.build_smoke_plan(
        config_path=CONFIG,
        output_root=tmp_path,
        python_executable="python",
    )

    assert len(plan) == 3
    assert sum(item["episodes"] for item in plan) == 15
    assert {item["opponent"] for item in plan} == set(smoke_runner.OPPONENTS)
    for item in plan:
        command = item["command"]
        assert "--run-status" in command and command[command.index("--run-status") + 1] == "smoke"
        assert "--backend" in command and command[command.index("--backend") + 1] == "jsbsim"
        assert "--jsbsim-root" in command
        assert command[command.index("--jsbsim-root") + 1] == "E:/uav-vpp-guidance-can20260705"
        assert "--methods" in command
        assert command[command.index("--methods") + 1 : command.index("--tasks")] == list(
            smoke_runner.METHODS
        )


def _step(method: str):
    point = {
        "range_m": 3000.0,
        "range_rate_mps": -20.0,
        "ata_deg": 5.0,
        "aa_deg": 90.0,
        "vp_forward_bias_m": 100.0,
        "vp_lateral_bias_m": 50.0,
        "offset_z": 0.0,
        "nz_cmd": 1.2,
        "nz_g": 1.1,
        "roll_rate_cmd": 0.1,
        "throttle_cmd": 0.5,
        "prediction_valid": True,
        "prediction_fallback": False,
    }
    if method in {"canonical_ppo", "ppo_unbootstrapped_crossing"}:
        point.update(
            {
                "commander_requested_mode_id": 1,
                "commander_mode_id": 1,
                "commander_crossing_initial_task_bootstrap_enabled": method
                == "canonical_ppo",
                "commander_crossing_initial_task_bootstrap_active": method
                == "canonical_ppo",
                "commander_crossing_pre_merge_mode_lock_active": False,
                "commander_mode_constraint_triggered": False,
            }
        )
    return point


def _record(method: str, opponent: str):
    return {
        "controller": method,
        "opponent_stage": opponent,
        "scenario": "taxonomy60_engineering_smoke_crossing_entry",
        "backend": "jsbsim",
        "strict_backend": True,
        "backend_fallback_occurred": False,
        "steps": 1,
        "trajectory": [_step(method)],
    }


def _write_valid_smoke_root(root: Path):
    verifier = _load_module("taxonomy60_p2_verifier_fixture", SMOKE_VERIFIER)
    root.mkdir(parents=True, exist_ok=True)
    (root / "smoke_execution_manifest.json").write_text(
        json.dumps(
            {
                "source_id": verifier.SOURCE_ID,
                "formal_source_id": verifier.FORMAL_SOURCE_ID,
            }
        ),
        encoding="utf-8",
    )
    for opponent in verifier.OPPONENTS:
        run_dir = root / "{}-{}".format(verifier.SOURCE_ID, opponent)
        (run_dir / "aggregate").mkdir(parents=True)
        (run_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "paper_safe": False,
                    "extra": {
                        "backend": "jsbsim",
                        "total_episodes": len(verifier.METHODS),
                        "artifact_validation": {"valid": True},
                    },
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "failures.json").write_text('{"failures": []}', encoding="utf-8")
        (run_dir / "artifact_contract.json").write_text(
            json.dumps({"required_files": ["aggregate/episode_records.json"]}),
            encoding="utf-8",
        )
        (run_dir / "aggregate" / "episode_records.json").write_text(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "n_episodes": len(verifier.METHODS),
                    "episodes": [
                        _record(method, opponent) for method in verifier.METHODS
                    ],
                }
            ),
            encoding="utf-8",
        )
    return verifier


def test_independent_smoke_verifier_accepts_complete_disposable_matrix(tmp_path):
    verifier = _write_valid_smoke_root(tmp_path)
    result = verifier.verify_smoke_root(tmp_path)

    assert result["verdict"] == "pass"
    assert result["records"] == 15
    assert result["prediction_fallback_steps"] == 0
    assert result["paper_evidence"] is False


def test_independent_smoke_verifier_rejects_backend_fallback(tmp_path):
    verifier = _write_valid_smoke_root(tmp_path)
    record_path = (
        tmp_path
        / "{}-expert".format(verifier.SOURCE_ID)
        / "aggregate"
        / "episode_records.json"
    )
    records = json.loads(record_path.read_text(encoding="utf-8"))
    records["episodes"][0]["backend_fallback_occurred"] = True
    record_path.write_text(json.dumps(records), encoding="utf-8")

    with pytest.raises(verifier.SmokeVerificationError, match="backend fallback"):
        verifier.verify_smoke_root(tmp_path)
