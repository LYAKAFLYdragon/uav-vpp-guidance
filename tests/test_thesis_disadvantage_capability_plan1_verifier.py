from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parent.parent


def _module():
    path = ROOT / "scripts" / "verify_thesis_disadvantage_capability_plan1.py"
    spec = importlib.util.spec_from_file_location("plan1_verifier", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(scenario: str, method: str, opponent: str, *, win: bool, target_zone: bool, valid_phase_rule: bool = True):
    trajectory = [{"target_in_attack_zone": target_zone}]
    if method == "phase_rule_head_to_cross":
        trajectory = [
            {
                "target_in_attack_zone": target_zone,
                "phase_rule_first_pass_complete": False,
                "phase_rule_selected_specialist": "head_on" if valid_phase_rule else "crossing_feasible",
                "phase_rule_uses_future_state": False,
            },
            {
                "target_in_attack_zone": target_zone,
                "phase_rule_first_pass_complete": True,
                "phase_rule_selected_specialist": "crossing_feasible",
                "phase_rule_uses_future_state": False,
            },
        ]
    return {
        "scenario": scenario,
        "controller": method,
        "opponent_stage": opponent,
        "backend": "jsbsim",
        "strict_backend": True,
        "win": win,
        "loss": not win,
        "draw": False,
        "total_time_s": 4.0,
        "combat_reason": "timeout_hp_advantage" if win else "timeout_hp_disadvantage",
        "trajectory": trajectory,
    }


def _write_fixture(tmp_path: Path, *, valid_phase_rule: bool = True):
    module = _module()
    manifest = module.yaml.safe_load(
        (ROOT / "config/experiment/manifests/thesis_disadvantage_capability_plan1_v1.yaml").read_text(encoding="utf-8")
    )
    scenarios = [item["name"] for item in manifest["scenarios"]]
    for opponent in module.OPPONENTS:
        run_dir = tmp_path / f"{module.SOURCE_ID}-{opponent}"
        (run_dir / "aggregate").mkdir(parents=True)
        (run_dir / "failures.json").write_text('{"failures": []}\n', encoding="utf-8")
        records = []
        for index, scenario in enumerate(scenarios):
            for method in module.METHODS:
                records.append(
                    _record(
                        scenario,
                        method,
                        opponent,
                        win=method == "phase_rule_head_to_cross" and index < 2,
                        target_zone=method != "phase_rule_head_to_cross",
                        valid_phase_rule=valid_phase_rule,
                    )
                )
        (run_dir / "aggregate" / "episode_records.json").write_text(
            json.dumps({"episodes": records}), encoding="utf-8"
        )
    return module


def test_plan1_verifier_accepts_complete_paired_ledgers(tmp_path: Path):
    module = _write_fixture(tmp_path)

    result = module.verify(ROOT / "config/experiment/thesis_disadvantage_capability_plan1_v1.yaml", tmp_path)

    assert result["total_records"] == 144
    assert result["routing_explanation_supported"] is True
    assert all(item["passes_practical_routing_gate"] for item in result["opponents"].values())


def test_plan1_verifier_rejects_phase_rule_that_violates_observed_phase(tmp_path: Path):
    module = _write_fixture(tmp_path, valid_phase_rule=False)

    with pytest.raises(module.Plan1VerificationError, match="inconsistent with observed phase"):
        module.verify(ROOT / "config/experiment/thesis_disadvantage_capability_plan1_v1.yaml", tmp_path)
