from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "analyze_thesis_dualskill_taxonomy60_formal.py"
MANIFEST = ROOT / "config" / "experiment" / "manifests" / "thesis_dualskill_taxonomy60_v1.yaml"


def _load_module():
    spec = importlib.util.spec_from_file_location("taxonomy60_formal_analysis", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_summary(path: Path, *, opponent: str, scenarios: list[dict], methods: tuple[str, ...]):
    fieldnames = [
        "scenario", "method", "opponent_stage", "seed", "episode", "task", "backend", "strict_backend",
        "backend_fallback_occurred", "run_status", "combat_outcome", "win", "loss", "draw",
        "termination_reason", "steps", "total_time_s", "mean_range_m", "mean_abs_ata_deg",
        "mean_speed_mps", "mean_altitude_m", "envelope_fraction", "saturation_fraction",
        "prediction_valid_rate", "prediction_fallback_rate", "commander_mode_head_on_fraction",
        "commander_mode_crossing_fraction", "commander_mode_recovery_fraction",
        "commander_mean_switch_count", "commander_first_switch_step",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for scenario_index, scenario in enumerate(scenarios):
            for method in methods:
                outcome = "loss" if method == "fixed_crossing" and scenario_index == 0 else "win"
                writer.writerow(
                    {
                        "scenario": scenario["name"], "method": method, "opponent_stage": opponent,
                        "seed": scenario["metadata"]["scenario_seed"], "episode": scenario_index,
                        "task": scenario["metadata"]["task_registry_key"], "backend": "jsbsim",
                        "strict_backend": "True", "backend_fallback_occurred": "False", "run_status": "formal",
                        "combat_outcome": outcome, "win": str(outcome == "win"), "loss": str(outcome == "loss"),
                        "draw": "False", "termination_reason": "timeout_hp_advantage" if outcome == "win" else "timeout_hp_disadvantage",
                        "steps": 160, "total_time_s": 32.0, "mean_range_m": 2000.0, "mean_abs_ata_deg": 40.0,
                        "mean_speed_mps": 250.0, "mean_altitude_m": 5000.0, "envelope_fraction": 0.1,
                        "saturation_fraction": 0.0, "prediction_valid_rate": 0.94, "prediction_fallback_rate": 0.06,
                        "commander_mode_head_on_fraction": 1.0 if method.startswith("canonical") else "",
                        "commander_mode_crossing_fraction": 0.0 if method.startswith("canonical") else "",
                        "commander_mode_recovery_fraction": 0.0 if method.startswith("canonical") else "",
                        "commander_mean_switch_count": 0.0 if method.startswith("canonical") else "",
                        "commander_first_switch_step": "",
                    }
                )


def _write_raw_episodes(run_dir: Path, *, opponent: str, scenarios: list[dict], methods: tuple[str, ...]):
    for scenario_index, scenario in enumerate(scenarios):
        metadata = scenario["metadata"]
        for method in methods:
            outcome = "loss" if method == "fixed_crossing" and scenario_index == 0 else "win"
            trajectory = [
                {
                    "time_s": 0.2, "pre_merge": True, "first_pass_complete": False,
                    "range_rate_mps": -100.0, "aa_deg": 20.0, "ata_deg": 20.0,
                    "ego_in_attack_zone": True, "target_in_attack_zone": False,
                    "range_m": 2400.0, "speed_mps": 250.0, "altitude_m": 5000.0,
                    "vpp_action": [0.1, -0.1, 0.2], "aoa_deg": 4.0, "nz_cmd": 1.3,
                    "nz_actual": 1.1, "clipping": {"any_saturated": False},
                },
                {
                    "time_s": 0.4, "pre_merge": False, "first_pass_complete": True,
                    "range_rate_mps": 30.0, "aa_deg": 100.0, "ata_deg": 20.0,
                    "ego_in_attack_zone": False, "target_in_attack_zone": True,
                    "range_m": 2500.0, "speed_mps": 245.0, "altitude_m": 5010.0,
                    "vpp_action": [0.2, -0.2, 0.1], "aoa_deg": 5.0, "nz_cmd": 1.4,
                    "nz_actual": 1.2, "clipping": {"any_saturated": False},
                },
                {
                    "time_s": 0.6, "pre_merge": False, "first_pass_complete": True,
                    "range_rate_mps": -30.0, "aa_deg": 100.0, "ata_deg": 100.0,
                    "ego_in_attack_zone": False, "target_in_attack_zone": False,
                    "range_m": 2450.0, "speed_mps": 240.0, "altitude_m": 5020.0,
                    "vpp_action": [0.3, -0.3, 0.0], "aoa_deg": 6.0, "nz_cmd": 1.5,
                    "nz_actual": 1.3, "clipping": {"any_saturated": True},
                },
            ]
            raw = {
                "controller": method, "task": metadata["task_registry_key"], "opponent_stage": opponent,
                "scenario": scenario["name"], "seed": metadata["scenario_seed"], "episode": scenario_index,
                "backend": "jsbsim", "strict_backend": True, "outcome": outcome,
                "terminal_reason": "timeout_hp_advantage" if outcome == "win" else "timeout_hp_disadvantage",
                "trajectory": trajectory,
            }
            raw_path = (
                run_dir / "raw" / metadata["task_registry_key"] / method
                / f"seed_{metadata['scenario_seed']}" / f"episode_{scenario_index:03d}.json"
            )
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(json.dumps(raw), encoding="utf-8")


def _write_formal_fixture(tmp_path: Path, module):
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    root = tmp_path / "formal"
    root.mkdir()
    (root / "formal_verification.json").write_text(json.dumps({"source_id": module.SOURCE_ID, "verdict": "pass", "records": 900}), encoding="utf-8")
    (root / "taxonomy60_route_decision.json").write_text(json.dumps({"source_id": module.SOURCE_ID, "formal_validity_verdict": "pass", "selected_study_route": "paper_closure", "guard_confounded_episode_ids": [], "prohibited_actions": ["training"]}), encoding="utf-8")
    (root / "formal_execution_manifest.json").write_text(json.dumps({"source_id": module.SOURCE_ID, "execution_status": "completed", "manifest_payload_sha256": manifest["integrity"]["payload_sha256"]}), encoding="utf-8")
    for opponent in module.OPPONENTS:
        run_dir = root / f"{module.SOURCE_ID}-{opponent}"
        run_dir.mkdir()
        _write_summary(run_dir / "summary.csv", opponent=opponent, scenarios=manifest["scenarios"], methods=module.METHODS)
        _write_raw_episodes(run_dir, opponent=opponent, scenarios=manifest["scenarios"], methods=module.METHODS)
    return root


def test_wilson_interval_excludes_empty_denominator():
    module = _load_module()
    assert module.wilson_95_interval(0, 0) == (None, None)
    lower, upper = module.wilson_95_interval(10, 10)
    assert lower is not None and upper is not None and upper > 0.99


def test_read_only_export_builds_complete_900_record_package(tmp_path):
    module = _load_module()
    formal_root = _write_formal_fixture(tmp_path, module)
    analysis_dir = tmp_path / "analysis"

    artifact = module.export_analysis(formal_root, analysis_dir)

    assert artifact["record_count"] == 900
    assert artifact["raw_phase_record_count"] == 900
    assert artifact["selected_study_route"] == "paper_closure"
    with (analysis_dir / "episode_index.csv").open(encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 900
    with (analysis_dir / "method_state_opponent_summary.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 90
    assert {row["opponent"] for row in rows} == set(module.OPPONENTS)
    assert (analysis_dir / "paper_closure_handoff.md").is_file()
    with (analysis_dir / "dynamic_taxonomy_phase_episode_index.csv").open(encoding="utf-8", newline="") as handle:
        phase_rows = list(csv.DictReader(handle))
    assert len(phase_rows) == 900
    assert {row["terminal_dynamic_taxonomy"] for row in phase_rows} == {"neutral"}
    assert (analysis_dir / "dynamic_taxonomy_phase_summary.csv").is_file()
    assert (analysis_dir / "phase_geometry_mechanism_summary.csv").is_file()
    assert (analysis_dir / "analysis_manifest.json").is_file()
