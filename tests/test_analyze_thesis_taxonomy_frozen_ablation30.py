from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parent.parent
ANALYZER_PATH = ROOT / "scripts" / "analyze_thesis_taxonomy_frozen_ablation30.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("taxonomy_audit", ANALYZER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trajectory(mode: str | None = None):
    frames = []
    for step, time_s in enumerate((0.2, 0.4, 0.6)):
        own_x = 44.0 * (step + 1)
        target_x = 2590.0 + 44.0 * (step + 1)
        frame = {
            "time_s": time_s,
            "ego_pos_x": own_x,
            "ego_pos_y": 0.0,
            "ego_pos_z": 5000.0,
            "target_pos_x": target_x,
            "target_pos_y": 0.0,
            "target_pos_z": 5000.0,
            "aa_deg": 0.0,
            "ata_deg": 180.0,
            "pre_merge": step < 2,
            "post_merge": step == 2,
            "range_rate_mps": -10.0 if step == 2 else -220.0,
            "range_m": 2600.0 - 40.0 * step,
            "ego_in_attack_zone": step >= 1,
            "target_in_attack_zone": step == 2,
            "ego_attack_score": 0.25 * step,
            "target_attack_score": 0.1 * step,
            "vp_pos_z": 5100.0 + 10.0 * step,
            "vp_forward_bias_m": 100.0,
            "vp_lateral_bias_m": -20.0,
            "altitude_m": 5000.0,
            "speed_mps": 220.0,
            "nz_cmd": 2.0,
            "nz_g": 1.5,
            "nz_saturated": 0.0,
            "roll_rate_saturated": 0.0,
            "throttle_saturated": 0.0,
        }
        if mode is not None:
            frame.update(
                {
                    "commander_requested_mode_name": mode,
                    "commander_mode_name": mode,
                    "commander_switch_count": 0,
                    "commander_mode_constraint_reason": "none",
                    "commander_mode_constraint_triggered": False,
                    "commander_mode_switched": False,
                    "commander_first_switch_step": None,
                }
            )
        frames.append(frame)
    return frames


def _episode(scenario: str, metadata: dict, method: str, *, win: bool, loss: bool, reason: str):
    return {
        "task": metadata["task_registry_key"],
        "controller": method,
        "seed": metadata["scenario_seed"],
        "episode": 0,
        "backend": "jsbsim",
        "strict_backend": True,
        "git_commit": "96fce9a",
        "config_sha256": "test-config",
        "opponent_config": {"type": "synthetic", "stage": "test"},
        "scenario": scenario,
        "scenario_metadata": metadata,
        "trajectory": _trajectory("head_on_specialist" if method.startswith("canonical") else None),
        "win": win,
        "loss": loss,
        "draw": False,
        "termination_reason": reason,
        "combat_reason": reason,
        "survived": win,
        "ego_hp": 90.0,
        "target_hp": 80.0,
        "hp_advantage": 10.0,
    }


def _scenario_rows():
    rows = []
    for pair_index in range(15):
        for sign in ("negative", "positive"):
            metadata = _metadata() | {
                "mirror_sign": sign,
                "mirror_pair_id": f"advantage__pair_{pair_index:02d}",
                "scenario_seed": 73000 + pair_index * 2 + (sign == "positive"),
            }
            rows.append((f"opaque_initial_geometry_case_{pair_index:02d}_{sign}", metadata))
    return rows


def _write_run(
    root: Path,
    split: str,
    outcomes: dict[str, tuple[bool, bool, str]],
    scenario_rows: list[tuple[str, dict]],
):
    for scenario, metadata in scenario_rows:
        for method, (win, loss, reason) in outcomes.items():
            raw_dir = root / "raw" / metadata["task_registry_key"] / method / f"seed_{metadata['scenario_seed']}"
            raw_dir.mkdir(parents=True, exist_ok=True)
            (raw_dir / "episode_000.json").write_text(
                json.dumps(_episode(scenario, metadata, method, win=win, loss=loss, reason=reason)),
                encoding="utf-8",
            )
    (root / "aggregate").mkdir(parents=True, exist_ok=True)
    for relative in (
        "run_manifest.json",
        "resolved_config.yaml",
        "aggregate/method_task_summary.json",
        "aggregate/combat_geometry_diagnostics.json",
    ):
        path = root / relative
        path.write_text("{}\n", encoding="utf-8")


def _write_manifest(path: Path, scenario_rows: list[tuple[str, dict]]):
    path.write_text(
        yaml.safe_dump(
            {
                "source_id": "THESIS-TAXONOMY-ABLATION30-V1",
                "scenarios": [
                    {"name": scenario, "metadata": metadata}
                    for scenario, metadata in scenario_rows
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _metadata() -> dict:
    return {
        "initial_class": "advantage",
        "height_condition": "co_altitude",
        "mirror_sign": "negative",
        "mirror_pair_id": "advantage__co_altitude",
        "task_registry_key": "head_on",
        "scenario_seed": 73000,
    }


def test_read_only_analyzer_writes_required_artifacts_and_labels(tmp_path):
    module = _load_module()
    scenario_rows = _scenario_rows()
    expert = tmp_path / "expert"
    end_to_end = tmp_path / "end_to_end"
    _write_run(
        expert,
        "expert",
        {
            "canonical_ppo_high_level_policy": (False, True, "ego_crash_or_out_of_bounds"),
            "legacy_static_oracle_task_gate": (True, False, "timeout_hp_advantage"),
            "head_on_vpp_specialist_no_routing": (True, False, "timeout_hp_advantage"),
            "crossing_vpp_specialist_no_routing": (False, True, "ego_killed"),
        },
        scenario_rows,
    )
    _write_run(
        end_to_end,
        "end_to_end",
        {
            "canonical_ppo_high_level_policy": (True, False, "timeout_hp_advantage"),
            "legacy_static_oracle_task_gate": (False, True, "ego_killed"),
            "head_on_vpp_specialist_no_routing": (False, True, "ego_killed"),
            "crossing_vpp_specialist_no_routing": (False, True, "ego_killed"),
        },
        scenario_rows,
    )
    manifest = tmp_path / "manifest.yaml"
    _write_manifest(manifest, scenario_rows)
    output = tmp_path / "analysis"

    result = module.analyze(
        expert_dir=expert,
        end_to_end_dir=end_to_end,
        manifest_path=manifest,
        output_dir=output,
    )

    assert result["record_count"] == 240
    for name in (
        "episode_pairing_verification.json",
        "taxonomy_initial_state_verification.json",
        "method_task_summary.json",
        "terminal_reason_summary.json",
        "geometry_taxonomy_audit.json",
        "geometry_taxonomy_audit.csv",
        "geometry_taxonomy_audit.md",
        "diagnostic_label_by_episode.csv",
        "initial_class_opponent_summary.csv",
        "mirror_pair_summary.csv",
        "thesis_taxonomy_ablation30_result_zh.md",
        "artifact_source_hash_manifest.json",
        "telemetry_summary_by_method.json",
        "telemetry_summary_by_method.csv",
        "opponent_capability_card.json",
        "opponent_capability_card_zh.md",
        "opponent_reference_pressure.csv",
        "expert_disadvantage_feasibility_audit.json",
        "expert_disadvantage_feasibility_audit_summary.csv",
        "expert_disadvantage_feasibility_audit.csv",
        "expert_disadvantage_feasibility_audit.md",
    ):
        assert (output / name).is_file()
    diagnostics = (output / "diagnostic_label_by_episode.csv").read_text(encoding="utf-8")
    assert "routing_opportunity" in diagnostics
    assert "learned_vs_oracle_gap" in diagnostics
    assert "composition_gain" in diagnostics
    assert "opponent_dependent" in diagnostics
    verification = json.loads((output / "taxonomy_initial_state_verification.json").read_text())
    assert verification["status"] == "passed"
    assert verification["match_count"] == 240
    telemetry = (output / "telemetry_summary_by_method.csv").read_text(encoding="utf-8")
    assert "mean_mean_vp_vertical_offset_m" in telemetry
    assert "ego_attack_zone_episode_entry_rate" in telemetry
    feasibility = json.loads(
        (output / "expert_disadvantage_feasibility_audit.json").read_text(encoding="utf-8")
    )
    assert feasibility["candidate_scenario_count"] == 0
    assert feasibility["decision"]["shared_skill_library_training_approved"] is False


def test_analyzer_rejects_incomplete_method_pairing(tmp_path):
    module = _load_module()
    scenario_rows = _scenario_rows()
    expert = tmp_path / "expert"
    end_to_end = tmp_path / "end_to_end"
    complete = {
        "canonical_ppo_high_level_policy": (True, False, "timeout_hp_advantage"),
        "legacy_static_oracle_task_gate": (True, False, "timeout_hp_advantage"),
        "head_on_vpp_specialist_no_routing": (True, False, "timeout_hp_advantage"),
        "crossing_vpp_specialist_no_routing": (True, False, "timeout_hp_advantage"),
    }
    _write_run(expert, "expert", complete, scenario_rows)
    incomplete = dict(complete)
    incomplete.pop("crossing_vpp_specialist_no_routing")
    _write_run(end_to_end, "end_to_end", incomplete, scenario_rows)
    manifest = tmp_path / "manifest.yaml"
    _write_manifest(manifest, scenario_rows)

    with pytest.raises(ValueError, match="raw episodes; expected"):
        module.analyze(
            expert_dir=expert,
            end_to_end_dir=end_to_end,
            manifest_path=manifest,
            output_dir=tmp_path / "analysis",
        )


def test_initial_geometry_does_not_read_scenario_name():
    module = _load_module()
    geometry = module._initial_geometry(_trajectory())
    assert geometry["geometry_state"] == "advantage"
    assert geometry["own_to_target_los_angle_deg"] == pytest.approx(0.0)
    assert geometry["target_velocity_to_own_los_angle_deg"] == pytest.approx(180.0)


def test_trajectory_telemetry_keeps_units_and_attack_zone_semantics():
    module = _load_module()
    telemetry = module._trajectory_telemetry(_trajectory("head_on_specialist"))

    assert telemetry["initial_range_m"] == pytest.approx(2600.0)
    assert telemetry["minimum_range_m"] == pytest.approx(2520.0)
    assert telemetry["ego_attack_zone_entry_time_s"] == pytest.approx(0.4)
    assert telemetry["target_attack_zone_entry_time_s"] == pytest.approx(0.6)
    assert telemetry["mean_vp_vertical_offset_m"] == pytest.approx(110.0)
    assert telemetry["mean_abs_nz_tracking_error_g"] == pytest.approx(0.5)
    assert telemetry["nz_saturation_step_fraction"] == pytest.approx(0.0)


def test_mode_fraction_aggregation_uses_zero_for_absent_modes():
    module = _load_module()
    records = [
        {"effective_mode_step_fractions": {"head_on_specialist": 1.0}},
        {"effective_mode_step_fractions": {"crossing_specialist": 1.0}},
    ]

    assert module._mean_mode_fraction(
        records, "effective_mode_step_fractions", "head_on_specialist"
    ) == pytest.approx(0.5)
    assert module._mean_mode_fraction(
        records, "effective_mode_step_fractions", "crossing_specialist"
    ) == pytest.approx(0.5)
