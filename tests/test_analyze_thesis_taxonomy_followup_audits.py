from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "analyze_thesis_taxonomy_followup_audits.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("followup_audits", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trajectory(method: str, *, target_speed_offset: float = 0.0):
    frames = []
    for step, time_s in enumerate((1.0, 5.0, 12.0, 16.0)):
        target_x = 2500.0 + (220.0 + target_speed_offset) * time_s
        frame = {
            "step": step + 1,
            "time_s": time_s,
            "ego_pos_x": 220.0 * time_s,
            "ego_pos_y": 0.0,
            "ego_pos_z": 5000.0,
            "target_pos_x": target_x,
            "target_pos_y": 0.0,
            "target_pos_z": 5400.0 + target_speed_offset,
            "range_m": 2500.0 - 40.0 * step,
            "range_rate_mps": -180.0 + target_speed_offset,
            "target_in_attack_zone": step >= 1,
            "ego_in_attack_zone": step >= 2,
            "post_merge": step >= 2,
            "vp_forward_bias_m": 100.0 + step,
            "vp_lateral_bias_m": -50.0,
            "vp_pos_z": 5100.0,
            "nz_cmd": 2.0,
            "nz_g": 1.5,
            "nz_saturated": 0.0,
            "roll_rate_saturated": 0.0,
            "throttle_saturated": 0.0,
        }
        if method == "canonical_ppo_high_level_policy":
            frame.update(
                {
                    "commander_requested_mode_name": "head_on_specialist",
                    "commander_mode_name": "head_on_specialist",
                    "commander_mode_constraint_triggered": False,
                    "commander_mode_constraint_reason": "none",
                    "commander_mode_switched": False,
                    "commander_first_switch_step": None,
                }
            )
        frames.append(frame)
    return frames


def _write_raw(path: Path, method: str, *, win: bool, target_speed_offset: float = 0.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "win": win,
                "loss": not win,
                "termination_reason": "target_killed" if win else "ego_killed",
                "trajectory": _trajectory(method, target_speed_offset=target_speed_offset),
            }
        ),
        encoding="utf-8",
    )


def test_frame_summary_uses_predeclared_windows():
    module = _load_module()
    early = module._frame_summary(_trajectory("fixed"), "early_t_le_10s")
    post = module._frame_summary(_trajectory("fixed"), "post_merge")

    assert early["frame_count"] == 2
    assert post["frame_count"] == 2
    assert early["mean_target_speed_mps"] == pytest.approx(220.0)
    assert early["mean_vp_vertical_offset_m"] == pytest.approx(100.0)


def test_followup_audits_select_only_mechanical_candidate_sets(tmp_path):
    module = _load_module()
    analysis_root = tmp_path / "base_analysis"
    analysis_root.mkdir()
    records = []
    diagnostics = []
    # Five expert/disadvantage candidates, each paired across two fixed methods
    # and two opponent splits.
    for index in range(5):
        scenario = f"disadvantage_{index}"
        diagnostics.append(
            {
                "opponent_stage": "expert",
                "initial_class": "disadvantage",
                "scenario": scenario,
                "height_condition": "co_altitude",
                "mirror_sign": "negative",
                "labels": "library_gap_candidate;all_method_failure",
                "ppo_outcome": "loss",
                "fixed_crossing_outcome": "loss",
            }
        )
        for split in ("expert", "end_to_end"):
            for method in (module.FIXED_HEAD_ON_METHOD, module.FIXED_CROSSING_METHOD):
                raw_path = tmp_path / "raw" / split / scenario / f"{method}.json"
                _write_raw(raw_path, method, win=split == "end_to_end", target_speed_offset=30.0 if split == "expert" else 0.0)
                records.append(
                    {
                        "opponent_stage": split,
                        "scenario": scenario,
                        "method": method,
                        "source_raw_path": str(raw_path),
                    }
                )
    # Three head-on routing candidates, each with PPO and fixed crossing raw files.
    for index in range(3):
        scenario = f"head_on_{index}"
        diagnostics.append(
            {
                "opponent_stage": "expert",
                "initial_class": "head_on",
                "scenario": scenario,
                "height_condition": "own_above",
                "mirror_sign": "positive",
                "labels": "routing_opportunity",
                "ppo_outcome": "loss",
                "fixed_crossing_outcome": "win",
            }
        )
        for method, win in (
            (module.PPO_METHOD, False),
            (module.FIXED_CROSSING_METHOD, True),
        ):
            raw_path = tmp_path / "raw" / "expert" / scenario / f"{method}.json"
            _write_raw(raw_path, method, win=win)
            records.append(
                {
                    "opponent_stage": "expert",
                    "scenario": scenario,
                    "method": method,
                    "source_raw_path": str(raw_path),
                }
            )
    (analysis_root / "geometry_taxonomy_audit.json").write_text(
        json.dumps({"episode_records": records}), encoding="utf-8"
    )
    with (analysis_root / "diagnostic_label_by_episode.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(diagnostics[0]))
        writer.writeheader()
        writer.writerows(diagnostics)
    (analysis_root / "artifact_source_hash_manifest.json").write_text("{}\n", encoding="utf-8")
    output = tmp_path / "followup"

    result = module.analyze(analysis_root=analysis_root, output_dir=output)

    assert result["opponent_divergence_scenarios"] == 5
    assert result["opponent_divergence_pairs"] == 10
    assert result["headon_routing_scenarios"] == 3
    for name in (
        "expert_disadvantage_opponent_divergence_audit.json",
        "expert_disadvantage_opponent_divergence_audit.csv",
        "expert_disadvantage_opponent_divergence_audit_zh.md",
        "expert_headon_routing_audit.json",
        "expert_headon_routing_audit.csv",
        "expert_headon_routing_audit_zh.md",
        "artifact_source_hash_manifest.json",
    ):
        assert (output / name).is_file()
    opponent = json.loads(
        (output / "expert_disadvantage_opponent_divergence_audit.json").read_text()
    )
    assert opponent["early_divergence_label_counts"] == {
        "early_opponent_behavior_divergence_candidate": 10
    }
