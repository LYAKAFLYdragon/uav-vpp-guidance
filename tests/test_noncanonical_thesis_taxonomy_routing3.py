from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "analyze_noncanonical_thesis_taxonomy_routing3.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("routing3_analysis", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _episode(module, scenario: str, method: str, *, win: bool, ego_crash: bool = False):
    frames = []
    for step in range(1, 15):
        forced = method == module.FORCED_METHOD and step <= 12
        frames.append(
            {
                "step": step,
                "commander_requested_mode_name": (
                    "crossing_specialist" if forced else "head_on_specialist"
                ),
                "commander_mode_name": (
                    "crossing_specialist" if forced else "head_on_specialist"
                ),
                "commander_initial_macro_mode_override_active": forced,
                "commander_initial_macro_mode_override_applied_this_step": forced
                and step == 1,
                "commander_mode_constraint_triggered": False,
                "vp_forward_bias_m": -100.0,
                "vp_lateral_bias_m": 50.0,
                "vp_pos_z": 5100.0,
                "ego_pos_z": 5000.0,
            }
        )
    return {
        "task": module.TASK,
        "opponent_stage": module.OPPONENT_STAGE,
        "controller": method,
        "scenario": scenario,
        "seed": 73000 + module.SCENARIOS.index(scenario),
        "win": win,
        "loss": not win,
        "combat_reason": "ego_crash_or_out_of_bounds" if ego_crash else "timeout",
        "trajectory": frames,
    }


def test_routing3_analyzer_applies_the_preregistered_two_of_three_gate(tmp_path):
    module = _load_module()
    run_dir = tmp_path / "run"
    (run_dir / "aggregate").mkdir(parents=True)
    episodes = []
    for index, scenario in enumerate(module.SCENARIOS):
        episodes.extend(
            [
                _episode(module, scenario, module.CANONICAL_METHOD, win=False),
                _episode(module, scenario, module.FORCED_METHOD, win=index < 2),
                _episode(module, scenario, module.FIXED_CROSSING_METHOD, win=True),
            ]
        )
    (run_dir / "aggregate" / "episode_records.json").write_text(
        json.dumps({"episodes": episodes}), encoding="utf-8"
    )
    (run_dir / "run_manifest.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "resolved_config.yaml").write_text("{}\n", encoding="utf-8")
    for episode in episodes:
        raw_dir = (
            run_dir
            / "raw"
            / module.TASK
            / episode["controller"]
            / f"seed_{episode['seed']}"
        )
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / "episode_000.json").write_text("{}\n", encoding="utf-8")

    output_dir = tmp_path / "analysis"
    result = module.analyze(run_dir=run_dir, output_dir=output_dir)

    assert result["gate_passed"] is True
    assert result["canonical_loss_to_forced_win"] == 2
    assert result["added_ego_crash_or_oob"] == 0
    for name in (
        "routing3_causal_summary.csv",
        "routing3_causal_summary.json",
        "routing3_causal_report_zh.md",
        "artifact_source_hash_manifest.json",
    ):
        assert (output_dir / name).is_file()
