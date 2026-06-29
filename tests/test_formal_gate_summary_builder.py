from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analyze_formal_engagement_gate import analyze_engagement_gate  # noqa: E402
from build_formal_gate_summary import (  # noqa: E402
    build_prediction_comparison_summary,
    build_executive_summary,
    filter_merged_payload,
    merge_summary_payloads,
)


def test_merge_summary_payloads_accepts_summary_and_rows_inputs(tmp_path):
    head_on_path = tmp_path / "head_on_summary.json"
    crossing_path = tmp_path / "crossing_rows.json"

    head_on_path.write_text(
        json.dumps(
            {
                "summary": {
                    "head_on::adversarial::end_to_end::dNone::rNone::aNone": {
                        "task": "head_on",
                        "mode": "adversarial",
                        "opponent_stage": "end_to_end",
                        "episodes": 100,
                        "win_rate": 1.0,
                        "effective_engagement_rate": 1.0,
                        "damaging_win_rate": 1.0,
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    crossing_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "group": "crossing_feasible::no_prediction_vpp::end_to_end::d0.5::r3.0::a60.0",
                        "task": "crossing_feasible",
                        "method": "no_prediction_vpp",
                        "mode": "hp_pilot",
                        "opponent_stage": "end_to_end",
                        "episodes": 20,
                        "win_rate": 1.0,
                        "effective_engagement_rate": 1.0,
                        "damaging_win_rate": 1.0,
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    merged = merge_summary_payloads([head_on_path, crossing_path])

    assert merged["sources"] == [str(head_on_path), str(crossing_path)]
    assert len(merged["rows"]) == 2
    groups = {row["group"]: row for row in merged["rows"]}
    assert "head_on::adversarial::end_to_end::dNone::rNone::aNone" in groups
    assert "crossing_feasible::no_prediction_vpp::end_to_end::d0.5::r3.0::a60.0" in groups
    assert groups["head_on::adversarial::end_to_end::dNone::rNone::aNone"]["source_summary"] == str(
        head_on_path
    )
    assert groups[
        "crossing_feasible::no_prediction_vpp::end_to_end::d0.5::r3.0::a60.0"
    ]["source_summary"] == str(crossing_path)


def test_build_executive_summary_selects_best_config_and_large_scale_readiness():
    merged = {
        "rows": [
            {
                "group": "head_on::adversarial::end_to_end::dNone::rNone::aNone",
                "task": "head_on",
                "mode": "adversarial",
                "opponent_stage": "end_to_end",
                "episodes": 100,
                "win_rate": 1.0,
                "effective_engagement_rate": 1.0,
                "damaging_win_rate": 1.0,
                "mean_damage_dealt": 32.5,
                "mean_damage_taken": 10.0,
                "ego_crashes": 0,
                "target_crash_or_oob": 100,
                "timeouts": 0,
            },
            {
                "group": "head_on::hp_pilot::expert::d1.0::r3.0::aNone",
                "task": "head_on",
                "mode": "hp_pilot",
                "opponent_stage": "expert",
                "damage_per_step": 1.0,
                "close_range_max_km": 3.0,
                "close_range_max_aoa_deg": None,
                "episodes": 50,
                "win_rate": 1.0,
                "effective_engagement_rate": 1.0,
                "damaging_win_rate": 1.0,
                "mean_damage_dealt": 100.0,
                "mean_damage_taken": 58.0,
                "ego_crashes": 0,
                "target_crash_or_oob": 0,
                "timeouts": 0,
            },
            {
                "group": "crossing_feasible::no_prediction_vpp::end_to_end::d0.5::r3.0::a60.0",
                "task": "crossing_feasible",
                "mode": "hp_pilot",
                "opponent_stage": "end_to_end",
                "damage_per_step": 0.5,
                "close_range_max_km": 3.0,
                "close_range_max_aoa_deg": 60.0,
                "episodes": 20,
                "win_rate": 1.0,
                "effective_engagement_rate": 1.0,
                "damaging_win_rate": 1.0,
                "mean_damage_dealt": 17.1,
                "mean_damage_taken": 14.5,
                "ego_crashes": 0,
                "target_crash_or_oob": 12,
                "timeouts": 8,
            },
        ]
    }

    gate = analyze_engagement_gate(merged)
    executive = build_executive_summary(merged, gate)

    head_on = executive["tasks"]["head_on"]
    crossing = executive["tasks"]["crossing_feasible"]

    assert head_on["best_config"]["group"] == "head_on::adversarial::end_to_end::dNone::rNone::aNone"
    assert head_on["next_action"] == "formal_expand"
    assert crossing["best_config"]["damage_per_step"] == 0.5
    assert crossing["best_config"]["close_range_max_km"] == 3.0
    assert crossing["best_config"]["close_range_max_aoa_deg"] == 60.0
    assert crossing["next_action"] == "formal_expand_with_explicit_aoa60"
    assert executive["overall"]["can_run_large_scale_air_combat_experiments"] is True
    assert executive["overall"]["recommended_action"] == "proceed_large_scale_combat_experiment"


def test_build_executive_summary_prefers_formal_rows_and_preserves_method_config():
    merged = {
        "rows": [
            {
                "task": "crossing_feasible",
                "method": "no_prediction_vpp",
                "mode": "hp_pilot",
                "opponent_stage": "expert",
                "damage_per_step": 0.5,
                "close_range_max_km": 3.0,
                "close_range_max_aoa_deg": 60.0,
                "episodes": 20,
                "win_rate": 1.0,
                "effective_engagement_rate": 1.0,
                "damaging_win_rate": 1.0,
                "mean_damage_dealt": 20.0,
                "mean_damage_taken": 10.0,
                "ego_crashes": 0,
                "target_crash_or_oob": 2,
                "timeouts": 2,
            },
            {
                "task": "crossing_feasible",
                "method": "no_prediction_vpp",
                "mode": "formal",
                "run_status": "formal",
                "opponent_stage": "expert",
                "damage_per_step": 0.5,
                "close_range_max_km": 3.0,
                "close_range_max_aoa_deg": 60.0,
                "episodes": 100,
                "win_rate": 1.0,
                "effective_engagement_rate": 1.0,
                "damaging_win_rate": 1.0,
                "mean_damage_dealt": 30.0,
                "mean_damage_taken": 8.0,
                "ego_crashes": 0,
                "target_crash_or_oob": 5,
                "timeouts": 1,
            },
            {
                "task": "head_on",
                "method": "no_prediction_vpp",
                "mode": "formal",
                "run_status": "formal",
                "opponent_stage": "expert",
                "episodes": 100,
                "win_rate": 1.0,
                "effective_engagement_rate": 1.0,
                "damaging_win_rate": 1.0,
                "mean_damage_dealt": 40.0,
                "mean_damage_taken": 5.0,
                "ego_crashes": 0,
                "target_crash_or_oob": 10,
                "timeouts": 0,
            },
        ]
    }

    gate = analyze_engagement_gate(merged)
    executive = build_executive_summary(merged, gate)
    crossing = executive["tasks"]["crossing_feasible"]

    assert crossing["best_config"]["method"] == "no_prediction_vpp"
    assert crossing["best_config"]["mode"] == "formal"
    assert crossing["best_config"]["run_status"] == "formal"
    assert crossing["best_config"]["close_range_max_aoa_deg"] == 60.0


def test_filter_merged_payload_keeps_only_requested_method_and_marks_filters():
    merged = {
        "sources": ["a.json", "b.json"],
        "rows": [
            {
                "group": "head_on::no_prediction_vpp::expert::d0.5::r3.0::a40.0",
                "task": "head_on",
                "method": "no_prediction_vpp",
                "mode": "formal",
                "opponent_stage": "expert",
                "episodes": 120,
                "win_rate": 1.0,
                "effective_engagement_rate": 1.0,
                "damaging_win_rate": 1.0,
                "mean_damage_dealt": 40.0,
                "mean_damage_taken": 10.0,
                "ego_crashes": 0,
                "target_crash_or_oob": 1,
                "timeouts": 0,
            },
            {
                "group": "head_on::prediction_vpp::expert::d0.5::r3.0::a40.0",
                "task": "head_on",
                "method": "prediction_vpp",
                "mode": "formal",
                "opponent_stage": "expert",
                "episodes": 120,
                "win_rate": 0.1,
                "effective_engagement_rate": 1.0,
                "damaging_win_rate": 0.1,
                "mean_damage_dealt": 15.0,
                "mean_damage_taken": 30.0,
                "ego_crashes": 0,
                "target_crash_or_oob": 0,
                "timeouts": 0,
            },
            {
                "group": "crossing_feasible::no_prediction_vpp::end_to_end::d0.5::r3.0::a60.0",
                "task": "crossing_feasible",
                "method": "no_prediction_vpp",
                "mode": "formal",
                "opponent_stage": "end_to_end",
                "episodes": 120,
                "win_rate": 0.8,
                "effective_engagement_rate": 1.0,
                "damaging_win_rate": 0.8,
                "mean_damage_dealt": 20.0,
                "mean_damage_taken": 12.0,
                "ego_crashes": 0,
                "target_crash_or_oob": 50,
                "timeouts": 20,
            },
        ],
    }

    filtered = filter_merged_payload(merged, include_methods=["no_prediction_vpp"])
    gate = analyze_engagement_gate(filtered)
    executive = build_executive_summary(filtered, gate)

    assert filtered["sources"] == ["a.json", "b.json"]
    assert len(filtered["rows"]) == 2
    assert {row["method"] for row in filtered["rows"]} == {"no_prediction_vpp"}
    assert filtered["filters"]["include_methods"] == ["no_prediction_vpp"]
    assert executive["overall"]["filters"] == {"include_methods": ["no_prediction_vpp"]}
    assert executive["overall"]["can_run_large_scale_air_combat_experiments"] is True


def test_build_prediction_comparison_summary_computes_deltas_vs_no_prediction():
    merged = {
        "rows": [
            {
                "group": "head_on::no_prediction_vpp::expert::d0.5::r3.0::a40.0",
                "task": "head_on",
                "method": "no_prediction_vpp",
                "mode": "formal",
                "run_status": "formal",
                "opponent_stage": "expert",
                "episodes": 480,
                "win_rate": 1.0,
                "effective_engagement_rate": 1.0,
                "damaging_win_rate": 1.0,
                "mean_damage_dealt": 56.0,
                "mean_damage_taken": 29.0,
                "ego_crashes": 0,
                "target_crash_or_oob": 1,
            },
            {
                "group": "head_on::prediction_vpp::expert::d0.5::r3.0::a40.0",
                "task": "head_on",
                "method": "prediction_vpp",
                "mode": "formal",
                "run_status": "formal",
                "opponent_stage": "expert",
                "episodes": 480,
                "win_rate": 0.15,
                "effective_engagement_rate": 1.0,
                "damaging_win_rate": 0.125,
                "mean_damage_dealt": 30.0,
                "mean_damage_taken": 58.0,
                "ego_crashes": 3,
                "target_crash_or_oob": 0,
            },
            {
                "group": "crossing_feasible::no_prediction_vpp::end_to_end::d0.5::r3.0::a60.0",
                "task": "crossing_feasible",
                "method": "no_prediction_vpp",
                "mode": "formal",
                "run_status": "formal",
                "opponent_stage": "end_to_end",
                "episodes": 480,
                "win_rate": 0.82,
                "effective_engagement_rate": 1.0,
                "damaging_win_rate": 0.82,
                "mean_damage_dealt": 18.5,
                "mean_damage_taken": 17.0,
                "ego_crashes": 0,
                "target_crash_or_oob": 262,
            },
            {
                "group": "crossing_feasible::prediction_vpp::end_to_end::d0.5::r3.0::a60.0",
                "task": "crossing_feasible",
                "method": "prediction_vpp",
                "mode": "formal",
                "run_status": "formal",
                "opponent_stage": "end_to_end",
                "episodes": 480,
                "win_rate": 0.91,
                "effective_engagement_rate": 1.0,
                "damaging_win_rate": 0.9,
                "mean_damage_dealt": 22.0,
                "mean_damage_taken": 15.0,
                "ego_crashes": 0,
                "target_crash_or_oob": 312,
            },
        ]
    }

    comparison = build_prediction_comparison_summary(merged)

    assert comparison["baseline_method"] == "no_prediction_vpp"
    assert comparison["methods"] == ["no_prediction_vpp", "prediction_vpp"]
    assert comparison["tasks"] == ["head_on", "crossing_feasible"]
    assert comparison["opponent_stages"] == ["expert", "end_to_end"]

    rows = {
        (row["task"], row["opponent_stage"], row["method"]): row
        for row in comparison["rows"]
    }
    assert comparison["missing_rows"] == [
        {"task": "head_on", "opponent_stage": "end_to_end", "method": "no_prediction_vpp"},
        {"task": "head_on", "opponent_stage": "end_to_end", "method": "prediction_vpp"},
        {"task": "crossing_feasible", "opponent_stage": "expert", "method": "no_prediction_vpp"},
        {"task": "crossing_feasible", "opponent_stage": "expert", "method": "prediction_vpp"},
    ]

    head_on_baseline = rows[("head_on", "expert", "no_prediction_vpp")]
    head_on_prediction = rows[("head_on", "expert", "prediction_vpp")]
    crossing_prediction = rows[("crossing_feasible", "end_to_end", "prediction_vpp")]

    assert head_on_baseline["delta_win_rate_vs_no_prediction"] == 0.0
    assert head_on_baseline["delta_damaging_win_rate_vs_no_prediction"] == 0.0
    assert head_on_baseline["delta_damage_margin_vs_no_prediction"] == 0.0

    assert head_on_prediction["damage_margin"] == -28.0
    assert head_on_prediction["delta_win_rate_vs_no_prediction"] == -0.85
    assert head_on_prediction["delta_damaging_win_rate_vs_no_prediction"] == -0.875
    assert head_on_prediction["delta_damage_margin_vs_no_prediction"] == -55.0

    assert crossing_prediction["delta_win_rate_vs_no_prediction"] == pytest.approx(0.09)
    assert crossing_prediction["delta_damaging_win_rate_vs_no_prediction"] == pytest.approx(0.08)
    assert crossing_prediction["delta_damage_margin_vs_no_prediction"] == pytest.approx(5.5)
