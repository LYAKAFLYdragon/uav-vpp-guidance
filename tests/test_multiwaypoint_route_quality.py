import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analyze_multiwaypoint_route_quality import analyze_route_quality, load_step_rows


def test_route_quality_detects_timeout_and_high_roll(tmp_path):
    csv_path = tmp_path / "steps.csv"
    csv_path.write_text(
        "\n".join(
            [
                "seed,step,time_s,altitude_m,roll_deg,active_waypoint_index,completed_waypoints",
                "1,0,0.0,5000,10,0,0",
                "1,1,0.2,4950,95,0,0",
                "1,2,0.4,4900,130,0,0",
                "1,3,0.6,4850,20,1,0",
                "1,4,0.8,4800,10,1,0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rows = load_step_rows(csv_path)
    payload = {
        "episodes": [
            {
                "seed": 1,
                "termination_reason": "waypoints_incomplete",
                "crashed": False,
                "completed_waypoints": 0,
            }
        ]
    }

    report = analyze_route_quality(
        payload,
        rows,
        segment_timeout_s=0.2,
        high_level_dt=0.2,
        timeout_tolerance_steps=0,
        high_roll_deg=85.0,
        severe_roll_deg=120.0,
    )

    seed = report["seeds"]["1"]
    assert seed["timeout_like_segments"] == [0, 1]
    assert seed["high_roll_segments"] == [0]
    assert seed["severe_roll_segments"] == [0]
    assert "segment_timeout" in seed["issues"]
    assert "high_roll" in seed["issues"]
    assert "severe_roll" in seed["issues"]
    assert report["overall_issue_counts"]["no_waypoint_completion"] == 1


def test_route_quality_reports_segment_closest_approach_and_shaping(tmp_path):
    csv_path = tmp_path / "steps.csv"
    csv_path.write_text(
        "\n".join(
            [
                (
                    "seed,step,time_s,altitude_m,roll_deg,segment_waypoint_index,"
                    "active_waypoint_index,completed_waypoints,waypoint_range_m,"
                    "segment_min_waypoint_range_m,segment_elapsed_s,"
                    "heading_error_to_waypoint_deg,"
                    "multi_waypoint_roll_moderation_active,"
                    "multi_waypoint_roll_recovery_active,"
                    "multi_waypoint_same_bank_guard_active,"
                    "multi_waypoint_same_bank_guard_soft_cap_active,"
                    "multi_waypoint_static_waypoint_blend,"
                    "near_miss_capture_active,capture_reason"
                ),
                "1,0,0.0,5000,10,0,0,0,1000,1000,0.2,30,true,false,false,false,0.25,false,",
                "1,1,0.2,4990,20,0,0,0,850,850,0.4,-45,false,true,true,true,0.75,false,",
                "1,2,0.4,4980,30,0,1,1,940,850,0.6,15,false,false,true,false,1.0,true,near_miss_passed_waypoint",
                "1,3,0.6,4970,15,1,1,1,1200,1200,0.2,5,false,false,false,false,0.0,false,",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rows = load_step_rows(csv_path)
    payload = {
        "episodes": [
            {
                "seed": 1,
                "termination_reason": "waypoints_incomplete",
                "crashed": False,
                "completed_waypoints": 1,
            }
        ]
    }

    report = analyze_route_quality(payload, rows)

    seed = report["seeds"]["1"]
    first_segment = seed["segments"][0]
    assert first_segment["active_waypoint_index"] == 0
    assert first_segment["end_step"] == 2
    assert first_segment["min_waypoint_range_m"] == 850.0
    assert first_segment["closest_approach_step"] == 1
    assert first_segment["range_at_segment_end_m"] == 940.0
    assert first_segment["range_receded_from_closest_m"] == 90.0
    assert first_segment["max_abs_heading_error_to_waypoint_deg"] == 45.0
    assert first_segment["roll_moderation_active_steps"] == 1
    assert first_segment["roll_recovery_active_steps"] == 1
    assert first_segment["same_bank_guard_active_steps"] == 2
    assert first_segment["same_bank_guard_soft_cap_steps"] == 1
    assert first_segment["same_bank_guard_zero_steps"] == 1
    assert first_segment["max_static_waypoint_blend"] == 1.0
    assert first_segment["near_miss_capture_active_steps"] == 1
    assert first_segment["capture_reasons"] == ["near_miss_passed_waypoint"]
    assert seed["near_miss_capture_segments"] == [0]
