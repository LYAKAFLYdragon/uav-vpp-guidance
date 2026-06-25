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
