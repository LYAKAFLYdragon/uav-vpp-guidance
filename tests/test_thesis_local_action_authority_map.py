from __future__ import annotations

from uav_vpp_guidance.evaluation.thesis_local_action_authority_map import METHODS, summarize


def _rows() -> list[dict]:
    rows = []
    opponents = ["expert"] * 12 + ["end_to_end"] * 5 + ["independent_ppo_vpp"] * 11
    for index, opponent in enumerate(opponents):
        for method in METHODS:
            metrics = {"aa_deg": 0.0, "ata_deg": 0.0, "range_m": 0.0, "range_rate_mps": 0.0, "specific_energy_height_m": 0.0, "altitude_m": 0.0}
            if method == "lateral_plus": metrics["aa_deg"] = 3.6
            if method == "lateral_minus": metrics["aa_deg"] = -3.6
            rows.append({"cohort_key": f"state-{index}", "opponent": opponent, "method": method, "replay_matched": True, "window_complete": True, "pulse_clipping": False, "ego_failure": False, "final_metrics": metrics})
    return rows


def test_two_opponent_signed_authority_passes_without_pooling() -> None:
    gate = {"expected_records": 196, "min_pairs_per_opponent": 2, "min_abs_median_response": 0.01, "min_sign_fraction": 0.75, "opponents_required": 2}
    result = summarize(_rows(), gate)
    assert result["passes"] is True
    assert {item["axis"] for item in result["local_authority_candidates"]} == {"lateral"}


def test_replay_mismatch_cannot_supply_a_pair() -> None:
    gate = {"expected_records": 196, "min_pairs_per_opponent": 12, "min_abs_median_response": 0.01, "min_sign_fraction": 0.75, "opponents_required": 2}
    rows = _rows(); rows[0]["replay_matched"] = False
    assert summarize(rows, gate)["passes"] is False
