from __future__ import annotations

from pathlib import Path

import yaml

from uav_vpp_guidance.training import thesis_neutral_postmerge_reentry_recovery_pilot_v2 as pilot


ROOT = Path(__file__).resolve().parent.parent
DEV = ROOT / "config" / "experiment" / "manifests" / "thesis_neutral_postmerge_reentry_recovery_pilot_v2_dev12.yaml"


def test_v2_episode_result_uses_manifest_pair_key_not_missing_scenario_signature():
    scenario = yaml.safe_load(DEV.read_text(encoding="utf-8"))["scenarios"][0]
    handoff = {"handoff_reached": True, "handoff_state_sha256": "handoff"}
    metrics = [{"valid_target_step": True, "after_intent_loss": 0.4, "range_rate_mps": 1.0, "specific_energy_height_delta_m": 2.0, "target_in_attack_zone": False, "ego_in_attack_zone": False, "terminal_reason": "timeout", "ego_failure": False, "win": False, "loss": False, "draw": True, "ego_hp": 100.0, "target_hp": 100.0} for _ in range(20)]
    ledger = {"summary": {"full_post_handoff_telemetry": True}}
    result = pilot._episode_result("expert", "candidate_fixed_reentry_recovery", scenario, handoff, metrics, ledger)
    assert result["pair_key"] == scenario["metadata"]["pair_key"]
    assert result["scenario_signature"] == scenario["metadata"]["pair_key"]
