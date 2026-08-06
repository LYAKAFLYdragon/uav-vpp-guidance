"""Static F14 observation availability audit tests."""
import json
from pathlib import Path
from uav_vpp_guidance.evaluation.f14_observation_evidence import BASE_FEATURES, audit
ROOT=Path(__file__).resolve().parents[1]; SPEC=ROOT / ".kiro/specs/control-guidance-remediation"
def test_f14_audit_preserves_immutable_16_feature_base_order():
    result=audit(); assert result["base_schema"]["dim"]==16; assert result["base_schema"]["ordered_features"]==BASE_FEATURES; assert result["base_schema"]["mutation"].startswith("prohibited")
def test_f14_audit_distinguishes_existing_optional_and_absent_candidates_with_leakage_policy():
    signals={row["signal"]:row for row in audit()["signals"]}; assert signals["guidance_gains"]["status"]=="existing_optional"; assert signals["aoa_beta_roll_angular_rates"]["status"]=="candidate_absent"; assert "causal" in signals["target_turn_rate"]["leakage"]
def test_f14_gate_blocks_schema_mutation_pending_protocol_and_checkpoint_migration():
    gate=json.loads((SPEC / "evidence/f14_observation_evidence_bundle/f14_evidence_gate.json").read_text(encoding="utf-8")); assert gate["status"]=="needs_more_evidence"; assert gate["protected_path_allowlist"]==[]; assert gate["runtime_disposition"]=="preserve_observation_schema_and_base_feature_order"