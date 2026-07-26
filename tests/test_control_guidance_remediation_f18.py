"""Static F18 preflight tests; no simulator rollout is claimed."""
import json
from pathlib import Path
import yaml
from uav_vpp_guidance.evaluation.f18_transfer_preflight import preflight
ROOT=Path(__file__).resolve().parents[1]; SPEC=ROOT / ".kiro/specs/control-guidance-remediation"
def test_f18_preflight_locks_the_declared_matrix_and_requires_strict_jsbsim_without_fallback():
 result=preflight(yaml.safe_load((ROOT / "config/experiment/jsbsim_hrl_comparison.yaml").read_text(encoding="utf-8")),ROOT); assert result["matrix"]["seeds"]==[0,1,2]; jsbsim=result["backend_pairs"][1]; assert jsbsim["strict_backend"] and not jsbsim["fallback_allowed"]
def test_f18_preflight_reports_missing_main_checkpoints_and_does_not_claim_transfer():
 result=preflight(yaml.safe_load((ROOT / "config/experiment/jsbsim_hrl_comparison.yaml").read_text(encoding="utf-8")),ROOT); assert any(item["role"]=="policy_checkpoint" and not item["exists"] for item in result["missing_required_inputs"])
def test_f18_gate_remains_blocked_with_no_runtime_allowlist():
 gate=json.loads((SPEC / "evidence/f18_transfer_preflight_bundle/f18_evidence_gate.json").read_text(encoding="utf-8")); assert gate["status"]=="needs_more_evidence"; assert gate["protected_path_allowlist"]==[]; assert gate["runtime_disposition"]=="no_transfer_claim_or_candidate_promotion"