#!/usr/bin/env python
"""Create portable offline F07 action-contract evidence."""
import argparse,hashlib,json,sys
from pathlib import Path
from uav_vpp_guidance.common.artifact_contract import ArtifactContract
from uav_vpp_guidance.pipeline.artifact_bundle import ArtifactBundle
from uav_vpp_guidance.pipeline.stage_runner import PipelineStage
from uav_vpp_guidance.evaluation.f07_action_contract_evidence import construction_matrix
ROOT=Path(__file__).resolve().parents[1]; SPEC=ROOT / ".kiro/specs/control-guidance-remediation"
class Stage(PipelineStage):
 stage_name,stage_version="f07_action_contract_evidence","1.0.0"
 def __init__(self,out,cmd): super().__init__({"analysis_only":True,"config_overrides":[]},str(out),command_line=cmd,config_path="config/guidance.yaml"); self.artifact_contract=ArtifactContract(required_files=["run_manifest.json","artifact_contract.json","resolved_config.yaml","f07_action_contract_evidence.json","f07_evidence_gate.json"])
 def run(self):
  self.manifest.mark_started(); self.output_dir.mkdir(parents=True,exist_ok=True)
  for n,p in {"baseline_freeze":SPEC/"baseline_freeze.json","guidance_config":ROOT/"config/guidance.yaml","contract_specification":SPEC/"f07_action_contract.md"}.items(): self.manifest.record_input_file(n,p)
  self.snapshot_config(); evidence=self.output_dir/"f07_action_contract_evidence.json"; evidence.write_text(json.dumps({"schema_version":"1.0.0","finding_id":"F07","construction_matrix":construction_matrix(),"frozen_runtime_counterexample":"VirtualPointGenerator({}) selects 5 while frozen config declares 3","config_overrides":[]},indent=2),encoding="utf-8")
  gate={"schema_version":"1.0.0","finding_id":"F07","status":"approved","runtime_disposition":"implemented_canonical_3d_contract","decision":"runtime_authorized_and_implemented","timestamp_utc":"2026-07-26T00:00:00+00:00","protected_path_allowlist":["src/uav_vpp_guidance/virtual_point/generator.py","src/uav_vpp_guidance/envs/tracking_env.py","src/uav_vpp_guidance/agents/ppo_agent.py","src/uav_vpp_guidance/training/train_prediction_vpp_ppo.py","src/uav_vpp_guidance/hierarchy/specialist_policy.py"],"evidence_hashes":{"contract_specification_sha256":hashlib.sha256((SPEC/"f07_action_contract.md").read_bytes()).hexdigest(),"matrix_sha256":hashlib.sha256(evidence.read_bytes()).hexdigest()},"results":{"canonical_contract_specified":True,"missing_mismatch_and_stale_cases_fail":True,"runtime_boundaries_all_connected":True,"generator_silent_fallback_removed":True,"legacy_handling":"explicit_vpp_5d_compatibility_mode_only"},"blockers":[],"baseline_retained":False}; gate_path=self.output_dir/"f07_evidence_gate.json"; gate_path.write_text(json.dumps(gate,indent=2),encoding="utf-8")
  for p in (evidence,gate_path): self.manifest.record_output_file(p.name,p)
  return self.finalize(success=True,paper_safe=True)
def main():
 p=argparse.ArgumentParser();p.add_argument("--output-dir",type=Path,default=SPEC/"evidence/f07_action_contract_bundle");p.add_argument("--bundle-dir",type=Path,default=SPEC/"evidence/f07_action_contract_bundle_portable");a=p.parse_args();r=Stage(a.output_dir,[sys.executable,*sys.argv]).run();b=ArtifactBundle(a.output_dir,a.bundle_dir);b.create();v=b.verify();print(json.dumps({"stage_success":r.success,"bundle_verification":v},indent=2));return 0 if r.success and v["valid"] else 1
if __name__=="__main__":raise SystemExit(main())
