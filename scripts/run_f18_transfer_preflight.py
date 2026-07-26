#!/usr/bin/env python
"""Create portable static F18 transfer preflight evidence."""
import argparse,hashlib,json,sys
from pathlib import Path
import yaml
from uav_vpp_guidance.common.artifact_contract import ArtifactContract
from uav_vpp_guidance.evaluation.f18_transfer_preflight import preflight
from uav_vpp_guidance.pipeline.artifact_bundle import ArtifactBundle
from uav_vpp_guidance.pipeline.stage_runner import PipelineStage
ROOT=Path(__file__).resolve().parents[1];SPEC=ROOT / ".kiro/specs/control-guidance-remediation"; MATRIX=ROOT / "config/experiment/jsbsim_hrl_comparison.yaml"
class F18Stage(PipelineStage):
 stage_name,stage_version="f18_transfer_preflight","1.0.0"
 def __init__(self,out,cmd):super().__init__({"analysis_only":True,"matrix":"config/experiment/jsbsim_hrl_comparison.yaml","config_overrides":[]},str(out),command_line=cmd,config_path="config/experiment/jsbsim_hrl_comparison.yaml");self.artifact_contract=ArtifactContract(required_files=["run_manifest.json","artifact_contract.json","resolved_config.yaml","f18_transfer_preflight.json","f18_evidence_gate.json"])
 def run(self):
  self.manifest.mark_started();self.output_dir.mkdir(parents=True,exist_ok=True);sources={"baseline_freeze":SPEC / "baseline_freeze.json","matrix_config":MATRIX,"comparison_runner":ROOT / "scripts/run_jsbsim_hrl_comparison.py","specification":SPEC / "f18_transfer_experiment.md"}
  for name,path in sources.items():self.manifest.record_input_file(name,path)
  self.snapshot_config();result=preflight(yaml.safe_load(MATRIX.read_text(encoding="utf-8")),ROOT);evidence={"schema_version":"1.0.0","finding_id":"F18","analysis_scope":"static preflight only; no environment, checkpoint, backend, or policy execution","results":result,"backend_provenance":{"requested":None,"final":None,"fallback_occurred":False,"status":"not_applicable_preflight_only"},"config_overrides":[]};path=self.output_dir / "f18_transfer_preflight.json";path.write_text(json.dumps(evidence,indent=2),encoding="utf-8")
  gate={"schema_version":"1.0.0","finding_id":"F18","status":"needs_more_evidence","runtime_disposition":"no_transfer_claim_or_candidate_promotion","decision":"no_runtime_authorization","timestamp_utc":"2026-07-26T00:00:00+00:00","protected_path_allowlist":[],"evidence_hashes":{"specification_sha256":hashlib.sha256(sources["specification"].read_bytes()).hexdigest(),"preflight_sha256":hashlib.sha256(path.read_bytes()).hexdigest()},"results":{"common_matrix_locked":True,"simple_and_strict_jsbsim_provenance_contract_defined":True,"required_checkpoint_inputs_available":False,"paired_runs_executed":False,"strict_jsbsim_evidence":"not_available","protected_runtime_or_config_changed":False},"blockers":["All three declared main-method policy checkpoints are absent.","No paired simple/strict-JSBSim results or per-seed deltas exist.","No run records final jsbsim backend with fallback_occurred=false."],"required_next_evidence":["Restore/hash required policy and gain artifacts.","Execute paired simple and strict-JSBSim runs for every task/seed/method with identical contracts.","Reject any fallback run as JSBSim evidence and bundle each result through the pipeline."],"baseline_retained":True};gate_path=self.output_dir / "f18_evidence_gate.json";gate_path.write_text(json.dumps(gate,indent=2),encoding="utf-8")
  for output in(path,gate_path):self.manifest.record_output_file(output.name,output)
  return self.finalize(success=True,paper_safe=True)
def main():
 p=argparse.ArgumentParser();p.add_argument("--output-dir",type=Path,default=SPEC / "evidence/f18_transfer_preflight_bundle");p.add_argument("--bundle-dir",type=Path,default=SPEC / "evidence/f18_transfer_preflight_bundle_portable");a=p.parse_args();r=F18Stage(a.output_dir,[sys.executable,*sys.argv]).run();b=ArtifactBundle(a.output_dir,a.bundle_dir);b.create();v=b.verify();print(json.dumps({"stage_success":r.success,"bundle_verification":v},indent=2));return 0 if r.success and v["valid"] else 1
if __name__=="__main__":raise SystemExit(main())