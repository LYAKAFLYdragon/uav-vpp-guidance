#!/usr/bin/env python
"""Create portable static F14 observation-information evidence."""
import argparse, hashlib, json, sys
from pathlib import Path
from uav_vpp_guidance.common.artifact_contract import ArtifactContract
from uav_vpp_guidance.evaluation.f14_observation_evidence import audit
from uav_vpp_guidance.pipeline.artifact_bundle import ArtifactBundle
from uav_vpp_guidance.pipeline.stage_runner import PipelineStage
ROOT=Path(__file__).resolve().parents[1]; SPEC=ROOT / ".kiro/specs/control-guidance-remediation"
class F14Stage(PipelineStage):
 stage_name,stage_version="f14_observation_evidence","1.0.0"
 def __init__(self,output_dir,command_line):
  super().__init__({"analysis_only":True,"input_config":"config/experiment/train_prediction_vpp_ppo_jsbsim_compare.yaml","config_overrides":[]},str(output_dir),command_line=command_line,config_path="config/experiment/train_prediction_vpp_ppo_jsbsim_compare.yaml");self.artifact_contract=ArtifactContract(required_files=["run_manifest.json","artifact_contract.json","resolved_config.yaml","f14_observation_evidence.json","f14_evidence_gate.json"])
 def run(self):
  self.manifest.mark_started();self.output_dir.mkdir(parents=True,exist_ok=True); sources={"baseline_freeze":SPEC / "baseline_freeze.json","observation_runtime_reference":ROOT / "src/uav_vpp_guidance/envs/observation.py","tracking_runtime_reference":ROOT / "src/uav_vpp_guidance/envs/tracking_env.py","training_config":ROOT / "config/experiment/train_prediction_vpp_ppo_jsbsim_compare.yaml","specification":SPEC / "f14_observation_experiment.md"}
  for name,path in sources.items():self.manifest.record_input_file(name,path)
  self.snapshot_config();evidence={"schema_version":"1.0.0","finding_id":"F14","analysis_scope":"static availability and leakage audit only; no observation implementation, config, schema, or checkpoint behavior changed","results":audit(),"backend_provenance":{"requested":None,"final":None,"fallback_occurred":False,"status":"not_applicable_static_audit"},"config_overrides":[]};path=self.output_dir / "f14_observation_evidence.json";path.write_text(json.dumps(evidence,indent=2),encoding="utf-8")
  gate={"schema_version":"1.0.0","finding_id":"F14","status":"needs_more_evidence","runtime_disposition":"preserve_observation_schema_and_base_feature_order","decision":"no_runtime_authorization","timestamp_utc":"2026-07-26T00:00:00+00:00","protected_path_allowlist":[],"evidence_hashes":{"specification_sha256":hashlib.sha256(sources["specification"].read_bytes()).hexdigest(),"offline_evidence_sha256":hashlib.sha256(path.read_bytes()).hexdigest()},"results":{"available_and_absent_signals_audited":True,"units_updates_normalization_availability_and_leakage_defined":True,"fixed_protocol_defined":True,"feature_benefit_and_backend_parity":"not_available","schema_migration":"not_approved","protected_runtime_or_config_changed":False},"blockers":["No fixed training/evaluation comparison exists for minimal candidates or history.","Frozen checkpoints and strict-JSBSim matrix evaluations are unavailable.","No schema-version/checkpoint migration plan is approved."],"required_next_evidence":["Compare baseline, one minimal signal addition, and causal history under fixed training/evaluation protocols.","Measure task/safety benefit, policy sensitivity, availability, and backend parity.","If approved, create a separate schema-version/checkpoint-migration task following AGENTS.md before touching observation.py or tracking_env.py."],"baseline_retained":True};gate_path=self.output_dir / "f14_evidence_gate.json";gate_path.write_text(json.dumps(gate,indent=2),encoding="utf-8")
  for output in(path,gate_path):self.manifest.record_output_file(output.name,output)
  return self.finalize(success=True,paper_safe=True)
def main():
 p=argparse.ArgumentParser();p.add_argument("--output-dir",type=Path,default=SPEC / "evidence/f14_observation_evidence_bundle");p.add_argument("--bundle-dir",type=Path,default=SPEC / "evidence/f14_observation_evidence_bundle_portable");a=p.parse_args();r=F14Stage(a.output_dir,[sys.executable,*sys.argv]).run();b=ArtifactBundle(a.output_dir,a.bundle_dir);b.create();v=b.verify();print(json.dumps({"stage_success":r.success,"bundle_verification":v},indent=2));return 0 if r.success and v["valid"] else 1
if __name__=="__main__":raise SystemExit(main())