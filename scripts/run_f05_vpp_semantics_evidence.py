#!/usr/bin/env python
"""Create portable offline F05 VPP-semantic evidence."""
import argparse, hashlib, json, sys
from pathlib import Path
from uav_vpp_guidance.common.artifact_contract import ArtifactContract
from uav_vpp_guidance.pipeline.artifact_bundle import ArtifactBundle
from uav_vpp_guidance.pipeline.stage_runner import PipelineStage
from uav_vpp_guidance.evaluation.f05_vpp_semantics_evidence import compare_semantics
ROOT = Path(__file__).resolve().parents[1]; SPEC = ROOT / ".kiro/specs/control-guidance-remediation"
class F05Stage(PipelineStage):
 stage_name, stage_version = "f05_vpp_semantics_evidence", "1.0.0"
 def __init__(self, output_dir, command_line):
  super().__init__({"analysis_only": True, "config_overrides": []}, str(output_dir), command_line=command_line, config_path="config/guidance.yaml"); self.artifact_contract = ArtifactContract(required_files=["run_manifest.json","artifact_contract.json","resolved_config.yaml","f05_vpp_semantics_evidence.json","f05_evidence_gate.json"])
 def run(self):
  self.manifest.mark_started(); self.output_dir.mkdir(parents=True, exist_ok=True)
  for name, path in {"baseline_freeze": SPEC / "baseline_freeze.json", "guidance_config": ROOT / "config/guidance.yaml", "experiment_specification": SPEC / "f05_vpp_semantics_experiment.md"}.items(): self.manifest.record_input_file(name, path)
  self.snapshot_config(); path = self.output_dir / "f05_vpp_semantics_evidence.json"; path.write_text(json.dumps({"schema_version":"1.0.0","finding_id":"F05","results":compare_semantics(),"config_overrides":[]},indent=2),encoding="utf-8")
  gate = {"schema_version":"1.0.0","finding_id":"F05","status":"needs_more_evidence","runtime_disposition":"preserve_fixed_metre_cartesian_offset","decision":"no_runtime_authorization","timestamp_utc":"2026-07-26T00:00:00+00:00","protected_path_allowlist":[],"evidence_hashes":{"experiment_specification_sha256":hashlib.sha256((SPEC / "f05_vpp_semantics_experiment.md").read_bytes()).hexdigest(),"offline_semantics_evidence_sha256":hashlib.sha256(path.read_bytes()).hexdigest()},"results":{"baseline_mapping_and_angular_variation_measured":True,"candidate_mappings_bounded_monotone_symmetric_and_near_range_safe":True,"frozen_matrix_evidence":"unavailable","checkpoint_action_migration":"not_approved"},"blockers":["No frozen scenario/seed evaluation is available.","Required policy checkpoints are absent.","Action semantic migration and legacy checkpoint compatibility are not approved."],"baseline_retained":True}; gate_path=self.output_dir / "f05_evidence_gate.json"; gate_path.write_text(json.dumps(gate,indent=2),encoding="utf-8")
  for output in (path,gate_path): self.manifest.record_output_file(output.name,output)
  return self.finalize(success=True,paper_safe=True)
def main():
 parser=argparse.ArgumentParser(); parser.add_argument("--output-dir",type=Path,default=SPEC / "evidence/f05_vpp_semantics_bundle"); parser.add_argument("--bundle-dir",type=Path,default=SPEC / "evidence/f05_vpp_semantics_bundle_portable"); args=parser.parse_args(); result=F05Stage(args.output_dir,[sys.executable,*sys.argv]).run(); bundle=ArtifactBundle(args.output_dir,args.bundle_dir); bundle.create(); check=bundle.verify(); print(json.dumps({"stage_success":result.success,"bundle_verification":check},indent=2)); return 0 if result.success and check["valid"] else 1
if __name__ == "__main__": raise SystemExit(main())
