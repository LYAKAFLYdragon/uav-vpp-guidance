#!/usr/bin/env python
"""Create portable offline F12/F13 reward evidence."""
import argparse, hashlib, json, sys
from pathlib import Path
from uav_vpp_guidance.common.artifact_contract import ArtifactContract
from uav_vpp_guidance.evaluation.f12_f13_reward_evidence import run_experiment
from uav_vpp_guidance.pipeline.artifact_bundle import ArtifactBundle
from uav_vpp_guidance.pipeline.stage_runner import PipelineStage
ROOT = Path(__file__).resolve().parents[1]; SPEC = ROOT / ".kiro/specs/control-guidance-remediation"
class F12F13Stage(PipelineStage):
    stage_name, stage_version = "f12_f13_reward_evidence", "1.0.0"
    def __init__(self, output_dir, command_line):
        super().__init__({"analysis_only": True, "input_matrix": "config/experiment/jsbsim_hrl_comparison.yaml", "config_overrides": []}, str(output_dir), command_line=command_line, config_path="config/experiment/jsbsim_hrl_comparison.yaml"); self.artifact_contract = ArtifactContract(required_files=["run_manifest.json", "artifact_contract.json", "resolved_config.yaml", "f12_f13_reward_evidence.json", "f12_f13_evidence_gate.json"])
    def run(self):
        self.manifest.mark_started(); self.output_dir.mkdir(parents=True, exist_ok=True)
        sources={"baseline_freeze": SPEC / "baseline_freeze.json", "reward_runtime_reference": ROOT / "src/uav_vpp_guidance/envs/reward.py", "matrix_config": ROOT / "config/experiment/jsbsim_hrl_comparison.yaml", "specification": SPEC / "f12_f13_reward_experiment.md"}
        for name,path in sources.items(): self.manifest.record_input_file(name,path)
        self.snapshot_config(); evidence={"schema_version":"1.0.0","finding_ids":["F12","F13"],"analysis_scope":"deterministic synthetic reward trace only; default reward and protected runtime unchanged","results":run_experiment(),"backend_provenance":{"requested":None,"final":None,"fallback_occurred":False,"status":"not_applicable_offline_analytic_only"},"config_overrides":[]}; path=self.output_dir / "f12_f13_reward_evidence.json"; path.write_text(json.dumps(evidence,indent=2),encoding="utf-8")
        gate={"schema_version":"1.0.0","finding_ids":["F12","F13"],"status":"needs_more_evidence","runtime_disposition":"preserve_default_reward_weights","decision":"no_runtime_authorization","timestamp_utc":"2026-07-26T00:00:00+00:00","protected_path_allowlist":[],"evidence_hashes":{"specification_sha256":hashlib.sha256(sources["specification"].read_bytes()).hexdigest(),"offline_evidence_sha256":hashlib.sha256(path.read_bytes()).hexdigest()},"results":{"step_and_terminal_decomposition_reported":True,"safety_and_saturation_occupancy_reported":True,"alive_and_closing_counterfactual_isolated":True,"policy_ablation_statistics":"not_available","strict_jsbsim_evidence":"not_available","protected_runtime_or_config_changed":False},"blockers":["No trained-policy frozen-matrix reward ablation exists.","Synthetic traces cannot establish exploit resistance, task, or safety outcomes.","No strict-JSBSim transfer evidence exists."],"required_next_evidence":["Run isolated default and one-weight-at-a-time ablations using identical training/evaluation protocols and seeds.","Report return composition, success/crash/safety, exploit indicators, and strict-JSBSim results.","Record every approved loaded-config mutation with record_config_override before a runtime/default change."],"baseline_retained":True}; gate_path=self.output_dir / "f12_f13_evidence_gate.json"; gate_path.write_text(json.dumps(gate,indent=2),encoding="utf-8")
        for output in (path,gate_path): self.manifest.record_output_file(output.name,output)
        return self.finalize(success=True,paper_safe=True)
def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--output-dir",type=Path,default=SPEC / "evidence/f12_f13_reward_evidence_bundle"); parser.add_argument("--bundle-dir",type=Path,default=SPEC / "evidence/f12_f13_reward_evidence_bundle_portable"); args=parser.parse_args(); result=F12F13Stage(args.output_dir,[sys.executable,*sys.argv]).run(); bundle=ArtifactBundle(args.output_dir,args.bundle_dir); bundle.create(); verified=bundle.verify(); print(json.dumps({"stage_success":result.success,"bundle_verification":verified},indent=2)); return 0 if result.success and verified["valid"] else 1
if __name__ == "__main__": raise SystemExit(main())
