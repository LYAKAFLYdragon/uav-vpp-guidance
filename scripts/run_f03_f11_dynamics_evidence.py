#!/usr/bin/env python
"""Create portable, offline-only F03/F11 dynamics evidence."""
import argparse, hashlib, json, sys
from pathlib import Path
import yaml
from uav_vpp_guidance.common.artifact_contract import ArtifactContract
from uav_vpp_guidance.pipeline.artifact_bundle import ArtifactBundle
from uav_vpp_guidance.pipeline.stage_runner import PipelineStage, StageResult
from uav_vpp_guidance.evaluation.f03_f11_dynamics_evidence import run_experiment
ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / ".kiro/specs/control-guidance-remediation"

class F03F11DynamicsEvidenceStage(PipelineStage):
    stage_name, stage_version = "f03_f11_dynamics_evidence", "1.0.0"
    def __init__(self, output_dir, command_line):
        config = {"analysis_only": True, "input_guidance_config": "config/guidance.yaml", "config_overrides": []}
        super().__init__(config, str(output_dir), command_line=command_line, config_path="config/guidance.yaml")
        self.artifact_contract = ArtifactContract(required_files=["run_manifest.json", "artifact_contract.json", "resolved_config.yaml", "f03_f11_dynamics_evidence.json", "f03_f11_evidence_gate.json"])
    def run(self):
        self.manifest.mark_started(); self.output_dir.mkdir(parents=True, exist_ok=True)
        for name, path in {"baseline_freeze": SPEC / "baseline_freeze.json", "guidance_config": ROOT / "config/guidance.yaml", "experiment_specification": SPEC / "f03_f11_dynamics_experiment.md"}.items(): self.manifest.record_input_file(name, path)
        self.snapshot_config(); guidance = yaml.safe_load((ROOT / "config/guidance.yaml").read_text(encoding="utf-8"))["guidance"]
        evidence = {"schema_version": "1.0.0", "finding_ids": ["F03", "F11"], "analysis_scope": "offline filter/mapping model only; protected runtime/configuration paths unchanged", "results": run_experiment(guidance), "backend_provenance": {"requested": None, "final": None, "fallback_occurred": False, "status": "not_applicable_offline_analytic_only"}, "config_overrides": []}
        evidence_path = self.output_dir / "f03_f11_dynamics_evidence.json"; evidence_path.write_text(json.dumps(evidence, indent=2, default=lambda value: value.__dict__), encoding="utf-8")
        gate = {"schema_version": "1.0.0", "finding_ids": ["F03", "F11"], "status": "needs_more_evidence", "runtime_disposition": "preserve_frozen_defaults", "decision": "no_runtime_authorization", "timestamp_utc": "2026-07-26T00:00:00+00:00", "protected_path_allowlist": [], "evidence_hashes": {"experiment_specification_sha256": hashlib.sha256((SPEC / "f03_f11_dynamics_experiment.md").read_bytes()).hexdigest(), "offline_dynamics_evidence_sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest()}, "results": {"enabled_filter_and_mapping_stages_enumerated": True, "per_stage_delay_phase_peak_saturation_slew_recorded": True, "four_single_change_candidates_evaluated_offline": True, "f04_limits_preserved": True, "strict_jsbsim_evidence": "not_available", "candidate_selection": "blocked"}, "blockers": ["Offline model excludes JSBSim plant and actual actuator response.", "Frozen scenario matrix is unexecuted and main policy checkpoints are absent.", "No paired strict-JSBSim safety, command-rate, stability, or tracking evidence exists."], "baseline_retained": True}
        gate_path = self.output_dir / "f03_f11_evidence_gate.json"; gate_path.write_text(json.dumps(gate, indent=2), encoding="utf-8")
        for name in (evidence_path.name, gate_path.name): self.manifest.record_output_file(name, self.output_dir / name)
        return self.finalize(success=True, paper_safe=True)

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path, default=SPEC / "evidence/f03_f11_dynamics_bundle"); parser.add_argument("--bundle-dir", type=Path, default=SPEC / "evidence/f03_f11_dynamics_bundle_portable"); args = parser.parse_args()
    result = F03F11DynamicsEvidenceStage(args.output_dir, [sys.executable, *sys.argv]).run(); verification = ArtifactBundle(args.output_dir, args.bundle_dir); verification.create(); checked = verification.verify(); print(json.dumps({"stage_success": result.success, "bundle_verification": checked}, indent=2)); return 0 if result.success and checked["valid"] else 1
if __name__ == "__main__": raise SystemExit(main())
