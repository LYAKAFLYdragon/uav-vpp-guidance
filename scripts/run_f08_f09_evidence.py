#!/usr/bin/env python
"""Create portable F08 CEM credibility and F09 terminology evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from uav_vpp_guidance.common.artifact_contract import ArtifactContract
from uav_vpp_guidance.evaluation.f08_f09_cem_evidence import compare_fixed_budget, f09_metric_definition
from uav_vpp_guidance.pipeline.artifact_bundle import ArtifactBundle
from uav_vpp_guidance.pipeline.stage_runner import PipelineStage, StageResult

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / ".kiro/specs/control-guidance-remediation"


class F08F09EvidenceStage(PipelineStage):
    stage_name = "f08_f09_evidence"
    stage_version = "1.0.0"

    def __init__(self, output_dir: Path, command_line: list[str]):
        config = {"analysis_only": True, "input_gain_only_config": "config/experiment/gain_only_cem.yaml", "config_overrides": []}
        super().__init__(config, str(output_dir), command_line=command_line, config_path="config/experiment/gain_only_cem.yaml")
        self.artifact_contract = ArtifactContract(required_files=["run_manifest.json", "artifact_contract.json", "resolved_config.yaml", "f08_f09_evidence.json", "f08_f09_evidence_gate.json"])

    def run(self) -> StageResult:
        self.manifest.mark_started(); self.output_dir.mkdir(parents=True, exist_ok=True)
        sources = {"baseline_freeze": SPEC / "baseline_freeze.json", "gain_only_cem_config": ROOT / "config/experiment/gain_only_cem.yaml", "cem_runtime_reference": ROOT / "src/uav_vpp_guidance/gain_optimizer/cem.py", "regret_runtime_reference": ROOT / "src/uav_vpp_guidance/gain_optimizer/regret.py", "specification": SPEC / "f08_f09_experiment.md", "dry_run_result": ROOT / "outputs/gain_only_cem/cem_results.json"}
        for name, path in sources.items(): self.manifest.record_input_file(name, path)
        self.snapshot_config()
        evidence = {"schema_version": "1.0.0", "finding_ids": ["F08", "F09"], "analysis_scope": "deterministic surrogate CEM comparison and metric-label audit only; no optimizer, rollout, checkpoint, or training behavior changed", "f08_fixed_budget_comparison": compare_fixed_budget(), "f09_metric_definition": f09_metric_definition(), "existing_cem_result_status": "dry_run_n_iterations_0_not_performance_evidence", "backend_provenance": {"requested": None, "final": None, "fallback_occurred": False, "status": "not_applicable_offline_analytic_only"}, "config_overrides": []}
        evidence_path = self.output_dir / "f08_f09_evidence.json"; evidence_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        gate = {"schema_version": "1.0.0", "finding_ids": ["F08", "F09"], "status": "needs_more_evidence", "runtime_disposition": "preserve_cem_and_bilevel_training_behavior", "decision": "no_runtime_authorization", "timestamp_utc": "2026-07-26T00:00:00+00:00", "protected_path_allowlist": [], "evidence_hashes": {"specification_sha256": hashlib.sha256(sources["specification"].read_bytes()).hexdigest(), "offline_evidence_sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest()}, "results": {"seed_controlled_fixed_budget_surrogate_comparison": True, "population_elite_covariance_early_stop_variants_enumerated": True, "f09_truthful_label_documented": True, "f09_training_behavior_changed": False, "actual_policy_or_jsbsim_evidence": "not_available", "protected_runtime_or_config_changed": False}, "blockers": ["Existing CEM results are zero-iteration dry runs.", "Frozen policy checkpoints and matrix evaluations are unavailable.", "Surrogate coverage/convergence cannot establish task, safety, compute, or strict-JSBSim benefit."], "required_next_evidence": ["Evaluate baseline and one CEM variant at a time with identical rollout budget, seeds, scenarios, policies, backend provenance, and wall-clock measurement.", "Report best/median/worst, coverage, convergence, safety and strict-JSBSim out-of-sample results.", "Use a separately approved spec before changing bilevel updates, snapshots, or rollback behavior."], "baseline_retained": True}
        gate_path = self.output_dir / "f08_f09_evidence_gate.json"; gate_path.write_text(json.dumps(gate, indent=2), encoding="utf-8")
        for output in (evidence_path, gate_path): self.manifest.record_output_file(output.name, output)
        return self.finalize(success=True, paper_safe=True)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path, default=SPEC / "evidence/f08_f09_evidence_bundle"); parser.add_argument("--bundle-dir", type=Path, default=SPEC / "evidence/f08_f09_evidence_bundle_portable"); args = parser.parse_args()
    result = F08F09EvidenceStage(args.output_dir, [sys.executable, *sys.argv]).run(); bundle = ArtifactBundle(args.output_dir, args.bundle_dir); bundle.create(); verification = bundle.verify(); print(json.dumps({"stage_success": result.success, "bundle_verification": verification}, indent=2)); return 0 if result.success and verification["valid"] else 1


if __name__ == "__main__": raise SystemExit(main())
