#!/usr/bin/env python
"""Create portable offline F17 hysteresis/dwell and F10 arbitration evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from uav_vpp_guidance.common.artifact_contract import ArtifactContract
from uav_vpp_guidance.evaluation.f17_f10_evidence import run_experiment
from uav_vpp_guidance.pipeline.artifact_bundle import ArtifactBundle
from uav_vpp_guidance.pipeline.stage_runner import PipelineStage, StageResult

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / ".kiro/specs/control-guidance-remediation"


class F17F10EvidenceStage(PipelineStage):
    stage_name = "f17_f10_evidence"
    stage_version = "1.0.0"

    def __init__(self, output_dir: Path, command_line: list[str]):
        config = {"analysis_only": True, "input_guidance_config": "config/guidance.yaml", "input_experiment_matrix": "config/experiment/jsbsim_hrl_comparison.yaml", "config_overrides": []}
        super().__init__(config, str(output_dir), command_line=command_line, config_path="config/guidance.yaml")
        self.artifact_contract = ArtifactContract(required_files=["run_manifest.json", "artifact_contract.json", "resolved_config.yaml", "f17_f10_evidence.json", "f17_f10_evidence_gate.json"])

    def run(self) -> StageResult:
        self.manifest.mark_started()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sources = {
            "baseline_freeze": SPEC / "baseline_freeze.json",
            "guidance_config": ROOT / "config/guidance.yaml",
            "f17_f10_specification": SPEC / "f17_f10_experiment.md",
            "frozen_matrix": ROOT / "config/experiment/jsbsim_hrl_comparison.yaml",
            "hybrid_runtime_reference": ROOT / "src/uav_vpp_guidance/guidance/hybrid_guidance.py",
            "flight_control_runtime_reference": ROOT / "src/uav_vpp_guidance/flight_control/enhanced_low_level_controller.py",
        }
        for name, path in sources.items():
            self.manifest.record_input_file(name, path)
        self.snapshot_config()
        evidence = {"schema_version": "1.0.0", "analysis_scope": "offline deterministic F17/F10 models only; protected runtime and configuration paths unchanged", "results": run_experiment(), "backend_provenance": {"requested": None, "final": None, "fallback_occurred": False, "status": "not_applicable_offline_analytic_only"}, "config_overrides": []}
        evidence_path = self.output_dir / "f17_f10_evidence.json"
        evidence_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        gate = {
            "schema_version": "1.0.0", "finding_ids": ["F17", "F10"], "status": "needs_more_evidence",
            "runtime_disposition": "preserve_frozen_dwell_hysteresis_and_controller_defaults", "decision": "no_runtime_authorization",
            "timestamp_utc": "2026-07-26T00:00:00+00:00", "protected_path_allowlist": [],
            "evidence_hashes": {"experiment_specification_sha256": hashlib.sha256(sources["f17_f10_specification"].read_bytes()).hexdigest(), "offline_evidence_sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest()},
            "results": {"f17_delay_travel_chatter_and_missed_opportunities_measured_offline": True, "f10_qbar_altitude_aoa_and_command_source_sweep_measured_offline": True, "frozen_arbitration_order_documented": True, "f04_limits_preserved": True, "scheduled_candidate_named_not_selected": True, "frozen_matrix_evidence": "unavailable", "strict_jsbsim_evidence": "not_available", "protected_runtime_or_config_changed": False},
            "blockers": ["The declared frozen matrix has not executed and required main-method checkpoints are unavailable.", "Offline models cannot establish closed-loop task, stability, safety, or controller authority outcomes.", "No paired strict-JSBSim baseline/candidate evidence exists, and fallback is not admissible evidence."],
            "required_next_evidence": ["Run baseline and one candidate at a time on the frozen scenario/seed/checkpoint matrix.", "Record F17 per-episode switch timing, travelled range, chatter, missed opportunities, task and safety outcomes.", "Record F10 qbar/AoA/altitude/command-source telemetry, saturation attribution, safety and stability outcomes under strict JSBSim.", "Approve a candidate and dedicated protected allowlist before any default mutation."],
            "baseline_retained": True,
        }
        gate_path = self.output_dir / "f17_f10_evidence_gate.json"
        gate_path.write_text(json.dumps(gate, indent=2), encoding="utf-8")
        for output in (evidence_path, gate_path):
            self.manifest.record_output_file(output.name, output)
        return self.finalize(success=True, paper_safe=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=SPEC / "evidence/f17_f10_evidence_bundle")
    parser.add_argument("--bundle-dir", type=Path, default=SPEC / "evidence/f17_f10_evidence_bundle_portable")
    args = parser.parse_args()
    result = F17F10EvidenceStage(args.output_dir, [sys.executable, *sys.argv]).run()
    bundle = ArtifactBundle(args.output_dir, args.bundle_dir)
    bundle.create()
    verification = bundle.verify()
    print(json.dumps({"stage_success": result.success, "bundle_verification": verification}, indent=2))
    return 0 if result.success and verification["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
