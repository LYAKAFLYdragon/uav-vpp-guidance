#!/usr/bin/env python
"""Create portable, deterministic, offline F16 state-machine evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

from uav_vpp_guidance.common.artifact_contract import ArtifactContract
from uav_vpp_guidance.evaluation.f16_mode_state_evidence import compare_candidates
from uav_vpp_guidance.pipeline.artifact_bundle import ArtifactBundle
from uav_vpp_guidance.pipeline.stage_runner import PipelineStage, StageResult

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / ".kiro/specs/control-guidance-remediation"


class F16ModeStateEvidenceStage(PipelineStage):
    """Portable F16 evidence stage; it never selects or mutates runtime behavior."""

    stage_name = "f16_mode_state_evidence"
    stage_version = "1.0.0"

    def __init__(self, output_dir: Path, command_line: list[str]):
        config = {"analysis_only": True, "input_guidance_config": "config/guidance.yaml", "config_overrides": []}
        super().__init__(config, str(output_dir), command_line=command_line, config_path="config/guidance.yaml")
        self.artifact_contract = ArtifactContract(required_files=[
            "run_manifest.json", "artifact_contract.json", "resolved_config.yaml",
            "f16_mode_state_evidence.json", "f16_evidence_gate.json",
        ])

    def run(self) -> StageResult:
        self.manifest.mark_started()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sources = {
            "baseline_freeze": SPEC / "baseline_freeze.json",
            "guidance_config": ROOT / "config/guidance.yaml",
            "state_machine_specification": SPEC / "f16_mode_state_machine.md",
            "runtime_gate_reference": ROOT / "src/uav_vpp_guidance/envs/tracking_env.py",
        }
        for name, path in sources.items():
            self.manifest.record_input_file(name, path)
        self.snapshot_config()
        guidance = yaml.safe_load((ROOT / "config/guidance.yaml").read_text(encoding="utf-8"))["guidance"]
        configured_gate = guidance.get("mode_switch", {})
        evidence = {
            "schema_version": "1.0.0",
            "finding_id": "F16",
            "analysis_scope": "pure deterministic candidate model; protected runtime/configuration paths unchanged",
            "frozen_runtime_observation": {
                "latch_policy": "sets latch on entry and clears only during reset",
                "missing_aspect_threshold_code_fallback_deg": 15.0,
                "audited_explicit_aspect_threshold_deg": configured_gate.get("aspect_threshold_deg"),
            },
            "candidate_comparison": compare_candidates(configured_gate),
            "backend_provenance": {"requested": None, "final": None, "fallback_occurred": False, "status": "not_applicable_offline_analytic_only"},
            "config_overrides": [],
        }
        evidence_path = self.output_dir / "f16_mode_state_evidence.json"
        evidence_path.write_text(json.dumps(evidence, indent=2, allow_nan=False), encoding="utf-8")
        gate = {
            "schema_version": "1.0.0", "finding_id": "F16", "candidate": "explicit_releaseable_mode_state_machine",
            "status": "needs_more_evidence", "runtime_disposition": "preserve_current_runtime_latch_pending_approved_gate",
            "decision": "no_runtime_authorization", "timestamp_utc": "2026-07-26T00:00:00+00:00",
            "protected_path_allowlist": [],
            "evidence_hashes": {
                "state_machine_specification_sha256": hashlib.sha256(sources["state_machine_specification"].read_bytes()).hexdigest(),
                "offline_state_machine_evidence_sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
            },
            "results": {
                "transition_table_tests_defined": True, "permanent_releaseable_and_state_machine_candidates_compared": True,
                "canonical_missing_aspect_threshold_deg": 25.0, "explicit_config_precedence": True,
                "transition_telemetry_specified": True, "frozen_matrix_evidence": "unavailable",
                "strict_jsbsim_evidence": "not_available", "protected_runtime_or_config_changed": False,
            },
            "blockers": [
                "The frozen scenario matrix is unexecuted and its required main-policy checkpoints are absent.",
                "No paired baseline/candidate scenario evidence establishes chatter, safety, tracking, or episode-behavior benefit.",
                "No strict-JSBSim evaluation verifies the candidate under the declared backend contract.",
                "Runtime migration requires an approved dedicated allowlist and compatibility review for existing permanent-latch tests.",
            ],
            "required_next_evidence": [
                "Paired frozen-matrix evaluations of permanent latch, releaseable latch, and state-machine candidates with identical seeds/artifacts.",
                "Per-episode transition/chatter/safety/tracking telemetry under strict JSBSim with no fallback.",
                "Review approving one state model and an explicit protected-path allowlist before runtime implementation.",
            ],
            "baseline_retained": True,
        }
        gate_path = self.output_dir / "f16_evidence_gate.json"
        gate_path.write_text(json.dumps(gate, indent=2), encoding="utf-8")
        for output in (evidence_path, gate_path):
            self.manifest.record_output_file(output.name, output)
        return self.finalize(success=True, paper_safe=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=SPEC / "evidence/f16_mode_state_bundle")
    parser.add_argument("--bundle-dir", type=Path, default=SPEC / "evidence/f16_mode_state_bundle_portable")
    args = parser.parse_args()
    result = F16ModeStateEvidenceStage(args.output_dir, [sys.executable, *sys.argv]).run()
    bundle = ArtifactBundle(args.output_dir, args.bundle_dir)
    bundle.create()
    verification = bundle.verify()
    print(json.dumps({"stage_success": result.success, "bundle_verification": verification}, indent=2))
    return 0 if result.success and verification["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
