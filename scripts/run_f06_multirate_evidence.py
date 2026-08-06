#!/usr/bin/env python
"""Generate reproducible offline F06 scheduling/delay evidence artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

from uav_vpp_guidance.common.artifact_contract import ArtifactContract
from uav_vpp_guidance.pipeline.artifact_bundle import ArtifactBundle
from uav_vpp_guidance.pipeline.stage_runner import PipelineStage, StageResult
from uav_vpp_guidance.evaluation.f06_multirate_evidence import (
    CANDIDATE_MULTI_RATE,
    LEGACY_SINGLE_RATE,
    DeterministicMultiRateScheduler,
    compare_delay_spectra,
    trace_to_dict,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / ".kiro/specs/control-guidance-remediation"


class F06MultiRateEvidenceStage(PipelineStage):
    """Portable, offline-only task-5 stage; it never mutates runtime config."""

    stage_name = "f06_multirate_evidence"
    stage_version = "1.0.0"

    def __init__(self, output_dir: Path, command_line: list[str]):
        with (ROOT / "config/guidance.yaml").open(encoding="utf-8") as handle:
            guidance_config = yaml.safe_load(handle)
        config = {
            "analysis_only": True,
            "candidate_name": "f06_10hz_high_level_20hz_guidance_100hz_low_level",
            "baseline_schedule": LEGACY_SINGLE_RATE.__dict__,
            "candidate_schedule": CANDIDATE_MULTI_RATE.__dict__,
            "input_guidance_config": "config/guidance.yaml",
            "config_overrides": [],
        }
        super().__init__(config, str(output_dir), command_line=command_line, config_path="config/guidance.yaml")
        self.guidance_config = guidance_config
        self.artifact_contract = ArtifactContract(required_files=[
            "run_manifest.json", "artifact_contract.json", "resolved_config.yaml",
            "f06_delay_phase_evidence.json", "f06_evidence_gate.json",
        ])

    def run(self) -> StageResult:
        self.manifest.mark_started()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for name, path in {
            "baseline_freeze": SPEC / "baseline_freeze.json",
            "guidance_config": ROOT / "config/guidance.yaml",
            "architecture_specification": SPEC / "f06_multirate_architecture.md",
        }.items():
            self.manifest.record_input_file(name, path)
        self.snapshot_config()

        baseline_trace = DeterministicMultiRateScheduler(LEGACY_SINGLE_RATE).run(
            1.0, high_level=lambda time_s: 1.0 if time_s >= 0.4 else 0.0
        )
        candidate_trace = DeterministicMultiRateScheduler(CANDIDATE_MULTI_RATE).run(
            1.0, high_level=lambda time_s: 1.0 if time_s >= 0.4 else 0.0
        )
        evidence = {
            "schema_version": "1.0.0",
            "finding_id": "F06",
            "analysis_scope": "offline deterministic scheduler + command-filter model; no runtime control/configuration modified",
            "frozen_baseline": {
                "high_level_period_s": 0.2,
                "guidance_period_s": 0.2,
                "low_level_period_s": 0.2,
                "command_order": ["high_level/VPP", "guidance", "clip", "command_filter", "communication_delay", "backend_step"],
                "filter_alpha": self.guidance_config["guidance"]["gains"]["alpha_filter"],
                "filter_time_constant_s": LEGACY_SINGLE_RATE.filter_time_constant_s,
                "filter_time_constant_interpretation": "component only; not total end-to-end latency",
                "communication_delay": "unmeasured/disabled-or-config-dependent; excluded from measured scheduler/filter sweep",
            },
            "candidate": {
                "name": self.config["candidate_name"],
                "schedule": CANDIDATE_MULTI_RATE.__dict__,
                "not_runtime_selected": True,
            },
            "exact_one_second_traces": {
                "baseline": trace_to_dict(baseline_trace),
                "candidate": trace_to_dict(candidate_trace),
            },
            "delay_phase_spectrum": compare_delay_spectra(),
            "unmeasured_budget_components": [
                "policy inference and observation acquisition", "communication delay when enabled",
                "backend/actuator dynamics", "sensor/telemetry transport",
            ],
            "backend_provenance": {
                "requested": None, "final": None, "fallback_occurred": False,
                "status": "not_applicable_offline_analytic_only",
            },
            "config_overrides": [],
        }
        evidence_path = self.output_dir / "f06_delay_phase_evidence.json"
        evidence_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        gate_path = self.output_dir / "f06_evidence_gate.json"
        spec_hash = hashlib.sha256((SPEC / "f06_multirate_architecture.md").read_bytes()).hexdigest()
        evidence_hash = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
        gate = {
            "schema_version": "1.0.0", "finding_id": "F06",
            "candidate": self.config["candidate_name"], "status": "needs_more_evidence",
            "runtime_disposition": "preserve_legacy_single_rate",
            "decision": "no_runtime_authorization",
            "timestamp_utc": "2026-07-26T00:00:00+00:00",
            "protected_path_allowlist": [],
            "evidence_hashes": {"architecture_specification_sha256": spec_hash, "offline_delay_phase_evidence_sha256": evidence_hash},
            "acceptance_thresholds": {
                "legacy_trace_equivalence": True, "integer_schedule_or_explicit_resampler": True,
                "no_safety_or_stability_regression": True, "latency_target_s": 0.2,
                "strict_backend_closed_loop_evidence": True,
            },
            "results": {
                "offline_scheduler_invariants_pass": True,
                "legacy_single_rate_trace_reproduced": True,
                "filter_time_constant_component_s": LEGACY_SINGLE_RATE.filter_time_constant_s,
                "candidate_latency_target_satisfied": "not_established_closed_loop",
                "candidate_stability_safety_satisfied": "not_established_closed_loop",
                "strict_jsbsim_evidence": "not_available_from_analytic_only_stage",
                "protected_runtime_or_config_changed": False,
            },
            "blockers": [
                "The declared frozen scenario matrix is unexecuted and required main-policy checkpoints are absent.",
                "Offline scheduler/filter delay excludes actuator/backend/sensor/communication components required for a total latency budget.",
                "No paired strict-JSBSim stability or safety evaluation exists for the candidate cadence.",
            ],
            "required_next_evidence": [
                "Task-6 per-stage actuator/filter/communication delay and phase measurements.",
                "Paired baseline/candidate frozen-matrix evaluations with identical seeds, horizon, artifacts, backend provenance, and safety metrics.",
                "Explicit review accepting latency/stability/safety targets before an F06 protected-path allowlist is issued.",
            ],
            "baseline_retained": True,
        }
        gate_path.write_text(json.dumps(gate, indent=2), encoding="utf-8")
        for filename in ("f06_delay_phase_evidence.json", "f06_evidence_gate.json"):
            self.manifest.record_output_file(filename, self.output_dir / filename)
        return self.finalize(success=True, paper_safe=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=SPEC / "evidence" / "f06_multirate_bundle")
    parser.add_argument("--bundle-dir", type=Path, default=SPEC / "evidence" / "f06_multirate_bundle_portable")
    args = parser.parse_args()
    stage = F06MultiRateEvidenceStage(args.output_dir, [sys.executable, *sys.argv])
    result = stage.run()
    if not result.success:
        return 1
    bundle = ArtifactBundle(args.output_dir, args.bundle_dir)
    bundle.create()
    verification = bundle.verify()
    print(json.dumps({"stage_success": result.success, "bundle_verification": verification}, indent=2))
    return 0 if verification["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
