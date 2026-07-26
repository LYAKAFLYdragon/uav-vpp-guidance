"""Phase-0 evidence freeze and protected-path preservation checks."""

import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_DIR = REPO_ROOT / ".kiro/specs/control-guidance-remediation"
BASELINE_PATH = SPEC_DIR / "baseline_freeze.json"
LEDGER_PATH = SPEC_DIR / "remediation_ledger.json"
ALLOWED_STATUSES = {
    "implemented_and_verified", "experiment_rejects_change", "needs_more_evidence",
    "by_design_no_change", "disproven_no_change",
}
PROTECTED_DIRECTORIES = (
    "src/uav_vpp_guidance/guidance", "src/uav_vpp_guidance/flight_control",
    "src/uav_vpp_guidance/virtual_point", "src/uav_vpp_guidance/gain_optimizer", "config",
)
PROTECTED_FILES = (
    "src/uav_vpp_guidance/envs/reward.py", "src/uav_vpp_guidance/envs/observation.py",
    "src/uav_vpp_guidance/envs/tracking_env.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _approved_protected_path_allowlist() -> set[str]:
    """Return protected files explicitly allowlisted by an approved gate."""
    allowlist: set[str] = set()
    for gate_path in (SPEC_DIR / "evidence").rglob("*_evidence_gate.json"):
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if gate.get("status") == "approved":
            allowlist.update(gate.get("protected_path_allowlist", []))
    return allowlist


def test_protected_paths_match_frozen_manifest_before_approval():
    """Only paths named by an approved gate may differ from the frozen bytes.

    Validates: Requirements 3.1, 3.2, 3.7, 3.8, 3.9
    """
    freeze = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    protected = freeze["protected_path_manifest"]
    assert protected["approved_implementation_gates"] == []
    manifest_path = REPO_ROOT / protected["path"]
    assert _sha256(manifest_path) == protected["sha256"]
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = {}
    for relative_directory in PROTECTED_DIRECTORIES:
        for path in sorted((REPO_ROOT / relative_directory).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                relative = path.relative_to(REPO_ROOT).as_posix()
                actual[relative] = _sha256(path)
    for relative in PROTECTED_FILES:
        actual[relative] = _sha256(REPO_ROOT / relative)
    actual = dict(sorted(actual.items()))
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    assert not missing, f"protected paths were removed: {missing}"
    assert not extra, f"protected paths were added: {extra}"
    allowlist = _approved_protected_path_allowlist()
    unexpected_changes = {
        path for path, expected_hash in expected.items()
        if actual[path] != expected_hash and path not in allowlist
    }
    assert not unexpected_changes, (
        "protected paths changed outside an approved gate allowlist: "
        f"{sorted(unexpected_changes)}"
    )
    assert allowlist == {
        "src/uav_vpp_guidance/guidance/los_rate_guidance.py",
        "src/uav_vpp_guidance/envs/simple_point_mass_env.py",
        "src/uav_vpp_guidance/virtual_point/generator.py",
        "src/uav_vpp_guidance/envs/tracking_env.py",
        "src/uav_vpp_guidance/agents/ppo_agent.py",
        "src/uav_vpp_guidance/training/train_prediction_vpp_ppo.py",
        "src/uav_vpp_guidance/hierarchy/specialist_policy.py",
    }


def test_remediation_ledger_is_complete_and_routes_no_change_items():
    """All findings have valid decisions and reserved no-change dispositions.

    Validates: Requirements 2.13, 3.7, 3.8
    """
    ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    entries = {entry["id"]: entry for entry in ledger["entries"]}
    assert list(entries) == [f"F{index:02d}" for index in range(1, 19)]
    assert all(entry["final_status"] in ALLOWED_STATUSES for entry in entries.values())
    assert all(entry["required_evidence"] and entry["decision_gate"] for entry in entries.values())
    assert entries["F04"]["final_status"] == "by_design_no_change"
    assert entries["F15"]["final_status"] == "disproven_no_change"
    assert entries["F09"]["final_status"] == "by_design_no_change"
    assert entries["F09"]["disposition"] == "documentation_only"
    assert not any(entries[item]["implementation_queue"] for item in ("F04", "F09", "F15"))


def test_baseline_freeze_records_required_provenance_and_unavailable_evidence():
    """The freeze is self-contained and does not imply unavailable run evidence."""
    freeze = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    assert len(freeze["source_revision"]["git_commit"]) == 40
    assert len(freeze["audit_validation"]["report_hashes"]) == 2
    assert len(freeze["config_hashes"]) >= 4
    assert freeze["protected_path_manifest"]["entry_count"] == 244
    assert freeze["baseline_scenario_matrix"]["formal_seeds"] == [0, 1, 2]
    assert freeze["baseline_scenario_matrix"]["status"] == "declared_not_executed_by_audit"
    backend = freeze["backend_provenance"]
    assert backend["status"] == "not_applicable_analytic_only"
    assert backend["audit_final_active_backend"] is None
    assert backend["strict_jsbsim_evidence_available"] is False
    assert freeze["unavailable_evidence"]
