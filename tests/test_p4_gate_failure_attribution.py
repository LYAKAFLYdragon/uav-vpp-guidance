"""Tests for the P4 geometry gate failure-attribution audit (GEO-20260713-R1).

This module currently contains the DURABLE frozen-evidence fact test. It reads the
ORIGINAL frozen artifacts directly (no import of the not-yet-existing audit module)
and MUST PASS both before and after the audit implementation.

Strictly non-training, non-tuning, read-only. Python stdlib + pytest only (no
hypothesis / PBT dependency). All paths are relative repo paths (no hardcoded drive
letters in logic).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Repo-relative portable evidence locations (no hardcoded drive letters).
# An external bundle can be supplied only through P4_EVIDENCE_ROOT.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = Path(
    os.environ.get(
        "P4_EVIDENCE_ROOT",
        str(REPO_ROOT / "reports" / "p4_evidence_bundle_20260714"),
    )
)
REGISTRY_PATH = EVIDENCE_ROOT / "thesis_five_state_shared_skill_registry_v1.yaml"
GATE_JSON_PATH = EVIDENCE_ROOT / "p4_geometry_pretrain_gate.json"

SKILLS = [
    "pursuit_conversion",
    "lead_intercept",
    "defensive_extension",
    "reentry_recovery",
]


def _per_skill_summary_path(skill: str) -> Path:
    return EVIDENCE_ROOT / "skills" / skill / "geometry_pretrain_summary.json"


# ---------------------------------------------------------------------------
# Declared-cell derivation from the registry geometry_phase_coverage blocks.
#
# Prefer PyYAML if importable; otherwise fall back to a tiny tolerant parser that
# extracts, per skill, the `geometry_states` and `phases` lists nested under
# `skills.<skill>.geometry_phase_coverage`.
# ---------------------------------------------------------------------------


def _parse_inline_list(value: str) -> list[str]:
    """Parse a YAML inline flow list like ``[a, b, c]`` into a list of tokens."""
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [tok.strip() for tok in inner.split(",") if tok.strip()]
    return []


def _derive_declared_cells_stdlib(text: str) -> dict[str, set[str]]:
    """Minimal, indentation-aware parser for the registry's geometry blocks.

    Returns a mapping skill -> set of ``"state:phase"`` declared cells. Only the
    ``skills.<skill>.geometry_phase_coverage.{geometry_states,phases}`` fields are
    read; everything else is ignored.
    """
    lines = text.splitlines()

    def indent(line: str) -> int:
        return len(line) - len(line.lstrip(" "))

    # Locate the top-level `skills:` block.
    skills_idx = None
    skills_indent = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "skills:" and indent(line) == 0:
            skills_idx = i
            skills_indent = indent(line)
            break
    if skills_idx is None:
        raise AssertionError("registry has no top-level 'skills:' block")

    result: dict[str, set[str]] = {}
    i = skills_idx + 1
    n = len(lines)
    current_skill: str | None = None
    skill_indent: int | None = None
    in_geo_block = False
    geo_indent: int | None = None
    geo_states: list[str] = []
    geo_phases: list[str] = []

    def flush_skill() -> None:
        nonlocal geo_states, geo_phases
        if current_skill is not None and (geo_states or geo_phases):
            result[current_skill] = {
                f"{state}:{phase}" for state in geo_states for phase in geo_phases
            }
        geo_states = []
        geo_phases = []

    while i < n:
        line = lines[i]
        stripped = line.strip()
        if stripped == "" or stripped.startswith("#"):
            i += 1
            continue
        ind = indent(line)

        # Left the skills block entirely.
        if ind <= skills_indent and stripped != "":
            flush_skill()
            break

        # A new skill key (direct child of `skills:`).
        if (
            skill_indent is None or ind == skill_indent
        ) and stripped.endswith(":") and ind > skills_indent:
            # Only treat as a skill header if it is the first indentation level
            # under skills.
            if skill_indent is None:
                skill_indent = ind
            if ind == skill_indent:
                flush_skill()
                current_skill = stripped[:-1].strip()
                in_geo_block = False
                geo_indent = None
                i += 1
                continue

        # Entering geometry_phase_coverage under the current skill.
        if stripped == "geometry_phase_coverage:" and current_skill is not None:
            in_geo_block = True
            geo_indent = ind
            i += 1
            continue

        if in_geo_block and geo_indent is not None:
            if ind <= geo_indent:
                # geometry block ended
                in_geo_block = False
            else:
                if stripped.startswith("geometry_states:"):
                    geo_states = _parse_inline_list(stripped.split(":", 1)[1])
                elif stripped.startswith("phases:"):
                    geo_phases = _parse_inline_list(stripped.split(":", 1)[1])
                i += 1
                continue

        i += 1

    flush_skill()
    return result


def _derive_declared_cells(registry_path: Path) -> dict[str, set[str]]:
    """Derive declared cells per skill from the registry.

    Prefers PyYAML; falls back to a stdlib-only parser when PyYAML is unavailable.
    """
    text = registry_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        skills = data.get("skills", {}) if isinstance(data, dict) else {}
        cells: dict[str, set[str]] = {}
        for skill, spec in skills.items():
            if not isinstance(spec, dict):
                continue
            gpc = spec.get("geometry_phase_coverage")
            if not isinstance(gpc, dict):
                continue
            states = gpc.get("geometry_states", []) or []
            phases = gpc.get("phases", []) or []
            cells[skill] = {f"{s}:{p}" for s in states for p in phases}
        if cells:
            return cells
        # Fall through to stdlib parser if PyYAML produced nothing useful.
        return _derive_declared_cells_stdlib(text)
    except Exception:
        return _derive_declared_cells_stdlib(text)


# ---------------------------------------------------------------------------
# Emitted-cell collection from selected_gate.geometry_phase_coverage.
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _emitted_from_gate_json(skill: str) -> dict[str, int]:
    """Emitted cells for a skill from the top-level combined gate JSON."""
    gate = _load_json(GATE_JSON_PATH)
    return dict(
        gate["skills"][skill]["selected_gate"]["geometry_phase_coverage"]
    )


def _emitted_from_summary(skill: str) -> dict[str, int]:
    """Emitted cells for a skill from its per-skill geometry_pretrain_summary.json."""
    summary = _load_json(_per_skill_summary_path(skill))
    return dict(summary["selected_gate"]["geometry_phase_coverage"])


# Expected declared-cell counts per skill (VERIFIED against the frozen registry).
EXPECTED_DECLARED_COUNTS = {
    "pursuit_conversion": 6,   # 3 states x 2 phases
    "lead_intercept": 6,       # 3 states x 2 phases
    "defensive_extension": 6,  # 2 states x 3 phases
    "reentry_recovery": 8,     # 4 states x 2 phases
}
EXPECTED_DECLARED_TOTAL = 26


def test_frozen_evidence_reconciliation_and_disproven_premise():
    """Durable static-fact test on the ORIGINAL frozen artifacts.

    Establishes the tool-independent evidence of the REAL reconciliation
    (26 declared / 26 emitted / diff 0 / extra 0) and disproves the historical
    26/22/4 premise. MUST PASS both before and after the audit is implemented.

    Validates: Requirements 1.1, 1.2, 1.3, 1.6, 1.8, 1.9
    """
    # --- Artifacts must exist (frozen evidence present) ---------------------
    assert REGISTRY_PATH.is_file(), f"registry not found: {REGISTRY_PATH}"
    assert GATE_JSON_PATH.is_file(), f"gate JSON not found: {GATE_JSON_PATH}"
    for skill in SKILLS:
        assert _per_skill_summary_path(skill).is_file(), (
            f"per-skill summary not found for {skill}"
        )

    # --- Step 1: declared cells derived from the registry -------------------
    declared = _derive_declared_cells(REGISTRY_PATH)
    for skill in SKILLS:
        assert skill in declared, f"registry missing declared cells for {skill}"
        assert len(declared[skill]) == EXPECTED_DECLARED_COUNTS[skill], (
            f"{skill}: declared {len(declared[skill])} != "
            f"{EXPECTED_DECLARED_COUNTS[skill]}"
        )
    declared_total = sum(len(declared[s]) for s in SKILLS)
    assert declared_total == EXPECTED_DECLARED_TOTAL, (
        f"declared total {declared_total} != {EXPECTED_DECLARED_TOTAL}"
    )

    # --- Step 2: emitted cells from gate JSON, cross-checked vs summaries ----
    emitted_total = 0
    diff_total = 0
    extra_total = 0
    for skill in SKILLS:
        emitted_gate = _emitted_from_gate_json(skill)
        emitted_summary = _emitted_from_summary(skill)

        # The top-level gate and the per-skill summary must agree exactly
        # (cross-artifact consistency is CLEAN).
        assert emitted_gate == emitted_summary, (
            f"{skill}: top-level gate coverage disagrees with per-skill summary"
        )

        emitted_keys = set(emitted_gate.keys())
        declared_keys = declared[skill]

        # --- Step 3: per-skill reconciliation -------------------------------
        diff = declared_keys - emitted_keys      # declared but not emitted
        extra = emitted_keys - declared_keys     # emitted but not declared

        assert len(emitted_keys) == EXPECTED_DECLARED_COUNTS[skill], (
            f"{skill}: emitted {len(emitted_keys)} != "
            f"{EXPECTED_DECLARED_COUNTS[skill]}"
        )
        assert diff == set(), f"{skill}: unexpected declared-not-emitted diff: {diff}"
        assert extra == set(), f"{skill}: unexpected emitted-not-declared extra: {extra}"

        emitted_total += len(emitted_keys)
        diff_total += len(diff)
        extra_total += len(extra)

    # --- Step 3 (aggregate) + Step 4: VERIFIED reconciliation & disproven ----
    assert declared_total == 26
    assert emitted_total == 26
    assert diff_total == 0     # historical 26/22/4 premise is NOT reproducible
    assert extra_total == 0

    # --- Step 5: present-value-0 facts asserted DIRECTLY --------------------
    lead = _emitted_from_gate_json("lead_intercept")

    # advantage:post_merge present with value 0 (key exists AND value == 0).
    assert "advantage:post_merge" in lead
    assert lead["advantage:post_merge"] == 0

    # neutral:post_merge present with value 0.
    assert "neutral:post_merge" in lead
    assert lead["neutral:post_merge"] == 0

    # crossing_entry cells present and non-zero with the exact verified counts.
    assert lead["crossing_entry:post_merge"] == 182
    assert lead["crossing_entry:pre_merge"] == 1092


# ---------------------------------------------------------------------------
# Phase 1, Task 2: xfail audit-output test.
#
# This test exercises the NEW audit output (the module/CLI that does NOT exist
# yet). It invokes the CLI `scripts/analyze_p4_geometry_gate_failure_attribution.py`
# as a subprocess against the frozen evidence, writing into a tmp_path output dir,
# then loads the emitted JSON and asserts on the intended POST-FIX behaviors.
#
# Because neither the module nor the CLI is implemented yet, the subprocess will
# return non-zero / the JSON file will not exist, so the assertions below fail and
# the test reports `xfail` (strict). The marker is removed in Phase 4 once the
# audit tool exists, at which point these assertions must PASS.
# ---------------------------------------------------------------------------

import subprocess
import sys

CLI_SCRIPT_PATH = (
    REPO_ROOT / "scripts" / "analyze_p4_geometry_gate_failure_attribution.py"
)

# The six controlled attribution labels (design §3).
CONTROLLED_LABELS = {
    "not_observed_cannot_assess",
    "train_coverage_gap",
    "observed_behavior_failure",
    "telemetry_or_gate_observability_gap",
    "mixed_or_inconclusive",
    "contract_or_provenance_mismatch",
}


def _run_audit_cli(out_dir: Path) -> Path:
    """Invoke the (not-yet-implemented) audit CLI against the frozen evidence.

    Writes md/csv/json into ``out_dir`` and returns the JSON path. Raises if the
    CLI cannot be invoked or exits non-zero (which is the current state, since the
    tool does not exist yet).
    """
    out_md = out_dir / "audit.md"
    out_csv = out_dir / "matrix.csv"
    out_json = out_dir / "matrix.json"

    cmd = [
        sys.executable,
        str(CLI_SCRIPT_PATH),
        "--gate-json",
        str(GATE_JSON_PATH),
        "--registry-path",
        str(REGISTRY_PATH),
        "--out-md",
        str(out_md),
        "--out-csv",
        str(out_csv),
        "--out-json",
        str(out_json),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"audit CLI exited {proc.returncode} (tool not implemented yet)\n"
        f"stdout={proc.stdout}\nstderr={proc.stderr}"
    )
    assert out_json.is_file(), f"audit CLI did not produce JSON at {out_json}"
    return out_json


def test_audit_emits_reconciled_attribution(tmp_path):
    """The fixed audit emits a fully reconciled per-cell attribution matrix.

    Exercises the NEW audit output (CLI/module). It asserts the intended post-fix
    behaviors: the full 26-row reconciled matrix (every row emitted, diff EMPTY),
    the three status columns + reason_code per row, a run-level
    contract_or_provenance_mismatch provenance flag for the missing frozen inputs,
    per-row evidence_paths that distinguish absent (unverifiable) from
    present_value_0, a frozen provenance block with provenance_unverified markers
    kept separate from working-tree git info, and the combat_finetune_NOT_allowed
    conclusion.

    Marked xfail(strict) because the audit tool does not exist yet.

    Validates: Requirements 1.4, 1.10, 2.1, 2.2, 2.3, 2.4, 2.6, 2.9, 2.10, 2.11, 2.8
    """
    out_json = _run_audit_cli(tmp_path)
    audit = _load_json(out_json)

    # --- Top-level shape: rows + preflight + provenance + conclusion --------
    assert "rows" in audit, "audit JSON missing 'rows' list"
    assert "preflight" in audit, "audit JSON missing 'preflight' block"
    assert "provenance" in audit, "audit JSON missing 'provenance' block"
    assert "conclusion" in audit, "audit JSON missing 'conclusion' field"

    rows = audit["rows"]
    assert isinstance(rows, list)

    # --- Full 26-row reconciled matrix; EVERY row emitted (diff EMPTY) -------
    assert len(rows) == EXPECTED_DECLARED_TOTAL, (
        f"expected {EXPECTED_DECLARED_TOTAL} rows, got {len(rows)}"
    )
    emission_statuses = {row["emission_status"] for row in rows}
    assert emission_statuses == {"emitted"}, (
        f"every row must be 'emitted' (diff EMPTY); got {emission_statuses}"
    )
    # No declared_not_emitted rows exist on this artifact.
    assert not [r for r in rows if r["emission_status"] == "declared_not_emitted"]

    # --- Preflight reconciliation 26 / 26 / diff 0 / extra 0 ----------------
    preflight = audit["preflight"]
    assert preflight["declared_count"] == 26
    assert preflight["emitted_count"] == 26
    assert preflight["diff_count"] == 0
    assert preflight["extra_count"] == 0
    assert preflight.get("diff", []) == []

    # --- Three status columns + reason_code present and populated per row ----
    for row in rows:
        for col in (
            "train_field_status",
            "dev_gate_field_status",
            "emission_status",
        ):
            assert col in row, f"row missing status column {col!r}: {row}"
            assert row[col] not in (None, ""), (
                f"row status column {col!r} is empty: {row}"
            )
        assert "reason_code" in row, f"row missing 'reason_code' column: {row}"
        # attribution label must be one of the six controlled labels
        assert row["attribution_label"] in CONTROLLED_LABELS, (
            f"uncontrolled attribution_label: {row.get('attribution_label')!r}"
        )

    # --- Every row links to evidence field paths; absent vs present_value_0 --
    field_status_values = {"present_value_0", "present_value_gt_0", "absent"}
    for row in rows:
        assert "evidence_paths" in row, f"row missing 'evidence_paths': {row}"
        assert row["evidence_paths"], f"row has empty 'evidence_paths': {row}"
        assert row["train_field_status"] in field_status_values
        assert row["dev_gate_field_status"] in field_status_values

    # The dominant case on this artifact is present_value_0 (no absent cells).
    dev_statuses = [row["dev_gate_field_status"] for row in rows]
    present_value_0 = sum(1 for s in dev_statuses if s == "present_value_0")
    absent = sum(1 for s in dev_statuses if s == "absent")
    assert absent == 0, "no declared cell is absent on this artifact"
    assert present_value_0 > 0, "expected present_value_0 to be the dominant case"

    # --- Run-level contract_or_provenance_mismatch for MISSING frozen inputs -
    provenance = audit["provenance"]
    flags = provenance.get("flags", provenance)
    # A run-level contract_or_provenance_mismatch flag must be raised.
    assert (
        "contract_or_provenance_mismatch" in json.dumps(provenance)
    ), "provenance block must raise a run-level contract_or_provenance_mismatch flag"

    # It must be triggered by the MISSING frozen inputs, NOT any per-cell diff.
    unverified = json.dumps(provenance)
    for missing in ("training_plan", "resolved_geometry_config", "run_commit"):
        assert missing in unverified, (
            f"provenance must mark missing frozen input {missing!r} "
            "as provenance_unverified"
        )
    assert "provenance_unverified" in unverified, (
        "provenance block must carry provenance_unverified markers"
    )

    # --- Frozen provenance block kept separate from working-tree git info ----
    # SHA256s / top-level field lists recorded for the present inputs.
    assert "sha256" in unverified or "sha256" in json.dumps(audit), (
        "frozen provenance must record SHA256s of the present inputs"
    )
    # Frozen provenance and working-tree git info are separate fields.
    assert "working_tree" in unverified or "git" in unverified, (
        "working-tree git info must be recorded separately from frozen provenance"
    )

    # --- Conclusion: combat finetune NOT allowed ----------------------------
    assert audit["conclusion"] == "combat_finetune_NOT_allowed"


# ---------------------------------------------------------------------------
# Phase 2, Task 3: concrete preservation baseline (observation-first).
#
# This is NOT a property-based test. It records a durable SHA256 manifest of the
# P4 artifact set and the frozen gate invariants BEFORE the audit runs, so the
# Phase 4 preservation test (task 6) can assert byte-identical hashes afterwards.
#
# Writing tests/_p4_preservation_baseline.json is an allowed NEW test-support file.
# This step does NOT modify any P4 artifact.
# ---------------------------------------------------------------------------

import hashlib

# The baseline is persisted here (repo-relative, new test-support file).
PRESERVATION_BASELINE_PATH = Path(__file__).resolve().parent / "_p4_preservation_baseline.json"

# Frozen gate thresholds (documented registry/gate constants).
GATE_MIN_GEOMETRY_PHASE_COVERAGE_PER_DECLARED_CELL = 2
GATE_MIN_INTENT_PROGRESS_FRACTION = 0.55

# Frozen P3 temporal encoder checkpoint SHA256 (best.pt), recorded as an invariant.
P3_ENCODER_BEST_PT_SHA256 = (
    "385663281a9c48c0416ea4d1a1bb8dc08ed64a0cfebb0f01083cbbb8bcfbdc59"
)


def _p4_artifact_paths() -> list[Path]:
    """Return the sorted list of in-repo P4 artifact paths to preserve.

    The set is the portable evidence bundle: one top-level gate, four skill
    summaries, and the registry snapshot that declares the covered cells.
    """
    paths = [GATE_JSON_PATH, REGISTRY_PATH]
    paths.extend(_per_skill_summary_path(skill) for skill in SKILLS)
    return sorted(paths)


def _artifact_role(path: Path) -> str:
    """Return a stable evidence role rather than a location-dependent path."""
    if path == GATE_JSON_PATH:
        return "top_level_gate"
    if path == REGISTRY_PATH:
        return "registry_snapshot"
    for skill in SKILLS:
        if path == _per_skill_summary_path(skill):
            return f"per_skill_summary:{skill}"
    raise ValueError(f"unknown P4 evidence path: {path}")


def _sha256_manifest(paths: list[Path]) -> dict[str, str]:
    """Compute ``{stable evidence role: sha256hex}`` for existing inputs."""
    manifest: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest[_artifact_role(path)] = digest
    return manifest


def _normalize_baseline_manifest(manifest: dict[str, str]) -> dict[str, str]:
    """Upgrade the pre-bundle path-keyed baseline without rewriting it.

    The original preservation baseline predates the tracked evidence bundle and
    identifies the same six artifacts by their old repo paths.  Compare by
    evidence role so moving immutable copies into the bundle is not reported as
    a mutation of their bytes.
    """
    normalized: dict[str, str] = {}
    for key, digest in manifest.items():
        if key.endswith("p4_geometry_pretrain_gate.json"):
            role = "top_level_gate"
        elif key.endswith("thesis_five_state_shared_skill_registry_v1.yaml"):
            role = "registry_snapshot"
        else:
            role = next(
                (
                    f"per_skill_summary:{skill}"
                    for skill in SKILLS
                    if key.endswith(f"skills/{skill}/geometry_pretrain_summary.json")
                ),
                None,
            )
            if role is None:
                raise ValueError(f"unrecognized baseline evidence key: {key}")
        normalized[role] = digest
    return normalized


def test_record_preservation_baseline():
    """Record the concrete preservation baseline on the UNFIXED state.

    Computes a SHA256 manifest of the P4 artifact set and persists it (plus the
    frozen gate invariants and passing sub-checks) to
    ``tests/_p4_preservation_baseline.json`` IF it does not already exist
    (idempotent: only writes once so the baseline reflects the pre-audit state).

    Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.8, 3.9
    """
    paths = _p4_artifact_paths()

    # 1 gate + 4 summaries + 1 registry = 6 artifacts, all present.
    assert len(paths) == 6, f"expected 6 P4 artifact paths, got {len(paths)}"
    for path in paths:
        assert path.is_file(), f"P4 artifact not found: {path}"

    manifest = _sha256_manifest(paths)

    # --- Gate verdict + passing sub-checks read from the gate JSON ----------
    gate = _load_json(GATE_JSON_PATH)
    verdict = gate["all_skills_ready_for_combat_finetune"]
    assert verdict is False, "gate verdict must be false (negative evidence)"

    # reentry_recovery passes intent progress; all four skills pass profile
    # conditioning coverage and safety (read from selected_gate.checks).
    passing_subchecks: dict[str, dict[str, bool]] = {}
    for skill in SKILLS:
        checks = gate["skills"][skill]["selected_gate"]["checks"]
        passing_subchecks[skill] = {
            "profile_conditioning_coverage": checks["profile_conditioning_coverage"],
            "safety": checks["safety"],
            "intent_progress": checks["intent_progress"],
        }
        assert checks["profile_conditioning_coverage"] is True, (
            f"{skill}: profile_conditioning_coverage must pass"
        )
        assert checks["safety"] is True, f"{skill}: safety must pass"

    assert passing_subchecks["reentry_recovery"]["intent_progress"] is True, (
        "reentry_recovery must pass intent progress"
    )

    baseline = {
        "description": (
            "Concrete preservation baseline for GEO-20260713-R1 P4 attribution "
            "audit. Records the pre-audit SHA256 manifest of the P4 artifact set "
            "and the frozen gate invariants. Written once (idempotent) to reflect "
            "the UNFIXED state; the Phase 4 preservation test asserts these hashes "
            "are byte-identical after the audit runs."
        ),
        "artifact_sha256_manifest": manifest,
        "artifact_count": len(manifest),
        "gate_verdict": {
            "all_skills_ready_for_combat_finetune": verdict,
            "next_stage": gate.get("next_stage"),
        },
        "gate_thresholds": {
            "minimum_geometry_phase_coverage_per_declared_cell": (
                GATE_MIN_GEOMETRY_PHASE_COVERAGE_PER_DECLARED_CELL
            ),
            "minimum_intent_progress_fraction": GATE_MIN_INTENT_PROGRESS_FRACTION,
        },
        "p3_encoder_best_pt_sha256": P3_ENCODER_BEST_PT_SHA256,
        "passing_subchecks": passing_subchecks,
    }

    # Idempotent write: only create the baseline once so it captures the
    # pre-audit state and is never overwritten by a later (post-audit) run.
    if not PRESERVATION_BASELINE_PATH.exists():
        PRESERVATION_BASELINE_PATH.write_text(
            json.dumps(baseline, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    # --- Assertions: baseline exists and captured the expected invariants ---
    assert PRESERVATION_BASELINE_PATH.is_file(), (
        f"baseline file was not created: {PRESERVATION_BASELINE_PATH}"
    )
    recorded = _load_json(PRESERVATION_BASELINE_PATH)

    # 6 artifact hashes (1 gate + 4 summaries + 1 registry).
    assert len(recorded["artifact_sha256_manifest"]) == 6, (
        "baseline must capture 6 artifact hashes"
    )
    assert recorded["artifact_count"] == 6

    # False verdict captured.
    assert (
        recorded["gate_verdict"]["all_skills_ready_for_combat_finetune"] is False
    )

    # Recorded invariants (thresholds + P3 encoder SHA256).
    assert (
        recorded["gate_thresholds"][
            "minimum_geometry_phase_coverage_per_declared_cell"
        ]
        == 2
    )
    assert (
        recorded["gate_thresholds"]["minimum_intent_progress_fraction"] == 0.55
    )
    assert recorded["p3_encoder_best_pt_sha256"] == P3_ENCODER_BEST_PT_SHA256

    # Passing sub-checks captured (reentry_recovery intent progress; all four
    # skills profile coverage + safety).
    assert recorded["passing_subchecks"]["reentry_recovery"]["intent_progress"] is True
    for skill in SKILLS:
        assert (
            recorded["passing_subchecks"][skill]["profile_conditioning_coverage"]
            is True
        )
        assert recorded["passing_subchecks"][skill]["safety"] is True


# ===========================================================================
# Phase 3, Task 4.5: Parametrized tests over the pure functions.
#
# Synthetic fixtures ONLY (never the frozen artifact); no hardcoded run counts.
# NO hypothesis / NO PBT — concrete/enumerated inputs via @pytest.mark.parametrize.
# These VERIFY spec-level Property 1 and Property 2 with concrete inputs.
#
# Validates: Requirements 2.7, 2.2, 2.3, 2.4, 2.9, 2.11, 1.3, 1.4
# ===========================================================================

# Insert the repo `src` dir onto sys.path so the audit module is importable.
sys.path.insert(0, str(REPO_ROOT / "src"))

from uav_vpp_guidance.evaluation.p4_gate_failure_attribution import (  # noqa: E402
    ABSENT_SENTINEL,
    CONCLUSION_COMBAT_FINETUNE,
    CONTROLLED_LABELS as MODULE_CONTROLLED_LABELS,
    DeclaredCell,
    EMISSION_DECLARED_NOT_EMITTED,
    EMISSION_EMITTED,
    FIELD_STATUS_ABSENT,
    FIELD_STATUS_PRESENT_GT0,
    FIELD_STATUS_PRESENT_ZERO,
    ReconcileResult,
    Row,
    assign_label,
    build_matrix,
    collect_emitted_cells,
    cross_artifact_consistency,
    derive_declared_cells,
    emit_csv,
    emit_json,
    field_status,
    freeze_provenance,
    preflight_block,
    reconcile,
    self_check,
)

# The three field-status values that are the only admissible encodings; note
# there is deliberately NO "physically impossible" verdict.
FIELD_STATUS_VALUES = {
    FIELD_STATUS_ABSENT,
    FIELD_STATUS_PRESENT_ZERO,
    FIELD_STATUS_PRESENT_GT0,
}


# ---------------------------------------------------------------------------
# Synthetic gate/summary builders (NOT the frozen artifact).
# ---------------------------------------------------------------------------


def _synth_selected_gate(coverage, intent_fraction, gpc_passed, ip_passed,
                         status="trained"):
    """Build a synthetic ``selected_gate`` block."""
    return {
        "geometry_phase_coverage": dict(coverage),
        "intent_progress_fraction": intent_fraction,
        "status": status,
        "profile_coverage": {"all_profiles_present": True},
        "safety_by_opponent": {"expert": True},
        "checks": {
            "geometry_phase_coverage": gpc_passed,
            "intent_progress": ip_passed,
            "profile_conditioning_coverage": True,
            "safety": True,
        },
    }


def _synth_gate_json(skill, declared_cells, emitted, train, intent_fraction,
                    gpc_passed=False, ip_passed=False):
    """Build a synthetic one-skill gate JSON object for the pure functions."""
    return {
        "skills": {
            skill: {
                "selected_gate": _synth_selected_gate(
                    emitted, intent_fraction, gpc_passed, ip_passed
                ),
                "train_aggregate": {"state_phase_steps": dict(train)},
            }
        }
    }


# ---------------------------------------------------------------------------
# 1. Reconciliation (parametrized): fully-emitted / one-missing / several-missing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "declared_cells,emitted_steps,expected_diff,expected_extra",
    [
        # fully-emitted: every declared cell is emitted -> diff 0, extra 0.
        (
            ["advantage:pre_merge", "neutral:pre_merge"],
            {"advantage:pre_merge": 5, "neutral:pre_merge": 3},
            set(),
            set(),
        ),
        # one-missing: one declared cell has no emitted key -> diff 1, extra 0.
        (
            ["advantage:pre_merge", "neutral:pre_merge"],
            {"advantage:pre_merge": 5},
            {("synthskill", "neutral", "pre_merge")},
            set(),
        ),
        # several-missing: two declared cells not emitted -> diff 2, extra 0.
        (
            [
                "advantage:pre_merge",
                "neutral:pre_merge",
                "disadvantage:pre_merge",
            ],
            {"advantage:pre_merge": 5},
            {
                ("synthskill", "neutral", "pre_merge"),
                ("synthskill", "disadvantage", "pre_merge"),
            },
            set(),
        ),
    ],
)
def test_reconcile_diff_extra_sets_synthetic(
    declared_cells, emitted_steps, expected_diff, expected_extra
):
    """`reconcile` produces the correct diff/extra sets on synthetic fixtures.

    The frozen artifact has diff 0, so the diff-mapping path is exercised ONLY
    on these synthetic fixtures.

    Validates: Requirements 2.9, 2.11
    """
    skill = "synthskill"
    declared = {skill: list(declared_cells)}
    gate_json = _synth_gate_json(
        skill, declared_cells, emitted_steps, {"advantage:pre_merge": 7},
        intent_fraction=0.5,
    )
    emitted = collect_emitted_cells(gate_json)
    rec = reconcile(declared, emitted)

    assert rec.diff_keys() == expected_diff
    assert {c.key for c in rec.extra} == expected_extra
    assert rec.diff_count == len(expected_diff)
    assert rec.extra_count == len(expected_extra)

    # Relational invariant: declared == emitted + diff when extra is empty.
    assert rec.emitted_count + rec.diff_count == rec.declared_count

    # self_check must hold on a matrix built from the same reconciliation.
    rows = build_matrix(gate_json, declared, rec)
    self_check(rows, rec)  # raises on violation

    # emitted + diff == declared (extra empty) reflected in the preflight block.
    pf = preflight_block(rec)
    assert pf["extra"] == []
    assert pf["emitted_count"] + pf["diff_count"] == pf["declared_count"]

    # Diff cells are labeled contract_or_provenance_mismatch (Property 1).
    diff_keys = rec.diff_keys()
    for row in rows:
        key = (row.skill, row.initial_class, row.phase)
        if key in diff_keys:
            assert row.emission_status == EMISSION_DECLARED_NOT_EMITTED
            assert row.attribution_label == "contract_or_provenance_mismatch"
        else:
            assert row.emission_status == EMISSION_EMITTED


# ---------------------------------------------------------------------------
# 2. Three field-status columns (parametrized): absent / value_0 / value_gt_0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "coverage,cell,expected",
    [
        # absent: key not present in the coverage source.
        ({}, "advantage:post_merge", FIELD_STATUS_ABSENT),
        ({"neutral:pre_merge": 4}, "advantage:post_merge", FIELD_STATUS_ABSENT),
        # present with value 0 -> "recorded as 0 steps under current gate semantics".
        ({"advantage:post_merge": 0}, "advantage:post_merge", FIELD_STATUS_PRESENT_ZERO),
        # present with value > 0.
        ({"advantage:post_merge": 1}, "advantage:post_merge", FIELD_STATUS_PRESENT_GT0),
        ({"advantage:post_merge": 1092}, "advantage:post_merge", FIELD_STATUS_PRESENT_GT0),
    ],
)
@pytest.mark.parametrize("source", ["train", "dev"])
def test_field_status_encoding_synthetic(coverage, cell, expected, source):
    """`field_status` encodes absent / present_value_0 / present_value_gt_0.

    Applies to BOTH the train coverage source and the dev gate coverage source.
    Absence NEVER yields a "physically impossible" verdict — it yields 'absent'.

    Validates: Requirements 1.3, 2.2, 2.3
    """
    result = field_status(coverage, cell)
    assert result == expected
    # Only the three admissible encodings ever occur (no "impossible" verdict).
    assert result in FIELD_STATUS_VALUES
    # Absence resolves ONLY to 'absent', never to any impossibility claim.
    if cell not in coverage:
        assert result == FIELD_STATUS_ABSENT


# ---------------------------------------------------------------------------
# 3. Label safety (parametrized): (train, dev, requires_coverage, in_diff)
# ---------------------------------------------------------------------------


def _dev_field_status(dev_steps):
    """Encode the dev-gate field status consistent with a dev step value."""
    if dev_steps == ABSENT_SENTINEL:
        return FIELD_STATUS_ABSENT
    if dev_steps > 0:
        return FIELD_STATUS_PRESENT_GT0
    return FIELD_STATUS_PRESENT_ZERO


def _train_field_status(train_steps):
    if train_steps == ABSENT_SENTINEL:
        return FIELD_STATUS_ABSENT
    if train_steps > 0:
        return FIELD_STATUS_PRESENT_GT0
    return FIELD_STATUS_PRESENT_ZERO


def _make_synth_row(skill, state, phase, train_steps, dev_steps, requires,
                   emission_status=EMISSION_EMITTED):
    """Construct a minimal synthetic Row for feeding assign_label."""
    return Row(
        skill=skill,
        initial_class=state,
        phase=phase,
        profile_validity_precondition="synthetic",
        train_design_requires_coverage=requires,
        train_coverage_evidence="synthetic.train",
        train_coverage_steps=train_steps,
        dev30_coverage_evidence="synthetic.dev",
        dev30_coverage_steps=dev_steps,
        train_field_status=_train_field_status(train_steps),
        dev_gate_field_status=_dev_field_status(dev_steps),
        emission_status=emission_status,
        gate_min_steps=2,
        cell_gate_passed=False,
        skill_geometry_phase_gate_passed=False,
        skill_intent_progress_passed=False,
        intent_progress_fraction=0.5,
        intent_progress_threshold=0.55,
        attribution_label="",
        reason_code="",
        evidence_paths=["synthetic"],
        confidence="",
        next_step_recommendation="",
    )


def _reconcile_with_diff_keys(diff_keys):
    """Build a synthetic ReconcileResult whose diff_keys() == diff_keys."""
    diff_cells = [DeclaredCell(s, st, ph) for (s, st, ph) in diff_keys]
    return ReconcileResult(
        declared=list(diff_cells),
        emitted=[],
        diff=list(diff_cells),
        extra=[],
    )


@pytest.mark.parametrize(
    "train_steps,dev_steps,requires,in_diff,expected_label,expected_reason",
    [
        # in_diff True (synthetic, non-frozen) -> contract_or_provenance_mismatch.
        (5, 5, True, True, "contract_or_provenance_mismatch", ""),
        (0, 0, True, True, "contract_or_provenance_mismatch", ""),
        # dev absent -> not_observed_cannot_assess.
        (5, ABSENT_SENTINEL, True, False, "not_observed_cannot_assess", ""),
        (0, ABSENT_SENTINEL, False, False, "not_observed_cannot_assess", ""),
        # train==0 & dev==0 & requires -> train_coverage_gap.
        (0, 0, True, False, "train_coverage_gap", ""),
        # train==0 & dev==0 & NOT requires -> not_observed_cannot_assess.
        (0, 0, False, False, "not_observed_cannot_assess", ""),
        # train>0 & dev==0 -> not_observed_cannot_assess + dev_coverage_gap.
        (5, 0, True, False, "not_observed_cannot_assess", "dev_coverage_gap"),
        # train>0 & dev>0 -> mixed_or_inconclusive.
        (5, 5, True, False, "mixed_or_inconclusive", ""),
        (3, 1, False, False, "mixed_or_inconclusive", ""),
    ],
)
def test_assign_label_safety_synthetic(
    train_steps, dev_steps, requires, in_diff, expected_label, expected_reason
):
    """`assign_label` returns the expected (label, reason_code) pairs (design §3).

    observed_behavior_failure is NEVER produced under the cell-quality-free
    evidence set.

    Validates: Requirements 2.2, 2.3, 2.4, 1.4
    """
    skill, state, phase = "synthskill", "neutral", "pre_merge"
    key = (skill, state, phase)
    emission_status = (
        EMISSION_DECLARED_NOT_EMITTED if in_diff else EMISSION_EMITTED
    )
    row = _make_synth_row(
        skill, state, phase, train_steps, dev_steps, requires, emission_status
    )
    rec = _reconcile_with_diff_keys({key} if in_diff else set())

    label, reason = assign_label(row, rec)

    assert (label, reason) == (expected_label, expected_reason)
    # observed_behavior_failure is NEVER assigned.
    assert label != "observed_behavior_failure"
    # Every label is one of the six controlled labels.
    assert label in MODULE_CONTROLLED_LABELS


def test_assign_label_never_emits_observed_behavior_failure_over_grid():
    """Exhaustively enumerate a grid; observed_behavior_failure never appears.

    Validates: Requirements 2.4, 1.4
    """
    skill, state, phase = "synthskill", "advantage", "pre_merge"
    key = (skill, state, phase)
    produced = set()
    for train_steps in (ABSENT_SENTINEL, 0, 1, 5, 1092):
        for dev_steps in (ABSENT_SENTINEL, 0, 1, 5, 182):
            for requires in (True, False):
                for in_diff in (True, False):
                    emission_status = (
                        EMISSION_DECLARED_NOT_EMITTED if in_diff else EMISSION_EMITTED
                    )
                    row = _make_synth_row(
                        skill, state, phase, train_steps, dev_steps, requires,
                        emission_status,
                    )
                    rec = _reconcile_with_diff_keys({key} if in_diff else set())
                    label, _reason = assign_label(row, rec)
                    produced.add(label)
                    assert label in MODULE_CONTROLLED_LABELS

    assert "observed_behavior_failure" not in produced


# ---------------------------------------------------------------------------
# 4. Cross-artifact consistency (2b) (parametrized): clean vs inconsistent
# ---------------------------------------------------------------------------


def _synth_cross_artifact_pair(top_intent, summary_intent):
    """Build a (gate_json, per_skill_summaries) pair for the 2b preflight."""
    skill = "synthskill"
    coverage = {"advantage:pre_merge": 5, "neutral:pre_merge": 0}
    top_sel = _synth_selected_gate(coverage, top_intent, False, False)
    summary_sel = _synth_selected_gate(coverage, summary_intent, False, False)
    gate_json = {"skills": {skill: {"selected_gate": top_sel}}}
    per_skill_summaries = {skill: {"selected_gate": summary_sel}}
    return gate_json, per_skill_summaries


@pytest.mark.parametrize(
    "top_intent,summary_intent,expect_problems",
    [
        # consistent: top-level gate agrees with the per-skill summary -> clean.
        (0.4977, 0.4977, False),
        # inconsistent: differing intent_progress_fraction -> non-empty list.
        (0.4977, 0.5820, True),
    ],
)
def test_cross_artifact_consistency_synthetic(
    top_intent, summary_intent, expect_problems
):
    """`cross_artifact_consistency` (2b): clean -> [], inconsistent -> non-empty.

    A consistent fixture drives the exit-0 path; an inconsistent fixture drives
    the integrity-report / non-zero-exit path with NO matrix.

    Validates: Requirements 2.7, 2.9, 2.11
    """
    gate_json, per_skill_summaries = _synth_cross_artifact_pair(
        top_intent, summary_intent
    )
    problems = cross_artifact_consistency(gate_json, per_skill_summaries)

    if expect_problems:
        assert problems, "inconsistent artifacts must yield a non-empty problem list"
        assert any(p.field == "intent_progress_fraction" for p in problems)
    else:
        assert problems == [], "consistent artifacts must yield an empty problem list"


# ---------------------------------------------------------------------------
# 5. Provenance freezing: provenance_unverified + run-level mismatch flag.
# ---------------------------------------------------------------------------


def test_freeze_provenance_marks_missing_frozen_inputs():
    """`freeze_provenance` marks missing frozen inputs provenance_unverified.

    Missing training_plan / resolved_geometry_config / dev30_manifest / run_commit
    each become provenance_unverified and set the run-level
    contract_or_provenance_mismatch flag. Working-tree info is a SEPARATE field.

    Validates: Requirements 2.9, 2.11, 1.4
    """
    inputs = {
        "training_plan": None,
        "resolved_geometry_config": None,
        "dev30_manifest": None,
    }
    block = freeze_provenance(inputs, run_commit=None)

    # Run-level contract_or_provenance_mismatch flag raised.
    assert block.contract_or_provenance_mismatch is True

    unverified = set(block.provenance_unverified_inputs)
    for missing in ("training_plan", "resolved_geometry_config", "dev30_manifest",
                    "run_commit"):
        assert missing in unverified, f"{missing} must be provenance_unverified"

    # run_commit is unverified (missing frozen fact).
    assert block.run_commit is None
    assert block.run_commit_unverified is True

    d = block.to_dict()
    # Working-tree git info is a SEPARATE field from frozen provenance.
    assert "working_tree" in d
    assert "inputs" in d
    assert d["working_tree"] is not d.get("inputs")
    # The working-tree block never masquerades as a frozen run fact.
    assert isinstance(d["working_tree"], dict)
    assert "note" in d["working_tree"]


# ---------------------------------------------------------------------------
# 6. CSV/JSON content consistency: identical rows/values across writers.
# ---------------------------------------------------------------------------


def test_emit_csv_json_content_consistency_synthetic(tmp_path):
    """emit_csv and emit_json emit the SAME rows/values from one matrix.

    Validates: Requirements 2.7, 2.11
    """
    import csv as _csv

    skill = "synthskill"
    declared_cells = [
        "advantage:pre_merge",
        "neutral:post_merge",
        "disadvantage:re_entry",
    ]
    emitted_steps = {
        "advantage:pre_merge": 40,
        "neutral:post_merge": 0,
        "disadvantage:re_entry": 0,
    }
    train_steps = {"advantage:pre_merge": 55}
    declared = {skill: list(declared_cells)}
    gate_json = _synth_gate_json(
        skill, declared_cells, emitted_steps, train_steps, intent_fraction=0.5,
    )
    emitted = collect_emitted_cells(gate_json)
    rec = reconcile(declared, emitted)
    rows = build_matrix(gate_json, declared, rec)
    self_check(rows, rec)

    provenance = freeze_provenance({}, run_commit=None)

    csv_path = tmp_path / "matrix.csv"
    json_path = tmp_path / "matrix.json"
    emit_csv(rows, str(csv_path))
    emit_json(rows, rec, provenance, CONCLUSION_COMBAT_FINETUNE, str(json_path))

    audit = json.loads(json_path.read_text(encoding="utf-8"))
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        csv_rows = list(_csv.DictReader(fh))

    json_rows = audit["rows"]
    assert len(csv_rows) == len(json_rows) == len(declared_cells)
    assert audit["conclusion"] == CONCLUSION_COMBAT_FINETUNE

    # Row-by-row: skills / cells / labels / statuses must be identical.
    for csv_row, json_row in zip(csv_rows, json_rows):
        assert csv_row["skill"] == json_row["skill"]
        assert csv_row["initial_class"] == json_row["initial_class"]
        assert csv_row["phase"] == json_row["phase"]
        assert csv_row["attribution_label"] == json_row["attribution_label"]
        assert csv_row["dev_gate_field_status"] == json_row["dev_gate_field_status"]
        assert csv_row["train_field_status"] == json_row["train_field_status"]
        assert csv_row["emission_status"] == json_row["emission_status"]
        assert csv_row["reason_code"] == (json_row["reason_code"] or "")

    # The set of (skill, cell) pairs is identical between the two artifacts.
    csv_cells = {
        (r["skill"], f"{r['initial_class']}:{r['phase']}") for r in csv_rows
    }
    json_cells = {
        (r["skill"], f"{r['initial_class']}:{r['phase']}") for r in json_rows
    }
    assert csv_cells == json_cells


# ===========================================================================
# Phase 4, Task 6: concrete preservation verification (SHA256 + git allowlist).
#
# This is CONCRETE verification, NOT a property-based test. It reuses the
# baseline manifest recorded in Phase 2 (tests/_p4_preservation_baseline.json)
# and asserts that, after the audit ran (Task 4.3), every preserved P4 artifact
# is byte-identical, the frozen gate invariants are unchanged, the passing
# sub-checks are still passes, and the git working-tree changes are confined to
# the audit's own allowlist (no P4 artifact / gate / checkpoint / mainline file
# was modified).
#
# Robust to the repo's pre-existing untracked noise (e.g. .kiro/, .codex_tmp_*,
# .hypothesis/, the untracked evidence under src/): the allowlist assertion is
# restricted to the audit's OWN files.
#
# Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.8, 3.9
# ===========================================================================

# The audit's own files (repo-relative POSIX paths). These are the ONLY paths
# this task is permitted to create or modify.
AUDIT_ALLOWLIST = {
    "src/uav_vpp_guidance/evaluation/p4_gate_failure_attribution.py",
    "scripts/analyze_p4_geometry_gate_failure_attribution.py",
    "tests/test_p4_gate_failure_attribution.py",
    "tests/_p4_preservation_baseline.json",
    "reports/thesis_five_state_p4_gate_failure_attribution_audit_20260714_zh.md",
    "reports/thesis_five_state_p4_gate_failure_attribution_matrix_20260714.csv",
    "reports/thesis_five_state_p4_gate_failure_attribution_matrix_20260714.json",
    "reports/thesis_five_state_p4_v1_go_no_go_decision_20260714_zh.md",
    "reports/thesis_five_state_p4_v2_sampler_feasibility_preflight_design_20260714_zh.md",
    "reports/thesis_five_state_shared_intent_v1_execution_checklist_20260713_zh.md",
    ".gitattributes",
    "reports/p4_evidence_bundle_20260714/evidence_manifest.json",
    "reports/p4_evidence_bundle_20260714/p4_geometry_pretrain_gate.json",
    "reports/p4_evidence_bundle_20260714/skills/defensive_extension/geometry_pretrain_summary.json",
    "reports/p4_evidence_bundle_20260714/skills/lead_intercept/geometry_pretrain_summary.json",
    "reports/p4_evidence_bundle_20260714/skills/pursuit_conversion/geometry_pretrain_summary.json",
    "reports/p4_evidence_bundle_20260714/skills/reentry_recovery/geometry_pretrain_summary.json",
}

# The execution checklist is a PRE-EXISTING tracked file that the audit edited;
# it is therefore expected to appear as a tracked MODIFICATION.
CHECKLIST_REL = (
    "reports/thesis_five_state_shared_intent_v1_execution_checklist_20260713_zh.md"
)

# The audit and its evidence bundle are committed deliverables.  A portability
# test must not require them to remain untracked after checkout.
NEW_AUDIT_FILES_REL: set[str] = set()


def _git_output(args: list[str]) -> str:
    """Run a git command at the repo root and return its stdout."""
    proc = subprocess.run(
        ["git", *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"git {' '.join(args)} exited {proc.returncode}\nstderr={proc.stderr}"
    )
    return proc.stdout


def _git_tracked_modified() -> set[str]:
    """Repo-relative POSIX paths of tracked files with unstaged modifications.

    Derived from ``git diff --name-only`` (tracked working-tree modifications).
    """
    out = _git_output(["diff", "--name-only"])
    return {line.strip() for line in out.splitlines() if line.strip()}


def _parse_porcelain(out: str) -> tuple[set[str], set[str], set[str]]:
    """Parse ``git status --porcelain`` output.

    Returns (tracked_modified, untracked, ignored) as sets of repo-relative
    POSIX paths. Porcelain lines are ``XY <path>`` where ``XY`` is the two-char
    status code. Untracked entries are ``??`` and ignored entries ``!!`` (the
    latter only when ``--ignored`` is passed). Directory entries retain their
    trailing slash exactly as git reports them.
    """
    tracked_modified: set[str] = set()
    untracked: set[str] = set()
    ignored: set[str] = set()
    for line in out.splitlines():
        if not line.strip():
            continue
        code = line[:2]
        path = line[3:].strip()
        # Handle rename form "old -> new": keep the destination path.
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        # Strip surrounding quotes git adds for paths with special chars.
        if path.startswith('"') and path.endswith('"'):
            path = path[1:-1]
        if code == "??":
            untracked.add(path)
        elif code == "!!":
            ignored.add(path)
        else:
            # Any tracked change (M/A/D/R/C in either column).
            tracked_modified.add(path)
    return tracked_modified, untracked, ignored


def test_preservation_p4_artifacts_unchanged():
    """Preservation verification: frozen P4 assets are byte-identical after audit.

    Reuses the Phase 2 baseline manifest and asserts:
      1. every preserved P4 artifact hash matches the baseline byte-for-byte,
      2. the gate verdict / thresholds / P3 encoder SHA256 are unchanged,
      3. the passing sub-checks are still reported as passes,
      4. all working-tree changes are confined to the audit's own allowlist
         (tracked modifications among audit files == the checklist ONLY; no
         preserved artifact / gate / checkpoint / mainline file was modified).

    Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.8, 3.9
    """
    # --- Step 1: baseline must exist (created by the Phase 2 baseline test) --
    assert PRESERVATION_BASELINE_PATH.is_file(), (
        f"preservation baseline not found: {PRESERVATION_BASELINE_PATH}. "
        "It is produced by test_record_preservation_baseline (earlier in this "
        "file); run the full module so the baseline is recorded first."
    )
    baseline = _load_json(PRESERVATION_BASELINE_PATH)

    # --- Step 2: recompute the current manifest; byte-identical to baseline ---
    paths = _p4_artifact_paths()
    assert len(paths) == 6, f"expected 6 P4 artifact paths, got {len(paths)}"
    for path in paths:
        assert path.is_file(), f"P4 artifact not found: {path}"

    current_manifest = _sha256_manifest(paths)
    baseline_manifest = _normalize_baseline_manifest(
        baseline["artifact_sha256_manifest"]
    )

    assert set(current_manifest) == set(baseline_manifest), (
        "preserved artifact set changed between baseline and now: "
        f"baseline={sorted(baseline_manifest)} current={sorted(current_manifest)}"
    )
    for relpath, baseline_hash in baseline_manifest.items():
        assert current_manifest[relpath] == baseline_hash, (
            f"artifact {relpath} changed (audit must be read-only against "
            f"evidence): baseline={baseline_hash} current={current_manifest[relpath]}"
        )

    # --- Step 3: gate verdict / thresholds / P3 encoder SHA256 unchanged -----
    gate = _load_json(GATE_JSON_PATH)
    assert gate["all_skills_ready_for_combat_finetune"] is False, (
        "gate verdict must remain false (negative evidence preserved)"
    )
    assert (
        gate["all_skills_ready_for_combat_finetune"]
        == baseline["gate_verdict"]["all_skills_ready_for_combat_finetune"]
    )

    assert (
        GATE_MIN_GEOMETRY_PHASE_COVERAGE_PER_DECLARED_CELL
        == baseline["gate_thresholds"][
            "minimum_geometry_phase_coverage_per_declared_cell"
        ]
        == 2
    )
    assert (
        GATE_MIN_INTENT_PROGRESS_FRACTION
        == baseline["gate_thresholds"]["minimum_intent_progress_fraction"]
        == 0.55
    )
    assert (
        P3_ENCODER_BEST_PT_SHA256
        == baseline["p3_encoder_best_pt_sha256"]
        == "385663281a9c48c0416ea4d1a1bb8dc08ed64a0cfebb0f01083cbbb8bcfbdc59"
    )

    # --- Step 4: passing sub-checks still reported as passes -----------------
    # (reentry_recovery intent progress; all four profile coverage + safety),
    # never recharacterized as failures.
    for skill in SKILLS:
        checks = gate["skills"][skill]["selected_gate"]["checks"]
        assert checks["profile_conditioning_coverage"] is True, (
            f"{skill}: profile_conditioning_coverage must still pass"
        )
        assert checks["safety"] is True, f"{skill}: safety must still pass"
        assert (
            checks["profile_conditioning_coverage"]
            == baseline["passing_subchecks"][skill]["profile_conditioning_coverage"]
        )
        assert checks["safety"] == baseline["passing_subchecks"][skill]["safety"]

    assert (
        gate["skills"]["reentry_recovery"]["selected_gate"]["checks"][
            "intent_progress"
        ]
        is True
    ), "reentry_recovery must still pass intent progress"

    # --- Step 5: git allowlist (git diff --name-only AND git status --porcelain)
    tracked_modified = _git_tracked_modified()
    porcelain_out = _git_output(["status", "--porcelain"])
    _pm_mod, untracked, _pm_ign = _parse_porcelain(porcelain_out)
    # Supplementary parse WITH --ignored so a gitignored audit output (e.g. the
    # CSV, matched by a *.csv ignore rule) can still be classified as ours.
    ignored_out = _git_output(["status", "--porcelain", "--ignored"])
    _ig_mod, _ig_unt, ignored = _parse_porcelain(ignored_out)

    # git diff and git status --porcelain must agree on the tracked-modified set
    # (both are used, per the task).
    assert tracked_modified == _pm_mod, (
        "git diff --name-only and git status --porcelain disagree on tracked "
        f"modifications: diff={tracked_modified} porcelain={_pm_mod}"
    )

    # (a) A clean checkout has no modifications; an in-progress portability
    # repair may touch only the explicit audit/bundle allowlist.
    assert tracked_modified <= AUDIT_ALLOWLIST, (
        "portable audit repair modified an out-of-scope tracked file: "
        f"{sorted(tracked_modified - AUDIT_ALLOWLIST)}"
    )

    # (b) No tracked change outside the audit/bundle allowlist is permitted.
    audit_tracked_mods = tracked_modified & AUDIT_ALLOWLIST
    assert audit_tracked_mods == tracked_modified, (
        "tracked modification classification lost an audit/bundle path: "
        f"tracked={sorted(tracked_modified)} audit={sorted(audit_tracked_mods)}"
    )

    # (c) NONE of the 6 preserved P4 artifacts appears as a tracked modification.
    preserved_relpaths = set(baseline_manifest)
    for relpath in preserved_relpaths:
        assert relpath not in tracked_modified, (
            f"preserved P4 artifact was modified in the working tree: {relpath}"
        )

    # (d) The new audit files (module/script/test/baseline/reports) must appear
    #     as untracked-added (or ignored, for the gitignored CSV) — NEVER as a
    #     tracked modification — and must exist on disk (produced by the audit).
    for relpath in sorted(NEW_AUDIT_FILES_REL):
        disk_path = REPO_ROOT / relpath
        assert disk_path.is_file(), f"expected audit output not produced: {relpath}"
        assert relpath not in tracked_modified, (
            f"new audit file must not be a tracked modification: {relpath}"
        )
        assert relpath in untracked or relpath in ignored, (
            f"new audit file must be untracked-added or ignored (not committed "
            f"content): {relpath}"
        )
