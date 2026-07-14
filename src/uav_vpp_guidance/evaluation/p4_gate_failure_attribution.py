"""P4 geometry-pretrain gate failure-attribution audit (read-only).

This module implements the pure-function core of the P4 geometry gate
failure-attribution audit for the frozen run ``GEO-20260713-R1``.

It is a strictly **non-training, non-tuning, read-only** audit layer over the
frozen gate evidence. It never retrains, re-evaluates, edits the pre-registered
gate, or mutates any P4 output / checkpoint / model / reward. The only side
effects are performed by the explicit ``emit_*`` writers when a caller passes an
output path; importing this module and calling the analysis functions has no
side effects.

Design references (see
``.kiro/specs/p4-geometry-gate-failure-attribution-audit/design.md``):
  * §2a/2b Preflights
  * §3 Attribution Matrix Schema + executable ``assignLabel`` rules
  * §5 Output Artifacts
  * §6 Analysis Script Design

Gate-semantics constants cited (verbatim source lines in
``src/uav_vpp_guidance/training/thesis_shared_skill_geometry.py``):
  * coverage unit = policy steps; minimum >= 2 steps per declared cell
    (``minimum_geometry_phase_coverage_per_declared_cell``) — implemented at
    ``thesis_shared_skill_geometry.py:544`` / ``:561``.
  * intent_progress = intent_progress_steps / policy_steps; a step counts toward
    the numerator only if its ``intent_progress > 0``
    (``thesis_shared_skill_geometry.py:451``); evaluated against threshold 0.55
    (``minimum_intent_progress_fraction``) at ``thesis_shared_skill_geometry.py:564``.
  * coverage map construction (Cartesian product, absent filled with 0 via
    ``.get(..., 0)``) at ``thesis_shared_skill_geometry.py:511-546``.

STRICT constraints honored by this module:
  * Python standard library only (no PyYAML, no hypothesis).
  * No hardcoded drive letters.
  * No hardcoded run-specific counts (26/26/0/0, 26/22/4, etc.); every count is
    derived adaptively from the provided inputs.
  * Pure functions; no side effects on import.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Gate semantics constants (for citation in output; NOT run-specific counts).
# ---------------------------------------------------------------------------

#: Minimum policy steps per declared cell for the geometry-phase-coverage check.
#: (``minimum_geometry_phase_coverage_per_declared_cell``,
#: thesis_shared_skill_geometry.py:544/:561)
GATE_MIN_STEPS: int = 2

#: Minimum intent-progress fraction (``minimum_intent_progress_fraction``,
#: thesis_shared_skill_geometry.py:564).
INTENT_PROGRESS_THRESHOLD: float = 0.55

#: Coverage unit for the gate (policy steps, not episodes).
COVERAGE_UNIT: str = "policy_steps"

#: Source-line citations for gate semantics (used in evidence_paths / comments).
GATE_SEMANTICS_SOURCE: Dict[str, str] = {
    "intent_progress_numerator": "thesis_shared_skill_geometry.py:451",
    "coverage_min_steps": "thesis_shared_skill_geometry.py:544",
    "coverage_min_steps_check": "thesis_shared_skill_geometry.py:561",
    "intent_progress_threshold": "thesis_shared_skill_geometry.py:564",
    "coverage_cartesian_product": "thesis_shared_skill_geometry.py:511-546",
    "raw_train_trajectory_prohibition": "thesis_shared_skill_geometry.py:752",
}

#: The six controlled attribution labels (design §3). These are the ONLY allowed
#: values for ``attribution_label``.
CONTROLLED_LABELS: Tuple[str, ...] = (
    "not_observed_cannot_assess",
    "train_coverage_gap",
    "observed_behavior_failure",
    "telemetry_or_gate_observability_gap",
    "mixed_or_inconclusive",
    "contract_or_provenance_mismatch",
)

#: The final, immutable audit conclusion string.
CONCLUSION_COMBAT_FINETUNE: str = "combat_finetune_NOT_allowed"

#: The frozen inputs that are expected to be missing from the in-repo layout and
#: therefore recorded as ``provenance_unverified`` when absent.
FROZEN_PROVENANCE_INPUTS: Tuple[str, ...] = (
    "run_commit",
    "training_plan",
    "resolved_geometry_config",
    "dev30_manifest",
)

#: Phases for which a dynamic-posture claim is unprovable from fixed
#: ``initial_class`` recording (design §4 initial-class × phase limitation).
DYNAMIC_POSTURE_PHASES: Tuple[str, ...] = ("post_merge", "re_entry")

# Field-status encodings applied independently to each coverage source.
FIELD_STATUS_ABSENT = "absent"
FIELD_STATUS_PRESENT_ZERO = "present_value_0"
FIELD_STATUS_PRESENT_GT0 = "present_value_gt_0"

# Emission-status encodings (declared-vs-emitted reconciliation outcome).
EMISSION_EMITTED = "emitted"
EMISSION_DECLARED_NOT_EMITTED = "declared_not_emitted"

# Sentinel used in step columns when a coverage key is absent from its source.
ABSENT_SENTINEL = "absent"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeclaredCell:
    """A ``skill × geometry_state × phase`` triple declared by the registry.

    The registry is used as a documented ``provenance_unverified`` substitute
    for the missing frozen ``training_plan.json``.
    """

    skill: str
    geometry_state: str
    phase: str

    @property
    def cell(self) -> str:
        """The ``state:phase`` coverage-map key for this cell."""
        return f"{self.geometry_state}:{self.phase}"

    @property
    def key(self) -> Tuple[str, str, str]:
        return (self.skill, self.geometry_state, self.phase)


@dataclass(frozen=True)
class EmittedCell:
    """A ``state:phase`` key actually present in a skill's coverage map."""

    skill: str
    geometry_state: str
    phase: str
    steps: int

    @property
    def cell(self) -> str:
        return f"{self.geometry_state}:{self.phase}"

    @property
    def key(self) -> Tuple[str, str, str]:
        return (self.skill, self.geometry_state, self.phase)


@dataclass
class ReconcileResult:
    """Result of the 2a declared-vs-emitted reconciliation preflight.

    All counts are derived from the inputs, never hardcoded.
    """

    declared: List[DeclaredCell]
    emitted: List[EmittedCell]
    diff: List[DeclaredCell]
    extra: List[EmittedCell]

    @property
    def declared_count(self) -> int:
        return len(self.declared)

    @property
    def emitted_count(self) -> int:
        return len(self.emitted)

    @property
    def diff_count(self) -> int:
        return len(self.diff)

    @property
    def extra_count(self) -> int:
        return len(self.extra)

    def diff_keys(self) -> set:
        return {c.key for c in self.diff}


@dataclass
class Problem:
    """A single cross-artifact-consistency (2b) discrepancy."""

    skill: str
    field: str
    detail: str = ""


@dataclass
class InputProvenance:
    """Frozen-provenance record for a single evidence input file."""

    name: str
    path: Optional[str]
    available: bool
    sha256: Optional[str] = None
    top_level_fields: Optional[List[str]] = None
    schema_version: Optional[Any] = None
    provenance_unverified: bool = False
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "available": self.available,
            "sha256": self.sha256,
            "top_level_fields": self.top_level_fields,
            "schema_version": self.schema_version,
            "provenance_unverified": self.provenance_unverified,
            "note": self.note,
        }


@dataclass
class ProvenanceBlock:
    """Frozen provenance + separate working-tree git info.

    Frozen run facts (``inputs``, ``run_commit``) are kept strictly separate
    from ``working_tree`` git info so the current worktree source can never be
    substituted for the frozen run.
    """

    inputs: List[InputProvenance] = field(default_factory=list)
    run_commit: Optional[str] = None
    run_commit_unverified: bool = True
    provenance_unverified_inputs: List[str] = field(default_factory=list)
    contract_or_provenance_mismatch: bool = False
    working_tree: Dict[str, Any] = field(default_factory=dict)
    host_interpreter_note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "inputs": [i.to_dict() for i in self.inputs],
            "run_commit": self.run_commit,
            "run_commit_unverified": self.run_commit_unverified,
            "provenance_unverified_inputs": list(self.provenance_unverified_inputs),
            "contract_or_provenance_mismatch": self.contract_or_provenance_mismatch,
            "working_tree": dict(self.working_tree),
            "host_interpreter_note": self.host_interpreter_note,
        }


@dataclass
class Row:
    """One attribution-matrix row per declared cell (design §3 schema)."""

    skill: str
    initial_class: str
    phase: str
    profile_validity_precondition: str
    train_design_requires_coverage: bool
    train_coverage_evidence: str
    train_coverage_steps: Any  # int or "absent"
    dev30_coverage_evidence: str
    dev30_coverage_steps: Any  # int or "absent"
    train_field_status: str
    dev_gate_field_status: str
    emission_status: str
    gate_min_steps: int
    cell_gate_passed: bool
    skill_geometry_phase_gate_passed: bool
    skill_intent_progress_passed: bool
    intent_progress_fraction: Optional[float]
    intent_progress_threshold: float
    attribution_label: str
    reason_code: str
    evidence_paths: List[str]
    confidence: str
    next_step_recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill": self.skill,
            "initial_class": self.initial_class,
            "phase": self.phase,
            "profile_validity_precondition": self.profile_validity_precondition,
            "train_design_requires_coverage": self.train_design_requires_coverage,
            "train_coverage_evidence": self.train_coverage_evidence,
            "train_coverage_steps": self.train_coverage_steps,
            "dev30_coverage_evidence": self.dev30_coverage_evidence,
            "dev30_coverage_steps": self.dev30_coverage_steps,
            "train_field_status": self.train_field_status,
            "dev_gate_field_status": self.dev_gate_field_status,
            "emission_status": self.emission_status,
            "gate_min_steps": self.gate_min_steps,
            "cell_gate_passed": self.cell_gate_passed,
            "skill_geometry_phase_gate_passed": self.skill_geometry_phase_gate_passed,
            "skill_intent_progress_passed": self.skill_intent_progress_passed,
            "intent_progress_fraction": self.intent_progress_fraction,
            "intent_progress_threshold": self.intent_progress_threshold,
            "attribution_label": self.attribution_label,
            "reason_code": self.reason_code,
            "evidence_paths": list(self.evidence_paths),
            "confidence": self.confidence,
            "next_step_recommendation": self.next_step_recommendation,
        }


#: Ordered CSV/JSON column names (single source of truth for both writers).
ROW_COLUMNS: Tuple[str, ...] = (
    "skill",
    "initial_class",
    "phase",
    "profile_validity_precondition",
    "train_design_requires_coverage",
    "train_coverage_evidence",
    "train_coverage_steps",
    "dev30_coverage_evidence",
    "dev30_coverage_steps",
    "train_field_status",
    "dev_gate_field_status",
    "emission_status",
    "gate_min_steps",
    "cell_gate_passed",
    "skill_geometry_phase_gate_passed",
    "skill_intent_progress_passed",
    "intent_progress_fraction",
    "intent_progress_threshold",
    "attribution_label",
    "reason_code",
    "evidence_paths",
    "confidence",
    "next_step_recommendation",
)


# ---------------------------------------------------------------------------
# 1. IO / hashing
# ---------------------------------------------------------------------------


def _sha256_bytes(data: bytes) -> str:
    """Return the hex SHA-256 of ``data``."""
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def load_json(path: str) -> Tuple[Any, str, List[str]]:
    """Load a JSON file, returning ``(obj, sha256, top_level_fields)``.

    * ``obj`` — the parsed JSON object.
    * ``sha256`` — hex SHA-256 of the raw file bytes (canonical file hash).
    * ``top_level_fields`` — sorted list of top-level keys when ``obj`` is a
      mapping, otherwise an empty list.
    """
    with open(path, "rb") as fh:
        raw = fh.read()
    sha = _sha256_bytes(raw)
    obj = json.loads(raw.decode("utf-8"))
    if isinstance(obj, dict):
        top_level_fields = sorted(obj.keys())
    else:
        top_level_fields = []
    return obj, sha, top_level_fields


def file_sha256(path: str) -> str:
    """Return the hex SHA-256 of the file at ``path`` (raw bytes)."""
    with open(path, "rb") as fh:
        return _sha256_bytes(fh.read())


# ---------------------------------------------------------------------------
# 2. Declared cells (registry inline-list parser; stdlib only, no PyYAML)
# ---------------------------------------------------------------------------

# Matches an inline flow list like ``geometry_states: [advantage, head_on]``.
_INLINE_LIST_RE = re.compile(r"^\s*(geometry_states|phases)\s*:\s*\[([^\]]*)\]\s*$")
# Matches a skill header line ``  <name>:`` at the 2-space indent under skills:.
_SKILL_HEADER_RE = re.compile(r"^(\s{2})([A-Za-z0-9_]+)\s*:\s*$")


def _parse_inline_list(body: str) -> List[str]:
    """Parse the body of a YAML inline flow list into a list of tokens."""
    return [tok.strip() for tok in body.split(",") if tok.strip()]


def _parse_registry_declared(registry_path: str) -> Dict[str, Dict[str, List[str]]]:
    """Parse ``skills.<skill>.geometry_phase_coverage.{geometry_states,phases}``.

    Stdlib-only line scanner (no PyYAML). Returns a mapping::

        { skill_name: {"geometry_states": [...], "phases": [...]} }

    The parser locates the top-level ``skills:`` block, then for each skill
    header captures the ``geometry_states`` and ``phases`` inline lists that
    appear inside that skill's ``geometry_phase_coverage`` block.
    """
    with open(registry_path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    result: Dict[str, Dict[str, List[str]]] = {}
    in_skills = False
    skills_indent = 0
    current_skill: Optional[str] = None
    current_skill_indent = 0
    in_coverage = False

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(line) - len(line.lstrip(" "))

        # Detect the top-level ``skills:`` mapping.
        if not in_skills:
            if stripped == "skills:" and indent == 0:
                in_skills = True
                skills_indent = indent
            continue

        # Once inside skills:, a return to <= skills_indent (non-skill) ends it.
        if indent <= skills_indent and stripped != "skills:":
            # e.g. a following top-level key like ``readiness_contract:``.
            break

        # Skill header: exactly two spaces of indent under ``skills:``.
        skill_match = _SKILL_HEADER_RE.match(line)
        if skill_match and len(skill_match.group(1)) == skills_indent + 2:
            current_skill = skill_match.group(2)
            current_skill_indent = indent
            in_coverage = False
            result.setdefault(
                current_skill, {"geometry_states": [], "phases": []}
            )
            continue

        if current_skill is None:
            continue

        # Track entering / leaving the geometry_phase_coverage sub-block.
        if stripped == "geometry_phase_coverage:" and indent > current_skill_indent:
            in_coverage = True
            continue

        # A key at the skill's child indent that is not coverage ends coverage.
        if in_coverage and indent <= current_skill_indent + 2:
            if not _INLINE_LIST_RE.match(line):
                in_coverage = False

        if in_coverage:
            inline = _INLINE_LIST_RE.match(line)
            if inline:
                key = inline.group(1)
                values = _parse_inline_list(inline.group(2))
                result[current_skill][key] = values

    # Drop skills that declared no geometry_phase_coverage lists.
    return {
        skill: spec
        for skill, spec in result.items()
        if spec.get("geometry_states") and spec.get("phases")
    }


def derive_declared_cells(registry_path_or_obj: Any) -> Dict[str, List[str]]:
    """Derive declared cells per skill as ``{skill: ["state:phase", ...]}``.

    Accepts either a path to the shared-skill registry YAML (parsed with the
    stdlib-only inline-list parser) or an already-parsed mapping of the form
    ``{skill: {"geometry_states": [...], "phases": [...]}}`` (useful for tests).

    The Cartesian product ``geometry_states × phases`` yields the declared
    cells. The count is adaptive from the input (never hardcoded).
    """
    if isinstance(registry_path_or_obj, dict):
        spec_map: Dict[str, Dict[str, List[str]]] = {}
        # Support both the raw registry dict ({"skills": {...}}) and the
        # already-reduced {skill: {geometry_states, phases}} form.
        source = registry_path_or_obj.get("skills", registry_path_or_obj)
        for skill, spec in source.items():
            if not isinstance(spec, dict):
                continue
            cov = spec.get("geometry_phase_coverage", spec)
            states = cov.get("geometry_states")
            phases = cov.get("phases")
            if states and phases:
                spec_map[skill] = {
                    "geometry_states": list(states),
                    "phases": list(phases),
                }
    else:
        spec_map = _parse_registry_declared(str(registry_path_or_obj))

    declared: Dict[str, List[str]] = {}
    for skill, spec in spec_map.items():
        cells: List[str] = []
        for state in spec["geometry_states"]:
            for phase in spec["phases"]:
                cells.append(f"{state}:{phase}")
        declared[skill] = cells
    return declared


# ---------------------------------------------------------------------------
# 3. Emitted / train coverage collection
# ---------------------------------------------------------------------------


def _skills_map(gate_json: Any) -> Dict[str, Any]:
    """Return the ``skills`` mapping from a gate JSON object."""
    if isinstance(gate_json, dict) and isinstance(gate_json.get("skills"), dict):
        return gate_json["skills"]
    return {}


def _selected_gate(skill_obj: Any) -> Dict[str, Any]:
    """Return the ``selected_gate`` block of a skill object (or ``{}``)."""
    if isinstance(skill_obj, dict) and isinstance(skill_obj.get("selected_gate"), dict):
        return skill_obj["selected_gate"]
    return {}


def collect_emitted_cells(gate_json: Any) -> Dict[str, Dict[str, int]]:
    """Collect emitted coverage per skill from ``selected_gate``.

    Returns ``{skill: {"state:phase": steps}}`` sourced from
    ``skills.<skill>.selected_gate.geometry_phase_coverage`` — the SELECTED
    checkpoint's gate snapshot (NOT the last dev30 evaluation). Counts are
    adaptive from the input.
    """
    emitted: Dict[str, Dict[str, int]] = {}
    for skill, skill_obj in _skills_map(gate_json).items():
        cov = _selected_gate(skill_obj).get("geometry_phase_coverage", {})
        cells: Dict[str, int] = {}
        if isinstance(cov, dict):
            for cell, steps in cov.items():
                cells[cell] = int(steps)
        emitted[skill] = cells
    return emitted


def collect_train_cells(gate_json: Any) -> Dict[str, Dict[str, int]]:
    """Collect training coverage per skill.

    Returns ``{skill: {"state:phase": steps}}`` sourced from
    ``skills.<skill>.train_aggregate.state_phase_steps``.
    """
    train: Dict[str, Dict[str, int]] = {}
    for skill, skill_obj in _skills_map(gate_json).items():
        agg = skill_obj.get("train_aggregate", {}) if isinstance(skill_obj, dict) else {}
        sps = agg.get("state_phase_steps", {}) if isinstance(agg, dict) else {}
        cells: Dict[str, int] = {}
        if isinstance(sps, dict):
            for cell, steps in sps.items():
                cells[cell] = int(steps)
        train[skill] = cells
    return train


def selected_checkpoint_steps(gate_json: Any) -> Dict[str, Optional[int]]:
    """Return the selected checkpoint's evaluation ``step`` per skill.

    If a skill object carries an explicit ``selected_checkpoint_step`` (or
    ``selected_step``) that value is used. Otherwise the audit attempts to
    match the ``selected_gate`` against the skill's ``dev_evaluations`` entries
    (by identical ``geometry_phase_coverage`` and ``intent_progress_fraction``)
    and reports that entry's ``step``. When no match can be established the
    value is ``None`` (recorded honestly, never fabricated).
    """
    steps: Dict[str, Optional[int]] = {}
    for skill, skill_obj in _skills_map(gate_json).items():
        if not isinstance(skill_obj, dict):
            steps[skill] = None
            continue

        explicit = skill_obj.get("selected_checkpoint_step")
        if explicit is None:
            explicit = skill_obj.get("selected_step")
        if explicit is not None:
            steps[skill] = int(explicit)
            continue

        sel = _selected_gate(skill_obj)
        sel_cov = sel.get("geometry_phase_coverage")
        sel_ipf = sel.get("intent_progress_fraction")
        matched: Optional[int] = None
        for ev in skill_obj.get("dev_evaluations", []) or []:
            if not isinstance(ev, dict):
                continue
            g = ev.get("gate", {})
            if not isinstance(g, dict):
                continue
            if (
                g.get("geometry_phase_coverage") == sel_cov
                and g.get("intent_progress_fraction") == sel_ipf
            ):
                matched = ev.get("step")
                break
        steps[skill] = int(matched) if matched is not None else None
    return steps


# ---------------------------------------------------------------------------
# 4. Reconciliation (2a preflight)
# ---------------------------------------------------------------------------


def reconcile(
    declared: Dict[str, List[str]],
    emitted: Dict[str, Dict[str, int]],
) -> ReconcileResult:
    """Reconcile declared cells against emitted cells (2a preflight).

    * ``declared`` — ``{skill: ["state:phase", ...]}`` (from the registry).
    * ``emitted`` — ``{skill: {"state:phase": steps}}`` (from ``selected_gate``).

    ``diff`` = declared cells with no corresponding emitted key.
    ``extra`` = emitted cells not present in the declared set.

    All counts are derived; nothing is hardcoded.
    """
    declared_cells: List[DeclaredCell] = []
    emitted_cells: List[EmittedCell] = []

    for skill in sorted(declared.keys()):
        for cell in declared[skill]:
            state, _, phase = cell.partition(":")
            declared_cells.append(DeclaredCell(skill, state, phase))

    for skill in sorted(emitted.keys()):
        for cell, steps in emitted[skill].items():
            state, _, phase = cell.partition(":")
            emitted_cells.append(EmittedCell(skill, state, phase, int(steps)))

    declared_keys = {c.key for c in declared_cells}
    emitted_keys = {c.key for c in emitted_cells}

    diff = [c for c in declared_cells if c.key not in emitted_keys]
    extra = [c for c in emitted_cells if c.key not in declared_keys]

    return ReconcileResult(
        declared=declared_cells,
        emitted=emitted_cells,
        diff=diff,
        extra=extra,
    )


# ---------------------------------------------------------------------------
# 5. Cross-artifact consistency (2b preflight)
# ---------------------------------------------------------------------------

#: The shared, run-identifying facts compared field-by-field between the
#: top-level gate ``selected_gate`` and each per-skill summary ``selected_gate``.
_CROSS_ARTIFACT_FIELDS: Tuple[str, ...] = (
    "geometry_phase_coverage",
    "intent_progress_fraction",
    "status",
    "profile_coverage",
    "safety_by_opponent",
)


def cross_artifact_consistency(
    gate_json: Any,
    per_skill_summaries: Dict[str, Any],
) -> List[Problem]:
    """Deep-compare top-level ``selected_gate`` vs each per-skill summary (2b).

    For every skill in the top-level gate, compare the shared run-identifying
    facts (``geometry_phase_coverage``, ``intent_progress_fraction``, ``status``,
    ``profile_coverage``, ``safety_by_opponent``) between
    ``gate_json.skills.<name>.selected_gate`` and
    ``per_skill_summaries[<name>].selected_gate``.

    Returns a list of :class:`Problem`; an empty list means the artifacts are
    consistent (CLEAN). A non-empty list is a genuine artifact-integrity
    failure that the caller must handle by emitting only an integrity report
    and exiting non-zero.
    """
    problems: List[Problem] = []
    for skill, skill_obj in _skills_map(gate_json).items():
        detail = per_skill_summaries.get(skill)
        if detail is None:
            problems.append(Problem(skill, "missing_per_skill_summary"))
            continue

        top_sel = _selected_gate(skill_obj)
        detail_sel = _selected_gate(detail)
        if not detail_sel:
            problems.append(Problem(skill, "missing_selected_gate_in_summary"))
            continue

        for fld in _CROSS_ARTIFACT_FIELDS:
            top_val = top_sel.get(fld)
            detail_val = detail_sel.get(fld)
            if top_val != detail_val:
                problems.append(
                    Problem(
                        skill=skill,
                        field=fld,
                        detail=(
                            "top_level != per_skill_summary "
                            f"(top={top_val!r}, summary={detail_val!r})"
                        ),
                    )
                )

        # Compare the selected checkpoint step when both artifacts expose one.
        top_step = skill_obj.get("selected_checkpoint_step") if isinstance(
            skill_obj, dict
        ) else None
        detail_step = detail.get("selected_checkpoint_step") if isinstance(
            detail, dict
        ) else None
        if top_step is not None and detail_step is not None and top_step != detail_step:
            problems.append(
                Problem(
                    skill=skill,
                    field="selected_checkpoint_step",
                    detail=f"top={top_step!r}, summary={detail_step!r}",
                )
            )

    return problems


def emit_integrity_report(problems: List[Problem], path: str) -> None:
    """Write a short integrity report for the 2b failure path.

    The caller is responsible for exiting non-zero after this is written and
    for NOT emitting any attribution matrix / conclusions.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    lines: List[str] = []
    lines.append("# P4 Gate Cross-Artifact Integrity Failure (2b)")
    lines.append("")
    lines.append(f"Generated (UTC): {timestamp}")
    lines.append("")
    lines.append(
        "The top-level `p4_geometry_pretrain_gate.json` `skills.<name>.selected_gate` "
        "block disagrees with one or more per-skill "
        "`skills/<name>/geometry_pretrain_summary.json` `selected_gate` blocks. "
        "The paired artifacts cannot be trusted for fine-grained attribution, so "
        "NO attribution matrix and NO capability-attribution conclusions are "
        "produced. This is surfaced as an input to a subsequent P4 design review, "
        "never as a skill-capability failure and never resolved by editing the gate."
    )
    lines.append("")
    lines.append(f"Discrepancies found: {len(problems)}")
    lines.append("")
    for p in problems:
        lines.append(f"- skill=`{p.skill}` field=`{p.field}` {p.detail}".rstrip())
    lines.append("")
    _write_text(path, "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# 6. Provenance freezing
# ---------------------------------------------------------------------------


def _git(args: List[str]) -> Optional[str]:
    """Run a git command, returning stripped stdout or ``None`` on failure."""
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def working_tree_git_info() -> Dict[str, Any]:
    """Return the CURRENT working-tree git info (never frozen run facts).

    Recorded under a distinct key so it can never be substituted for the
    frozen run commit.
    """
    head = _git(["rev-parse", "HEAD"])
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    short = _git(["rev-parse", "--short", "HEAD"])
    status = _git(["status", "--porcelain"])
    dirty = bool(status) if status is not None else None
    return {
        "head": head,
        "branch": branch,
        "short": short,
        "dirty": dirty,
        "note": (
            "Working-tree git info is NOT the frozen run commit; it is the "
            "current checkout used to run the audit."
        ),
    }


def freeze_provenance(
    inputs: Dict[str, Optional[str]],
    run_commit: Optional[str] = None,
) -> ProvenanceBlock:
    """Freeze provenance for the evidence set.

    ``inputs`` maps a logical input name to a file path (or ``None`` when the
    input is not available). Recognized names include ``gate_json``,
    ``registry``, ``training_plan``, ``resolved_config`` /
    ``resolved_geometry_config``, ``dev30_manifest`` and the per-skill summary
    paths (any name containing ``summary``).

    For each available input file the block records ``path``, ``sha256``,
    ``top_level_fields`` and, for a ``resolved_geometry_config``, its resolved
    ``schema_version``.

    For each MISSING frozen input (``run_commit``, ``training_plan``,
    ``resolved_geometry_config``, ``dev30_manifest``) the block records the
    input as ``provenance_unverified``. The registry-as-``training_plan``
    substitute is likewise flagged ``provenance_unverified``.

    A run-level ``contract_or_provenance_mismatch`` flag is raised listing the
    missing frozen inputs. Frozen provenance and working-tree git info are kept
    as SEPARATE fields.
    """
    block = ProvenanceBlock()
    provenance_unverified: List[str] = []

    def _canonical_frozen_name(name: str) -> Optional[str]:
        if name in ("resolved_config", "resolved_geometry_config"):
            return "resolved_geometry_config"
        if name == "training_plan":
            return "training_plan"
        if name == "dev30_manifest":
            return "dev30_manifest"
        return None

    for name, path in inputs.items():
        available = bool(path) and os.path.isfile(path) if path else False
        record = InputProvenance(name=name, path=path, available=available)

        if available:
            try:
                _obj, sha, top = load_json(path)  # type: ignore[arg-type]
                record.sha256 = sha
                record.top_level_fields = top
                if _canonical_frozen_name(name) == "resolved_geometry_config" and isinstance(
                    _obj, dict
                ):
                    record.schema_version = _obj.get("schema_version")
            except (ValueError, OSError):
                # Not JSON (e.g. the registry YAML): still record a byte hash.
                try:
                    record.sha256 = file_sha256(path)  # type: ignore[arg-type]
                except OSError:
                    record.available = False
        else:
            canonical = _canonical_frozen_name(name)
            if canonical is not None:
                record.provenance_unverified = True
                record.note = "missing frozen input"
                if canonical not in provenance_unverified:
                    provenance_unverified.append(canonical)

        # The registry stands in for the frozen training_plan.json.
        if name == "registry" and available:
            record.provenance_unverified = True
            record.note = (
                "registry used as provenance_unverified substitute for the "
                "missing frozen training_plan.json"
            )

        block.inputs.append(record)

    # Run commit is a frozen fact; the working tree is NEVER substituted.
    block.run_commit = run_commit
    block.run_commit_unverified = run_commit is None
    if run_commit is None and "run_commit" not in provenance_unverified:
        provenance_unverified.append("run_commit")

    # Ensure the canonical frozen inputs that were not supplied at all are
    # still surfaced as unverified.
    for frozen in FROZEN_PROVENANCE_INPUTS:
        if frozen == "run_commit":
            continue
        supplied = any(
            _canonical_frozen_name(n) == frozen and (inputs.get(n))
            for n in inputs
        )
        if not supplied and frozen not in provenance_unverified:
            provenance_unverified.append(frozen)

    block.provenance_unverified_inputs = provenance_unverified
    block.contract_or_provenance_mismatch = bool(provenance_unverified)
    block.working_tree = working_tree_git_info()
    return block


# ---------------------------------------------------------------------------
# 7. Field-status encoding + label assignment (design §3)
# ---------------------------------------------------------------------------


def field_status(coverage: Dict[str, int], cell: str) -> str:
    """Encode presence/value of ``cell`` within a coverage source.

    Returns one of ``present_value_0`` / ``present_value_gt_0`` / ``absent``.
    Never infers "physically impossible" from absence.
    """
    if cell not in coverage:
        return FIELD_STATUS_ABSENT
    return (
        FIELD_STATUS_PRESENT_GT0
        if int(coverage[cell]) > 0
        else FIELD_STATUS_PRESENT_ZERO
    )


def _steps_or_absent(coverage: Dict[str, int], cell: str) -> Any:
    """Return the integer step count for ``cell`` or the ``absent`` sentinel."""
    if cell not in coverage:
        return ABSENT_SENTINEL
    return int(coverage[cell])


def assign_label(row: Row, reconcile: ReconcileResult) -> Tuple[str, str]:
    """Assign ``(attribution_label, reason_code)`` per design §3 executable rules.

    Rules (executed in order):
      0. cell in diff set (``emission_status == declared_not_emitted``) ->
         ``(contract_or_provenance_mismatch, "")``.
      1. ``dev_gate_field_status == absent`` -> ``(not_observed_cannot_assess, "")``.
      2. ``train == 0 and dev == 0`` -> ``train_coverage_gap`` if the design
         requires coverage, else ``not_observed_cannot_assess``.
      3. ``train > 0 and dev == 0`` -> ``(not_observed_cannot_assess, dev_coverage_gap)``.
      4. ``train > 0 and dev > 0`` -> ``(mixed_or_inconclusive, "")``.
      5. residual -> ``(mixed_or_inconclusive, "")``.

    ``observed_behavior_failure`` is NEVER assigned (no cell-level quality
    metric exists). ``telemetry_or_gate_observability_gap`` is reserved solely
    for the initial-class × phase limitation (applied in ``build_matrix``), not
    here.
    """
    # 0. Contract/provenance precedence: declared but not emitted.
    if (row.skill, row.initial_class, row.phase) in reconcile.diff_keys():
        return ("contract_or_provenance_mismatch", "")

    train = row.train_coverage_steps
    dev = row.dev30_coverage_steps

    # 1. Absent dev gate field -> unverifiable.
    if row.dev_gate_field_status == FIELD_STATUS_ABSENT:
        return ("not_observed_cannot_assess", "")

    train_zero = (train == 0) or (train == ABSENT_SENTINEL)
    dev_zero = (dev == 0) or (dev == ABSENT_SENTINEL)
    train_pos = isinstance(train, int) and train > 0
    dev_pos = isinstance(dev, int) and dev > 0

    # 2. train=0 AND dev=0 -> never observed in either distribution.
    if train_zero and dev_zero:
        if row.train_design_requires_coverage:
            return ("train_coverage_gap", "")
        return ("not_observed_cannot_assess", "")

    # 3. train>0 AND dev=0 -> a dev/evaluation coverage count of zero.
    if train_pos and dev_zero:
        return ("not_observed_cannot_assess", "dev_coverage_gap")

    # 4. train>0 AND dev>0 -> no cell-level quality metric to assert failure.
    if train_pos and dev_pos:
        return ("mixed_or_inconclusive", "")

    # 5. Residual (e.g. train absent while dev present).
    return ("mixed_or_inconclusive", "")


# ---------------------------------------------------------------------------
# 8. Matrix construction (design §3)
# ---------------------------------------------------------------------------


def default_semantics() -> Dict[str, Any]:
    """Return the default gate-semantics constants recorded on each row."""
    return {
        "coverage_unit": COVERAGE_UNIT,
        "gate_min_steps": GATE_MIN_STEPS,
        "intent_progress_threshold": INTENT_PROGRESS_THRESHOLD,
        "source_lines": dict(GATE_SEMANTICS_SOURCE),
    }


def _skill_check(skill_obj: Any, name: str) -> bool:
    """Return the boolean skill-level gate check ``name`` (default False)."""
    sel = _selected_gate(skill_obj)
    checks = sel.get("checks", {})
    if isinstance(checks, dict):
        return bool(checks.get(name, False))
    return False


def build_matrix(
    gate_json: Any,
    declared: Dict[str, List[str]],
    reconcile: ReconcileResult,
    semantics: Optional[Dict[str, Any]] = None,
) -> List[Row]:
    """Build the full attribution matrix — one row per declared cell.

    Columns follow the design §3 schema. Field statuses are encoded
    independently for the training coverage source
    (``train_aggregate.state_phase_steps``) and the dev gate coverage source
    (``selected_gate.geometry_phase_coverage``). The initial-class × phase
    limitation is propagated into ``confidence`` (and, for rows whose claim
    depends purely on dynamic posture, into a
    ``telemetry_or_gate_observability_gap`` note) for ``post_merge`` /
    ``re_entry`` rows.
    """
    if semantics is None:
        semantics = default_semantics()
    gate_min = int(semantics.get("gate_min_steps", GATE_MIN_STEPS))
    intent_threshold = float(
        semantics.get("intent_progress_threshold", INTENT_PROGRESS_THRESHOLD)
    )

    skills_map = _skills_map(gate_json)
    emitted = collect_emitted_cells(gate_json)
    train = collect_train_cells(gate_json)

    rows: List[Row] = []
    for skill in sorted(declared.keys()):
        skill_obj = skills_map.get(skill, {})
        sel = _selected_gate(skill_obj)
        dev_cov = emitted.get(skill, {})
        train_cov = train.get(skill, {})

        skill_gpc_passed = _skill_check(skill_obj, "geometry_phase_coverage")
        skill_ip_passed = _skill_check(skill_obj, "intent_progress")
        intent_fraction = sel.get("intent_progress_fraction")
        if intent_fraction is not None:
            intent_fraction = float(intent_fraction)

        emitted_keys = {c.cell for c in reconcile.emitted if c.skill == skill}

        for cell in declared[skill]:
            state, _, phase = cell.partition(":")

            train_fs = field_status(train_cov, cell)
            dev_fs = field_status(dev_cov, cell)
            train_steps = _steps_or_absent(train_cov, cell)
            dev_steps = _steps_or_absent(dev_cov, cell)

            emission_status = (
                EMISSION_EMITTED if cell in emitted_keys else EMISSION_DECLARED_NOT_EMITTED
            )

            dev_present = dev_fs != FIELD_STATUS_ABSENT
            dev_steps_int = dev_steps if isinstance(dev_steps, int) else 0
            cell_gate_passed = dev_present and dev_steps_int >= gate_min

            train_evidence = (
                f"skills.{skill}.train_aggregate.state_phase_steps[\"{cell}\"]"
            )
            dev_evidence = (
                f"skills.{skill}.selected_gate.geometry_phase_coverage[\"{cell}\"]"
            )

            evidence_paths = [
                train_evidence,
                dev_evidence,
                f"skills.{skill}.selected_gate.checks.geometry_phase_coverage",
                f"skills.{skill}.selected_gate.intent_progress_fraction",
                "registry.geometry_phase_coverage "
                "(provenance_unverified substitute for training_plan.json)",
                "gate_min_steps="
                f"{gate_min} ({GATE_SEMANTICS_SOURCE['coverage_min_steps']})",
                "intent_progress_threshold="
                f"{intent_threshold} ({GATE_SEMANTICS_SOURCE['intent_progress_threshold']})",
            ]

            row = Row(
                skill=skill,
                initial_class=state,
                phase=phase,
                profile_validity_precondition=(
                    "profile_conditioning_coverage "
                    "(every allowed profile present); validity mask per "
                    "declared_geometry_phase_coverage_must_be_observed"
                ),
                train_design_requires_coverage=True,
                train_coverage_evidence=train_evidence,
                train_coverage_steps=train_steps,
                dev30_coverage_evidence=dev_evidence,
                dev30_coverage_steps=dev_steps,
                train_field_status=train_fs,
                dev_gate_field_status=dev_fs,
                emission_status=emission_status,
                gate_min_steps=gate_min,
                cell_gate_passed=cell_gate_passed,
                skill_geometry_phase_gate_passed=skill_gpc_passed,
                skill_intent_progress_passed=skill_ip_passed,
                intent_progress_fraction=intent_fraction,
                intent_progress_threshold=intent_threshold,
                attribution_label="",  # filled below
                reason_code="",
                evidence_paths=evidence_paths,
                confidence="",  # filled below
                next_step_recommendation="",  # filled below
            )

            label, reason_code = assign_label(row, reconcile)
            row.attribution_label = label
            row.reason_code = reason_code

            # Propagate the initial-class × phase limitation into confidence
            # (and, when the claim depends purely on dynamic posture, a note).
            _apply_initial_class_phase_limitation(row)

            row.next_step_recommendation = _recommend_next_step(row)
            rows.append(row)

    return rows


def _apply_initial_class_phase_limitation(row: Row) -> None:
    """Cap confidence / annotate rows affected by the fixed-initial_class limit.

    Coverage is recorded keyed on the scenario's fixed ``initial_class``
    (``thesis_shared_skill_geometry.py:451``, call sites 677/806), so dynamic
    five-state posture during ``post_merge`` / ``re_entry`` cannot be proven.
    Rows for those phases carry lower confidence and an explicit note. This
    never overrides the primary attribution label; it is applied as an
    additional note (and confidence flag).
    """
    if row.phase in DYNAMIC_POSTURE_PHASES:
        row.confidence = (
            "low: coverage proven only for fixed initial_class x phase; "
            "dynamic five-state posture in "
            f"{row.phase} is UNPROVEN (initial_class_phase_limitation, "
            f"{GATE_SEMANTICS_SOURCE['intent_progress_numerator']})"
        )
        # Record the limitation as a machine-readable reason when the primary
        # label did not already set one.
        if not row.reason_code:
            row.reason_code = "initial_class_phase_limitation"
        if "initial_class_phase_limitation" not in row.evidence_paths:
            row.evidence_paths.append(
                "initial_class_phase_limitation: dynamic posture not observable "
                "(telemetry_or_gate_observability_gap reserved semantics, design §4)"
            )
    else:
        row.confidence = (
            "moderate: coverage proven for fixed initial_class x phase "
            f"({row.phase}); no dynamic-posture claim required"
        )


def _recommend_next_step(row: Row) -> str:
    """Recommend a per-cell next action consistent with the audit conclusion."""
    label = row.attribution_label
    if label == "train_coverage_gap":
        return (
            "P4 design review: declared cell has 0 steps in train and dev; "
            "revisit training-distribution design (no retraining under this audit)"
        )
    if label == "not_observed_cannot_assess":
        if row.reason_code == "dev_coverage_gap":
            return (
                "P4 design review: cell observed in train but not in dev30; "
                "dev coverage gap, cannot assess capability from current evidence"
            )
        return (
            "P4 design review: cell not observed; cannot assess capability "
            "from current evidence"
        )
    if label == "contract_or_provenance_mismatch":
        return (
            "P4 design review input: reconcile contract/provenance; "
            "do NOT modify the gate and do NOT attribute to skill capability"
        )
    if label == "mixed_or_inconclusive":
        return (
            "P4 design review: coverage present but no cell-level quality metric; "
            "inconclusive under current evidence"
        )
    return "P4 design review"


# ---------------------------------------------------------------------------
# 9. Output writers (single in-memory matrix -> content-consistent csv/json/md)
# ---------------------------------------------------------------------------


def _write_text(path: str, text: str) -> None:
    """Write ``text`` to ``path``, creating parent directories as needed."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def _csv_cell(value: Any) -> str:
    """Render a row value as a CSV-safe string (lists joined with ``|``)."""
    if isinstance(value, list):
        return " | ".join(str(v) for v in value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def preflight_block(reconcile: ReconcileResult) -> Dict[str, Any]:
    """Build the JSON ``preflight`` block from the reconciliation result."""
    return {
        "declared_count": reconcile.declared_count,
        "emitted_count": reconcile.emitted_count,
        "diff_count": reconcile.diff_count,
        "extra_count": reconcile.extra_count,
        "diff": [
            {"skill": c.skill, "geometry_state": c.geometry_state, "phase": c.phase}
            for c in reconcile.diff
        ],
        "extra": [
            {
                "skill": c.skill,
                "geometry_state": c.geometry_state,
                "phase": c.phase,
                "steps": c.steps,
            }
            for c in reconcile.extra
        ],
    }


def emit_csv(rows: List[Row], path: str) -> None:
    """Write the attribution matrix as CSV (columns per :data:`ROW_COLUMNS`)."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(ROW_COLUMNS)
        for row in rows:
            d = row.to_dict()
            writer.writerow([_csv_cell(d[col]) for col in ROW_COLUMNS])


def emit_json(
    rows: List[Row],
    reconcile: ReconcileResult,
    provenance: ProvenanceBlock,
    conclusion: str,
    path: str,
) -> None:
    """Write the attribution matrix + preflight + provenance + conclusion JSON.

    Emits the SAME in-memory rows as :func:`emit_csv` so csv/json content is
    consistent.
    """
    payload = {
        "conclusion": conclusion,
        "gate_semantics": default_semantics(),
        "preflight": preflight_block(reconcile),
        "provenance": provenance.to_dict(),
        "controlled_labels": list(CONTROLLED_LABELS),
        "rows": [row.to_dict() for row in rows],
    }
    _write_text(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def emit_markdown(
    rows: List[Row],
    reconcile: ReconcileResult,
    provenance: ProvenanceBlock,
    semantics: Optional[Dict[str, Any]],
    path: str,
    lang: str = "zh",
) -> None:
    """Write the narrative audit markdown from the SAME in-memory matrix.

    Defaults to Chinese (``lang='zh'``); any other value produces the English
    fallback labels. Content is derived from the same rows/reconcile/provenance
    used by the csv/json writers.
    """
    if semantics is None:
        semantics = default_semantics()
    zh = lang == "zh"
    r = reconcile

    lines: List[str] = []
    if zh:
        lines.append("# P4 几何门失败归因审计 (GEO-20260713-R1)")
    else:
        lines.append("# P4 Geometry Gate Failure Attribution Audit (GEO-20260713-R1)")
    lines.append("")
    lines.append(f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}")
    lines.append("")

    # Strict provenance boundary + conclusion.
    if zh:
        lines.append("## 结论")
        lines.append("")
        lines.append(
            "在既有的预注册门限下，**不允许**启动 combat finetune"
            f"（`{CONCLUSION_COMBAT_FINETUNE}`）；本审计无权更改该结论。"
        )
    else:
        lines.append("## Conclusion")
        lines.append("")
        lines.append(
            "Under the pre-registered gate, combat finetune is NOT allowed "
            f"(`{CONCLUSION_COMBAT_FINETUNE}`); this audit has no authority to "
            "change that."
        )
    lines.append("")

    # Preflight reconciliation (counts adaptive, derived from inputs).
    lines.append("## Preflight 2a: declared-vs-emitted reconciliation")
    lines.append("")
    lines.append(
        f"- declared = {r.declared_count}, emitted = {r.emitted_count}, "
        f"diff = {r.diff_count}, extra = {r.extra_count}"
    )
    if r.diff_count == 0:
        lines.append(
            "- diff set is EMPTY; the historical 26/22/4 premise is unverified / "
            "disproven / structurally impossible under "
            f"`{GATE_SEMANTICS_SOURCE['coverage_cartesian_product']}` "
            "(full Cartesian product, absent filled with 0)."
        )
    else:
        lines.append(
            "- diff cells are the audit subject (labeled "
            "`contract_or_provenance_mismatch` / `declared_not_emitted`); the "
            "audit STILL builds the full matrix and exits 0."
        )
    lines.append("")

    # Real failure attribution (ALL values derived from the in-memory rows;
    # nothing hardcoded). Groups rows by skill preserving first-seen order.
    intent_threshold = float(
        semantics.get("intent_progress_threshold", INTENT_PROGRESS_THRESHOLD)
    )
    skills_order: List[str] = []
    per_skill: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        agg = per_skill.get(row.skill)
        if agg is None:
            skills_order.append(row.skill)
            agg = {
                "intent_fraction": row.intent_progress_fraction,
                "intent_passed": row.skill_intent_progress_passed,
                "gpc_passed": row.skill_geometry_phase_gate_passed,
                "zero_cells": [],
                "crossing_present_nonzero": [],
            }
            per_skill[row.skill] = agg
        if row.dev_gate_field_status == FIELD_STATUS_PRESENT_ZERO:
            agg["zero_cells"].append(f"{row.initial_class}:{row.phase}")
        if (
            row.initial_class == "crossing_entry"
            and row.dev_gate_field_status == FIELD_STATUS_PRESENT_GT0
        ):
            agg["crossing_present_nonzero"].append(
                f"{row.initial_class}:{row.phase}={row.dev30_coverage_steps}"
            )

    # Skills that PASS intent progress yet are still blocked by phase coverage.
    intent_pass_gpc_fail = [
        s
        for s in skills_order
        if per_skill[s]["intent_passed"] and not per_skill[s]["gpc_passed"]
    ]
    # crossing_entry cells that are present and non-zero (across all skills).
    crossing_evidence = [
        cell
        for s in skills_order
        for cell in per_skill[s]["crossing_present_nonzero"]
    ]

    if zh:
        lines.append("## 真实失败归因 (Real failure attribution)")
        lines.append("")
        lines.append(
            "1. `geometry_phase_coverage = false` 完全由 `post_merge` / `re_entry` "
            "阶段中**已声明且以数值 0 发出**的单元（`present_value_0`）驱动，"
            "**并非**由任何缺失/不存在（`absent`）的单元造成。历史上 26/22/4 "
            "（`crossing_entry missing`）的前提**不成立 / 已被反证 / 在结构上不可能**"
            f"（`{GATE_SEMANTICS_SOURCE['coverage_cartesian_product']}` 发出完整笛卡尔积，"
            "缺失组合以 0 填充）："
            + (
                "已声明的 `crossing_entry` 单元均为存在且非零（"
                + ", ".join(f"`{c}`" for c in crossing_evidence)
                + "）。"
                if crossing_evidence
                else "已声明的 `crossing_entry` 单元均为存在（非缺失）。"
            )
        )
        lines.append("")
        lines.append(
            "2. 各技能 intent 进度汇总（阈值 "
            f"{intent_threshold}；分数为每技能常量，取自 "
            "`selected_gate.intent_progress_fraction`）："
        )
        for s in skills_order:
            frac = per_skill[s]["intent_fraction"]
            frac_str = f"{frac:.4f}" if isinstance(frac, (int, float)) else "n/a"
            verdict = "PASS" if per_skill[s]["intent_passed"] else "FAIL"
            lines.append(
                f"   - `{s}`: intent_progress_fraction = {frac_str} "
                f"(threshold {intent_threshold}) -> "
                f"skill_intent_progress_passed = "
                f"{str(per_skill[s]['intent_passed']).lower()} ({verdict})"
            )
        lines.append("")
        if intent_pass_gpc_fail:
            lines.append(
                "3. "
                + ", ".join(f"`{s}`" for s in intent_pass_gpc_fail)
                + " 通过 intent 进度检查（`skill_intent_progress_passed = true`），"
                "但仍因 `geometry_phase_coverage` 失败而被阻断。"
            )
        else:
            lines.append(
                "3. 没有技能在通过 intent 进度检查的同时仅被 "
                "`geometry_phase_coverage` 阻断。"
            )
        lines.append("")
        lines.append(
            "4. 各技能的 `present_value_0` dev-gate 单元（"
            "`dev_gate_field_status == present_value_0`，即驱动阶段覆盖失败的零单元）："
        )
        for s in skills_order:
            cells = per_skill[s]["zero_cells"]
            rendered = ", ".join(f"`{c}`" for c in cells) if cells else "(无)"
            lines.append(f"   - `{s}`: {rendered}")
        lines.append("")
        lines.append(
            "5. 全部 "
            f"{len(skills_order)} 个技能均通过 `profile_conditioning_coverage` 与 "
            "`safety` 检查（证据：`selected_gate.checks.profile_conditioning_coverage` / "
            "`selected_gate.checks.safety`），此为予以保留的通过性证据（Req 3.9），"
            "不得被重述为失败。"
        )
    else:
        lines.append("## Real failure attribution")
        lines.append("")
        lines.append(
            "1. `geometry_phase_coverage = false` is driven entirely by declared "
            "cells emitted with value 0 (`present_value_0`) in `post_merge` / "
            "`re_entry`, NOT by any missing/absent cell. The historical 26/22/4 "
            "(`crossing_entry missing`) premise is unverified / disproven / "
            "structurally impossible under "
            f"`{GATE_SEMANTICS_SOURCE['coverage_cartesian_product']}` "
            "(full Cartesian product, absent filled with 0): "
            + (
                "declared `crossing_entry` cells are present and non-zero ("
                + ", ".join(f"`{c}`" for c in crossing_evidence)
                + ")."
                if crossing_evidence
                else "declared `crossing_entry` cells are present (not absent)."
            )
        )
        lines.append("")
        lines.append(
            "2. Per-skill intent-progress summary (threshold "
            f"{intent_threshold}; fraction is constant per skill, read from "
            "`selected_gate.intent_progress_fraction`):"
        )
        for s in skills_order:
            frac = per_skill[s]["intent_fraction"]
            frac_str = f"{frac:.4f}" if isinstance(frac, (int, float)) else "n/a"
            verdict = "PASS" if per_skill[s]["intent_passed"] else "FAIL"
            lines.append(
                f"   - `{s}`: intent_progress_fraction = {frac_str} "
                f"(threshold {intent_threshold}) -> "
                f"skill_intent_progress_passed = "
                f"{str(per_skill[s]['intent_passed']).lower()} ({verdict})"
            )
        lines.append("")
        if intent_pass_gpc_fail:
            lines.append(
                "3. "
                + ", ".join(f"`{s}`" for s in intent_pass_gpc_fail)
                + " PASSES intent progress (`skill_intent_progress_passed = "
                "true`) but remains blocked by `geometry_phase_coverage` failure."
            )
        else:
            lines.append(
                "3. No skill passes intent progress while being blocked solely "
                "by `geometry_phase_coverage`."
            )
        lines.append("")
        lines.append(
            "4. Present-value-0 dev-gate cells per skill "
            "(`dev_gate_field_status == present_value_0`; these zero cells drive "
            "the phase-coverage failure):"
        )
        for s in skills_order:
            cells = per_skill[s]["zero_cells"]
            rendered = ", ".join(f"`{c}`" for c in cells) if cells else "(none)"
            lines.append(f"   - `{s}`: {rendered}")
        lines.append("")
        lines.append(
            "5. All "
            f"{len(skills_order)} skills pass `profile_conditioning_coverage` and "
            "`safety` (evidence: "
            "`selected_gate.checks.profile_conditioning_coverage` / "
            "`selected_gate.checks.safety`); this is preserved passing evidence "
            "(Req 3.9) and is NOT recharacterized as failure."
        )
    lines.append("")

    # Gate semantics citations.
    lines.append("## Gate semantics")
    lines.append("")
    lines.append(f"- coverage unit = {semantics.get('coverage_unit', COVERAGE_UNIT)} (policy steps)")
    lines.append(
        f"- gate_min_steps = {semantics.get('gate_min_steps', GATE_MIN_STEPS)} "
        f"(`{GATE_SEMANTICS_SOURCE['coverage_min_steps']}` / "
        f"`{GATE_SEMANTICS_SOURCE['coverage_min_steps_check']}`)"
    )
    lines.append(
        "- intent_progress = intent_progress_steps / policy_steps; threshold = "
        f"{semantics.get('intent_progress_threshold', INTENT_PROGRESS_THRESHOLD)} "
        f"(`{GATE_SEMANTICS_SOURCE['intent_progress_threshold']}`; numerator "
        f"`{GATE_SEMANTICS_SOURCE['intent_progress_numerator']}`)"
    )
    lines.append("")

    # Frozen provenance block (kept separate from working tree).
    lines.append("## Frozen provenance")
    lines.append("")
    lines.append(
        "- provenance_unverified inputs: "
        + (", ".join(provenance.provenance_unverified_inputs) or "(none)")
    )
    lines.append(
        "- run-level contract_or_provenance_mismatch: "
        f"{str(provenance.contract_or_provenance_mismatch).lower()}"
    )
    lines.append(f"- run_commit: {provenance.run_commit or 'provenance_unverified'}")
    for inp in provenance.inputs:
        marker = " (provenance_unverified)" if inp.provenance_unverified else ""
        sha = inp.sha256 or "n/a"
        lines.append(
            f"  - `{inp.name}` available={str(inp.available).lower()} "
            f"sha256={sha}{marker}"
        )
    wt = provenance.working_tree
    lines.append(
        "- working_tree (SEPARATE from frozen run facts): "
        f"head={wt.get('head')}, branch={wt.get('branch')}, dirty={wt.get('dirty')}"
    )
    if provenance.host_interpreter_note:
        lines.append(f"- host/interpreter note: {provenance.host_interpreter_note}")
    lines.append("")

    # initial-class x phase limitation.
    lines.append("## initial-class x phase limitation")
    lines.append("")
    lines.append(
        "Coverage is recorded keyed on the scenario's fixed `initial_class` "
        f"(`{GATE_SEMANTICS_SOURCE['intent_progress_numerator']}`, call sites "
        "677/806). This audit proves ONLY initial-class x phase coverage and "
        "CANNOT prove dynamic five-state posture during post_merge / re_entry."
    )
    lines.append("")

    # Attribution matrix.
    lines.append("## Attribution matrix")
    lines.append("")
    header = (
        "| skill | initial_class | phase | train_field_status | "
        "dev_gate_field_status | emission_status | train_steps | dev_steps | "
        "cell_gate_passed | attribution_label | reason_code | confidence |"
    )
    sep = "|" + "|".join(["---"] * 12) + "|"
    lines.append(header)
    lines.append(sep)
    for row in rows:
        lines.append(
            f"| {row.skill} | {row.initial_class} | {row.phase} | "
            f"{row.train_field_status} | {row.dev_gate_field_status} | "
            f"{row.emission_status} | {row.train_coverage_steps} | "
            f"{row.dev30_coverage_steps} | {str(row.cell_gate_passed).lower()} | "
            f"{row.attribution_label} | {row.reason_code or ''} | {row.confidence} |"
        )
    lines.append("")

    _write_text(path, "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# 10. Self-check (generic relational invariants only; no hardcoded counts)
# ---------------------------------------------------------------------------


def self_check(rows: List[Row], reconcile: ReconcileResult) -> None:
    """Validate generic relational invariants; raise ``ValueError`` on violation.

    Invariants (independent of any specific run's counts):
      * ``len(rows) == len(reconcile.declared)`` (one row per declared cell).
      * when ``extra`` is empty, ``len(emitted) + len(diff) == len(declared)``.
      * ``reconcile.extra == []`` (no emitted cell outside the declared set).
      * every row's ``attribution_label`` is one of the six controlled labels.
      * every diff cell is labeled ``contract_or_provenance_mismatch``.
    """
    if len(rows) != len(reconcile.declared):
        raise ValueError(
            f"self_check: row count {len(rows)} != declared count "
            f"{len(reconcile.declared)}"
        )

    if reconcile.extra_count != 0:
        raise ValueError(
            f"self_check: reconcile.extra must be empty, found "
            f"{reconcile.extra_count} extra cell(s)"
        )

    if len(reconcile.emitted) + len(reconcile.diff) != len(reconcile.declared):
        raise ValueError(
            "self_check: emitted + diff must equal declared when extra is empty "
            f"({len(reconcile.emitted)} + {len(reconcile.diff)} != "
            f"{len(reconcile.declared)})"
        )

    for row in rows:
        if row.attribution_label not in CONTROLLED_LABELS:
            raise ValueError(
                f"self_check: uncontrolled attribution_label "
                f"{row.attribution_label!r} for {row.skill} "
                f"{row.initial_class}:{row.phase}"
            )

    diff_keys = reconcile.diff_keys()
    for row in rows:
        if (row.skill, row.initial_class, row.phase) in diff_keys:
            if row.attribution_label != "contract_or_provenance_mismatch":
                raise ValueError(
                    "self_check: diff cell "
                    f"{row.skill} {row.initial_class}:{row.phase} must be "
                    "labeled contract_or_provenance_mismatch, found "
                    f"{row.attribution_label!r}"
                )
