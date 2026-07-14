#!/usr/bin/env python
"""CLI for the P4 geometry-pretrain gate failure-attribution audit (read-only).

This is a thin argparse front-end over the pure-function module
``uav_vpp_guidance.evaluation.p4_gate_failure_attribution``. It is a strictly
**non-training, non-tuning, read-only** audit against the frozen gate evidence
of the run ``GEO-20260713-R1``. It never retrains, re-evaluates, edits the
pre-registered gate, or mutates any P4 output / checkpoint / model / reward. The
only writes it performs are the allowlisted output artifacts passed via
``--out-md`` / ``--out-csv`` / ``--out-json`` (or, on a 2b integrity failure, a
single integrity report to ``--out-md``).

Design references (see
``.kiro/specs/p4-geometry-gate-failure-attribution-audit/design.md`` §6):
  * CLI behavior (2b runs FIRST and halts non-zero on integrity failure; 2a
    reconciliation continues to the full matrix and exits 0).
  * CLI surface (configurable paths; no hardcoded drive letters).

STRICT constraints honored here:
  * Python standard library only.
  * No hardcoded drive letters (all defaults are relative repo paths).
  * No hardcoded run-specific counts (26/26/0/0 lives only in the integration
    test, never in this CLI).
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make ``src/uav_vpp_guidance`` importable when run directly from the repo.
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_EVIDENCE_ROOT = REPO_ROOT / "reports" / "p4_evidence_bundle_20260714"

from uav_vpp_guidance.evaluation.p4_gate_failure_attribution import (  # noqa: E402
    CONCLUSION_COMBAT_FINETUNE,
    build_matrix,
    collect_emitted_cells,
    cross_artifact_consistency,
    derive_declared_cells,
    emit_csv,
    emit_integrity_report,
    emit_json,
    emit_markdown,
    freeze_provenance,
    load_json,
    reconcile,
    self_check,
)

#: The four skills that MUST be present in a valid P4 gate.
REQUIRED_SKILLS: Tuple[str, ...] = (
    "pursuit_conversion",
    "lead_intercept",
    "defensive_extension",
    "reentry_recovery",
)

#: The gate fields that MUST be readable under each skill's ``selected_gate``.
REQUIRED_GATE_FIELDS: Tuple[str, ...] = (
    "checks",
    "geometry_phase_coverage",
    "intent_progress_fraction",
    "profile_coverage",
    "safety_by_opponent",
)

#: Per-skill summary filename inside ``<summaries-root>/<skill>/``.
SUMMARY_FILENAME = "geometry_pretrain_summary.json"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser (defaults overridable, no drive letters)."""
    default_out_md = "reports/thesis_five_state_p4_gate_failure_attribution_audit_20260714_zh.md"
    default_out_csv = "reports/thesis_five_state_p4_gate_failure_attribution_matrix_20260714.csv"
    default_out_json = "reports/thesis_five_state_p4_gate_failure_attribution_matrix_20260714.json"

    parser = argparse.ArgumentParser(
        prog="analyze_p4_geometry_gate_failure_attribution",
        description=(
            "Read-only P4 geometry-pretrain gate failure-attribution audit. "
            "Produces a reconciled attribution matrix (md/csv/json) from the "
            "frozen gate evidence. Does NOT retrain, re-evaluate, or edit the "
            "pre-registered gate."
        ),
    )
    parser.add_argument(
        "--evidence-root",
        default=os.environ.get("P4_EVIDENCE_ROOT", str(DEFAULT_EVIDENCE_ROOT)),
        help=(
            "Portable P4 evidence bundle root. Default: "
            "reports/p4_evidence_bundle_20260714 (or P4_EVIDENCE_ROOT)."
        ),
    )
    parser.add_argument(
        "--results-root",
        default=None,
        help=(
            "Legacy results root directory. When supplied, it provides the default "
            "gate path; otherwise the portable evidence bundle is used."
        ),
    )
    parser.add_argument(
        "--gate-json",
        default=None,
        help=(
            "Path to p4_geometry_pretrain_gate.json. Default: "
            "<evidence-root>/p4_geometry_pretrain_gate.json, or "
            "<results-root>/p4_geometry_pretrain_gate.json when --results-root is set."
        ),
    )
    parser.add_argument(
        "--registry-path",
        default=None,
        help=(
            "Shared-skill registry YAML. Default: the registry snapshot in "
            "--evidence-root; it is a provenance_unverified substitute for the "
            "missing training_plan.json."
        ),
    )
    parser.add_argument(
        "--training-plan",
        default=None,
        help="OPTIONAL. Frozen training_plan.json; if absent -> provenance_unverified.",
    )
    parser.add_argument(
        "--resolved-config",
        default=None,
        help="OPTIONAL. Frozen resolved_geometry_config.json; if absent -> provenance_unverified.",
    )
    parser.add_argument(
        "--dev30-manifest-path",
        default=None,
        help=(
            "OPTIONAL (separate from --registry-path). dev30 evaluation manifest; "
            "if absent -> provenance_unverified."
        ),
    )
    parser.add_argument(
        "--run-commit",
        default=None,
        help="OPTIONAL. Frozen run commit SHA; if absent -> provenance_unverified.",
    )
    parser.add_argument(
        "--summaries-root",
        default=None,
        help=(
            "OPTIONAL. Directory containing per-skill <skill>/"
            f"{SUMMARY_FILENAME}. Default: <dir of gate-json>/skills."
        ),
    )
    parser.add_argument(
        "--out-md",
        default=default_out_md,
        help=f"Output audit markdown path. Default: {default_out_md}",
    )
    parser.add_argument(
        "--out-csv",
        default=default_out_csv,
        help=f"Output matrix CSV path. Default: {default_out_csv}",
    )
    parser.add_argument(
        "--out-json",
        default=default_out_json,
        help=f"Output matrix JSON path. Default: {default_out_json}",
    )
    return parser


def resolve_paths(args: argparse.Namespace) -> Dict[str, Any]:
    """Resolve the effective input paths from the parsed arguments.

    The portable bundle is the default source. ``--results-root`` remains a
    backward-compatible override for an external result directory.
    """
    evidence_root = os.path.abspath(args.evidence_root)
    results_root = args.results_root

    if args.gate_json:
        gate_json = args.gate_json
    elif results_root:
        gate_json = os.path.join(results_root, "p4_geometry_pretrain_gate.json")
    else:
        gate_json = os.path.join(evidence_root, "p4_geometry_pretrain_gate.json")

    gate_dir = os.path.dirname(os.path.abspath(gate_json))
    summaries_root = args.summaries_root or os.path.join(gate_dir, "skills")
    registry_path = args.registry_path or os.path.join(
        evidence_root, "thesis_five_state_shared_skill_registry_v1.yaml"
    )

    return {
        "evidence_root": evidence_root,
        "results_root": results_root,
        "gate_json": gate_json,
        "summaries_root": summaries_root,
        "registry_path": registry_path,
    }


# ---------------------------------------------------------------------------
# Validation + loading
# ---------------------------------------------------------------------------


def validate_gate(gate_obj: Any) -> List[str]:
    """Validate the gate JSON structure. Return a list of error messages."""
    errors: List[str] = []
    if not isinstance(gate_obj, dict):
        return ["gate JSON top-level is not a JSON object/mapping"]

    skills = gate_obj.get("skills")
    if not isinstance(skills, dict):
        return ["gate JSON has no readable 'skills' mapping"]

    for skill in REQUIRED_SKILLS:
        skill_obj = skills.get(skill)
        if not isinstance(skill_obj, dict):
            errors.append(f"missing or unreadable skill '{skill}'")
            continue
        selected_gate = skill_obj.get("selected_gate")
        if not isinstance(selected_gate, dict):
            errors.append(f"skill '{skill}' has no readable 'selected_gate' block")
            continue
        for fld in REQUIRED_GATE_FIELDS:
            if fld not in selected_gate:
                errors.append(
                    f"skill '{skill}' selected_gate is missing field '{fld}'"
                )
    return errors


def locate_summary_paths(summaries_root: str) -> Dict[str, str]:
    """Return ``{skill: path}`` for each located per-skill summary file."""
    located: Dict[str, str] = {}
    for skill in REQUIRED_SKILLS:
        candidate = os.path.join(summaries_root, skill, SUMMARY_FILENAME)
        if os.path.isfile(candidate):
            located[skill] = candidate
    return located


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _host_interpreter_note() -> str:
    return (
        "audit executed on Python "
        f"{platform.python_version()} ({platform.system()} {platform.machine()}); "
        "stdlib only, no third-party dependencies"
    )


def run(args: argparse.Namespace) -> int:
    """Execute the audit. Return a process exit code (0 = success)."""
    paths = resolve_paths(args)
    gate_json_path = paths["gate_json"]
    summaries_root = paths["summaries_root"]
    registry_path = paths["registry_path"]

    # ---- Step 1: load + validate the gate JSON --------------------------
    if not os.path.isfile(gate_json_path):
        print(f"ERROR: gate JSON not found: {gate_json_path}", file=sys.stderr)
        return 2
    try:
        gate_obj, _gate_sha, _gate_fields = load_json(gate_json_path)
    except (ValueError, OSError) as exc:
        print(f"ERROR: could not load gate JSON {gate_json_path}: {exc}", file=sys.stderr)
        return 2

    gate_errors = validate_gate(gate_obj)
    if gate_errors:
        print("ERROR: gate JSON validation failed:", file=sys.stderr)
        for msg in gate_errors:
            print(f"  - {msg}", file=sys.stderr)
        return 2

    if not os.path.isfile(registry_path):
        print(
            f"ERROR: registry path not found: {registry_path}",
            file=sys.stderr,
        )
        return 2

    # ---- Step 2: load the four per-skill summaries ----------------------
    summary_paths = locate_summary_paths(summaries_root)
    missing_summaries = [s for s in REQUIRED_SKILLS if s not in summary_paths]
    if missing_summaries:
        print(
            "ERROR: could not locate per-skill summaries under "
            f"{summaries_root} for: {', '.join(missing_summaries)}",
            file=sys.stderr,
        )
        return 2

    per_skill_summaries: Dict[str, Any] = {}
    for skill, spath in summary_paths.items():
        try:
            summary_obj, _sha, _fields = load_json(spath)
        except (ValueError, OSError) as exc:
            print(f"ERROR: could not load summary {spath}: {exc}", file=sys.stderr)
            return 2
        per_skill_summaries[skill] = summary_obj

    # ---- Step 3: 2b Cross-Artifact Consistency FIRST --------------------
    problems = cross_artifact_consistency(gate_obj, per_skill_summaries)
    if problems:
        emit_integrity_report(problems, args.out_md)
        print(
            "INTEGRITY FAILURE (2b): top-level gate disagrees with per-skill "
            f"summaries in {len(problems)} field(s). Wrote integrity report to "
            f"{args.out_md}. NO attribution matrix produced.",
            file=sys.stderr,
        )
        return 3

    # ---- Step 4: 2a declared-vs-emitted reconciliation (continue) -------
    declared = derive_declared_cells(registry_path)
    emitted = collect_emitted_cells(gate_obj)
    reconcile_result = reconcile(declared, emitted)

    # ---- Step 5: freeze provenance --------------------------------------
    prov_inputs: Dict[str, Optional[str]] = {
        "gate_json": gate_json_path,
        "registry": registry_path,
        "training_plan": args.training_plan,
        "resolved_geometry_config": args.resolved_config,
        "dev30_manifest": args.dev30_manifest_path,
    }
    for skill in REQUIRED_SKILLS:
        prov_inputs[f"summary_{skill}"] = summary_paths[skill]

    provenance = freeze_provenance(prov_inputs, run_commit=args.run_commit)
    provenance.host_interpreter_note = _host_interpreter_note()

    # ---- Step 6: build matrix, self-check, emit artifacts ---------------
    rows = build_matrix(gate_obj, declared, reconcile_result)
    self_check(rows, reconcile_result)

    emit_json(
        rows,
        reconcile_result,
        provenance,
        CONCLUSION_COMBAT_FINETUNE,
        args.out_json,
    )
    emit_csv(rows, args.out_csv)
    emit_markdown(
        rows,
        reconcile_result,
        provenance,
        None,
        args.out_md,
        lang="zh",
    )

    # ---- Step 7: concise summary + exit 0 -------------------------------
    print("P4 geometry gate failure-attribution audit complete.")
    print("2b cross-artifact consistency: CLEAN")
    print(
        "2a reconciliation (derived): "
        f"declared={reconcile_result.declared_count} "
        f"emitted={reconcile_result.emitted_count} "
        f"diff={reconcile_result.diff_count} "
        f"extra={reconcile_result.extra_count}"
    )
    print(f"matrix rows: {len(rows)}")
    if provenance.provenance_unverified_inputs:
        print(
            "provenance_unverified: "
            + ", ".join(provenance.provenance_unverified_inputs)
        )
    else:
        print("provenance_unverified: (none)")
    print(
        "run-level contract_or_provenance_mismatch: "
        f"{provenance.contract_or_provenance_mismatch}"
    )
    print(f"conclusion: {CONCLUSION_COMBAT_FINETUNE}")
    print("outputs:")
    print(f"  md:   {args.out_md}")
    print(f"  csv:  {args.out_csv}")
    print(f"  json: {args.out_json}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
