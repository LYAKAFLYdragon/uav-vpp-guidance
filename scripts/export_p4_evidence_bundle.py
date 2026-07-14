#!/usr/bin/env python
"""Export a minimal, read-only, SHA256-verified P4 evidence bundle.

This packages the exact frozen P4 evidence needed to close P4 evidence
portability on another machine:

  * ``p4_geometry_pretrain_gate.json``           (top-level combined gate)
  * four ``geometry_pretrain_summary.json``      (per-skill dev30 gate + selected_gate)
  * the shared-skill ``registry`` snapshot       (declared-cell source)
  * ``evidence_manifest.json``                   (file hashes, source machine,
                                                  generation time, provenance boundaries)

Strictly READ-ONLY against the source evidence: the source files are only
hashed and copied byte-for-byte; nothing is retrained, re-evaluated, or
modified. No hardcoded drive letters — all paths are configurable and default
to relative repo paths.

Usage (export)::

    python scripts/export_p4_evidence_bundle.py \
        --gate-json src/p4_geometry_pretrain_gate.json \
        --registry-path config/experiment/thesis_five_state_shared_skill_registry_v1.yaml \
        --out-dir reports/p4_evidence_bundle_20260714

Usage (verify on the target machine)::

    python scripts/export_p4_evidence_bundle.py --verify --out-dir <bundle-dir>

The verifier recomputes every SHA256 and re-runs the declared-vs-emitted
reconciliation from the bundled files alone, printing PASS/FAIL and exiting
non-zero on any mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SKILLS: Tuple[str, ...] = (
    "pursuit_conversion",
    "lead_intercept",
    "defensive_extension",
    "reentry_recovery",
)

SUMMARY_FILENAME = "geometry_pretrain_summary.json"

#: Frozen inputs that are NOT present in this repo and are therefore recorded as
#: provenance_unverified boundaries in the manifest.
PROVENANCE_UNVERIFIED = (
    "run_commit",
    "training_plan.json",
    "resolved_geometry_config.json",
    "dev30_manifest",
)

BUNDLE_SCHEMA_VERSION = "p4_evidence_bundle_v1"


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: str) -> Any:
    with open(path, "rb") as fh:
        return json.loads(fh.read().decode("utf-8"))


def _json_top_level_fields(path: str) -> List[str]:
    try:
        obj = _load_json(path)
    except (ValueError, OSError):
        return []
    return sorted(obj.keys()) if isinstance(obj, dict) else []


# ---------------------------------------------------------------------------
# Declared-cell derivation (stdlib inline-list parser; no PyYAML).
# ---------------------------------------------------------------------------


def _derive_declared_cells(registry_path: str) -> Dict[str, List[str]]:
    """Parse ``skills.<skill>.geometry_phase_coverage.{geometry_states,phases}``."""
    import re

    inline = re.compile(r"^\s*(geometry_states|phases)\s*:\s*\[([^\]]*)\]\s*$")
    header = re.compile(r"^(\s{2})([A-Za-z0-9_]+)\s*:\s*$")
    with open(registry_path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    result: Dict[str, Dict[str, List[str]]] = {}
    in_skills = False
    current_skill: Optional[str] = None
    current_indent = 0
    in_cov = False

    for raw in lines:
        line = raw.rstrip("\n")
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if not in_skills:
            if stripped == "skills:" and indent == 0:
                in_skills = True
            continue
        if indent == 0 and stripped != "skills:":
            break
        m = header.match(line)
        if m and len(m.group(1)) == 2:
            current_skill = m.group(2)
            current_indent = indent
            in_cov = False
            result.setdefault(current_skill, {"geometry_states": [], "phases": []})
            continue
        if current_skill is None:
            continue
        if stripped == "geometry_phase_coverage:" and indent > current_indent:
            in_cov = True
            continue
        if in_cov and indent <= current_indent + 2 and not inline.match(line):
            in_cov = False
        if in_cov:
            im = inline.match(line)
            if im:
                key = im.group(1)
                vals = [t.strip() for t in im.group(2).split(",") if t.strip()]
                result[current_skill][key] = vals

    declared: Dict[str, List[str]] = {}
    for skill, spec in result.items():
        states = spec.get("geometry_states") or []
        phases = spec.get("phases") or []
        if states and phases:
            declared[skill] = [f"{s}:{p}" for s in states for p in phases]
    return declared


def _emitted_cells(gate_obj: Any, skill: str) -> Dict[str, int]:
    sel = gate_obj["skills"][skill]["selected_gate"]["geometry_phase_coverage"]
    return {k: int(v) for k, v in sel.items()}


def _reconcile(gate_obj: Any, declared: Dict[str, List[str]]) -> Dict[str, Any]:
    declared_total = emitted_total = diff_total = extra_total = 0
    per_skill: Dict[str, Dict[str, int]] = {}
    for skill in SKILLS:
        dcells = set(declared.get(skill, []))
        ecells = set(_emitted_cells(gate_obj, skill).keys())
        diff = dcells - ecells
        extra = ecells - dcells
        per_skill[skill] = {
            "declared": len(dcells),
            "emitted": len(ecells),
            "diff": len(diff),
            "extra": len(extra),
        }
        declared_total += len(dcells)
        emitted_total += len(ecells)
        diff_total += len(diff)
        extra_total += len(extra)
    return {
        "declared_count": declared_total,
        "emitted_count": emitted_total,
        "diff_count": diff_total,
        "extra_count": extra_total,
        "per_skill": per_skill,
    }


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_bundle(args: argparse.Namespace) -> int:
    gate_json = args.gate_json or os.path.join(
        REPO_ROOT, "src", "p4_geometry_pretrain_gate.json"
    )
    registry_path = args.registry_path
    gate_dir = os.path.dirname(os.path.abspath(gate_json))
    summaries_root = args.summaries_root or os.path.join(gate_dir, "skills")
    out_dir = os.path.abspath(args.out_dir)

    # Validate sources exist (read-only).
    if not os.path.isfile(gate_json):
        print(f"ERROR: gate JSON not found: {gate_json}", file=sys.stderr)
        return 2
    if not os.path.isfile(registry_path):
        print(f"ERROR: registry not found: {registry_path}", file=sys.stderr)
        return 2
    summary_srcs: Dict[str, str] = {}
    for skill in SKILLS:
        p = os.path.join(summaries_root, skill, SUMMARY_FILENAME)
        if not os.path.isfile(p):
            print(f"ERROR: per-skill summary not found: {p}", file=sys.stderr)
            return 2
        summary_srcs[skill] = p

    os.makedirs(out_dir, exist_ok=True)
    skills_out = os.path.join(out_dir, "skills")
    os.makedirs(skills_out, exist_ok=True)

    files: List[Dict[str, Any]] = []

    def _copy_record(src: str, rel_dst: str, role: str) -> None:
        dst = os.path.join(out_dir, rel_dst)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        src_sha = _sha256_file(src)
        dst_sha = _sha256_file(dst)
        if src_sha != dst_sha:
            raise RuntimeError(f"copy hash mismatch for {rel_dst}")
        files.append(
            {
                "role": role,
                "bundle_path": rel_dst.replace(os.sep, "/"),
                "source_path": os.path.relpath(src, REPO_ROOT).replace(os.sep, "/"),
                "sha256": dst_sha,
                "size_bytes": os.path.getsize(dst),
                "json_top_level_fields": _json_top_level_fields(dst)
                if dst.endswith(".json")
                else None,
            }
        )

    _copy_record(gate_json, "p4_geometry_pretrain_gate.json", "top_level_gate")
    _copy_record(
        registry_path,
        "thesis_five_state_shared_skill_registry_v1.yaml",
        "registry_snapshot",
    )
    for skill in SKILLS:
        _copy_record(
            summary_srcs[skill],
            os.path.join("skills", skill, SUMMARY_FILENAME),
            f"per_skill_summary:{skill}",
        )

    # Reconciliation computed from the bundled files (verification anchor).
    gate_obj = _load_json(gate_json)
    declared = _derive_declared_cells(registry_path)
    reconciliation = _reconcile(gate_obj, declared)

    gate_verdict = {
        "all_skills_ready_for_combat_finetune": gate_obj.get(
            "all_skills_ready_for_combat_finetune"
        ),
        "next_stage": gate_obj.get("next_stage"),
        "p3_encoder_sha256": gate_obj.get("p3_encoder_sha256"),
    }

    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "description": (
            "Minimal read-only P4 evidence bundle for GEO-20260713-R1 "
            "portability closure. Contains the top-level gate, the four "
            "per-skill geometry_pretrain_summary.json, and the shared-skill "
            "registry snapshot used as the declared-cell source. All source "
            "files were hashed and copied byte-for-byte; nothing was retrained, "
            "re-evaluated, or modified."
        ),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_machine": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
        },
        "provenance_boundaries": {
            "note": (
                "This bundle is NOT a reproduction of the original E: drive "
                "GEO-20260713-R1 run. The following frozen inputs are NOT "
                "available in the source repo and are recorded as "
                "provenance_unverified; the working-tree source is NEVER "
                "substituted for frozen run facts."
            ),
            "run_commit": "provenance_unverified",
            "training_plan.json": "provenance_unverified",
            "resolved_geometry_config.json": "provenance_unverified",
            "dev30_manifest": "provenance_unverified",
            "declared_cells_source": (
                "registry snapshot (provenance_unverified substitute for the "
                "missing frozen training_plan.json)"
            ),
        },
        "reconciliation": reconciliation,
        "gate_verdict": gate_verdict,
        "gate_thresholds": {
            "minimum_geometry_phase_coverage_per_declared_cell": 2,
            "minimum_intent_progress_fraction": 0.55,
        },
        "files": sorted(files, key=lambda f: f["bundle_path"]),
        "verify_hint": (
            "python scripts/export_p4_evidence_bundle.py --verify --out-dir <this-dir>"
        ),
    }

    manifest_path = os.path.join(out_dir, "evidence_manifest.json")
    with open(manifest_path, "w", encoding="utf-8", newline="") as fh:
        fh.write(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

    print("P4 evidence bundle exported (read-only).")
    print(f"  out-dir: {out_dir}")
    print(f"  files:   {len(files)} + evidence_manifest.json")
    print(
        "  reconciliation (from bundle): "
        f"declared={reconciliation['declared_count']} "
        f"emitted={reconciliation['emitted_count']} "
        f"diff={reconciliation['diff_count']} "
        f"extra={reconciliation['extra_count']}"
    )
    print("  provenance_unverified: " + ", ".join(PROVENANCE_UNVERIFIED))
    print(f"  manifest: {os.path.relpath(manifest_path, REPO_ROOT)}")
    return 0


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------


def verify_bundle(args: argparse.Namespace) -> int:
    out_dir = os.path.abspath(args.out_dir)
    manifest_path = os.path.join(out_dir, "evidence_manifest.json")
    if not os.path.isfile(manifest_path):
        print(f"ERROR: evidence_manifest.json not found in {out_dir}", file=sys.stderr)
        return 2
    manifest = _load_json(manifest_path)

    ok = True
    for entry in manifest["files"]:
        bundle_file = os.path.join(out_dir, entry["bundle_path"])
        if not os.path.isfile(bundle_file):
            print(f"FAIL  missing: {entry['bundle_path']}")
            ok = False
            continue
        actual = _sha256_file(bundle_file)
        if actual != entry["sha256"]:
            print(f"FAIL  sha256 mismatch: {entry['bundle_path']}")
            print(f"        expected={entry['sha256']}")
            print(f"        actual  ={actual}")
            ok = False
        else:
            print(f"OK    {entry['bundle_path']}  {actual}")

    # Re-run the reconciliation from the bundled files alone.
    gate_json = os.path.join(out_dir, "p4_geometry_pretrain_gate.json")
    registry_path = os.path.join(
        out_dir, "thesis_five_state_shared_skill_registry_v1.yaml"
    )
    try:
        gate_obj = _load_json(gate_json)
        declared = _derive_declared_cells(registry_path)
        recon = _reconcile(gate_obj, declared)
    except (OSError, ValueError, KeyError) as exc:
        print(f"FAIL  reconciliation could not run: {exc}")
        return 1

    expected = manifest.get("reconciliation", {})
    for key in ("declared_count", "emitted_count", "diff_count", "extra_count"):
        if recon[key] != expected.get(key):
            print(
                f"FAIL  reconciliation {key}: recomputed={recon[key]} "
                f"manifest={expected.get(key)}"
            )
            ok = False
    if ok:
        print(
            "OK    reconciliation (recomputed from bundle): "
            f"declared={recon['declared_count']} emitted={recon['emitted_count']} "
            f"diff={recon['diff_count']} extra={recon['extra_count']}"
        )

    print("RESULT: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="export_p4_evidence_bundle",
        description=(
            "Export or verify a minimal, read-only, SHA256-verified P4 evidence "
            "bundle for portability closure on another machine."
        ),
    )
    parser.add_argument("--verify", action="store_true", help="Verify an existing bundle.")
    parser.add_argument("--gate-json", default=None, help="Path to p4_geometry_pretrain_gate.json.")
    parser.add_argument(
        "--registry-path",
        default=os.path.join(
            "config", "experiment", "thesis_five_state_shared_skill_registry_v1.yaml"
        ),
        help="Path to the shared-skill registry snapshot (declared-cell source).",
    )
    parser.add_argument(
        "--summaries-root",
        default=None,
        help="Directory containing <skill>/geometry_pretrain_summary.json (default: <gate-dir>/skills).",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join("reports", "p4_evidence_bundle_20260714"),
        help="Bundle output directory (export) or bundle directory to verify.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verify:
        return verify_bundle(args)
    return export_bundle(args)


if __name__ == "__main__":
    raise SystemExit(main())
