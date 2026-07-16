#!/usr/bin/env python3
"""Create and verify a portable, slim paper-evidence bundle for Taxonomy60.

The bundle contains every paper-facing receipt, table, figure, frozen contract
and generator needed to inspect the formal claim.  Raw trajectories are not
duplicated because the source already stores 3.4+ GiB of them; their joint hash
and the per-episode raw hashes are retained as an explicit external provenance
boundary rather than silently pretending the slim bundle is a raw-data archive.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_ID = "THESIS-DUALSKILL-TAXONOMY60-3OPP-V1"
FORMAL_FILES = (
    "formal_artifact_contract.json",
    "formal_execution_manifest.json",
    "formal_execution_plan.json",
    "formal_verification.json",
    "taxonomy60_route_decision.json",
)
REPO_FILES = (
    "config/experiment/jsbsim_hrl_thesis_dualskill_taxonomy60_formal.yaml",
    "config/experiment/manifests/thesis_dualskill_taxonomy60_v1.yaml",
    "config/experiment/decision_contracts/thesis_dualskill_taxonomy60_no_go_v3.yaml",
    "scripts/analyze_thesis_dualskill_taxonomy60_formal.py",
    "scripts/build_thesis_dualskill_taxonomy60_paper_materials.py",
    "scripts/build_thesis_dualskill_taxonomy60_paper_bundle.py",
    "scripts/verify_thesis_dualskill_taxonomy60_formal.py",
    "tests/test_thesis_dualskill_taxonomy60_formal_analysis.py",
    "tests/test_thesis_dualskill_taxonomy60_formal_runner.py",
    "reports/thesis_master_evidence_closure_checklist_v2_20260716_zh.md",
    "reports/thesis_dualskill_taxonomy60_evidence_matrix_and_final_memo_20260716_zh.md",
)
PORTABLE_PATH_COLUMNS = {"source_raw_path"}


class PaperBundleError(RuntimeError):
    """Raised when an input artifact is invalid or a bundle is incomplete."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _copy_file(source: Path, destination: Path, *, make_portable: bool) -> tuple[str, bool]:
    """Copy one artifact, replacing machine-local raw paths with hash-addressed references."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    transformed = False
    if make_portable and source.suffix.lower() == ".csv":
        with source.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fieldnames = reader.fieldnames or []
        if any(column in fieldnames for column in PORTABLE_PATH_COLUMNS):
            for row in rows:
                raw_hash = row.get("source_raw_sha256")
                if raw_hash and row.get("source_raw_path"):
                    row["source_raw_path"] = "raw_not_embedded/{}.json".format(raw_hash)
                    transformed = True
            with destination.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        else:
            shutil.copy2(source, destination)
    elif make_portable and source.name == "paper_materials_manifest.json":
        payload = _load_json(source)
        payload["analysis_dir"] = "bundle/analysis"
        payload["formal_root"] = "external_raw_formal_source_not_embedded"
        destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        transformed = True
    else:
        shutil.copy2(source, destination)
    return _sha256(destination), transformed


def _iter_files(root: Path) -> Iterable[Path]:
    return (path for path in sorted(root.rglob("*")) if path.is_file())


def _add_tree(
    entries: list[dict[str, Any]],
    *,
    source_root: Path,
    destination_root: Path,
    bundle_root: Path,
    role: str,
    make_portable: bool,
) -> None:
    for source in _iter_files(source_root):
        destination = destination_root / source.relative_to(source_root)
        bundle_sha, transformed = _copy_file(source, destination, make_portable=make_portable)
        entries.append(
            {
                "bundle_path": destination.relative_to(bundle_root).as_posix(),
                "role": role,
                "source_sha256": _sha256(source),
                "bundle_sha256": bundle_sha,
                "size_bytes": destination.stat().st_size,
                "transformed_for_portability": transformed,
            }
        )


def _validate_inputs(formal_root: Path, analysis_dir: Path, materials_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    for name in FORMAL_FILES:
        if not (formal_root / name).is_file():
            raise PaperBundleError("formal root is missing {}".format(name))
    verification = _load_json(formal_root / "formal_verification.json")
    decision = _load_json(formal_root / "taxonomy60_route_decision.json")
    analysis = _load_json(analysis_dir / "analysis_manifest.json")
    materials = _load_json(materials_dir / "paper_materials_manifest.json")
    if verification.get("source_id") != SOURCE_ID or verification.get("verdict") != "pass" or int(verification.get("records", 0)) != 900:
        raise PaperBundleError("formal receipt is not a valid 900-record source")
    if decision.get("selected_study_route") != "paper_closure":
        raise PaperBundleError("formal route is not paper_closure")
    if analysis.get("source_id") != SOURCE_ID or int(analysis.get("raw_phase_record_count", 0)) != 900:
        raise PaperBundleError("analysis package lacks the 900-record raw phase receipt")
    if materials.get("source_id") != SOURCE_ID or int(materials.get("record_count", 0)) != 900:
        raise PaperBundleError("paper materials package lacks the 900-record receipt")
    return analysis, materials


def build_bundle(formal_root: Path, analysis_dir: Path, materials_dir: Path, bundle_dir: Path, *, overwrite: bool = False) -> dict[str, Any]:
    formal_root = Path(formal_root).resolve()
    analysis_dir = Path(analysis_dir).resolve()
    materials_dir = Path(materials_dir).resolve()
    bundle_dir = Path(bundle_dir).resolve()
    analysis_manifest, materials_manifest = _validate_inputs(formal_root, analysis_dir, materials_dir)
    if bundle_dir.exists():
        if not overwrite:
            raise PaperBundleError("bundle directory already exists; use --overwrite only for disposable bundles")
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)
    entries: list[dict[str, Any]] = []
    for name in FORMAL_FILES:
        source = formal_root / name
        destination = bundle_dir / "formal_receipts" / name
        bundle_sha, transformed = _copy_file(source, destination, make_portable=False)
        entries.append({
            "bundle_path": destination.relative_to(bundle_dir).as_posix(), "role": "formal_receipt",
            "source_sha256": _sha256(source), "bundle_sha256": bundle_sha,
            "size_bytes": destination.stat().st_size, "transformed_for_portability": transformed,
        })
    _add_tree(entries, source_root=analysis_dir, destination_root=bundle_dir / "analysis", bundle_root=bundle_dir, role="read_only_analysis", make_portable=True)
    _add_tree(entries, source_root=materials_dir, destination_root=bundle_dir / "paper_materials", bundle_root=bundle_dir, role="paper_material", make_portable=True)
    for relative in REPO_FILES:
        source = REPO_ROOT / relative
        if not source.is_file():
            raise PaperBundleError("repository input is missing {}".format(relative))
        destination = bundle_dir / "frozen_contracts_and_code" / relative
        bundle_sha, transformed = _copy_file(source, destination, make_portable=False)
        entries.append({
            "bundle_path": destination.relative_to(bundle_dir).as_posix(), "role": "frozen_contract_or_code",
            "source_sha256": _sha256(source), "bundle_sha256": bundle_sha,
            "size_bytes": destination.stat().st_size, "transformed_for_portability": transformed,
        })
    readme = "\n".join(
        [
            "# Taxonomy60 Slim Paper-Evidence Bundle",
            "",
            "- Source ID: `{}`.".format(SOURCE_ID),
            "- Contents: frozen contracts, formal receipts, read-only tables, capability cards, ablation report and figures.",
            "- Raw trajectories are intentionally not duplicated. The analysis package stores the 900 per-episode raw SHA-256 values and joint raw digest `{}`.".format(analysis_manifest.get("raw_phase_joint_sha256")),
            "- This is a paper-evidence bundle, not a replacement for the external raw telemetry archive.",
            "- Verify with `python scripts/build_thesis_dualskill_taxonomy60_paper_bundle.py verify --bundle-dir <bundle>`.",
        ]
    )
    (bundle_dir / "README.md").write_text(readme + "\n", encoding="utf-8")
    manifest = {
        "source_id": SOURCE_ID,
        "bundle_kind": "slim_portable_paper_evidence_not_raw_archive",
        "record_count": 900,
        "route": "paper_closure",
        "raw_provenance": {
            "raw_episodes_embedded": False,
            "raw_phase_joint_sha256": analysis_manifest.get("raw_phase_joint_sha256"),
            "per_episode_sha256_table": "analysis/dynamic_taxonomy_phase_episode_index.csv",
        },
        "analysis_manifest_source_sha256": _sha256(analysis_dir / "analysis_manifest.json"),
        "paper_materials_manifest_source_sha256": _sha256(materials_dir / "paper_materials_manifest.json"),
        "files": entries,
        "prohibited_actions": materials_manifest.get("prohibited_actions", []),
    }
    (bundle_dir / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify_bundle(bundle_dir: Path) -> dict[str, Any]:
    bundle_dir = Path(bundle_dir).resolve()
    manifest_path = bundle_dir / "bundle_manifest.json"
    if not manifest_path.is_file():
        raise PaperBundleError("bundle manifest is missing")
    manifest = _load_json(manifest_path)
    if manifest.get("source_id") != SOURCE_ID or manifest.get("route") != "paper_closure":
        raise PaperBundleError("bundle identity or route mismatch")
    failures: list[str] = []
    for entry in manifest.get("files", []):
        path = bundle_dir / entry["bundle_path"]
        if not path.is_file() or _sha256(path) != entry["bundle_sha256"]:
            failures.append(entry["bundle_path"])
    raw_dirs = list(bundle_dir.rglob("raw"))
    if raw_dirs:
        failures.append("raw_trajectory_directory_must_not_be_embedded")
    phase_index = bundle_dir / manifest["raw_provenance"]["per_episode_sha256_table"]
    if not phase_index.is_file():
        failures.append("per_episode_raw_sha_table_missing")
    return {
        "source_id": SOURCE_ID,
        "verdict": "pass" if not failures else "fail",
        "file_count": len(manifest.get("files", [])),
        "failures": failures,
        "raw_phase_joint_sha256": manifest["raw_provenance"]["raw_phase_joint_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--formal-root", type=Path, required=True)
    build.add_argument("--analysis-dir", type=Path, required=True)
    build.add_argument("--materials-dir", type=Path, required=True)
    build.add_argument("--bundle-dir", type=Path, required=True)
    build.add_argument("--overwrite", action="store_true")
    verify = subparsers.add_parser("verify")
    verify.add_argument("--bundle-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = (
            build_bundle(args.formal_root, args.analysis_dir, args.materials_dir, args.bundle_dir, overwrite=args.overwrite)
            if args.command == "build"
            else verify_bundle(args.bundle_dir)
        )
    except PaperBundleError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))
    return 0 if result.get("verdict", "pass") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
