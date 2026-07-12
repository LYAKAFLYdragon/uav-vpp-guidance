#!/usr/bin/env python3
"""Build a single AST submission zip from a checked submission manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> Dict[str, Any]:
    manifest = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    required = {"submission_id", "canonical_evidence_id", "canonical_commit", "latex_documents", "files"}
    missing = sorted(required.difference(manifest))
    if missing:
        raise ValueError(f"submission manifest is missing required keys: {', '.join(missing)}")
    return manifest


def assert_sources_exist(repo_root: Path, manifest: Dict[str, Any]) -> None:
    entries = list(manifest["latex_documents"]) + list(manifest["files"])
    missing = [str(entry["source"]) for entry in entries if not (repo_root / entry["source"]).exists()]
    if missing:
        raise FileNotFoundError(f"submission manifest references missing files: {', '.join(missing)}")


def compile_latex(source: Path, build_dir: Path, jobname: str, latex_bin: str) -> Path:
    build_dir.mkdir(parents=True, exist_ok=True)
    command = [
        latex_bin,
        "-interaction=batchmode",
        "-halt-on-error",
        f"-jobname={jobname}",
        f"-output-directory={build_dir}",
        source.name,
    ]
    for _ in range(2):
        result = subprocess.run(command, cwd=source.parent, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(
                f"LaTeX compilation failed for {source}:\n{result.stdout}\n{result.stderr}"
            )
    pdf = build_dir / f"{jobname}.pdf"
    if not pdf.exists():
        raise RuntimeError(f"LaTeX did not produce expected PDF: {pdf}")
    return pdf


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def file_index(root: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "submission_bundle_manifest.json":
            records.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return records


def build_bundle(
    *,
    repo_root: Path,
    manifest_path: Path,
    output_root: Path,
    latex_bin: str,
    overwrite: bool,
) -> Path:
    manifest = load_manifest(manifest_path)
    assert_sources_exist(repo_root, manifest)
    submission_id = str(manifest["submission_id"])
    destination = output_root / submission_id
    archive = output_root / f"{submission_id}.zip"
    if destination.exists() or archive.exists():
        if not overwrite:
            raise FileExistsError(
                f"submission output already exists ({destination} or {archive}); use --overwrite after review"
            )
        shutil.rmtree(destination, ignore_errors=True)
        archive.unlink(missing_ok=True)
    destination.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="submission-build-") as temporary:
        build_dir = Path(temporary)
        for document in manifest["latex_documents"]:
            source = repo_root / document["source"]
            pdf = compile_latex(source, build_dir, str(document["jobname"]), latex_bin)
            copy_file(pdf, destination / document["output"])
    for item in manifest["files"]:
        copy_file(repo_root / item["source"], destination / item["output"])
    readme = destination / "README.md"
    readme.write_text(
        "# AST submission package\n\n"
        f"Submission ID: `{submission_id}`\n\n"
        f"Canonical evidence: `{manifest['canonical_evidence_id']}` at `{manifest['canonical_commit']}`.\n\n"
        "This zip is generated only from `drones/submission_manifest.yaml`.\n",
        encoding="utf-8",
    )
    package_manifest = {
        "schema_version": "1.0",
        "submission_id": submission_id,
        "canonical_evidence_id": manifest["canonical_evidence_id"],
        "canonical_commit": manifest["canonical_commit"],
        "source_manifest": str(manifest_path.relative_to(repo_root)).replace("\\", "/"),
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": file_index(destination),
    }
    (destination / "submission_bundle_manifest.json").write_text(
        json.dumps(package_manifest, indent=2), encoding="utf-8"
    )
    shutil.make_archive(str(archive.with_suffix("")), "zip", output_root, submission_id)
    return archive


def verify_bundle(bundle_dir: Path) -> Dict[str, Any]:
    manifest_path = bundle_dir / "submission_bundle_manifest.json"
    if not manifest_path.exists():
        return {"valid": False, "issues": ["submission bundle manifest is missing"]}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    issues: List[str] = []
    for record in manifest.get("files", []):
        path = bundle_dir / record["path"]
        if not path.exists():
            issues.append(f"missing file: {record['path']}")
        elif path.stat().st_size != record["size_bytes"] or sha256_file(path) != record["sha256"]:
            issues.append(f"checksum mismatch: {record['path']}")
    for required in [
        "manuscript/dfar_ast_final_v5.pdf",
        "title_page/title_page_ast.pdf",
        "supplementary/supplementary_material_v4.pdf",
        "supplementary/supplement_index.pdf",
    ]:
        if not (bundle_dir / required).exists():
            issues.append(f"missing required submission deliverable: {required}")
    return {"valid": not issues, "issues": issues, "submission_id": manifest.get("submission_id")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="compile and package the submission zip")
    build.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    build.add_argument("--manifest", type=Path, default=Path("drones/submission_manifest.yaml"))
    build.add_argument("--output-root", type=Path, default=Path("drones/submission_dist"))
    build.add_argument("--latex-bin", default="pdflatex")
    build.add_argument("--overwrite", action="store_true")
    verify = subparsers.add_parser("verify", help="verify an unpacked submission directory")
    verify.add_argument("--bundle-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "build":
        repo_root = args.repo_root.resolve()
        manifest_path = args.manifest
        if not manifest_path.is_absolute():
            manifest_path = repo_root / manifest_path
        output_root = args.output_root
        if not output_root.is_absolute():
            output_root = repo_root / output_root
        archive = build_bundle(
            repo_root=repo_root,
            manifest_path=manifest_path.resolve(),
            output_root=output_root.resolve(),
            latex_bin=args.latex_bin,
            overwrite=args.overwrite,
        )
        result = verify_bundle(archive.with_suffix(""))
        result["archive"] = str(archive)
    else:
        result = verify_bundle(args.bundle_dir.resolve())
    print(json.dumps(result, indent=2))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
