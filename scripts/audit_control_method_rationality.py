#!/usr/bin/env python3
"""Run the read-only control-method rationality audit.

This command only emits the two audit report formats after the audit module's
invariants pass.  It never changes control source, configuration, checkpoints,
or training state.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import platform
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from uav_vpp_guidance.common.git import get_git_info
from uav_vpp_guidance.evaluation.control_method_audit import (
    emit_reports,
    load_source_facts,
    run_diagnostics,
    self_check,
    triage,
)

DEFAULT_OUT_MD = REPO_ROOT / "reports" / "control_method_rationality_audit_findings_20260725_zh.md"
DEFAULT_OUT_JSON = REPO_ROOT / "reports" / "control_method_rationality_audit_findings_20260725.json"


def _resolve_path(value: str) -> Path:
    """Return an absolute path without requiring the destination to exist."""
    return Path(value).expanduser().resolve()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_output_paths(out_md: Path, out_json: Path) -> None:
    """Reject repository writes other than the two allowlisted audit artifacts.

    External caller-provided scratch paths are supported for isolated integration
    tests; they must remain outside this repository and retain the report suffix.
    """
    if out_md == out_json:
        raise ValueError("--out-md and --out-json must name different files")
    if out_md.suffix.lower() != ".md" or out_json.suffix.lower() != ".json":
        raise ValueError("--out-md must end in .md and --out-json must end in .json")

    allowed = {DEFAULT_OUT_MD.resolve(), DEFAULT_OUT_JSON.resolve()}
    for output in (out_md, out_json):
        if output in allowed:
            continue
        if _is_relative_to(output, REPO_ROOT):
            raise ValueError(
                f"output path is outside the report allowlist: {output}; "
                "use the canonical reports paths or an external scratch directory"
            )


def _provenance() -> dict[str, str]:
    """Collect reproducibility metadata without changing repository state."""
    git_info = get_git_info(str(REPO_ROOT))
    return {
        "git_commit": git_info.get("commit") or "unavailable",
        "git_branch": git_info.get("branch") or "unavailable",
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the read-only control-method rationality audit reports."
    )
    parser.add_argument(
        "--guidance-config",
        default="config/guidance.yaml",
        help="Guidance YAML used to derive effective audit facts (default: %(default)s).",
    )
    parser.add_argument(
        "--source-root",
        default="src/uav_vpp_guidance",
        help="Control source package root used for source evidence (default: %(default)s).",
    )
    parser.add_argument(
        "--out-md",
        default=str(DEFAULT_OUT_MD.relative_to(REPO_ROOT)),
        help="Allowlisted Markdown report destination (default: %(default)s).",
    )
    parser.add_argument(
        "--out-json",
        default=str(DEFAULT_OUT_JSON.relative_to(REPO_ROOT)),
        help="Allowlisted JSON report destination (default: %(default)s).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    out_md = _resolve_path(args.out_md)
    out_json = _resolve_path(args.out_json)

    try:
        _validate_output_paths(out_md, out_json)
        facts = load_source_facts(
            {
                "guidance_config": args.guidance_config,
                "source_root": args.source_root,
            }
        )
        diagnostics = run_diagnostics(facts)
        records = triage(facts, diagnostics)
        self_check(records)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"control-method audit failed before writing reports: {exc}", file=sys.stderr)
        return 1

    try:
        provenance = _provenance()
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        emit_reports(records, out_md, out_json, facts, diagnostics, provenance)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"control-method audit failed while writing reports: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote audit reports: {out_md} and {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
