#!/usr/bin/env python3
"""Classify dirty and untracked worktree files into explicit research lanes."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List


POLICY = {
    "canonical": {
        "owner": "paper/reproducibility steward",
        "lane": "CAN-20260705 and v5 submission package",
        "retention": "preserve; merge only through a reviewed minimal change set",
    },
    "noncanonical": {
        "owner": "research-lane owner",
        "lane": "explicit non-canonical exploration",
        "retention": "preserve in a separate worktree/output root; never promote silently",
    },
    "historical": {
        "owner": "archive steward",
        "lane": "historical or superseded material",
        "retention": "retain read-only for traceability; exclude from current claims",
    },
    "build": {
        "owner": "release/build steward",
        "lane": "generated package, cache, or temporary artifact",
        "retention": "regenerate from a manifest; do not treat as source evidence",
    },
    "review-required": {
        "owner": "repository steward",
        "lane": "unassigned",
        "retention": "preserve unchanged until a human assigns a lane",
    },
}


def classify_path(path: str) -> str:
    normalized = path.replace("\\", "/").lower()
    name = Path(normalized).name
    if (
        normalized.startswith("drones/build")
        or normalized.startswith("drones/submission_dist")
        or normalized.startswith(".codex_tmp")
        or normalized.startswith((".agents/", ".codex/", "_temp_"))
        or normalized.startswith(".pytest_cache")
        or normalized.startswith(".ruff_cache")
        or name.endswith((".aux", ".out", ".spl", ".synctex.gz", ".log"))
    ):
        return "build"
    if any(token in normalized for token in ("noncanonical", "future_geometry", "prediction_decision_bridge", "explore_rl")):
        return "noncanonical"
    if any(
        token in normalized
        for token in (
            "historical",
            "double_dqn",
            "ddqn",
            "sac",
            "legacy",
            "dfartv2",
            "stage6",
            "stage7",
            "stage8",
            "stage9",
            "launch_",
            "paper_front20",
            "paper_full",
            "reviews/",
            "rl_algorithm_additions",
            "run_manifest90",
        )
    ):
        return "historical"
    if normalized.startswith(("src/", "config/", "tests/", "reports/", "drones/", "docs/", "scripts/")):
        return "canonical"
    if normalized in {
        "readme.md",
        "state.md",
        "loop.md",
        "loop-budget.md",
        "loop-run-log.md",
        "loop_audit_report.md",
        "loop_audit_report_v2.md",
    } or normalized.startswith("run_formal_"):
        return "canonical"
    return "review-required"


def git_status(repo_root: Path) -> Iterable[Dict[str, str]]:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    for line in result.stdout.splitlines():
        if not line:
            continue
        yield {"status": line[:2], "path": line[3:]}


def render_markdown(report: Dict[str, object]) -> str:
    lines = [
        "# Worktree Inventory Classification",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        f"Repository: `{report['repo_root']}`",
        "",
        "## Category Summary",
        "",
        "| Category | Files | Owner | Lane | Retention |",
        "|---|---:|---|---|---|",
    ]
    categories = report["categories"]
    for category in POLICY:
        info = POLICY[category]
        count = len(categories.get(category, []))
        lines.append(
            f"| `{category}` | {count} | {info['owner']} | {info['lane']} | {info['retention']} |"
        )
    lines.extend(
        [
            "",
            "## Assignment Rules",
            "",
            "See `docs/worktree_lane_policy.md`. `review-required` files are deliberately retained and blocked from merge or paper citation until assigned.",
        ]
    )
    for category in POLICY:
        entries = categories.get(category, [])
        lines.extend(["", f"## `{category}` Entries ({len(entries)})", ""])
        for entry in entries:
            lines.append(f"- `{entry['status']}` `{entry['path']}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    categories: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for entry in git_status(repo_root):
        categories[classify_path(entry["path"])].append(entry)
    report: Dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "policy": POLICY,
        "categories": {name: categories.get(name, []) for name in POLICY},
    }
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    unassigned = len(categories.get("review-required", []))
    print(json.dumps({"valid": True, "review_required": unassigned, "categories": {k: len(v) for k, v in categories.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
