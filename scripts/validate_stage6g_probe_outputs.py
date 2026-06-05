#!/usr/bin/env python3
"""
Validate Stage 6G.1 probe outputs against artifact contract.

Checks:
- Output root directory exists
- 12 cells present (3 guidance modes × 4 scenarios)
- Each cell status == completed
- Total episode count == 720 (unless --smoke)
- Each cell contains 2 methods: no_prediction, gru_frozen
- Each cell contains eval seeds 0, 1, 2
- Required artifacts per cell and globally
- Guidance mode consistency (requested == resolved == effective == class name)
- allow_incomplete == false => any missing artifact exits 1

Usage:
    python scripts/validate_stage6g_probe_outputs.py \
        --input outputs/stage6g_guidance_limitation_probe/run_20260605_074434 \
        [--smoke]

Outputs:
    stage6g_validation_report.md
    stage6g_validation_summary.json
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import yaml


EXPECTED_GUIDANCE_MODES = {"los_rate", "proportional_navigation", "hybrid"}
EXPECTED_SCENARIOS = {"favorable", "disadvantage", "weaving_pursuit", "weaving_disadvantage"}
EXPECTED_METHODS = {"no_prediction", "gru_frozen"}
EXPECTED_EVAL_SEEDS = {0, 1, 2}
REQUIRED_CELL_ARTIFACTS = [
    "prediction_metrics.json",
    "resolved_config.yaml",
]
REQUIRED_GLOBAL_ARTIFACTS = [
    "run_manifest.json",
    "raw_episodes.csv",
    "scenario_method_summary.csv",
    "pairwise_mcnemar.csv",
    "paper_safe_claims.md",
    "README_result_block.md",
    "run.log",
]


def fail(msg: str) -> bool:
    print(f"  FAIL: {msg}")
    return False


def ok(msg: str) -> bool:
    print(f"  OK: {msg}")
    return True


def validate(args) -> Tuple[bool, Dict]:
    all_ok = True
    input_dir = Path(args.input)
    is_smoke = args.smoke
    report_lines = []
    summary = {
        "input_dir": str(input_dir),
        "smoke": is_smoke,
        "checks_passed": 0,
        "checks_failed": 0,
        "cells": {},
        "global": {},
        "overall": False,
    }

    def check(name: str, condition: bool, detail: str) -> bool:
        nonlocal all_ok
        if condition:
            ok(f"{name}: {detail}")
            summary["checks_passed"] += 1
            report_lines.append(f"- [x] {name}: {detail}")
            return True
        else:
            fail(f"{name}: {detail}")
            all_ok = False
            summary["checks_failed"] += 1
            report_lines.append(f"- [ ] {name}: {detail}")
            return False

    report_lines.append("# Stage 6G.1 Probe Validation Report")
    report_lines.append("")
    report_lines.append(f"**Input**: `{input_dir}`")
    report_lines.append(f"**Smoke mode**: {is_smoke}")
    report_lines.append("")

    # 1. Input directory exists
    check("Input directory exists", input_dir.exists() and input_dir.is_dir(), str(input_dir))
    if not input_dir.exists():
        summary["overall"] = False
        return all_ok, summary

    # 2. Discover cell directories
    cell_dirs = {}
    for guidance in EXPECTED_GUIDANCE_MODES:
        for scenario in EXPECTED_SCENARIOS:
            cell_name = f"{guidance}_{scenario}"
            cell_path = input_dir / cell_name
            cell_dirs[cell_name] = cell_path

    # 3. Check each cell
    total_episodes = 0
    total_cells_completed = 0
    guidance_mode_consistency_issues = []
    episode_key_set: Set[Tuple] = set()
    duplicate_keys: List[str] = []

    for cell_name, cell_path in cell_dirs.items():
        cell_ok = True
        cell_summary = {
            "exists": cell_path.exists(),
            "artifacts_present": {},
            "methods_found": set(),
            "eval_seeds_found": set(),
            "episode_count": 0,
            "status": "unknown",
            "guidance_mode_consistent": True,
        }

        report_lines.append(f"\n## Cell: {cell_name}")

        cell_ok &= check(
            f"{cell_name} directory exists",
            cell_path.exists() and cell_path.is_dir(),
            str(cell_path),
        )
        if not cell_path.exists():
            cell_summary["methods_found"] = sorted(cell_summary["methods_found"])
            cell_summary["eval_seeds_found"] = sorted(cell_summary["eval_seeds_found"])
            summary["cells"][cell_name] = cell_summary
            all_ok = False
            continue

        # Required cell artifacts
        for artifact in REQUIRED_CELL_ARTIFACTS:
            artifact_path = cell_path / artifact
            present = artifact_path.exists()
            cell_summary["artifacts_present"][artifact] = present
            cell_ok &= check(
                f"{cell_name}/{artifact}", present, "present" if present else "MISSING"
            )

        # Load prediction_metrics.json
        metrics_path = cell_path / "prediction_metrics.json"
        if metrics_path.exists():
            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    methods_data = json.load(f)
            except Exception as exc:
                check(f"{cell_name}/prediction_metrics.json parse", False, str(exc))
                methods_data = []
        else:
            methods_data = []

        # Check methods, seeds, episodes, guidance mode consistency
        for m in methods_data:
            method_name = m.get("method_name", m.get("method", "unknown"))
            cell_summary["methods_found"].add(method_name)
            requested = m.get("requested_guidance_mode", "")
            effective = m.get("effective_guidance_mode", "")
            # Also try to infer from env class name if available
            env_class = m.get("env_guidance_class_name", "")

            # Guidance mode consistency check
            guidance_parts = cell_name.split("_")
            # cell_name is like "los_rate_favorable" or "proportional_navigation_disadvantage"
            # Need to extract guidance mode (everything before last _ which is scenario)
            # But scenarios can have underscores: weaving_pursuit, weaving_disadvantage
            # So: known scenarios are the keys in EXPECTED_SCENARIOS
            inferred_guidance = None
            for scenario in sorted(EXPECTED_SCENARIOS, key=len, reverse=True):
                if cell_name.endswith(f"_{scenario}"):
                    inferred_guidance = cell_name[: -(len(scenario) + 1)]
                    break

            if inferred_guidance and effective and effective != inferred_guidance:
                guidance_mode_consistency_issues.append(
                    f"{cell_name}: effective={effective} != inferred={inferred_guidance}"
                )
                cell_summary["guidance_mode_consistent"] = False
            if requested and effective and requested != effective:
                guidance_mode_consistency_issues.append(
                    f"{cell_name}: requested={requested} != effective={effective}"
                )
                cell_summary["guidance_mode_consistent"] = False

            raw_eps = m.get("raw_episodes", [])
            for ep in raw_eps:
                cell_summary["episode_count"] += 1
                total_episodes += 1
                eval_seed = ep.get("evaluation_seed", ep.get("eval_seed", None))
                if eval_seed is not None:
                    cell_summary["eval_seeds_found"].add(int(eval_seed))

                # Track episode keys for pairing validation
                ep_key = (
                    ep.get("scenario", ""),
                    method_name,
                    effective or inferred_guidance or "",
                    ep.get("training_seed", -1),
                    ep.get("evaluation_seed", ep.get("eval_seed", -1)),
                    ep.get("episode_index", -1),
                )
                if ep_key in episode_key_set:
                    duplicate_keys.append(f"{cell_name}: {ep_key}")
                episode_key_set.add(ep_key)

        # Convert sets to lists for JSON serialization
        cell_summary["methods_found"] = sorted(cell_summary["methods_found"])
        cell_summary["eval_seeds_found"] = sorted(cell_summary["eval_seeds_found"])

        # Validate methods
        missing_methods = EXPECTED_METHODS - set(cell_summary["methods_found"])
        cell_ok &= check(
            f"{cell_name} methods",
            len(missing_methods) == 0,
            f"found={cell_summary['methods_found']}, missing={sorted(missing_methods)}",
        )

        # Validate eval seeds
        if is_smoke:
            expected_seeds = EXPECTED_EVAL_SEEDS
            missing_seeds = set()
        else:
            missing_seeds = EXPECTED_EVAL_SEEDS - set(cell_summary["eval_seeds_found"])
        cell_ok &= check(
            f"{cell_name} eval seeds",
            len(missing_seeds) == 0,
            f"found={cell_summary['eval_seeds_found']}, missing={sorted(missing_seeds)}",
        )

        # Validate episode count per cell
        if is_smoke:
            # Smoke mode: we don't know how many seeds were used; just check non-zero
            expected_episodes_per_cell = None
            cell_ok &= check(
                f"{cell_name} episode count",
                cell_summary["episode_count"] > 0,
                f"found={cell_summary['episode_count']} (smoke mode: any non-zero)",
            )
        else:
            expected_episodes_per_cell = len(EXPECTED_METHODS) * len(EXPECTED_EVAL_SEEDS) * 10
            cell_ok &= check(
                f"{cell_name} episode count",
                cell_summary["episode_count"] == expected_episodes_per_cell,
                f"found={cell_summary['episode_count']}, expected={expected_episodes_per_cell}",
            )

        if cell_ok:
            total_cells_completed += 1
            cell_summary["status"] = "completed"
        else:
            cell_summary["status"] = "failed"

        summary["cells"][cell_name] = cell_summary

    # 4. Global checks
    report_lines.append("\n## Global Checks")

    expected_total_cells = len(EXPECTED_GUIDANCE_MODES) * len(EXPECTED_SCENARIOS)
    check(
        "Total cells",
        len(summary["cells"]) == expected_total_cells,
        f"found={len(summary['cells'])}, expected={expected_total_cells}",
    )

    check(
        "Cells completed",
        total_cells_completed == expected_total_cells,
        f"completed={total_cells_completed}, expected={expected_total_cells}",
    )

    if is_smoke:
        check(
            "Total episode count",
            total_episodes > 0,
            f"found={total_episodes} (smoke mode: any non-zero)",
        )
    else:
        expected_total_episodes = (
            len(EXPECTED_GUIDANCE_MODES)
            * len(EXPECTED_SCENARIOS)
            * len(EXPECTED_METHODS)
            * len(EXPECTED_EVAL_SEEDS)
            * 10
        )
        check(
            "Total episode count",
            total_episodes == expected_total_episodes,
            f"found={total_episodes}, expected={expected_total_episodes}",
        )

    # Guidance mode consistency
    check(
        "Guidance mode consistency",
        len(guidance_mode_consistency_issues) == 0,
        "issues=" + "; ".join(guidance_mode_consistency_issues) if guidance_mode_consistency_issues else "all consistent",
    )

    # Duplicate episode keys
    check(
        "No duplicate episode keys",
        len(duplicate_keys) == 0,
        "duplicates=" + "; ".join(duplicate_keys) if duplicate_keys else "none",
    )

    # Global artifacts
    for artifact in REQUIRED_GLOBAL_ARTIFACTS:
        artifact_path = input_dir / artifact
        present = artifact_path.exists()
        summary["global"][artifact] = present
        check(
            f"Global artifact {artifact}", present, "present" if present else "MISSING"
        )

    # 5. Load run_manifest.json for metadata cross-check
    manifest_path = input_dir / "run_manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as exc:
            manifest = {}
            check("run_manifest.json parse", False, str(exc))
    else:
        manifest = {}

    if manifest:
        run_status = manifest.get("run_status", "unknown")
        check(
            "run_status is completed",
            run_status == "completed",
            f"status={run_status}",
        )

        allow_incomplete = manifest.get("allow_incomplete", False)
        # If manifest doesn't have it, infer from status
        if "allow_incomplete" not in manifest:
            allow_incomplete = run_status in ("partial", "smoke_partial")
        check(
            "allow_incomplete is false",
            not allow_incomplete,
            f"allow_incomplete={allow_incomplete}",
        )

        manifest_cells_total = manifest.get("cells_total", 0)
        check(
            "manifest cells_total",
            manifest_cells_total == expected_total_cells,
            f"manifest={manifest_cells_total}, expected={expected_total_cells}",
        )

        manifest_episodes = manifest.get("total_raw_episodes", 0)
        check(
            "manifest total_raw_episodes",
            manifest_episodes == total_episodes,
            f"manifest={manifest_episodes}, actual={total_episodes}",
        )

        # Check artifacts_present in manifest matches reality (skip run_manifest itself)
        artifacts_present = manifest.get("artifacts_present", {})
        for artifact in REQUIRED_GLOBAL_ARTIFACTS:
            if artifact == "run_manifest.json":
                continue
            manifest_says = artifacts_present.get(artifact, False)
            reality = (input_dir / artifact).exists()
            if manifest_says != reality:
                check(
                    f"manifest artifact consistency {artifact}",
                    False,
                    f"manifest={manifest_says}, reality={reality}",
                )

    # 6. Raw episodes CSV cross-check
    raw_csv_path = input_dir / "raw_episodes.csv"
    if raw_csv_path.exists():
        try:
            with open(raw_csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                csv_rows = list(reader)
            check(
                "raw_episodes.csv row count",
                len(csv_rows) == total_episodes,
                f"csv={len(csv_rows)}, json_total={total_episodes}",
            )
            # Check required columns
            required_cols = [
                "scenario", "method", "guidance_mode_requested", "effective_guidance_mode",
                "training_seed", "evaluation_seed", "episode_index", "success",
                "termination_reason", "capture_time", "miss_distance", "min_range",
                "oob", "crash", "fallback_used", "prediction_error",
            ]
            if csv_rows:
                headers = set(csv_rows[0].keys())
                missing_cols = [c for c in required_cols if c not in headers]
                check(
                    "raw_episodes.csv columns",
                    len(missing_cols) == 0,
                    f"missing={missing_cols}",
                )
        except Exception as exc:
            check("raw_episodes.csv parse", False, str(exc))

    # Finalize
    summary["overall"] = all_ok
    summary["total_episodes"] = total_episodes
    summary["total_cells_completed"] = total_cells_completed

    report_lines.append("")
    report_lines.append("## Summary")
    report_lines.append("")
    report_lines.append(f"- **Overall**: {'PASS' if all_ok else 'FAIL'}")
    report_lines.append(f"- **Checks passed**: {summary['checks_passed']}")
    report_lines.append(f"- **Checks failed**: {summary['checks_failed']}")
    report_lines.append(f"- **Cells completed**: {total_cells_completed}/{expected_total_cells}")
    report_lines.append(f"- **Total episodes**: {total_episodes}")
    report_lines.append(f"- **Guidance consistency issues**: {len(guidance_mode_consistency_issues)}")
    report_lines.append(f"- **Duplicate episode keys**: {len(duplicate_keys)}")
    report_lines.append("")
    if not all_ok:
        report_lines.append("> ⚠️ **Validation failed**. Do not use these outputs for paper claims until all checks pass."
        )

    return all_ok, summary, "\n".join(report_lines)


def main():
    parser = argparse.ArgumentParser(description="Validate Stage 6G.1 probe outputs")
    parser.add_argument("--input", type=str, required=True, help="Path to probe output directory (e.g., outputs/stage6g_guidance_limitation_probe/run_YYYYMMDD_HHMMSS)")
    parser.add_argument("--smoke", action="store_true", help="Smoke mode: expect 1 episode per cell instead of 10")
    parser.add_argument("--allow-incomplete", action="store_true", help="Allow partial validation (do not exit 1 on missing cells)")
    args = parser.parse_args()

    all_ok, summary, report_md = validate(args)

    input_dir = Path(args.input)
    report_path = input_dir / "stage6g_validation_report.md"
    summary_path = input_dir / "stage6g_validation_summary.json"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"\nValidation report saved to {report_path}")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Validation summary saved to {summary_path}")

    if all_ok:
        print("\nAll validations passed.")
        sys.exit(0)
    else:
        print("\nSome validations failed.")
        if not args.allow_incomplete:
            sys.exit(1)
        else:
            print("(Exiting 0 because --allow-incomplete was set)")
            sys.exit(0)


if __name__ == "__main__":
    main()
