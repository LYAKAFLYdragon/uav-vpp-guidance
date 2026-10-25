#!/usr/bin/env python
"""Analyze completed H0/H1/H2 artifacts; never runs JSBSim."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from uav_vpp_guidance.evaluation.prediction_highlevel_observation_v1 import (
    aggregate_episodes,
    paired_deltas,
    SOURCE_ID,
)


def load_episodes(root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(root.rglob("metrics.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        method = path.parent.parent.parent.name if path.parent.parent.parent else "unknown"
        opponent = path.parent.name
        for episode in payload.get("raw_episodes", []):
            row = dict(episode)
            row.setdefault("method", method)
            row.setdefault("opponent", opponent)
            rows.append(row)
    return rows


def analyze(episodes: list[dict], source_id: str = SOURCE_ID) -> dict:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    by_method: dict[str, list[dict]] = defaultdict(list)
    for row in episodes:
        key = (str(row.get("opponent", "unknown")), str(row.get("method", "unknown")), str(row.get("scenario", "unknown")))
        grouped[key].append(row)
        by_method[str(row.get("method", "unknown"))].append(row)
    summaries = []
    for (opponent, method, scenario), rows in sorted(grouped.items()):
        summary = aggregate_episodes(rows)
        summaries.append({"opponent": opponent, "method": method, "scenario": scenario, **summary})
    deltas = paired_deltas(dict(by_method))
    return {"source_id": source_id, "episode_count": len(episodes), "summaries": summaries, "paired_deltas": deltas}


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_markdown(path: Path, result: dict) -> None:
    lines = [
        "# Predicted Geometry High-Level Observation V1 Analysis",
        "",
        f"Source ID: `{result['source_id']}`",
        f"Raw episodes: `{result['episode_count']}`",
        "",
        "## Grouped Results",
        "",
        "| Opponent | Method | Scenario | n | Success | Wilson 95% | Mean return | Mean prediction error (m) | Fallback rate |",
        "|---|---|---|---:|---:|---|---:|---:|---:|",
    ]
    for row in result["summaries"]:
        lines.append(
            f"| {row['opponent']} | {row['method']} | {row['scenario']} | {row['episodes']} | "
            f"{row['success_rate']:.3f} | [{row['success_wilson_low']:.3f}, {row['success_wilson_high']:.3f}] | "
            f"{row['mean_return']:.3f} | {row['mean_prediction_error_m']:.3f} | {row['mean_prediction_fallback_rate']:.3f} |"
        )
    lines.extend(["", "## Paired Deltas", "", "Paired deltas are candidate minus H0 on the same opponent, scenario and episode seed.", ""])
    if result["paired_deltas"]:
        lines.extend([
            "| Opponent | Scenario | Training seed | Method | dReturn | dRange (m) | dATA (deg) | dPrediction error (m) | dSuccess |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|",
        ])
        for row in result["paired_deltas"]:
            def fmt(key):
                value = row.get(key)
                return "NA" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)
            lines.append(f"| {row['opponent']} | {row['scenario']} | {row.get('training_seed', '')} | {row['method']} | {fmt('delta_return')} | {fmt('delta_final_range_m')} | {fmt('delta_final_ata_deg')} | {fmt('delta_mean_prediction_error_m')} | {row['success_delta']} |")
    else:
        lines.append("No complete paired keys were available.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summary_row(row: dict) -> dict:
    return {
        "opponent": row.get("opponent"),
        "method": row.get("method"),
        "scenario": row.get("scenario"),
        "episodes": row.get("episodes"),
        "successes": row.get("successes"),
        "success_rate": row.get("success_rate"),
        "success_wilson_low": row.get("success_wilson_low"),
        "success_wilson_high": row.get("success_wilson_high"),
        "mean_return": row.get("mean_return"),
        "mean_final_range_m": row.get("mean_final_range_m"),
        "mean_final_ata_deg": row.get("mean_final_ata_deg"),
        "mean_prediction_error_m": row.get("mean_prediction_error_m"),
        "mean_prediction_fallback_rate": row.get("mean_prediction_fallback_rate"),
        "mode_switch_rate": row.get("mode_switch_rate"),
        "terminal_counts": json.dumps(row.get("terminal_counts", {}), ensure_ascii=False, sort_keys=True),
    }


def _write_required_artifacts(root: Path, result: dict, episodes: list[dict]) -> None:
    """Materialize the formal/pilot tables required by the protocol."""
    root.mkdir(parents=True, exist_ok=True)
    source_id = result["source_id"]
    summaries = [_summary_row(row) for row in result["summaries"]]
    _write_csv(root / "episode_level_results.csv", episodes)
    _write_csv(root / "metrics_summary.csv", summaries)

    def aggregate_groups(keys: tuple[str, ...], rows: list[dict]) -> list[dict]:
        groups: dict[tuple[str, ...], list[dict]] = defaultdict(list)
        for row in rows:
            groups[tuple(str(row.get(key, "")) for key in keys)].append(row)
        output = []
        for key, group_rows in sorted(groups.items()):
            summary = aggregate_episodes(group_rows)
            output.append({
                **_summary_row(summary),
                **{name: value for name, value in zip(keys, key)},
            })
        return output

    _write_csv(
        root / "metrics_per_seed.csv",
        aggregate_groups(
            ("method", "training_seed", "opponent", "evaluation_seed"),
            episodes,
        ),
    )
    _write_csv(
        root / "metrics_by_opponent.csv",
        aggregate_groups(("opponent", "method"), episodes),
    )
    _write_csv(
        root / "metrics_by_attitude.csv",
        aggregate_groups(("scenario", "opponent", "method"), episodes),
    )
    _write_csv(
        root / "prediction_geometry_error.csv",
        [row for row in summaries if row["method"] != "h0_current_geometry"],
    )
    _write_csv(
        root / "mode_switch_summary.csv",
        [
            {key: row.get(key) for key in ("opponent", "method", "scenario", "episodes", "mode_switch_rate")}
            for row in summaries
        ],
    )

    terminal_rows = []
    for row in result["summaries"]:
        for reason, count in row.get("terminal_counts", {}).items():
            terminal_rows.append({
                "opponent": row["opponent"],
                "method": row["method"],
                "scenario": row["scenario"],
                "reason": reason,
                "count": count,
            })
    _write_csv(root / "terminal_breakdown.csv", terminal_rows)

    convergence_rows = []
    for path in sorted(root.rglob("run_manifest.json")):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("training_started"):
            convergence_rows.append({
                "method": manifest.get("method"),
                "training_seed": manifest.get("training_seed"),
                "training_scope": manifest.get("training_scope"),
                "checkpoint_sha256": (manifest.get("checkpoint") or {}).get("sha256"),
                "status": "checkpoint_present" if (manifest.get("checkpoint") or {}).get("exists") else "checkpoint_missing",
            })
    _write_csv(root / "convergence_status.csv", convergence_rows)

    # Keep the runner's immutable execution/provenance manifest intact. The
    # analysis has its own manifest so later re-analysis cannot rewrite the
    # record of how the episodes were produced.
    if (root / "formal_manifest.json").exists():
        analysis_manifest_name = "formal_analysis_manifest.json"
    else:
        analysis_manifest_name = "analysis_manifest.json"
    _write_json(root / analysis_manifest_name, {
        "source_id": source_id,
        "phase": "formal" if (root / "formal_manifest.json").exists() else "analysis",
        "analysis_complete": True,
        "raw_episode_count": len(episodes),
        "paired_delta_count": len(result["paired_deltas"]),
        "execution_manifest_preserved": (root / "formal_manifest.json").exists(),
    })

    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    all_runs = len({(row.get("method"), row.get("opponent")) for row in summaries})
    (reports / "prediction_highlevel_observation_status.md").write_text(
        f"# Prediction High-Level Observation Status\n\nSource ID: `{source_id}`\n\nRaw episodes: `{len(episodes)}`\n\nMethod/opponent groups: `{all_runs}`\n\nThis report is generated from raw episode records and does not modify the thesis manuscript.\n",
        encoding="utf-8",
    )
    (reports / "prediction_highlevel_observation_completion_audit.md").write_text(
        "# Completion Audit\n\n"
        f"Raw episode records: `{len(episodes)}`\n\n"
        f"Paired records against H0: `{len(result['paired_deltas'])}`\n\n"
        "All claims remain conditional on verifier success and the recorded denominators.\n",
        encoding="utf-8",
    )
    _write_csv(
        reports / "claim_evidence_table.csv",
        [
            {"claim": "predicted geometry reaches high-level observation", "source_id": source_id, "evidence": "observation_contract_version and observation schema", "episodes": len(episodes)},
            {"claim": "H1 freezes PPO", "source_id": source_id, "evidence": "run_manifest.training_started=false and ppo_update_permitted=false", "episodes": len(episodes)},
            {"claim": "H2 adapts only high-level PPO", "source_id": source_id, "evidence": "training_scope=high_level_ppo_only", "episodes": len(episodes)},
        ],
    )
    _write_csv(reports / "protocol_comparison_table.csv", summaries)
    (reports / "causality_audit.json").write_text(
        json.dumps({
            "source_id": source_id,
            "future_state_access_allowed": False,
            "predicted_geometry_definition": "predicted position plus current own state; estimated velocity from predicted displacement",
            "runtime_verification_required": True,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (reports / "artifact_validation.json").write_text(
        json.dumps({"source_id": source_id, "raw_episode_count": len(episodes), "paired_delta_count": len(result["paired_deltas"]), "analysis_generated": True}, indent=2),
        encoding="utf-8",
    )
    (reports / "thesis_insertion_notes.md").write_text(
        "# Thesis Insertion Notes\n\n"
        "Use the generated tables only after the artifact verifier passes. Report H0, H1 and H2 separately by opponent and initial attitude; retain the raw episode denominator and avoid claiming universal improvement.\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    episodes = load_episodes(root)
    if not episodes:
        print("No metrics.json raw_episodes found", file=sys.stderr)
        return 2
    source_id = SOURCE_ID
    formal_manifest = root / "formal_manifest.json"
    if formal_manifest.is_file():
        try:
            manifest = json.loads(formal_manifest.read_text(encoding="utf-8"))
            source_id = str(manifest.get("source_id") or SOURCE_ID)
        except (OSError, json.JSONDecodeError):
            pass
    result = analyze(episodes, source_id=source_id)
    output = (args.output_dir or root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(output / "grouped_results.csv", result["summaries"])
    _write_csv(output / "paired_deltas.csv", result["paired_deltas"])
    _write_markdown(output / "analysis.md", result)
    _write_required_artifacts(output, result, episodes)
    print(f"Analysis written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
