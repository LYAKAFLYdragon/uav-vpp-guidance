"""Read-only analysis and terminal-discriminability gate for Heldout40.

This analyzer consumes completed comparison-run JSON only.  It never imports a
policy, constructs an environment, or alters the frozen evaluation outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from uav_vpp_guidance.evaluation.engagement_geometry_taxonomy import (  # noqa: E402
    classify_relative_geometry,
)


SOURCE_ID = "THESIS-FIVE-STATE-HELDOUT40-V1"
METHODS = (
    "canonical_ppo_high_level_policy",
    "legacy_static_oracle_task_gate",
    "head_on_vpp_specialist_no_routing",
    "crossing_vpp_specialist_no_routing",
)
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
FIXED_METHODS = METHODS[2:]


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _mean(values: Iterable[Any]) -> float | None:
    finite = [number for value in values if (number := _finite(value)) is not None]
    return sum(finite) / len(finite) if finite else None


def _std(values: Iterable[Any]) -> float | None:
    finite = [number for value in values if (number := _finite(value)) is not None]
    if not finite:
        return None
    average = sum(finite) / len(finite)
    return math.sqrt(sum((value - average) ** 2 for value in finite) / len(finite))


def _bool(value: Any) -> bool:
    return bool(value) if value is not None else False


def _outcome(record: Mapping[str, Any]) -> str:
    if _bool(record.get("win")):
        return "win"
    if _bool(record.get("loss")):
        return "loss"
    if _bool(record.get("draw")):
        return "draw"
    return "unresolved"


def _terminal_reason(record: Mapping[str, Any]) -> str:
    return str(record.get("combat_reason") or record.get("termination_reason") or "unknown")


def _is_ego_crash_or_oob(reason: str) -> bool:
    lowered = reason.lower()
    return "ego" in lowered and ("crash" in lowered or "out_of_bounds" in lowered or "oob" in lowered)


def _phase_name(frame: Mapping[str, Any]) -> str:
    if _bool(frame.get("pre_merge")):
        return "pre_merge"
    range_rate = _finite(frame.get("range_rate_mps"))
    return "re_entry" if range_rate is not None and range_rate < 0.0 else "post_merge"


def _dynamic_taxonomy(frame: Mapping[str, Any]) -> str:
    return classify_relative_geometry(frame.get("aa_deg"), frame.get("ata_deg"))


def _fraction(count: int, total: int) -> float | None:
    return count / total if total else None


def _first_true_time(trajectory: Sequence[Mapping[str, Any]], field: str) -> float | None:
    for frame in trajectory:
        if _bool(frame.get(field)):
            return _finite(frame.get("time_s"))
    return None


def _vpp_vertical_offset(frame: Mapping[str, Any]) -> float | None:
    vp_z = _finite(frame.get("vp_pos_z"))
    ego_z = _finite(frame.get("ego_pos_z"))
    return vp_z - ego_z if vp_z is not None and ego_z is not None else None


def summarize_episode(record: Mapping[str, Any]) -> dict[str, Any]:
    """Compress one raw trajectory without losing required held-out metrics."""

    trajectory = [frame for frame in record.get("trajectory", []) if isinstance(frame, Mapping)]
    phase_counts = {name: 0 for name in ("pre_merge", "post_merge", "re_entry")}
    taxonomy_counts: dict[str, int] = defaultdict(int)
    for frame in trajectory:
        phase_counts[_phase_name(frame)] += 1
        taxonomy_counts[_dynamic_taxonomy(frame)] += 1
    steps = len(trajectory)
    reason = _terminal_reason(record)
    outcome = _outcome(record)
    ego_attack_fraction = _mean(1.0 if _bool(frame.get("ego_in_attack_zone")) else 0.0 for frame in trajectory)
    target_attack_fraction = _mean(1.0 if _bool(frame.get("target_in_attack_zone")) else 0.0 for frame in trajectory)
    dt = _mean(
        _finite(right.get("time_s")) - _finite(left.get("time_s"))
        for left, right in zip(trajectory, trajectory[1:])
        if _finite(right.get("time_s")) is not None
        and _finite(left.get("time_s")) is not None
    )
    if dt is None:
        dt = 0.2
    return {
        "opponent": str(record.get("opponent_stage")),
        "method": str(record.get("controller") or record.get("method")),
        "scenario": str(record.get("scenario")),
        "seed": record.get("seed"),
        "initial_class": record.get("scenario_metadata", {}).get("initial_class"),
        "height_condition": record.get("scenario_metadata", {}).get("height_condition"),
        "mirror_sign": record.get("scenario_metadata", {}).get("mirror_sign"),
        "distance_speed_package": record.get("scenario_metadata", {}).get("distance_speed_package"),
        "outcome": outcome,
        "resolved": outcome in {"win", "loss"},
        "terminal_reason": reason,
        "ego_crash_or_oob": _is_ego_crash_or_oob(reason),
        "ego_hp": _finite(record.get("ego_hp")),
        "target_hp": _finite(record.get("target_hp")),
        "hp_advantage": _finite(record.get("hp_advantage")),
        "steps": steps,
        "total_time_s": _finite(record.get("total_time_s")),
        "ego_attack_zone_fraction": ego_attack_fraction,
        "target_attack_zone_fraction": target_attack_fraction,
        "ego_attack_zone_time_s": ego_attack_fraction * steps * dt if ego_attack_fraction is not None else None,
        "target_attack_zone_time_s": target_attack_fraction * steps * dt if target_attack_fraction is not None else None,
        "first_ego_attack_zone_time_s": _first_true_time(trajectory, "ego_in_attack_zone"),
        "first_target_attack_zone_time_s": _first_true_time(trajectory, "target_in_attack_zone"),
        "pre_merge_fraction": _fraction(phase_counts["pre_merge"], steps),
        "post_merge_fraction": _fraction(phase_counts["post_merge"], steps),
        "re_entry_fraction": _fraction(phase_counts["re_entry"], steps),
        "dynamic_taxonomy_step_counts": dict(sorted(taxonomy_counts.items())),
        "mean_vp_forward_bias_m": _mean(frame.get("vp_forward_bias_m") for frame in trajectory),
        "std_vp_forward_bias_m": _std(frame.get("vp_forward_bias_m") for frame in trajectory),
        "mean_vp_lateral_bias_m": _mean(frame.get("vp_lateral_bias_m") for frame in trajectory),
        "std_vp_lateral_bias_m": _std(frame.get("vp_lateral_bias_m") for frame in trajectory),
        "mean_vp_vertical_offset_m": _mean(_vpp_vertical_offset(frame) for frame in trajectory),
        "std_vp_vertical_offset_m": _std(_vpp_vertical_offset(frame) for frame in trajectory),
    }


def _load_records(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "aggregate" / "episode_records.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    episodes = payload.get("episodes") if isinstance(payload, Mapping) else None
    if not isinstance(episodes, list):
        raise ValueError(f"No episode list in {path}")
    return [dict(record) for record in episodes if isinstance(record, Mapping)]


def _validate_run(
    records: Sequence[Mapping[str, Any]],
    *,
    opponent: str,
    manifest_scenarios: Mapping[str, Mapping[str, Any]],
) -> None:
    expected_keys = {
        (scenario_name, method)
        for scenario_name in manifest_scenarios
        for method in METHODS
    }
    actual_keys = set()
    for record in records:
        if str(record.get("opponent_stage")) != opponent:
            raise ValueError(f"Opponent provenance mismatch in {opponent}: {record.get('opponent_stage')}")
        method = str(record.get("controller") or record.get("method"))
        scenario = str(record.get("scenario"))
        if method not in METHODS:
            raise ValueError(f"Unexpected method {method!r} in {opponent}")
        if scenario not in manifest_scenarios:
            raise ValueError(f"Unexpected scenario {scenario!r} in {opponent}")
        key = (scenario, method)
        if key in actual_keys:
            raise ValueError(f"Duplicate paired episode {key} in {opponent}")
        actual_keys.add(key)
        if record.get("backend") != "jsbsim" or not _bool(record.get("strict_backend")):
            raise ValueError(f"Heldout40 requires strict JSBSim: {opponent}/{key}")
        metadata = record.get("scenario_metadata") or {}
        for field in ("initial_class", "height_condition", "mirror_sign", "distance_speed_package"):
            if metadata.get(field) != manifest_scenarios[scenario].get(field):
                raise ValueError(f"Scenario metadata mismatch for {opponent}/{key}: {field}")
    if actual_keys != expected_keys:
        missing = sorted(expected_keys.difference(actual_keys))
        extra = sorted(actual_keys.difference(expected_keys))
        raise ValueError(
            f"Incomplete paired design for {opponent}: missing={missing[:3]}, extra={extra[:3]}"
        )
    for scenario_name in manifest_scenarios:
        seeds = {
            record.get("seed")
            for record in records
            if str(record.get("scenario")) == scenario_name
        }
        if len(seeds) != 1:
            raise ValueError(f"Methods do not share a paired seed: {opponent}/{scenario_name}")


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    wins = sum(row["outcome"] == "win" for row in rows)
    losses = sum(row["outcome"] == "loss" for row in rows)
    draws = sum(row["outcome"] == "draw" for row in rows)
    resolved = wins + losses
    return {
        "n_total": total,
        "n_resolved": resolved,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "unresolved": total - resolved - draws,
        "resolved_ratio": _fraction(resolved, total),
        "win_rate": _fraction(wins, resolved),
        "ego_crash_or_oob_rate": _mean(1.0 if row["ego_crash_or_oob"] else 0.0 for row in rows),
        "mean_hp_advantage": _mean(row["hp_advantage"] for row in rows),
        "mean_ego_attack_zone_time_s": _mean(row["ego_attack_zone_time_s"] for row in rows),
        "mean_target_attack_zone_time_s": _mean(row["target_attack_zone_time_s"] for row in rows),
        "mean_first_ego_attack_zone_time_s": _mean(row["first_ego_attack_zone_time_s"] for row in rows),
        "mean_first_target_attack_zone_time_s": _mean(row["first_target_attack_zone_time_s"] for row in rows),
        "mean_pre_merge_fraction": _mean(row["pre_merge_fraction"] for row in rows),
        "mean_post_merge_fraction": _mean(row["post_merge_fraction"] for row in rows),
        "mean_re_entry_fraction": _mean(row["re_entry_fraction"] for row in rows),
        "mean_vp_forward_bias_m": _mean(row["mean_vp_forward_bias_m"] for row in rows),
        "mean_vp_lateral_bias_m": _mean(row["mean_vp_lateral_bias_m"] for row in rows),
        "mean_vp_vertical_offset_m": _mean(row["mean_vp_vertical_offset_m"] for row in rows),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _pairing_verification(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for opponent in OPPONENTS:
        scenarios = sorted(
            {str(row["scenario"]) for row in rows if row["opponent"] == opponent}
        )
        for scenario in scenarios:
            selected = [
                row for row in rows if row["opponent"] == opponent and row["scenario"] == scenario
            ]
            methods = sorted(str(row["method"]) for row in selected)
            seeds = {row["seed"] for row in selected}
            cells.append(
                {
                    "opponent": opponent,
                    "scenario": scenario,
                    "method_count": len(methods),
                    "methods": methods,
                    "shared_seed": next(iter(seeds)) if len(seeds) == 1 else None,
                    "passed": methods == sorted(METHODS) and len(seeds) == 1,
                }
            )
    return {
        "source_id": SOURCE_ID,
        "expected_cells": len(OPPONENTS) * 40,
        "actual_cells": len(cells),
        "all_cells_passed": all(cell["passed"] for cell in cells),
        "cells": cells,
    }


def _format_ratio(value: Any) -> str:
    number = _finite(value)
    return "NA" if number is None else f"{number:.3f}"


def _execution_report_markdown(
    *,
    gate: Mapping[str, Any],
    pairing: Mapping[str, Any],
    run_dirs: Mapping[str, Path],
) -> str:
    lines = [
        "# Heldout40 Execution Report",
        "",
        f"- Source ID: `{SOURCE_ID}`",
        "- Scope: 40 explicit five-state scenarios x 4 frozen methods x 3 separately reported opponents = 480 strict-JSBSim episodes.",
        "- Training/tuning: none. The analyzer read immutable formal outputs only.",
        f"- Pairing: `{pairing['actual_cells']}/{pairing['expected_cells']}` scenario-opponent cells passed shared-seed four-method pairing.",
        "",
        "## Head-On Terminal Discriminability",
        "",
        "The gate uses the eight `initial_class=head_on` scenarios per opponent. A routing comparison is terminally interpretable only when canonical PPO and at least one fixed control have a resolved ratio >= 0.60.",
        "",
        "| Opponent | Canonical resolved ratio | Fixed head-on | Fixed crossing | Verdict |",
        "|---|---:|---:|---:|---|",
    ]
    for opponent in OPPONENTS:
        item = gate["opponents"][opponent]
        fixed = item["fixed_resolved_ratios"]
        lines.append(
            "| {opponent} | {canonical} | {head_on} | {crossing} | {label} |".format(
                opponent=opponent,
                canonical=_format_ratio(item["canonical_resolved_ratio"]),
                head_on=_format_ratio(fixed["head_on_vpp_specialist_no_routing"]),
                crossing=_format_ratio(fixed["crossing_vpp_specialist_no_routing"]),
                label=item["label"],
            )
        )
    lines.extend(
        [
            "",
            "All three opponent-specific gates passed. This establishes terminal discriminability for this envelope; it does not by itself establish superiority or universal generalization.",
            "",
            "## Evidence Index",
            "",
            "- `heldout40_episode_metrics.csv`: terminal semantics, HP, attack-zone, phase, dynamic taxonomy, and VPP geometry per episode.",
            "- `heldout40_summary.csv`: opponent/method/initial-state summaries; opponents are not pooled.",
            "- `heldout40_terminal_discriminability_gate.json`: preregistered gate verdicts.",
            "- `heldout40_pairing_verification.json`: shared-seed four-method pairing proof.",
            "",
            "## Formal Inputs",
            "",
        ]
    )
    lines.extend(f"- `{opponent}`: `{run_dirs[opponent]}`" for opponent in OPPONENTS)
    return "\n".join(lines) + "\n"


def analyze(
    *,
    run_dirs: Mapping[str, Path],
    manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    if manifest.get("source_id") != SOURCE_ID:
        raise ValueError(f"Unexpected manifest source: {manifest.get('source_id')}")
    scenarios = manifest.get("scenarios")
    if not isinstance(scenarios, list) or len(scenarios) != 40:
        raise ValueError("Heldout40 manifest must contain exactly 40 scenarios")
    scenario_metadata = {
        str(item["name"]): dict(item.get("metadata") or {})
        for item in scenarios
        if isinstance(item, Mapping)
    }
    if set(run_dirs) != set(OPPONENTS):
        raise ValueError(f"Expected run directories for {OPPONENTS}, got {sorted(run_dirs)}")

    episode_rows: list[dict[str, Any]] = []
    for opponent, run_dir in run_dirs.items():
        records = _load_records(run_dir)
        _validate_run(records, opponent=opponent, manifest_scenarios=scenario_metadata)
        episode_rows.extend(summarize_episode(record) for record in records)

    summary_rows: list[dict[str, Any]] = []
    for opponent in OPPONENTS:
        for method in METHODS:
            for initial_class in ("all", "advantage", "head_on", "disadvantage", "neutral", "crossing_entry"):
                selected = [
                    row
                    for row in episode_rows
                    if row["opponent"] == opponent
                    and row["method"] == method
                    and (initial_class == "all" or row["initial_class"] == initial_class)
                ]
                if not selected:
                    continue
                summary_rows.append(
                    {
                        "opponent": opponent,
                        "method": method,
                        "initial_class": initial_class,
                        **_aggregate(selected),
                    }
                )

    gate: dict[str, Any] = {
        "source_id": SOURCE_ID,
        "criterion": {
            "subset_initial_class": "head_on",
            "minimum_resolved_ratio": 0.6,
            "canonical_method": METHODS[0],
            "fixed_methods": list(FIXED_METHODS),
        },
        "opponents": {},
    }
    for opponent in OPPONENTS:
        head_rows = [row for row in summary_rows if row["opponent"] == opponent and row["initial_class"] == "head_on"]
        by_method = {str(row["method"]): row for row in head_rows}
        canonical_ratio = by_method[METHODS[0]]["resolved_ratio"]
        fixed_ratios = {method: by_method[method]["resolved_ratio"] for method in FIXED_METHODS}
        passed = bool(
            canonical_ratio is not None
            and canonical_ratio >= 0.6
            and any(ratio is not None and ratio >= 0.6 for ratio in fixed_ratios.values())
        )
        gate["opponents"][opponent] = {
            "passed": passed,
            "label": "terminal_win_rate_discriminative" if passed else "terminal_win_rate_not_discriminative",
            "canonical_resolved_ratio": canonical_ratio,
            "fixed_resolved_ratios": fixed_ratios,
            "head_on_summary_by_method": by_method,
        }

    pairing = _pairing_verification(episode_rows)
    if not pairing["all_cells_passed"]:
        raise ValueError("Heldout40 pairing verification failed after record validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "heldout40_episode_metrics.csv", episode_rows)
    _write_csv(output_dir / "heldout40_summary.csv", summary_rows)
    (output_dir / "heldout40_terminal_discriminability_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "heldout40_pairing_verification.json").write_text(
        json.dumps(pairing, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = {
        "source_id": SOURCE_ID,
        "manifest": str(manifest_path),
        "n_episode_rows": len(episode_rows),
        "n_summary_rows": len(summary_rows),
        "opponents_reported_separately": list(OPPONENTS),
        "gate": gate,
        "pairing": {
            "expected_cells": pairing["expected_cells"],
            "actual_cells": pairing["actual_cells"],
            "all_cells_passed": pairing["all_cells_passed"],
        },
        "artifacts": {
            "episode_metrics_csv": "heldout40_episode_metrics.csv",
            "summary_csv": "heldout40_summary.csv",
            "terminal_discriminability_gate": "heldout40_terminal_discriminability_gate.json",
            "pairing_verification": "heldout40_pairing_verification.json",
            "execution_report": "heldout40_execution_report.md",
        },
    }
    (output_dir / "heldout40_analysis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "heldout40_execution_report.md").write_text(
        _execution_report_markdown(gate=gate, pairing=pairing, run_dirs=run_dirs),
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expert-run", type=Path, required=True)
    parser.add_argument("--end-to-end-run", type=Path, required=True)
    parser.add_argument("--independent-ppo-vpp-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        run_dirs={
            "expert": args.expert_run,
            "end_to_end": args.end_to_end_run,
            "independent_ppo_vpp": args.independent_ppo_vpp_run,
        },
        manifest_path=args.manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps(report["gate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
