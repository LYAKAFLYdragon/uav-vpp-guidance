#!/usr/bin/env python3
"""Read-only analysis for the verified pressure-exposed M0--M4 source."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


SOURCE_ID = "THESIS-DISADVANTAGE-PRESSURE-EXPOSED-M0M4-3OPP-V1"
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
METHODS = ("fixed_head_on", "fixed_crossing", "static_oracle_gate", "canonical_ppo", "ppo_unbootstrapped_crossing")
DEFAULT_ROOT = Path("E:/uav-vpp-guidance-disadvantage-capability-v1-results/pressure_exposed_m0m4_compact_ablation_v1")


class AnalysisError(RuntimeError):
    pass


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _mean(values: Iterable[Any]) -> float | None:
    clean = [value for item in values if (value := _finite(item)) is not None]
    return sum(clean) / len(clean) if clean else None


def _phase(step: Mapping[str, Any]) -> str:
    if bool(step.get("pre_merge")):
        return "pre_merge"
    rate = _finite(step.get("range_rate_mps"))
    return "re_entry" if rate is not None and rate < 0.0 else "post_merge"


def _fraction(trajectory: list[Mapping[str, Any]], field: str) -> float:
    return sum(bool(step.get(field, False)) for step in trajectory) / len(trajectory)


def _episode_metrics(record: Mapping[str, Any]) -> dict[str, Any]:
    trajectory = record.get("trajectory")
    if not isinstance(trajectory, list) or not trajectory:
        raise AnalysisError("raw trajectory is unavailable")
    phases = Counter(_phase(step) for step in trajectory)
    return {
        "opponent": record["opponent_stage"],
        "scenario": record["scenario"],
        "method": record["controller"],
        "outcome": record["outcome"],
        "terminal_reason": record["terminal_reason"],
        "hp_advantage": _finite(record.get("hp_advantage")),
        "total_reward": _finite(record.get("total_reward")),
        "mean_range_m": _mean(step.get("range_m") for step in trajectory),
        "target_attack_zone_fraction": _fraction(trajectory, "target_in_attack_zone"),
        "ego_attack_zone_fraction": _fraction(trajectory, "ego_in_attack_zone"),
        "pre_merge_fraction": phases["pre_merge"] / len(trajectory),
        "post_merge_fraction": phases["post_merge"] / len(trajectory),
        "re_entry_fraction": phases["re_entry"] / len(trajectory),
        "mean_vpp_action_x": _mean(step.get("vpp_action", [None])[0] for step in trajectory),
        "mean_vpp_action_y": _mean(step.get("vpp_action", [None, None])[1] for step in trajectory),
        "mean_vpp_action_z": _mean(step.get("vpp_action", [None, None, None])[2] for step in trajectory),
    }


def _action_l1(left: Mapping[str, Any], right: Mapping[str, Any]) -> float | None:
    left_steps, right_steps = left.get("trajectory"), right.get("trajectory")
    if not isinstance(left_steps, list) or not isinstance(right_steps, list) or len(left_steps) != len(right_steps):
        return None
    values = []
    for left_step, right_step in zip(left_steps, right_steps):
        left_action, right_action = left_step.get("vpp_action"), right_step.get("vpp_action")
        if not isinstance(left_action, list) or not isinstance(right_action, list) or len(left_action) != 3 or len(right_action) != 3:
            return None
        values.extend(abs(float(a) - float(b)) for a, b in zip(left_action, right_action))
    return sum(values)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise AnalysisError("refusing to write an empty result table")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    verification = json.loads((root / "formal_verification.json").read_text(encoding="utf-8"))
    if verification.get("source_id") != SOURCE_ID or verification.get("verdict") != "pass" or int(verification.get("records", 0)) != 240:
        raise AnalysisError("requires a passing verified 240-record source")
    records: list[dict[str, Any]] = []
    raw_by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for opponent in OPPONENTS:
        path = root / "{}-{}".format(SOURCE_ID, opponent) / "aggregate" / "episode_records.json"
        episodes = json.loads(path.read_text(encoding="utf-8")).get("episodes")
        if not isinstance(episodes, list) or len(episodes) != 80:
            raise AnalysisError("{} does not provide 80 episodes".format(opponent))
        for record in episodes:
            key = (opponent, str(record["scenario"]), str(record["controller"]))
            raw_by_key[key] = record
            records.append(_episode_metrics(record))
    if len(records) != 240:
        raise AnalysisError("matrix is incomplete")

    summary: list[dict[str, Any]] = []
    for opponent in OPPONENTS:
        for method in METHODS:
            group = [row for row in records if row["opponent"] == opponent and row["method"] == method]
            if len(group) != 16:
                raise AnalysisError("{} / {} is not a 16-episode cell".format(opponent, method))
            terminals = Counter(row["terminal_reason"] for row in group)
            summary.append({
                "opponent": opponent, "method": method, "n": len(group),
                "wins": sum(row["outcome"] == "win" for row in group), "losses": sum(row["outcome"] == "loss" for row in group), "draws": sum(row["outcome"] == "draw" for row in group),
                "terminal_mix": json.dumps(dict(sorted(terminals.items())), sort_keys=True),
                **{key: _mean(row[key] for row in group) for key in ("hp_advantage", "total_reward", "mean_range_m", "target_attack_zone_fraction", "ego_attack_zone_fraction", "pre_merge_fraction", "post_merge_fraction", "re_entry_fraction", "mean_vpp_action_x", "mean_vpp_action_y", "mean_vpp_action_z")},
            })
    paired: list[dict[str, Any]] = []
    references = ("fixed_head_on", "fixed_crossing", "static_oracle_gate", "ppo_unbootstrapped_crossing")
    scenarios = sorted({row["scenario"] for row in records})
    for opponent in OPPONENTS:
        for reference in references:
            for scenario in scenarios:
                ppo = next(row for row in records if row["opponent"] == opponent and row["method"] == "canonical_ppo" and row["scenario"] == scenario)
                baseline = next(row for row in records if row["opponent"] == opponent and row["method"] == reference and row["scenario"] == scenario)
                paired.append({"opponent": opponent, "scenario": scenario, "reference_method": reference, "ppo_hp_advantage_delta": ppo["hp_advantage"] - baseline["hp_advantage"], "ppo_target_attack_fraction_delta": ppo["target_attack_zone_fraction"] - baseline["target_attack_zone_fraction"], "ppo_ego_attack_fraction_delta": ppo["ego_attack_zone_fraction"] - baseline["ego_attack_zone_fraction"], "ppo_range_delta_m": ppo["mean_range_m"] - baseline["mean_range_m"]})
    identity = {}
    for opponent in OPPONENTS:
        for left, right in (("canonical_ppo", "ppo_unbootstrapped_crossing"), ("fixed_head_on", "static_oracle_gate")):
            distances = [_action_l1(raw_by_key[(opponent, scenario, left)], raw_by_key[(opponent, scenario, right)]) for scenario in scenarios]
            identity["{}:{}_vs_{}".format(opponent, left, right)] = {"paired_episodes": len(distances), "action_identical_episodes": sum(distance == 0.0 for distance in distances), "mean_action_l1": _mean(distances)}
    result = {
        "source_id": SOURCE_ID,
        "verification_sha256": __import__("hashlib").sha256((root / "formal_verification.json").read_bytes()).hexdigest(),
        "records": len(records),
        "summary_cells": len(summary),
        "terminal_discriminative": any(row["wins"] or row["draws"] for row in summary),
        "treatment_identity": identity,
        "interpretation": "All rows remain opponent-separated. M3/M4 identity on the all-head_on task-bit cohort makes the crossing-bootstrap ablation non-informative; the source is valid descriptive evidence, not a five-independent-treatment causal result.",
    }
    return result, records, summary, paired


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output_dir.resolve() if args.output_dir else root / "analysis"
    if output.exists():
        parser.error("refusing to overwrite analysis output")
    try:
        result, records, summary, paired = analyze(root)
    except (OSError, ValueError, TypeError, AnalysisError) as error:
        parser.error(str(error))
    output.mkdir(parents=True)
    _write_csv(output / "episode_metrics.csv", records)
    _write_csv(output / "method_opponent_summary.csv", summary)
    _write_csv(output / "paired_ppo_deltas.csv", paired)
    (output / "analysis_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Pressure-Exposed M0--M4 Compact Ablation", "", "- Source: `{}`".format(SOURCE_ID), "- Verified records: `240/240`.", "- All results remain separated by opponent.", "- Terminal discrimination: `{}`.".format(result["terminal_discriminative"]), "- M3/M4 is non-informative here because every scenario carries the `head_on` task bit and their VPP actions are treatment-identical.", "- This valid source must not support a claim of five independent causal treatments or universal advantage."]
    (output / "analysis_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
