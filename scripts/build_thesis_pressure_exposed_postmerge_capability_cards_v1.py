#!/usr/bin/env python3
"""Build opponent-specific pressure cards from the corrected frozen cohort."""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from uav_vpp_guidance.evaluation.engagement_geometry_taxonomy import classify_relative_geometry  # noqa: E402

RUNNER_PATH = ROOT / "scripts" / "run_thesis_disadvantage_pressure_exposed_postmerge_collector_v1.py"
CORRECTION_FILENAME = "pressure_gate_taxonomy_correction_r1.json"


def _runner():
    spec = importlib.util.spec_from_file_location("pressure_collector", RUNNER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def build(output_root: Path, config_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    runner = _runner()
    config = runner.load_config(config_path)
    correction = json.loads((output_root / CORRECTION_FILENAME).read_text(encoding="utf-8"))
    if correction.get("corrected_gate", {}).get("passes") is not True:
        raise ValueError("corrected pressure gate did not pass")
    index = json.loads((output_root / "records_index.json").read_text(encoding="utf-8"))["records"]
    rows = []
    for entry in index:
        record = json.loads((output_root / entry["path"]).read_text(encoding="utf-8"))["record"]
        trace = copy.deepcopy(record["phase_trace"])
        for step in trace:
            step["dynamic_state"] = classify_relative_geometry(step["aa_deg"], step["ata_deg"])
        window = runner.find_first_pressure_window(trace, config["gates"])
        if window is None:
            raise ValueError("corrected passing gate contains an unqualified record")
        actions = [step["reference_action"] for step in window]
        nz = [value for step in window if (value := _finite(step.get("actual_nz_g"))) is not None]
        rows.append({
            "opponent": record["opponent"],
            "scenario_name": record["scenario_name"],
            "scenario_signature": record["scenario_signature"],
            "mirror_sign": record["mirror_sign"],
            "window_start_step": window[0]["step"],
            "target_attack_exposure_fraction": _mean([float(step["target_in_attack_zone"]) for step in window]),
            "mean_aa_deg": _mean([float(step["aa_deg"]) for step in window]),
            "mean_ata_deg": _mean([float(step["ata_deg"]) for step in window]),
            "mean_abs_vpp_longitudinal": _mean([abs(float(action[0])) for action in actions]),
            "mean_abs_vpp_lateral": _mean([abs(float(action[1])) for action in actions]),
            "mean_abs_vpp_vertical": _mean([abs(float(action[2])) for action in actions]),
            "mean_actual_nz_g": _mean(nz),
            "terminal_reason": record["result"]["terminal_reason"],
        })
    cards = []
    for opponent in runner.OPPONENTS:
        group = [row for row in rows if row["opponent"] == opponent]
        cards.append({
            "opponent": opponent,
            "episodes": len(group),
            "distinct_signatures": len({row["scenario_signature"] for row in group}),
            "mirrors": sorted({row["mirror_sign"] for row in group}),
            "mean_target_attack_exposure_fraction": _mean([row["target_attack_exposure_fraction"] for row in group]),
            "mean_aa_deg": _mean([row["mean_aa_deg"] for row in group]),
            "mean_ata_deg": _mean([row["mean_ata_deg"] for row in group]),
            "mean_abs_vpp_longitudinal": _mean([row["mean_abs_vpp_longitudinal"] for row in group]),
            "mean_abs_vpp_lateral": _mean([row["mean_abs_vpp_lateral"] for row in group]),
            "mean_abs_vpp_vertical": _mean([row["mean_abs_vpp_vertical"] for row in group]),
            "mean_actual_nz_g": _mean([row["mean_actual_nz_g"] for row in group]),
            "terminal_reasons": {reason: sum(row["terminal_reason"] == reason for row in group) for reason in sorted({row["terminal_reason"] for row in group})},
            "interpretation_boundary": "pressure/interface profile under the common frozen fixed-head-on reference; not Elo, absolute opponent strength, or a physical range/energy card",
        })
    return rows, {"cards": cards, "physical_telemetry_limit": "V1 phase trace stores normalized observation features, not raw physical range/speed/altitude; those metrics are intentionally absent."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_root / "capability_cards_taxonomy_corrected_r1"
    if output_dir.exists():
        parser.error("refusing to overwrite existing capability cards")
    rows, result = build(args.output_root.resolve(), args.config.resolve())
    output_dir.mkdir(parents=True)
    with (output_dir / "pressure_window_index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "opponent_capability_cards.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Pressure Capability Cards (Taxonomy-Corrected R1)", "", "All cards use the same 16-scenario frozen fixed-head-on cohort and are reported separately.", "", "| Opponent | Episodes | Pressure exposure | Mean AA | Mean ATA |", "| --- | ---: | ---: | ---: | ---: |"]
    for card in result["cards"]:
        lines.append(f"| {card['opponent']} | {card['episodes']} | {card['mean_target_attack_exposure_fraction']:.3f} | {card['mean_aa_deg']:.1f} | {card['mean_ata_deg']:.1f} |")
    lines.extend(["", "Physical range, speed, altitude and energy are intentionally not reported because V1 preserved normalized observation features rather than raw physical telemetry."])
    (output_dir / "opponent_capability_cards.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
