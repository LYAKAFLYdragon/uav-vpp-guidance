#!/usr/bin/env python3
"""Read-only feasibility check for a new bounded temporal-corridor action interface."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from uav_vpp_guidance.evaluation.thesis_temporal_corridor_feasibility import summarize_rows
from uav_vpp_guidance.virtual_point.temporal_corridor import TemporalCorridorConfig, compile_temporal_corridor_action


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _records(root: Path):
    for path in sorted(root.rglob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8")).get("record", {})
        if record.get("method") == "progress_triggered_reference_path" and record.get("window"):
            yield record


def analyze(config_path: Path) -> dict:
    config = _load(config_path)
    if config.get("kind") != "read_only_temporal_corridor_feasibility":
        raise RuntimeError("temporal-corridor config identity drifted")
    root = Path(config["inputs"]["v16_output_root"])
    rows = []
    for record in _records(root):
        for candidate in config["candidates"]:
            compiler_config = TemporalCorridorConfig(**candidate["compiler"])
            previous = np.zeros(3, dtype=np.float64)
            for step in record["window"]:
                receipt = compile_temporal_corridor_action(
                    own_position_neu=step["own_position_neu"],
                    predicted_target_position_neu=step["predicted_target_position_neu"],
                    phase_step=int(step["local_step"]),
                    previous_action=previous,
                    raw_action_ranges=config["raw_action_ranges"],
                    config=compiler_config,
                )
                previous = np.asarray(receipt["applied_action"], dtype=np.float64)
                rows.append(
                    {
                        "candidate_id": candidate["id"],
                        "opponent": record["opponent"],
                        "pair_key": record["pair_key"],
                        "local_step": step["local_step"],
                        "applied_action_norm": float(np.linalg.norm(previous)),
                        **receipt,
                    }
                )
    if not rows:
        raise RuntimeError("V16 raw telemetry contains no replayable paths")
    gate = summarize_rows(rows, **config["gate"])
    return {"config": config, "rows": rows, "gate": gate}


def write_outputs(result: dict, output: Path, report: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    rows = result["rows"]
    with (output / "temporal_corridor_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, list) else value for key, value in row.items()})
    summary = {"analysis_kind": "read_only_not_a_scientific_source", "config": result["config"], "gate": result["gate"]}
    (output / "temporal_corridor_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Bounded Temporal Corridor Compiler 可执行性分析（只读）",
        "",
        "本分析仅利用冻结 V16 state telemetry 构造当前信息 corridor action；不运行 JSBSim、不训练，且不主张 AA/ATA/能量改善。",
        "",
        f"`passing_candidates={result['gate']['passing_candidates']}`；`passes={result['gate']['passes']}`。",
        "",
        "| Candidate | Opponent | Episodes | Steps | Mean projection residual (m) | Peak action norm | Hard-bound steps | Pass |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for candidate_id, opponent_matrix in result["gate"]["matrix"].items():
        for opponent, cell in opponent_matrix.items():
            lines.append(
                f"| {candidate_id} | {opponent} | {cell['qualifying_episodes']} | {cell['steps']} | {cell['mean_projection_residual_m']:.3f} | {cell['max_applied_action_norm']:.3f} | {cell['hard_bound_steps']} | {cell['passes']} |"
            )
    lines.extend(
        [
            "",
            "通过仅授权起草全新、非学习 JSBSim action-to-trajectory transfer probe；它不授权训练。",
            "不通过则说明该 action parameterization 在当前冻结包线内仍无可执行候选。",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise RuntimeError("temporal-corridor outputs must be absent")
    result = analyze(args.config)
    write_outputs(result, args.output, args.report)
    print(json.dumps({"output": str(args.output), **result["gate"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
