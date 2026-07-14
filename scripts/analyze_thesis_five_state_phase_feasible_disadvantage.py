"""Evaluate whether a frozen reference rollout reaches sustained post-merge and re-entry."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


SOURCE_ID = "THESIS-FIVE-STATE-PHASE-FEASIBLE-DISADVANTAGE-V1"
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
REFERENCE_METHOD = "legacy_static_oracle_task_gate"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _truthy(value: Any) -> bool:
    return value.strip().lower() in {"true", "1", "yes"} if isinstance(value, str) else bool(value)


def _finite(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _episode_phase_counts(record: Mapping[str, Any], gate: Mapping[str, Any]) -> dict[str, Any]:
    frames = [frame for frame in record.get("trajectory", []) if isinstance(frame, Mapping)]
    if not frames:
        raise ValueError("Missing trajectory")
    required_consecutive = int(gate["re_entry_consecutive_post_merge_steps"])
    threshold = float(gate["re_entry_rate_threshold_mps"])
    post_steps = 0
    reentry_steps = 0
    consecutive = 0
    for frame in frames:
        if _truthy(frame.get("pre_merge")):
            consecutive = 0
            continue
        post_steps += 1
        rate = _finite(frame.get("range_rate_mps"))
        consecutive = consecutive + 1 if rate is not None and rate <= threshold else 0
        if consecutive >= required_consecutive:
            reentry_steps += 1
    return {"post_merge_steps": post_steps, "re_entry_steps": reentry_steps, "trajectory_steps": len(frames)}


def analyze(*, manifest_path: Path, run_dirs: Mapping[str, Path], output_dir: Path) -> dict[str, Any]:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    if manifest.get("source_id") != SOURCE_ID or len(manifest.get("scenarios", [])) != 12:
        raise ValueError("Phase-feasible manifest identity mismatch")
    gate = manifest["phase_gate"]
    if set(run_dirs) != set(OPPONENTS):
        raise ValueError("Expected all three opponents")
    rows = []
    input_records = {"manifest": {"path": str(manifest_path), "sha256": _sha256_file(manifest_path)}, "formal_episode_records": {}}
    opponent_gate = {}
    for opponent in OPPONENTS:
        path = run_dirs[opponent] / "aggregate" / "episode_records.json"
        input_records["formal_episode_records"][opponent] = {"path": str(path), "sha256": _sha256_file(path)}
        episodes = json.loads(path.read_text(encoding="utf-8")).get("episodes", [])
        selected = [record for record in episodes if record.get("controller") == REFERENCE_METHOD]
        if len(selected) != 12 or any(record.get("opponent_stage") != opponent or record.get("backend") != "jsbsim" or not _truthy(record.get("strict_backend")) for record in selected):
            raise ValueError(f"Strict reference evidence is incomplete for {opponent}")
        for record in selected:
            metrics = _episode_phase_counts(record, gate)
            rows.append({"opponent": opponent, "scenario": record["scenario"], "seed": record["seed"], **metrics, "terminal_reason": record.get("combat_reason")})
        opponent_rows = [row for row in rows if row["opponent"] == opponent]
        qualified = [
            row for row in opponent_rows
            if row["post_merge_steps"] >= int(gate["post_merge_step_minimum_per_episode"])
            and row["re_entry_steps"] >= int(gate["re_entry_step_minimum_per_episode"])
        ]
        ratio = len(qualified) / len(opponent_rows)
        opponent_gate[opponent] = {"episodes": len(opponent_rows), "qualified_episodes": len(qualified), "qualified_fraction": ratio, "passed": ratio >= float(gate["minimum_qualifying_episode_fraction_per_opponent"])}
    passed = all(item["passed"] for item in opponent_gate.values())
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "phase_feasible_disadvantage_episode_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {"source_id": SOURCE_ID, "mode": "evaluation_only", "reference_controller": REFERENCE_METHOD, "input_artifacts": input_records, "gate": opponent_gate, "verdict": "phase_feasible_envelope_established" if passed else gate["failure_verdict"]}
    (output_dir / "phase_feasible_disadvantage_gate.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = ["# Phase-Feasible Disadvantage Gate", "", f"Verdict: `{report['verdict']}`", "", "The gate is a reachability test only; it does not score combat quality or authorize training."]
    for opponent, result in opponent_gate.items():
        lines.append(f"- `{opponent}`: {result['qualified_episodes']}/{result['episodes']} qualified; pass={result['passed']}.")
    (output_dir / "phase_feasible_disadvantage_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expert-run", type=Path, required=True)
    parser.add_argument("--end-to-end-run", type=Path, required=True)
    parser.add_argument("--independent-ppo-vpp-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(manifest_path=args.manifest, run_dirs={"expert": args.expert_run, "end_to_end": args.end_to_end_run, "independent_ppo_vpp": args.independent_ppo_vpp_run}, output_dir=args.output_dir)
    print(json.dumps({"verdict": result["verdict"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
