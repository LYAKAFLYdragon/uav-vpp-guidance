"""Evaluate strict-JSBSim v2 phase reachability with a physical-continuity ledger."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(REPO_ROOT / "src"))

from uav_vpp_guidance.evaluation.phase_reachability_handoff_contract import (  # noqa: E402
    PhaseReachabilityContractError,
    validate_handoff_continuity,
)


SOURCE_ID = "THESIS-PHASE-REACHABILITY-V2-RUN-IN-HANDOFF-R1"
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


def _phase_counts(record: Mapping[str, Any], gate: Mapping[str, Any]) -> dict[str, int]:
    frames = [frame for frame in record.get("trajectory", []) if isinstance(frame, Mapping)]
    if not frames:
        raise ValueError("Missing trajectory")
    consecutive = post_merge_steps = re_entry_steps = 0
    threshold = float(gate["re_entry_rate_threshold_mps"])
    required_consecutive = int(gate["re_entry_consecutive_post_merge_steps"])
    for frame in frames:
        if _truthy(frame.get("pre_merge")):
            consecutive = 0
            continue
        post_merge_steps += 1
        rate = _finite(frame.get("range_rate_mps"))
        consecutive = consecutive + 1 if rate is not None and rate <= threshold else 0
        if consecutive >= required_consecutive:
            re_entry_steps += 1
    return {
        "trajectory_steps": len(frames),
        "post_merge_steps": post_merge_steps,
        "re_entry_steps": re_entry_steps,
    }


def _continuity_status(record: Mapping[str, Any]) -> tuple[bool, str | None]:
    ledger = record.get("phase_reachability_handoff_ledger")
    if not isinstance(ledger, Mapping):
        return False, "missing_sidecar_ledger"
    if ledger.get("sidecar_mode") != "observe_only" or ledger.get("action_replacement_permitted") is not False:
        return False, "sidecar_not_observe_only"
    boundary = ledger.get("run_in_boundary")
    sampler = ledger.get("sampler_observation")
    if not isinstance(boundary, Mapping) or not isinstance(sampler, Mapping):
        return False, str(ledger.get("continuity_error") or "missing_k_to_k_plus_1_ledger")
    try:
        validate_handoff_continuity(boundary, sampler)
    except PhaseReachabilityContractError as error:
        return False, str(error)
    return True, None


def analyze(*, manifest_path: Path, run_dirs: Mapping[str, Path], output_dir: Path) -> dict[str, Any]:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    scenarios = manifest.get("scenarios", [])
    if manifest.get("source_id") != SOURCE_ID or len(scenarios) != 12:
        raise ValueError("v2 manifest identity mismatch")
    if manifest.get("independence", {}).get("v1_manifest_reused") is not False:
        raise ValueError("v2 must not reuse the v1 manifest")
    if set(run_dirs) != set(OPPONENTS):
        raise ValueError("Expected all three v2 opponents")
    gate = manifest["phase_gate"]
    expected_scenarios = {str(item["name"]) for item in scenarios}
    rows: list[dict[str, Any]] = []
    inputs: dict[str, Any] = {
        "manifest": {"path": str(manifest_path), "sha256": _sha256_file(manifest_path)},
        "formal_episode_records": {},
    }
    opponent_gate: dict[str, Any] = {}
    for opponent in OPPONENTS:
        path = run_dirs[opponent] / "aggregate" / "episode_records.json"
        inputs["formal_episode_records"][opponent] = {"path": str(path), "sha256": _sha256_file(path)}
        episodes = json.loads(path.read_text(encoding="utf-8")).get("episodes", [])
        selected = [record for record in episodes if record.get("controller") == REFERENCE_METHOD]
        if len(selected) != len(scenarios):
            raise ValueError(f"Expected {len(scenarios)} reference episodes for {opponent}")
        if {record.get("scenario") for record in selected} != expected_scenarios:
            raise ValueError(f"Scenario set mismatch for {opponent}")
        for record in selected:
            if (
                record.get("opponent_stage") != opponent
                or record.get("backend") != "jsbsim"
                or not _truthy(record.get("strict_backend"))
                or _truthy(record.get("backend_fallback_occurred"))
            ):
                raise ValueError(f"Strict JSBSim evidence is incomplete for {opponent}")
            metrics = _phase_counts(record, gate)
            continuity_valid, continuity_error = _continuity_status(record)
            rows.append(
                {
                    "opponent": opponent,
                    "scenario": record["scenario"],
                    "seed": record["seed"],
                    **metrics,
                    "continuity_valid": continuity_valid,
                    "continuity_error": continuity_error,
                    "terminal_reason": record.get("combat_reason"),
                }
            )
        opponent_rows = [row for row in rows if row["opponent"] == opponent]
        qualified = [
            row
            for row in opponent_rows
            if row["continuity_valid"]
            and row["post_merge_steps"] >= int(gate["post_merge_step_minimum_per_episode"])
            and row["re_entry_steps"] >= int(gate["re_entry_step_minimum_per_episode"])
        ]
        ratio = len(qualified) / len(opponent_rows)
        opponent_gate[opponent] = {
            "episodes": len(opponent_rows),
            "continuity_valid_episodes": sum(1 for row in opponent_rows if row["continuity_valid"]),
            "qualified_episodes": len(qualified),
            "qualified_fraction": ratio,
            "passed": ratio >= float(gate["minimum_qualifying_episode_fraction_per_opponent"]),
        }
    passed = all(item["passed"] for item in opponent_gate.values())
    report = {
        "source_id": SOURCE_ID,
        "mode": "evaluation_only_observe_only_sidecar",
        "reference_controller": REFERENCE_METHOD,
        "input_artifacts": inputs,
        "gate": opponent_gate,
        "verdict": "phase_reachability_v2_established" if passed else gate["failure_verdict"],
        "training_authorized": False,
        "stop_rule": manifest["protocol"]["stop_rule"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (output_dir / "phase_reachability_v2_episode_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "phase_reachability_v2_gate.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Phase Reachability v2 Gate",
        "",
        f"Verdict: `{report['verdict']}`",
        "",
        "This is a strict JSBSim physical-continuity test. It does not authorise training, tuning, action replacement, or a v1 rerun.",
    ]
    for opponent, result in opponent_gate.items():
        lines.append(
            f"- `{opponent}`: continuity {result['continuity_valid_episodes']}/{result['episodes']}; "
            f"qualified {result['qualified_episodes']}/{result['episodes']}; pass={result['passed']}."
        )
    (output_dir / "phase_reachability_v2_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expert-run", type=Path, required=True)
    parser.add_argument("--end-to-end-run", type=Path, required=True)
    parser.add_argument("--independent-ppo-vpp-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        manifest_path=args.manifest,
        run_dirs={"expert": args.expert_run, "end_to_end": args.end_to_end_run, "independent_ppo_vpp": args.independent_ppo_vpp_run},
        output_dir=args.output_dir,
    )
    print(json.dumps({"verdict": result["verdict"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
