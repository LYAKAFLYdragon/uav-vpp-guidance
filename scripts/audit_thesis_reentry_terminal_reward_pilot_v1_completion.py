"""Read-only completion audit for the terminal-reward pilot V1.

The pilot's runner writes its scientific artifacts before rendering the final
JSON to stdout.  This audit independently reconstructs the Dev paired deltas
from those persisted records, verifies checkpoint selection, and separates a
post-completion output-channel error from an incomplete scientific run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence


SOURCE_ID = "THESIS-REENTRY-TERMINAL-REWARD-PILOT-V1"
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
RUNNER_PATH = "scripts/run_thesis_reentry_terminal_reward_pilot_v1.py"


class CompletionAuditError(ValueError):
    """Raised when persisted pilot artifacts cannot support a completion audit."""


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise CompletionAuditError(f"missing required artifact: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _qualified_by_signature(records: Iterable[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    return {
        str(record["scenario_signature"]): record
        for record in records
        if bool(record.get("qualifying"))
    }


def recompute_dev_evaluation(
    candidate: Mapping[str, Sequence[Mapping[str, Any]]],
    baseline: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> Dict[str, Any]:
    """Rebuild the frozen minimax Dev score without importing the pilot runtime."""
    per_opponent: Dict[str, Any] = {}
    worst_delta = -math.inf
    safety_pass = True
    for opponent in OPPONENTS:
        candidate_records = candidate.get(opponent, [])
        head_on_records = baseline.get(opponent, {}).get("frozen_fixed_head_on", [])
        head_on_by_signature = _qualified_by_signature(head_on_records)
        paired = [
            float(record["intent_loss_auc20"]) - float(head_on_by_signature[str(record["scenario_signature"])]["intent_loss_auc20"])
            for record in candidate_records
            if bool(record.get("qualifying"))
            and str(record["scenario_signature"]) in head_on_by_signature
        ]
        candidate_crash_rate = sum(bool(record.get("ego_failure")) for record in candidate_records) / max(len(candidate_records), 1)
        head_on_crash_rate = sum(bool(record.get("ego_failure")) for record in head_on_records) / max(len(head_on_records), 1)
        mean_delta = _mean(paired)
        crash_delta = candidate_crash_rate - head_on_crash_rate
        passed_safety = crash_delta <= 0.05
        worst_delta = max(worst_delta, math.inf if mean_delta is None else mean_delta)
        safety_pass = safety_pass and passed_safety
        per_opponent[opponent] = {
            "paired_count": len(paired),
            "mean_paired_intent_loss_delta_vs_head_on": mean_delta,
            "ego_crash_oob_delta_vs_head_on": crash_delta,
            "safety_pass": passed_safety,
        }
    return {
        "selection_score_worst_opponent_mean_delta": worst_delta,
        "safety_pass": safety_pass,
        "per_opponent": per_opponent,
    }


def _close(left: Any, right: Any, *, tolerance: float = 1e-12) -> bool:
    return isinstance(left, (int, float)) and isinstance(right, (int, float)) and math.isclose(float(left), float(right), abs_tol=tolerance, rel_tol=0.0)


def _summary_matches(recomputed: Mapping[str, Any], stored: Mapping[str, Any]) -> bool:
    if bool(recomputed["safety_pass"]) != bool(stored.get("safety_pass")):
        return False
    if not _close(recomputed["selection_score_worst_opponent_mean_delta"], stored.get("selection_score_worst_opponent_mean_delta")):
        return False
    for opponent in OPPONENTS:
        expected = recomputed["per_opponent"][opponent]
        observed = stored.get("per_opponent", {}).get(opponent, {})
        if expected["paired_count"] != observed.get("paired_count"):
            return False
        if not _close(expected["mean_paired_intent_loss_delta_vs_head_on"], observed.get("mean_paired_intent_loss_delta_vs_head_on")):
            return False
        if not _close(expected["ego_crash_oob_delta_vs_head_on"], observed.get("ego_crash_oob_delta_vs_head_on")):
            return False
        if bool(expected["safety_pass"]) != bool(observed.get("safety_pass")):
            return False
    return True


def _runner_stdout_hypothesis(repository: Optional[Path], git_sha: Any) -> Dict[str, Any]:
    if repository is None or not isinstance(git_sha, str) or not git_sha:
        return {"checked": False, "supported": False, "reason": "repository_or_git_sha_unavailable"}
    command = ["git", "show", f"{git_sha}:{RUNNER_PATH}"]
    completed = subprocess.run(command, cwd=repository, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode:
        return {"checked": True, "supported": False, "reason": "git_show_failed"}
    source = completed.stdout
    manifest_write = source.find('"run_manifest.json"')
    stdout_print = source.find("print(json.dumps(result, indent=2))")
    except_block = source.find("except Exception as error")
    failure_write = source.find('"failure.json"')
    supported = -1 not in (manifest_write, stdout_print, except_block, failure_write) and manifest_write < stdout_print < except_block < failure_write
    return {
        "checked": True,
        "supported": supported,
        "reason": "manifest_then_stdout_inside_exception_handler" if supported else "runner_order_not_confirmed",
        "runner_git_sha": git_sha,
    }


def audit_completion(result_root: Path, *, repository: Optional[Path] = None, expected_training_steps: int = 20000) -> Dict[str, Any]:
    """Audit persisted V1 pilot evidence without modifying it."""
    result_root = result_root.resolve()
    baseline = _load_json(result_root / "dev_baseline.json")
    summary = _load_json(result_root / "training_summary.json")
    manifest = _load_json(result_root / "run_manifest.json")
    failure = _load_json(result_root / "failure.json")
    candidate_paths = sorted(
        (result_root / "dev_candidate").glob("step_*.json"),
        key=lambda path: int(path.stem.split("_")[-1]),
    )
    if not candidate_paths:
        raise CompletionAuditError("missing Dev candidate artifacts")

    evaluations = []
    for path in candidate_paths:
        candidate_artifact = _load_json(path)
        stored = candidate_artifact.get("summary", {})
        recomputed = recompute_dev_evaluation(candidate_artifact.get("records", {}), baseline)
        step = int(stored.get("step", path.stem.split("_")[-1]))
        evaluations.append({
            "step": step,
            "artifact": str(path),
            "stored_summary_matches_recomputed": _summary_matches(recomputed, stored),
            "recomputed": recomputed,
        })
    eligible = [
        item for item in evaluations
        if item["stored_summary_matches_recomputed"]
        and item["recomputed"]["safety_pass"]
        and math.isfinite(item["recomputed"]["selection_score_worst_opponent_mean_delta"])
    ]
    selected = min(eligible, key=lambda item: (item["recomputed"]["selection_score_worst_opponent_mean_delta"], item["step"])) if eligible else None
    if selected is None:
        raise CompletionAuditError("no independently valid Dev checkpoint")

    selected_checkpoint = result_root / "checkpoints" / f"step_{selected['step']}.pt"
    best_checkpoint = result_root / "checkpoints" / "best.pt"
    if not selected_checkpoint.is_file() or not best_checkpoint.is_file():
        raise CompletionAuditError("selected checkpoint or best.pt is missing")
    completion_paths = [
        result_root / "training_summary.json",
        result_root / "run_manifest.json",
        result_root / "failure.json",
    ]
    timestamps = {path.name: path.stat().st_mtime_ns for path in completion_paths}
    artifacts_precede_failure = timestamps["training_summary.json"] <= timestamps["run_manifest.json"] <= timestamps["failure.json"]
    practical_threshold = -0.05
    practical_improvement_opponents = [
        opponent for opponent in OPPONENTS
        if selected["recomputed"]["per_opponent"][opponent]["mean_paired_intent_loss_delta_vs_head_on"] <= practical_threshold
    ]
    stdout_hypothesis = _runner_stdout_hypothesis(repository, manifest.get("git_sha"))
    completion_recorded = (
        summary.get("source_id") == SOURCE_ID
        and manifest.get("source_id") == SOURCE_ID
        and int(summary.get("training_steps", -1)) == expected_training_steps
        and not bool(summary.get("stopped"))
        and not bool(summary.get("heldout_executed"))
        and artifacts_precede_failure
        and _sha256(selected_checkpoint) == _sha256(best_checkpoint)
    )
    classification = "post_completion_output_channel_failure" if completion_recorded and stdout_hypothesis["supported"] and str(failure.get("error")) == "OSError(22, 'Invalid argument')" else "infrastructure_failure_attribution_inconclusive"
    return {
        "source_id": SOURCE_ID,
        "result_root": str(result_root),
        "training": {
            "expected_steps": expected_training_steps,
            "recorded_steps": summary.get("training_steps"),
            "attempts": summary.get("attempts"),
            "stopped": bool(summary.get("stopped")),
            "stop_reason": summary.get("stop_reason"),
            "heldout_executed": bool(summary.get("heldout_executed")),
        },
        "completion_timestamps_ns": timestamps,
        "completion_artifacts_precede_failure": artifacts_precede_failure,
        "failure": failure,
        "runner_stdout_hypothesis": stdout_hypothesis,
        "dev_recomputation": evaluations,
        "selected_checkpoint": {
            "step": selected["step"],
            "best_matches_selected_sha256": _sha256(selected_checkpoint) == _sha256(best_checkpoint),
            "best_sha256": _sha256(best_checkpoint),
        },
        "decision": {
            "dev_safety_pass": selected["recomputed"]["safety_pass"],
            "practical_improvement_threshold": practical_threshold,
            "practical_improvement_opponents": practical_improvement_opponents,
            "heldout_permitted": False,
            "heldout_must_remain_locked": True,
        },
        "classification": classification,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_completion(args.result_root, repository=args.repository)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
