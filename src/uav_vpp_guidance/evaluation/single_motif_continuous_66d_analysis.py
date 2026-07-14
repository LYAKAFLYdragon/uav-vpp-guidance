"""Read-only gate analysis for the single-motif continuous 66-D collector."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from uav_vpp_guidance.evaluation.single_motif_continuous_66d import (
    REQUIRED_OPPONENTS,
    SOURCE_ID,
    SingleMotifPlan,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_collector_runs(run_dirs: Sequence[Path]) -> list[dict[str, Any]]:
    """Load and hash-verify independent per-episode collector artifacts."""

    episodes: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        root = Path(run_dir)
        manifest_path = root / "run_manifest.json"
        run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        index_path = root / "single_motif_continuous_66d" / "index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if index.get("source_id") != SOURCE_ID:
            raise ValueError(f"unexpected collector source in {index_path}")
        for entry in index.get("entries", []):
            artifact_path = root / str(entry["artifact_path"])
            actual_sha = sha256_file(artifact_path)
            if actual_sha != str(entry.get("sha256", "")):
                raise ValueError(f"collector artifact SHA mismatch: {artifact_path}")
            ledger = json.loads(artifact_path.read_text(encoding="utf-8"))
            if ledger.get("header", {}).get("source_id") != SOURCE_ID:
                raise ValueError(f"collector ledger source mismatch: {artifact_path}")
            episodes.append(
                {
                    "run_dir": str(root),
                    "artifact_path": str(artifact_path),
                    "artifact_sha256": actual_sha,
                    "run_manifest": run_manifest,
                    "index_entry": entry,
                    "ledger": ledger,
                }
            )
    return episodes


def evaluate_gate(
    episodes: Sequence[Mapping[str, Any]],
    plan: SingleMotifPlan,
) -> dict[str, Any]:
    """Apply every preregistered threshold independently per opponent."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        entry = episode["index_entry"]
        ledger = episode["ledger"]
        header = ledger["header"]
        summary = ledger["summary"]
        opponent = str(header["opponent"])
        grouped[opponent].append(episode)
        valid_steps = int(summary.get("valid_target_steps", 0))
        structural = bool(summary.get("all_ready_structural_contract_valid", False))
        prediction = bool(summary.get("all_ready_prediction_without_fallback", False))
        action_identity = bool(summary.get("all_actions_identity_preserved", False))
        no_reset = bool(summary.get("no_reset", False))
        no_padding = bool(summary.get("no_padding", False))
        strict_jsbsim = bool(summary.get("strict_jsbsim_lineage", False))
        v2_continuity = bool(entry.get("v2_continuity_valid", False))
        episode_contract_valid = all(
            (
                structural,
                prediction,
                action_identity,
                no_reset,
                no_padding,
                strict_jsbsim,
                v2_continuity,
            )
        )
        rows.append(
            {
                "opponent": opponent,
                "scenario": header.get("scenario"),
                "scenario_signature": header.get("scenario_signature"),
                "mirror_sign": header.get("mirror_sign"),
                "seed": header.get("seed"),
                "episode": header.get("episode"),
                "recorded_steps": int(summary.get("recorded_steps", 0)),
                "contract_ready_steps": int(summary.get("contract_ready_steps", 0)),
                "target_motif_steps": int(summary.get("target_motif_steps", 0)),
                "valid_target_steps": valid_steps,
                "episode_contract_valid": episode_contract_valid,
                "v2_continuity_valid": v2_continuity,
                "qualifying_episode": episode_contract_valid and valid_steps > 0,
                "artifact_path": episode["artifact_path"],
                "artifact_sha256": episode["artifact_sha256"],
            }
        )

    opponent_results: dict[str, Any] = {}
    for opponent in REQUIRED_OPPONENTS:
        opponent_rows = [row for row in rows if row["opponent"] == opponent]
        qualifying = [row for row in opponent_rows if row["qualifying_episode"]]
        valid_steps = sum(int(row["valid_target_steps"]) for row in qualifying)
        signatures = sorted({str(row["scenario_signature"]) for row in qualifying})
        mirrors = sorted({str(row["mirror_sign"]) for row in qualifying})
        checks = {
            "episodes_present": len(opponent_rows) > 0,
            "minimum_qualifying_episodes": len(qualifying) >= plan.min_episodes_per_opponent,
            "minimum_valid_target_steps": valid_steps >= plan.min_target_steps_per_opponent,
            "minimum_distinct_scenario_signatures": len(signatures) >= plan.min_signatures_per_opponent,
            "minimum_distinct_mirror_signs": len(mirrors) >= plan.min_mirror_signs_per_opponent,
            "all_episode_contracts_valid": bool(opponent_rows) and all(
                bool(row["episode_contract_valid"]) for row in opponent_rows
            ),
        }
        opponent_results[opponent] = {
            "episodes": len(opponent_rows),
            "qualifying_episodes": len(qualifying),
            "valid_target_steps": valid_steps,
            "distinct_scenario_signatures": signatures,
            "distinct_mirror_signs": mirrors,
            "checks": checks,
            "passed": all(checks.values()),
        }
    gate_passed = all(result["passed"] for result in opponent_results.values())
    run_provenance: dict[str, dict[str, Any]] = {}
    for episode in episodes:
        run_dir = str(episode["run_dir"])
        manifest = episode.get("run_manifest", {})
        git_info = manifest.get("git_info", {}) if isinstance(manifest, Mapping) else {}
        run_provenance[run_dir] = {
            "git_commit": git_info.get("commit"),
            "git_dirty": bool(git_info.get("dirty", True)),
            "paper_safe": bool(manifest.get("paper_safe", False)),
            "invalid_for_paper_reasons": list(manifest.get("invalid_for_paper_reasons", [])),
        }
    provenance_clean = bool(run_provenance) and all(
        not item["git_dirty"] and item["paper_safe"]
        for item in run_provenance.values()
    )
    return {
        "source_id": SOURCE_ID,
        "gate_passed": gate_passed,
        "verdict": (
            "executable_66d_to_3d_contract_established"
            if gate_passed
            else "executable_66d_to_3d_contract_not_established"
        ),
        "interpretation": (
            "input_output_contract_established_missing_defensive_extension_skill_candidate"
            if gate_passed
            else "cannot_claim_missing_skill_input_output_contract_still_missing"
        ),
        "training_authorized": False,
        "pilot_preregistration_draft_permitted": gate_passed,
        "paper_safe": gate_passed and provenance_clean,
        "provenance_clean": provenance_clean,
        "run_provenance": run_provenance,
        "opponents": opponent_results,
        "episode_rows": rows,
    }
