#!/usr/bin/env python3
"""Audit the preregistered first-macro routing intervention without rerunning it."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


SOURCE_ID = "THESIS-TAXONOMY-ROUTING3-V1"
TASK = "head_on"
OPPONENT_STAGE = "expert"
CANONICAL_METHOD = "canonical_ppo_high_level_policy"
FORCED_METHOD = "noncanonical_forced_first_crossing_then_ppo"
FIXED_CROSSING_METHOD = "crossing_vpp_specialist_no_routing"
METHODS = (CANONICAL_METHOD, FORCED_METHOD, FIXED_CROSSING_METHOD)
SCENARIOS = (
    "taxonomy30_head_on_above400_neg",
    "taxonomy30_head_on_below400_pos",
    "taxonomy30_head_on_level_neg",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(values: Iterable[Any]) -> float | None:
    finite = [value for item in values if (value := _finite(item)) is not None]
    return sum(finite) / len(finite) if finite else None


def _outcome(episode: Mapping[str, Any]) -> str:
    if bool(episode.get("win", False)):
        return "win"
    if bool(episode.get("loss", False)):
        return "loss"
    return "draw"


def _is_ego_crash_or_oob(episode: Mapping[str, Any]) -> bool:
    reason = str(
        episode.get("combat_reason") or episode.get("termination_reason") or ""
    ).lower()
    return "ego_crash" in reason or "ego_out_of_bounds" in reason


def _first_present(frames: Iterable[Mapping[str, Any]], key: str) -> Any:
    for frame in frames:
        value = frame.get(key)
        if value not in (None, "", "null"):
            return value
    return None


def _first_switch_step(frames: Iterable[Mapping[str, Any]]) -> int | None:
    for frame in frames:
        value = _finite(frame.get("commander_first_switch_step"))
        if value is not None:
            return int(value)
    return None


def _trajectory_summary(episode: Mapping[str, Any]) -> dict[str, Any]:
    frames = list(episode.get("trajectory", []) or [])
    active_override = [
        frame
        for frame in frames
        if bool(frame.get("commander_initial_macro_mode_override_active", False))
    ]
    return {
        "trajectory_steps": len(frames),
        "first_requested_mode": _first_present(
            frames, "commander_requested_mode_name"
        ),
        "first_effective_mode": _first_present(frames, "commander_mode_name"),
        "first_switch_step": _first_switch_step(frames),
        "override_active_steps": len(active_override),
        "override_applied_steps": sum(
            bool(frame.get("commander_initial_macro_mode_override_applied_this_step"))
            for frame in frames
        ),
        "override_active_effective_modes": sorted(
            {
                str(frame.get("commander_mode_name"))
                for frame in active_override
                if frame.get("commander_mode_name") not in (None, "", "null")
            }
        ),
        "first_macro_mean_vp_forward_bias_m": _mean(
            frame.get("vp_forward_bias_m") for frame in active_override
        ),
        "first_macro_mean_vp_lateral_bias_m": _mean(
            frame.get("vp_lateral_bias_m") for frame in active_override
        ),
        "first_macro_mean_vp_vertical_offset_m": _mean(
            (
                float(vp_z) - float(ego_z)
                if (vp_z := _finite(frame.get("vp_pos_z"))) is not None
                and (ego_z := _finite(frame.get("ego_pos_z"))) is not None
                else None
            )
            for frame in active_override
        ),
        "guard_reasons": sorted(
            {
                str(frame.get("commander_mode_constraint_reason"))
                for frame in frames
                if bool(frame.get("commander_mode_constraint_triggered", False))
                and frame.get("commander_mode_constraint_reason") not in (None, "")
            }
        ),
    }


def _load_episodes(run_dir: Path) -> list[dict[str, Any]]:
    records_path = run_dir / "aggregate" / "episode_records.json"
    payload = json.loads(records_path.read_text(encoding="utf-8"))
    episodes = payload.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError(f"{records_path} does not contain an episodes list")
    selected = [
        episode
        for episode in episodes
        if episode.get("task") == TASK
        and episode.get("opponent_stage") == OPPONENT_STAGE
        and episode.get("controller") in METHODS
    ]
    if len(selected) != len(SCENARIOS) * len(METHODS):
        raise ValueError(
            "Expected exactly 9 selected routing episodes, "
            f"found {len(selected)}"
        )
    return selected


def _hash_raw_sources(run_dir: Path, episodes: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for episode in episodes:
        method = str(episode["controller"])
        seed = int(episode["seed"])
        raw_dir = run_dir / "raw" / TASK / method / f"seed_{seed}"
        matches = sorted(raw_dir.glob("episode_*.json"))
        if len(matches) != 1:
            raise ValueError(
                f"Expected one raw episode for {method}/{seed}, found {len(matches)}"
            )
        hashes[str(matches[0])] = _sha256(matches[0])
    return hashes


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(*, run_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Validate intervention execution and apply the preregistered 2/3 gate."""
    run_dir = run_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    episodes = _load_episodes(run_dir)
    grouped = {
        (str(episode["scenario"]), str(episode["controller"])): episode
        for episode in episodes
    }
    expected_keys = {(scenario, method) for scenario in SCENARIOS for method in METHODS}
    if set(grouped) != expected_keys:
        raise ValueError("Scenario/method pairing does not match the preregistered design")

    rows: list[dict[str, Any]] = []
    violations: list[str] = []
    conversions = 0
    added_ego_crash_or_oob = 0
    for scenario in SCENARIOS:
        canonical = grouped[(scenario, CANONICAL_METHOD)]
        forced = grouped[(scenario, FORCED_METHOD)]
        fixed_crossing = grouped[(scenario, FIXED_CROSSING_METHOD)]
        canonical_summary = _trajectory_summary(canonical)
        forced_summary = _trajectory_summary(forced)
        fixed_summary = _trajectory_summary(fixed_crossing)
        if _outcome(canonical) != "loss":
            violations.append(f"{scenario}: canonical PPO is not the preregistered loss")
        if _outcome(fixed_crossing) != "win":
            violations.append(f"{scenario}: fixed crossing is not the preregistered win")
        if forced_summary["first_effective_mode"] != "crossing_specialist":
            violations.append(f"{scenario}: forced first effective mode is not crossing")
        if forced_summary["override_applied_steps"] != 1:
            violations.append(f"{scenario}: override was not applied exactly once")
        if forced_summary["override_active_steps"] <= 0:
            violations.append(f"{scenario}: no active override telemetry")
        if forced_summary["override_active_effective_modes"] != ["crossing_specialist"]:
            violations.append(f"{scenario}: a guard altered the initial forced mode")

        converted = _outcome(canonical) == "loss" and _outcome(forced) == "win"
        added_terminal = _is_ego_crash_or_oob(forced) and not _is_ego_crash_or_oob(
            canonical
        )
        conversions += int(converted)
        added_ego_crash_or_oob += int(added_terminal)
        rows.append(
            {
                "scenario": scenario,
                "seed": int(forced["seed"]),
                "canonical_outcome": _outcome(canonical),
                "canonical_terminal": canonical.get("combat_reason"),
                "forced_outcome": _outcome(forced),
                "forced_terminal": forced.get("combat_reason"),
                "fixed_crossing_outcome": _outcome(fixed_crossing),
                "fixed_crossing_terminal": fixed_crossing.get("combat_reason"),
                "canonical_first_effective_mode": canonical_summary[
                    "first_effective_mode"
                ],
                "forced_first_requested_mode": forced_summary["first_requested_mode"],
                "forced_first_effective_mode": forced_summary["first_effective_mode"],
                "forced_first_switch_step": forced_summary["first_switch_step"],
                "override_active_steps": forced_summary["override_active_steps"],
                "override_applied_steps": forced_summary["override_applied_steps"],
                "override_active_effective_modes": ";".join(
                    forced_summary["override_active_effective_modes"]
                ),
                "forced_first_macro_vp_forward_bias_m": forced_summary[
                    "first_macro_mean_vp_forward_bias_m"
                ],
                "forced_first_macro_vp_lateral_bias_m": forced_summary[
                    "first_macro_mean_vp_lateral_bias_m"
                ],
                "forced_first_macro_vp_vertical_offset_m": forced_summary[
                    "first_macro_mean_vp_vertical_offset_m"
                ],
                "forced_guard_reasons": ";".join(forced_summary["guard_reasons"]),
                "canonical_loss_to_forced_win": converted,
                "added_ego_crash_or_oob": added_terminal,
                "fixed_crossing_first_effective_mode": fixed_summary[
                    "first_effective_mode"
                ],
            }
        )

    gate_passed = not violations and conversions >= 2 and added_ego_crash_or_oob == 0
    result = {
        "source_id": SOURCE_ID,
        "run_dir": str(run_dir),
        "expected_episodes": 9,
        "observed_episodes": len(episodes),
        "canonical_loss_to_forced_win": conversions,
        "added_ego_crash_or_oob": added_ego_crash_or_oob,
        "execution_violations": violations,
        "gate_passed": gate_passed,
        "next_action": (
            "Expand the unchanged intervention to all six expert/head_on scenarios."
            if gate_passed
            else "Freeze negative routing evidence; do not patch routing or run end-to-end."
        ),
        "scenario_rows": rows,
    }
    _write_csv(output_dir / "routing3_causal_summary.csv", rows)
    _write_json(output_dir / "routing3_causal_summary.json", result)
    _write_json(
        output_dir / "artifact_source_hash_manifest.json",
        {
            "run_manifest": _sha256(run_dir / "run_manifest.json"),
            "resolved_config": _sha256(run_dir / "resolved_config.yaml"),
            "raw_episode_hashes": _hash_raw_sources(run_dir, episodes),
        },
    )
    report_lines = [
        "# Non-canonical Routing-Only Causal Validation",
        "",
        "## Frozen Boundary",
        "",
        "This evaluation changes only the first head-on macro routing request to crossing, then resumes the identical frozen canonical PPO. No training, reward, VPP, guidance, PID, checkpoint, or canonical configuration changed.",
        "",
        "## Preregistered Gate",
        "",
        f"- Canonical-loss to forced-win conversions: `{conversions}/3` (required: `>=2`).",
        f"- Added ego crash/OOB: `{added_ego_crash_or_oob}` (required: `0`).",
        f"- Execution violations: `{len(violations)}`.",
        f"- Gate: **{'PASS' if gate_passed else 'FAIL'}**.",
        "",
        "## Decision",
        "",
        result["next_action"],
        "",
        "## Interpretation Boundary",
        "",
        "A pass identifies a limited initial-routing contribution in these three preselected counterexamples only. It does not establish crossing-first as a globally deployable policy or change the canonical family.",
    ]
    if violations:
        report_lines.extend(["", "## Execution Violations", ""])
        report_lines.extend(f"- {violation}" for violation in violations)
    (output_dir / "routing3_causal_report_zh.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = analyze(run_dir=args.run_dir, output_dir=args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
