"""Audit the frozen Heldout40 disadvantage motif without training or tuning.

This analysis consumes only the three immutable Heldout40 formal aggregates,
the frozen scenario manifest, and a verified capability-atlas output. It
selects two deterministic representative scenarios per opponent and reports
VPP geometry, phase, attack-zone, HP, and actual-response telemetry. The
result is evidence attribution, not a causal proof or a training authorization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


SOURCE_ID = "THESIS-FIVE-STATE-HELDOUT40-V1"
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
METHODS = (
    "canonical_ppo_high_level_policy",
    "legacy_static_oracle_task_gate",
    "head_on_vpp_specialist_no_routing",
    "crossing_vpp_specialist_no_routing",
)
CANONICAL_METHOD = METHODS[0]
LABEL_LIBRARY = "shared-specialist insufficiency candidate"
PHASES = ("pre_merge", "post_merge", "re_entry")
GRAVITY_MPS2 = 9.80665

SELECTION_RULE = {
    "scope": "disadvantage only; formal Heldout40 episodes only",
    "per_opponent": 2,
    "boundary_highest_canonical_hp": "Highest canonical PPO final HP advantage; this may be a shared-success boundary case.",
    "deepest_canonical_hp": "Lowest canonical PPO final HP advantage; this is the deepest canonical loss.",
    "tie_break": "Lexicographic scenario name, then seed.",
    "phase_sentinel": "Every non-representative opponent-scenario pair with any observed post-merge fraction is appended for coverage attribution only; it never replaces a representative.",
    "boundary": "Selection is descriptive and does not choose checkpoints, tune parameters, or pool opponents.",
}


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _mean(values: Iterable[Any]) -> float | None:
    numbers = [number for value in values if (number := _finite(value)) is not None]
    return sum(numbers) / len(numbers) if numbers else None


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": _sha256_file(path), "size_bytes": path.stat().st_size}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def _phase(frame: Mapping[str, Any]) -> str:
    if _truthy(frame.get("pre_merge")):
        return "pre_merge"
    range_rate = _finite(frame.get("range_rate_mps"))
    return "re_entry" if range_rate is not None and range_rate < 0.0 else "post_merge"


def _outcome(record: Mapping[str, Any]) -> str:
    if _truthy(record.get("win")):
        return "win"
    if _truthy(record.get("loss")):
        return "loss"
    if _truthy(record.get("draw")):
        return "draw"
    return "unresolved"


def _reason(record: Mapping[str, Any]) -> str:
    return str(record.get("combat_reason") or record.get("termination_reason") or "unknown")


def _position(frame: Mapping[str, Any], prefix: str) -> tuple[float, float, float] | None:
    names = (f"{prefix}_pos_x", f"{prefix}_pos_y", f"{prefix}_pos_z")
    vector = tuple(_finite(frame.get(name)) for name in names)
    if all(value is not None for value in vector):
        return vector  # type: ignore[return-value]
    nested = frame.get(f"{prefix}_pos_m")
    if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes)) and len(nested) == 3:
        alternative = tuple(_finite(value) for value in nested)
        if all(value is not None for value in alternative):
            return alternative  # type: ignore[return-value]
    return None


def _step_seconds(frames: Sequence[Mapping[str, Any]]) -> float:
    deltas = []
    for previous, current in zip(frames, frames[1:]):
        before = _finite(previous.get("time_s"))
        after = _finite(current.get("time_s"))
        if before is not None and after is not None and after > before:
            deltas.append(after - before)
    return statistics.median(deltas) if deltas else 0.0


def _target_speeds(frames: Sequence[Mapping[str, Any]]) -> list[float | None]:
    speeds: list[float | None] = [None] * len(frames)
    for index in range(1, len(frames)):
        first = _position(frames[index - 1], "target")
        second = _position(frames[index], "target")
        before = _finite(frames[index - 1].get("time_s"))
        after = _finite(frames[index].get("time_s"))
        if first is None or second is None or before is None or after is None or after <= before:
            continue
        speeds[index] = math.dist(first, second) / (after - before)
    return speeds


def _vpp_vector(frame: Mapping[str, Any]) -> tuple[float, float, float] | None:
    forward = _finite(frame.get("vp_forward_bias_m"))
    lateral = _finite(frame.get("vp_lateral_bias_m"))
    vp_z = _finite(frame.get("vp_pos_z"))
    ego_z = _finite(frame.get("ego_pos_z"))
    if forward is None or lateral is None or vp_z is None or ego_z is None:
        return None
    return forward, lateral, vp_z - ego_z


def _mean_vector(vectors: Iterable[tuple[float, float, float] | None]) -> tuple[float, float, float] | None:
    values = [vector for vector in vectors if vector is not None]
    if not values:
        return None
    return tuple(sum(vector[index] for vector in values) / len(values) for index in range(3))  # type: ignore[return-value]


def _specific_energy(speed_mps: float | None, altitude_m: float | None) -> float | None:
    if speed_mps is None or altitude_m is None:
        return None
    return 0.5 * speed_mps**2 + GRAVITY_MPS2 * altitude_m


def _phase_summary(
    *, frames: Sequence[Mapping[str, Any]], target_speeds: Sequence[float | None], phase: str, dt_s: float
) -> dict[str, Any]:
    indexed = [(index, frame) for index, frame in enumerate(frames) if _phase(frame) == phase]
    selected = [frame for _, frame in indexed]
    selected_target_speeds = [target_speeds[index] for index, _ in indexed]
    ego_speeds = [_finite(frame.get("speed_mps")) for frame in selected]
    ego_altitudes = [_finite(frame.get("altitude_m")) for frame in selected]
    target_altitudes = [(_position(frame, "target") or (None, None, None))[2] for frame in selected]
    target_energies = [
        _specific_energy(speed, altitude) for speed, altitude in zip(selected_target_speeds, target_altitudes)
    ]
    response_errors = []
    for frame in selected:
        command = _finite(frame.get("nz_cmd"))
        response = _finite(frame.get("nz_g"))
        if command is not None and response is not None:
            response_errors.append(abs(command - response))
    ego_hp = [_finite(frame.get("ego_hp")) for frame in selected]
    target_hp = [_finite(frame.get("target_hp")) for frame in selected]
    return {
        "phase": phase,
        "phase_steps": len(selected),
        "phase_duration_s": len(selected) * dt_s,
        "mean_vp_vector_m": _mean_vector(_vpp_vector(frame) for frame in selected),
        "mean_vp_forward_bias_m": _mean(vector[0] for vector in (_vpp_vector(frame) for frame in selected) if vector),
        "mean_vp_lateral_bias_m": _mean(vector[1] for vector in (_vpp_vector(frame) for frame in selected) if vector),
        "mean_vp_vertical_offset_m": _mean(vector[2] for vector in (_vpp_vector(frame) for frame in selected) if vector),
        "ego_attack_zone_fraction": _mean(1.0 if _truthy(frame.get("ego_in_attack_zone")) else 0.0 for frame in selected),
        "target_attack_zone_fraction": _mean(1.0 if _truthy(frame.get("target_in_attack_zone")) else 0.0 for frame in selected),
        "mean_ego_speed_mps": _mean(ego_speeds),
        "mean_ego_altitude_m": _mean(ego_altitudes),
        "mean_ego_specific_energy_m2ps2": _mean(
            _specific_energy(speed, altitude) for speed, altitude in zip(ego_speeds, ego_altitudes)
        ),
        "mean_target_speed_mps": _mean(selected_target_speeds),
        "mean_target_altitude_m": _mean(target_altitudes),
        "mean_target_specific_energy_m2ps2": _mean(target_energies),
        "mean_abs_nz_tracking_error_g": _mean(response_errors),
        "nz_saturation_fraction": _mean(1.0 if _truthy(frame.get("nz_saturated")) else 0.0 for frame in selected),
        "roll_rate_saturation_fraction": _mean(
            1.0 if _truthy(frame.get("roll_rate_saturated")) else 0.0 for frame in selected
        ),
        "throttle_saturation_fraction": _mean(
            1.0 if _truthy(frame.get("throttle_saturated")) else 0.0 for frame in selected
        ),
        "ego_hp_start": next((value for value in ego_hp if value is not None), None),
        "ego_hp_end": next((value for value in reversed(ego_hp) if value is not None), None),
        "target_hp_start": next((value for value in target_hp if value is not None), None),
        "target_hp_end": next((value for value in reversed(target_hp) if value is not None), None),
    }


def _episode_audit(record: Mapping[str, Any], opponent: str, selection_role: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    frames = [frame for frame in record.get("trajectory", []) if isinstance(frame, Mapping)]
    if not frames:
        raise ValueError(f"Missing trajectory: {opponent}/{record.get('scenario')}/{record.get('controller')}")
    target_speeds = _target_speeds(frames)
    dt_s = _step_seconds(frames)
    phase_rows = []
    phase_summaries = {}
    for phase in PHASES:
        summary = _phase_summary(frames=frames, target_speeds=target_speeds, phase=phase, dt_s=dt_s)
        phase_summaries[phase] = summary
        phase_rows.append(
            {
                "opponent": opponent,
                "scenario": record["scenario"],
                "seed": int(record["seed"]),
                "selection_role": selection_role,
                "method": record.get("controller") or record.get("method"),
                **summary,
            }
        )
    terminal_reason = _reason(record)
    all_response_errors = [
        abs(command - response)
        for frame in frames
        if (command := _finite(frame.get("nz_cmd"))) is not None and (response := _finite(frame.get("nz_g"))) is not None
    ]
    episode_row = {
        "opponent": opponent,
        "scenario": record["scenario"],
        "seed": int(record["seed"]),
        "selection_role": selection_role,
        "method": record.get("controller") or record.get("method"),
        "outcome": _outcome(record),
        "terminal_reason": terminal_reason,
        "ego_hp_final": _finite(record.get("ego_hp")),
        "target_hp_final": _finite(record.get("target_hp")),
        "hp_advantage_final": _finite(record.get("hp_advantage")),
        "total_time_s": _finite(record.get("total_time_s")),
        "trajectory_steps": len(frames),
        "sample_period_s": dt_s,
        "ego_crash_or_oob_terminal": "ego" in terminal_reason.lower()
        and ("crash" in terminal_reason.lower() or "out_of_bounds" in terminal_reason.lower()),
        "mean_abs_nz_tracking_error_g": _mean(all_response_errors),
        "nz_saturation_fraction": _mean(1.0 if _truthy(frame.get("nz_saturated")) else 0.0 for frame in frames),
        "roll_rate_saturation_fraction": _mean(
            1.0 if _truthy(frame.get("roll_rate_saturated")) else 0.0 for frame in frames
        ),
        "throttle_saturation_fraction": _mean(
            1.0 if _truthy(frame.get("throttle_saturated")) else 0.0 for frame in frames
        ),
        **{
            f"{phase}_{key}": value
            for phase, summary in phase_summaries.items()
            for key, value in summary.items()
            if key != "phase"
        },
    }
    return episode_row, phase_rows


def _max_vector_span(vectors: Iterable[tuple[float, float, float] | None]) -> float | None:
    usable = [vector for vector in vectors if vector is not None]
    if len(usable) < 2:
        return None
    return max(math.dist(first, second) for index, first in enumerate(usable) for second in usable[index + 1 :])


def _range(values: Iterable[Any]) -> float | None:
    usable = [number for value in values if (number := _finite(value)) is not None]
    return max(usable) - min(usable) if len(usable) >= 2 else None


def _select_representatives(atlas_rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    selections = []
    for opponent in OPPONENTS:
        rows = [row for row in atlas_rows if row.get("opponent") == opponent]
        if len(rows) != 8:
            raise ValueError(f"Expected 8 disadvantage scenarios for {opponent}, found {len(rows)}")
        ordered = sorted(
            rows,
            key=lambda row: (float(row[f"{CANONICAL_METHOD}_hp_advantage"]), row["scenario"], int(row["seed"])),
        )
        chosen = (("deepest_canonical_hp", ordered[0]), ("boundary_highest_canonical_hp", ordered[-1]))
        if chosen[0][1]["scenario"] == chosen[1][1]["scenario"]:
            raise ValueError(f"Representative selection collapsed for {opponent}")
        for role, row in chosen:
            selections.append(
                {
                    "opponent": opponent,
                    "scenario": row["scenario"],
                    "seed": int(row["seed"]),
                    "selection_role": role,
                    "canonical_hp_advantage": float(row[f"{CANONICAL_METHOD}_hp_advantage"]),
                    "all_methods_loss": row["all_methods_loss"].strip().lower() == "true",
                }
            )
    return selections


def _full_phase_coverage(atlas_rows: Sequence[Mapping[str, str]]) -> dict[str, int]:
    """Count phase presence in all 96 disadvantage method episodes, not just the six selections."""

    coverage = {"method_episode_rows": len(atlas_rows) * len(METHODS)}
    for phase in PHASES:
        coverage[f"{phase}_nonzero_method_rows"] = sum(
            (_finite(row.get(f"{method}_{phase}_fraction")) or 0.0) > 0.0
            for row in atlas_rows
            for method in METHODS
        )
    return coverage


def _phase_sentinels(
    atlas_rows: Sequence[Mapping[str, str]], selections: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Preserve rare post-merge cases without retrospectively changing representative selection."""

    representative_keys = {(str(row["opponent"]), str(row["scenario"])) for row in selections}
    sentinels = []
    for row in atlas_rows:
        opponent = str(row["opponent"])
        scenario = str(row["scenario"])
        if (opponent, scenario) in representative_keys:
            continue
        fractions = [_finite(row.get(f"{method}_post_merge_fraction")) or 0.0 for method in METHODS]
        if max(fractions) <= 0.0:
            continue
        sentinels.append(
            {
                "opponent": opponent,
                "scenario": scenario,
                "seed": int(row["seed"]),
                "selection_role": "phase_coverage_sentinel",
                "max_post_merge_fraction": max(fractions),
                "all_methods_loss": row["all_methods_loss"].strip().lower() == "true",
            }
        )
    return sorted(sentinels, key=lambda row: (row["opponent"], row["scenario"], row["seed"]))


def _validate_atlas(atlas_dir: Path, manifest_path: Path, run_dirs: Mapping[str, Path]) -> tuple[dict[str, Any], list[dict[str, str]]]:
    report_path = atlas_dir / "capability_atlas_analysis.json"
    audit_path = atlas_dir / "capability_atlas_disadvantage_audit.csv"
    if not report_path.is_file() or not audit_path.is_file():
        raise ValueError("Capability atlas artifacts are incomplete")
    report = _read_json(report_path)
    if report.get("source_id") != SOURCE_ID or report.get("mode") != "read_only" or report.get("episode_count") != 480:
        raise ValueError("Capability atlas identity mismatch")
    if not report.get("pairing_input_passed"):
        raise ValueError("Capability atlas lacks a passing pairing proof")
    expected_hashes = report.get("input_artifacts", {})
    if expected_hashes.get("manifest", {}).get("sha256") != _sha256_file(manifest_path):
        raise ValueError("Manifest hash no longer matches the capability atlas")
    for opponent in OPPONENTS:
        aggregate = run_dirs[opponent] / "aggregate" / "episode_records.json"
        expected = expected_hashes.get("formal_episode_records", {}).get(opponent, {}).get("sha256")
        if expected != _sha256_file(aggregate):
            raise ValueError(f"Formal aggregate hash no longer matches the capability atlas: {opponent}")
    with audit_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 24:
        raise ValueError(f"Expected 24 disadvantage audit rows, found {len(rows)}")
    return report, rows


def _decision_memo(
    *,
    selections: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    atlas_rows: Sequence[Mapping[str, str]],
    full_phase_coverage: Mapping[str, int],
    selected_phase_steps: Mapping[str, int],
    sentinel_phase_steps: Mapping[str, int],
    sentinels: Sequence[Mapping[str, Any]],
) -> str:
    full_shared_losses = sum(row["all_methods_loss"].strip().lower() == "true" for row in atlas_rows)
    selected_shared_losses = sum(bool(row["all_methods_loss"]) for row in selections)
    selected_successes = len(selections) - selected_shared_losses
    lines = [
        "# Heldout40 Capability Atlas Decision Memo",
        "",
        "## Frozen Decision",
        "",
        "- `disadvantage` remains a cross-opponent shared-specialist-insufficiency candidate: it is not classified as a general routing failure.",
        "- `head_on/end_to_end` remains a local routing-sensitive candidate only; this memo does not reopen it or combine it with the disadvantage conclusion.",
        "- `advantage` and `crossing_entry` remain ceiling/non-discriminative under Heldout40, while opponent sensitivity is reported separately rather than pooled.",
        "",
        "## Representative Selection",
        "",
        "The selection rule was fixed before raw-trajectory inspection: for each opponent, choose the highest and lowest canonical PPO final HP advantage, breaking ties by scenario then seed.",
    ]
    for selection in selections:
        lines.append(
            f"- `{selection['opponent']}` / `{selection['selection_role']}`: `{selection['scenario']}` "
            f"(seed {selection['seed']}, canonical HP advantage {selection['canonical_hp_advantage']:+.1f}, "
            f"all-method loss={selection['all_methods_loss']})."
        )
    phase_coverage_blocked = full_phase_coverage["re_entry_nonzero_method_rows"] == 0
    lines.extend(
        [
            "",
            "## Attribution Reading",
            "",
            f"Across all 24 disadvantage opponent-scenario cells, {full_shared_losses}/24 are four-method shared losses. Among the six selected boundary/deep cases, {selected_shared_losses}/6 are shared losses and {selected_successes}/6 is a shared boundary success.",
            "",
            "The phase audit preserves separate opponent rows and reports target pre-merge speed, altitude, and specific energy alongside VPP vectors, attack-zone exposure, HP evolution, and actual n_z response. Thus early opponent behavior is measured rather than assumed away.",
            "",
            "The current evidence rules out a purely high-level routing explanation for the repeated shared-loss cells because canonical PPO, oracle gate, fixed head-on, and fixed crossing all lose together.",
            "",
            "All four methods also share the downstream guidance/PID execution chain. Absence of ego crash/OOB or sustained saturation is only a negative warning signal, not proof of execution adequacy; the phase audit retains n_z command-response error precisely because this confound remains open.",
            "",
            "## Guardrails",
            "",
            "- No policy, VPP, reward, guidance, PID, manifest, canonical result, or MVP-2 asset was modified.",
            "- Opponent summaries are never pooled into an Elo or global-strength claim.",
        ]
    )
    if phase_coverage_blocked:
        lines.extend(
            [
                "",
                "## Phase-Coverage No-Go",
                "",
                f"Across the full disadvantage matrix, post-merge is present in {full_phase_coverage['post_merge_nonzero_method_rows']}/{full_phase_coverage['method_episode_rows']} method episodes and re-entry is present in 0/{full_phase_coverage['method_episode_rows']}. The six fixed representatives have phase steps pre-merge/post-merge/re-entry = {selected_phase_steps['pre_merge']}/{selected_phase_steps['post_merge']}/{selected_phase_steps['re_entry']}.",
                "",
                f"The {len(sentinels)} appended coverage sentinel pairs contribute phase steps pre-merge/post-merge/re-entry = {sentinel_phase_steps['pre_merge']}/{sentinel_phase_steps['post_merge']}/{sentinel_phase_steps['re_entry']}; they are retained because they are the only observed post-merge evidence, not because they are representative selections.",
                "",
                "Consequently, this evidence cannot diagnose a disengagement, energy-recovery, or re-entry skill gap. It supports only a pre-merge disadvantage-pressure motif plus a sparse post-merge failure signal, and blocks any training claim for a post-merge/re-entry specialist. The next permitted action is an evaluation-only phase-feasible manifest or sampler preflight that demonstrably reaches sustained post-merge and re-entry before any low-level training decision.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "The observed post-merge/re-entry coverage supports a future feasibility preflight for one disadvantage motif, but does not authorize full shared-skill-library training.",
            ]
        )
    return "\n".join(lines) + "\n"


def analyze(
    *,
    manifest_path: Path,
    atlas_dir: Path,
    run_dirs: Mapping[str, Path],
    output_dir: Path,
) -> dict[str, Any]:
    if set(run_dirs) != set(OPPONENTS):
        raise ValueError(f"Expected formal runs for {OPPONENTS}")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    if manifest.get("source_id") != SOURCE_ID or len(manifest.get("scenarios", [])) != 40:
        raise ValueError("Frozen Heldout40 manifest mismatch")
    atlas_report, atlas_rows = _validate_atlas(atlas_dir, manifest_path, run_dirs)
    selections = _select_representatives(atlas_rows)
    full_phase_coverage = _full_phase_coverage(atlas_rows)
    sentinels = _phase_sentinels(atlas_rows, selections)
    selection_index = {(item["opponent"], item["scenario"]): item for item in selections}
    sentinel_index = {(item["opponent"], item["scenario"]): item for item in sentinels}
    audit_index = {**selection_index, **sentinel_index}

    episode_audits: list[dict[str, Any]] = []
    phase_audits: list[dict[str, Any]] = []
    audit_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for opponent in OPPONENTS:
        aggregate_path = run_dirs[opponent] / "aggregate" / "episode_records.json"
        payload = _read_json(aggregate_path)
        records = payload.get("episodes") if isinstance(payload, Mapping) else None
        if not isinstance(records, list):
            raise ValueError(f"Invalid aggregate record payload: {aggregate_path}")
        selected = [
            record
            for record in records
            if isinstance(record, Mapping)
            and (opponent, str(record.get("scenario"))) in audit_index
            and str(record.get("controller") or record.get("method")) in METHODS
        ]
        expected_count = 4 * sum(1 for key in audit_index if key[0] == opponent)
        if len(selected) != expected_count:
            raise ValueError(f"Expected {expected_count} selected method records for {opponent}, found {len(selected)}")
        for record in selected:
            scenario = str(record["scenario"])
            method = str(record.get("controller") or record.get("method"))
            selection = audit_index[(opponent, scenario)]
            episode_row, phase_rows = _episode_audit(record, opponent, str(selection["selection_role"]))
            episode_audits.append(episode_row)
            phase_audits.extend(phase_rows)
            audit_by_key[(opponent, scenario, method)] = episode_row

    comparisons = []
    for selection in selections:
        opponent = str(selection["opponent"])
        scenario = str(selection["scenario"])
        rows = [audit_by_key[(opponent, scenario, method)] for method in METHODS]
        phase_spans = {}
        for phase in PHASES:
            phase_spans[phase] = _max_vector_span(
                row.get(f"{phase}_mean_vp_vector_m") for row in rows
            )
        comparisons.append(
            {
                **selection,
                "method_outcomes": {row["method"]: row["outcome"] for row in rows},
                "terminal_reasons": {row["method"]: row["terminal_reason"] for row in rows},
                "all_methods_same_outcome": len({row["outcome"] for row in rows}) == 1,
                "all_methods_same_terminal_reason": len({row["terminal_reason"] for row in rows}) == 1,
                "any_ego_crash_or_oob_terminal": any(row["ego_crash_or_oob_terminal"] for row in rows),
                "max_nz_saturation_fraction": max(row["nz_saturation_fraction"] or 0.0 for row in rows),
                "max_roll_rate_saturation_fraction": max(row["roll_rate_saturation_fraction"] or 0.0 for row in rows),
                "max_throttle_saturation_fraction": max(row["throttle_saturation_fraction"] or 0.0 for row in rows),
                "max_mean_abs_nz_tracking_error_g": max(
                    row["mean_abs_nz_tracking_error_g"] or 0.0 for row in rows
                ),
                "vpp_mean_vector_span_m_by_phase": phase_spans,
                "target_pre_merge_speed_span_mps": _range(row["pre_merge_mean_target_speed_mps"] for row in rows),
                "target_pre_merge_altitude_span_m": _range(row["pre_merge_mean_target_altitude_m"] for row in rows),
                "target_pre_merge_specific_energy_span_m2ps2": _range(
                    row["pre_merge_mean_target_specific_energy_m2ps2"] for row in rows
                ),
                "ego_attack_zone_time_s_by_method": {
                    row["method"]: (row["pre_merge_ego_attack_zone_fraction"] or 0.0)
                    * (row["pre_merge_phase_duration_s"] or 0.0)
                    + (row["post_merge_ego_attack_zone_fraction"] or 0.0)
                    * (row["post_merge_phase_duration_s"] or 0.0)
                    + (row["re_entry_ego_attack_zone_fraction"] or 0.0)
                    * (row["re_entry_phase_duration_s"] or 0.0)
                    for row in rows
                },
                "hp_advantage_by_method": {row["method"]: row["hp_advantage_final"] for row in rows},
            }
        )

    selected_phase_steps = {
        phase: sum(
            int(row["phase_steps"])
            for row in phase_audits
            if row["phase"] == phase and row["selection_role"] != "phase_coverage_sentinel"
        )
        for phase in PHASES
    }
    sentinel_phase_steps = {
        phase: sum(
            int(row["phase_steps"])
            for row in phase_audits
            if row["phase"] == phase and row["selection_role"] == "phase_coverage_sentinel"
        )
        for phase in PHASES
    }
    phase_coverage_blocked = full_phase_coverage["re_entry_nonzero_method_rows"] == 0

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "disadvantage_motif_selection_rule.json", SELECTION_RULE)
    _write_csv(output_dir / "disadvantage_motif_representatives.csv", selections)
    _write_csv(output_dir / "disadvantage_motif_phase_sentinels.csv", sentinels)
    _write_csv(output_dir / "disadvantage_motif_episode_audit.csv", episode_audits)
    _write_csv(output_dir / "disadvantage_motif_phase_audit.csv", phase_audits)
    _write_csv(
        output_dir / "disadvantage_motif_phase_sentinel_episode_audit.csv",
        [row for row in episode_audits if row["selection_role"] == "phase_coverage_sentinel"],
    )
    _write_csv(
        output_dir / "disadvantage_motif_phase_sentinel_phase_audit.csv",
        [row for row in phase_audits if row["selection_role"] == "phase_coverage_sentinel"],
    )
    _write_csv(output_dir / "disadvantage_motif_comparisons.csv", comparisons)
    memo = _decision_memo(
        selections=selections,
        comparisons=comparisons,
        atlas_rows=atlas_rows,
        full_phase_coverage=full_phase_coverage,
        selected_phase_steps=selected_phase_steps,
        sentinel_phase_steps=sentinel_phase_steps,
        sentinels=sentinels,
    )
    (output_dir / "heldout40_atlas_decision_memo_zh.md").write_text(memo, encoding="utf-8")
    input_artifacts = {
        "manifest": _file_record(manifest_path),
        "capability_atlas": {
            "analysis": _file_record(atlas_dir / "capability_atlas_analysis.json"),
            "disadvantage_audit": _file_record(atlas_dir / "capability_atlas_disadvantage_audit.csv"),
        },
        "formal_episode_records": {
            opponent: _file_record(run_dirs[opponent] / "aggregate" / "episode_records.json")
            for opponent in OPPONENTS
        },
    }
    report = {
        "source_id": SOURCE_ID,
        "mode": "read_only",
        "input_artifacts": input_artifacts,
        "atlas_source_id": atlas_report["source_id"],
        "selected_scenario_count": len(selections),
        "selected_episode_count": len(selections) * len(METHODS),
        "audited_episode_count": len(episode_audits),
        "phase_sentinel_scenario_count": len(sentinels),
        "phase_sentinel_episode_count": len(sentinels) * len(METHODS),
        "phase_row_count": len(phase_audits),
        "full_disadvantage_all_method_loss_pairs": sum(
            row["all_methods_loss"].strip().lower() == "true" for row in atlas_rows
        ),
        "representative_all_method_loss_pairs": sum(bool(row["all_methods_loss"]) for row in selections),
        "full_phase_coverage": full_phase_coverage,
        "selected_phase_steps": selected_phase_steps,
        "phase_sentinel_steps": sentinel_phase_steps,
        "verdict": "blocked_missing_reentry_coverage"
        if phase_coverage_blocked
        else "preflight_only",
        "artifacts": [
            "disadvantage_motif_selection_rule.json",
            "disadvantage_motif_representatives.csv",
            "disadvantage_motif_phase_sentinels.csv",
            "disadvantage_motif_episode_audit.csv",
            "disadvantage_motif_phase_audit.csv",
            "disadvantage_motif_phase_sentinel_episode_audit.csv",
            "disadvantage_motif_phase_sentinel_phase_audit.csv",
            "disadvantage_motif_comparisons.csv",
            "heldout40_atlas_decision_memo_zh.md",
        ],
    }
    _write_json(output_dir / "disadvantage_motif_analysis.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--atlas-dir", type=Path, required=True)
    parser.add_argument("--expert-run", type=Path, required=True)
    parser.add_argument("--end-to-end-run", type=Path, required=True)
    parser.add_argument("--independent-ppo-vpp-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        manifest_path=args.manifest,
        atlas_dir=args.atlas_dir,
        run_dirs={
            "expert": args.expert_run,
            "end_to_end": args.end_to_end_run,
            "independent_ppo_vpp": args.independent_ppo_vpp_run,
        },
        output_dir=args.output_dir,
    )
    print(json.dumps({"mode": report["mode"], "audited_episode_count": report["audited_episode_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
