"""Build non-ranking opponent pressure cards from frozen Heldout40 formal runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


SOURCE_ID = "THESIS-FIVE-STATE-HELDOUT40-V1"
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
REFERENCE_METHOD = "legacy_static_oracle_task_gate"
STATES = ("advantage", "head_on", "disadvantage", "neutral", "crossing_entry")
GRAVITY_MPS2 = 9.80665


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(values: Iterable[Any]) -> float | None:
    numbers = [number for value in values if (number := _finite(value)) is not None]
    return sum(numbers) / len(numbers) if numbers else None


def _truthy(value: Any) -> bool:
    return value.strip().lower() in {"true", "1", "yes"} if isinstance(value, str) else bool(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _outcome(record: Mapping[str, Any]) -> str:
    if _truthy(record.get("win")):
        return "win"
    if _truthy(record.get("loss")):
        return "loss"
    if _truthy(record.get("draw")):
        return "draw"
    return "unresolved"


def _phase(frame: Mapping[str, Any]) -> str:
    return "pre_merge" if _truthy(frame.get("pre_merge")) else "post_merge"


def _position(frame: Mapping[str, Any], prefix: str) -> tuple[float, float, float] | None:
    values = tuple(_finite(frame.get(f"{prefix}_pos_{axis}")) for axis in ("x", "y", "z"))
    return values if all(value is not None for value in values) else None  # type: ignore[return-value]


def _target_speed_series(frames: Sequence[Mapping[str, Any]]) -> list[float | None]:
    speeds: list[float | None] = [None] * len(frames)
    for index in range(1, len(frames)):
        previous, current = frames[index - 1], frames[index]
        first = _position(previous, "target")
        second = _position(current, "target")
        before, after = _finite(previous.get("time_s")), _finite(current.get("time_s"))
        if first is not None and second is not None and before is not None and after is not None and after > before:
            speeds[index] = math.dist(first, second) / (after - before)
    return speeds


def _first_true_time(frames: Sequence[Mapping[str, Any]], key: str) -> float | None:
    for frame in frames:
        if _truthy(frame.get(key)):
            return _finite(frame.get("time_s"))
    return None


def _episode_metrics(record: Mapping[str, Any], opponent: str) -> dict[str, Any]:
    frames = [frame for frame in record.get("trajectory", []) if isinstance(frame, Mapping)]
    if not frames:
        raise ValueError(f"Missing raw telemetry: {opponent}/{record.get('scenario')}")
    target_speeds = _target_speed_series(frames)
    target_altitudes = [(_position(frame, "target") or (None, None, None))[2] for frame in frames]
    target_energy = [
        0.5 * speed**2 + GRAVITY_MPS2 * altitude
        if speed is not None and altitude is not None
        else None
        for speed, altitude in zip(target_speeds, target_altitudes)
    ]
    duration = _finite(record.get("total_time_s")) or 0.0
    return {
        "opponent": opponent,
        "scenario": str(record["scenario"]),
        "seed": int(record["seed"]),
        "initial_class": str(record["scenario_metadata"]["initial_class"]),
        "outcome": _outcome(record),
        "terminal_reason": str(record.get("combat_reason") or record.get("termination_reason") or "unknown"),
        "hp_advantage": _finite(record.get("hp_advantage")),
        "ego_hp": _finite(record.get("ego_hp")),
        "target_hp": _finite(record.get("target_hp")),
        "duration_s": duration,
        "ego_attack_zone_time_s": duration * sum(_truthy(frame.get("ego_in_attack_zone")) for frame in frames) / len(frames),
        "target_attack_zone_time_s": duration * sum(_truthy(frame.get("target_in_attack_zone")) for frame in frames) / len(frames),
        "first_ego_attack_zone_time_s": _first_true_time(frames, "ego_in_attack_zone"),
        "first_target_attack_zone_time_s": _first_true_time(frames, "target_in_attack_zone"),
        "target_pre_merge_speed_mps": _mean(
            speed for speed, frame in zip(target_speeds, frames) if _phase(frame) == "pre_merge"
        ),
        "target_pre_merge_altitude_m": _mean(
            altitude for altitude, frame in zip(target_altitudes, frames) if _phase(frame) == "pre_merge"
        ),
        "target_pre_merge_specific_energy_m2ps2": _mean(
            energy for energy, frame in zip(target_energy, frames) if _phase(frame) == "pre_merge"
        ),
        "post_merge_fraction": sum(_phase(frame) == "post_merge" for frame in frames) / len(frames),
        "ego_crash_or_oob": "ego" in str(record.get("combat_reason", "")).lower()
        and ("crash" in str(record.get("combat_reason", "")).lower() or "out_of_bounds" in str(record.get("combat_reason", "")).lower()),
    }


def _summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    outcomes = Counter(str(row["outcome"]) for row in rows)
    terminal_reasons = Counter(str(row["terminal_reason"]) for row in rows)
    return {
        "episodes": len(rows),
        "wins": outcomes["win"],
        "losses": outcomes["loss"],
        "draws": outcomes["draw"],
        "unresolved": outcomes["unresolved"],
        "mean_hp_advantage": _mean(row["hp_advantage"] for row in rows),
        "mean_target_attack_zone_time_s": _mean(row["target_attack_zone_time_s"] for row in rows),
        "mean_ego_attack_zone_time_s": _mean(row["ego_attack_zone_time_s"] for row in rows),
        "mean_net_target_pressure_s": _mean(
            (row["target_attack_zone_time_s"] or 0.0) - (row["ego_attack_zone_time_s"] or 0.0)
            for row in rows
        ),
        "mean_first_target_attack_zone_time_s": _mean(row["first_target_attack_zone_time_s"] for row in rows),
        "mean_first_ego_attack_zone_time_s": _mean(row["first_ego_attack_zone_time_s"] for row in rows),
        "mean_target_pre_merge_speed_mps": _mean(row["target_pre_merge_speed_mps"] for row in rows),
        "mean_target_pre_merge_altitude_m": _mean(row["target_pre_merge_altitude_m"] for row in rows),
        "mean_target_pre_merge_specific_energy_m2ps2": _mean(row["target_pre_merge_specific_energy_m2ps2"] for row in rows),
        "mean_post_merge_fraction": _mean(row["post_merge_fraction"] for row in rows),
        "ego_crash_or_oob_count": sum(bool(row["ego_crash_or_oob"]) for row in rows),
        "terminal_reason_counts": dict(sorted(terminal_reasons.items())),
    }


def analyze(*, manifest_path: Path, run_dirs: Mapping[str, Path], output_dir: Path) -> dict[str, Any]:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    scenarios = manifest.get("scenarios")
    if manifest.get("source_id") != SOURCE_ID or not isinstance(scenarios, list) or len(scenarios) != 40:
        raise ValueError("Expected frozen Heldout40 manifest")
    if set(run_dirs) != set(OPPONENTS):
        raise ValueError(f"Expected opponent set {OPPONENTS}")

    rows: list[dict[str, Any]] = []
    inputs = {"manifest": {"path": str(manifest_path), "sha256": _sha256_file(manifest_path)}}
    inputs["formal_episode_records"] = {}
    for opponent in OPPONENTS:
        aggregate_path = run_dirs[opponent] / "aggregate" / "episode_records.json"
        inputs["formal_episode_records"][opponent] = {"path": str(aggregate_path), "sha256": _sha256_file(aggregate_path)}
        payload = json.loads(aggregate_path.read_text(encoding="utf-8"))
        episodes = payload.get("episodes") if isinstance(payload, Mapping) else None
        selected = [
            episode for episode in episodes or []
            if isinstance(episode, Mapping) and episode.get("controller") == REFERENCE_METHOD
        ]
        if len(selected) != 40 or any(episode.get("opponent_stage") != opponent for episode in selected):
            raise ValueError(f"Expected 40 {REFERENCE_METHOD} episodes for {opponent}")
        if any(episode.get("backend") != "jsbsim" or not _truthy(episode.get("strict_backend")) for episode in selected):
            raise ValueError(f"Calibration requires strict JSBSim: {opponent}")
        rows.extend(_episode_metrics(episode, opponent) for episode in selected)

    matrix = []
    cards = {}
    for opponent in OPPONENTS:
        opponent_rows = [row for row in rows if row["opponent"] == opponent]
        cards[opponent] = {
            "reference_controller": REFERENCE_METHOD,
            "fixed_scenario_set": SOURCE_ID,
            "all_states": _summarize(opponent_rows),
            "by_initial_state": {},
            "claim_boundary": "Calibrated pressure profile under one fixed reference only; not an Elo, total-strength, or transitive rank.",
        }
        for state in STATES:
            cell = [row for row in opponent_rows if row["initial_class"] == state]
            if len(cell) != 8:
                raise ValueError(f"Expected 8 reference episodes for {opponent}/{state}")
            summary = _summarize(cell)
            cards[opponent]["by_initial_state"][state] = summary
            matrix.append({"opponent": opponent, "initial_class": state, **summary})

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "opponent_calibration_v2_reference_ledger.csv", rows)
    _write_csv(output_dir / "opponent_calibration_v2_state_matrix.csv", matrix)
    _write_json(output_dir / "opponent_calibration_v2_cards.json", cards)
    report_lines = [
        "# Opponent Calibration / Capability Card v2",
        "",
        f"- Fixed scenario set: `{SOURCE_ID}` (40 scenarios).",
        f"- Common reference controller: `{REFERENCE_METHOD}`.",
        "- All metrics remain opponent-specific. This document reports pressure and response envelopes, not an Elo or a total-strength ranking.",
        "",
        "## Interpretation",
        "",
        "Target attack-zone exposure, first target attack-zone entry, final HP margin, terminal mix, target pre-merge kinematics, and post-merge occupancy quantify how the three frozen opponent implementations pressure the same reference controller. They are controller- and scenario-conditioned quantities, not intrinsic opponent strength.",
    ]
    for opponent in OPPONENTS:
        summary = cards[opponent]["all_states"]
        report_lines.append(
            f"- `{opponent}`: reference wins/losses/draws = {summary['wins']}/{summary['losses']}/{summary['draws']}; "
            f"mean target pressure = {summary['mean_net_target_pressure_s']:.3f} s."
        )
    (output_dir / "opponent_calibration_v2_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    report = {
        "source_id": SOURCE_ID,
        "mode": "read_only",
        "reference_controller": REFERENCE_METHOD,
        "episode_count": len(rows),
        "input_artifacts": inputs,
        "artifacts": [
            "opponent_calibration_v2_reference_ledger.csv",
            "opponent_calibration_v2_state_matrix.csv",
            "opponent_calibration_v2_cards.json",
            "opponent_calibration_v2_report.md",
        ],
    }
    _write_json(output_dir / "opponent_calibration_v2_analysis.json", report)
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
        manifest_path=args.manifest,
        run_dirs={"expert": args.expert_run, "end_to_end": args.end_to_end_run, "independent_ppo_vpp": args.independent_ppo_vpp_run},
        output_dir=args.output_dir,
    )
    print(json.dumps({"mode": report["mode"], "episode_count": report["episode_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
