#!/usr/bin/env python3
"""Read-only raw-SI phase-semantics audit for a completed V4 bundle."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from uav_vpp_guidance.evaluation.raw_si_first_pass_event import (
    RawSIFirstPassConfig,
    RawSIFirstPassEventError,
    replay_raw_si_first_pass,
)
from uav_vpp_guidance.evaluation.thesis_neutral_postmerge_v4_contract import SOURCE_ID


OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
FAMILIES = (
    "close_balanced",
    "close_high_closure",
    "extended_balanced",
    "extended_high_closure",
)


def _load_episode(path: Path) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    record, ledger = payload.get("record"), payload.get("ledger")
    if not isinstance(record, Mapping) or not isinstance(ledger, Mapping):
        raise ValueError(f"invalid V4 episode artifact: {path.name}")
    if record.get("source_id") != SOURCE_ID:
        raise ValueError(f"unexpected source in {path.name}")
    return record, ledger


def _post_event_window(steps: Sequence[Mapping[str, Any]], event_step: int) -> tuple[int, int, set[str], set[str]]:
    flags = [
        bool(
            step.get("dynamic_state") == "neutral"
            and step.get("phase") == "post_merge"
            and step.get("prediction_valid")
            and not step.get("prediction_fallback")
        )
        for step in steps[event_step:]
    ]
    best = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        best = max(best, current)
    return sum(flags), best, set(), set()


def analyze_episodes(
    episodes: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    event_config: RawSIFirstPassConfig | None = None,
) -> dict[str, Any]:
    """Aggregate first-pass semantics from raw ledger steps only."""

    config = event_config or RawSIFirstPassConfig()
    groups: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "episodes": 0,
            "raw_event_episodes": 0,
            "environment_first_pass_episodes": 0,
            "handoff_episodes": 0,
            "post_event_neutral_prediction_episodes": 0,
            "post_event_neutral_prediction_steps": 0,
            "post_event_neutral_prediction_window_ge20": 0,
            "post_event_neutral_prediction_mirrors": set(),
            "post_event_neutral_prediction_heights": set(),
            "terminals": Counter(),
        }
    )
    for record, ledger in episodes:
        metadata = record.get("metadata")
        result = record.get("result")
        steps = ledger.get("steps")
        if not isinstance(metadata, Mapping) or not isinstance(result, Mapping) or not isinstance(steps, list):
            raise ValueError("V4 record lacks metadata/result/steps")
        opponent, family = record.get("opponent"), record.get("family_id")
        if opponent not in OPPONENTS or family not in FAMILIES:
            raise ValueError("V4 record has unexpected opponent or family")
        group = groups[(str(opponent), str(family))]
        group["episodes"] += 1
        group["environment_first_pass_episodes"] += int(any(bool(step.get("first_pass_complete")) for step in steps if isinstance(step, Mapping)))
        group["handoff_episodes"] += int(bool(result.get("handoff_reached")))
        group["terminals"][str(result.get("terminal_reason"))] += 1
        try:
            event = replay_raw_si_first_pass(steps, config)
        except RawSIFirstPassEventError as error:
            raise ValueError(f"V4 raw-SI replay failed: {error}") from error
        if event is None:
            continue
        group["raw_event_episodes"] += 1
        qualifying_steps, longest_window, _mirrors, _heights = _post_event_window(steps, event.event_step_index)
        if qualifying_steps:
            group["post_event_neutral_prediction_episodes"] += 1
            group["post_event_neutral_prediction_steps"] += qualifying_steps
            group["post_event_neutral_prediction_mirrors"].add(str(metadata.get("mirror_sign")))
            group["post_event_neutral_prediction_heights"].add(str(metadata.get("height_condition")))
        if longest_window >= 20:
            group["post_event_neutral_prediction_window_ge20"] += 1
    matrix: dict[str, dict[str, Any]] = {}
    totals: dict[str, dict[str, int]] = {}
    for opponent in OPPONENTS:
        matrix[opponent] = {}
        total = Counter()
        for family in FAMILIES:
            group = groups[(opponent, family)]
            item = {
                key: int(group[key])
                for key in (
                    "episodes",
                    "raw_event_episodes",
                    "environment_first_pass_episodes",
                    "handoff_episodes",
                    "post_event_neutral_prediction_episodes",
                    "post_event_neutral_prediction_steps",
                    "post_event_neutral_prediction_window_ge20",
                )
            }
            item["post_event_neutral_prediction_mirrors"] = sorted(group["post_event_neutral_prediction_mirrors"])
            item["post_event_neutral_prediction_heights"] = sorted(group["post_event_neutral_prediction_heights"])
            item["terminal_reasons"] = dict(sorted(group["terminals"].items()))
            matrix[opponent][family] = item
            total.update({key: item[key] for key in item if isinstance(item[key], int)})
        totals[opponent] = {key: int(value) for key, value in total.items()}
    return {
        "schema_version": 1,
        "source_id": SOURCE_ID,
        "analysis_kind": "raw_si_phase_semantics_read_only",
        "event_contract": {
            "closing_rate_threshold_mps": config.closing_rate_threshold_mps,
            "opening_rate_threshold_mps": config.opening_rate_threshold_mps,
            "minimum_closing_steps": config.minimum_closing_steps,
            "minimum_opening_steps": config.minimum_opening_steps,
        },
        "opponent_family_matrix": matrix,
        "opponent_totals": totals,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted((args.output_dir / "episodes").rglob("*.json"))
    if len(paths) != 144:
        raise SystemExit(f"expected 144 V4 episode artifacts, found {len(paths)}")
    report = analyze_episodes([_load_episode(path) for path in paths])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
