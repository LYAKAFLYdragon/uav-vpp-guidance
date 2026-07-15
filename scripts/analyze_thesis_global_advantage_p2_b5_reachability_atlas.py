#!/usr/bin/env python3
"""Read the completed B5 run without rerunning it or pooling opponents."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping


SOURCE_ID = "THESIS-GLOBAL-ADVANTAGE-V1-P2-PHASE-FEASIBLE-SAMPLER-B5-R1"
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
STATES = ("advantage", "head_on", "disadvantage", "neutral", "crossing_entry")
PHASES = ("post_merge", "re_entry")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _step_is_eligible(step: Mapping[str, Any]) -> bool:
    """A profile-free continuous physical sample suitable for atlas counting."""

    return bool(
        step.get("first_pass_complete_before_step", False)
        and step.get("structural_contract_valid", False)
        and step.get("prediction_valid", False)
        and not step.get("prediction_fallback", False)
    )


def analyze(run_dir: Path) -> dict[str, Any]:
    """Verify B5 episode hashes and quantify target geometry per opponent."""

    root = Path(run_dir)
    run_manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    gate = json.loads((root / "phase_feasible_sampler_gate.json").read_text(encoding="utf-8"))
    if run_manifest.get("source_id") != SOURCE_ID or gate.get("source_id") != SOURCE_ID:
        raise ValueError("unexpected B5 source ID")
    records = list(run_manifest.get("records") or [])
    if len(records) != 36:
        raise ValueError("B5 run manifest must contain all 36 one-time records")

    buckets: dict[tuple[str, str, str], dict[str, Any]] = {}
    episode_rows: list[dict[str, Any]] = []
    for record in records:
        opponent = str(record.get("opponent", ""))
        if opponent not in OPPONENTS:
            raise ValueError(f"unexpected B5 opponent: {opponent}")
        path = root / str(record.get("path", ""))
        if not path.is_file() or sha256_file(path) != str(record.get("sha256", "")):
            raise ValueError(f"episode artifact hash mismatch: {path}")
        episode = json.loads(path.read_text(encoding="utf-8"))
        ledger = _mapping(episode.get("ledger"), "episode.ledger")
        header = _mapping(ledger.get("header"), "episode.ledger.header")
        summary = _mapping(ledger.get("summary"), "episode.ledger.summary")
        if header.get("source_id") != SOURCE_ID or header.get("opponent") != opponent:
            raise ValueError(f"episode provenance mismatch: {path}")
        signature = str(header.get("scenario_signature", ""))
        mirror = str(header.get("mirror_sign", ""))
        steps = list(ledger.get("steps") or [])
        eligible_by_cell: defaultdict[tuple[str, str], int] = defaultdict(int)
        profile_allowed_by_cell: defaultdict[tuple[str, str], int] = defaultdict(int)
        for step in steps:
            if not isinstance(step, Mapping) or not _step_is_eligible(step):
                continue
            state = str(step.get("dynamic_state", ""))
            phase = str(step.get("phase", ""))
            if state not in STATES or phase not in PHASES:
                continue
            cell = (state, phase)
            eligible_by_cell[cell] += 1
            if bool(step.get("target_pair_allowed", False)):
                profile_allowed_by_cell[cell] += 1
        episode_rows.append(
            {
                "opponent": opponent,
                "scenario": header.get("scenario"),
                "scenario_signature": signature,
                "mirror_sign": mirror,
                "terminal_reason": episode.get("terminal_reason"),
                "first_pass_observed": bool(summary.get("first_pass_observed", False)),
                "continuity_valid": bool(summary.get("continuity_valid", False)),
                "eligible_postmerge_reentry_steps": sum(eligible_by_cell.values()),
                "cells": {f"{state}:{phase}": count for (state, phase), count in eligible_by_cell.items()},
            }
        )
        for state, phase in eligible_by_cell:
            key = (opponent, state, phase)
            bucket = buckets.setdefault(
                key,
                {
                    "opponent": opponent,
                    "state": state,
                    "phase": phase,
                    "eligible_steps": 0,
                    "profile_allowed_steps": 0,
                    "episodes": set(),
                    "signatures": set(),
                    "mirrors": set(),
                },
            )
            bucket["eligible_steps"] += eligible_by_cell[(state, phase)]
            bucket["profile_allowed_steps"] += profile_allowed_by_cell[(state, phase)]
            bucket["episodes"].add(str(header.get("scenario")))
            bucket["signatures"].add(signature)
            bucket["mirrors"].add(mirror)

    matrix: list[dict[str, Any]] = []
    for state in STATES:
        for phase in PHASES:
            for opponent in OPPONENTS:
                bucket = buckets.get((opponent, state, phase))
                matrix.append(
                    {
                        "state": state,
                        "phase": phase,
                        "opponent": opponent,
                        "eligible_steps": int(bucket["eligible_steps"]) if bucket else 0,
                        "profile_allowed_steps": int(bucket["profile_allowed_steps"]) if bucket else 0,
                        "qualifying_episodes": len(bucket["episodes"]) if bucket else 0,
                        "distinct_signatures": len(bucket["signatures"]) if bucket else 0,
                        "distinct_mirrors": len(bucket["mirrors"]) if bucket else 0,
                    }
                )

    candidates: list[dict[str, Any]] = []
    for state in STATES:
        for phase in PHASES:
            cells = [
                next(item for item in matrix if item["state"] == state and item["phase"] == phase and item["opponent"] == opponent)
                for opponent in OPPONENTS
            ]
            minimums = {
                "eligible_steps": min(item["eligible_steps"] for item in cells),
                "qualifying_episodes": min(item["qualifying_episodes"] for item in cells),
                "distinct_signatures": min(item["distinct_signatures"] for item in cells),
                "distinct_mirrors": min(item["distinct_mirrors"] for item in cells),
            }
            candidates.append(
                {
                    "state": state,
                    "phase": phase,
                    "per_opponent": {item["opponent"]: item for item in cells},
                    "cross_opponent_minimums": minimums,
                    "meets_b5_coverage_gate": (
                        minimums["eligible_steps"] >= 20
                        and minimums["qualifying_episodes"] >= 2
                        and minimums["distinct_signatures"] >= 2
                        and minimums["distinct_mirrors"] >= 2
                    ),
                }
            )
    candidates.sort(
        key=lambda item: (
            item["cross_opponent_minimums"]["eligible_steps"],
            item["cross_opponent_minimums"]["qualifying_episodes"],
            item["cross_opponent_minimums"]["distinct_signatures"],
            item["cross_opponent_minimums"]["distinct_mirrors"],
        ),
        reverse=True,
    )
    return {
        "source_id": SOURCE_ID,
        "mode": "read_only_completed_b5_analysis",
        "input": {
            "run_dir": str(root),
            "run_manifest_sha256": sha256_file(root / "run_manifest.json"),
            "gate_sha256": sha256_file(root / "phase_feasible_sampler_gate.json"),
            "gate_verdict": gate.get("verdict"),
        },
        "pooling_prohibited": True,
        "matrix": matrix,
        "candidates": candidates,
        "episode_rows": episode_rows,
        "cross_opponent_candidate_exists": any(item["meets_b5_coverage_gate"] for item in candidates),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _report(result: Mapping[str, Any]) -> str:
    lines = [
        "# P2-B5 Completed Reachability Atlas",
        "",
        "该 readout 只读取已完成的 B5 artifacts；不重跑 JSBSim，不改变 B5 gate，也不将 opponent 合并为强度排序。",
        "",
        "| State / phase | Min valid steps across opponents | Min episodes | Min signatures | Min mirrors | B5 coverage gate |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for item in result["candidates"]:
        minimums = item["cross_opponent_minimums"]
        lines.append(
            "| {state} / {phase} | {steps} | {episodes} | {signatures} | {mirrors} | {passed} |".format(
                state=item["state"],
                phase=item["phase"],
                steps=minimums["eligible_steps"],
                episodes=minimums["qualifying_episodes"],
                signatures=minimums["distinct_signatures"],
                mirrors=minimums["distinct_mirrors"],
                passed="pass" if item["meets_b5_coverage_gate"] else "fail",
            )
        )
    lines.extend(
        [
            "",
            "结论：" + ("存在跨对手候选，但该 atlas 不等价于训练授权。" if result["cross_opponent_candidate_exists"] else "当前 B5 包线中不存在满足全部三对手最小覆盖门槛的 state x phase 候选。"),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "b5_reachability_atlas.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_csv(
        args.out_dir / "b5_reachability_atlas.csv",
        result["matrix"],
        [
            "state",
            "phase",
            "opponent",
            "eligible_steps",
            "profile_allowed_steps",
            "qualifying_episodes",
            "distinct_signatures",
            "distinct_mirrors",
        ],
    )
    (args.out_dir / "b5_reachability_atlas_zh.md").write_text(_report(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "cross_opponent_candidate_exists": result["cross_opponent_candidate_exists"],
                "json": str(args.out_dir / "b5_reachability_atlas.json"),
                "csv": str(args.out_dir / "b5_reachability_atlas.csv"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
