#!/usr/bin/env python3
"""Read-only B6 candidate analysis with artifact hash verification."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from uav_vpp_guidance.evaluation.global_advantage_p2_b6_reachability import (  # noqa: E402
    SOURCE_ID,
    build_b6_plan,
    evaluate_candidate_atlas,
    sha256_file,
)


DEFAULT_CONFIG = ROOT / "config" / "experiment" / "thesis_global_advantage_v1_p2_b6_opponent_conditional_reachability.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return payload


def analyze(run_dir: Path, config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    root = Path(run_dir)
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("source_id") != SOURCE_ID:
        raise ValueError("unexpected B6 run source")
    records = list(manifest.get("records") or [])
    if len(records) != 180:
        raise ValueError("B6 must contain all 180 preregistered records")
    ledgers: list[dict[str, Any]] = []
    for record in records:
        path = root / str(record["path"])
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise ValueError(f"B6 episode artifact hash mismatch: {path}")
        episode = json.loads(path.read_text(encoding="utf-8"))
        ledger = episode.get("ledger")
        if not isinstance(ledger, dict) or ledger.get("header", {}).get("source_id") != SOURCE_ID:
            raise ValueError(f"B6 episode provenance mismatch: {path}")
        ledgers.append(ledger)
    plan = build_b6_plan(_load_yaml(config_path))
    result = evaluate_candidate_atlas(ledgers, plan)
    result["input"] = {
        "run_dir": str(root),
        "run_manifest_sha256": sha256_file(root / "run_manifest.json"),
        "gate_sha256": sha256_file(root / "b6_candidate_atlas_gate.json"),
    }
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["state", "phase", "opponent", "eligible_steps", "qualifying_episodes", "distinct_signatures", "distinct_mirrors"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _report(result: Mapping[str, Any]) -> str:
    lines = [
        "# B6 Opponent-Conditional Reachability Atlas",
        "",
        "本分析只读取已经完成且 SHA 验证通过的 B6 artifacts，不重跑 episode，不汇总 opponent 成绩为单一强度。",
        "",
        "| State / phase | Min steps | Min episodes | Min signatures | Min mirrors | Pilot-input candidate |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for candidate in result["candidates"]:
        minimums = candidate["cross_opponent_minimums"]
        lines.append(
            "| {state} / {phase} | {steps} | {episodes} | {signatures} | {mirrors} | {candidate} |".format(
                state=candidate["state"],
                phase=candidate["phase"],
                steps=minimums["eligible_steps"],
                episodes=minimums["qualifying_episodes"],
                signatures=minimums["distinct_signatures"],
                mirrors=minimums["distinct_mirrors"],
                candidate="yes" if candidate["pilot_input_candidate"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "结论：" + ("发现一个可用于**起草** pilot 输入预注册的 cell；训练仍未授权。" if result["selected_candidate"] else "没有跨三对手的 pilot-input candidate；训练仍未授权。"),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    result = analyze(args.run_dir, args.config)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "b6_reachability_atlas.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(args.out_dir / "b6_reachability_atlas.csv", result["matrix"])
    (args.out_dir / "b6_reachability_atlas_zh.md").write_text(_report(result), encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"], "selected_candidate": result["selected_candidate"], "pilot_authorized": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
