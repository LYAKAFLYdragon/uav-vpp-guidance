#!/usr/bin/env python3
"""Independently verify V1 reward preflight phase and terminal reward semantics."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from uav_vpp_guidance.evaluation.terminal_outcome_reward import TerminalOutcomeRewardConfig
from uav_vpp_guidance.evaluation.thesis_reentry_reward_preflight import annotate_ledger


SOURCE_ID = "THESIS-REENTRY-TERMINAL-REWARD-PILOT-V1-PREFLIGHT"
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")


def _load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping: {path}")
    return value


def _reward_config(config: dict) -> TerminalOutcomeRewardConfig:
    value = config["reward"]["terminal_outcome_reward"]
    return TerminalOutcomeRewardConfig(
        win_bonus=float(value["win_bonus"]),
        loss_penalty=float(value["loss_penalty"]),
        draw_penalty=float(value["draw_penalty"]),
        ego_crash_oob_supplemental_penalty=float(value["ego_crash_oob_supplemental_penalty"]),
    )


def verify(root: Path, config: dict) -> dict:
    rows = []
    for opponent in OPPONENTS:
        paths = sorted((root / "episodes" / opponent).glob("*.json"))
        if len(paths) != 12:
            raise ValueError(f"missing preflight records for {opponent}")
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            record = payload["record"]
            if record.get("source_id") != SOURCE_ID or record.get("opponent") != opponent:
                raise ValueError("record identity mismatch")
            replay_ledger = copy.deepcopy(payload["ledger"])
            replay = annotate_ledger(
                replay_ledger,
                terminal_reason=str(record["result"]["terminal_reason"]),
                first_pass_event=config["observability"]["raw_si_first_pass_event"],
                reward_config=_reward_config(config),
            )
            if replay != {key: record["result"][key] for key in replay}:
                raise ValueError("annotation replay mismatch")
            stored_terms = [step["pilot_v1_terminal_reward"] for step in payload["ledger"]["steps"]]
            replay_terms = [step["pilot_v1_terminal_reward"] for step in replay_ledger["steps"]]
            if stored_terms != replay_terms or any(value != 0.0 for value in stored_terms[:-1]):
                raise ValueError("terminal reward timing mismatch")
            rows.append(record)
    matrix = {}
    for opponent in OPPONENTS:
        group = [row for row in rows if row["opponent"] == opponent]
        qualifying = [row for row in group if int(row["result"]["reentry_66d_steps"]) > 0]
        cell = {
            "records": len(group),
            "qualifying_episodes": len(qualifying),
            "reentry_66d_steps": sum(int(row["result"]["reentry_66d_steps"]) for row in group),
            "mirrors": len({row["metadata"]["mirror_sign"] for row in qualifying}),
            "signatures": len({row["metadata"]["geometry_cell_id"] for row in qualifying}),
            "terminal_reward_semantics_pass": all(bool(row["result"]["terminal_reward_final_step_only"]) for row in group),
        }
        cell["passes"] = cell["records"] == 12 and cell["qualifying_episodes"] >= 2 and cell["reentry_66d_steps"] >= 20 and cell["mirrors"] == 2 and cell["signatures"] >= 2 and cell["terminal_reward_semantics_pass"]
        matrix[opponent] = cell
    gate = json.loads((root / "preflight_gate.json").read_text(encoding="utf-8"))
    if gate.get("matrix") != matrix:
        raise ValueError("gate matrix mismatch")
    return {"source_id": SOURCE_ID, "verification_status": "pass", "records": len(rows), "matrix": matrix, "passes": all(cell["passes"] for cell in matrix.values())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        result, code = verify(args.output_dir, _load(args.config)), 0
    except Exception as error:
        result, code = {"source_id": SOURCE_ID, "verification_status": "fail", "error": str(error)}, 2
    text = json.dumps(result, indent=2)
    print(text)
    if args.report:
        args.report.write_text(text + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
