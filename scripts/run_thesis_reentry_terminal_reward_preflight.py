#!/usr/bin/env python3
"""Run the non-learning JSBSim preflight for terminal-reward pilot V1."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from uav_vpp_guidance.evaluation.terminal_outcome_reward import TerminalOutcomeRewardConfig
from uav_vpp_guidance.evaluation.thesis_reentry_reward_preflight import annotate_ledger
from uav_vpp_guidance.evaluation.thesis_five_state_raw_si_v5_record_only import (
    FiveStateRawSIV5Episode,
    build_record_only_env,
)
from uav_vpp_guidance.training import thesis_neutral_postmerge_reentry_recovery_pilot as engine


SOURCE_ID = "THESIS-REENTRY-TERMINAL-REWARD-PILOT-V1-PREFLIGHT"
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")


def _load(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping: {path}")
    return value


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _reward_config(config: Mapping[str, Any]) -> TerminalOutcomeRewardConfig:
    values = config["reward"]["terminal_outcome_reward"]
    return TerminalOutcomeRewardConfig(
        win_bonus=float(values["win_bonus"]),
        loss_penalty=float(values["loss_penalty"]),
        draw_penalty=float(values["draw_penalty"]),
        ego_crash_oob_supplemental_penalty=float(values["ego_crash_oob_supplemental_penalty"]),
    )


def _gate_cell(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    qualifying = [row for row in rows if int(row["result"]["reentry_66d_steps"]) > 0]
    mirrors = {str(row["metadata"]["mirror_sign"]) for row in qualifying}
    signatures = {str(row["metadata"]["geometry_cell_id"]) for row in qualifying}
    semantic_ok = all(bool(row["result"]["terminal_reward_final_step_only"]) for row in rows)
    return {
        "records": len(rows),
        "qualifying_episodes": len(qualifying),
        "reentry_66d_steps": sum(int(row["result"]["reentry_66d_steps"]) for row in rows),
        "mirrors": len(mirrors),
        "signatures": len(signatures),
        "terminal_reward_semantics_pass": semantic_ok,
        "passes": len(rows) == 12 and len(qualifying) >= 2
        and sum(int(row["result"]["reentry_66d_steps"]) for row in rows) >= 20
        and len(mirrors) == 2 and len(signatures) >= 2 and semantic_ok,
    }


def _validate_inputs(config: Mapping[str, Any]) -> None:
    for label in ("train_distribution", "dev_manifest", "heldout_manifest", "runtime_template", "runtime_registry"):
        path = _resolve(config["inputs"][label])
        expected = str(config["inputs"][f"{label}_sha256"]).lower()
        if not path.is_file() or _hash(path) != expected:
            raise ValueError(f"{label} hash mismatch")
    encoder = Path(config["inputs"]["frozen_p3_encoder"])
    if not encoder.is_file() or _hash(encoder) != str(config["inputs"]["frozen_p3_encoder_sha256"]).lower():
        raise ValueError("frozen P3 encoder hash mismatch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = _load(args.config)
    if not args.execute:
        print(json.dumps({"source_id": SOURCE_ID, "mode": "preflight"}))
        return 0
    if config.get("authorization", {}).get("preflight_execution_permitted") is not True:
        raise ValueError("preflight is not authorized")
    if config.get("authorization", {}).get("training_permitted") is not False:
        raise ValueError("preflight config must continue to prohibit training")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("preflight requires a clean worktree")
    _validate_inputs(config)

    out = _resolve(config["outputs"]["preflight_root"])
    if out.exists():
        raise ValueError("preflight output directory already exists")
    if shutil.disk_usage(out.parent).free / 1024 ** 3 < float(config["observability"]["min_free_disk_gb"]):
        raise ValueError("insufficient free disk")
    manifest = _load(_resolve(config["inputs"]["dev_manifest"]))
    if len(manifest.get("scenarios", [])) != 12:
        raise ValueError("preflight requires dev12 manifest")
    registry = _load(_resolve(config["inputs"]["runtime_registry"]))
    reward_config = _reward_config(config)
    out.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    try:
        for opponent in OPPONENTS:
            for scenario in manifest["scenarios"]:
                env = build_record_only_env(config, registry, opponent)
                policy = engine._specialist(registry, "frozen_fixed_head_on")
                encoder = engine.FrozenP3Encoder(
                    Path(config["inputs"]["frozen_p3_encoder"]),
                    config["inputs"]["frozen_p3_encoder_sha256"],
                )
                episode = FiveStateRawSIV5Episode(env, policy, encoder, config)
                try:
                    result, ledger = episode.run(scenario, int(scenario["metadata"]["scenario_seed"]))
                finally:
                    episode.close()
                annotation = annotate_ledger(
                    ledger,
                    terminal_reason=str(result["terminal_reason"]),
                    first_pass_event=config["observability"]["raw_si_first_pass_event"],
                    reward_config=reward_config,
                )
                terms = [float(step["pilot_v1_terminal_reward"]) for step in ledger["steps"]]
                final_only = all(value == 0.0 for value in terms[:-1]) and terms[-1] == float(annotation["terminal_reward_total"])
                row = {
                    "source_id": SOURCE_ID,
                    "opponent": opponent,
                    "controller": "frozen_fixed_head_on",
                    "metadata": scenario["metadata"],
                    "result": {
                        **result,
                        **annotation,
                        "terminal_reward_final_step_only": final_only,
                        "environment_reward_recording": "prohibited",
                    },
                }
                records.append(row)
                engine.write_json(out / "episodes" / opponent / f"{scenario['name']}.json", {"record": row, "ledger": ledger})
        matrix = {opponent: _gate_cell([row for row in records if row["opponent"] == opponent]) for opponent in OPPONENTS}
        gate = {"source_id": SOURCE_ID, "records": len(records), "matrix": matrix, "passes": all(cell["passes"] for cell in matrix.values())}
        engine.write_json(out / "preflight_gate.json", gate)
        engine.write_json(out / "run_manifest.json", {
            "source_id": SOURCE_ID,
            "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "config_sha256": _hash(args.config),
            "dev_manifest_sha256": _hash(_resolve(config["inputs"]["dev_manifest"])),
            "records": len(records),
            "training_permitted": False,
            "environment_reward_recording": "prohibited",
        })
        print(json.dumps(gate, indent=2))
        return 0
    except Exception as error:
        engine.write_json(out / "failure.json", {"source_id": SOURCE_ID, "error": repr(error), "completed_records": len(records)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
