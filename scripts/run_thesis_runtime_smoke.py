#!/usr/bin/env python3
"""Run one disposable full-episode runtime-contract smoke per frozen opponent."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import run_thesis_five_state_v17_path_context_preflight as v17
import run_thesis_five_state_v22_observer_release_probe as v22
from uav_vpp_guidance.evaluation.thesis_runtime_smoke import OPPONENTS, summarize_records
from uav_vpp_guidance.training import thesis_neutral_postmerge_reentry_recovery_pilot as engine
from uav_vpp_guidance.training.thesis_shared_skill_geometry import FrozenP3Encoder


SMOKE_ID = "ENGINEERING-RUNTIME-SMOKE-NOT-A-SOURCE-ID"


def _load(path: Path) -> dict[str, Any]:
    return v17._load(path)


def _resolve(value: str) -> Path:
    return v17._resolve(value)


def _finite_vector(value: Any, name: str) -> list[float]:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"runtime smoke lacks {name}") from error
    if array.shape != (3,) or not np.isfinite(array).all():
        raise RuntimeError(f"runtime smoke lacks finite {name}")
    return array.tolist()


def _actual_aoa_rad(own_state: Mapping[str, Any]) -> float:
    """Derive physical AoA from the JSBSim body-velocity receipt.

    The unified JSBSim state exposes body velocity but does not currently carry
    an ``alpha_rad`` field.  This is the same actual-response definition used
    by the shared-skill geometry evaluator, rather than a commanded AoA proxy.
    """
    velocity = np.asarray(_finite_vector(own_state.get("velocity_body"), "velocity_body"), dtype=np.float64)
    return float(np.arctan2(velocity[2], velocity[0]))


class RuntimeSmokeEpisode(v22.Episode):
    """V22 observer protocol plus complete per-step execution receipts."""

    def run(self, scenario: Mapping[str, Any], seed: int) -> dict[str, Any]:
        self.runtime_trace: list[dict[str, Any]] = []
        result = super().run(scenario, seed)
        result["runtime_trace"] = list(self.runtime_trace)
        return result

    def _reference_step(self):
        info, done = super()._reference_step()
        own_state = info.get("own_state") if isinstance(info.get("own_state"), Mapping) else {}
        guidance = info.get("guidance_command") if isinstance(info.get("guidance_command"), Mapping) else {}
        self.runtime_trace.append(
            {
                "step": int(self.total_steps),
                "phase": self._phase(),
                "raw_range_m": float(self.base[0] * v17.v16.v15.v14.v11.RANGE_REFERENCE_M),
                "raw_range_rate_mps": float(self.base[1] * v17.v16.v15.v14.v11.RANGE_RATE_REFERENCE_MPS),
                "vp_world_offset_m": _finite_vector(info.get("vp_world_offset"), "vp_world_offset"),
                "guidance_command_normalized": [
                    float(guidance["nz_cmd"]) / 9.0,
                    float(guidance["roll_rate_cmd"]) / 1.5,
                    float(guidance["throttle_cmd"]),
                ],
                "aoa_rad": _actual_aoa_rad(own_state),
                "actual_nz_g": float(own_state.get("nz_g", info.get("nz_g"))),
                "runtime_vpp_anchor_mode": info.get("runtime_vpp_anchor_mode"),
            }
        )
        return info, done


def _scenario(config: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _load(_resolve(str(config["inputs"]["manifest"])))
    pair_key = str(config["scenario"]["pair_key"])
    matches = [item for item in manifest.get("scenarios", []) if item.get("metadata", {}).get("pair_key") == pair_key]
    if len(matches) != 1:
        raise RuntimeError("runtime smoke scenario pair_key must select exactly one frozen scenario")
    return copy.deepcopy(matches[0])


def run(config_path: Path, output: Path) -> dict[str, Any]:
    config = _load(config_path)
    if config.get("kind") != "engineering_runtime_smoke_not_scientific":
        raise RuntimeError("runtime smoke config identity drifted")
    if bool(config.get("training_permitted", True)) or bool(config.get("reward_change_permitted", True)):
        raise RuntimeError("runtime smoke must prohibit training and reward changes")
    if output.exists():
        raise RuntimeError("runtime smoke output path must be absent")
    scenario = _scenario(config)
    runtime = copy.deepcopy(_load(_resolve(str(config["inputs"]["runtime_template"]))))
    registry = _load(_resolve(str(config["inputs"]["runtime_registry"])))
    output.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    try:
        for opponent in OPPONENTS:
            env = v17.v16.v15.v14.build_record_only_env(runtime, registry, opponent)
            v22._configure_observer_only_warmup(env)
            policy = engine._specialist(registry, "frozen_fixed_head_on")
            encoder = FrozenP3Encoder(
                Path(runtime["inputs"]["frozen_p3_encoder"]),
                runtime["inputs"]["frozen_p3_encoder_sha256"],
            )
            episode = RuntimeSmokeEpisode(env, policy, encoder, runtime, v17.v16._compiler(runtime))
            try:
                result = episode.run(scenario, int(scenario["metadata"]["scenario_seed"]))
            finally:
                episode.close()
            record = {"smoke_id": SMOKE_ID, "opponent": opponent, "pair_key": scenario["metadata"]["pair_key"], **result}
            records.append(record)
            v17._write_json(output / "episodes" / f"{opponent}.json", {"record": record})
        summary = summarize_records(records)
        v17._write_json(output / "smoke_gate.json", {"smoke_id": SMOKE_ID, **summary})
        v17._write_json(
            output / "smoke_manifest.json",
            {
                "smoke_id": SMOKE_ID,
                "scientific_source_id": None,
                "not_paper_safe": True,
                "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "scenario_pair_key": scenario["metadata"]["pair_key"],
                "records": len(records),
                "config_path": str(config_path),
            },
        )
        return summary
    except Exception as error:
        v17._write_json(output / "failure.json", {"smoke_id": SMOKE_ID, "error": repr(error), "completed_records": len(records)})
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path("outputs/diagnostics/runtime_smoke") / datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    if not output.is_absolute():
        output = ROOT / output
    result = run(args.config, output)
    print(json.dumps({"output": str(output), **result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
