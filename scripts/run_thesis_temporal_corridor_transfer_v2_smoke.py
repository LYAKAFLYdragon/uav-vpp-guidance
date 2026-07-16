#!/usr/bin/env python3
"""Run a disposable three-opponent build/reset/telemetry smoke for transfer V2."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import run_thesis_temporal_corridor_transfer_v2 as runner
from uav_vpp_guidance.envs.observation import compute_relative_geometry
from uav_vpp_guidance.evaluation.thesis_temporal_corridor_transfer_v2 import OPPONENTS
from uav_vpp_guidance.training import thesis_neutral_postmerge_reentry_recovery_pilot as engine
from uav_vpp_guidance.training.thesis_shared_skill_geometry import FrozenP3Encoder


SMOKE_ID = "ENGINEERING-TEMPORAL-CORRIDOR-TRANSFER-V2-BUILD-SMOKE-NOT-A-SOURCE-ID"


class TemporalCorridorTransferV2SmokeError(RuntimeError):
    """Raised when the repair has not reached a usable record-only environment."""


def _finite_vector(value: Any, name: str) -> list[float]:
    try:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as error:
        raise TemporalCorridorTransferV2SmokeError(f"smoke lacks {name}") from error
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise TemporalCorridorTransferV2SmokeError(f"smoke lacks finite {name}")
    return vector.tolist()


class SmokeEpisode(runner.Episode):
    """Record the V16 run-in receipts before making the predictor-ready assertion."""

    def run(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.smoke_reference_trace: list[Mapping[str, Any]] = []
        return super().run(*args, **kwargs)

    def _reference_step(self):
        info, done = super()._reference_step()
        self.smoke_reference_trace.append(dict(info))
        return info, done


def _record_one(config: Mapping[str, Any], registry: Mapping[str, Any], scenario: Mapping[str, Any], opponent: str) -> dict[str, Any]:
    env = runner.build_record_only_env(config, registry, opponent)
    policy = engine._specialist(registry, "frozen_fixed_head_on")
    encoder = FrozenP3Encoder(
        Path(config["inputs"]["frozen_p3_encoder"]),
        config["inputs"]["frozen_p3_encoder_sha256"],
    )
    episode = SmokeEpisode(env, policy, encoder, config, runner.v16._compiler(config))
    try:
        if env._backend != "jsbsim":
            raise TemporalCorridorTransferV2SmokeError("smoke did not construct strict JSBSim")
        result = episode.run(
            copy.deepcopy(dict(scenario)),
            int(scenario["metadata"]["scenario_seed"]),
            runner.REFERENCE_METHOD,
            None,
        )
        observation = episode.observation
        if not isinstance(observation, Mapping):
            raise TemporalCorridorTransferV2SmokeError("smoke lost the final observation")
        vector = np.asarray(observation["observation_vector"], dtype=np.float64).reshape(-1)
        if vector.size == 0 or not np.isfinite(vector).all():
            raise TemporalCorridorTransferV2SmokeError("smoke observation is invalid")
        if not episode.smoke_reference_trace:
            raise TemporalCorridorTransferV2SmokeError("smoke did not execute the V16 run-in")
        if any(bool(info.get("backend_fallback_occurred", False)) for info in episode.smoke_reference_trace):
            raise TemporalCorridorTransferV2SmokeError("smoke backend fallback")
        info = episode.smoke_reference_trace[-1]
        if info.get("prediction_valid") is not True or info.get("prediction_fallback") is True:
            raise TemporalCorridorTransferV2SmokeError("smoke predictor was not ready after V16 run-in")
        _own, _target, predicted = episode._context()
        guidance = info.get("guidance_command")
        if not isinstance(guidance, Mapping):
            raise TemporalCorridorTransferV2SmokeError("smoke guidance command is missing")
        command = [
            float(guidance["nz_cmd"]),
            float(guidance["roll_rate_cmd"]),
            float(guidance["throttle_cmd"]),
        ]
        if not np.isfinite(np.asarray(command, dtype=np.float64)).all():
            raise TemporalCorridorTransferV2SmokeError("smoke guidance command is non-finite")
        own_state = info.get("own_state")
        if not isinstance(own_state, Mapping) or not np.isfinite(float(own_state.get("nz_g"))):
            raise TemporalCorridorTransferV2SmokeError("smoke actual nz response is missing")
        return {
            "opponent": opponent,
            "backend": env._backend,
            "observation_dim": int(vector.size),
            "predicted_target_position_neu": predicted,
            "run_in_steps": len(episode.smoke_reference_trace),
            "handoff_reached": bool(result["handoff_reached"]),
            "phase_trace_steps": len(result["phase_trace"]),
            "vpp_world_offset_m": _finite_vector(info.get("vp_world_offset"), "vp_world_offset"),
            "guidance_command": command,
            "actual_nz_g": float(own_state["nz_g"]),
            "backend_fallback_occurred": False,
            "prediction_fallback": False,
            "telemetry_passed": True,
        }
    finally:
        episode.close()


def run(config_path: Path, output: Path) -> dict[str, Any]:
    config = runner._config(config_path)
    if config["authorization"].get("execution_permitted") is not False:
        raise TemporalCorridorTransferV2SmokeError("smoke must use the non-authorized V2 design")
    runner.validate_design(config_path)
    if output.exists():
        raise TemporalCorridorTransferV2SmokeError("smoke output must be absent")
    manifest = runner._load(runner._resolve(str(config["inputs"]["manifest"])))
    registry = runner._load(runner._resolve(str(config["inputs"]["runtime_registry"])))
    scenario = manifest["scenarios"][0]
    output.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    try:
        for opponent in OPPONENTS:
            record = _record_one(config, registry, scenario, opponent)
            records.append(record)
            runner._write(output / "records" / f"{opponent}.json", {"record": record})
        gate = {
            "smoke_id": SMOKE_ID,
            "source_id_under_test": runner.SOURCE_ID,
            "expected_opponents": list(OPPONENTS),
            "records": len(records),
            "all_opponents_present": {record["opponent"] for record in records} == set(OPPONENTS),
            "all_telemetry_passed": all(bool(record["telemetry_passed"]) for record in records),
        }
        gate["passes"] = bool(gate["all_opponents_present"] and gate["all_telemetry_passed"])
        runner._write(output / "smoke_gate.json", gate)
        runner._write(
            output / "smoke_manifest.json",
            {
                "smoke_id": SMOKE_ID,
                "scientific_source_id": None,
                "source_id_under_test": runner.SOURCE_ID,
                "not_paper_safe": True,
                "config_sha256": runner._hash(config_path),
                "scenario_signature": scenario["metadata"]["scenario_signature"],
                "records": len(records),
                "generated_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        return gate
    except Exception as error:
        runner._write(output / "failure.json", {"smoke_id": SMOKE_ID, "error": repr(error), "completed_records": len(records)})
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / "outputs" / "diagnostics" / "temporal_corridor_transfer_v2_runtime_smoke" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not output.is_absolute():
        output = ROOT / output
    print(json.dumps({"output": str(output), **run(args.config, output)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
