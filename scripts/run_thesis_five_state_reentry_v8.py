#!/usr/bin/env python3
"""V8 non-learning strict-JSBSim probe for the corrected re-entry contract."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from uav_vpp_guidance.evaluation.raw_si_first_pass_event import RawSIFirstPassConfig
from uav_vpp_guidance.evaluation.raw_si_reentry_phase import (
    RawSIReentryPhaseConfig,
    RawSIReentryPhaseTracker,
)
from uav_vpp_guidance.evaluation.thesis_five_state_raw_si_v5_record_only import (
    FiveStateRawSIV5Episode,
    build_record_only_env,
)
from uav_vpp_guidance.training import thesis_neutral_postmerge_reentry_recovery_pilot as engine


SOURCE_ID = "THESIS-FIVE-STATE-REENTRY-PHASE-CONTRACT-PROBE-V8"
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
RANGE_REFERENCE_M = 5000.0
RANGE_RATE_REFERENCE_MPS = 200.0


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


def relabel_physical_phase(ledger: dict[str, Any], event_config: Mapping[str, Any]) -> dict[str, int]:
    """Attach a fresh physical phase label without overwriting legacy fields."""

    tracker = RawSIReentryPhaseTracker(
        RawSIReentryPhaseConfig(first_pass=RawSIFirstPassConfig(**dict(event_config)))
    )
    reentry_steps = 0
    reentry_with_66d = 0
    first_pass_steps = 0
    for index, step in enumerate(ledger["steps"]):
        base = np.asarray(step["base_observation_16d"], dtype=np.float64)
        if base.shape != (16,) or not np.isfinite(base).all():
            raise ValueError("V8 ledger contains invalid base observation")
        raw_range_m = float(base[0] * RANGE_REFERENCE_M)
        raw_rate_mps = float(base[1] * RANGE_RATE_REFERENCE_MPS)
        phase = tracker.update(raw_range_m, raw_rate_mps, step_index=index)
        if bool(step["raw_si_first_pass_latched"]) != tracker.first_pass_complete:
            raise ValueError("V8 raw first-pass replay mismatch")
        step["v8_raw_range_m"] = raw_range_m
        step["v8_raw_range_rate_mps"] = raw_rate_mps
        step["v8_phase"] = phase
        step["v8_first_pass_latched"] = tracker.first_pass_complete
        if tracker.first_pass_complete:
            first_pass_steps += 1
        if phase == "re_entry":
            reentry_steps += 1
            reentry_with_66d += int(step["observation_66d"] is not None)
    ledger["v8_phase_contract"] = {
        "version": "raw_physical_first_pass_aware_v1",
        "range_reference_m": RANGE_REFERENCE_M,
        "range_rate_reference_mps": RANGE_RATE_REFERENCE_MPS,
        "reentry_rate_mps": -25.0,
        "first_pass_event": dict(event_config),
    }
    return {
        "first_pass_steps": first_pass_steps,
        "reentry_steps": reentry_steps,
        "reentry_66d_steps": reentry_with_66d,
    }


def _gate_cell(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    qualifying = [row for row in rows if int(row["result"]["reentry_66d_steps"]) > 0]
    mirrors = {str(row["metadata"]["mirror_sign"]) for row in qualifying}
    signatures = {str(row["metadata"]["geometry_cell_id"]) for row in qualifying}
    reentry_steps = sum(int(row["result"]["reentry_66d_steps"]) for row in rows)
    return {
        "records": len(rows),
        "qualifying_episodes": len(qualifying),
        "reentry_66d_steps": reentry_steps,
        "mirrors": len(mirrors),
        "signatures": len(signatures),
        "passes": len(rows) == 12 and len(qualifying) >= 2 and reentry_steps >= 20 and len(mirrors) == 2 and len(signatures) >= 2,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = _load(args.config)
    if not args.execute:
        print(json.dumps({"source_id": SOURCE_ID, "mode": "preflight"}))
        return 0
    if config.get("source_id") != SOURCE_ID or config.get("authorization", {}).get("execution_permitted") is not True:
        raise ValueError("V8 execution is not authorized")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("V8 requires a clean worktree")

    manifest_path = _resolve(config["inputs"]["manifest"])
    if _hash(manifest_path) != config["inputs"]["manifest_sha256"]:
        raise ValueError("V8 manifest hash mismatch")
    manifest = _load(manifest_path)
    if len(manifest.get("scenarios", [])) != 12:
        raise ValueError("V8 requires exactly 12 frozen scenarios")
    out = _resolve(config["outputs"]["root"])
    if out.exists():
        raise ValueError("V8 output directory already exists")
    free_gb = shutil.disk_usage(out.parent).free / 1024 ** 3
    if free_gb < float(config["observability"]["min_free_disk_gb"]):
        raise ValueError("insufficient free disk for V8")

    registry = _load(_resolve(config["inputs"]["runtime_registry"]))
    out.mkdir(parents=True)
    with (out / "authorized_config.yaml").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(yaml.safe_dump(config, sort_keys=False))
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
                phase_summary = relabel_physical_phase(ledger, config["observability"]["raw_si_first_pass_event"])
                row = {
                    "source_id": SOURCE_ID,
                    "opponent": opponent,
                    "controller": "frozen_fixed_head_on",
                    "metadata": scenario["metadata"],
                    "result": {**result, **phase_summary, "phase_contract_passed": True},
                }
                records.append(row)
                engine.write_json(out / "episodes" / opponent / f"{scenario['name']}.json", {"record": row, "ledger": ledger})
        matrix = {
            opponent: _gate_cell([row for row in records if row["opponent"] == opponent])
            for opponent in OPPONENTS
        }
        gate = {"source_id": SOURCE_ID, "records": len(records), "matrix": matrix, "passes": all(cell["passes"] for cell in matrix.values())}
        engine.write_json(out / "phase_contract_gate.json", gate)
        engine.write_json(out / "run_manifest.json", {
            "source_id": SOURCE_ID,
            "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "config_sha256": _hash(args.config),
            "manifest_sha256": _hash(manifest_path),
            "records": len(records),
            "training_reward": "prohibited",
            "environment_reward_recording": "prohibited",
        })
        print(json.dumps(gate, indent=2))
        return 0
    except Exception as error:
        engine.write_json(out / "failure.json", {"source_id": SOURCE_ID, "error": repr(error), "completed_records": len(records)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
