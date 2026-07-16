#!/usr/bin/env python3
"""Collect V22 observer-only predictor-release evidence without learning."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import run_thesis_five_state_v17_path_context_preflight as v17
from uav_vpp_guidance.common.provenance import record_config_override
from uav_vpp_guidance.evaluation.v22_observer_release_protocol import V22ObserverReleaseMixin
from uav_vpp_guidance.training import thesis_neutral_postmerge_reentry_recovery_pilot as engine
from uav_vpp_guidance.training.thesis_shared_skill_geometry import FrozenP3Encoder


SOURCE_ID = "THESIS-FIVE-STATE-V22-OBSERVER-RELEASE-PROBE"
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")


def _load_config(path: Path) -> dict[str, Any]:
    overlay = v17._load(path)
    if "base_config" not in overlay:
        return overlay
    base_path = v17._resolve(overlay["base_config"])
    if v17._hash(base_path) != overlay.get("base_config_sha256"):
        raise RuntimeError("V22 authorization base hash mismatch")
    result = copy.deepcopy(v17._load(base_path))
    result["authorization"] = copy.deepcopy(overlay.get("authorization", {}))
    result["_authorization_execution"] = copy.deepcopy(overlay.get("execution", {}))
    return result


def validate_design(config_path: Path, *, allow_existing_output: bool = False) -> dict[str, Any]:
    """Fail closed unless the non-learning V22 design remains unchanged."""
    config = _load_config(config_path)
    if config.get("source_id") != SOURCE_ID:
        raise RuntimeError("V22 source ID drifted")
    for key in (
        "training_permitted",
        "tuning_permitted",
        "reward_change_permitted",
        "candidate_policy_loading_permitted",
        "combat_finetune_permitted",
    ):
        if config.get("authorization", {}).get(key) is not False:
            raise RuntimeError(f"V22 must prohibit {key}")
    if tuple(config.get("opponents", {}).get("order", ())) != OPPONENTS:
        raise RuntimeError("V22 opponent order drifted")
    if config.get("opponents", {}).get("pooled_gate") != "prohibited":
        raise RuntimeError("V22 must retain separate opponent gates")
    protocol = config.get("protocol", {})
    expected_protocol = {
        "observer_only_warmup": True,
        "initial_anchor_mode": "current_target",
        "release_anchor_mode": "predicted_target",
        "history_frames_required": 10,
        "max_anchor_switches": 1,
        "post_release_prediction_fallback": "fail_closed",
    }
    if {key: protocol.get(key) for key in expected_protocol} != expected_protocol:
        raise RuntimeError("V22 observer-release protocol drifted")
    manifest_path = v17._resolve(config["inputs"]["manifest"])
    if v17._hash(manifest_path) != config["inputs"].get("manifest_sha256"):
        raise RuntimeError("V22 manifest hash mismatch")
    manifest = v17._load(manifest_path)
    if len(manifest.get("scenarios", [])) != 24:
        raise RuntimeError("V22 requires exactly 24 frozen scenarios")
    for key in ("runtime_template", "runtime_registry"):
        path = v17._resolve(config["inputs"][key])
        if not path.is_file() or v17._hash(path) != config["inputs"].get(f"{key}_sha256"):
            raise RuntimeError(f"V22 frozen input mismatch: {key}")
    output = v17._resolve(config["outputs"]["root"])
    if not output.is_absolute() or (output.exists() and not allow_existing_output):
        raise RuntimeError("V22 output root must be absolute and absent")
    expected_records = len(OPPONENTS) * len(manifest["scenarios"])
    if int(config["outputs"].get("expected_records", -1)) != expected_records:
        raise RuntimeError("V22 design record universe drifted")
    return {
        "source_id": SOURCE_ID,
        "planned_records": expected_records,
        "output_root_absent": not output.exists(),
        "execution_permitted": bool(config.get("authorization", {}).get("execution_permitted")),
    }


def _require_execution_authorization(config: Mapping[str, Any], expected_records: int) -> None:
    execution = config.get("_authorization_execution")
    if config.get("authorization", {}).get("execution_permitted") is not True or not isinstance(execution, Mapping):
        raise RuntimeError("V22 execution is not authorized")
    if int(execution.get("expected_records", -1)) != expected_records:
        raise RuntimeError("V22 authorization record universe drifted")
    required_sha = str(execution.get("required_implementation_git_sha") or "")
    actual_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if not required_sha or actual_sha != required_sha:
        raise RuntimeError("V22 implementation SHA is not exactly authorized")
    files = execution.get("authorized_code_files")
    if not isinstance(files, list) or not files:
        raise RuntimeError("V22 authorization file whitelist is missing")
    for item in files:
        path = v17._resolve(str(item.get("path") or ""))
        if not path.is_file() or v17._hash(path) != str(item.get("sha256") or ""):
            raise RuntimeError(f"V22 authorized code hash mismatch: {path}")


class Episode(V22ObserverReleaseMixin, v17.PathContextEpisode):
    def _observe(self, observation: Mapping[str, Any]) -> None:
        """Record each observer receipt immediately after the base observation update."""
        super()._observe(observation)
        self._v22_after_observation()

    def run(self, scenario: Mapping[str, Any], seed: int) -> dict[str, Any]:
        self._v22_reset_observer_protocol()
        result = super().run(scenario, seed)
        result.update(
            anchor_switches=list(self._v21_switches),
            released=self._v21_released,
            observer_trace=list(self._v22_observer_trace),
            warmup_control_isolated=not self._v22_warmup_control_consumption,
            post_release_valid_steps=self._v22_post_release_valid_steps,
            post_release_prediction_fallback=self._v22_post_release_prediction_fallback,
        )
        return result


def _configure_observer_only_warmup(env: Any) -> None:
    integration = env.config.setdefault("trajectory_prediction", {}).setdefault("integration", {})
    old_value = integration.get("observer_only_warmup")
    integration["observer_only_warmup"] = True
    record_config_override(
        env.config,
        "trajectory_prediction.integration.observer_only_warmup",
        True,
        old_value,
        "V22 observer-only release",
    )


def _summarize(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    matrix = {
        opponent: {
            "records": sum(row["opponent"] == opponent for row in rows),
            "released": sum(row["opponent"] == opponent and row["released"] for row in rows),
            "post_release_valid_steps": sum(
                int(row["post_release_valid_steps"])
                for row in rows
                if row["opponent"] == opponent
            ),
            "warmup_control_isolated": all(
                bool(row["warmup_control_isolated"])
                for row in rows
                if row["opponent"] == opponent
            ),
            "post_release_prediction_fallback": any(
                bool(row["post_release_prediction_fallback"])
                for row in rows
                if row["opponent"] == opponent
            ),
        }
        for opponent in OPPONENTS
    }
    passes = len(rows) == 72 and all(
        values["released"] >= 2
        and values["post_release_valid_steps"] >= 20
        and values["warmup_control_isolated"]
        and not values["post_release_prediction_fallback"]
        for values in matrix.values()
    )
    return {"source_id": SOURCE_ID, "records": len(rows), "matrix": matrix, "passes": passes}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    config = _load_config(args.config)
    preflight = validate_design(args.config)
    if not args.execute:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0
    _require_execution_authorization(config, int(preflight["planned_records"]))
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("V22 requires a clean worktree")

    manifest = v17._load(v17._resolve(config["inputs"]["manifest"]))
    runtime = copy.deepcopy(v17._load(v17._resolve(config["inputs"]["runtime_template"])))
    registry = v17._load(v17._resolve(config["inputs"]["runtime_registry"]))
    output = v17._resolve(config["outputs"]["root"])
    output.mkdir(parents=True)
    v17._write_json(
        output / "authorized_config.json",
        {"config": config, "config_sha256": v17._hash(args.config)},
    )
    rows: list[dict[str, Any]] = []
    try:
        for opponent in OPPONENTS:
            for scenario in manifest["scenarios"]:
                env = v17.v16.v15.v14.build_record_only_env(runtime, registry, opponent)
                _configure_observer_only_warmup(env)
                policy = engine._specialist(registry, "frozen_fixed_head_on")
                encoder = FrozenP3Encoder(
                    Path(runtime["inputs"]["frozen_p3_encoder"]),
                    runtime["inputs"]["frozen_p3_encoder_sha256"],
                )
                episode = Episode(env, policy, encoder, runtime, v17.v16._compiler(runtime))
                try:
                    result = episode.run(scenario, int(scenario["metadata"]["scenario_seed"]))
                finally:
                    episode.close()
                row = {
                    "source_id": SOURCE_ID,
                    "opponent": opponent,
                    "pair_key": scenario["metadata"]["pair_key"],
                    **result,
                }
                rows.append(row)
                v17._write_json(output / "episodes" / opponent / f"{scenario['name']}.json", {"record": row})
        gate = _summarize(rows)
        v17._write_json(output / "gate.json", gate)
        v17._write_json(
            output / "run_manifest.json",
            {
                "source_id": SOURCE_ID,
                "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "config_sha256": v17._hash(args.config),
                "manifest_sha256": v17._hash(v17._resolve(config["inputs"]["manifest"])),
                "records": len(rows),
                "training": "prohibited",
                "authorized_code_files": copy.deepcopy(config["_authorization_execution"]["authorized_code_files"]),
            },
        )
        print(json.dumps(gate, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        v17._write_json(
            output / "failure.json",
            {"source_id": SOURCE_ID, "error": repr(error), "completed_records": len(rows)},
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
