#!/usr/bin/env python3
"""Run one independent paired strict-JSBSim probe for the simplest sequence."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uav_vpp_guidance.common.hash import config_sha256, file_sha256
from uav_vpp_guidance.common.git import get_git_info
from uav_vpp_guidance.common.provenance import record_config_override_if_changed
from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.hierarchy.specialist_policy import FrozenSpecialistPolicy
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv

SOURCE_ID = "THESIS-MILD-DISADVANTAGE-TWO-NODE-ACTION-SEQUENCE-RUNTIME-PROBE-V1-20260904"
DEFAULT_CONFIG = ROOT / "config/experiment/mild_disadvantage_two_node_action_sequence_runtime_probe_v1.yaml"
DEFAULT_OUTPUT = ROOT / "outputs/mild_disadvantage_two_node_action_sequence_runtime_probe_v1"
SOURCE_TRACE_SOURCE_ID = "THESIS-MILD-DISADVANTAGE-HANDOFF-SOURCE-TRACES-ISOLATED-V1-20260904"
MATCHED_TELEMETRY = ROOT / "scripts/run_mild_disadvantage_envelope_recovery_matched_telemetry_v1.py"
LOCAL_COLLECTOR = ROOT / "scripts/run_mild_disadvantage_handoff_local_response_collector_v1.py"
OFFSETS = (4, 12, 48)
METRICS = ("range_m", "range_rate_mps", "ata_deg", "aa_deg", "specific_energy_m2ps2", "altitude_m")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def receipt(path: Path) -> dict[str, Any]:
    exists = path.is_file()
    return {"path": str(path.resolve()), "exists": exists, "size_bytes": path.stat().st_size if exists else None, "sha256": file_sha256(path) if exists else None}


def set_nested(config: dict[str, Any], key: str, value: Any, source: str) -> None:
    current: Any = config
    parts = key.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    old = current.get(parts[-1])
    current[parts[-1]] = value
    record_config_override_if_changed(config, key, value, old, source)


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


class TwoNodePolicy:
    """Add two fixed deltas to successive H0 actions, without clipping."""

    def __init__(self, base: Any, enabled: bool, node0: np.ndarray, node1: np.ndarray) -> None:
        self.base = base
        self.enabled = enabled
        self.node0 = np.asarray(node0, dtype=np.float64)
        self.node1 = np.asarray(node1, dtype=np.float64)
        self.calls = 0
        self.records: list[dict[str, Any]] = []
        self.out_of_bounds = False

    def get_deterministic_action(self, observation: Any) -> np.ndarray:
        base = np.asarray(self.base.get_deterministic_action(observation), dtype=np.float64).reshape(-1)
        if base.shape != (3,) or not np.isfinite(base).all():
            raise ValueError(f"invalid base H0 action: {base}")
        delta = np.zeros(3, dtype=np.float64)
        node = "none"
        if self.enabled and self.calls == 0:
            delta = self.node0.copy()
            node = "node0"
        elif self.enabled and self.calls == 8:
            delta = self.node1.copy()
            node = "node1"
        action = base + delta
        in_bounds = bool(np.all(action >= -1.0) and np.all(action <= 1.0))
        self.out_of_bounds = self.out_of_bounds or not in_bounds
        self.records.append({
            "h0_call_index": self.calls,
            "node": node,
            "base_action_0": float(base[0]), "base_action_1": float(base[1]), "base_action_2": float(base[2]),
            "delta_0": float(delta[0]), "delta_1": float(delta[1]), "delta_2": float(delta[2]),
            "action_0": float(action[0]), "action_1": float(action[1]), "action_2": float(action[2]),
            "action_in_bounds": in_bounds,
        })
        self.calls += 1
        return action


def prepare_config(base: dict[str, Any], spec: dict[str, Any], matched_spec: dict[str, Any], matched_probe: Any, opponent: str, max_steps: int) -> dict[str, Any]:
    cfg = matched_probe._prepare_config(base, matched_spec, opponent)
    source = "run_mild_disadvantage_two_node_action_sequence_runtime_probe_v1.py"
    set_nested(cfg, "experiment.name", spec["experiment"]["name"], f"{source}:name")
    set_nested(cfg, "experiment.source_id", SOURCE_ID, f"{source}:source_id")
    set_nested(cfg, "experiment.mode", "evaluation", f"{source}:mode")
    set_nested(cfg, "protocol.source_id", SOURCE_ID, f"{source}:protocol_source_id")
    set_nested(cfg, "protocol.opponent", opponent, f"{source}:protocol_opponent")
    set_nested(cfg, "env.legacy_project_root", str((ROOT.parent / "uav-vpp-guidance-artifacts/CAN-20260705/assets").resolve()), f"{source}:jsbsim_root")
    set_nested(cfg, "env.max_high_level_steps", int(max_steps), f"{source}:response_window")
    set_nested(cfg, "trajectory_prediction.allow_device_fallback", False, f"{source}:predictor_fallback")
    return cfg


def build_opponent(name: str, cfg: dict[str, Any]):
    from uav_vpp_guidance.envs.expert_opponent import ExpertOpponent
    from uav_vpp_guidance.envs.end_to_end_opponent import EndToEndOpponent
    entry = cfg["opponent_registry"][name]
    if name == "expert":
        return ExpertOpponent(entry.get("config", {})), {"stage": "expert"}
    return EndToEndOpponent.from_registry(entry), {"stage": "end_to_end", **entry}


def run_condition(source: dict[str, Any], condition: str, cfg: dict[str, Any], matched: Any, legacy: Any, matched_spec: dict[str, Any], paths: dict[str, Path]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    opponent_policy, opponent_config = build_opponent(source["opponent"], cfg)
    env = CloseRangeTrackingEnv(cfg, opponent_policy=opponent_policy, opponent_config=opponent_config)
    try:
        h0_agent = PPOAgent(obs_dim=16, action_dim=3, config=cfg, device="cpu")
        h0_load = h0_agent.load(str(paths["h0"]))
        h0 = load_module(LOCAL_COLLECTOR, "two_node_local_collector_helpers").H0Adapter(h0_agent)
        crossing = FrozenSpecialistPolicy(str(paths["crossing"]), str(paths["crossing_config"]), device="cpu")
        node0 = np.asarray([0.0, 0.08, 0.0], dtype=np.float64)
        node1 = np.asarray([-0.08, 0.0, 0.0], dtype=np.float64)
        policy = TwoNodePolicy(h0, condition == "candidate", node0, node1)
        stable_points = [legacy.EnvelopePoint(float(p[0]), float(p[1])) for p in matched_spec["recovery"]["stable_points_m"]]
        recovery_cfg = matched_spec["recovery"]
        recovery = legacy.CapabilityEnvelopeRecoveryPolicy(stable_points, **{
            key: recovery_cfg[key] for key in ("nominal_mode", "recovery_mode", "x_tolerance_m", "y_tolerance_m", "max_recovery_steps", "required_dwell_steps", "max_clipping_rate", "min_altitude_m", "max_range_m")
        })
        scenario = legacy._make_scenario(source["candidate"], matched_spec["probe"]["candidates"][source["candidate"]], matched_spec["probe"]["fixed_target_conditions"])
        result, trace = matched._run_episode(env, policy, crossing, recovery, scenario, int(source["seed"]), source["opponent"], source["candidate"], matched_spec["probe"]["calibration"], legacy)
        handoff_rows = [row for row in trace if int(row["step"]) == int(source["handoff_index"])]
        replay_hash = handoff_rows[0].get("state_hash_before_action", "") if handoff_rows else ""
        for row in trace:
            row["source_id"] = SOURCE_ID
            row["source_trace_id"] = source["source_trace_id"]
            row["condition"] = condition
            row["handoff_step"] = int(source["handoff_index"])
            row["response_offset"] = int(row["step"]) - int(source["handoff_index"])
            row["source_handoff_hash"] = source["handoff_hash"]
            row["handoff_replay_hash"] = replay_hash
            row["handoff_hash_match"] = replay_hash == source["handoff_hash"]
            row["sequence_node"] = next((item["node"] for item in policy.records if item["h0_call_index"] == max(0, int(row["step"]) - int(source["handoff_index"]))), "none") if row["mode"] == "h0_current_geometry" else "none"
            row["sequence_delta_0"] = 0.0
            row["sequence_delta_1"] = 0.0
            row["sequence_delta_2"] = 0.0
            speed = float(row.get("own_speed_mps", float("nan")))
            altitude = float(row.get("altitude_m", float("nan")))
            row["specific_energy_m2ps2"] = 0.5 * speed * speed + 9.80665 * altitude
            if condition == "candidate" and row["sequence_node"] == "node0":
                row["sequence_delta_0"], row["sequence_delta_1"], row["sequence_delta_2"] = 0.0, 0.08, 0.0
            elif condition == "candidate" and row["sequence_node"] == "node1":
                row["sequence_delta_0"], row["sequence_delta_1"], row["sequence_delta_2"] = -0.08, 0.0, 0.0
        response_rows = {int(row["response_offset"]): row for row in trace}
        result.update({
            "source_id": SOURCE_ID, "source_trace_id": source["source_trace_id"], "condition": condition,
            "source_handoff_hash": source["handoff_hash"], "handoff_replay_hash": replay_hash,
            "handoff_hash_match": replay_hash == source["handoff_hash"],
            "sequence_node0": json.dumps(node0.tolist()), "sequence_node1": json.dumps(node1.tolist()),
            "node0_observed": bool(any(item["node"] == "node0" for item in policy.records)),
            "node1_observed": bool(any(item["node"] == "node1" for item in policy.records)),
            "action_out_of_bounds": bool(policy.out_of_bounds),
            "response_window_complete": all(offset in response_rows for offset in OFFSETS),
            "response_window_prediction_fallback": any(bool(response_rows[offset].get("prediction_fallback", False)) for offset in OFFSETS if offset in response_rows),
            "response_window_backend_fallback": any(bool(response_rows[offset].get("backend_fallback_occurred", False)) for offset in OFFSETS if offset in response_rows),
            "response_window_clipping": any(float(response_rows[offset].get("clipping_rate", 0.0)) > 0.0 for offset in OFFSETS if offset in response_rows),
            "h0_checkpoint_obs_dim": int(h0_load.get("obs_dim", 0)),
            "h0_checkpoint_action_dim": int(h0_load.get("action_dim", 0)),
            "policy_call_count": policy.calls,
            "terminal_reason": result.get("reason", "") or "window_end",
        })
        return result, trace
    finally:
        env.close()


def pair_rows(results: list[dict[str, Any]], traces: dict[tuple[str, str], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for key in sorted({(row["source_trace_id"], row["opponent"], row["candidate"]) for row in results}):
        source_trace_id, opponent, candidate = key
        base = next(row for row in results if row["source_trace_id"] == source_trace_id and row["condition"] == "baseline")
        cand = next(row for row in results if row["source_trace_id"] == source_trace_id and row["condition"] == "candidate")
        base_trace = {int(row["response_offset"]): row for row in traces[(source_trace_id, "baseline")]}
        cand_trace = {int(row["response_offset"]): row for row in traces[(source_trace_id, "candidate")]}
        for offset in OFFSETS:
            row: dict[str, Any] = {"source_id": SOURCE_ID, "source_trace_id": source_trace_id, "opponent": opponent, "candidate": candidate, "response_offset": offset, "pair_valid": offset in base_trace and offset in cand_trace}
            for metric in METRICS:
                b = float(base_trace[offset][metric]) if offset in base_trace else float("nan")
                c = float(cand_trace[offset][metric]) if offset in cand_trace else float("nan")
                row[f"baseline_{metric}"] = b
                row[f"candidate_{metric}"] = c
                row[f"delta_{metric}"] = c - b if math.isfinite(b) and math.isfinite(c) else float("nan")
            output.append(row)
    return output


def run(spec: dict[str, Any], output_root: Path) -> dict[str, Any]:
    if spec["experiment"]["source_id"] != SOURCE_ID:
        raise RuntimeError("Source ID mismatch")
    if output_root.exists() and any(output_root.rglob("*")):
        raise RuntimeError(f"refusing to overwrite non-empty output: {output_root}")
    collector = load_module(LOCAL_COLLECTOR, "two_node_local_collector_runtime_helpers")
    matched = load_module(MATCHED_TELEMETRY, "two_node_matched_telemetry_helpers")
    matched_probe = load_module(matched.MATCHED_RUNNER, "two_node_matched_probe_helpers")
    legacy = load_module(matched_probe.LEGACY_RUNNER, "two_node_recovery_helpers")
    matched_spec = load_yaml(collector.MATCHED_PROBE_CONFIG)
    source_root = collector.resolve(spec["source_traces_root"])
    sources = [item for opponent in spec["protocol"]["opponents"] for item in collector.source_traces(source_root, opponent)]
    if len(sources) != int(spec["protocol"]["required_source_traces"]):
        raise RuntimeError(f"expected 12 source traces, found {len(sources)}")
    base = load_yaml(collector.resolve(spec["base_config"]))
    paths = {
        "h0": collector.resolve(spec["assets"]["h0_checkpoint"]),
        "crossing": collector.resolve(spec["assets"]["crossing_checkpoint"]),
        "crossing_config": collector.resolve(spec["assets"]["crossing_config"]),
        "crossing_predictor": collector.resolve(spec["assets"]["crossing_predictor"]),
        "end_to_end": collector.resolve(spec["assets"]["end_to_end_checkpoint"]),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    results: list[dict[str, Any]] = []
    traces: dict[tuple[str, str], list[dict[str, Any]]] = {}
    manifests: list[dict[str, Any]] = []
    max_steps = max(int(item["handoff_index"]) for item in sources) + max(OFFSETS) + 1
    for opponent in spec["protocol"]["opponents"]:
        own_sources = [item for item in sources if item["opponent"] == opponent]
        cfg = prepare_config(base, spec, matched_spec, matched_probe, opponent, max_steps)
        resolved = output_root / opponent / "resolved_config.yaml"
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        for source in own_sources:
            for condition in spec["protocol"]["conditions"]:
                result, trace = run_condition(source, condition, cfg, matched, legacy, matched_spec, paths)
                results.append(result)
                traces[(source["source_trace_id"], condition)] = trace
                name = source["source_trace_id"].replace("/", "__").replace(".csv", "")
                write_csv(output_root / opponent / "traces" / f"{name}__{condition}.csv", trace)
        manifests.append({"source_id": SOURCE_ID, "opponent": opponent, "source_trace_count": len(own_sources), "condition_count": len(own_sources) * 2, "resolved_config": receipt(resolved), "backend_requested": "jsbsim", "strict_backend": True})
    pairs = pair_rows(results, traces)
    write_csv(output_root / "condition_results.csv", results)
    write_csv(output_root / "paired_response_records.csv", pairs)
    manifest = {
        "source_id": SOURCE_ID, "status": "completed", "training_started": False, "training_permitted": False,
        "jsbsim_started": True, "source_trace_source_id": SOURCE_TRACE_SOURCE_ID, "source_trace_count": len(sources),
        "condition_count": len(results), "expected_condition_count": len(sources) * 2, "pair_count": len(sources),
        "opponents": list(spec["protocol"]["opponents"]), "conditions": list(spec["protocol"]["conditions"]),
        "node_steps_after_handoff": list(spec["protocol"]["node_steps_after_handoff"]),
        "node0_delta": spec["protocol"]["node0_delta"], "node1_delta": spec["protocol"]["node1_delta"],
        "response_offsets": list(OFFSETS), "base_config": receipt(collector.resolve(spec["base_config"])),
        "config": receipt(DEFAULT_CONFIG), "config_canonical_sha256": config_sha256(spec, truncate=0),
        "git_info": get_git_info(str(ROOT)), "run_manifests": manifests,
        "asset_receipts": {key: receipt(path) for key, path in paths.items()},
        "preregistered_gate": spec["preregistered_gate"],
        "contract": spec["protocol"],
    }
    write_json(output_root / "run_manifest.json", manifest)
    write_json(output_root / "probe_result.json", {**manifest, "decision": "pending_verifier", "paired_response_records": len(pairs)})
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--execute", action="store_true", help="Required to execute JSBSim")
    args = parser.parse_args()
    spec = load_yaml(args.config.resolve())
    if not args.execute:
        print(f"LOCKED: no JSBSim execution. Planned Source ID: {spec['experiment']['source_id']}")
        return 0
    manifest = run(spec, args.output_root.resolve())
    print(json.dumps({"source_id": manifest["source_id"], "pair_count": manifest["pair_count"], "output_root": str(args.output_root.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
