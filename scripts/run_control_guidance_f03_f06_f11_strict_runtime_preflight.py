#!/usr/bin/env python3
"""Run a read-only strict-JSBSim preflight for F03, F06, and F11.

This probe does not modify runtime code or YAML.  It compares one isolated
candidate at a time against a same-state legacy path:

* F03: PN LOS-rate filter alpha 0.3 versus 0.5;
* F06: 5 Hz synchronous updates versus 10/20/100 Hz multi-rate updates;
* F11: no probe-side slew limit versus a 2.0 normalized-command/s limit.

The result is an interface/plant response witness, not a combat-performance
evaluation and not an authorization to change protected defaults.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uav_vpp_guidance.envs.jsbsim_env import JSBSimEnv, neu2lla
from uav_vpp_guidance.flight_control.low_level_controller import LowLevelController
from uav_vpp_guidance.guidance.los_rate_guidance import LOSRateGuidance
from uav_vpp_guidance.guidance.proportional_navigation import ProportionalNavigationGuidance


SOURCE_ID = "THESIS-CONTROL-GUIDANCE-F03-F06-F11-STRICT-RUNTIME-PREFLIGHT-V1"
SIM_FREQ_HZ = 100
WINDOW_TICKS = 200
ORIGIN = (120.0, 60.0, 0.0)
DEFAULT_OUTPUT = "outputs/control_guidance_f03_f06_f11_strict_runtime_preflight_v1"

SCENARIOS = (
    {
        "scenario_id": "head_on_level",
        "signature": "head_on|level|2500m|mirror_0",
        "own_neu_m": [0.0, 0.0, 5000.0],
        "target_neu_m": [2500.0, 0.0, 5000.0],
        "own_heading_deg": 0.0,
        "target_heading_deg": 180.0,
    },
    {
        "scenario_id": "crossing_left",
        "signature": "crossing|left|2500m|mirror_0",
        "own_neu_m": [0.0, 0.0, 5000.0],
        "target_neu_m": [1200.0, 2200.0, 5000.0],
        "own_heading_deg": 0.0,
        "target_heading_deg": 225.0,
    },
    {
        "scenario_id": "crossing_right_elevated",
        "signature": "crossing|right|2500m|elevated_target",
        "own_neu_m": [0.0, 0.0, 5000.0],
        "target_neu_m": [1200.0, -2200.0, 5200.0],
        "own_heading_deg": 0.0,
        "target_heading_deg": 135.0,
    },
    {
        "scenario_id": "offset_closing",
        "signature": "offset_closing|right|2500m|vertical_offset",
        "own_neu_m": [0.0, 0.0, 5000.0],
        "target_neu_m": [2400.0, 500.0, 5600.0],
        "own_heading_deg": 0.0,
        "target_heading_deg": 190.0,
    },
)

PAIR_SPECS = (
    {
        "finding_id": "F03",
        "comparison_id": "pn_filter_alpha",
        "baseline_id": "pn_alpha_0_3",
        "candidate_id": "pn_alpha_0_5",
    },
    {
        "finding_id": "F06",
        "comparison_id": "cadence",
        "baseline_id": "single_rate_5hz",
        "candidate_id": "multirate_10_20_100hz",
    },
    {
        "finding_id": "F11",
        "comparison_id": "slew_limit",
        "baseline_id": "no_slew_limit",
        "candidate_id": "slew_2_norm_per_s",
    },
)


def finite(value: Any) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): jsonable(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def finite_tree(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def stable_hash(value: Any) -> str:
    payload = json.dumps(jsonable(value), ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_info() -> Dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
        return result.stdout.strip()

    return {"commit": run("rev-parse", "HEAD"), "branch": run("branch", "--show-current"), "dirty": bool(run("status", "--porcelain"))}


def load_guidance_config() -> Dict[str, Any]:
    import yaml

    payload = yaml.safe_load((ROOT / "config/guidance.yaml").read_text(encoding="utf-8")) or {}
    return copy.deepcopy(payload.get("guidance", payload))


def init_state(position_neu_m: List[float], heading_deg: float) -> Dict[str, float]:
    lon, lat, altitude_m = neu2lla(*position_neu_m, *ORIGIN)
    return {
        "ic/long-gc-deg": float(lon), "ic/lat-geod-deg": float(lat),
        "ic/h-sl-ft": float(altitude_m / 0.3048), "ic/psi-true-deg": float(heading_deg),
        "ic/u-fps": 800.0, "ic/v-fps": 0.0, "ic/w-fps": 0.0,
        "ic/p-rad_sec": 0.0, "ic/q-rad_sec": 0.0, "ic/r-rad_sec": 0.0,
        "ic/roc-fpm": 0.0, "ic/terrain-elevation-ft": 0.0,
    }


def state_anchor(state: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "position_neu": state.get("position_neu"), "velocity_vector_mps": state.get("velocity_vector_mps"),
        "attitude_rpy": state.get("attitude_rpy"), "body_rates_rps": state.get("body_rates_rps"),
        "speed_mps": state.get("speed_mps"), "altitude_m": state.get("altitude_m"), "sim_time": state.get("sim_time"),
    }


def geometry(own: Mapping[str, Any], target: Mapping[str, Any]) -> Dict[str, float]:
    position = np.asarray(target["position_neu"]) - np.asarray(own["position_neu"])
    velocity = np.asarray(target["velocity_vector_mps"]) - np.asarray(own["velocity_vector_mps"])
    distance = max(float(np.linalg.norm(position)), 1.0e-9)
    own_velocity = np.asarray(own["velocity_vector_mps"])
    target_velocity = np.asarray(target["velocity_vector_mps"])
    own_speed = max(float(np.linalg.norm(own_velocity)), 1.0e-9)
    target_speed = max(float(np.linalg.norm(target_velocity)), 1.0e-9)
    ata = np.clip(float(np.dot(own_velocity, position)) / (own_speed * distance), -1.0, 1.0)
    aa = np.clip(float(np.dot(target_velocity, -position)) / (target_speed * distance), -1.0, 1.0)
    return {
        "range_m": distance, "range_rate_mps": float(np.dot(position, velocity) / distance),
        "ata_deg": math.degrees(math.acos(ata)), "aa_deg": math.degrees(math.acos(aa)),
    }


def property_value(env: JSBSimEnv, uid: str, name: str) -> Optional[float]:
    try:
        value = env._aircraft[uid].get_property_value(name)
    except Exception:
        return None
    return finite(value)


def actuator_actual(env: JSBSimEnv) -> Dict[str, Optional[float]]:
    return {
        "elevator_pos_rad": property_value(env, "own", "fcs/elevator-pos-rad"),
        "aileron_pos_rad": property_value(env, "own", "fcs/aileron-pos-rad"),
        "left_aileron_pos_rad": property_value(env, "own", "fcs/left-aileron-pos-rad"),
        "right_aileron_pos_rad": property_value(env, "own", "fcs/right-aileron-pos-rad"),
        "throttle_pos_norm": property_value(env, "own", "fcs/throttle-pos-norm"),
    }


def mode_spec(mode_id: str, base_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    if mode_id == "pn_alpha_0_3":
        return {"law": "pn", "pn_alpha": 0.3, "high_ticks": 20, "guidance_ticks": 20, "low_ticks": 20, "slew_per_s": None}
    if mode_id == "pn_alpha_0_5":
        return {"law": "pn", "pn_alpha": 0.5, "high_ticks": 20, "guidance_ticks": 20, "low_ticks": 20, "slew_per_s": None}
    if mode_id == "single_rate_5hz":
        return {"law": "los", "high_ticks": 20, "guidance_ticks": 20, "low_ticks": 20, "slew_per_s": None}
    if mode_id == "multirate_10_20_100hz":
        return {"law": "los", "high_ticks": 10, "guidance_ticks": 5, "low_ticks": 1, "slew_per_s": None}
    if mode_id == "no_slew_limit":
        return {"law": "los", "high_ticks": 20, "guidance_ticks": 20, "low_ticks": 20, "slew_per_s": None}
    if mode_id == "slew_2_norm_per_s":
        return {"law": "los", "high_ticks": 20, "guidance_ticks": 20, "low_ticks": 20, "slew_per_s": 2.0}
    raise ValueError(f"unknown preflight mode: {mode_id}")


def apply_slew(command: Mapping[str, float], previous: Optional[Mapping[str, float]], slew_per_s: Optional[float], dt_s: float) -> tuple[Dict[str, float], bool]:
    current = {key: float(value) for key, value in command.items()}
    if previous is None or slew_per_s is None:
        return current, False
    step = float(slew_per_s) * dt_s
    limited, changed = {}, False
    for key, value in current.items():
        old = float(previous.get(key, value))
        clipped = float(np.clip(value, old - step, old + step))
        limited[key] = clipped
        changed = changed or abs(clipped - value) > 1.0e-12
    return limited, changed


def run_episode(scenario: Mapping[str, Any], mode_id: str, base_cfg: Mapping[str, Any], jsbsim_root: Path) -> Dict[str, Any]:
    spec = mode_spec(mode_id, base_cfg)
    cfg = copy.deepcopy(dict(base_cfg))
    if spec["law"] == "pn":
        cfg.setdefault("params", {})["los_rate_filter_alpha"] = spec["pn_alpha"]
        cfg["params"]["dt"] = spec["guidance_ticks"] / SIM_FREQ_HZ
        guidance = ProportionalNavigationGuidance(cfg)
    else:
        guidance = LOSRateGuidance(cfg)
    controller = LowLevelController({
        "alpha_filter": float(cfg.get("gains", {}).get("alpha_filter", 0.3)),
        "actuator": {"nz_to_elevator_gain": 1.0 / 7.0, "roll_rate_to_aileron_gain": 1.0 / 1.5},
    })
    env = JSBSimEnv({"sim_freq": SIM_FREQ_HZ, "legacy_project_root": str(jsbsim_root), "origin": ORIGIN})
    env.add_aircraft("own", {"model": "f16"})
    env.add_aircraft("target", {"model": "f16"})
    rows: List[Dict[str, Any]] = []
    held_fcs = {"fcs/elevator-cmd-norm": 0.0, "fcs/aileron-cmd-norm": 0.0, "fcs/rudder-cmd-norm": 0.0, "fcs/throttle-cmd-norm": 0.7}
    previous_fcs: Optional[Dict[str, float]] = None
    raw_command: Dict[str, float] = {"nz_cmd": 1.0, "roll_rate_cmd": 0.0, "throttle_cmd": 0.7}
    vpp_position: Optional[np.ndarray] = None
    initial_hash: Optional[str] = None
    reset_count = 0
    try:
        states = env.reset({"own": init_state(scenario["own_neu_m"], scenario["own_heading_deg"]), "target": init_state(scenario["target_neu_m"], scenario["target_heading_deg"])})
        reset_count = 1
        initial_hash = stable_hash({"own": state_anchor(states["own"]), "target": state_anchor(states["target"])})
        for tick in range(WINDOW_TICKS):
            pre = env.get_state()
            own, target = pre["own"], pre["target"]
            high_update = tick % spec["high_ticks"] == 0
            guidance_update = tick % spec["guidance_ticks"] == 0
            low_update = tick % spec["low_ticks"] == 0
            if high_update or vpp_position is None:
                vpp_position = np.asarray(target["position_neu"], dtype=float).copy()
            if guidance_update:
                raw_command = guidance.compute_command(own, target, {"position_neu": vpp_position})
            controller_record: Dict[str, Any] = {}
            rate_limited = False
            if low_update:
                controller_record = controller.compute_actuator(raw_command, own)
                command = {key: float(value) for key, value in controller_record.items() if key.startswith("fcs/")}
                held_fcs, rate_limited = apply_slew(command, previous_fcs, spec["slew_per_s"], 1.0 / SIM_FREQ_HZ)
                previous_fcs = dict(held_fcs)
            post = env.step({"own": held_fcs}, active_uids=["own", "target"])
            row = {
                "tick": tick, "time_s": finite(post["own"].get("sim_time")), "mode_id": mode_id,
                "high_level_update": high_update, "guidance_update": guidance_update, "low_level_update": low_update,
                "pre_action_state_hash": stable_hash({"own": state_anchor(own), "target": state_anchor(target)}),
                "target_state_hash": stable_hash(state_anchor(post["target"])),
                "raw_command": jsonable(raw_command), "fcs_command": jsonable(held_fcs),
                "actuator_actual": jsonable(actuator_actual(env)), "own_state": jsonable(post["own"]),
                "target_state": jsonable(post["target"]), "geometry": geometry(post["own"], post["target"]),
                "p_rps": finite(post["own"].get("p_rps")), "nz_g": finite(post["own"].get("nz_g")),
                "roll_rad": finite(post["own"].get("roll_rad")), "controller_saturated": bool(controller_record.get("saturation_flag", False)),
                "rate_limited": rate_limited,
            }
            if not finite_tree(row):
                return {"status": "fail", "error": "nonfinite_telemetry", "trajectory": rows, "reset_count": reset_count, "initial_state_hash": initial_hash, "mode_id": mode_id, "scenario_id": scenario["scenario_id"], "backend": "jsbsim", "strict_backend": True, "backend_fallback_occurred": False, "mid_episode_reset_count": 0}
            rows.append(row)
    except Exception as exc:
        return {"status": "fail", "error": repr(exc), "trajectory": rows, "reset_count": reset_count, "initial_state_hash": initial_hash, "mode_id": mode_id, "scenario_id": scenario["scenario_id"], "backend": "jsbsim", "strict_backend": True, "backend_fallback_occurred": False, "mid_episode_reset_count": 0}
    finally:
        env.close()
    return {
        "status": "pass", "scenario_id": scenario["scenario_id"], "scenario_signature": scenario["signature"],
        "mode_id": mode_id, "backend": "jsbsim", "strict_backend": True, "backend_fallback_occurred": False,
        "checkpoint_fallback_occurred": False, "reset_count": reset_count, "mid_episode_reset_count": 0,
        "initial_state_hash": initial_hash, "trajectory_steps": len(rows), "trajectory": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsbsim-root", type=Path, default=Path(os.environ.get("JSBSIM_ROOT", "")))
    parser.add_argument("--output-root", type=Path, default=Path(DEFAULT_OUTPUT))
    args = parser.parse_args()
    jsbsim_root = args.jsbsim_root.resolve()
    if not (jsbsim_root / "envs" / "JSBSim" / "data").is_dir():
        raise SystemExit("strict JSBSim data directory is missing; no evidence was generated")
    output = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    output.mkdir(parents=True, exist_ok=True)
    base_cfg = load_guidance_config()
    records: List[Dict[str, Any]] = []
    for pair_spec in PAIR_SPECS:
        for scenario in SCENARIOS:
            baseline = run_episode(scenario, pair_spec["baseline_id"], base_cfg, jsbsim_root)
            candidate = run_episode(scenario, pair_spec["candidate_id"], base_cfg, jsbsim_root)
            records.append({"pair_id": f"{pair_spec['comparison_id']}__{scenario['scenario_id']}", "source_id": SOURCE_ID, "finding_id": pair_spec["finding_id"], "comparison_id": pair_spec["comparison_id"], "baseline_id": pair_spec["baseline_id"], "candidate_id": pair_spec["candidate_id"], "scenario_id": scenario["scenario_id"], "baseline": baseline, "candidate": candidate})
    (output / "paired_records.json").write_text(json.dumps(jsonable(records), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "1.0.0", "source_id": SOURCE_ID,
        "status": "pass" if all(p["baseline"]["status"] == "pass" and p["candidate"]["status"] == "pass" for p in records) else "fail",
        "backend": "jsbsim", "strict_backend": True, "backend_fallback_occurred": False,
        "jsbsim_root": str(jsbsim_root), "sim_freq_hz": SIM_FREQ_HZ, "window_ticks": WINDOW_TICKS,
        "pair_count": len(records), "scenario_count": len(SCENARIOS), "findings": ["F03", "F06", "F11"],
        "pair_specs": list(PAIR_SPECS), "git": git_info(), "platform": platform.platform(), "python_version": platform.python_version(),
        "protected_sources": {
            "guidance_sha256": sha256_file(ROOT / "src/uav_vpp_guidance/guidance/proportional_navigation.py"),
            "low_level_sha256": sha256_file(ROOT / "src/uav_vpp_guidance/flight_control/low_level_controller.py"),
            "config_sha256": sha256_file(ROOT / "config/guidance.yaml"),
        },
        "notes": [
            "This is a non-learning plant/interface preflight; no YAML or protected runtime path was modified.",
            "A pass establishes telemetry continuity and measurable response only; it does not select a candidate.",
            "Frozen policy checkpoints and combat matrix evidence remain outside this preflight.",
        ],
    }
    (output / "run_manifest.json").write_text(json.dumps(jsonable(manifest), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"source_id": SOURCE_ID, "status": manifest["status"], "pair_count": len(records), "output_root": str(output)}, ensure_ascii=False))
    return 0 if manifest["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
