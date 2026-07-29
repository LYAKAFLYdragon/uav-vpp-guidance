#!/usr/bin/env python3
"""Run a non-learning strict-JSBSim legacy versus opt-in paired regression.

The candidate is the explicit ``params.remediation_mode`` runtime mode.  Each
scenario is reset independently for the legacy and candidate episodes, while
the target follows the same deterministic JSBSim trajectory.  This is a
physical response regression, not a combat-performance or training run.
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

from uav_vpp_guidance.common.provenance import record_config_override
from uav_vpp_guidance.envs.jsbsim_env import JSBSimEnv, neu2lla
from uav_vpp_guidance.flight_control.low_level_controller import LowLevelController
from uav_vpp_guidance.guidance.los_rate_guidance import LOSRateGuidance


SOURCE_ID = "THESIS-CONTROL-GUIDANCE-F01-F02-OPTIN-PAIRED-REGRESSION-V1"
WINDOW_STEPS = 48
SIM_FREQ_HZ = 60
ORIGIN = (120.0, 60.0, 0.0)
DEFAULT_OUTPUT = "outputs/control_guidance_f01_f02_optin_paired_regression_v1"
# The runtime exposes one approved opt-in mode containing both remediations.
# Do not label repeated executions as component ablations.
VARIANTS = ("f01_f02",)

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


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        jsonable(value), ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
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
        completed = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace", check=False,
        )
        return completed.stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


def load_guidance_config() -> Dict[str, Any]:
    import yaml

    payload = yaml.safe_load((ROOT / "config/guidance.yaml").read_text(encoding="utf-8")) or {}
    return copy.deepcopy(payload.get("guidance", payload))


def mode_config(base: Mapping[str, Any], mode: str) -> Dict[str, Any]:
    config = copy.deepcopy(dict(base))
    params = config.setdefault("params", {})
    old = params.get("remediation_mode", "legacy")
    record_config_override(
        config, "params.remediation_mode", mode, old,
        source="run_control_guidance_f01_f02_optin_paired_regression.py",
    )
    params["remediation_mode"] = mode
    return config


def init_state(position_neu_m: List[float], heading_deg: float) -> Dict[str, float]:
    lon, lat, altitude_m = neu2lla(*position_neu_m, *ORIGIN)
    return {
        "ic/long-gc-deg": float(lon),
        "ic/lat-geod-deg": float(lat),
        "ic/h-sl-ft": float(altitude_m / 0.3048),
        "ic/psi-true-deg": float(heading_deg),
        "ic/u-fps": 800.0,
        "ic/v-fps": 0.0,
        "ic/w-fps": 0.0,
        "ic/p-rad_sec": 0.0,
        "ic/q-rad_sec": 0.0,
        "ic/r-rad_sec": 0.0,
        "ic/roc-fpm": 0.0,
        "ic/terrain-elevation-ft": 0.0,
    }


def state_anchor(state: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "position_neu": state.get("position_neu"),
        "velocity_vector_mps": state.get("velocity_vector_mps"),
        "attitude_rpy": state.get("attitude_rpy"),
        "body_rates_rps": state.get("body_rates_rps"),
        "speed_mps": state.get("speed_mps"),
        "altitude_m": state.get("altitude_m"),
        "sim_time": state.get("sim_time"),
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
        "range_m": distance,
        "range_rate_mps": float(np.dot(position, velocity) / distance),
        "closing_speed_mps": float(-np.dot(position, velocity) / distance),
        "ata_deg": math.degrees(math.acos(ata)),
        "aa_deg": math.degrees(math.acos(aa)),
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
        "rudder_pos_rad": property_value(env, "own", "fcs/rudder-pos-rad"),
        "throttle_pos_norm": property_value(env, "own", "fcs/throttle-pos-norm"),
    }


def finite_tree(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def run_episode(scenario: Mapping[str, Any], mode: str, variant: str,
                guidance_cfg: Mapping[str, Any], jsbsim_root: Path) -> Dict[str, Any]:
    env = JSBSimEnv({"sim_freq": SIM_FREQ_HZ, "legacy_project_root": str(jsbsim_root), "origin": ORIGIN})
    env.add_aircraft("own", {"model": "f16"})
    env.add_aircraft("target", {"model": "f16"})
    guidance = LOSRateGuidance(dict(guidance_cfg))
    controller = LowLevelController({
        "alpha_filter": float(guidance_cfg.get("gains", {}).get("alpha_filter", 0.3)),
        "actuator": {"nz_to_elevator_gain": 1.0 / 7.0, "roll_rate_to_aileron_gain": 1.0 / 1.5},
    })
    trajectory: List[Dict[str, Any]] = []
    reset_count = 0
    initial_hash: Optional[str] = None
    terminal_reason = "window_complete"
    try:
        states = env.reset({
            "own": init_state(scenario["own_neu_m"], scenario["own_heading_deg"]),
            "target": init_state(scenario["target_neu_m"], scenario["target_heading_deg"]),
        })
        reset_count = 1
        initial_hash = stable_hash({"own": state_anchor(states["own"]), "target": state_anchor(states["target"])})
        for step_index in range(WINDOW_STEPS):
            pre = env.get_state()
            own, target = pre["own"], pre["target"]
            raw = guidance.compute_command(own, target, {"position_neu": target["position_neu"]})
            actuators = controller.compute_actuator(raw, own)
            fcs = {key: float(value) for key, value in actuators.items() if key.startswith("fcs/")}
            post = env.step({"own": fcs}, active_uids=["own", "target"])
            row = {
                "step": step_index,
                "time_s": finite(post["own"].get("sim_time")),
                "pre_action_state_hash": stable_hash({"own": state_anchor(own), "target": state_anchor(target)}),
                "target_state_hash": stable_hash(state_anchor(post["target"])),
                "mode": mode,
                "variant": variant,
                "raw_command": jsonable(raw),
                "filtered_command": jsonable(actuators.get("filtered_command", {})),
                "fcs_command": jsonable(fcs),
                "actuator_actual": jsonable(actuator_actual(env)),
                "own_state": jsonable(post["own"]),
                "target_state": jsonable(post["target"]),
                "geometry": geometry(post["own"], post["target"]),
                "p_rps": finite(post["own"].get("p_rps")),
                "nz_g": finite(post["own"].get("nz_g")),
                "alpha_deg": finite(post["own"].get("alpha_deg")),
                "roll_rad": finite(post["own"].get("roll_rad")),
                "clipping": bool(actuators.get("saturation_flag", False)),
            }
            if not finite_tree(row):
                terminal_reason = "nonfinite_telemetry"
                break
            trajectory.append(row)
    except Exception as exc:
        terminal_reason = "jsbsim_exception"
        return {
            "status": "fail", "scenario_id": scenario["scenario_id"], "mode": mode,
            "variant": variant, "backend": "jsbsim", "strict_backend": True,
            "backend_fallback_occurred": False, "reset_count": reset_count,
            "mid_episode_reset_count": 0, "initial_state_hash": initial_hash,
            "trajectory": trajectory, "terminal_reason": terminal_reason,
            "error": repr(exc),
        }
    finally:
        env.close()
    return {
        "status": "pass" if len(trajectory) == WINDOW_STEPS else "fail",
        "scenario_id": scenario["scenario_id"], "scenario_signature": scenario["signature"],
        "mode": mode, "variant": variant, "backend": "jsbsim", "strict_backend": True,
        "backend_fallback_occurred": False, "checkpoint_fallback_occurred": False,
        "reset_count": reset_count, "mid_episode_reset_count": 0,
        "initial_state_hash": initial_hash, "trajectory_steps": len(trajectory),
        "terminal_reason": terminal_reason, "trajectory": trajectory,
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
    legacy_cfg = mode_config(base_cfg, "legacy")
    candidate_cfg = mode_config(base_cfg, "f01_f02_remediated")
    records: List[Dict[str, Any]] = []
    for scenario in SCENARIOS:
        baseline = run_episode(scenario, "legacy", "baseline", legacy_cfg, jsbsim_root)
        for variant in VARIANTS:
            candidate = run_episode(scenario, "f01_f02_remediated", variant, candidate_cfg, jsbsim_root)
            records.append({
                "pair_id": f"{scenario['scenario_id']}__{variant}",
                "source_id": SOURCE_ID, "scenario_id": scenario["scenario_id"], "variant": variant,
                "baseline": baseline, "candidate": candidate,
            })
    (output / "paired_records.json").write_text(json.dumps(jsonable(records), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "1.0.0", "source_id": SOURCE_ID,
        "status": "pass" if all(p["baseline"]["status"] == "pass" and p["candidate"]["status"] == "pass" for p in records) else "fail",
        "backend": "jsbsim", "strict_backend": True, "backend_fallback_occurred": False,
        "jsbsim_root": str(jsbsim_root), "window_steps": WINDOW_STEPS, "sim_freq_hz": SIM_FREQ_HZ,
        "scenario_count": len(SCENARIOS), "pair_count": len(records), "variants": list(VARIANTS),
        "baseline_mode": "legacy", "candidate_mode": "f01_f02_remediated",
        "config": {
            "guidance_source": "config/guidance.yaml",
            "guidance_sha256": sha256_file(ROOT / "config/guidance.yaml"),
            "legacy_overrides": legacy_cfg.get("provenance", {}).get("config_overrides", []),
            "candidate_overrides": candidate_cfg.get("provenance", {}).get("config_overrides", []),
        },
        "protected_sources": {
            "guidance_sha256": sha256_file(ROOT / "src/uav_vpp_guidance/guidance/los_rate_guidance.py"),
            "controller_sha256": sha256_file(ROOT / "src/uav_vpp_guidance/flight_control/low_level_controller.py"),
        },
        "git": git_info(), "platform": platform.platform(), "python_version": platform.python_version(),
        "scenarios": list(SCENARIOS), "paired_records": "paired_records.json",
        "notes": [
            "The candidate is explicit opt-in only; legacy remains the default.",
            "This is a strict-JSBSim same-state physical response regression, not a performance claim.",
            "No training, reward change, checkpoint replacement, or runtime parameter tuning was performed.",
        ],
    }
    (output / "run_manifest.json").write_text(json.dumps(jsonable(manifest), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"source_id": SOURCE_ID, "status": manifest["status"], "pair_count": len(records), "output_root": str(output)}, ensure_ascii=False))
    return 0 if manifest["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
