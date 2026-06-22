#!/usr/bin/env python3
"""
High-dynamic A/B test: Enhanced PID vs Gain-Scheduled PID.

Generates dynamic flight scenarios using:
- Random action sequences (per-step noise)
- Preset aggressive maneuver patterns (sine, sweep, bang-bang)
- Records dynamic metrics: speed range, max nz/roll/pitch, climb/descent rates.

Usage:
    python scripts/ab_test_dynamic.py --n-episodes 30 --output results/ab_test_dynamic.json
"""
import sys, json, traceback, math
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.utils.config import merge_config
import yaml


def load_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def build_config(controller: str):
    config = {}
    for name in ("guidance", "reward", "virtual_point", "gain_space"):
        config = merge_config(config, load_yaml(PROJECT_ROOT / "config" / "canonical" / f"{name}.yaml"))
    config = merge_config(config, load_yaml(PROJECT_ROOT / "config" / "env.yaml"))
    config = merge_config(config, load_yaml(PROJECT_ROOT / "config" / "success_criteria" / "medium.yaml"))
    config["backend"] = "jsbsim"
    config["env"]["backend"] = "jsbsim"
    config["env"]["use_jsbsim"] = True
    config["env"]["jsbsim_data_dir"] = str(PROJECT_ROOT / "data" / "jsbsim")
    config.setdefault("guidance", {})
    config["guidance"].setdefault("mode_switch", {})
    config["guidance"]["mode_switch"]["enabled"] = False
    config["low_level_controller"] = controller
    if controller in ("enhanced", "gain_scheduled"):
        config["guidance"]["limits"]["nz_max"] = 9.0
        config["guidance"]["limits"]["roll_rate_max"] = 3.0
        config["guidance"]["limits"]["roll_rate_min"] = -3.0
    return config


# ---------------------------------------------------------------------------
# Action generators for high-dynamic scenarios
# ---------------------------------------------------------------------------

def generate_action_random(t_step: int, seed_offset: int, scale: float = 1.0):
    """Per-step random action with smoothing."""
    np.random.seed(seed_offset + t_step)
    raw = np.random.randn(3) * scale
    return np.clip(raw, -2.0, 2.0).astype(np.float32)


def generate_action_sine_sweep(t_step: int, dt: float, freq: float = 0.5):
    """Sinusoidal sweep action for sustained oscillation."""
    t = t_step * dt
    a0 = 1.5 * math.sin(2 * math.pi * freq * t)
    a1 = 1.2 * math.cos(2 * math.pi * freq * 0.7 * t)
    a2 = 0.5 * math.sin(2 * math.pi * freq * 1.3 * t)
    return np.array([a0, a1, a2], dtype=np.float32)


def generate_action_bang_bang(t_step: int, period: int = 40):
    """Bang-bang action: full deflection alternating."""
    phase = (t_step // period) % 4
    if phase == 0:
        return np.array([1.5, 0.0, 0.0], dtype=np.float32)
    elif phase == 1:
        return np.array([0.0, 1.5, 0.0], dtype=np.float32)
    elif phase == 2:
        return np.array([-1.5, 0.0, 0.0], dtype=np.float32)
    else:
        return np.array([0.0, -1.5, 0.0], dtype=np.float32)


def select_action_generator(episode_idx: int, total_episodes: int):
    """Rotate between action patterns for diversity."""
    patterns = ["random", "sine", "bang_bang", "random_large", "sine_fast"]
    pattern = patterns[episode_idx % len(patterns)]
    if pattern == "random":
        return lambda t, s: generate_action_random(t, s, scale=0.8)
    elif pattern == "random_large":
        return lambda t, s: generate_action_random(t, s, scale=1.5)
    elif pattern == "sine":
        return lambda t, s: generate_action_sine_sweep(t, 0.2, freq=0.3)
    elif pattern == "sine_fast":
        return lambda t, s: generate_action_sine_sweep(t, 0.2, freq=0.8)
    elif pattern == "bang_bang":
        return lambda t, s: generate_action_bang_bang(t, period=30)
    return lambda t, s: generate_action_random(t, s, scale=0.8)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(env, seed, max_steps=512, action_gen=None):
    np.random.seed(seed)
    obs = env.reset(seed=seed)
    done = False
    step_count = 0

    # Metrics containers
    nz_cmds, nz_actuals = [], []
    roll_cmds, roll_actuals = [], []
    betas, alphas, speeds, altitudes = [], [], [], []
    pitches, rolls, vds = [], [], []
    elevator_cmds, aileron_cmds = [], []
    scheduled_gain_log = []

    while not done and step_count < max_steps:
        if action_gen is None:
            action = np.zeros(3, dtype=np.float32)
        else:
            action = action_gen(step_count, seed)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1

        own = info.get("own_state", {})
        if own:
            nz_cmds.append(info.get("guidance_command", {}).get("nz_cmd", 0))
            nz_actuals.append(own.get("nz_g", 0))
            roll_cmds.append(info.get("guidance_command", {}).get("roll_rate_cmd", 0))
            roll_actuals.append(own.get("p_rps", 0))
            betas.append(abs(own.get("beta_rad", 0)))
            alphas.append(abs(own.get("alpha_rad", 0)))
            speeds.append(own.get("speed_mps", 0))
            altitudes.append(own.get("altitude_m", 0))
            pitches.append(own.get("pitch_rad", 0))
            rolls.append(own.get("roll_rad", 0))
            vds.append(own.get("vd_mps", 0))

            # Try to extract actuator commands from info
            if "actuator" in info:
                act = info["actuator"]
                elevator_cmds.append(abs(act.get("elevator_raw", 0)))
                aileron_cmds.append(abs(act.get("aileron_raw", 0)))

            # Try to extract scheduled gains from info
            if "scheduled_gains" in info:
                scheduled_gain_log.append(info["scheduled_gains"])

    result = {
        "success": info.get("termination_reason") == "success",
        "steps": step_count,
    }

    if nz_cmds and nz_actuals:
        nz_err = np.array(nz_cmds) - np.array(nz_actuals)
        result["nz_rmse"] = float(np.sqrt(np.mean(nz_err**2)))
        result["nz_mae"] = float(np.mean(np.abs(nz_err)))
        result["nz_max"] = float(np.max(np.abs(nz_actuals)))
        result["nz_min"] = float(np.min(nz_actuals))

    if roll_cmds and roll_actuals:
        roll_err = np.array(roll_cmds) - np.array(roll_actuals)
        result["roll_rmse"] = float(np.sqrt(np.mean(roll_err**2)))
        result["roll_mae"] = float(np.mean(np.abs(roll_err)))
        result["max_roll_rate"] = float(np.max(np.abs(roll_actuals)))

    result["mean_beta_deg"] = float(np.degrees(np.mean(betas))) if betas else 0
    result["max_beta_deg"] = float(np.degrees(np.max(betas))) if betas else 0
    result["mean_alpha_deg"] = float(np.degrees(np.mean(alphas))) if alphas else 0
    result["max_alpha_deg"] = float(np.degrees(np.max(alphas))) if alphas else 0
    result["mean_speed_mps"] = float(np.mean(speeds)) if speeds else 0
    result["min_speed_mps"] = float(np.min(speeds)) if speeds else 0
    result["max_speed_mps"] = float(np.max(speeds)) if speeds else 0
    result["speed_range_mps"] = result["max_speed_mps"] - result["min_speed_mps"]
    result["mean_altitude_m"] = float(np.mean(altitudes)) if altitudes else 0
    result["min_altitude_m"] = float(np.min(altitudes)) if altitudes else 0
    result["max_altitude_m"] = float(np.max(altitudes)) if altitudes else 0
    result["altitude_range_m"] = result["max_altitude_m"] - result["min_altitude_m"]
    result["max_pitch_deg"] = float(np.degrees(np.max(np.abs(pitches)))) if pitches else 0
    result["max_roll_deg"] = float(np.degrees(np.max(np.abs(rolls)))) if rolls else 0
    result["max_vd_mps"] = float(np.max(vds)) if vds else 0
    result["min_vd_mps"] = float(np.min(vds)) if vds else 0

    if elevator_cmds:
        result["mean_elevator"] = float(np.mean(elevator_cmds))
        result["max_elevator"] = float(np.max(elevator_cmds))
    if aileron_cmds:
        result["mean_aileron"] = float(np.mean(aileron_cmds))
        result["max_aileron"] = float(np.max(aileron_cmds))

    if scheduled_gain_log:
        result["mean_scheduled_Kp_nz"] = float(np.mean([g["Kp_nz"] for g in scheduled_gain_log]))
        result["mean_scheduled_Kp_roll"] = float(np.mean([g["Kp_roll"] for g in scheduled_gain_log]))

    return result


def aggregate(results):
    if not results:
        return {}

    def _mean(key):
        vals = [r[key] for r in results if key in r]
        return float(np.mean(vals)) if vals else 0.0

    def _std(key):
        vals = [r[key] for r in results if key in r]
        return float(np.std(vals)) if vals else 0.0

    def _max(key):
        vals = [r[key] for r in results if key in r]
        return float(np.max(vals)) if vals else 0.0

    agg = {
        "n_episodes": len(results),
        "success_rate": len([r for r in results if r.get("success")]) / len(results),
        "mean_steps": _mean("steps"),
        "nz_rmse_mean": _mean("nz_rmse"), "nz_rmse_std": _std("nz_rmse"),
        "roll_rmse_mean": _mean("roll_rmse"), "roll_rmse_std": _std("roll_rmse"),
        "mean_beta_deg": _mean("mean_beta_deg"), "max_beta_deg": _mean("max_beta_deg"),
        "mean_alpha_deg": _mean("mean_alpha_deg"), "max_alpha_deg": _mean("max_alpha_deg"),
        "mean_speed_mps": _mean("mean_speed_mps"),
        "min_speed_mps": _mean("min_speed_mps"), "max_speed_mps": _mean("max_speed_mps"),
        "speed_range_mps": _mean("speed_range_mps"),
        "mean_altitude_m": _mean("mean_altitude_m"),
        "altitude_range_m": _mean("altitude_range_m"),
        "max_roll_deg": _mean("max_roll_deg"),
        "max_pitch_deg": _mean("max_pitch_deg"),
        "max_vd_mps": _mean("max_vd_mps"),
        "min_vd_mps": _mean("min_vd_mps"),
    }
    # Only include scheduled gain metrics if available
    if any("mean_scheduled_Kp_nz" in r for r in results):
        agg["mean_scheduled_Kp_nz"] = _mean("mean_scheduled_Kp_nz")
        agg["mean_scheduled_Kp_roll"] = _mean("mean_scheduled_Kp_roll")
    return agg


def test_controller(name, config, seeds, max_steps=512):
    print(f"\n[{name}] Running {len(seeds)} high-dynamic episodes...")
    env = CloseRangeTrackingEnv(config)
    results = []
    for i, seed in enumerate(seeds):
        action_gen = select_action_generator(i, len(seeds))
        try:
            result = run_episode(env, seed, max_steps, action_gen)
            results.append(result)
            if (i + 1) % 5 == 0 or len(seeds) <= 10:
                print(f"  {i+1}/{len(seeds)}: steps={result['steps']}, "
                      f"nz_rmse={result.get('nz_rmse',0):.2f}, "
                      f"roll_rmse={result.get('roll_rmse',0):.2f}, "
                      f"speed_range={result.get('speed_range_mps',0):.1f}, "
                      f"max_roll={result.get('max_roll_deg',0):.1f}°")
        except Exception as exc:
            print(f"  Episode {i+1} failed: {exc}")
            traceback.print_exc()
    return aggregate(results), results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--output", type=str, default="results/ab_test_dynamic.json")
    args = parser.parse_args()

    seeds = list(range(args.seed, args.seed + args.n_episodes))
    print(f"\n{'='*70}")
    print(f"HIGH-DYNAMIC A/B TEST: Enhanced PID vs Gain-Scheduled PID")
    print(f"Episodes: {args.n_episodes}, Seeds: {seeds[0]}..{seeds[-1]}")
    print(f"Action patterns: random, sine_sweep, bang_bang, random_large, sine_fast")
    print(f"{'='*70}")

    enhanced_config = build_config("enhanced")
    gs_config = build_config("gain_scheduled")

    enhanced_agg, enhanced_raw = test_controller("ENHANCED", enhanced_config, seeds, args.max_steps)
    gs_agg, gs_raw = test_controller("GAIN_SCHEDULED", gs_config, seeds, args.max_steps)

    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Metric':<30} {'Enhanced':>15} {'GainScheduled':>15} {'Delta':>15}")
    print("-" * 80)

    metrics = [
        ("Success Rate", "success_rate", ".1%"),
        ("Mean Steps", "mean_steps", ".1f"),
        ("NZ RMSE (g)", "nz_rmse_mean", ".3f"),
        ("NZ RMSE std", "nz_rmse_std", ".3f"),
        ("Roll Rate RMSE (rad/s)", "roll_rmse_mean", ".3f"),
        ("Roll Rate RMSE std", "roll_rmse_std", ".3f"),
        ("Mean Beta (deg)", "mean_beta_deg", ".2f"),
        ("Max Beta (deg)", "max_beta_deg", ".2f"),
        ("Mean Alpha (deg)", "mean_alpha_deg", ".2f"),
        ("Max Alpha (deg)", "max_alpha_deg", ".2f"),
        ("Mean Speed (m/s)", "mean_speed_mps", ".1f"),
        ("Min Speed (m/s)", "min_speed_mps", ".1f"),
        ("Max Speed (m/s)", "max_speed_mps", ".1f"),
        ("Speed Range (m/s)", "speed_range_mps", ".1f"),
        ("Mean Altitude (m)", "mean_altitude_m", ".1f"),
        ("Altitude Range (m)", "altitude_range_m", ".1f"),
        ("Max Roll (deg)", "max_roll_deg", ".1f"),
        ("Max Pitch (deg)", "max_pitch_deg", ".1f"),
    ]

    comparison = {}
    for label, key, fmt_str in metrics:
        b = enhanced_agg.get(key, 0.0)
        e = gs_agg.get(key, 0.0)
        if b != 0:
            delta = (e - b) / abs(b) * 100
        else:
            delta = 0.0 if e == 0 else float('inf')
        print(f"{label:<30} {b:>15{fmt_str.replace('.','')}} {e:>15{fmt_str.replace('.','')}} {delta:>+14.1f}%")
        comparison[key] = {"enhanced": b, "gain_scheduled": e, "delta_pct": delta}

    # Scheduled gain metrics (only for gain_scheduled)
    if "mean_scheduled_Kp_nz" in gs_agg:
        print(f"\n{'--- Scheduled Gain Diagnostics ---':^80}")
        print(f"{'Mean Kp_nz (scheduled)':<30} {'---':>15} {gs_agg['mean_scheduled_Kp_nz']:>15.3f} {'---':>15}")
        print(f"{'Mean Kp_roll (scheduled)':<30} {'---':>15} {gs_agg['mean_scheduled_Kp_roll']:>15.3f} {'---':>15}")

    print(f"\n{'='*70}")
    print("RECOMMENDATION")
    print(f"{'='*70}")

    nz_delta = comparison.get("nz_rmse_mean", {}).get("delta_pct", 0)
    roll_delta = comparison.get("roll_rmse_mean", {}).get("delta_pct", 0)
    beta_delta = comparison.get("mean_beta_deg", {}).get("delta_pct", 0)
    alpha_delta = comparison.get("mean_alpha_deg", {}).get("delta_pct", 0)
    speed_range_delta = comparison.get("speed_range_mps", {}).get("delta_pct", 0)

    if nz_delta < -5 and roll_delta < -5:
        print("[PASS] GAIN-SCHEDULED controller shows SIGNIFICANT improvement in tracking.")
        print("   Recommendation: ADOPT gain_scheduled controller.")
    elif abs(nz_delta) < 5 and abs(roll_delta) < 5 and abs(beta_delta) < 10 and abs(alpha_delta) < 10:
        print("[WARN] No significant difference between controllers.")
        print("   Recommendation: Keep Enhanced PID (simpler, less risk).")
    else:
        print("[INFO] GAIN-SCHEDULED shows moderate differences.")
        if beta_delta < -10 or alpha_delta < -10:
            print("   Better flight quality (alpha/beta reduced).")
        if nz_delta > 5:
            print("   WARNING: NZ tracking precision degraded.")
        if speed_range_delta > 10:
            print("   WARNING: Larger speed variations detected.")
        print("   Recommendation: ADOPT with further tuning OR keep Enhanced.")

    output_data = {
        "config": {
            "n_episodes": args.n_episodes,
            "seed_start": args.seed,
            "max_steps": args.max_steps,
            "test_type": "high_dynamic",
            "action_patterns": ["random", "sine_sweep", "bang_bang", "random_large", "sine_fast"],
        },
        "enhanced": {"aggregate": enhanced_agg, "episodes": enhanced_raw},
        "gain_scheduled": {"aggregate": gs_agg, "episodes": gs_raw},
        "comparison": comparison,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output_data, indent=2, default=str), encoding="utf-8")
    print(f"\n[SAVE] Results saved to: {out_path}")


if __name__ == "__main__":
    main()
