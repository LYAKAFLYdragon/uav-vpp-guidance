#!/usr/bin/env python3
"""Quick A/B test: Enhanced PID vs Gain-Scheduled PID."""
import sys, json, traceback
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

def run_episode(env, seed, max_steps=512):
    np.random.seed(seed)
    obs = env.reset(seed=seed)
    done = False
    step_count = 0
    nz_cmds, nz_actuals = [], []
    roll_cmds, roll_actuals = [], []
    betas, alphas, speeds = [], [], []
    while not done and step_count < max_steps:
        action = np.zeros(3, dtype=np.float32)
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
    result = {
        "success": info.get("termination_reason") == "success",
        "steps": step_count,
    }
    if nz_cmds and nz_actuals:
        nz_err = np.array(nz_cmds) - np.array(nz_actuals)
        result["nz_rmse"] = float(np.sqrt(np.mean(nz_err**2)))
        result["nz_mae"] = float(np.mean(np.abs(nz_err)))
    if roll_cmds and roll_actuals:
        roll_err = np.array(roll_cmds) - np.array(roll_actuals)
        result["roll_rmse"] = float(np.sqrt(np.mean(roll_err**2)))
        result["roll_mae"] = float(np.mean(np.abs(roll_err)))
    result["mean_beta_deg"] = float(np.degrees(np.mean(betas))) if betas else 0
    result["max_beta_deg"] = float(np.degrees(np.max(betas))) if betas else 0
    result["mean_alpha_deg"] = float(np.degrees(np.mean(alphas))) if alphas else 0
    result["max_alpha_deg"] = float(np.degrees(np.max(alphas))) if alphas else 0
    result["mean_speed_mps"] = float(np.mean(speeds)) if speeds else 0
    result["min_speed_mps"] = float(np.min(speeds)) if speeds else 0
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
    return {
        "n_episodes": len(results),
        "success_rate": len([r for r in results if r.get("success")]) / len(results),
        "mean_steps": _mean("steps"),
        "nz_rmse_mean": _mean("nz_rmse"), "nz_rmse_std": _std("nz_rmse"),
        "roll_rmse_mean": _mean("roll_rmse"), "roll_rmse_std": _std("roll_rmse"),
        "mean_beta_deg": _mean("mean_beta_deg"), "max_beta_deg": _mean("max_beta_deg"),
        "mean_alpha_deg": _mean("mean_alpha_deg"), "max_alpha_deg": _mean("max_alpha_deg"),
        "mean_speed_mps": _mean("mean_speed_mps"), "min_speed_mps": _mean("min_speed_mps"),
    }

def test_controller(name, config, seeds, max_steps=512):
    print(f"\n[{name}] Running {len(seeds)} episodes...")
    env = CloseRangeTrackingEnv(config)
    results = []
    for i, seed in enumerate(seeds):
        try:
            result = run_episode(env, seed, max_steps)
            results.append(result)
            if (i + 1) % 5 == 0 or len(seeds) <= 10:
                print(f"  {i+1}/{len(seeds)}: success={result['success']}, steps={result['steps']}, "
                      f"nz_rmse={result.get('nz_rmse',0):.2f}, roll_rmse={result.get('roll_rmse',0):.2f}")
        except Exception as exc:
            print(f"  Episode {i+1} failed: {exc}")
            traceback.print_exc()
    return aggregate(results), results

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--output", type=str, default="results/ab_test_gain_scheduled.json")
    args = parser.parse_args()

    seeds = list(range(args.seed, args.seed + args.n_episodes))
    print(f"\n{'='*70}")
    print(f"A/B Test: Enhanced PID vs Gain-Scheduled PID")
    print(f"Episodes: {args.n_episodes}, Seeds: {seeds[0]}..{seeds[-1]}")
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

    print(f"\n{'='*70}")
    print("RECOMMENDATION")
    print(f"{'='*70}")

    nz_delta = comparison.get("nz_rmse_mean", {}).get("delta_pct", 0)
    roll_delta = comparison.get("roll_rmse_mean", {}).get("delta_pct", 0)
    beta_delta = comparison.get("mean_beta_deg", {}).get("delta_pct", 0)

    if nz_delta < -5 or roll_delta < -5 or beta_delta < -10:
        print("[PASS] GAIN-SCHEDULED controller shows SIGNIFICANT improvement.")
        print("   Recommendation: ADOPT gain_scheduled controller.")
    elif abs(nz_delta) < 3 and abs(roll_delta) < 3 and abs(beta_delta) < 5:
        print("[WARN] No significant difference between controllers.")
        print("   Recommendation: Keep Enhanced PID (simpler, less risk).")
    else:
        print("[INFO] GAIN-SCHEDULED shows moderate differences.")
        print("   Recommendation: ADOPT with further tuning.")

    output_data = {
        "config": {"n_episodes": args.n_episodes, "seed_start": args.seed, "max_steps": args.max_steps},
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
