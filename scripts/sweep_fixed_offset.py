"""Sweep fixed VPP offsets on the JSBSim disadvantage scenario."""
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, "src")


def run_one(args):
    config, checkpoint, offset, label, n_eps = args
    off_str = ",".join(str(float(x)) for x in offset)
    out = f"outputs/sweep_fixed_offset/{label}_{off_str.replace(',', '_')}.json"
    cmd = [
        "python", "scripts/eval_checkpoint_scenarios.py",
        "--config", config,
        "--checkpoint", checkpoint,
        "--scenarios", "disadvantage",
        "--episodes", str(n_eps),
        "--seed-base", "5000",
        f"--fixed-offset={off_str}",
        "--output", out,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        data = json.load(open(out))
        return {"label": label, "offset": off_str, "result": data[0]}
    except Exception as e:
        return {"label": label, "offset": off_str, "error": str(e)}


def sweep(label, config, checkpoint, offsets, n_eps=10, max_workers=8):
    os.makedirs("outputs/sweep_fixed_offset", exist_ok=True)
    tasks = [(config, checkpoint, off, label, n_eps) for off in offsets]
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for fut in as_completed([ex.submit(run_one, t) for t in tasks]):
            results.append(fut.result())
    return results


def main():
    # VPP seed 0 checkpoint (existing)
    vpp_cfg = "config/experiment/train_no_prediction_vpp_ppo_jsbsim_maneuver.yaml"
    vpp_ckpt = "outputs/experiments/no_prediction_vpp_ppo_jsbsim_maneuver/checkpoints/best.pt"
    # No-VPP checkpoint (existing)
    novpp_cfg = "config/experiment/train_no_vpp_ppo_jsbsim_maneuver.yaml"
    novpp_ckpt = "outputs/experiments/no_vpp_ppo_jsbsim_maneuver/checkpoints/best.pt"

    offsets = []
    for dx in [-1500, -1000, -500, 0, 500, 1000, 1500]:
        for dy in [-500, 0, 500]:
            offsets.append([dx, dy, 0])

    print(f"Sweeping {len(offsets)} fixed offsets for VPP ...")
    vpp_results = sweep("vpp", vpp_cfg, vpp_ckpt, offsets, n_eps=10)
    print(f"Sweeping {len(offsets)} fixed offsets for No-VPP ...")
    novpp_results = sweep("novpp", novpp_cfg, novpp_ckpt, offsets, n_eps=10)

    summary = {"vpp": vpp_results, "novpp": novpp_results}
    out = "outputs/sweep_fixed_offset/summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {out}")

    # Print best offsets
    for label, results in [("VPP", vpp_results), ("No-VPP", novpp_results)]:
        valid = [r for r in results if "error" not in r]
        best = max(valid, key=lambda r: r["result"]["success_rate"])
        print(
            f"[{label}] best offset={best['offset']} "
            f"SR={best['result']['success_rate']:.2f} "
            f"(success={best['result']['success']}/{best['result']['episodes']}, "
            f"crash={best['result']['crash']}, timeout={best['result']['timeout']})"
        )


if __name__ == "__main__":
    main()
