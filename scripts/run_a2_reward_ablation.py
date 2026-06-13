#!/usr/bin/env python3
"""
A2' Reward-Design Ablation (Phase 2 reward audit).

Compares three reward designs, all built ONLY from existing canonical reward
parameters in ``config/canonical/reward.yaml`` (no new defaults are introduced):

  - ``dense_only``    : canonical dense mixture + terminal sparse events, PBS off.
                        This is the Baseline (standard PPO + canonical config).
  - ``dense_pbs``     : dense_only + potential-based shaping enabled
                        (toggles the existing ``potential_based_shaping.enabled``).
  - ``terminal_only`` : all existing dense weights set to 0.0, terminal sparse
                        events kept. PBS off.

The runnable scaffold (env / scenarios / ppo / policy / curriculum /
actuator_dynamics) is inherited verbatim from the existing committed config
``config/method_innovation_comparison.yaml`` (simple backend). The reward,
guidance and virtual_point blocks are overridden by ``config/canonical/``.

Each condition is trained with standard PPO on the SIMPLE backend for >= 3 seeds.
Results (per-seed eval logs, success-rate comparison table, statistical
significance) are written to ``outputs/ablation_reward_design/``.

Usage:
    python scripts/run_a2_reward_ablation.py --seeds 3
    python scripts/run_a2_reward_ablation.py --seeds 3 --smoke   # fast plumbing test
    python scripts/run_a2_reward_ablation.py --aggregate-only     # rebuild tables
"""
import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from uav_vpp_guidance.utils.config import load_yaml_config, merge_config
from uav_vpp_guidance.evaluation.statistical_comparison import (
    paired_t_test,
    cohens_d,
    mann_whitney_u,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CANONICAL_DIR = REPO_ROOT / "config" / "canonical"
SCAFFOLD_CONFIG = REPO_ROOT / "config" / "method_innovation_comparison.yaml"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "ablation_reward_design"
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_curriculum_ppo.py"

# Dense per-step weight keys present in config/canonical/reward.yaml. The
# terminal_only condition zeroes exactly these existing keys.
DENSE_WEIGHT_KEYS = [
    "w_range", "w_angle", "w_energy", "w_safety", "w_saturation",
    "w_smooth", "w_turn_rate", "w_closing", "w_alive",
    "w_overshoot", "w_boundary",
]

CONDITIONS = ["dense_only", "dense_pbs", "terminal_only"]
BASELINE = "dense_only"


def _load_with_includes(path: Path) -> dict:
    """Replicate train_curriculum_ppo.load_experiment_config include handling."""
    base = load_yaml_config(str(path))
    includes = base.pop("includes", [])
    merged: dict = {}
    for inc in includes:
        inc_full = path.parent / inc
        if inc_full.exists():
            merged = merge_config(merged, load_yaml_config(str(inc_full)))
    return merge_config(merged, base)


def build_base_config() -> dict:
    """Simple-backend scaffold with canonical reward/guidance/virtual_point."""
    cfg = _load_with_includes(SCAFFOLD_CONFIG)

    # Force simple backend (the ablation runs on the simple point-mass env).
    cfg["backend"] = "simple"
    cfg.setdefault("env", {})["backend"] = "simple"
    cfg["env"]["use_jsbsim"] = False

    # Override reward / guidance / virtual_point with the frozen canonical blocks.
    canonical_reward = load_yaml_config(str(CANONICAL_DIR / "reward.yaml"))
    canonical_guidance = load_yaml_config(str(CANONICAL_DIR / "guidance.yaml"))
    canonical_vpp = load_yaml_config(str(CANONICAL_DIR / "virtual_point.yaml"))

    cfg["reward"] = copy.deepcopy(canonical_reward["reward"])
    if "guidance" in canonical_guidance:
        cfg["guidance"] = copy.deepcopy(canonical_guidance["guidance"])
    if "virtual_point" in canonical_vpp:
        cfg["virtual_point"] = copy.deepcopy(canonical_vpp["virtual_point"])
    # guidance.yaml also carries limits; keep canonical limits for consistency.
    if "limits" in canonical_guidance:
        cfg["limits"] = copy.deepcopy(canonical_guidance["limits"])

    return cfg


def make_condition_config(base: dict, condition: str) -> dict:
    """Derive a condition config by toggling ONLY existing canonical keys."""
    cfg = copy.deepcopy(base)
    reward = cfg["reward"]
    pbs = reward.setdefault("potential_based_shaping", {})

    if condition == "dense_only":
        pbs["enabled"] = False
    elif condition == "dense_pbs":
        pbs["enabled"] = True
    elif condition == "terminal_only":
        pbs["enabled"] = False
        for key in DENSE_WEIGHT_KEYS:
            if key in reward:
                reward[key] = 0.0
    else:
        raise ValueError(f"Unknown condition: {condition}")

    cfg.setdefault("experiment", {})["name"] = f"a2_reward_{condition}"
    return cfg


def run_training(condition: str, seed: int, smoke: bool, device: str) -> int:
    """Train one (condition, seed) run via the standard training entry point."""
    cond_dir = OUTPUT_ROOT / condition / f"seed{seed}"
    cond_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--config", str(OUTPUT_ROOT / "configs" / f"{condition}.yaml"),
        "--backend", "simple",
        "--algorithm", "ppo",
        "--seed", str(seed),
        "--output-dir", str(cond_dir),
        "--device", device,
    ]
    if smoke:
        cmd.append("--smoke")

    log_path = cond_dir / "train.log"
    print(f"[RUN] {condition} seed={seed} -> {cond_dir}")
    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    return proc.returncode


def _final_success_rate(condition: str, seed: int):
    """Read the last eval_log.csv row's success_rate for one run."""
    eval_csv = OUTPUT_ROOT / condition / f"seed{seed}" / "logs" / "eval_log.csv"
    if not eval_csv.exists():
        return None
    import csv
    rows = list(csv.DictReader(open(eval_csv, encoding="utf-8")))
    if not rows:
        return None
    last = rows[-1]
    return {
        "success_rate": float(last["success_rate"]),
        "mean_return": float(last["mean_return"]),
        "crash_rate": float(last["crash_rate"]),
        "out_of_bounds_rate": float(last["out_of_bounds_rate"]),
        "final_step": int(last["step"]),
    }


def aggregate(seeds: list[int]) -> dict:
    """Collect per-seed final metrics and compute significance vs baseline."""
    per_condition = {}
    for cond in CONDITIONS:
        seed_metrics = {}
        for s in seeds:
            m = _final_success_rate(cond, s)
            if m is not None:
                seed_metrics[s] = m
        per_condition[cond] = seed_metrics

    # Per-seed final success rate vectors (paired by seed).
    sr_vectors = {
        cond: [per_condition[cond][s]["success_rate"]
               for s in seeds if s in per_condition[cond]]
        for cond in CONDITIONS
    }

    def summary(cond):
        vals = sr_vectors[cond]
        if not vals:
            return {"n": 0, "mean": float("nan"), "std": float("nan")}
        return {
            "n": len(vals),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "values": vals,
        }

    condition_summary = {cond: summary(cond) for cond in CONDITIONS}

    # Pairwise significance on per-seed final success rate (paired by seed).
    pairwise = {}
    pairs = [
        (BASELINE, "dense_pbs"),
        (BASELINE, "terminal_only"),
        ("dense_pbs", "terminal_only"),
    ]
    for a, b in pairs:
        # Align on common seeds.
        common = [s for s in seeds
                  if s in per_condition[a] and s in per_condition[b]]
        a_vals = [per_condition[a][s]["success_rate"] for s in common]
        b_vals = [per_condition[b][s]["success_rate"] for s in common]
        pairwise[f"{a}_vs_{b}"] = {
            "n_pairs": len(common),
            "paired_t_test": paired_t_test(a_vals, b_vals),
            "cohens_d": cohens_d(a_vals, b_vals),
            "mann_whitney_u": mann_whitney_u(a_vals, b_vals),
        }

    return {
        "seeds": seeds,
        "baseline": BASELINE,
        "per_condition": per_condition,
        "condition_summary": condition_summary,
        "pairwise": pairwise,
    }


def write_reports(result: dict) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "ablation_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    cs = result["condition_summary"]
    pw = result["pairwise"]
    label = {
        "dense_only": "Dense-only (Baseline)",
        "dense_pbs": "Dense + PBS",
        "terminal_only": "Terminal-only",
    }

    lines = ["# A2' Reward-Design Ablation — Results", ""]
    lines.append(f"*Backend*: simple  |  *Algorithm*: standard PPO  "
                 f"|  *Seeds*: {result['seeds']}  |  *Baseline*: dense_only")
    lines.append("")
    lines.append("All three conditions are built only from existing canonical "
                 "reward parameters (`config/canonical/reward.yaml`). No new "
                 "default parameters were introduced.")
    lines.append("")
    lines.append("## Final Success Rate (mean ± std across seeds)")
    lines.append("")
    lines.append("| Condition | Seeds | Final SR | Per-seed SR |")
    lines.append("|-----------|-------|----------|-------------|")
    for cond in CONDITIONS:
        s = cs[cond]
        if s["n"] == 0:
            lines.append(f"| {label[cond]} | 0 | — | — |")
            continue
        per_seed = ", ".join(f"{v:.1%}" for v in s.get("values", []))
        lines.append(
            f"| {label[cond]} | {s['n']} | "
            f"{s['mean']:.1%} ± {s['std']:.1%} | {per_seed} |"
        )
    lines.append("")

    lines.append("## Statistical Significance (paired by seed)")
    lines.append("")
    lines.append("| Comparison | n | Δ SR (mean) | t | p (t-test) | Cohen's d | "
                 "Mann–Whitney p | Significant @0.05 |")
    lines.append("|------------|---|-------------|---|------------|-----------|"
                 "----------------|-------------------|")
    for key, d in pw.items():
        a, b = key.split("_vs_")
        tt = d["paired_t_test"]
        cd = d["cohens_d"]
        mw = d["mann_whitney_u"]
        sig = "yes" if tt.get("significant_at_05") else "no"
        lines.append(
            f"| {label.get(a, a)} vs {label.get(b, b)} | {d['n_pairs']} | "
            f"{tt['mean_diff']:+.1%} | {tt['t_statistic']:.3f} | "
            f"{tt['p_value']:.4f} | {cd['d']:.3f} ({cd['magnitude']}) | "
            f"{mw['p_value']:.4f} | {sig} |"
        )
    lines.append("")

    # PBS redundancy verdict.
    pbs_cmp = pw.get("dense_only_vs_dense_pbs", {})
    pbs_tt = pbs_cmp.get("paired_t_test", {})
    pbs_sig = bool(pbs_tt.get("significant_at_05"))
    lines.append("## PBS Redundancy Verdict")
    lines.append("")
    if pbs_cmp.get("n_pairs", 0) == 0:
        verdict = "INCONCLUSIVE (no paired data)."
    elif not pbs_sig:
        verdict = (
            "**PBS is REDUNDANT.** Dense-only and Dense+PBS final success rates "
            "are not statistically distinguishable (p >= 0.05). The dense mixture "
            "already provides sufficient signal; PBS should remain disabled in "
            "canonical configs."
        )
    else:
        verdict = (
            "**PBS is NOT redundant.** Dense+PBS differs significantly from "
            "dense-only (p < 0.05). Consider enabling PBS or reporting it as a "
            "beneficial component."
        )
    lines.append(verdict)
    lines.append("")

    (OUTPUT_ROOT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Wrote {OUTPUT_ROOT/'summary.md'}")
    print(f"[OK] Wrote {OUTPUT_ROOT/'ablation_results.json'}")


def main():
    parser = argparse.ArgumentParser(description="A2' reward-design ablation")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of seeds per condition (>=3 required by audit)")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"])
    parser.add_argument("--smoke", action="store_true",
                        help="Fast plumbing test (tiny budget)")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Skip training, only rebuild tables from existing logs")
    args = parser.parse_args()

    seeds = list(range(args.seeds))

    if not args.aggregate_only:
        # Materialize the three resolved condition configs.
        (OUTPUT_ROOT / "configs").mkdir(parents=True, exist_ok=True)
        base = build_base_config()
        for cond in CONDITIONS:
            cfg = make_condition_config(base, cond)
            out = OUTPUT_ROOT / "configs" / f"{cond}.yaml"
            out.write_text(
                yaml.dump(cfg, default_flow_style=False, allow_unicode=True,
                          sort_keys=False),
                encoding="utf-8",
            )
            print(f"[CONFIG] {cond} -> {out}")

        failures = []
        for cond in CONDITIONS:
            for s in seeds:
                rc = run_training(cond, s, args.smoke, args.device)
                if rc != 0:
                    failures.append((cond, s, rc))
        if failures:
            print("FAILED runs:")
            for c, s, rc in failures:
                print(f"  {c}/seed{s}: exit {rc}")
            sys.exit(1)

    result = aggregate(seeds)
    write_reports(result)
    print("Done.")


if __name__ == "__main__":
    main()
