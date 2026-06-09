#!/usr/bin/env python3
"""
Maneuver Parameter Sweep Experiment.

Validates the "conditional advantage" hypothesis: predictor benefit
stratifies with target maneuver intensity.

For each (amplitude, frequency) grid cell, runs the full 5-method ablation
matrix and records success rates.  Generates a summary table, statistical
tests, and visualizations.

Usage (full sweep, 10 seeds, 10 episodes/scenario):
    python scripts/run_maneuver_sweep.py \
        --checkpoints-config config/experiment/ablation_checkpoints.yaml \
        --sweep-type sinusoidal_weaving \
        --amplitude-range 1.0 3.0 5.0 \
        --frequency-range 0.5 1.0 2.0 \
        --scenarios favorable neutral disadvantage challenging \
        --seeds 10 --episodes-per-scenario 10 \
        --output-dir outputs/maneuver_sweep

Usage (smoke test):
    python scripts/run_maneuver_sweep.py ... --smoke

Usage (resume interrupted sweep):
    python scripts/run_maneuver_sweep.py ... --resume
"""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


METHOD_KEYS = [
    "no_prediction",
    "cv_prediction",
    "ca_prediction",
    "lstm_frozen",
    "gru_frozen",
]
METHOD_DISPLAY_NAMES = {
    "no_prediction": "No-Pred",
    "cv_prediction": "CV",
    "ca_prediction": "CA",
    "lstm_frozen": "LSTM",
    "gru_frozen": "GRU",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Maneuver Parameter Sweep Experiment"
    )

    # Checkpoints
    parser.add_argument(
        "--checkpoints-config",
        type=str,
        required=True,
        help="YAML with checkpoint paths for all 5 methods",
    )

    # Sweep parameters
    parser.add_argument(
        "--sweep-type",
        type=str,
        default="sinusoidal_weaving",
        choices=["sinusoidal_weaving", "bang_bang", "barrel_roll"],
        help="Target maneuver type",
    )
    parser.add_argument(
        "--amplitude-range",
        type=float,
        nargs="+",
        default=[1.0, 3.0, 5.0],
        help="Amplitude values to sweep (in g for weaving/bang-bang)",
    )
    parser.add_argument(
        "--frequency-range",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0],
        help="Frequency values to sweep (in rad/s)",
    )

    # Scenarios & evaluation scale
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="+",
        default=["favorable", "neutral", "disadvantage", "challenging"],
        help="Scenarios to evaluate",
    )
    parser.add_argument(
        "--seeds", type=int, default=10, help="Number of evaluation seeds"
    )
    parser.add_argument(
        "--episodes-per-scenario",
        type=int,
        default=10,
        help="Episodes per scenario",
    )

    # Base template
    parser.add_argument(
        "--base-config",
        type=str,
        default="config/experiment/evaluate_vpp_prediction_comparison.yaml",
        help="Base evaluation config template",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/maneuver_sweep",
        help="Root output directory",
    )

    # Modes
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test: 1 config, 1 seed, 1 episode",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed sweep configurations",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="simple",
        choices=["simple", "jsbsim"],
    )
    parser.add_argument(
        "--allow-random-policy",
        action="store_true",
        help="Allow random policy fallback for smoke/debug",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Override PPO device for evaluation",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: str) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _save_yaml(config: dict, path: str) -> None:
    import yaml

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def load_checkpoints_config(path: str) -> dict:
    """Load checkpoint paths YAML."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoints config not found: {path}")
    return _load_yaml(path)


def build_sweep_configs(args) -> List[dict]:
    """Generate sweep parameter grid."""
    configs = []
    for amp in args.amplitude_range:
        for freq in args.frequency_range:
            cfg = {"amplitude": float(amp), "frequency": float(freq)}
            if args.sweep_type == "sinusoidal_weaving":
                cfg["env_overrides"] = {
                    "target_mode": "sinusoidal_weaving",
                    "weaving_amplitude_g": float(amp),
                    "weaving_frequency_rad_s": float(freq),
                }
            elif args.sweep_type == "bang_bang":
                # Map frequency (rad/s) to switch interval (s)
                # sign switches every pi rad -> T = pi / freq
                switch_interval = math.pi / float(freq) if freq > 0 else 2.0
                cfg["env_overrides"] = {
                    "target_mode": "bang_bang",
                    "bang_bang_max_g": float(amp),
                    "bang_bang_switch_interval_s": switch_interval,
                }
            elif args.sweep_type == "barrel_roll":
                cfg["env_overrides"] = {
                    "target_mode": "barrel_roll",
                    "barrel_roll_rate_rad_s": float(freq),
                    "barrel_roll_vertical_amp_m": float(amp) * 10.0,
                }
            configs.append(cfg)
    return configs


def build_temp_eval_config(
    base_config_path: str,
    sweep_cfg: dict,
    ckpts: dict,
    temp_dir: Path,
) -> str:
    """
    Build a temporary evaluation config that injects maneuver parameters
    and checkpoint paths into the base template.
    """
    config = _load_yaml(base_config_path)

    # Inject env overrides
    if "env" not in config:
        config["env"] = {}
    config["env"].update(sweep_cfg["env_overrides"])

    # Inject checkpoints into methods block
    methods = config.get("methods", {})
    cp_map = ckpts.get("checkpoints", {})
    for method_key in METHOD_KEYS:
        if method_key in methods and method_key in cp_map:
            methods[method_key]["checkpoint"] = cp_map[method_key]

    # Inject predictor checkpoints
    pred_cp = ckpts.get("predictor_checkpoints", {})
    if "lstm_frozen" in methods and pred_cp.get("lstm"):
        tp = methods["lstm_frozen"].setdefault("trajectory_prediction", {})
        tp["checkpoint_path"] = pred_cp["lstm"]
    if "gru_frozen" in methods and pred_cp.get("gru"):
        tp = methods["gru_frozen"].setdefault("trajectory_prediction", {})
        tp["checkpoint_path"] = pred_cp["gru"]

    # Save temp file
    temp_dir.mkdir(parents=True, exist_ok=True)
    fname = f"sweep_A{sweep_cfg['amplitude']:.2f}_f{sweep_cfg['frequency']:.2f}.yaml"
    temp_path = temp_dir / fname
    _save_yaml(config, str(temp_path))
    return str(temp_path)


# ---------------------------------------------------------------------------
# Single sweep runner
# ---------------------------------------------------------------------------


def run_single_sweep(
    args,
    sweep_cfg: dict,
    ckpts: dict,
    sub_output_dir: str,
    temp_dir: Path,
) -> dict:
    """Run the ablation matrix for one sweep cell."""
    temp_config = build_temp_eval_config(args.base_config, sweep_cfg, ckpts, temp_dir)

    cp_map = ckpts.get("checkpoints", {})
    cmd = [
        sys.executable,
        "scripts/run_ablation_matrix.py",
        "--scenarios-config",
        temp_config,
        "--no-pred-checkpoint",
        cp_map.get("no_prediction", ""),
        "--cv-checkpoint",
        cp_map.get("cv_prediction", ""),
        "--ca-checkpoint",
        cp_map.get("ca_prediction", ""),
        "--lstm-checkpoint",
        cp_map.get("lstm_frozen", ""),
        "--gru-checkpoint",
        cp_map.get("gru_frozen", ""),
        "--seeds",
        str(args.seeds),
        "--episodes-per-scenario",
        str(args.episodes_per_scenario),
        "--scenarios",
    ] + list(args.scenarios) + [
        "--output-dir",
        sub_output_dir,
        "--backend",
        args.backend,
    ]

    pred_cp = ckpts.get("predictor_checkpoints", {})
    if pred_cp.get("lstm"):
        cmd.extend(["--lstm-predictor-ckpt", pred_cp["lstm"]])
    if pred_cp.get("gru"):
        cmd.extend(["--gru-predictor-ckpt", pred_cp["gru"]])

    if args.allow_random_policy:
        cmd.append("--allow-random-policy")

    if args.device:
        # run_ablation_matrix.py does not accept --device directly,
        # but evaluate_prediction_comparison does not either.
        # We skip device override here for simplicity.
        pass

    print(f"\n[SWEEP] {'=' * 60}")
    print(
        f"Config: A={sweep_cfg['amplitude']:.1f}g, ω={sweep_cfg['frequency']:.1f}rad/s"
    )
    print(f"Output: {sub_output_dir}")
    start = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - start

    success = result.returncode == 0
    print(f"[SWEEP] {'OK' if success else 'FAILED'} in {elapsed:.1f}s")
    return {"success": success, "elapsed_s": elapsed}


# ---------------------------------------------------------------------------
# Result loaders
# ---------------------------------------------------------------------------


def read_method_success_rates(sub_output_dir: str) -> Dict[str, dict]:
    """
    Read per-method success rates from ablation evaluation output.

    Returns:
        dict: {method_key: {"success_rate": float, "mean_return": float, "per_seed": {seed: float}}}
    """
    json_path = Path(sub_output_dir) / "evaluation" / "prediction_metrics.json"
    if not json_path.exists():
        return {}

    with open(json_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    results = {}
    for m in metrics:
        method = m.get("method", "unknown")
        sr = m.get("instant_success_rate", m.get("success_rate", np.nan))
        mr = m.get("mean_return", np.nan)

        # Extract per-seed success rates for paired testing
        per_seed_sr = {}
        per_seed = m.get("per_seed", {})
        for seed_key, episodes in per_seed.items():
            if not episodes:
                continue
            successes = sum(1 for ep in episodes if ep.get("is_success", False))
            per_seed_sr[seed_key] = successes / len(episodes)

        results[method] = {
            "success_rate": float(sr) if np.isfinite(sr) else np.nan,
            "mean_return": float(mr) if np.isfinite(mr) else np.nan,
            "episodes": int(m.get("episodes", 0)),
            "per_seed_sr": per_seed_sr,
        }
    return results


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------


def paired_ttest_and_cohens_d(
    a_vals: List[float], b_vals: List[float]
) -> Tuple[dict, dict]:
    """
    Paired t-test and Cohen's d for matched samples.

    Args:
        a_vals, b_vals: Lists of paired observations (must be same length).

    Returns:
        tuple: (test_result dict, cohens_d dict)
    """
    try:
        from scipy import stats
    except ImportError:
        return {}, {}

    a = np.array(a_vals, dtype=float)
    b = np.array(b_vals, dtype=float)
    if len(a) != len(b) or len(a) < 2:
        return {"error": "insufficient paired samples"}, {}

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(a, b)

    # Cohen's d for paired samples
    diff = a - b
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))
    cohens_d = mean_diff / std_diff if std_diff > 1e-12 else 0.0

    test_result = {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "n_pairs": len(a),
        "significant_05": bool(p_value < 0.05),
        "significant_01": bool(p_value < 0.01),
    }
    cohens_d_result = {
        "cohens_d": float(cohens_d),
        "magnitude": (
            "negligible"
            if abs(cohens_d) < 0.2
            else "small"
            if abs(cohens_d) < 0.5
            else "medium"
            if abs(cohens_d) < 0.8
            else "large"
        ),
    }
    return test_result, cohens_d_result


def run_statistical_analysis(
    all_results: List[dict], output_dir: str
) -> List[dict]:
    """
    Aggregate success rates and run pairwise statistical tests.

    Returns:
        list of summary rows (one per sweep cell).
    """
    summary_rows = []
    for r in all_results:
        amp = r["amplitude"]
        freq = r["frequency"]
        res = r["results"]

        row = {
            "amplitude_g": amp,
            "frequency_rad_s": freq,
            "no_pred_sr": res.get("no_prediction", {}).get("success_rate", np.nan),
            "cv_sr": res.get("cv_prediction", {}).get("success_rate", np.nan),
            "ca_sr": res.get("ca_prediction", {}).get("success_rate", np.nan),
            "lstm_sr": res.get("lstm_frozen", {}).get("success_rate", np.nan),
            "gru_sr": res.get("gru_frozen", {}).get("success_rate", np.nan),
        }

        # Predictor benefit = LSTM - No-Pred
        lstm_sr = row["lstm_sr"]
        no_pred_sr = row["no_pred_sr"]
        row["predictor_benefit"] = (
            lstm_sr - no_pred_sr
            if np.isfinite(lstm_sr) and np.isfinite(no_pred_sr)
            else np.nan
        )

        # Statistical tests using per-seed success rates
        lstm_per_seed = res.get("lstm_frozen", {}).get("per_seed_sr", {})
        no_pred_per_seed = res.get("no_prediction", {}).get("per_seed_sr", {})
        ca_per_seed = res.get("ca_prediction", {}).get("per_seed_sr", {})
        cv_per_seed = res.get("cv_prediction", {}).get("per_seed_sr", {})

        # LSTM vs No-Pred
        common_seeds = set(lstm_per_seed.keys()) & set(no_pred_per_seed.keys())
        if len(common_seeds) >= 2:
            seeds_sorted = sorted(common_seeds)
            lstm_vals = [lstm_per_seed[s] for s in seeds_sorted]
            no_pred_vals = [no_pred_per_seed[s] for s in seeds_sorted]
            test, d = paired_ttest_and_cohens_d(lstm_vals, no_pred_vals)
            row["lstm_vs_no_pred_t"] = test.get("t_statistic", np.nan)
            row["lstm_vs_no_pred_p"] = test.get("p_value", np.nan)
            row["lstm_vs_no_pred_cohens_d"] = d.get("cohens_d", np.nan)
            row["lstm_vs_no_pred_sig"] = test.get("significant_05", False)
        else:
            row["lstm_vs_no_pred_t"] = np.nan
            row["lstm_vs_no_pred_p"] = np.nan
            row["lstm_vs_no_pred_cohens_d"] = np.nan
            row["lstm_vs_no_pred_sig"] = False

        # CA vs CV
        common_seeds2 = set(ca_per_seed.keys()) & set(cv_per_seed.keys())
        if len(common_seeds2) >= 2:
            seeds_sorted = sorted(common_seeds2)
            ca_vals = [ca_per_seed[s] for s in seeds_sorted]
            cv_vals = [cv_per_seed[s] for s in seeds_sorted]
            test2, d2 = paired_ttest_and_cohens_d(ca_vals, cv_vals)
            row["ca_vs_cv_t"] = test2.get("t_statistic", np.nan)
            row["ca_vs_cv_p"] = test2.get("p_value", np.nan)
            row["ca_vs_cv_cohens_d"] = d2.get("cohens_d", np.nan)
            row["ca_vs_cv_sig"] = test2.get("significant_05", False)
        else:
            row["ca_vs_cv_t"] = np.nan
            row["ca_vs_cv_p"] = np.nan
            row["ca_vs_cv_cohens_d"] = np.nan
            row["ca_vs_cv_sig"] = False

        summary_rows.append(row)

    # Save summary CSV
    csv_path = Path(output_dir) / "summary.csv"
    if summary_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"[SUMMARY] {csv_path}")

    return summary_rows


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def generate_visualizations(summary: List[dict], output_dir: str):
    """Generate heatmap and line plots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = Path(output_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    amplitudes = sorted({r["amplitude_g"] for r in summary})
    frequencies = sorted({r["frequency_rad_s"] for r in summary})

    # --- Heatmap 1: Predictor benefit (LSTM - No-Pred) ---
    benefit_grid = np.full((len(amplitudes), len(frequencies)), np.nan)
    for r in summary:
        i = amplitudes.index(r["amplitude_g"])
        j = frequencies.index(r["frequency_rad_s"])
        benefit_grid[i, j] = r["predictor_benefit"]

    fig, ax = plt.subplots(figsize=(8, 6))
    vmax = max(0.5, np.nanmax(np.abs(benefit_grid)))
    im = ax.imshow(
        benefit_grid,
        cmap="RdYlGn",
        aspect="auto",
        vmin=-vmax,
        vmax=vmax,
        origin="lower",
    )
    ax.set_xticks(range(len(frequencies)))
    ax.set_yticks(range(len(amplitudes)))
    ax.set_xticklabels([f"{f:.1f}" for f in frequencies])
    ax.set_yticklabels([f"{a:.1f}" for a in amplitudes])
    ax.set_xlabel("Frequency (rad/s)")
    ax.set_ylabel("Amplitude (g)")
    ax.set_title("Predictor Benefit: LSTM vs No-Prediction (Δ Success Rate)")
    # Annotate cells
    for i in range(len(amplitudes)):
        for j in range(len(frequencies)):
            val = benefit_grid[i, j]
            if np.isfinite(val):
                ax.text(
                    j,
                    i,
                    f"{val:+.2f}",
                    ha="center",
                    va="center",
                    color="black" if abs(val) < 0.25 else "white",
                    fontsize=9,
                )
    plt.colorbar(im, ax=ax, label="Δ Success Rate")
    plt.tight_layout()
    heatmap_path = fig_dir / "heatmap_predictor_benefit.png"
    fig.savefig(heatmap_path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] {heatmap_path}")

    # --- Line plot: success rate vs frequency for each amplitude ---
    fig, ax = plt.subplots(figsize=(10, 6))
    method_keys = ["no_pred_sr", "cv_sr", "ca_sr", "lstm_sr", "gru_sr"]
    method_labels = ["No-Pred", "CV", "CA", "LSTM", "GRU"]
    colors = ["#7f7f7f", "#1f77b4", "#2ca02c", "#d62728", "#9467bd"]
    markers = ["o", "s", "^", "D", "v"]

    for method_key, label, color, marker in zip(
        method_keys, method_labels, colors, markers
    ):
        for amp in amplitudes:
            subset = [r for r in summary if r["amplitude_g"] == amp]
            subset.sort(key=lambda x: x["frequency_rad_s"])
            freqs = [r["frequency_rad_s"] for r in subset]
            srs = [r[method_key] for r in subset]
            ax.plot(
                freqs,
                srs,
                marker=marker,
                label=f"{label} (A={amp}g)"
                if method_key == "no_pred_sr"
                else None,
                color=color,
                alpha=0.7,
                linestyle="-" if amp == amplitudes[0] else "--" if amp == amplitudes[1] else ":",
                linewidth=2,
            )

    ax.set_xlabel("Frequency (rad/s)", fontsize=12)
    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_title("Success Rate vs Frequency by Amplitude", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    line_path = fig_dir / "line_success_rate_by_frequency.png"
    fig.savefig(line_path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] {line_path}")

    # --- Line plot 2: Predictor benefit vs frequency ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for amp in amplitudes:
        subset = [r for r in summary if r["amplitude_g"] == amp]
        subset.sort(key=lambda x: x["frequency_rad_s"])
        freqs = [r["frequency_rad_s"] for r in subset]
        benefits = [r["predictor_benefit"] for r in subset]
        ax.plot(
            freqs,
            benefits,
            marker="o",
            label=f"A={amp}g",
            linewidth=2,
        )
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Frequency (rad/s)", fontsize=12)
    ax.set_ylabel("Predictor Benefit (Δ Success Rate)", fontsize=12)
    ax.set_title("LSTM vs No-Pred: Predictor Benefit by Maneuver Intensity", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    plt.tight_layout()
    benefit_path = fig_dir / "line_predictor_benefit.png"
    fig.savefig(benefit_path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] {benefit_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    # Smoke overrides
    if args.smoke:
        args.amplitude_range = (
            [args.amplitude_range[0]] if args.amplitude_range else [1.0]
        )
        args.frequency_range = (
            [args.frequency_range[0]] if args.frequency_range else [0.5]
        )
        args.seeds = 1
        args.episodes_per_scenario = 1
        args.allow_random_policy = True
        print("[SMOKE] Reduced sweep for smoke test")
        print(
            f"  A={args.amplitude_range}, f={args.frequency_range}, "
            f"seeds={args.seeds}, eps={args.episodes_per_scenario}"
        )

    # Load checkpoints
    ckpts = load_checkpoints_config(args.checkpoints_config)
    print(f"[INFO] Loaded checkpoints config: {args.checkpoints_config}")

    # Build sweep grid
    sweep_configs = build_sweep_configs(args)
    print(f"[INFO] Sweep matrix: {len(sweep_configs)} configurations")
    print(f"       Amplitudes: {args.amplitude_range}")
    print(f"       Frequencies: {args.frequency_range}")
    print(f"       Maneuver: {args.sweep_type}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / "_temp"

    all_results = []
    failed_cells = []

    for sweep_cfg in sweep_configs:
        amp = sweep_cfg["amplitude"]
        freq = sweep_cfg["frequency"]
        sub_name = f"A{amp:.1f}_f{freq:.1f}"
        sub_output_dir = str(output_dir / sub_name)

        # Resume check
        if args.resume:
            manifest_path = Path(sub_output_dir) / "ablation_manifest.json"
            if manifest_path.exists():
                print(f"\n[RESUME] Skipping completed config: {sub_name}")
                res = read_method_success_rates(sub_output_dir)
                if res:
                    all_results.append(
                        {"amplitude": amp, "frequency": freq, "results": res}
                    )
                continue

        # Run ablation for this sweep cell
        run_result = run_single_sweep(
            args, sweep_cfg, ckpts, sub_output_dir, temp_dir
        )

        if run_result["success"]:
            res = read_method_success_rates(sub_output_dir)
            if res:
                all_results.append(
                    {"amplitude": amp, "frequency": freq, "results": res}
                )
            else:
                print(f"[WARN] No results found for {sub_name}")
                failed_cells.append(sub_name)
        else:
            print(f"[ERROR] Sweep cell failed: {sub_name}")
            failed_cells.append(sub_name)

    # Aggregate & analyze
    if all_results:
        print("\n" + "=" * 60)
        print("AGGREGATING & ANALYZING")
        print("=" * 60)
        summary = run_statistical_analysis(all_results, str(output_dir))
        generate_visualizations(summary, str(output_dir))

        # Save full manifest
        manifest = {
            "sweep_type": args.sweep_type,
            "amplitude_range": args.amplitude_range,
            "frequency_range": args.frequency_range,
            "scenarios": args.scenarios,
            "seeds": args.seeds,
            "episodes_per_scenario": args.episodes_per_scenario,
            "backend": args.backend,
            "num_cells_total": len(sweep_configs),
            "num_cells_success": len(all_results),
            "num_cells_failed": len(failed_cells),
            "failed_cells": failed_cells,
            "summary": summary,
        }
        manifest_path = output_dir / "sweep_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)
        print(f"[MANIFEST] {manifest_path}")

        # Quick print
        print("\nQuick Summary:")
        for row in summary:
            print(
                f"  A={row['amplitude_g']:.1f}g ω={row['frequency_rad_s']:.1f}rad/s | "
                f"No-Pred={row['no_pred_sr']:.2%} LSTM={row['lstm_sr']:.2%} "
                f"Benefit={row['predictor_benefit']:+.2%} "
                f"p={row['lstm_vs_no_pred_p']:.3f}"
            )

    print("\n" + "=" * 60)
    print("SWEEP COMPLETE")
    print(f"Successful cells: {len(all_results)} / {len(sweep_configs)}")
    if failed_cells:
        print(f"Failed cells: {failed_cells}")
    print("=" * 60)

    if failed_cells:
        sys.exit(1)


if __name__ == "__main__":
    main()
