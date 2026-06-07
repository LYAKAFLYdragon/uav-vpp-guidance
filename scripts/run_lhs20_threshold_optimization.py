#!/usr/bin/env python3
"""Stage 6H.2: Formal LHS20 threshold optimization for mode-switch gate.

Scans 20 Latin-Hypercube-sampled threshold configurations and reports
those that satisfy all hard constraints.
"""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uav_vpp_guidance.evaluation.threshold_runner import ThresholdOptimizationRunner


# Parameter grid definitions
ASPECT_OPTIONS = np.array([10.0, 15.0, 20.0, 25.0, 30.0])
RANGE_OPTIONS = np.array([1500.0, 2000.0, 2500.0, 3000.0, 4000.0])
SPEED_OPTIONS = np.array([50.0, 80.0, 100.0, 150.0, 200.0])


def generate_lhs_samples(n_samples: int = 20, seed: int = 42) -> np.ndarray:
    """Generate n_samples in [0,1]^3 using LHS."""
    sampler = qmc.LatinHypercube(d=3, seed=seed)
    samples = sampler.random(n=n_samples)
    return samples


def discretize_samples(samples: np.ndarray) -> np.ndarray:
    """Map continuous LHS samples to nearest discrete parameter values.

    Returns array of shape (n_samples, 3) with columns:
      [aspect_threshold_deg, range_threshold_m, closing_speed_threshold_mps]
    """
    def _nearest(val, options):
        return float(options[np.argmin(np.abs(options - val))])

    configs = []
    for s in samples:
        aspect = _nearest(s[0] * (ASPECT_OPTIONS.max() - ASPECT_OPTIONS.min()) + ASPECT_OPTIONS.min(), ASPECT_OPTIONS)
        range_m = _nearest(s[1] * (RANGE_OPTIONS.max() - RANGE_OPTIONS.min()) + RANGE_OPTIONS.min(), RANGE_OPTIONS)
        speed = _nearest(s[2] * (SPEED_OPTIONS.max() - SPEED_OPTIONS.min()) + SPEED_OPTIONS.min(), SPEED_OPTIONS)
        configs.append([aspect, range_m, speed])
    return np.array(configs)


def build_gate_config(aspect_th: float, range_th: float, speed_th: float) -> dict:
    return {
        "enabled": True,
        "aspect_threshold_deg": aspect_th,
        "range_threshold_m": range_th,
        "closing_speed_threshold_mps": speed_th,
        "crossing_aspect_threshold_deg": None,
    }


def main():
    parser = argparse.ArgumentParser(description="LHS20 mode-switch threshold optimization")
    parser.add_argument("--config", type=str, default="config/experiment/stage6f5_feasible_geometry.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/audit_no_pred_final/checkpoints/best.pt")
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--lhs-seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/lhs20_threshold_optimization")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Per-scenario eval seeds (default 0..9)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load base config
    config_path = Path(args.config)
    full_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    method_override = full_config.get("methods", {}).get("no_prediction", {})
    base_config = merge_config(copy.deepcopy(full_config), copy.deepcopy(method_override))

    seeds = tuple(args.seeds) if args.seeds else tuple(range(10))

    # Generate LHS samples
    raw_samples = generate_lhs_samples(n_samples=args.n_samples, seed=args.lhs_seed)
    configs = discretize_samples(raw_samples)

    # Deduplicate (LHS can map to same discrete cell)
    unique_configs = []
    seen = set()
    for c in configs:
        key = (round(c[0], 1), round(c[1], 1), round(c[2], 1))
        if key not in seen:
            seen.add(key)
            unique_configs.append(c)

    print(f"LHS generated {args.n_samples} samples; {len(unique_configs)} unique after discretization.")
    print(f"Evaluating with seeds {seeds} ...\n")

    runner = ThresholdOptimizationRunner(
        base_config=base_config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        seeds=seeds,
    )

    results = []
    for idx, (aspect_th, range_th, speed_th) in enumerate(unique_configs, start=1):
        gate_cfg = build_gate_config(aspect_th, range_th, speed_th)
        print(f"[{idx}/{len(unique_configs)}] aspect={aspect_th:.0f}° range={range_th:.0f}m speed={speed_th:.0f}m/s ... ", end="", flush=True)
        verdict_dict = runner.evaluate_config(gate_cfg)
        print(f"reg={verdict_dict['regression_success']}/{verdict_dict['regression_total']} "
              f"cand={verdict_dict['candidate_success']}/{verdict_dict['candidate_total']} "
              f"tc={verdict_dict['tail_chase_success']}/{verdict_dict['tail_chase_total']} "
              f"verdict={verdict_dict['verdict']}")
        results.append(verdict_dict)

    df = pd.DataFrame(results)

    # Save CSV
    csv_path = output_dir / "lhs20_threshold_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}")

    # Build markdown report
    report_lines = [
        "# Stage 6H.2: LHS20 Mode-Switch Threshold Optimization Report",
        "",
        f"- LHS seed: {args.lhs_seed}",
        f"- Total samples (raw): {args.n_samples}",
        f"- Unique after discretization: {len(unique_configs)}",
        f"- Eval seeds per scenario: {list(seeds)}",
        f"- Checkpoint: `{args.checkpoint}`",
        "",
        "## Parameter Grid",
        "",
        "| Parameter | Options |",
        "|---|---|",
        f"| aspect_threshold_deg | {list(ASPECT_OPTIONS)} |",
        f"| range_threshold_m | {list(RANGE_OPTIONS)} |",
        f"| closing_speed_threshold_mps | {list(SPEED_OPTIONS)} |",
        "",
        "## Hard Constraints",
        "",
        "1. Regression baseline: 40/40 success (100%)",
        "2. Candidate search: ≥38/40 success (≥95%)",
        "3. Negative tail_chase: 10/10 success AND 10/10 mode_switch",
        "4. Negative fleeing: 0/10 success",
        "5. Negative offset_attack: 0/10 success",
        "",
        "## Results Summary",
        "",
        f"- PASS: {(df['verdict'] == 'PASS').sum()} / {len(df)}",
        f"- FAIL: {(df['verdict'] == 'FAIL').sum()} / {len(df)}",
        "",
    ]

    pass_df = df[df["verdict"] == "PASS"].sort_values(
        by=["candidate_success", "regression_success"], ascending=[False, False]
    )

    if not pass_df.empty:
        report_lines.append("## PASS Configurations (sorted by candidate success)")
        report_lines.append("")
        report_lines.append("| # | aspect_th | range_th | speed_th | regression | candidate | tc_success | tc_switch | fleeing | offset | violations |")
        report_lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for i, row in pass_df.iterrows():
            report_lines.append(
                f"| {i+1} | {row['aspect_threshold_deg']:.0f} | {row['range_threshold_m']:.0f} | {row['closing_speed_threshold_mps']:.0f} | "
                f"{row['regression_success']}/{row['regression_total']} | "
                f"{row['candidate_success']}/{row['candidate_total']} | "
                f"{row['tail_chase_success']}/{row['tail_chase_total']} | "
                f"{row['tail_chase_switch']}/{row['tail_chase_total']} | "
                f"{row['fleeing_success']}/{row['fleeing_total']} | "
                f"{row['offset_success']}/{row['offset_total']} | "
                f"{row['violations']} |"
            )
        report_lines.append("")
        best = pass_df.iloc[0]
        report_lines.append("### Recommended Configuration")
        report_lines.append("")
        report_lines.append("```yaml")
        report_lines.append("guidance:")
        report_lines.append("  mode_switch:")
        report_lines.append("    enabled: true")
        report_lines.append(f"    aspect_threshold_deg: {best['aspect_threshold_deg']:.0f}")
        report_lines.append(f"    range_threshold_m: {best['range_threshold_m']:.0f}")
        report_lines.append(f"    closing_speed_threshold_mps: {best['closing_speed_threshold_mps']:.0f}")
        report_lines.append("    crossing_aspect_threshold_deg: null")
        report_lines.append("```")
        report_lines.append("")
    else:
        report_lines.append("## PASS Configurations")
        report_lines.append("")
        report_lines.append("_No configuration satisfied all hard constraints._")
        report_lines.append("")

    # Add full table
    report_lines.append("## Full Results Table")
    report_lines.append("")
    report_lines.append("| # | aspect | range | speed | regression | candidate | tc_suc | tc_sw | flee | off | verdict | violations |")
    report_lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, row in df.iterrows():
        report_lines.append(
            f"| {i+1} | {row['aspect_threshold_deg']:.0f} | {row['range_threshold_m']:.0f} | {row['closing_speed_threshold_mps']:.0f} | "
            f"{row['regression_success']}/{row['regression_total']} | "
            f"{row['candidate_success']}/{row['candidate_total']} | "
            f"{row['tail_chase_success']}/{row['tail_chase_total']} | "
            f"{row['tail_chase_switch']}/{row['tail_chase_total']} | "
            f"{row['fleeing_success']}/{row['fleeing_total']} | "
            f"{row['offset_success']}/{row['offset_total']} | "
            f"{row['verdict']} | {row['violations']} |"
        )
    report_lines.append("")

    md_path = output_dir / "lhs20_threshold_report.md"
    md_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved report: {md_path}")


def merge_config(base, override):
    """Shallow merge dict (same as utils.config.merge_config)."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and k in result and isinstance(result[k], dict):
            result[k] = merge_config(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


if __name__ == "__main__":
    main()
