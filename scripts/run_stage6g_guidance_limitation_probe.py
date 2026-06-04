#!/usr/bin/env python3
"""
Stage 6G Guidance-Law Limitation Probe.

Lightweight comparison of guidance laws on tail-chase / stern-conversion
dead-zone scenarios to determine whether the dead zone is specific to
LOS-rate guidance or inherent to the VPP formulation.

Methods: no_prediction, gru_frozen (training_seed 0 only)
Guidance modes: los_rate, proportional_navigation, hybrid
Scenarios: favorable, disadvantage, weaving_pursuit, weaving_disadvantage
Episodes: 10 per scenario
Eval seeds: 0, 1, 2

Usage:
    python scripts/run_stage6g_guidance_limitation_probe.py --dry-run
    python scripts/run_stage6g_guidance_limitation_probe.py
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

GUIDANCE_MODES = ["los_rate", "proportional_navigation", "hybrid"]

METHODS = [
    {
        "name": "no_prediction",
        "checkpoint": "outputs/experiments/no_prediction_vpp_ppo_seed0/checkpoints/best.pt",
    },
    {
        "name": "gru_frozen",
        "checkpoint": "outputs/experiments/vpp_ppo_gru_frozen_seed0/checkpoints/best.pt",
    },
]

SCENARIO_CONFIGS = {
    "favorable": {
        "base_config": "config/experiment/stage6f5_feasible_geometry.yaml",
        "scenario": "favorable",
    },
    "disadvantage": {
        "base_config": "config/experiment/stage6f5_feasible_geometry.yaml",
        "scenario": "disadvantage",
    },
    "weaving_pursuit": {
        "base_config": "config/experiment/stage6f5_maneuvering_target.yaml",
        "scenario": "weaving_pursuit",
    },
    "weaving_disadvantage": {
        "base_config": "config/experiment/stage6f5_maneuvering_target.yaml",
        "scenario": "weaving_disadvantage",
    },
}

OUTPUT_ROOT = "outputs/tables/stage6g_guidance_limitation_probe"


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def build_probe_config(base_config_path: str, guidance_mode: str, scenario_name: str) -> dict:
    """Load base config and override guidance mode + scenario subset + methods subset."""
    cfg = load_yaml(base_config_path)
    # Override guidance mode
    if "guidance" not in cfg:
        cfg["guidance"] = {}
    cfg["guidance"]["mode"] = guidance_mode
    # Keep only the requested scenario
    scenarios = cfg.get("scenarios", {})
    if scenario_name in scenarios:
        cfg["scenarios"] = {scenario_name: scenarios[scenario_name]}
    else:
        raise ValueError(f"Scenario {scenario_name} not found in {base_config_path}")
    # Keep only the two probe methods in config
    if "methods" in cfg:
        wanted = {m["name"] for m in METHODS}
        filtered = {k: v for k, v in cfg["methods"].items() if k in wanted}
        cfg["methods"] = filtered
    return cfg


def run_probe(
    guidance_mode: str,
    scenario_name: str,
    base_config_path: str,
    episodes_per_scenario: int,
    eval_seeds: list,
    dry_run: bool,
) -> bool:
    """Run a single guidance-mode x scenario probe."""
    print(f"\n{'='*60}")
    print(f"Probe: guidance={guidance_mode} | scenario={scenario_name}")
    print(f"{'='*60}")

    # Build resolved config and save to output dir
    probe_cfg = build_probe_config(base_config_path, guidance_mode, scenario_name)
    output_dir = os.path.join(OUTPUT_ROOT, f"{guidance_mode}_{scenario_name}")
    os.makedirs(output_dir, exist_ok=True)
    resolved_config_path = os.path.join(output_dir, "resolved_config.yaml")
    save_yaml(resolved_config_path, probe_cfg)

    # Verify resolved config matches requested mode
    resolved_mode = probe_cfg.get("guidance", {}).get("mode", "unknown")
    if resolved_mode != guidance_mode:
        print(f"  ERROR: Resolved config guidance mode mismatch: {resolved_mode} != {guidance_mode}")
        return False

    method_overrides = []
    for method in METHODS:
        ckpt = method["checkpoint"]
        if not os.path.exists(ckpt) and not dry_run:
            print(f"  WARNING: Checkpoint not found: {ckpt}")
        method_overrides.append(f"{method['name']}={ckpt}")

    cmd = [
        sys.executable,
        "-m",
        "uav_vpp_guidance.evaluation.evaluate_prediction_comparison",
        "--config", resolved_config_path,
        "--backend", "simple",
        "--training-seed", "0",
        "--episodes-per-scenario", str(episodes_per_scenario),
        "--seeds", *map(str, eval_seeds),
        "--scenarios", scenario_name,
        "--output-dir", output_dir,
        "--validation-mode", "raise",
    ]
    for override in method_overrides:
        cmd.extend(["--method-checkpoint", override])

    if dry_run:
        print(f"  [DRY-RUN] {' '.join(cmd)}")
        return True

    result = subprocess.run(cmd, cwd=os.getcwd())

    if result.returncode != 0:
        print(f"ERROR: Probe failed for {guidance_mode} / {scenario_name}")
        return False

    print(f"  Results saved to {output_dir}")
    return True


def aggregate_results() -> dict:
    """Aggregate all probe outputs into a single summary."""
    rows = []
    for guidance_mode in GUIDANCE_MODES:
        for scenario_name in SCENARIO_CONFIGS:
            probe_dir = Path(OUTPUT_ROOT) / f"{guidance_mode}_{scenario_name}"
            metrics_json = probe_dir / "prediction_metrics.json"
            if not metrics_json.exists():
                continue
            with open(metrics_json, "r", encoding="utf-8") as f:
                methods_data = json.load(f)
            for m in methods_data:
                per_scenario = m.get("per_scenario", {}).get(scenario_name, {})
                rows.append({
                    "guidance_mode": guidance_mode,
                    "scenario": scenario_name,
                    "method": m.get("method_name", m.get("method", "unknown")),
                    "success_rate": per_scenario.get("success_rate", m.get("success_rate", 0)),
                    "mean_return": per_scenario.get("mean_return", m.get("mean_return", 0)),
                    "crash_rate": per_scenario.get("crash_rate", m.get("crash_rate", 0)),
                    "out_of_bounds_rate": per_scenario.get("out_of_bounds_rate", m.get("out_of_bounds_rate", 0)),
                    "mean_final_range_m": per_scenario.get("mean_final_range_m", m.get("mean_final_range_m", 0)),
                    "reason": "N/A",
                })
    return rows


def render_probe_summary(rows: list, complete: bool, failed_probes: list) -> str:
    import pandas as pd
    df = pd.DataFrame(rows)
    lines = []
    lines.append("# Stage 6G Guidance-Law Limitation Probe Summary")
    lines.append("")
    lines.append(f"**Complete**: {complete}")
    lines.append(f"**Failed Probes**: {len(failed_probes)}")
    if failed_probes:
        lines.append(f"**Failed List**: {failed_probes}")
    lines.append("")

    if df.empty:
        lines.append("No probe results found.")
        return "\n".join(lines)

    # Pivot: guidance_mode x scenario x method -> success_rate
    for scenario in sorted(df["scenario"].unique()):
        lines.append(f"## Scenario: {scenario}")
        lines.append("")
        sdf = df[df["scenario"] == scenario]
        lines.append("| Guidance Mode | Method | Success Rate | Mean Return | Crash Rate | OOB Rate | Final Range (m) |")
        lines.append("|---------------|--------|-------------:|------------:|-----------:|---------:|----------------:|")
        for _, row in sdf.iterrows():
            lines.append(
                f"| {row['guidance_mode']} | {row['method']} | {row['success_rate']:.1%} | "
                f"{row['mean_return']:.1f} | {row['crash_rate']:.1%} | {row['out_of_bounds_rate']:.1%} | "
                f"{row['mean_final_range_m']:.1f} |"
            )
        lines.append("")

    # Guidance comparison conclusion
    lines.append("## Conclusion")
    lines.append("")
    for scenario in sorted(df["scenario"].unique()):
        sdf = df[df["scenario"] == scenario]
        los_sr = sdf[sdf["guidance_mode"] == "los_rate"]["success_rate"].mean()
        pn_sr = sdf[sdf["guidance_mode"] == "proportional_navigation"]["success_rate"].mean()
        hybrid_sr = sdf[sdf["guidance_mode"] == "hybrid"]["success_rate"].mean()
        lines.append(f"- **{scenario}**: LOS-rate SR={los_sr:.1%}, PN SR={pn_sr:.1%}, Hybrid SR={hybrid_sr:.1%}")
    lines.append("")

    # Determine if dead zone is guidance-specific
    all_los_zero = True
    any_pn_nonzero = False
    any_hybrid_nonzero = False
    for scenario in sorted(df["scenario"].unique()):
        sdf = df[df["scenario"] == scenario]
        los_sr = sdf[sdf["guidance_mode"] == "los_rate"]["success_rate"].mean()
        pn_sr = sdf[sdf["guidance_mode"] == "proportional_navigation"]["success_rate"].mean()
        hybrid_sr = sdf[sdf["guidance_mode"] == "hybrid"]["success_rate"].mean()
        if los_sr > 0:
            all_los_zero = False
        if pn_sr > 0:
            any_pn_nonzero = True
        if hybrid_sr > 0:
            any_hybrid_nonzero = True

    if all_los_zero and not any_pn_nonzero and not any_hybrid_nonzero:
        lines.append(
            "**Interpretation**: All guidance modes show 0% success in dead-zone scenarios. "
            "This suggests the tail-chase dead zone is a **VPP formulation / pursuit geometry limitation**, "
            "not specific to LOS-rate guidance."
        )
    elif all_los_zero and (any_pn_nonzero or any_hybrid_nonzero):
        lines.append(
            "**Interpretation**: Alternative guidance laws (PN/hybrid) achieve non-zero success where LOS-rate fails. "
            "This suggests the tail-chase dead zone is primarily a **LOS-rate guidance limitation**."
        )
    else:
        lines.append(
            "**Interpretation**: Results are mixed. Some dead-zone scenarios show partial success under LOS-rate, "
            "indicating scenario-dependent behavior rather than a universal structural limitation."
        )
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Stage 6G Guidance-Law Limitation Probe")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Allow aggregation even if some probes fail")
    parser.add_argument("--episodes-per-scenario", type=int, default=10)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    print("Stage 6G Guidance-Law Limitation Probe")
    print(f"Guidance modes: {GUIDANCE_MODES}")
    print(f"Scenarios: {list(SCENARIO_CONFIGS.keys())}")
    print(f"Methods: {[m['name'] for m in METHODS]}")
    print(f"Episodes per scenario: {args.episodes_per_scenario}")
    print(f"Eval seeds: {args.eval_seeds}")
    print(f"Allow incomplete: {args.allow_incomplete}")
    print(f"Dry-run: {args.dry_run}")
    print("")

    overall_ok = []
    failed_probes = []
    for guidance_mode in GUIDANCE_MODES:
        for scenario_name, sc_cfg in SCENARIO_CONFIGS.items():
            ok = run_probe(
                guidance_mode=guidance_mode,
                scenario_name=scenario_name,
                base_config_path=sc_cfg["base_config"],
                episodes_per_scenario=args.episodes_per_scenario,
                eval_seeds=args.eval_seeds,
                dry_run=args.dry_run,
            )
            overall_ok.append(ok)
            if not ok:
                failed_probes.append(f"{guidance_mode}_{scenario_name}")

    if args.dry_run:
        print("\n[DRY-RUN] All probe commands prepared.")
        return

    complete = all(overall_ok)
    if not complete:
        print(f"\nFAILED PROBES ({len(failed_probes)}): {failed_probes}")
        if not args.allow_incomplete:
            print("ERROR: Use --allow-incomplete to generate partial summary, or fix failures.")
            sys.exit(1)
        print("WARNING: --allow-incomplete set; generating partial summary.")

    # Aggregate and save summary
    rows = aggregate_results()
    summary_md = render_probe_summary(rows, complete=complete, failed_probes=failed_probes)
    summary_path = os.path.join(OUTPUT_ROOT, "guidance_probe.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_md)
    print(f"\nProbe summary saved to: {summary_path}")

    # Save CSV
    import pandas as pd
    df = pd.DataFrame(rows)
    if not df.empty:
        csv_path = os.path.join(OUTPUT_ROOT, "guidance_probe_summary.csv")
        df.to_csv(csv_path, index=False, float_format="%.6f")
        print(f"Probe CSV saved to: {csv_path}")

        # Save failure reason breakdown
        failure_rows = []
        for guidance_mode in GUIDANCE_MODES:
            for scenario_name in SCENARIO_CONFIGS:
                probe_dir = Path(OUTPUT_ROOT) / f"{guidance_mode}_{scenario_name}"
                metrics_json = probe_dir / "prediction_metrics.json"
                if not metrics_json.exists():
                    continue
                with open(metrics_json, "r", encoding="utf-8") as f:
                    methods_data = json.load(f)
                for m in methods_data:
                    raw_eps = m.get("raw_episodes", [])
                    for ep in raw_eps:
                        if not ep.get("is_success", False):
                            failure_rows.append({
                                "guidance_mode": guidance_mode,
                                "scenario": scenario_name,
                                "method": m.get("method_name", m.get("method", "unknown")),
                                "reason": ep.get("reason", "unknown"),
                                "return": ep.get("return", 0),
                                "final_range_m": ep.get("final_range_m", 0),
                            })
        if failure_rows:
            fail_df = pd.DataFrame(failure_rows)
            fail_csv = os.path.join(OUTPUT_ROOT, "failure_reason_by_guidance.csv")
            fail_df.to_csv(fail_csv, index=False, float_format="%.6f")
            print(f"Failure breakdown CSV saved to: {fail_csv}")


if __name__ == "__main__":
    main()
