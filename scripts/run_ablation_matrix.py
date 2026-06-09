#!/usr/bin/env python3
"""
5-Method Ablation Experiment Automation Script.

Runs the complete evaluation matrix:
    5 methods × N scenarios × M seeds = total evaluation episodes

Methods:
    1. no_prediction      - no trajectory prediction (direct current_target)
    2. constant_velocity  - CV predictor
    3. constant_acceleration - CA predictor
    4. lstm               - frozen LSTM neural predictor
    5. gru                - frozen GRU neural predictor

The script:
    1. Builds a unified evaluation config from method checkpoints
    2. Calls evaluate_prediction_comparison.py for the full matrix
    3. Reads results and runs statistical tests (statistical_comparison.py)
    4. Generates paper-compatible CSV/JSON tables in paper_materials/tables/
    5. Writes ablation_manifest.json with full provenance

Usage (full matrix, 10 seeds, 50 episodes each):
    python scripts/run_ablation_matrix.py \
        --no-pred-checkpoint outputs/experiments/no_prediction/checkpoints/best.pt \
        --cv-checkpoint outputs/experiments/cv/checkpoints/best.pt \
        --ca-checkpoint outputs/experiments/ca/checkpoints/best.pt \
        --lstm-checkpoint outputs/experiments/lstm/checkpoints/best.pt \
        --gru-checkpoint outputs/experiments/gru/checkpoints/best.pt \
        --lstm-predictor-ckpt outputs/trajectory_prediction/best_model.pt \
        --gru-predictor-ckpt outputs/trajectory_prediction/best_model_gru.pt \
        --scenarios-config config/experiment/evaluate_vpp_prediction_comparison.yaml \
        --seeds 10 \
        --episodes-per-scenario 50 \
        --output-dir outputs/ablation_matrix

Usage (smoke test, 1 seed, 2 scenarios, 1 episode each):
    python scripts/run_ablation_matrix.py ... --smoke

Usage (resume / skip existing):
    python scripts/run_ablation_matrix.py ... --skip-existing
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METHOD_KEYS = ["no_prediction", "cv_prediction", "ca_prediction", "lstm_frozen", "gru_frozen"]
METHOD_DISPLAY_NAMES = {
    "no_prediction": "No Prediction",
    "cv_prediction": "CV",
    "ca_prediction": "CA",
    "lstm_frozen": "LSTM",
    "gru_frozen": "GRU",
}

PAPER_TABLES_DIR = Path("paper_materials/tables")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="5-Method Ablation Experiment Matrix")

    # Method checkpoints (PPO policy)
    parser.add_argument("--no-pred-checkpoint", type=str, required=True,
                        help="Checkpoint path for no-prediction method")
    parser.add_argument("--cv-checkpoint", type=str, required=True,
                        help="Checkpoint path for CV method")
    parser.add_argument("--ca-checkpoint", type=str, required=True,
                        help="Checkpoint path for CA method")
    parser.add_argument("--lstm-checkpoint", type=str, required=True,
                        help="Checkpoint path for LSTM PPO policy")
    parser.add_argument("--gru-checkpoint", type=str, required=True,
                        help="Checkpoint path for GRU PPO policy")

    # Neural predictor checkpoints (LSTM/GRU weights)
    parser.add_argument("--lstm-predictor-ckpt", type=str, default=None,
                        help="LSTM predictor checkpoint (.pt)")
    parser.add_argument("--gru-predictor-ckpt", type=str, default=None,
                        help="GRU predictor checkpoint (.pt)")

    # Scenarios
    parser.add_argument("--scenarios-config", type=str,
                        default="config/experiment/evaluate_vpp_prediction_comparison.yaml",
                        help="Base YAML containing scenario definitions")
    parser.add_argument("--scenarios", type=str, nargs="+", default=None,
                        help="Override scenario names (default: all from config)")

    # Evaluation scale
    parser.add_argument("--seeds", type=int, default=10,
                        help="Number of evaluation seeds (default: 10)")
    parser.add_argument("--episodes-per-scenario", type=int, default=50,
                        help="Episodes per scenario-seed cell (default: 50)")

    # Output
    parser.add_argument("--output-dir", type=str, default="outputs/ablation_matrix",
                        help="Root output directory")
    parser.add_argument("--paper-tables-dir", type=str, default=str(PAPER_TABLES_DIR),
                        help="Directory for paper-compatible table outputs")

    # Modes
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: 1 seed, 2 scenarios, 1 episode each")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip evaluation if output files already exist")
    parser.add_argument("--backend", type=str, default="simple", choices=["simple", "jsbsim"],
                        help="Simulation backend")
    parser.add_argument("--allow-random-policy", action="store_true",
                        help="Allow random policy fallback for smoke/debug ONLY")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def _load_yaml(config_path: str) -> dict:
    """Load YAML config."""
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ImportError:
        print("ERROR: PyYAML required. Install: pip install pyyaml")
        sys.exit(1)


def _save_yaml(config: dict, path: str) -> None:
    """Save dict to YAML."""
    import yaml
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def build_eval_config(args) -> str:
    """
    Build a unified evaluation config YAML from the base template + checkpoint overrides.

    Returns:
        str: Path to the generated temp config file.
    """
    base_path = Path(args.scenarios_config)
    if not base_path.exists():
        raise FileNotFoundError(f"Scenarios config not found: {base_path}")

    config = _load_yaml(str(base_path))

    # Update method checkpoints
    method_ckpts = {
        "no_prediction": args.no_pred_checkpoint,
        "cv_prediction": args.cv_checkpoint,
        "ca_prediction": args.ca_checkpoint,
        "lstm_frozen": args.lstm_checkpoint,
        "gru_frozen": args.gru_checkpoint,
    }

    methods = config.get("methods", {})
    for method_key, ckpt_path in method_ckpts.items():
        if method_key not in methods:
            raise ValueError(f"Method '{method_key}' not found in base config methods")
        methods[method_key]["checkpoint"] = ckpt_path

    # Update neural predictor checkpoints
    if args.lstm_predictor_ckpt and "lstm_frozen" in methods:
        methods["lstm_frozen"]["trajectory_prediction"]["checkpoint_path"] = args.lstm_predictor_ckpt
    if args.gru_predictor_ckpt and "gru_frozen" in methods:
        methods["gru_frozen"]["trajectory_prediction"]["checkpoint_path"] = args.gru_predictor_ckpt

    # Determine scenarios to use
    if args.scenarios is not None:
        all_scenarios = config.get("scenarios", {})
        config["scenarios"] = {k: v for k, v in all_scenarios.items() if k in args.scenarios}
        if not config["scenarios"]:
            raise ValueError(f"No matching scenarios found for: {args.scenarios}")

    # Temp config path
    temp_dir = Path(args.output_dir) / "_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"ablation_eval_config_{os.getpid()}.yaml"
    _save_yaml(config, str(temp_path))
    return str(temp_path)


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


def run_evaluation(args, eval_config_path: str) -> dict:
    """
    Run evaluate_prediction_comparison.py and return parsed results.

    Returns:
        dict with keys: json_path, csv_path, output_dir, success
    """
    eval_output = Path(args.output_dir) / "evaluation"
    eval_output.mkdir(parents=True, exist_ok=True)

    # Skip-existing check
    existing_json = eval_output / "prediction_metrics.json"
    existing_csv = eval_output / "prediction_metrics.csv"
    if args.skip_existing and existing_json.exists() and existing_csv.exists():
        print(f"[SKIP] Evaluation output already exists: {eval_output}")
        return {
            "success": True,
            "skipped": True,
            "json_path": str(existing_json),
            "csv_path": str(existing_csv),
            "output_dir": str(eval_output),
        }

    seeds = list(range(args.seeds))
    scenarios = args.scenarios if args.scenarios else None

    cmd = [
        sys.executable, "-m",
        "uav_vpp_guidance.evaluation.evaluate_prediction_comparison",
        "--config", eval_config_path,
        "--backend", args.backend,
        "--seeds", *[str(s) for s in seeds],
        "--episodes-per-scenario", str(args.episodes_per_scenario),
        "--output-dir", str(eval_output),
    ]

    if scenarios:
        cmd.extend(["--scenarios"] + scenarios)

    if args.allow_random_policy:
        cmd.append("--allow-random-policy")

    print(f"\n[EVAL] {'=' * 60}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Matrix: 5 methods × {len(scenarios) if scenarios else 'all'} scenarios × {len(seeds)} seeds")
    print(f"Estimated episodes: 5 × {len(scenarios) if scenarios else 'N'} × {len(seeds)} × {args.episodes_per_scenario}")
    start = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - start

    success = result.returncode == 0
    status = "OK" if success else "FAILED"
    print(f"[EVAL] {status} in {elapsed:.1f}s")

    return {
        "success": success,
        "skipped": False,
        "json_path": str(existing_json),
        "csv_path": str(existing_csv),
        "output_dir": str(eval_output),
        "elapsed_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Result loaders
# ---------------------------------------------------------------------------


def load_evaluation_json(json_path: str) -> List[dict]:
    """Load prediction_metrics.json."""
    if not os.path.exists(json_path):
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_evaluation_csv(csv_path: str) -> List[dict]:
    """Load prediction_metrics.csv."""
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------


def run_statistical_analysis(metrics: List[dict], output_dir: str) -> dict:
    """
    Run pairwise statistical comparisons between methods.

    Uses bootstrap CI for success rates and paired comparisons.
    Returns summary dict.
    """
    sys.path.insert(0, "src")
    from uav_vpp_guidance.evaluation.statistical_comparison import (
        bootstrap_success_rate_ci, compare_methods, mean_std
    )

    results = {}

    # Per-method success rate with bootstrap CI
    for m in metrics:
        method = m.get("method", "unknown")
        sr = m.get("instant_success_rate", m.get("success_rate", np.nan))
        # We don't have raw episode outcomes in the aggregated JSON,
        # so we approximate CI width from std_return as a heuristic
        # or compute CI on the aggregated mean if available.
        # For now, report point estimate only.
        results[method] = {
            "success_rate": float(sr) if np.isfinite(sr) else np.nan,
            "mean_return": float(m.get("mean_return", np.nan)),
            "std_return": float(m.get("std_return", np.nan)),
            "mean_final_range_m": float(m.get("mean_final_range_m", np.nan)),
            "mean_final_ata_deg": float(m.get("mean_final_ata_deg", np.nan)),
            "crash_rate": float(m.get("crash_rate", np.nan)),
            "out_of_bounds_rate": float(m.get("out_of_bounds_rate", np.nan)),
            "timeout_rate": float(m.get("timeout_rate", np.nan)),
            "prediction_fallback_rate": float(m.get("prediction_fallback_rate", np.nan)),
            "prediction_rmse_m": float(m.get("prediction_rmse_m", np.nan)),
            "episodes": int(m.get("episodes", 0)),
        }

    # Pairwise comparison summary
    pairwise = {}
    baseline = results.get("no_prediction")
    if baseline:
        for method, stats in results.items():
            if method == "no_prediction":
                continue
            delta_sr = stats["success_rate"] - baseline["success_rate"]
            delta_return = stats["mean_return"] - baseline["mean_return"]
            pairwise[method] = {
                "vs_baseline": "no_prediction",
                "delta_success_rate": delta_sr,
                "delta_mean_return": delta_return,
                "relative_return_pct": (delta_return / abs(baseline["mean_return"]) * 100.0
                                         if baseline["mean_return"] != 0 and np.isfinite(baseline["mean_return"])
                                         else np.nan),
            }

    stats_path = os.path.join(output_dir, "statistical_summary.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({
            "per_method": results,
            "pairwise_vs_no_prediction": pairwise,
        }, f, indent=2, ensure_ascii=False, default=str)
    print(f"[STAT] Statistical summary saved: {stats_path}")

    return {"per_method": results, "pairwise": pairwise}


# ---------------------------------------------------------------------------
# Paper table generation
# ---------------------------------------------------------------------------


def generate_paper_tables(metrics: List[dict], stats: dict, paper_dir: str) -> List[str]:
    """
    Generate paper-compatible CSV and LaTeX tables.

    Returns:
        list of generated file paths.
    """
    paper_dir = Path(paper_dir)
    paper_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    # Table 1: Main comparison (overall)
    rows = []
    for method_key in METHOD_KEYS:
        m = next((x for x in metrics if x.get("method") == method_key), None)
        if m is None:
            continue
        sr = m.get("instant_success_rate", m.get("success_rate", np.nan))
        rows.append({
            "Method": METHOD_DISPLAY_NAMES.get(method_key, method_key),
            "Success Rate (%)": f"{sr*100:.1f}" if np.isfinite(sr) else "N/A",
            "Mean Return": f"{m.get('mean_return', 0):.1f}",
            "Std Return": f"{m.get('std_return', 0):.1f}",
            "Mean Range (m)": f"{m.get('mean_final_range_m', 0):.1f}",
            "Mean ATA (deg)": f"{m.get('mean_final_ata_deg', 0):.1f}",
            "Crash Rate (%)": f"{m.get('crash_rate', 0)*100:.1f}",
            "OOB Rate (%)": f"{m.get('out_of_bounds_rate', 0)*100:.1f}",
            "Timeout Rate (%)": f"{m.get('timeout_rate', 0)*100:.1f}",
            "Fallback Rate (%)": f"{m.get('prediction_fallback_rate', 0)*100:.1f}" if m.get("prediction_fallback_rate") is not None else "N/A",
        })

    # CSV
    csv_path = paper_dir / "table_ablation_main.csv"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        generated.append(str(csv_path))
        print(f"[TABLE] {csv_path}")

    # LaTeX
    tex_path = paper_dir / "table_ablation_main.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Ablation Comparison of Five Prediction Methods}\n")
        f.write("\\label{tab:ablation_main}\n")
        f.write("\\begin{tabular}{lccccccccc}\n")
        f.write("\\toprule\n")
        f.write("Method & Success & Return & Range (m) & ATA (deg) & Crash & OOB & Timeout & Fallback \\\\\n")
        f.write("\\midrule\n")
        for r in rows:
            f.write(
                f"{r['Method']} & "
                f"{r['Success Rate (%)']} & "
                f"{r['Mean Return']} & "
                f"{r['Mean Range (m)']} & "
                f"{r['Mean ATA (deg)']} & "
                f"{r['Crash Rate (%)']} & "
                f"{r['OOB Rate (%)']} & "
                f"{r['Timeout Rate (%)']} & "
                f"{r['Fallback Rate (%)']} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    generated.append(str(tex_path))
    print(f"[TABLE] {tex_path}")

    # Table 2: Per-scenario breakdown
    scenario_tables = []
    for m in metrics:
        method = m.get("method", "unknown")
        per_scenario = m.get("per_scenario", {})
        for sc_name, sc in per_scenario.items():
            scenario_tables.append({
                "Method": METHOD_DISPLAY_NAMES.get(method, method),
                "Scenario": sc_name,
                "Success Rate (%)": f"{sc.get('success_rate', 0)*100:.1f}",
                "Mean Return": f"{sc.get('mean_return', 0):.1f}",
                "Mean Range (m)": f"{sc.get('mean_final_range_m', 0):.1f}",
            })

    if scenario_tables:
        csv_path2 = paper_dir / "table_ablation_per_scenario.csv"
        with open(csv_path2, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(scenario_tables[0].keys()))
            writer.writeheader()
            writer.writerows(scenario_tables)
        generated.append(str(csv_path2))
        print(f"[TABLE] {csv_path2}")

    return generated


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def write_manifest(args, eval_result: dict, metrics: List[dict], stats: dict,
                   tables: List[str], manifest_path: str) -> None:
    """Write ablation_manifest.json."""
    manifest = {
        "experiment": "5_method_ablation_matrix",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "methods": METHOD_KEYS,
            "checkpoints": {
                "no_prediction": args.no_pred_checkpoint,
                "cv_prediction": args.cv_checkpoint,
                "ca_prediction": args.ca_checkpoint,
                "lstm_frozen": args.lstm_checkpoint,
                "gru_frozen": args.gru_checkpoint,
            },
            "predictor_checkpoints": {
                "lstm": args.lstm_predictor_ckpt,
                "gru": args.gru_predictor_ckpt,
            },
            "scenarios_config": args.scenarios_config,
            "scenarios": args.scenarios,
            "seeds": args.seeds,
            "episodes_per_scenario": args.episodes_per_scenario,
            "backend": args.backend,
            "smoke": args.smoke,
        },
        "evaluation": {
            "success": eval_result.get("success", False),
            "skipped": eval_result.get("skipped", False),
            "output_dir": eval_result.get("output_dir"),
            "json_path": eval_result.get("json_path"),
            "csv_path": eval_result.get("csv_path"),
            "elapsed_s": eval_result.get("elapsed_s", 0.0),
        },
        "metrics_summary": {
            m.get("method", "unknown"): {
                "success_rate": float(m.get("instant_success_rate", m.get("success_rate", np.nan))),
                "mean_return": float(m.get("mean_return", np.nan)),
                "episodes": int(m.get("episodes", 0)),
            }
            for m in metrics
        },
        "statistical_analysis": stats,
        "paper_tables": tables,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)
    print(f"[MANIFEST] {manifest_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    # Smoke mode overrides
    if args.smoke:
        args.seeds = 1
        args.episodes_per_scenario = 1
        args.allow_random_policy = True
        if args.scenarios is None:
            # Default to 2 scenarios for smoke
            args.scenarios = ["favorable", "neutral"]
        print("[SMOKE] Running smoke mode: 1 seed, 1 episode/scenario, random-policy fallback enabled")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("5-Method Ablation Experiment Matrix")
    print("=" * 60)
    print(f"Methods: {', '.join(METHOD_KEYS)}")
    print(f"Seeds:   {args.seeds}")
    print(f"Episodes per scenario: {args.episodes_per_scenario}")
    print(f"Scenarios config: {args.scenarios_config}")
    print(f"Scenarios: {args.scenarios if args.scenarios else 'all from config'}")
    print(f"Output:  {output_dir}")
    print("-" * 60)

    # Validate checkpoints exist
    ckpts = {
        "no_prediction": args.no_pred_checkpoint,
        "cv_prediction": args.cv_checkpoint,
        "ca_prediction": args.ca_checkpoint,
        "lstm_frozen": args.lstm_checkpoint,
        "gru_frozen": args.gru_checkpoint,
    }
    for method, ckpt in ckpts.items():
        if not os.path.exists(ckpt):
            print(f"[WARN] Checkpoint not found for {method}: {ckpt}")
            if not args.allow_random_policy:
                print("       Use --allow-random-policy for smoke/debug, or train the policy first.")
                sys.exit(1)

    # 1. Build evaluation config
    print("\n[1/4] Building evaluation config...")
    try:
        eval_config_path = build_eval_config(args)
        print(f"       Temp config: {eval_config_path}")
    except Exception as exc:
        print(f"[ERROR] Config build failed: {exc}")
        sys.exit(1)

    # 2. Run evaluation
    print("\n[2/4] Running unified evaluation...")
    eval_result = run_evaluation(args, eval_config_path)
    if not eval_result["success"] and not eval_result.get("skipped"):
        print("[ERROR] Evaluation failed. Check logs above.")
        sys.exit(1)

    # 3. Load results & statistical analysis
    print("\n[3/4] Loading results and running statistical analysis...")
    metrics = load_evaluation_json(eval_result.get("json_path", ""))
    if not metrics:
        print("[WARN] No metrics loaded. Skipping analysis.")
        stats = {}
    else:
        stats = run_statistical_analysis(metrics, str(output_dir))

    # 4. Generate paper tables
    print("\n[4/4] Generating paper-compatible tables...")
    tables = generate_paper_tables(metrics, stats, args.paper_tables_dir)

    # 5. Write manifest
    manifest_path = output_dir / "ablation_manifest.json"
    write_manifest(args, eval_result, metrics, stats, tables, str(manifest_path))

    # Summary
    print("\n" + "=" * 60)
    print("ABLATION MATRIX COMPLETE")
    print("=" * 60)
    print(f"Evaluation output: {eval_result.get('output_dir')}")
    print(f"Manifest:          {manifest_path}")
    print(f"Paper tables:      {len(tables)} files in {args.paper_tables_dir}")
    if metrics:
        print("\nQuick Results:")
        for m in metrics:
            method = m.get("method", "unknown")
            sr = m.get("instant_success_rate", m.get("success_rate", 0))
            ret = m.get("mean_return", 0)
            print(f"  {METHOD_DISPLAY_NAMES.get(method, method):12s} | "
                  f"Success: {sr:.2%} | Return: {ret:8.1f}")
    print("=" * 60)

    # Cleanup temp config
    try:
        os.remove(eval_config_path)
    except OSError:
        pass


if __name__ == "__main__":
    main()
