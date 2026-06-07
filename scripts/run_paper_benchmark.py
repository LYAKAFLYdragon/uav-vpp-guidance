#!/usr/bin/env python3
"""Stage 8C: Paper-safe experiment readiness benchmark.

Evaluates all methods and generates:
- summary.md: Full text report with statistical comparison and reproducibility metadata
- results.csv: Raw data
- figures/*.png: Paper figures
- tables/*.md: Markdown tables
"""

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.envs.scenario_registry import (
    ScenarioRegistry,
    initialize_canonical_scenarios,
)
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.evaluation.evaluate_prediction_comparison import (
    evaluate_single_episode,
)
from uav_vpp_guidance.evaluation.statistical_comparison import paired_t_test, cohens_d
from uav_vpp_guidance.guidance.gain_config import GuidanceGains


METHODS = {
    "no_prediction": {
        "checkpoint": "outputs/audit_no_pred_final/checkpoints/best.pt",
        "config_method": "no_prediction",
    },
    "cv_prediction": {
        "checkpoint": "outputs/audit_cv_final/checkpoints/best.pt",
        "config_method": "cv_prediction",
    },
    "ca_prediction": {
        "checkpoint": "outputs/audit_ca_final/checkpoints/best.pt",
        "config_method": "ca_prediction",
    },
    "gain_only": {
        "checkpoint": "outputs/audit_no_pred_final/checkpoints/best.pt",
        "config_method": "no_prediction",
        "gains_path": "outputs/gain_only_cem/cem_results.json",
        "note": "Same policy as no_prediction but with CEM-optimized gains",
    },
}


def _config_hash(config: dict) -> str:
    """Compute a simple hash of the resolved config for reproducibility tracking."""
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def load_config(config_path: str, method_name: str) -> dict:
    full_config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    method_override = full_config.get("methods", {}).get(method_name, {})
    base_config = copy.deepcopy(full_config)
    for k, v in method_override.items():
        if isinstance(v, dict) and k in base_config and isinstance(base_config[k], dict):
            base_config[k].update(copy.deepcopy(v))
        else:
            base_config[k] = copy.deepcopy(v)
    return base_config


def _load_gain_only_gains(method_cfg: dict, allow_random_smoke: bool) -> dict:
    """Load CEM-optimized gains for gain_only method."""
    gains_path = method_cfg.get("gains_path")
    if not gains_path:
        raise FileNotFoundError(
            f"gains_path not configured for gain_only. "
            f"Use --allow-random-smoke to proceed without gains."
        )
    gains_file = Path(gains_path)
    if not gains_file.exists():
        if not allow_random_smoke:
            raise FileNotFoundError(
                f"Gains file not found for gain_only: {gains_path}. "
                f"Use --allow-random-smoke to proceed without gains."
            )
        print(f"WARNING: Gains file not found for gain_only: {gains_path}")
        return {}
    data = json.loads(gains_file.read_text(encoding="utf-8"))
    best = data.get("best_gains", {})
    if not best:
        print(f"WARNING: No best_gains found in {gains_path}")
    return best


def evaluate_method(
    method_name: str,
    method_cfg: dict,
    scenarios: list,
    seeds: tuple,
    backend: str,
    config_path: str,
    allow_random_smoke: bool = False,
) -> dict:
    """Evaluate a single method across all scenarios and seeds."""
    config = load_config(config_path, method_cfg["config_method"])
    config["backend"] = backend
    if "env" not in config:
        config["env"] = {}
    config["env"]["backend"] = backend
    config["env"]["use_jsbsim"] = backend == "jsbsim"

    # Disable mode-switch for clean comparison
    if "guidance" not in config:
        config["guidance"] = {}
    if "mode_switch" not in config["guidance"]:
        config["guidance"]["mode_switch"] = {}
    config["guidance"]["mode_switch"]["enabled"] = False

    # gain_only: load and apply CEM-optimized gains
    loaded_gains = {}
    gains_exists = False
    if method_name == "gain_only":
        loaded_gains = _load_gain_only_gains(method_cfg, allow_random_smoke)
        gains_exists = bool(loaded_gains)
        if loaded_gains and "guidance" in config:
            if "gains" not in config["guidance"]:
                config["guidance"]["gains"] = {}
            config["guidance"]["gains"].update(copy.deepcopy(loaded_gains))

    env = CloseRangeTrackingEnv(config)
    obs = env.reset(seed=0)
    obs_dim = int(obs["observation_vector"].shape[0])

    agent = PPOAgent(obs_dim=obs_dim, action_dim=3, config=config, device="cpu")
    ckpt_path = method_cfg["checkpoint"]
    ckpt_exists = Path(ckpt_path).exists()
    if ckpt_exists:
        agent.load(ckpt_path)
    else:
        if not allow_random_smoke:
            raise FileNotFoundError(
                f"Checkpoint not found for {method_name}: {ckpt_path}. "
                f"Use --allow-random-smoke to proceed with random policy."
            )
        print(
            f"WARNING: Checkpoint not found for {method_name}: {ckpt_path}"
        )
        print("Using random policy (results marked as invalid_for_paper)")

    episodes = []
    for scen in scenarios:
        for seed in seeds:
            result, _ = evaluate_single_episode(
                env=env,
                agent=agent,
                config=config,
                scenario=scen,
                seed=seed,
                save_trajectory=False,
                method_name=method_name,
            )
            result["method"] = method_name
            result["scenario"] = scen.get("name", "unknown")
            episodes.append(result)

    env.close()

    # Method metadata
    invalid_for_paper = not ckpt_exists or (method_name == "gain_only" and not gains_exists)
    metadata = {
        "method": method_name,
        "config_path": config_path,
        "resolved_config_hash": _config_hash(config),
        "method_override_name": method_cfg["config_method"],
        "backend": backend,
        "scenarios": [s.get("name", "unknown") for s in scenarios],
        "seeds": list(seeds),
        "prediction_mode": method_cfg["config_method"],
        "guidance_mode": config.get("guidance", {}).get("mode", "unknown"),
        "gain_source": "cem" if method_name == "gain_only" else "default",
        "policy_checkpoint": ckpt_path,
        "checkpoint_exists": ckpt_exists,
        "gains_path": method_cfg.get("gains_path"),
        "gains_exists": gains_exists,
        "loaded_gains": loaded_gains,
        "is_random_smoke": not ckpt_exists,
        "invalid_for_paper": invalid_for_paper,
        "note": method_cfg.get("note", ""),
    }

    return {
        "method": method_name,
        "episodes": episodes,
        "metadata": metadata,
    }


def serialize(obj):
    """Serialize numpy types for JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return bool(obj)
    if isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [serialize(v) for v in obj]
    return obj


def _git_commit_hash() -> str:
    """Get current git commit hash, or 'unknown' if not available."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def generate_figures(results: list, output_dir: Path):
    """Generate paper figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    methods = []
    means = []
    stds = []
    for r in results:
        method = r["method"]
        returns = [ep.get("return", 0) for ep in r["episodes"]]
        methods.append(method)
        means.append(np.mean(returns) if returns else 0)
        stds.append(np.std(returns, ddof=1) if len(returns) > 1 else 0)

    x = np.arange(len(methods))
    ax.bar(x, means, yerr=stds, capsize=5, color="steelblue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylabel("Mean Return")
    ax.set_title("Method Comparison (Mean Return ± Std)")
    ax.grid(axis="y", alpha=0.3)

    fig_path = output_dir / "figures" / "figure1_method_comparison.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {fig_path}")

    return [fig_path]


def generate_tables(results: list, output_dir: Path):
    """Generate Markdown and CSV tables."""
    rows = []
    baseline = None
    for r in results:
        method = r["method"]
        episodes = r["episodes"]
        returns = [ep.get("return", 0) for ep in episodes]
        successes = sum(1 for ep in episodes if ep.get("is_success", False))
        total = len(episodes)

        sr = successes / total if total > 0 else 0
        mean_ret = np.mean(returns) if returns else 0
        std_ret = np.std(returns, ddof=1) if len(returns) > 1 else 0

        row = {
            "Method": method,
            "Success Rate": f"{sr:.2%}",
            "Mean Return": f"{mean_ret:.2f} ± {std_ret:.2f}",
            "N Episodes": total,
        }

        if baseline is not None and method != baseline["method"]:
            method_by_seed = {
                (ep.get("scenario"), ep.get("seed")): ep.get("return", 0)
                for ep in episodes
            }
            baseline_by_seed = {
                (ep.get("scenario"), ep.get("seed")): ep.get("return", 0)
                for ep in baseline["episodes"]
            }
            common_keys = sorted(
                set(method_by_seed.keys()) & set(baseline_by_seed.keys())
            )
            if common_keys:
                a_vals = [baseline_by_seed[k] for k in common_keys]
                b_vals = [method_by_seed[k] for k in common_keys]
                ttest = paired_t_test(a_vals, b_vals)
                d = cohens_d(a_vals, b_vals)
                sig = "*" if ttest["significant_at_05"] else ""
                row["vs Baseline p"] = f"{ttest['p_value']:.4f}{sig}"
                row["Cohen's d"] = f"{d['d']:.3f} ({d['magnitude']})"

        rows.append(row)
        if baseline is None:
            baseline = r

    df = pd.DataFrame(rows)
    md_path = output_dir / "tables" / "comparison_table.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)

    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    separator = "|" + "|".join([" --- " for _ in cols]) + "|"
    lines = [header, separator]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {md_path}")

    csv_path = output_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    return md_path, csv_path


def generate_summary(results: list, output_dir: Path, backend: str, args: argparse.Namespace):
    """Generate summary.md report with full reproducibility metadata."""
    commit_hash = _git_commit_hash()
    lines = [
        "# UAV VPP Guidance — Paper Benchmark Report",
        "",
        f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Git Commit**: `{commit_hash}`",
        f"**Backend**: {backend}",
        f"**Config**: {args.config}",
        f"**Methods**: {', '.join(r['method'] for r in results)}",
        f"**Scenarios**: {args.scenarios}",
        f"**Seeds**: {args.seeds}",
        f"**Allow Random Smoke**: {args.allow_random_smoke}",
        "",
        "## Benchmark Type",
        "",
    ]

    any_invalid = any(r["metadata"]["invalid_for_paper"] for r in results)
    if any_invalid:
        lines.append("- ⚠️ **SMOKE BENCHMARK**: At least one method uses random policy or missing gains.")
        lines.append("- **NOT PAPER-SAFE**: Do not use these results for paper claims.")
    else:
        lines.append("- ✅ **PAPER-SAFE BENCHMARK**: All checkpoints and gains loaded successfully.")
    lines.append("")

    lines.append("## Results Summary")
    lines.append("")

    for r in results:
        method = r["method"]
        meta = r["metadata"]
        episodes = r["episodes"]
        successes = sum(1 for ep in episodes if ep.get("is_success", False))
        total = len(episodes)
        sr = successes / total if total > 0 else 0
        lines.append(f"### {method}")
        lines.append(f"- Success Rate: {sr:.2%} ({successes}/{total})")
        lines.append(f"- Config Path: {meta['config_path']}")
        lines.append(f"- Resolved Config Hash: {meta['resolved_config_hash']}")
        lines.append(f"- Method Override: {meta['method_override_name']}")
        lines.append(f"- Prediction Mode: {meta['prediction_mode']}")
        lines.append(f"- Guidance Mode: {meta['guidance_mode']}")
        lines.append(f"- Gain Source: {meta['gain_source']}")
        lines.append(f"- Policy Checkpoint: {meta['policy_checkpoint']}")
        lines.append(f"- Checkpoint Exists: {meta['checkpoint_exists']}")
        if meta.get("gains_path"):
            lines.append(f"- Gains Path: {meta['gains_path']}")
            lines.append(f"- Gains Exists: {meta['gains_exists']}")
            if meta.get("loaded_gains"):
                lines.append(f"- Loaded Gains: {meta['loaded_gains']}")
        if meta['invalid_for_paper']:
            lines.append("- ⚠️ **INVALID FOR PAPER**: Using random policy or missing gains")
        if meta.get('note'):
            lines.append(f"- Note: {meta['note']}")
        lines.append("")

    lines.extend(
        [
            "## Statistical Comparison",
            "See `tables/comparison_table.md` for paired t-test and Cohen's d.",
            "",
            "## Figures",
            "See `figures/` directory.",
            "",
            "## Reproducibility",
            "```bash",
            f"python scripts/run_paper_benchmark.py \\",
            f"  --config {args.config} \\",
            f"  --backend {backend} \\",
            f"  --scenarios {args.scenarios} \\",
            f"  --seeds {' '.join(str(s) for s in args.seeds)}",
            "```",
        ]
    )

    summary_path = output_dir / "summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {summary_path}")

    return summary_path


def main():
    parser = argparse.ArgumentParser(description="Paper benchmark")
    parser.add_argument(
        "--config",
        type=str,
        default="config/experiment/stage6f5_feasible_geometry.yaml",
    )
    parser.add_argument(
        "--backend", type=str, default="simple", choices=["simple", "jsbsim"]
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default="regression",
        choices=["regression", "candidate", "all"],
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=None,
        help="Methods to evaluate (default: all)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/paper_benchmark"
    )
    parser.add_argument(
        "--allow-random-smoke",
        action="store_true",
        help="Allow evaluation with missing checkpoints or gains. Results will be marked invalid_for_paper.",
    )
    parser.add_argument(
        "--allow-missing-methods",
        action="store_true",
        help="Skip unknown methods instead of raising an error.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    initialize_canonical_scenarios()

    if args.scenarios == "regression":
        scenarios = ScenarioRegistry.get_regression_suite()
    elif args.scenarios == "candidate":
        scenarios = ScenarioRegistry.get_candidate_suite()
    else:
        scenarios = (
            ScenarioRegistry.get_regression_suite()
            + ScenarioRegistry.get_candidate_suite()
        )

    methods_to_run = args.methods or list(METHODS.keys())

    # Validate that gain_only and no_prediction are semantically different
    if "gain_only" in methods_to_run and "no_prediction" in methods_to_run:
        gain_cfg = METHODS["gain_only"]
        no_pred_cfg = METHODS["no_prediction"]
        # They must differ in at least gains_path or note
        if gain_cfg.get("gains_path") == no_pred_cfg.get("gains_path") and not gain_cfg.get("note"):
            raise ValueError(
                "gain_only and no_prediction must have distinct configuration. "
                "gain_only requires a gains_path or note to differentiate from no_prediction."
            )

    results = []
    for method_name in methods_to_run:
        if method_name not in METHODS:
            msg = f"Unknown method '{method_name}'. Available: {list(METHODS.keys())}"
            if args.allow_missing_methods:
                print(f"WARNING: {msg}, skipping")
                continue
            raise ValueError(msg)
        print(f"\n{'='*50}")
        print(f"Evaluating: {method_name}")
        print(f"{'='*50}")
        result = evaluate_method(
            method_name,
            METHODS[method_name],
            scenarios,
            tuple(args.seeds),
            args.backend,
            config_path=args.config,
            allow_random_smoke=args.allow_random_smoke,
        )
        results.append(result)
        sr = sum(1 for ep in result["episodes"] if ep.get("is_success", False)) / max(
            1, len(result["episodes"])
        )
        print(f"Success Rate: {sr:.2%}")

    # Generate outputs
    print(f"\n{'='*50}")
    print("Generating figures and tables...")
    generate_figures(results, output_dir)
    generate_tables(results, output_dir)
    generate_summary(results, output_dir, args.backend, args)

    print(f"\n{'='*50}")
    print("Benchmark Complete!")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
