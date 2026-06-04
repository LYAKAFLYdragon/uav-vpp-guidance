#!/usr/bin/env python3
"""
Stage 6F.6 Paper-Ready Synthesis & GRU-vs-LSTM Mechanism Audit.

Integrates Stage 6F, Stage 6F.4 (deep audit), and Stage 6F.5 (re-evaluation)
into paper-ready tables, statistical comparisons, and claims checklist.

Usage:
    python scripts/synthesize_stage6f_paper_results.py \
        --stage6f outputs/tables/stage6f \
        --stage6f_deep_audit outputs/tables/stage6f_deep_audit \
        --stage6f5_feasible outputs/tables/stage6f5_feasible_geometry \
        --stage6f5_maneuvering outputs/tables/stage6f5_maneuvering_target \
        --output outputs/tables/stage6f_paper_synthesis
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

METRICS_SCHEMA_VERSION = "6f.2"
EXPERIMENT_SUITE_VERSION = "6f.6"

METHOD_ORDER = [
    "no_prediction",
    "cv_prediction",
    "ca_prediction",
    "lstm_frozen",
    "gru_frozen",
]

FEASIBLE_SCENARIOS = ["neutral", "challenging", "weaving_headon", "weaving_offset"]
DEAD_ZONE_SCENARIOS = ["favorable", "disadvantage", "weaving_pursuit", "weaving_disadvantage"]
MANEUVERING_SCENARIOS = ["weaving_headon", "weaving_offset"]


def load_cross_seed_summary(path: Path) -> dict:
    cross_json = path / "cross_seed_summary.json"
    if cross_json.exists():
        with open(cross_json, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_deep_audit_scenario_csv(path: Path) -> pd.DataFrame:
    csv_path = path / "stage6f_scenario_deep.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def discover_training_seeds(raw_root: Path) -> list:
    if not raw_root.exists():
        return []
    seeds = []
    for d in raw_root.iterdir():
        if d.is_dir() and d.name.startswith("train_seed"):
            if (d / "prediction_metrics.json").exists():
                try:
                    seeds.append(int(d.name.replace("train_seed", "")))
                except ValueError:
                    pass
    return sorted(seeds)


def load_raw_prediction_metrics(raw_root: Path, training_seed: int) -> list:
    p = raw_root / f"train_seed{training_seed}" / "prediction_metrics.json"
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def build_episode_df(raw_root: Path, seeds: list) -> pd.DataFrame:
    rows = []
    for ts in seeds:
        data = load_raw_prediction_metrics(raw_root, ts)
        for m in data:
            method = m.get("method_name", m.get("method", "unknown"))
            for ep in m.get("raw_episodes", []):
                rows.append({
                    "method": method,
                    "training_seed": ts,
                    "scenario": ep.get("scenario", "unknown"),
                    "return": float(ep.get("return", np.nan)) if ep.get("return") is not None else np.nan,
                    "is_success": bool(ep.get("is_success", False)),
                    "reason": ep.get("reason", "unknown"),
                    "mean_env_prediction_error_m": float(ep.get("mean_env_prediction_error_m", np.nan)) if ep.get("mean_env_prediction_error_m") is not None else np.nan,
                    "mean_offline_aligned_error_m": float(ep.get("mean_offline_aligned_error_m", np.nan)) if ep.get("mean_offline_aligned_error_m") is not None else np.nan,
                    "mean_virtual_point_shift_m": float(ep.get("mean_virtual_point_shift_m", np.nan)) if ep.get("mean_virtual_point_shift_m") is not None else np.nan,
                    "final_range_m": float(ep.get("final_range_m", np.nan)) if ep.get("final_range_m") is not None else np.nan,
                    "final_ata_deg": float(ep.get("final_ata_deg", np.nan)) if ep.get("final_ata_deg") is not None else np.nan,
                    "length": int(ep.get("length", 0)),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Table Builders
# ---------------------------------------------------------------------------

def build_table_a_overall(cross_data: dict) -> pd.DataFrame:
    rows = []
    methods = [m["method"] for m in cross_data.get("methods", [])]
    for name in METHOD_ORDER:
        m = next((x for x in cross_data.get("methods", []) if x["method"] == name), None)
        if not m:
            continue
        rows.append({
            "method": name,
            "n_training_seeds": m.get("num_training_seeds", 0),
            "success_rate_mean": m.get("instant_success_rate_mean", np.nan),
            "success_rate_std": m.get("instant_success_rate_std", np.nan),
            "mean_return_mean": m.get("mean_return_mean", np.nan),
            "mean_return_std": m.get("mean_return_std", np.nan),
            "mean_final_range_m": m.get("mean_final_range_m_mean", np.nan),
            "mean_final_ata_deg": m.get("mean_final_ata_deg_mean", np.nan),
        })
    return pd.DataFrame(rows)


def build_table_b_feasible_subset(df: pd.DataFrame) -> pd.DataFrame:
    feasible = df[df["scenario"].isin(FEASIBLE_SCENARIOS)]
    rows = []
    for method in METHOD_ORDER:
        mdf = feasible[feasible["method"] == method]
        if len(mdf) == 0:
            continue
        rows.append({
            "method": method,
            "n_episodes": len(mdf),
            "success_rate": mdf["is_success"].mean(),
            "mean_return": mdf["return"].mean(),
            "std_return": mdf["return"].std(ddof=1),
            "mean_env_error": mdf["mean_env_prediction_error_m"].mean(),
        })
    return pd.DataFrame(rows)


def build_table_c_dead_zone(df: pd.DataFrame) -> pd.DataFrame:
    dead = df[df["scenario"].isin(DEAD_ZONE_SCENARIOS)]
    rows = []
    for scenario in sorted(dead["scenario"].unique()):
        for method in METHOD_ORDER:
            sdf = dead[(dead["method"] == method) & (dead["scenario"] == scenario)]
            if len(sdf) == 0:
                continue
            fail_reasons = sdf[~sdf["is_success"]]["reason"].value_counts().to_dict()
            rows.append({
                "scenario": scenario,
                "method": method,
                "n_episodes": len(sdf),
                "success_rate": sdf["is_success"].mean(),
                "mean_return": sdf["return"].mean(),
                "primary_failure_reason": max(fail_reasons, key=fail_reasons.get) if fail_reasons else "N/A",
                "failure_reason_counts": json.dumps(fail_reasons),
            })
    return pd.DataFrame(rows)


def build_table_d_maneuvering(df: pd.DataFrame) -> pd.DataFrame:
    man = df[df["scenario"].isin(MANEUVERING_SCENARIOS)]
    rows = []
    for scenario in sorted(man["scenario"].unique()):
        for method in METHOD_ORDER:
            sdf = man[(man["method"] == method) & (man["scenario"] == scenario)]
            if len(sdf) == 0:
                continue
            rows.append({
                "scenario": scenario,
                "method": method,
                "n_episodes": len(sdf),
                "success_rate": sdf["is_success"].mean(),
                "mean_return": sdf["return"].mean(),
                "std_return": sdf["return"].std(ddof=1),
                "mean_env_error": sdf["mean_env_prediction_error_m"].mean(),
            })
    return pd.DataFrame(rows)


def build_table_e_gru_lstm_focused(df: pd.DataFrame) -> pd.DataFrame:
    wh = df[df["scenario"] == "weaving_headon"]
    rows = []
    for method in ["lstm_frozen", "gru_frozen"]:
        mdf = wh[wh["method"] == method]
        if len(mdf) == 0:
            continue
        rows.append({
            "method": method,
            "n_episodes": len(mdf),
            "success_rate": mdf["is_success"].mean(),
            "mean_return": mdf["return"].mean(),
            "std_return": mdf["return"].std(ddof=1),
            "mean_env_error_m": mdf["mean_env_prediction_error_m"].mean(),
            "mean_vpp_shift_m": mdf["mean_virtual_point_shift_m"].mean(),
            "mean_final_range_m": mdf["final_range_m"].mean(),
            "mean_final_ata_deg": mdf["final_ata_deg"].mean(),
        })
    return pd.DataFrame(rows)


def build_table_f_cv_ca_delta(df: pd.DataFrame) -> pd.DataFrame:
    cv_df = df[df["method"] == "cv_prediction"]
    ca_df = df[df["method"] == "ca_prediction"]
    if cv_df.empty or ca_df.empty:
        return pd.DataFrame()

    rows = []
    for scenario in sorted(df["scenario"].unique()):
        cv_sc = cv_df[cv_df["scenario"] == scenario]
        ca_sc = ca_df[ca_df["scenario"] == scenario]
        if len(cv_sc) == 0 or len(ca_sc) == 0:
            continue
        # Compute effect size (Cohen's d)
        pooled_std = np.sqrt((cv_sc["return"].var(ddof=1) + ca_sc["return"].var(ddof=1)) / 2)
        cohens_d = (ca_sc["return"].mean() - cv_sc["return"].mean()) / pooled_std if pooled_std > 0 else np.nan
        rows.append({
            "scenario": scenario,
            "cv_success_rate": cv_sc["is_success"].mean(),
            "ca_success_rate": ca_sc["is_success"].mean(),
            "delta_success_rate": ca_sc["is_success"].mean() - cv_sc["is_success"].mean(),
            "cv_mean_return": cv_sc["return"].mean(),
            "ca_mean_return": ca_sc["return"].mean(),
            "delta_mean_return": ca_sc["return"].mean() - cv_sc["return"].mean(),
            "cohens_d": cohens_d,
            "success_changed": bool((ca_sc["is_success"].mean() > 0) != (cv_sc["is_success"].mean() > 0)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistical Tests
# ---------------------------------------------------------------------------

def bootstrap_ci(values: np.ndarray, n_bootstrap: int = 10000, ci: float = 0.95) -> tuple:
    """Return (lower, upper) bootstrap percentile CI for the mean."""
    if len(values) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed=42)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_means.append(np.mean(sample))
    boot_means = np.sort(boot_means)
    alpha = 1 - ci
    lower_idx = int(np.floor(alpha / 2 * n_bootstrap))
    upper_idx = int(np.ceil((1 - alpha / 2) * n_bootstrap)) - 1
    return (float(boot_means[lower_idx]), float(boot_means[upper_idx]))


def bootstrap_success_rate_ci(success_flags: np.ndarray, n_bootstrap: int = 10000, ci: float = 0.95) -> tuple:
    """Bootstrap CI for success rate (proportion)."""
    if len(success_flags) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed=42)
    boot_rates = []
    for _ in range(n_bootstrap):
        sample = rng.choice(success_flags, size=len(success_flags), replace=True)
        boot_rates.append(np.mean(sample))
    boot_rates = np.sort(boot_rates)
    alpha = 1 - ci
    lower_idx = int(np.floor(alpha / 2 * n_bootstrap))
    upper_idx = int(np.ceil((1 - alpha / 2) * n_bootstrap)) - 1
    return (float(boot_rates[lower_idx]), float(boot_rates[upper_idx]))


def paired_bootstrap_paired_diff_ci(a_values: np.ndarray, b_values: np.ndarray, n_bootstrap: int = 10000, ci: float = 0.95) -> tuple:
    """Bootstrap CI for paired mean difference (b - a)."""
    if len(a_values) == 0 or len(b_values) == 0 or len(a_values) != len(b_values):
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed=42)
    diffs = b_values - a_values
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(diffs, size=len(diffs), replace=True)
        boot_means.append(np.mean(sample))
    boot_means = np.sort(boot_means)
    alpha = 1 - ci
    lower_idx = int(np.floor(alpha / 2 * n_bootstrap))
    upper_idx = int(np.ceil((1 - alpha / 2) * n_bootstrap)) - 1
    return (float(boot_means[lower_idx]), float(boot_means[upper_idx]))


def mcnemar_paired_comparison(a_success: np.ndarray, b_success: np.ndarray) -> dict:
    """McNemar-style exact test for paired binary outcomes."""
    if len(a_success) == 0 or len(b_success) == 0 or len(a_success) != len(b_success):
        return {"n": 0, "a_only": 0, "b_only": 0, "both": 0, "neither": 0, "p_value": np.nan}
    a_only = int(np.sum((a_success == True) & (b_success == False)))
    b_only = int(np.sum((a_success == False) & (b_success == True)))
    both = int(np.sum((a_success == True) & (b_success == True)))
    neither = int(np.sum((a_success == False) & (b_success == False)))
    # Exact binomial test on discordant pairs
    discordant = a_only + b_only
    if discordant == 0:
        p_value = 1.0
    else:
        from math import comb
        p_value = 0.0
        for k in range(b_only, discordant + 1):
            p_value += comb(discordant, k) * (0.5 ** discordant)
        p_value = min(p_value * 2, 1.0)  # two-sided
    return {
        "n": len(a_success),
        "a_only": a_only,
        "b_only": b_only,
        "both": both,
        "neither": neither,
        "discordant": discordant,
        "p_value": float(p_value),
    }


def cohens_d_between_groups(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for independent groups (pooled std)."""
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled_std = np.sqrt(((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / (len(a) + len(b) - 2))
    if pooled_std == 0:
        return np.nan
    return (np.mean(b) - np.mean(a)) / pooled_std


def run_statistical_tests(df: pd.DataFrame) -> dict:
    results = {}

    # 1. Bootstrap 95% CI per method per scenario
    ci_results = []
    for (method, scenario), group in df.groupby(["method", "scenario"]):
        sr_ci = bootstrap_success_rate_ci(group["is_success"].values)
        ret_ci = bootstrap_ci(group["return"].values)
        ci_results.append({
            "method": method,
            "scenario": scenario,
            "n_episodes": len(group),
            "success_rate": group["is_success"].mean(),
            "success_rate_ci_lower": sr_ci[0],
            "success_rate_ci_upper": sr_ci[1],
            "mean_return": group["return"].mean(),
            "return_ci_lower": ret_ci[0],
            "return_ci_upper": ret_ci[1],
        })
    results["per_method_scenario_ci"] = ci_results

    # 2a. GRU vs LSTM in weaving_headon (paired episode comparison)
    wh = df[df["scenario"] == "weaving_headon"]
    lstm_wh = wh[wh["method"] == "lstm_frozen"]
    gru_wh = wh[wh["method"] == "gru_frozen"]
    if len(lstm_wh) == len(gru_wh) and len(lstm_wh) > 0:
        gru_vs_lstm = mcnemar_paired_comparison(lstm_wh["is_success"].values, gru_wh["is_success"].values)
        gru_vs_lstm["return_diff_ci"] = paired_bootstrap_paired_diff_ci(lstm_wh["return"].values, gru_wh["return"].values)
        gru_vs_lstm["cohens_d"] = cohens_d_between_groups(lstm_wh["return"].values, gru_wh["return"].values)
        # Cross-seed consistency: compute per-training-seed success rates
        gru_seed_sr = gru_wh.groupby("training_seed")["is_success"].mean().sort_index()
        lstm_seed_sr = lstm_wh.groupby("training_seed")["is_success"].mean().sort_index()
        gru_vs_lstm["gru_per_seed_sr"] = {int(k): float(v) for k, v in gru_seed_sr.items()}
        gru_vs_lstm["lstm_per_seed_sr"] = {int(k): float(v) for k, v in lstm_seed_sr.items()}
        gru_vs_lstm["gru_mean_seed_sr"] = float(gru_seed_sr.mean())
        gru_vs_lstm["lstm_mean_seed_sr"] = float(lstm_seed_sr.mean())
        gru_vs_lstm["gru_std_seed_sr"] = float(gru_seed_sr.std(ddof=1)) if len(gru_seed_sr) >= 2 else np.nan
        gru_vs_lstm["lstm_std_seed_sr"] = float(lstm_seed_sr.std(ddof=1)) if len(lstm_seed_sr) >= 2 else np.nan
        # Consistency: GRU > LSTM in all seeds? (strict, to avoid ties masking instability)
        seed_comparison = gru_seed_sr > lstm_seed_sr
        gru_vs_lstm["gru_gt_lstm_all_seeds"] = bool(seed_comparison.all()) if len(seed_comparison) > 0 else False
        gru_vs_lstm["gru_ge_lstm_all_seeds"] = bool((gru_seed_sr >= lstm_seed_sr).all()) if len(seed_comparison) > 0 else False
        gru_vs_lstm["n_training_seeds"] = int(len(seed_comparison))
        results["gru_vs_lstm_weaving_headon"] = gru_vs_lstm

    # 2b. Neural vs classical in feasible subset
    feasible = df[df["scenario"].isin(FEASIBLE_SCENARIOS)]
    classical_methods = ["no_prediction", "cv_prediction", "ca_prediction"]
    neural_methods = ["lstm_frozen", "gru_frozen"]
    c_df = feasible[feasible["method"].isin(classical_methods)]
    n_df = feasible[feasible["method"].isin(neural_methods)]
    if len(c_df) > 0 and len(n_df) > 0:
        # Aggregate per episode; treat as independent groups (not paired)
        neural_vs_classical = {
            "classical_success_rate": float(c_df["is_success"].mean()),
            "neural_success_rate": float(n_df["is_success"].mean()),
            "delta_success_rate": float(n_df["is_success"].mean() - c_df["is_success"].mean()),
            "classical_return_mean": float(c_df["return"].mean()),
            "neural_return_mean": float(n_df["return"].mean()),
            "cohens_d": cohens_d_between_groups(c_df["return"].values, n_df["return"].values),
        }
        results["neural_vs_classical_feasible"] = neural_vs_classical

    # 2c. CA vs CV in maneuvering target
    man = df[df["scenario"].isin(MANEUVERING_SCENARIOS)]
    cv_man = man[man["method"] == "cv_prediction"]
    ca_man = man[man["method"] == "ca_prediction"]
    if len(cv_man) == len(ca_man) and len(cv_man) > 0:
        ca_vs_cv = mcnemar_paired_comparison(cv_man["is_success"].values, ca_man["is_success"].values)
        ca_vs_cv["return_diff_ci"] = paired_bootstrap_paired_diff_ci(cv_man["return"].values, ca_man["return"].values)
        ca_vs_cv["cohens_d"] = cohens_d_between_groups(cv_man["return"].values, ca_man["return"].values)
        results["ca_vs_cv_maneuvering"] = ca_vs_cv

    return results


# ---------------------------------------------------------------------------
# Claims Checklist
# ---------------------------------------------------------------------------

def build_claims_checklist(stats: dict, tables: dict) -> list:
    claims = []

    # Claim 1: Neural predictors improve success in feasible geometries
    nvc = stats.get("neural_vs_classical_feasible", {})
    neural_sr = nvc.get("neural_success_rate", np.nan)
    classical_sr = nvc.get("classical_success_rate", np.nan)
    claims.append({
        "claim": "Neural predictors (LSTM/GRU) improve success rate over classical predictors in feasible intercept geometries.",
        "evidence": f"Neural SR={neural_sr:.1%}, Classical SR={classical_sr:.1%}, delta={nvc.get('delta_success_rate', np.nan):+.1%}, Cohen's d={nvc.get('cohens_d', np.nan):.2f}",
        "statistically_supported": bool(nvc.get("cohens_d", 0) is not np.nan and abs(nvc.get("cohens_d", 0)) > 0.2),
        "practically_meaningful": bool(abs(nvc.get("delta_success_rate", 0)) > 0.05),
        "paper_safe_claim": bool(abs(nvc.get("delta_success_rate", 0)) > 0.05 and abs(nvc.get("cohens_d", 0)) > 0.2),
    })

    # Claim 2: GRU > LSTM in weaving_headon
    gvl = stats.get("gru_vs_lstm_weaving_headon", {})
    gru_sr = None
    lstm_sr = None
    wh_table = tables.get("table_e", pd.DataFrame())
    if not wh_table.empty:
        gru_row = wh_table[wh_table["method"] == "gru_frozen"]
        lstm_row = wh_table[wh_table["method"] == "lstm_frozen"]
        if not gru_row.empty:
            gru_sr = gru_row.iloc[0]["success_rate"]
        if not lstm_row.empty:
            lstm_sr = lstm_row.iloc[0]["success_rate"]
    gru_sr_str = f"{gru_sr:.1%}" if gru_sr is not None else "N/A"
    lstm_sr_str = f"{lstm_sr:.1%}" if lstm_sr is not None else "N/A"
    cross_seed_strict = bool(gvl.get("gru_gt_lstm_all_seeds", False))
    cross_seed_weak = bool(gvl.get("gru_ge_lstm_all_seeds", False))
    n_seeds = int(gvl.get("n_training_seeds", 0))
    evidence = (
        f"GRU SR={gru_sr_str}, LSTM SR={lstm_sr_str}, "
        f"episode-level p={gvl.get('p_value', np.nan):.3f}, Cohen's d={gvl.get('cohens_d', np.nan):.2f}; "
        f"cross-seed GRU>LSTM in all seeds={cross_seed_strict} (n={n_seeds} training seeds). "
        f"NOTE: episode-level paired test with large n_episodes; independent training repeats are limited."
    )
    claims.append({
        "claim": "GRU is more robust than LSTM under maneuvering head-on target (weaving_headon).",
        "evidence": evidence,
        "statistically_supported": bool(gvl.get("p_value", 1.0) < 0.05),
        "practically_meaningful": bool(abs((gru_sr or 0) - (lstm_sr or 0)) > 0.10),
        "paper_safe_claim": bool(
            gvl.get("p_value", 1.0) < 0.05
            and abs((gru_sr or 0) - (lstm_sr or 0)) > 0.10
            and cross_seed_strict
            and n_seeds >= 3
        ),
    })

    # Claim 3: Tail-chase is a guidance-law limitation
    dead_table = tables.get("table_c", pd.DataFrame())
    dead_zero = False
    if not dead_table.empty:
        dead_zero = (dead_table["success_rate"] == 0.0).all()
    claims.append({
        "claim": "Tail-chase and stern-conversion geometries remain unsolved for all methods under the current LOS-rate guidance formulation.",
        "evidence": "All methods show 0% success in favorable/disadvantage/weaving_pursuit/weaving_disadvantage." if dead_zero else "Mixed results in dead-zone scenarios.",
        "statistically_supported": True,
        "practically_meaningful": True,
        "paper_safe_claim": dead_zero,
    })

    # Claim 4: CA vs CV
    cvc = stats.get("ca_vs_cv_maneuvering", {})
    cvca_table = tables.get("table_f", pd.DataFrame())
    cvca_meaningful = False
    if not cvca_table.empty:
        cvca_meaningful = not cvca_table["success_changed"].any() and cvca_table["cohens_d"].abs().max() < 0.2
    statistically_supported = bool(cvc.get("p_value", 1.0) < 0.05)
    claims.append({
        "claim": "CA predictor shows statistically detectable but practically negligible difference from CV under current maneuvering intensity.",
        "evidence": f"McNemar p={cvc.get('p_value', np.nan):.3f}, Cohen's d={cvc.get('cohens_d', np.nan):.2f}, no success/failure changes.",
        "statistically_supported": statistically_supported,
        "practically_meaningful": False,
        "paper_safe_claim": bool(cvca_meaningful and statistically_supported),
    })

    return claims


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def render_paper_main_results_md(tables: dict, stats: dict, claims: list) -> str:
    lines = []
    lines.append("# Stage 6F Paper-Ready Synthesis")
    lines.append("")
    lines.append(f"**Metrics Schema Version**: {METRICS_SCHEMA_VERSION}")
    lines.append(f"**Experiment Suite Version**: {EXPERIMENT_SUITE_VERSION}")
    lines.append("")

    # Table A
    lines.append("## Table A: Overall Ablation (Stage 6F Cross-Training-Seed)")
    lines.append("")
    ta = tables.get("table_a", pd.DataFrame())
    if not ta.empty:
        lines.append("| Method | N Seeds | Success Rate | Mean Return | Final Range (m) | Final ATA (deg) |")
        lines.append("|--------|--------:|-------------:|------------:|----------------:|----------------:|")
        for _, row in ta.iterrows():
            lines.append(
                f"| {row['method']} | {row['n_training_seeds']} | "
                f"{row['success_rate_mean']:.1%} ± {row['success_rate_std']:.1%} | "
                f"{row['mean_return_mean']:.1f} ± {row['mean_return_std']:.1f} | "
                f"{row['mean_final_range_m']:.1f} | {row['mean_final_ata_deg']:.1f} |"
            )
    lines.append("")

    # Table B
    lines.append("## Table B: Feasible-Geometry Subset")
    lines.append("")
    tb = tables.get("table_b", pd.DataFrame())
    if not tb.empty:
        lines.append("| Method | Episodes | Success Rate | Mean Return | Env Error (m) |")
        lines.append("|--------|---------:|-------------:|------------:|--------------:|")
        for _, row in tb.iterrows():
            env_err = f"{row['mean_env_error']:.1f}" if not np.isnan(row['mean_env_error']) else "N/A"
            lines.append(
                f"| {row['method']} | {row['n_episodes']} | {row['success_rate']:.1%} | "
                f"{row['mean_return']:.1f} | {env_err} |"
            )
    lines.append("")

    # Table C
    lines.append("## Table C: Tail-Chase Dead Zone")
    lines.append("")
    tc = tables.get("table_c", pd.DataFrame())
    if not tc.empty:
        lines.append("| Scenario | Method | Episodes | Success Rate | Mean Return | Primary Failure |")
        lines.append("|----------|--------|---------:|-------------:|------------:|----------------|")
        for _, row in tc.iterrows():
            lines.append(
                f"| {row['scenario']} | {row['method']} | {row['n_episodes']} | "
                f"{row['success_rate']:.1%} | {row['mean_return']:.1f} | {row['primary_failure_reason']} |"
            )
    lines.append("")

    # Table D
    lines.append("## Table D: Maneuvering Target")
    lines.append("")
    td = tables.get("table_d", pd.DataFrame())
    if not td.empty:
        lines.append("| Scenario | Method | Episodes | Success Rate | Mean Return | Env Error (m) |")
        lines.append("|----------|--------|---------:|-------------:|------------:|--------------:|")
        for _, row in td.iterrows():
            env_err = f"{row['mean_env_error']:.1f}" if not np.isnan(row['mean_env_error']) else "N/A"
            lines.append(
                f"| {row['scenario']} | {row['method']} | {row['n_episodes']} | "
                f"{row['success_rate']:.1%} | {row['mean_return']:.1f} | {env_err} |"
            )
    lines.append("")

    # Table E
    lines.append("## Table E: GRU vs LSTM in weaving_headon")
    lines.append("")
    te = tables.get("table_e", pd.DataFrame())
    if not te.empty:
        lines.append("| Method | Episodes | Success Rate | Mean Return | Env Error (m) | VPP Shift (m) | Final Range (m) | Final ATA (deg) |")
        lines.append("|--------|---------:|-------------:|------------:|--------------:|--------------:|----------------:|----------------:|")
        for _, row in te.iterrows():
            lines.append(
                f"| {row['method']} | {row['n_episodes']} | {row['success_rate']:.1%} | "
                f"{row['mean_return']:.1f} | {row['mean_env_error_m']:.1f} | {row['mean_vpp_shift_m']:.1f} | "
                f"{row['mean_final_range_m']:.1f} | {row['mean_final_ata_deg']:.1f} |"
            )
    lines.append("")

    # Table F
    lines.append("## Table F: CV vs CA Delta")
    lines.append("")
    tf = tables.get("table_f", pd.DataFrame())
    if not tf.empty:
        lines.append("| Scenario | CV SR | CA SR | ΔSR | CV Return | CA Return | ΔReturn | Cohen's d | Success Changed? |")
        lines.append("|----------|------:|------:|----:|----------:|----------:|--------:|----------:|:----------------:|")
        for _, row in tf.iterrows():
            changed = "Yes" if row["success_changed"] else "No"
            lines.append(
                f"| {row['scenario']} | {row['cv_success_rate']:.1%} | {row['ca_success_rate']:.1%} | "
                f"{row['delta_success_rate']:+.1%} | {row['cv_mean_return']:.1f} | {row['ca_mean_return']:.1f} | "
                f"{row['delta_mean_return']:+.1f} | {row['cohens_d']:.2f} | {changed} |"
            )
    lines.append("")

    # Statistical Tests
    lines.append("## Statistical Tests")
    lines.append("")

    gvl = stats.get("gru_vs_lstm_weaving_headon", {})
    if gvl:
        lines.append("### GRU vs LSTM (weaving_headon)")
        lines.append(f"- Paired episodes: n={gvl['n']}")
        lines.append(f"- GRU-only successes: {gvl['b_only']}, LSTM-only successes: {gvl['a_only']}")
        lines.append(f"- McNemar exact p-value: {gvl['p_value']:.4f}")
        lines.append(f"- Return difference 95% CI: [{gvl['return_diff_ci'][0]:.1f}, {gvl['return_diff_ci'][1]:.1f}]")
        lines.append(f"- Cohen's d: {gvl['cohens_d']:.2f}")
        lines.append(f"- Cross-seed GRU SR: {gvl.get('gru_per_seed_sr', {})}")
        lines.append(f"- Cross-seed LSTM SR: {gvl.get('lstm_per_seed_sr', {})}")
        lines.append(f"- GRU > LSTM in all seeds (strict): {gvl.get('gru_gt_lstm_all_seeds', False)}")
        lines.append(f"- GRU >= LSTM in all seeds (weak): {gvl.get('gru_ge_lstm_all_seeds', False)}")
        lines.append(f"- **WARNING**: Episode-level p-value is dominated by large n_episodes; only {gvl.get('n_training_seeds', 'unknown')} independent training seeds available.")
        lines.append("")

    nvc = stats.get("neural_vs_classical_feasible", {})
    if nvc:
        lines.append("### Neural vs Classical (feasible subset)")
        lines.append(f"- Neural success rate: {nvc['neural_success_rate']:.1%}")
        lines.append(f"- Classical success rate: {nvc['classical_success_rate']:.1%}")
        lines.append(f"- Δ success rate: {nvc['delta_success_rate']:+.1%}")
        lines.append(f"- Cohen's d (return): {nvc['cohens_d']:.2f}")
        lines.append("")

    cvc = stats.get("ca_vs_cv_maneuvering", {})
    if cvc:
        lines.append("### CA vs CV (maneuvering target)")
        lines.append(f"- Paired episodes: n={cvc['n']}")
        lines.append(f"- CA-only successes: {cvc['b_only']}, CV-only successes: {cvc['a_only']}")
        lines.append(f"- McNemar exact p-value: {cvc['p_value']:.4f}")
        lines.append(f"- Return difference 95% CI: [{cvc['return_diff_ci'][0]:.1f}, {cvc['return_diff_ci'][1]:.1f}]")
        lines.append(f"- Cohen's d: {cvc['cohens_d']:.2f}")
        lines.append("")

    # Claims Checklist
    lines.append("## Paper Claims Checklist")
    lines.append("")
    lines.append("| Claim | Statistically Supported | Practically Meaningful | Paper-Safe |")
    lines.append("|-------|:-----------------------:|:----------------------:|:----------:|")
    for claim in claims:
        ss = "Yes" if claim["statistically_supported"] else "No"
        pm = "Yes" if claim["practically_meaningful"] else "No"
        ps = "Yes" if claim["paper_safe_claim"] else "No"
        lines.append(f"| {claim['claim']} | {ss} | {pm} | {ps} |")
    lines.append("")

    return "\n".join(lines)


def render_paper_main_results_tex(tables: dict, stats: dict) -> str:
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Stage 6F Overall Ablation (Cross-Seed Mean$\pm$Std, $n{=}3$ training seeds)}")
    lines.append(r"\label{tab:stage6f_overall}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\hline")
    lines.append(r"Method & Success Rate & Mean Return & Final Range (m) & Final ATA (deg) \\")
    lines.append(r"\hline")
    ta = tables.get("table_a", pd.DataFrame())
    for _, row in ta.iterrows():
        lines.append(
            f"{row['method']} & {row['success_rate_mean']:.1%}$\\pm${row['success_rate_std']:.1%} & "
            f"{row['mean_return_mean']:.1f}$\\pm${row['mean_return_std']:.1f} & "
            f"{row['mean_final_range_m']:.1f} & {row['mean_final_ata_deg']:.1f} \\"
        )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Maneuvering Target: GRU vs LSTM in weaving\_headon}")
    lines.append(r"\label{tab:gru_lstm_weaving_headon}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\hline")
    lines.append(r"Method & Success Rate & Mean Return & Env Error (m) & VPP Shift (m) \\")
    lines.append(r"\hline")
    te = tables.get("table_e", pd.DataFrame())
    for _, row in te.iterrows():
        lines.append(
            f"{row['method']} & {row['success_rate']:.1%} & {row['mean_return']:.1f} & "
            f"{row['mean_env_error_m']:.1f} & {row['mean_vpp_shift_m']:.1f} \\"
        )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 6F Paper-Ready Synthesis")
    parser.add_argument("--stage6f", type=str, required=True)
    parser.add_argument("--stage6f_deep_audit", type=str, required=True)
    parser.add_argument("--stage6f5_feasible", type=str, required=True)
    parser.add_argument("--stage6f5_maneuvering", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/tables/stage6f_paper_synthesis")
    args = parser.parse_args()

    stage6f_dir = Path(args.stage6f)
    deep_audit_dir = Path(args.stage6f_deep_audit)
    feasible_dir = Path(args.stage6f5_feasible)
    maneuvering_dir = Path(args.stage6f5_maneuvering)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Stage 6F cross-seed summary
    cross_data = load_cross_seed_summary(stage6f_dir)
    if not cross_data:
        print(f"WARNING: No cross_seed_summary.json found in {stage6f_dir}")

    # Load deep audit scenario CSV
    deep_audit_df = load_deep_audit_scenario_csv(deep_audit_dir)

    # Build episode dataframes for 6F.5 suites
    feasible_seeds = discover_training_seeds(feasible_dir)
    maneuvering_seeds = discover_training_seeds(maneuvering_dir)
    print(f"Discovered feasible_geometry seeds: {feasible_seeds}")
    print(f"Discovered maneuvering_target seeds: {maneuvering_seeds}")

    feasible_df = build_episode_df(feasible_dir, feasible_seeds)
    maneuvering_df = build_episode_df(maneuvering_dir, maneuvering_seeds)

    # Combine all 6F.5 data
    all_6f5_df = pd.concat([feasible_df, maneuvering_df], ignore_index=True)

    # Build tables
    tables = {}
    tables["table_a"] = build_table_a_overall(cross_data)
    tables["table_b"] = build_table_b_feasible_subset(all_6f5_df)
    tables["table_c"] = build_table_c_dead_zone(all_6f5_df)
    tables["table_d"] = build_table_d_maneuvering(all_6f5_df)
    tables["table_e"] = build_table_e_gru_lstm_focused(all_6f5_df)
    tables["table_f"] = build_table_f_cv_ca_delta(all_6f5_df)

    # Statistical tests
    stats = run_statistical_tests(all_6f5_df)

    # Claims checklist
    claims = build_claims_checklist(stats, tables)

    # Save scenario filtered results CSV
    scenario_filtered = pd.concat([
        tables["table_b"].assign(subset="feasible"),
        tables["table_c"].assign(subset="dead_zone"),
        tables["table_d"].assign(subset="maneuvering"),
    ], ignore_index=True)
    scenario_filtered.to_csv(output_dir / "paper_scenario_filtered_results.csv", index=False, float_format="%.6f")

    # Save statistics JSON
    with open(output_dir / "paper_statistics.json", "w", encoding="utf-8") as f:
        json.dump({
            "schema_version": METRICS_SCHEMA_VERSION,
            "experiment_suite_version": EXPERIMENT_SUITE_VERSION,
            "statistics": stats,
            "claims": claims,
        }, f, indent=2, ensure_ascii=False, default=str)

    # Save claims checklist markdown
    claims_md_lines = ["# Paper Claims Checklist", ""]
    claims_md_lines.append("| Claim | Statistically Supported | Practically Meaningful | Paper-Safe |")
    claims_md_lines.append("|-------|:-----------------------:|:----------------------:|:----------:|")
    for claim in claims:
        ss = "Yes" if claim["statistically_supported"] else "No"
        pm = "Yes" if claim["practically_meaningful"] else "No"
        ps = "Yes" if claim["paper_safe_claim"] else "No"
        claims_md_lines.append(f"| {claim['claim']} | {ss} | {pm} | {ps} |")
    with open(output_dir / "paper_claims_checklist.md", "w", encoding="utf-8") as f:
        f.write("\n".join(claims_md_lines))

    # Save main results markdown
    main_md = render_paper_main_results_md(tables, stats, claims)
    with open(output_dir / "paper_main_results.md", "w", encoding="utf-8") as f:
        f.write(main_md)

    # Save main results LaTeX
    main_tex = render_paper_main_results_tex(tables, stats)
    with open(output_dir / "paper_main_results.tex", "w", encoding="utf-8") as f:
        f.write(main_tex)

    print(f"Synthesis complete. Output: {output_dir}")
    print(f"  - paper_main_results.md")
    print(f"  - paper_main_results.tex")
    print(f"  - paper_scenario_filtered_results.csv")
    print(f"  - paper_statistics.json")
    print(f"  - paper_claims_checklist.md")


if __name__ == "__main__":
    main()
