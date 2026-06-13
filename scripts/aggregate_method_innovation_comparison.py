#!/usr/bin/env python3
"""Aggregate method-innovation comparison results."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scipy import stats


DEFAULT_ROOT = Path("outputs/method_innovation_compare")
ALGORITHMS = {
    "baseline": "Baseline PPO",
    "cr_ppo": "CR-PPO",
    "intentional": "Intentional PPO",
    "intentional_c": "Intentional PPO-C",
    "intentional_a": "Intentional PPO-A",
}
COLORS = {
    "baseline": "#7f7f7f",
    "cr_ppo": "#1f77b4",
    "intentional": "#ff7f0e",
    "intentional_c": "#2ca02c",
    "intentional_a": "#d62728",
}
SEED_GLOB = [0, 1, 2, 3, 4]

EVAL_METRICS = [
    "mean_return", "success_rate", "crash_rate",
    "out_of_bounds_rate", "timeout_rate",
]
UPDATE_METRICS_COMMON = [
    "policy_loss", "value_loss", "entropy",
    "approx_kl", "clip_fraction", "explained_variance",
]
UPDATE_METRICS_EXTRA = {
    "baseline": [],
    "cr_ppo": ["complexity"],
    "intentional": ["scale_actor", "scale_critic", "ema_abs_adv"],
    "intentional_c": ["scale_critic", "ema_abs_adv"],
    "intentional_a": ["scale_actor", "ema_abs_adv"],
}


def _read_csv(path):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_eval_log(algo, seed):
    path = ROOT / algo / f"seed{seed}" / "logs" / "eval_log.csv"
    return _read_csv(path)


def read_update_log(algo, seed):
    path = ROOT / algo / f"seed{seed}" / "logs" / "update_train_log.csv"
    return _read_csv(path)


def to_floats(rows, key):
    vals = []
    for r in rows:
        try:
            v = float(r.get(key, np.nan))
            if np.isfinite(v):
                vals.append(v)
        except (TypeError, ValueError):
            pass
    return np.array(vals)


def final_eval_metrics(algo, seed):
    rows = read_eval_log(algo, seed)
    if not rows:
        return None
    last = rows[-1]
    return {m: float(last.get(m, np.nan)) for m in EVAL_METRICS}


def sample_efficiency(algo, seed, threshold=0.5):
    rows = read_eval_log(algo, seed)
    if not rows:
        return np.nan, np.nan
    total_steps = int(rows[-1].get("step", np.nan))
    for r in rows:
        try:
            if float(r.get("success_rate", 0.0)) >= threshold:
                return int(r["step"]), total_steps
        except (TypeError, ValueError):
            continue
    return total_steps, total_steps


def stability_metrics(algo, seed):
    rows = read_update_log(algo, seed)
    if not rows:
        return None
    out = {}
    for m in UPDATE_METRICS_COMMON + UPDATE_METRICS_EXTRA.get(algo, []):
        vals = to_floats(rows, m)
        if len(vals) > 0:
            out[f"{m}_mean"] = float(np.mean(vals))
            out[f"{m}_std"] = float(np.std(vals))
        else:
            out[f"{m}_mean"] = np.nan
            out[f"{m}_std"] = np.nan
    return out


def aggregate_algorithm(algo):
    final_eval = []
    steps_to_threshold = []
    stability_list = []
    learning_curves = {}
    for seed in SEED_GLOB:
        ev = final_eval_metrics(algo, seed)
        if ev is not None:
            final_eval.append(ev)
            rows = read_eval_log(algo, seed)
            for r in rows:
                step = int(r["step"])
                learning_curves.setdefault(step, []).append(float(r.get("success_rate", np.nan)))
        st = sample_efficiency(algo, seed)
        if not np.isnan(st[0]):
            steps_to_threshold.append(st[0])
        sm = stability_metrics(algo, seed)
        if sm is not None:
            stability_list.append(sm)

    if not final_eval:
        return None

    result = {"n_seeds": len(final_eval)}
    for m in EVAL_METRICS:
        vals = [s[m] for s in final_eval]
        result[f"{m}_mean"] = float(np.mean(vals))
        result[f"{m}_std"] = float(np.std(vals))

    if steps_to_threshold:
        result["steps_to_50sr_mean"] = float(np.mean(steps_to_threshold))
        result["steps_to_50sr_std"] = float(np.std(steps_to_threshold))
    else:
        result["steps_to_50sr_mean"] = np.nan
        result["steps_to_50sr_std"] = np.nan

    if stability_list:
        keys = list(stability_list[0].keys())
        for k in keys:
            vals = [s[k] for s in stability_list if k in s and np.isfinite(s[k])]
            if vals:
                result[k] = float(np.mean(vals))
            else:
                result[k] = np.nan

    result["learning_curve"] = {}
    for step, vals in sorted(learning_curves.items()):
        result["learning_curve"][step] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }
    return result


def ttest_pair(algo_a, algo_b, metric="success_rate"):
    vals_a = []
    vals_b = []
    for seed in SEED_GLOB:
        ev_a = final_eval_metrics(algo_a, seed)
        ev_b = final_eval_metrics(algo_b, seed)
        if ev_a is not None and ev_b is not None:
            vals_a.append(ev_a[metric])
            vals_b.append(ev_b[metric])
    if len(vals_a) < 2 or len(vals_b) < 2:
        return np.nan, np.nan
    t, p = stats.ttest_ind(vals_a, vals_b, equal_var=False)
    return float(t), float(p)


def plot_learning_curves(summary):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for key in ALGORITHMS:
        if key not in summary:
            continue
        curve = summary[key].get("learning_curve", {})
        if not curve:
            continue
        steps = sorted(curve.keys())
        means = np.array([curve[s]["mean"] for s in steps])
        stds = np.array([curve[s]["std"] for s in steps])
        ax.plot(steps, means, label=ALGORITHMS[key], color=COLORS[key])
        ax.fill_between(steps, np.clip(means - stds, 0, 1), np.clip(means + stds, 0, 1),
                        color=COLORS[key], alpha=0.2)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Success rate")
    ax.set_title("Learning curves (mean +- std over seeds)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    algos = [k for k in ALGORITHMS if k in summary]
    x = np.arange(len(algos))
    means = [summary[k]["success_rate_mean"] for k in algos]
    stds = [summary[k]["success_rate_std"] for k in algos]
    ax.bar(x, means, yerr=stds, color=[COLORS[k] for k in algos], capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels([ALGORITHMS[k] for k in algos], rotation=15, ha="right")
    ax.set_ylabel("Final success rate")
    ax.set_title("Final success rate by algorithm")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = ROOT / "learning_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_stability_bars(summary):
    algos = [k for k in ALGORITHMS if k in summary]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    metrics = [
        ("value_loss_std", "Value loss std"),
        ("approx_kl_mean", "Approx KL mean"),
        ("clip_fraction_mean", "Clip fraction mean"),
        ("policy_loss_std", "Policy loss std"),
    ]
    for ax, (key, title) in zip(axes.flat, metrics):
        vals = [summary[a].get(key, np.nan) for a in algos]
        ax.bar(range(len(algos)), vals, color=[COLORS[a] for a in algos])
        ax.set_xticks(range(len(algos)))
        ax.set_xticklabels([ALGORITHMS[a] for a in algos], rotation=15, ha="right")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = ROOT / "stability_bars.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def _parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=str, default=None)
    args = parser.parse_args()
    if args.output_root:
        global ROOT
        ROOT = Path(args.output_root)
    return args


def main():
    _parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for key in ALGORITHMS:
        agg = aggregate_algorithm(key)
        if agg is not None:
            summary[key] = agg
            summary[key]["name"] = ALGORITHMS[key]

    if not summary:
        print(f"No evaluation logs found under {ROOT}")
        return

    baseline = "baseline"
    for key in ALGORITHMS:
        if key == baseline or key not in summary:
            continue
        t, p = ttest_pair(baseline, key, metric="success_rate")
        summary[key]["ttest_success_rate_t"] = t
        summary[key]["ttest_success_rate_p"] = p

    md_lines = [
        "# Method Innovation Comparison Summary",
        "",
        "## Final Performance Metrics",
        "",
        "| Algorithm | Return | Success Rate | Crash Rate | OOB Rate | Timeout Rate | Steps to 50% SR |",
        "|-----------|--------|--------------|------------|----------|--------------|-----------------|",
    ]
    for key in ALGORITHMS:
        if key not in summary:
            continue
        s = summary[key]
        md_lines.append(
            "| " + s['name'] + " | "
            + f"{s['mean_return_mean']:.1f}+-{s['mean_return_std']:.1f} | "
            + f"{s['success_rate_mean']:.2%}+-{s['success_rate_std']:.2%} | "
            + f"{s['crash_rate_mean']:.2%}+-{s['crash_rate_std']:.2%} | "
            + f"{s['out_of_bounds_rate_mean']:.2%}+-{s['out_of_bounds_rate_std']:.2%} | "
            + f"{s['timeout_rate_mean']:.2%}+-{s['timeout_rate_std']:.2%} | "
            + f"{s['steps_to_50sr_mean']:.0f}+-{s['steps_to_50sr_std']:.0f} |"
        )

    md_lines.append("")
    md_lines.append("## Stability Metrics (training update logs)")
    md_lines.append("")
    md_lines.append("| Algorithm | Policy Loss | Value Loss | Approx KL | Clip Fraction | Explained Var |")
    md_lines.append("|-----------|-------------|------------|-----------|---------------|---------------|")
    for key in ALGORITHMS:
        if key not in summary:
            continue
        s = summary[key]
        md_lines.append(
            "| " + s['name'] + " | "
            + f"{s.get('policy_loss_mean', np.nan):.4f}+-{s.get('policy_loss_std', 0):.4f} | "
            + f"{s.get('value_loss_mean', np.nan):.1f}+-{s.get('value_loss_std', 0):.1f} | "
            + f"{s.get('approx_kl_mean', np.nan):.4f}+-{s.get('approx_kl_std', 0):.4f} | "
            + f"{s.get('clip_fraction_mean', np.nan):.2%}+-{s.get('clip_fraction_std', 0):.2%} | "
            + f"{s.get('explained_variance_mean', np.nan):.4f}+-{s.get('explained_variance_std', 0):.4f} |"
        )

    md_lines.append("")
    md_lines.append("## Algorithm-Specific Diagnostics")
    md_lines.append("")
    md_lines.append("| Algorithm | Extra Metrics |")
    md_lines.append("|-----------|---------------|")
    for key in ALGORITHMS:
        if key not in summary:
            continue
        s = summary[key]
        extras = UPDATE_METRICS_EXTRA.get(key, [])
        if not extras:
            md_lines.append("| " + s['name'] + " | — |")
            continue
        parts = []
        for m in extras:
            mean = s.get(f"{m}_mean", np.nan)
            std = s.get(f"{m}_std", 0)
            parts.append(f"{m}={mean:.4f}+-{std:.4f}")
        md_lines.append("| " + s['name'] + " | " + ", ".join(parts) + " |")

    md_lines.append("")
    md_lines.append("## Pairwise t-test vs Baseline (success rate)")
    md_lines.append("")
    md_lines.append("| Algorithm | t-statistic | p-value | Significant (p<0.05) |")
    md_lines.append("|-----------|-------------|---------|----------------------|")
    for key in ALGORITHMS:
        if key == baseline or key not in summary:
            continue
        s = summary[key]
        p = s.get("ttest_success_rate_p", np.nan)
        sig = "yes" if not np.isnan(p) and p < 0.05 else "no"
        md_lines.append(
            "| " + s['name'] + " | "
            + f"{s.get('ttest_success_rate_t', np.nan):.3f} | "
            + f"{p:.4f} | {sig} |"
        )

    md_text = "\n".join(md_lines)
    print(md_text)

    md_path = ROOT / "summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"\nSaved markdown summary to {md_path}")

    csv_path = ROOT / "summary.csv"
    all_update_keys = UPDATE_METRICS_COMMON.copy()
    for extras in UPDATE_METRICS_EXTRA.values():
        all_update_keys.extend(extras)
    all_update_keys = list(dict.fromkeys(all_update_keys))
    fieldnames = ["algorithm", "n_seeds"] + [
        f"{m}_{stat}" for m in EVAL_METRICS for stat in ("mean", "std")
    ] + [
        "steps_to_50sr_mean", "steps_to_50sr_std"
    ] + [
        f"{m}_{stat}" for m in all_update_keys for stat in ("mean", "std")
    ] + ["ttest_success_rate_t", "ttest_success_rate_p"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in ALGORITHMS:
            if key not in summary:
                continue
            s = summary[key]
            row = {"algorithm": s["name"], "n_seeds": s["n_seeds"]}
            for m in EVAL_METRICS:
                row[f"{m}_mean"] = s[f"{m}_mean"]
                row[f"{m}_std"] = s[f"{m}_std"]
            row["steps_to_50sr_mean"] = s["steps_to_50sr_mean"]
            row["steps_to_50sr_std"] = s["steps_to_50sr_std"]
            for m in all_update_keys:
                row[f"{m}_mean"] = s.get(f"{m}_mean", np.nan)
                row[f"{m}_std"] = s.get(f"{m}_std", np.nan)
            row["ttest_success_rate_t"] = s.get("ttest_success_rate_t", np.nan)
            row["ttest_success_rate_p"] = s.get("ttest_success_rate_p", np.nan)
            writer.writerow(row)
    print(f"Saved CSV summary to {csv_path}")

    plot_learning_curves(summary)
    plot_stability_bars(summary)


if __name__ == "__main__":
    main()
