#!/usr/bin/env python3
"""Compare CEM standard, EMA, and GD convergence from run_gain_only_cem outputs."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent.resolve()
INPUT_DIR = ROOT.parent / "outputs" / "cem_compare"
OUTPUT_DIR = INPUT_DIR


def load_history(mode):
    path = INPUT_DIR / mode / "cem_results.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def main():
    modes = ["standard", "ema", "gd"]
    records = {m: load_history(m) for m in modes}
    available = {m: r for m, r in records.items() if r is not None}
    if not available:
        print(f"No CEM results found in {INPUT_DIR}. Run scripts/run_gain_only_cem.py first.")
        return

    # Convergence plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = {"standard": "#1f77b4", "ema": "#ff7f0e", "gd": "#2ca02c"}

    for mode, data in available.items():
        history = data.get("history", [])
        if not history:
            continue
        iters = [h["iteration"] for h in history]
        best = [h["best_score"] for h in history]
        mean = [h["mean_score"] for h in history]
        axes[0].plot(iters, best, label=mode.upper(), color=colors[mode], marker="o", markersize=4)
        axes[1].plot(iters, mean, label=mode.upper(), color=colors[mode], marker="o", markersize=4)

    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Best score")
    axes[0].set_title("Best score per iteration")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Mean candidate score")
    axes[1].set_title("Mean score per iteration")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "convergence.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved convergence plot to {out_path}")

    # Summary markdown
    lines = ["# CEM Variant Comparison", ""]
    lines.append("| Mode | Final best score | Final mean score | Iterations |")
    lines.append("|------|------------------|------------------|------------|")
    for mode, data in available.items():
        history = data.get("history", [])
        if history:
            final = history[-1]
            lines.append(
                f"| {mode.upper()} | {final['best_score']:.4f} | {final['mean_score']:.4f} | {len(history)} |"
            )
        else:
            lines.append(f"| {mode.upper()} | — | — | 0 |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- **Standard CEM** fits the Gaussian directly to elite candidates; fast but can oscillate on noisy landscapes.")
    lines.append("- **CEM-EMA** smooths the mean/std updates; expected to reduce variance and improve stability.")
    lines.append("- **CEM-GD** uses score-weighted gradient ascent on the distribution mean; may be unstable on flat landscapes, supporting the theoretical recommendation to use EMA instead.")

    summary_path = OUTPUT_DIR / "summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
