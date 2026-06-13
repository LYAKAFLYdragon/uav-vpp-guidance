#!/usr/bin/env python3
"""Generate VPP-vs-PN success-rate-difference heatmaps from capture-region summary."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.resolve()
INPUT_PATH = ROOT.parent / "docs" / "results" / "capture_region_full" / "summary.csv"
OUTPUT_DIR = ROOT.parent / "docs" / "results" / "capture_region_full"


def plot_speed_ratio_heatmap(df, speed_ratio, output_path):
    sub = df[np.isclose(df["speed_ratio"], speed_ratio)]
    ranges = np.sort(sub["range_m"].unique())
    headings = np.sort(sub["heading_error_deg"].unique())

    grid = np.full((len(ranges), len(headings)), np.nan)
    for _, row in sub.iterrows():
        i = np.where(np.isclose(ranges, row["range_m"]))[0][0]
        j = np.where(np.isclose(headings, row["heading_error_deg"]))[0][0]
        grid[i, j] = row["sr_diff"]

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        grid,
        aspect="auto",
        origin="lower",
        cmap="RdYlGn",
        vmin=-1.0,
        vmax=1.0,
        extent=[headings.min(), headings.max(), ranges.min(), ranges.max()],
    )
    ax.set_xlabel("Heading error $\eta$ (deg)", fontsize=12)
    ax.set_ylabel("Initial range $R$ (m)", fontsize=12)
    ax.set_title(f"VPP+LOS vs PN direct success-rate difference ($v_o/v_t = {speed_ratio:.2f}$)", fontsize=13)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("$\\Delta SR$ = VPP SR $-$ PN SR", fontsize=11)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_mean_advantage(df, output_path):
    """Average sr_diff across all speed ratios."""
    piv = df.groupby(["range_m", "heading_error_deg"])["sr_diff"].mean().reset_index()
    ranges = np.sort(piv["range_m"].unique())
    headings = np.sort(piv["heading_error_deg"].unique())
    grid = np.full((len(ranges), len(headings)), np.nan)
    for _, row in piv.iterrows():
        i = np.where(np.isclose(ranges, row["range_m"]))[0][0]
        j = np.where(np.isclose(headings, row["heading_error_deg"]))[0][0]
        grid[i, j] = row["sr_diff"]

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        grid,
        aspect="auto",
        origin="lower",
        cmap="RdYlGn",
        vmin=-1.0,
        vmax=1.0,
        extent=[headings.min(), headings.max(), ranges.min(), ranges.max()],
    )
    ax.set_xlabel("Heading error $\eta$ (deg)", fontsize=12)
    ax.set_ylabel("Initial range $R$ (m)", fontsize=12)
    ax.set_title("VPP+LOS vs PN direct: mean success-rate difference across speed ratios", fontsize=13)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Mean $\\Delta SR$", fontsize=11)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    if not INPUT_PATH.exists():
        print(f"Input not found: {INPUT_PATH}")
        print("Run scripts/analyze_capture_region.py first.")
        return

    df = pd.read_csv(INPUT_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_mean_advantage(df, OUTPUT_DIR / "vpp_pn_advantage_mean.png")
    for speed in sorted(df["speed_ratio"].unique()):
        plot_speed_ratio_heatmap(df, speed, OUTPUT_DIR / f"vpp_pn_advantage_s{speed:.2f}.png")


if __name__ == "__main__":
    main()
