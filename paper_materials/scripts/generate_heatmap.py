#!/usr/bin/env python3
"""Generate per-scenario success-rate heatmap for paper materials."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

# Data from docs/results/stage6b/per_scenario_analysis.md
# and docs/results/lstm_gru_benchmark/raw_episodes.csv
# Success rates (%)
data = np.array([
    [100, 100, 100, 100, 100, 0, 0, 100],   # no_prediction
    [100, 100, 100, 0, 100, 0, 0, 100],     # parametric_prediction (cv/ca)
    [100, 100, 100, 100, 100, 0, 0, 100],   # lstm_frozen
    [100, 100, 100, 0, 100, 0, 0, 100],     # gru_frozen
], dtype=float)

methods = ["No-Prediction", "Parametric Pred.", "LSTM (Frozen)", "GRU (Frozen)"]

scenario_labels = [
    "Head-on\n(Neutral)",
    "Crossing\n(Challenging)",
    "Crossing Left",
    "Crossing Right",
    "Head-on\n(Close)",
    "Head-on\n(Medium)",
    "Head-on\n(Far)",
    "Crossing\n(Close)",
]

fig, ax = plt.subplots(figsize=(11, 4.5))

# Custom colormap: red (0) -> yellow (50) -> green (100)
from matplotlib.colors import LinearSegmentedColormap
colors = [(0.8, 0.2, 0.2), (1.0, 0.9, 0.4), (0.2, 0.7, 0.3)]
cmap = LinearSegmentedColormap.from_list("rg", colors, N=256)

im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=100)

# Add text annotations
for i in range(len(methods)):
    for j in range(len(scenario_labels)):
        val = data[i, j]
        text_color = "white" if val < 30 or val > 80 else "black"
        ax.text(j, i, f"{val:.0f}", ha="center", va="center", color=text_color, fontsize=11, fontweight="bold")

ax.set_xticks(np.arange(len(scenario_labels)))
ax.set_yticks(np.arange(len(methods)))
ax.set_xticklabels(scenario_labels, fontsize=9)
ax.set_yticklabels(methods, fontsize=10)

ax.set_title("Per-Scenario Success Rate (Feasible Geometry, 10 seeds)", fontsize=12, pad=12)
ax.set_xlabel("Scenario", fontsize=11)
ax.set_ylabel("Method", fontsize=11)

# Colorbar
cbar = plt.colorbar(im, ax=ax, label="Success Rate (%)", shrink=0.8)
cbar.ax.tick_params(labelsize=9)

# Grid lines
ax.set_xticks(np.arange(-.5, len(scenario_labels), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(methods), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=1.5)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "..", "figures", "fig5_per_scenario_heatmap.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved heatmap to {out_path}")
