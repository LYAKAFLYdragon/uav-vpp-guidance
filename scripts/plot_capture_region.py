#!/usr/bin/env python3
"""Generate capture region heatmaps from raw_results.json."""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_results(path):
    with open(path) as f:
        return json.load(f)


def plot_heatmap(data, x_values, y_values, xlabel, ylabel, title, output_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(data, aspect='auto', origin='lower', cmap='RdYlGn', vmin=0, vmax=1,
                   extent=[min(x_values), max(x_values), min(y_values), max(y_values)])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Success Rate')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else "docs/results/capture_region/raw_results.json"
    output_dir = os.path.dirname(input_path)
    
    results = load_results(input_path)
    
    # Organize by label
    methods = {}
    for r in results:
        label = r["label"]
        if label not in methods:
            methods[label] = []
        methods[label].append(r)
    
    for label, data in methods.items():
        # Extract unique values
        ranges = sorted(set(d["range_m"] for d in data))
        headings = sorted(set(d["heading_error_deg"] for d in data))
        speeds = sorted(set(d["speed_ratio"] for d in data))
        
        print(f"{label}: {len(ranges)} ranges, {len(headings)} headings, {len(speeds)} speeds")
        
        # For each speed ratio, plot range vs heading heatmap
        for speed in speeds:
            grid = np.zeros((len(ranges), len(headings)))
            for d in data:
                if abs(d["speed_ratio"] - speed) < 1e-6:
                    i = ranges.index(d["range_m"])
                    j = headings.index(d["heading_error_deg"])
                    grid[i, j] = d["success_rate"]
            
            title = f"{label} (speed_ratio={speed:.2f})"
            out_path = os.path.join(output_dir, f"heatmap_{label}_s{speed:.2f}.png")
            plot_heatmap(grid, headings, ranges, "Heading Error (deg)", "Initial Range (m)", title, out_path)
    
    print(f"\nAll heatmaps saved to {output_dir}/")


if __name__ == "__main__":
    main()
