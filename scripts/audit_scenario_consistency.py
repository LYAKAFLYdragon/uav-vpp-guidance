#!/usr/bin/env python3
"""Audit scenario parameter consistency across config files."""

import json
from pathlib import Path
import numpy as np
import pandas as pd

from uav_vpp_guidance.utils.config import load_yaml_config, merge_config

CONFIG_DIR = Path("config")


def load_recursive(path: Path, visited=None):
    if visited is None:
        visited = set()
    path = path.resolve()
    if path in visited:
        return {}
    visited.add(path)
    cfg = load_yaml_config(str(path))
    includes = cfg.pop("includes", [])
    merged = {}
    for inc in includes:
        inc_full = path.parent / inc
        if inc_full.exists():
            merged = merge_config(merged, load_recursive(inc_full, visited))
    return merge_config(merged, cfg)


def extract_scenarios(cfg: dict, source: str):
    """Extract scenario rows from a config."""
    rows = []
    scenarios = cfg.get("scenarios") or {}
    if isinstance(scenarios, list):
        scenarios = {s.get("name", f"scenario_{i}"): s for i, s in enumerate(scenarios)}
    for name, scen in scenarios.items():
        own = scen.get("own_init", {})
        tgt = scen.get("target_init", {})
        own_pos = np.asarray(own.get("position_m", [np.nan, np.nan, np.nan]))
        tgt_pos = np.asarray(tgt.get("position_m", [np.nan, np.nan, np.nan]))
        own_vel = float(own.get("velocity_mps", np.nan))
        tgt_vel = float(tgt.get("velocity_mps", np.nan))
        range_m = float(np.linalg.norm(own_pos - tgt_pos))
        ratio = own_vel / tgt_vel if tgt_vel and tgt_vel > 0 else np.nan
        rows.append({
            "source": source,
            "scenario": name,
            "own_vel": own_vel,
            "tgt_vel": tgt_vel,
            "ratio": ratio,
            "range_m": range_m,
            "own_alt": float(own_pos[2]) if len(own_pos) > 2 else np.nan,
            "tgt_alt": float(tgt_pos[2]) if len(tgt_pos) > 2 else np.nan,
            "own_heading": float(own.get("heading_deg", np.nan)),
            "tgt_heading": float(tgt.get("heading_deg", np.nan)),
        })
    return rows


def main():
    all_rows = []
    for path in sorted(CONFIG_DIR.rglob("*.yaml")):
        try:
            cfg = load_recursive(path)
        except Exception as exc:
            print(f"SKIP {path}: {exc}")
            continue
        rows = extract_scenarios(cfg, str(path.relative_to(CONFIG_DIR)))
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("No scenarios found.")
        return

    # Group by scenario and show statistics for numeric fields
    print("=== Scenario parameter consistency audit ===\n")
    for scenario in sorted(df["scenario"].unique()):
        sub = df[df["scenario"] == scenario]
        print(f"--- {scenario} ({len(sub)} occurrences) ---")
        for col in ["own_vel", "tgt_vel", "ratio", "range_m", "own_alt", "tgt_alt", "own_heading", "tgt_heading"]:
            vals = sub[col].dropna().unique()
            if len(vals) <= 1:
                continue
            print(f"  {col}: {len(vals)} distinct values")
            for v in sorted(vals):
                sources = sub[sub[col] == v]["source"].unique()
                print(f"    {v:.4g} -> {', '.join(sources)}")
        print()

    # Dump full table for reference
    out_path = Path("outputs/scenario_consistency_audit.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Full table written to {out_path}")


if __name__ == "__main__":
    main()
