#!/usr/bin/env python3
"""
Run all maneuver primitives in JSBSim and collect a summary.

This script invokes scripts/demo_maneuver_jsbsim.py for each maneuver with
maneuver-specific initial conditions and parameters, then writes a summary
of entry/exit state and completion status.

Usage:
    source activate jsbenv
    python scripts/run_maneuver_verification.py
"""
from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).parent.parent
DEMO_SCRIPT = PROJECT_ROOT / "scripts" / "demo_maneuver_jsbsim.py"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "maneuver_demo"

# Maneuver-specific ICs and parameters chosen to satisfy can_enter() checks.
MANEUVER_CONFIGS = {
    "straight_level": {
        "vt_mps": 250.0,
        "altitude_m": 5000.0,
        "steps": 1000,
        "params": {"duration_s": 5.0},
    },
    "coordinated_turn": {
        "vt_mps": 250.0,
        "altitude_m": 5000.0,
        "steps": 3000,
        "params": {"heading_change_deg": 90.0, "bank_angle_deg": 45.0, "velocity_ref_mps": 220.0},
    },
    "dive": {
        "vt_mps": 250.0,
        "altitude_m": 5000.0,
        "steps": 2500,
        "params": {"gamma_deg": 25.0, "target_altitude_m": 3000.0},
    },
    "loop": {
        "vt_mps": 350.0,
        "altitude_m": 6000.0,
        "steps": 5000,
        "params": {"entry_speed_mps": 350.0, "nz_target": 3.0},
    },
    "barrel_roll": {
        "vt_mps": 250.0,
        "altitude_m": 5000.0,
        "steps": 2000,
        "params": {"roll_rate_dps": 60.0},
    },
    "high_yoyo": {
        "vt_mps": 300.0,
        "altitude_m": 6000.0,
        "steps": 3000,
        "params": {},
    },
    "low_yoyo": {
        "vt_mps": 250.0,
        "altitude_m": 5000.0,
        "steps": 3000,
        "params": {},
    },
    "scissors": {
        "vt_mps": 250.0,
        "altitude_m": 5000.0,
        "steps": 5000,
        "params": {"cycles": 2, "bank_angle_deg": 45.0, "turn_angle_deg": 60.0},
    },
    "split_s": {
        "vt_mps": 350.0,
        "altitude_m": 10000.0,
        "steps": 4000,
        "params": {"entry_speed_mps": 350.0, "nz_pull": 5.0},
    },
    "immelmann": {
        "vt_mps": 350.0,
        "altitude_m": 4000.0,
        "steps": 6000,
        "params": {"entry_speed_mps": 350.0, "nz_pull": 2.0},
    },
}


def run_maneuver(name: str, cfg: dict) -> dict:
    output_path = OUTPUT_ROOT / f"{name}.csv"
    cmd = [
        "python", str(DEMO_SCRIPT),
        "--maneuver", name,
        "--output", str(output_path),
        "--vt-mps", str(cfg["vt_mps"]),
        "--altitude-m", str(cfg["altitude_m"]),
        "--steps", str(cfg["steps"]),
        "--settle-steps", "120",
        "--params", json.dumps(cfg["params"]),
    ]
    if cfg.get("pre_accel_s", 0.0) > 0.0:
        cmd.extend(["--pre-accel-s", str(cfg["pre_accel_s"])])
    print(f"\n[RUN] {name}")
    result = subprocess.run(
        cmd, cwd=PROJECT_ROOT, capture_output=True, text=True
    )

    summary = {
        "maneuver": name,
        "completed": False,
        "completed_step": None,
        "initial_alt_m": None,
        "initial_vt_mps": None,
        "final_alt_m": None,
        "final_vt_mps": None,
        "final_theta_deg": None,
        "final_phi_deg": None,
        "error": None,
    }

    # Parse stdout for completion / entry failure.
    for line in result.stdout.splitlines():
        if "Maneuver completed at step" in line:
            summary["completed"] = True
            summary["completed_step"] = int(line.split("step")[-1].strip())
        if "cannot enter from current state" in line:
            summary["error"] = "cannot_enter"

    if output_path.exists():
        df = pd.read_csv(output_path)
        if not df.empty:
            first = df.iloc[0]
            summary["initial_alt_m"] = float(first["alt_m"])
            summary["initial_vt_mps"] = float(first["vt_mps"])
            last = df.iloc[-1]
            summary["final_alt_m"] = float(last["alt_m"])
            summary["final_vt_mps"] = float(last["vt_mps"])
            summary["final_theta_deg"] = float(last["theta_deg"])
            summary["final_phi_deg"] = float(last["phi_deg"])

    if result.returncode != 0 and summary["error"] is None:
        summary["error"] = result.stderr.strip()[:200]

    print(f"  completed={summary['completed']}, "
          f"final_alt={summary['final_alt_m']:.1f}m, "
          f"final_vt={summary['final_vt_mps']:.1f}m/s")
    return summary


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name, cfg in MANEUVER_CONFIGS.items():
        summaries.append(run_maneuver(name, cfg))

    # Save CSV summary
    summary_csv = OUTPUT_ROOT / "summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    print(f"\nSaved CSV summary to {summary_csv}")

    # Save Markdown summary
    md_lines = ["# JSBSim Maneuver Verification Summary", ""]
    md_lines.append("| Maneuver | Completed | Init Alt (m) | Init Vt (m/s) | Final Alt (m) | Final Vt (m/s) | Final θ (deg) | Final φ (deg) | Error |")
    md_lines.append("|----------|-----------|--------------|---------------|---------------|----------------|---------------|---------------|-------|")
    for s in summaries:
        def fmt(value: float | None) -> str:
            return f"{value:.1f}" if value is not None else "—"

        md_lines.append(
            f"| {s['maneuver']} | {s['completed']} | "
            f"{fmt(s['initial_alt_m'])} | "
            f"{fmt(s['initial_vt_mps'])} | "
            f"{fmt(s['final_alt_m'])} | "
            f"{fmt(s['final_vt_mps'])} | "
            f"{fmt(s['final_theta_deg'])} | "
            f"{fmt(s['final_phi_deg'])} | "
            f"{s['error'] or ''} |"
        )
    summary_md = OUTPUT_ROOT / "summary.md"
    summary_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Saved Markdown summary to {summary_md}")


if __name__ == "__main__":
    main()
