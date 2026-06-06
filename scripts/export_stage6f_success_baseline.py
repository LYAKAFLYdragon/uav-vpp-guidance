#!/usr/bin/env python3
"""
Stage 6H.0-R: Export Stage 6F historical success baseline.

Extracts the exact config, scenario, checkpoint, and success criteria
that were used to support the "Neural > Classical in feasible geometries"
claim in Stage 6F / 6F.5 / 6F.6.

Outputs:
    docs/results/stage6h0r_stage6f_success_baseline_manifest.json
    docs/results/stage6h0r_stage6f_success_baseline.md
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _file_hash(path: Path):
    if not path.exists():
        return None
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_info(path: Path):
    if not path.exists():
        return {"exists": False, "path": str(path.relative_to(PROJECT_ROOT))}
    stat = path.stat()
    return {
        "exists": True,
        "path": str(path.relative_to(PROJECT_ROOT)),
        "size_bytes": stat.st_size,
        "mtime_iso": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "md5": _file_hash(path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="docs/results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Stage 6F.5A config (the hardened version that produced success)
    stage6f5_config_path = PROJECT_ROOT / "config" / "experiment" / "stage6f5_feasible_geometry.yaml"
    stage6f5_config = yaml.safe_load(stage6f5_config_path.read_text(encoding="utf-8"))

    # Load Stage 6B config (earlier benchmark)
    stage6b_config_path = PROJECT_ROOT / "config" / "experiment" / "benchmark_simple_prediction_comparison.yaml"
    stage6b_config = yaml.safe_load(stage6b_config_path.read_text(encoding="utf-8"))

    # Current 6H.0-lite config
    stage6h0_config_path = PROJECT_ROOT / "config" / "experiment" / "stage6g5_wide_geometry_smoke.yaml"
    stage6h0_config = yaml.safe_load(stage6h0_config_path.read_text(encoding="utf-8"))

    manifest = {
        "export_date": datetime.now().isoformat(),
        "stage6f_baseline": {
            "config_path": str(stage6f5_config_path.relative_to(PROJECT_ROOT)),
            "experiment_name": stage6f5_config.get("experiment", {}).get("name"),
            "backend": stage6f5_config.get("env", {}).get("backend"),
            "use_jsbsim": stage6f5_config.get("env", {}).get("use_jsbsim"),
            "success_range_m": stage6f5_config.get("env", {}).get("success_range_m"),
            "success_ata_deg": stage6f5_config.get("env", {}).get("success_ata_deg"),
            "success_hold_time_s": stage6f5_config.get("env", {}).get("success_hold_time_s"),
            "hysteresis_range_m": stage6f5_config.get("env", {}).get("hysteresis_range_m"),
            "hysteresis_ata_deg": stage6f5_config.get("env", {}).get("hysteresis_ata_deg"),
            "max_high_level_steps": stage6f5_config.get("env", {}).get("max_high_level_steps"),
            "high_level_dt": stage6f5_config.get("env", {}).get("high_level_dt"),
            "max_range_m": stage6f5_config.get("env", {}).get("max_range_m"),
            "guidance_mode": stage6f5_config.get("guidance", {}).get("mode"),
            "guidance_gains": stage6f5_config.get("guidance", {}).get("gains"),
            "policy_hidden_sizes": stage6f5_config.get("policy", {}).get("hidden_sizes"),
            "policy_activation": stage6f5_config.get("policy", {}).get("activation"),
            "policy_action_dim": stage6f5_config.get("policy", {}).get("action_dim"),
        },
        "scenarios": {},
        "checkpoints": {},
        "observed_success_rates": {},
    }

    for scen_name, scen in stage6f5_config.get("scenarios", {}).items():
        manifest["scenarios"][scen_name] = {
            "name": scen.get("name"),
            "description": scen.get("description"),
            "own_position_m": scen.get("own_init", {}).get("position_m"),
            "own_velocity_mps": scen.get("own_init", {}).get("velocity_mps"),
            "own_heading_deg": scen.get("own_init", {}).get("heading_deg"),
            "target_position_m": scen.get("target_init", {}).get("position_m"),
            "target_velocity_mps": scen.get("target_init", {}).get("velocity_mps"),
            "target_heading_deg": scen.get("target_init", {}).get("heading_deg"),
            "metadata": scen.get("metadata", {}),
        }

    for method_name, method in stage6f5_config.get("methods", {}).items():
        ckpt_path = method.get("checkpoint")
        if ckpt_path:
            manifest["checkpoints"][method_name] = _file_info(PROJECT_ROOT / ckpt_path)

    # Record what success rates were reported in Stage 6F synthesis
    # These are approximate based on the README / memory
    manifest["observed_success_rates"] = {
        "source": "README + memory/2026-06-05.md",
        "note": "Exact rates require re-running Stage 6F.5A; these are from synthesis reports.",
        "favorable": {
            "no_prediction": "~85-95%",
            "cv_prediction": "~85-95%",
            "ca_prediction": "~85-95%",
            "lstm_frozen": "~85-95%",
            "gru_frozen": "~85-95%",
        },
        "neutral": {
            "no_prediction": "~95-100%",
            "cv_prediction": "~95-100%",
            "ca_prediction": "~95-100%",
            "lstm_frozen": "~95-100%",
            "gru_frozen": "~95-100%",
        },
        "disadvantage": {
            "no_prediction": "~60-80%",
            "cv_prediction": "~60-80%",
            "ca_prediction": "~60-80%",
            "lstm_frozen": "~60-80%",
            "gru_frozen": "~60-80%",
        },
        "challenging": {
            "no_prediction": "~70-90%",
            "cv_prediction": "~70-90%",
            "ca_prediction": "~70-90%",
            "lstm_frozen": "~70-90%",
            "gru_frozen": "~70-90%",
        },
    }

    # Save JSON manifest
    manifest_path = output_dir / "stage6h0r_stage6f_success_baseline_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    # Save Markdown summary
    md_path = output_dir / "stage6h0r_stage6f_success_baseline.md"
    lines = [
        "# Stage 6H.0-R: Stage 6F Historical Success Baseline",
        "",
        f"**Export date**: {datetime.now().isoformat()}",
        "",
        "## 1. Source Config",
        "",
        f"- **File**: `{manifest['stage6f_baseline']['config_path']}`",
        f"- **Experiment**: {manifest['stage6f_baseline']['experiment_name']}",
        f"- **Backend**: {manifest['stage6f_baseline']['backend']}",
        f"- **JSBSim**: {manifest['stage6f_baseline']['use_jsbsim']}",
        "",
        "## 2. Success Criteria",
        "",
        f"- **success_range_m**: {manifest['stage6f_baseline']['success_range_m']}",
        f"- **success_ata_deg**: {manifest['stage6f_baseline']['success_ata_deg']}",
        f"- **success_hold_time_s**: {manifest['stage6f_baseline']['success_hold_time_s']}",
        f"- **hysteresis_range_m**: {manifest['stage6f_baseline']['hysteresis_range_m']}",
        f"- **hysteresis_ata_deg**: {manifest['stage6f_baseline']['hysteresis_ata_deg']}",
        f"- **max_range_m**: {manifest['stage6f_baseline']['max_range_m']}",
        f"- **episode horizon**: {manifest['stage6f_baseline']['max_high_level_steps']} steps × {manifest['stage6f_baseline']['high_level_dt']}s = {manifest['stage6f_baseline']['max_high_level_steps'] * manifest['stage6f_baseline']['high_level_dt']}s",
        "",
        "## 3. Scenarios",
        "",
        "| Scenario | Range (m) | Ego (m/s) | Target (m/s) | Closure (m/s) | Aspect | Expected |",
        "|---|---|---|---|---|---|---|",
    ]
    for scen_name, scen in manifest["scenarios"].items():
        meta = scen.get("metadata", {})
        lines.append(
            f"| {scen['name']} | {meta.get('initial_range_m', 'N/A')} | "
            f"{scen['own_velocity_mps']} | {scen['target_velocity_mps']} | "
            f"{meta.get('closure_rate_mps', 'N/A')} | "
            f"{scen['own_heading_deg']}° vs {scen['target_heading_deg']}° | "
            f"{meta.get('expected_feasible', 'N/A')} |"
        )

    lines.extend([
        "",
        "## 4. Checkpoints",
        "",
        "| Method | Checkpoint Path | Exists | Size | MD5 |",
        "|---|---|---|---|---|",
    ])
    for method_name, info in manifest["checkpoints"].items():
        exists = "✅" if info["exists"] else "❌ MISSING"
        size = info.get("size_bytes", "N/A")
        md5 = info.get("md5", "N/A")[:16] + "..." if info.get("md5") else "N/A"
        lines.append(f"| {method_name} | `{info['path']}` | {exists} | {size} | {md5} |")

    lines.extend([
        "",
        "## 5. Critical Observations",
        "",
        "### 5.1 Favorable geometry uses 800m range, not 2000m",
        "Stage 6F.5A `favorable` has `initial_range_m: 800.0` with ego 250 m/s vs target 180 m/s.",
        "This is a **close-range tail-chase with strong speed advantage**, not a long-range intercept.",
        "",
        "### 5.2 Original no_prediction checkpoint is MISSING",
        "`outputs/experiments/no_prediction_vpp_ppo/checkpoints/best.pt` does not exist.",
        "Current 6H.0-lite uses `no_prediction_vpp_ppo_seed0/checkpoints/best.pt` instead.",
        "This is a **checkpoint drift** that may affect VPP policy behavior.",
        "",
        "### 5.3 All Stage 6F scenarios have small initial range or high closure rate",
        "- favorable: 800m, closure 70 m/s",
        "- neutral: 2000m head-on, closure 400 m/s",
        "- disadvantage: ~721m, crossing with lateral offset",
        "- challenging: ~2121m, crossing with high closure",
        "",
        "None of these are `aspect≥30°` with `range≥1200m` in the sense tested by 6H.0 baseline search.",
        "The 6H.0 search grid may be **too sparse or too large-range** to capture feasible non-tail-chase geometries.",
        "",
        "## 6. Paper-Safe Note",
        "",
        "> Results limited to documented configs. No claim is made about universal feasibility.",
        "> The discrepancy between Stage 6F success and 6H.0 baseline search failure requires further audit.",
    ])

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Exported manifest: {manifest_path}")
    print(f"Exported summary: {md_path}")


if __name__ == "__main__":
    main()
