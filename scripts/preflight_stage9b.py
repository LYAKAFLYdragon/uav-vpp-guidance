#!/usr/bin/env python3
"""Stage 9B.0 preflight: verify environment readiness for official paper-safe benchmark.

Checks:
  1. Git working tree is clean (or warn)
  2. Git commit hash matches origin/main (or warn)
  3. Config file exists and is readable
  4. All method checkpoints exist
  5. gain_only gains file exists and schema is valid
  6. Output directory does not already contain official results
  7. Config hash for provenance tracking

Exit code:
  0 = all critical checks pass, benchmark may proceed
  1 = at least one critical check failed
"""

import argparse
import json
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from uav_vpp_guidance.guidance.gain_config import GuidanceGains

# Must stay in sync with run_paper_benchmark.py
METHODS = {
    "no_prediction": {
        "checkpoint": "outputs/experiments/no_prediction_vpp_ppo/checkpoints/best.pt",
        "config_method": "no_prediction",
    },
    "cv_prediction": {
        "checkpoint": "outputs/experiments/vpp_ppo_cv_prediction/checkpoints/best.pt",
        "config_method": "cv_prediction",
    },
    "ca_prediction": {
        "checkpoint": "outputs/experiments/vpp_ppo_ca_prediction/checkpoints/best.pt",
        "config_method": "ca_prediction",
    },
    "gain_only": {
        "checkpoint": "outputs/audit_no_pred_final/checkpoints/best.pt",
        "config_method": "no_prediction",
        "gains_path": "outputs/gain_only_cem/cem_results.json",
        "note": "Same policy as no_prediction but with CEM-optimized gains",
    },
}

_GAIN_FIELD_NAMES = {f.name for f in fields(GuidanceGains)}


def _git_info() -> dict:
    info = {"commit": "unknown", "dirty": True, "branch": "unknown"}
    try:
        info["commit"] = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True)
            .strip()
        )
        info["dirty"] = (
            len(subprocess.check_output(["git", "status", "--short"], text=True).strip())
            > 0
        )
        info["branch"] = (
            subprocess.check_output(["git", "branch", "--show-current"], text=True)
            .strip()
        )
    except Exception:
        pass
    return info


def _origin_main_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "origin/main"], text=True
            )
            .strip()
        )
    except Exception:
        return "unknown"


def _config_hash(config_path: str) -> str:
    import hashlib
    data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    canonical = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _check_gains_schema(gains_path: str) -> dict:
    result = {"exists": False, "valid": False, "loaded": {}, "ignored": []}
    p = Path(gains_path)
    if not p.exists():
        return result
    result["exists"] = True
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return result
    best = data.get("best_gains")
    if not isinstance(best, dict) or not best:
        return result
    loaded = {}
    ignored = []
    for k, v in best.items():
        if k in _GAIN_FIELD_NAMES:
            loaded[k] = v
        else:
            ignored.append(k)
    result["loaded"] = loaded
    result["ignored"] = ignored
    result["valid"] = bool(loaded)
    return result


def main():
    parser = argparse.ArgumentParser(description="Stage 9B.0 preflight")
    parser.add_argument(
        "--config",
        type=str,
        default="config/experiment/stage6f5_feasible_geometry.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/paper_benchmark_stage9b_simple_official",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=list(METHODS.keys()),
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="simple",
        choices=["simple", "jsbsim"],
    )
    args = parser.parse_args()

    report = {
        "status": "pass",
        "critical_failures": [],
        "warnings": [],
        "git": {},
        "config": {},
        "checkpoints": {},
        "gains": {},
        "output_dir": {},
    }

    # 1. Git info
    git = _git_info()
    origin_main = _origin_main_commit()
    report["git"] = {
        "commit": git["commit"],
        "dirty": git["dirty"],
        "branch": git["branch"],
        "origin_main": origin_main,
    }
    if git["dirty"]:
        report["warnings"].append("Git working tree is dirty. Benchmark should be run from a clean tree for reproducibility.")
    if git["commit"] != origin_main and origin_main != "unknown":
        report["warnings"].append(
            f"Local commit {git['commit']} does not match origin/main {origin_main}."
        )

    # 2. Config
    config_path = Path(args.config)
    if not config_path.exists():
        report["critical_failures"].append(f"Config file not found: {args.config}")
        report["config"]["exists"] = False
    else:
        report["config"]["exists"] = True
        report["config"]["hash"] = _config_hash(args.config)

    # 3. Checkpoints
    full_config = {}
    if config_path.exists():
        full_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    for method_name in args.methods:
        if method_name not in METHODS:
            report["critical_failures"].append(f"Unknown method: {method_name}")
            continue
        method_cfg = METHODS[method_name]
        # resolve same precedence as benchmark
        config_override = full_config.get("methods", {}).get(method_name, {})
        ckpt_path = config_override.get("checkpoint", method_cfg["checkpoint"])
        exists = Path(ckpt_path).exists()
        report["checkpoints"][method_name] = {
            "path": ckpt_path,
            "exists": exists,
        }
        if not exists:
            report["critical_failures"].append(
                f"Missing checkpoint for {method_name}: {ckpt_path}"
            )

    # 4. Gains (gain_only only)
    if "gain_only" in args.methods:
        gains_path = METHODS["gain_only"].get("gains_path")
        if gains_path:
            gains_report = _check_gains_schema(gains_path)
            report["gains"] = gains_report
            if not gains_report["exists"]:
                report["critical_failures"].append(
                    f"Missing gains file for gain_only: {gains_path}"
                )
            elif not gains_report["valid"]:
                report["critical_failures"].append(
                    f"Invalid gains schema for gain_only: {gains_path}"
                )
        else:
            report["critical_failures"].append("gains_path not set for gain_only")

    # 5. Output dir collision
    output_dir = Path(args.output_dir)
    report["output_dir"]["path"] = str(output_dir)
    collision = False
    for artifact in ("summary.md", "run_manifest.json", "results.csv"):
        if (output_dir / artifact).exists():
            collision = True
            report["critical_failures"].append(
                f"Output directory already contains {artifact}. Use a fresh directory to avoid overwriting official results."
            )
    report["output_dir"]["collision"] = collision

    if report["critical_failures"]:
        report["status"] = "fail"
        print("=" * 60)
        print("STAGE 9B.0 PREFLIGHT FAILED")
        print("=" * 60)
        for f in report["critical_failures"]:
            print(f"  [FAIL] {f}")
        if report["warnings"]:
            print("\nWarnings:")
            for w in report["warnings"]:
                print(f"  [WARN] {w}")
        print("\nFull report:")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        sys.exit(1)

    print("=" * 60)
    print("STAGE 9B.0 PREFLIGHT PASSED")
    print("=" * 60)
    print(f"Git commit : {git['commit']} (dirty={git['dirty']}, branch={git['branch']})")
    print(f"Origin/main: {origin_main}")
    print(f"Config     : {args.config}")
    print(f"Config hash: {report['config']['hash']}")
    print(f"Methods    : {args.methods}")
    print(f"Backend    : {args.backend}")
    print(f"Output dir : {output_dir}")
    if report["warnings"]:
        print("\nWarnings:")
        for w in report["warnings"]:
            print(f"  [WARN] {w}")
    print("\nFull report:")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
