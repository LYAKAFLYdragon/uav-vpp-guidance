#!/usr/bin/env python3
"""
Verify a machine is ready to run the distributed experiment campaign.

Checks:
- Python version (>=3.8)
- Required packages (torch, numpy, pyyaml, scipy, stable-baselines3 if used)
- Project root and script paths
- CPU cores and available memory
- GPU availability (optional, informational)
- Git status and commit
- Write permissions in outputs/

Usage:
    python scripts/verify_machine_setup.py
    python scripts/verify_machine_setup.py --manifest scripts/distributed_manifest.json
"""
import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()


def check_python_version():
    v = sys.version_info
    ok = v.major == 3 and v.minor >= 8
    return ok, f"Python {v.major}.{v.minor}.{v.micro}", None


def check_packages():
    required = [
        "numpy",
        "yaml",
        "scipy",
        "torch",
        "matplotlib",
        "pandas",
    ]
    missing = []
    versions = {}
    for pkg in required:
        try:
            mod = importlib.import_module(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            missing.append(pkg)
    ok = len(missing) == 0
    return ok, f"packages ok={ok}, missing={missing}", {"missing": missing, "versions": versions}


def check_paths():
    checks = {
        "src": ROOT / "src" / "uav_vpp_guidance",
        "config": ROOT / "config" / "experiment",
        "scripts": ROOT / "scripts",
        "outputs": ROOT / "outputs",
    }
    missing = [name for name, path in checks.items() if not path.exists()]
    ok = len(missing) == 0
    return ok, f"paths ok={ok}, missing={missing}", {"missing": missing}


def check_write_permissions():
    try:
        test_file = ROOT / "outputs" / ".setup_test_write"
        test_file.write_text("ok")
        test_file.unlink()
        return True, "outputs/ writable", None
    except Exception as e:
        return False, f"outputs/ not writable: {e}", None


def check_cpu_memory():
    try:
        import psutil
        cpu_count = psutil.cpu_count(logical=True)
        mem_gb = psutil.virtual_memory().total / (1024**3)
        ok = cpu_count >= 4 and mem_gb >= 8
        return ok, f"CPU cores={cpu_count}, RAM={mem_gb:.1f}GB", {"cpu_count": cpu_count, "memory_gb": mem_gb}
    except ImportError:
        return None, "psutil not installed; cannot check CPU/memory", None


def check_gpu():
    try:
        import torch
        available = torch.cuda.is_available()
        count = torch.cuda.device_count() if available else 0
        names = [torch.cuda.get_device_name(i) for i in range(count)] if available else []
        return True, f"GPU available={available}, count={count}, names={names}", {"available": available, "count": count, "names": names}
    except Exception as e:
        return True, f"GPU check skipped (torch import failed: {e})", None


def check_git():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, text=True).strip()
        status = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).strip()
        dirty = len(status) > 0
        return not dirty, f"git branch={branch}, commit={commit[:8]}, dirty={dirty}", {"branch": branch, "commit": commit, "dirty": dirty}
    except Exception as e:
        return None, f"git check failed: {e}", None


def check_manifest(manifest_path):
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        total = sum(len(g.get("experiments", [])) for g in manifest.get("groups", []))
        return True, f"manifest valid: {len(manifest['groups'])} groups, {total} experiments", {"groups": len(manifest["groups"]), "experiments": total}
    except Exception as e:
        return False, f"manifest invalid: {e}", None


def main():
    parser = argparse.ArgumentParser(description="Verify machine setup for distributed experiments")
    parser.add_argument("--manifest", type=str, default=str(ROOT / "scripts" / "distributed_manifest.json"))
    parser.add_argument("--output", type=str, default=None, help="Write report to JSON file")
    args = parser.parse_args()

    checks = [
        ("Python version", check_python_version),
        ("Required packages", check_packages),
        ("Project paths", check_paths),
        ("Write permissions", check_write_permissions),
        ("CPU/Memory", check_cpu_memory),
        ("GPU", check_gpu),
        ("Git status", check_git),
        ("Manifest", lambda: check_manifest(args.manifest)),
    ]

    print("=" * 70)
    print("MACHINE SETUP VERIFICATION")
    print("=" * 70)

    report = {"checks": {}, "overall_ready": True, "warnings": []}
    for name, fn in checks:
        try:
            ok, msg, detail = fn()
        except Exception as e:
            ok = False
            msg = f"exception: {e}"
            detail = None

        status = "PASS" if ok else ("WARN" if ok is None else "FAIL")
        if ok is False:
            report["overall_ready"] = False
        elif ok is None:
            report["warnings"].append(name)

        print(f"[{status:4s}] {name:20s}: {msg}")
        report["checks"][name] = {
            "status": status,
            "message": msg,
            "detail": detail,
        }

    print("=" * 70)
    if report["overall_ready"]:
        print("RESULT: Machine is READY for distributed experiments.")
    else:
        print("RESULT: Machine is NOT READY. Fix FAIL items above.")
    if report["warnings"]:
        print(f"WARNINGS: {', '.join(report['warnings'])}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {args.output}")

    sys.exit(0 if report["overall_ready"] else 1)


if __name__ == "__main__":
    main()
