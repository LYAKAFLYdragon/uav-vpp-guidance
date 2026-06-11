#!/usr/bin/env python3
"""
Distributed experiment runner for a single machine.

Reads distributed_manifest.json, selects a subset of experiments by --machine-id
or --experiment-ids, and executes them sequentially. Generates a run_manifest.json
for each experiment and a machine-level summary.

Usage:
    python scripts/run_machine_subset.py --machine-id 1
    python scripts/run_machine_subset.py --experiment-ids A1 B1 F1
    python scripts/run_machine_subset.py --group A --group B
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
MANIFEST_PATH = ROOT / "scripts" / "distributed_manifest.json"


def load_manifest(path=MANIFEST_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_experiments(manifest):
    """Return flat list of all experiments from all groups."""
    exps = []
    for group in manifest.get("groups", []):
        for exp in group.get("experiments", []):
            exp = dict(exp)
            exp["group_id"] = group["group_id"]
            exp["group_name"] = group["name"]
            exps.append(exp)
    return exps


def select_experiments(manifest, args):
    exps = flatten_experiments(manifest)
    selected = []
    if args.experiment_ids:
        ids = set(args.experiment_ids)
        selected = [e for e in exps if e["id"] in ids]
    elif args.groups:
        gids = set(args.groups)
        selected = [e for e in exps if e["group_id"] in gids]
    elif args.machine_id is not None:
        selected = assign_by_machine_id(exps, args.machine_id, args.total_machines)
    else:
        raise ValueError("Must specify --machine-id, --experiment-ids, or --group")
    return selected


def assign_by_machine_id(exps, machine_id, total_machines):
    """Round-robin assign experiments to machines by index."""
    if machine_id < 0 or machine_id >= total_machines:
        raise ValueError(f"machine_id must be in [0, {total_machines-1}]")
    # Keep order stable; assign by hashing experiment id
    assigned = []
    for exp in exps:
        h = hash(exp["id"]) % total_machines
        if h == machine_id:
            assigned.append(exp)
    return assigned


def make_config_with_overrides(base_config, overrides):
    """Deep merge overrides into base config and write to temp file."""
    import yaml
    with open(base_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    def merge(base, over):
        for k, v in over.items():
            if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                merge(base[k], v)
            else:
                base[k] = v

    merge(cfg, overrides)

    fd, tmp_path = tempfile.mkstemp(suffix=".yaml", prefix="config_override_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return tmp_path


def build_command(exp, seed):
    """Build shell command for a single seed of an experiment."""
    cmd = exp["script"].split()

    # If script is a python module invocation, we need config
    if "config" in exp and any("train" in c for c in cmd):
        config_path = ROOT / exp["config"]
        if "override_config" in exp:
            config_path = make_config_with_overrides(config_path, exp["override_config"])
        cmd.extend(["--config", str(config_path)])
        cmd.extend(["--seed", str(seed)])
        output_dir = exp["output_dir"].format(seed=seed)
        cmd.extend(["--output-dir", str(ROOT / output_dir)])
    elif "input_checkpoint" in exp and "evaluate" in exp["script"]:
        # Evaluation scripts typically take --checkpoint
        cmd.extend(["--checkpoint", str(ROOT / exp["input_checkpoint"])])
        if "scales" in exp:
            cmd.extend(["--scales"] + [str(s) for s in exp["scales"]])
        if "episodes_per_scale" in exp:
            cmd.extend(["--episodes", str(exp["episodes_per_scale"])])
    elif "input_checkpoint" in exp and "compute_capture_region" in exp["script"]:
        cmd.extend(["--checkpoint", str(ROOT / exp["input_checkpoint"])])
    elif "input_checkpoint" in exp and "benchmark_inference_time" in exp["script"]:
        cmd.extend(["--checkpoint", str(ROOT / exp["input_checkpoint"])])

    return cmd, config_path if "override_config" in exp else None


def run_experiment(exp, args):
    """Run all seeds of a single experiment sequentially."""
    print(f"\n{'='*70}")
    print(f"[{exp['group_id']}{exp['id']}] {exp['name']}")
    print(f"Description: {exp['description']}")
    print(f"Paper claim: {exp.get('paper_claim', 'N/A')}")
    print(f"{'='*70}")

    seeds = exp.get("seeds", [None])
    tmp_configs = []
    results = []

    for seed in seeds:
        cmd, tmp_cfg = build_command(exp, seed)
        if tmp_cfg:
            tmp_configs.append(tmp_cfg)

        print(f"\n>>> Seed {seed}: {' '.join(cmd)}")
        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd,
                cwd=ROOT,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            elapsed = time.time() - t0
            success = proc.returncode == 0
            print(proc.stdout)
            print(f"<<< Seed {seed}: {'OK' if success else f'FAIL({proc.returncode})'} ({elapsed:.0f}s)")
            results.append({
                "seed": seed,
                "success": success,
                "returncode": proc.returncode,
                "elapsed_seconds": elapsed,
                "command": " ".join(cmd),
            })

            if not success and not args.continue_on_error:
                print("Stopping due to failure. Use --continue-on-error to keep going.")
                break
        except Exception as e:
            print(f"<<< Seed {seed}: EXCEPTION {e}")
            results.append({"seed": seed, "success": False, "exception": str(e)})
            if not args.continue_on_error:
                break

    # Cleanup temporary config files
    for tmp in tmp_configs:
        try:
            os.remove(tmp)
        except OSError:
            pass

    # Write run_manifest.json
    output_base = ROOT / "outputs" / "distributed_runs"
    output_base.mkdir(parents=True, exist_ok=True)
    manifest_path = output_base / f"{exp['id']}_run_manifest.json"
    run_manifest = {
        "experiment_id": exp["id"],
        "experiment_name": exp["name"],
        "group": exp["group_id"],
        "machine_id": args.machine_id,
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": get_git_commit(),
        "results": results,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)
    print(f"Run manifest saved to {manifest_path}")

    return all(r.get("success", False) for r in results)


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Run a subset of experiments on one machine")
    parser.add_argument("--machine-id", type=int, default=None, help="Machine ID (0-based)")
    parser.add_argument("--total-machines", type=int, default=5, help="Total number of machines")
    parser.add_argument("--experiment-ids", nargs="+", default=None, help="Specific experiment IDs to run")
    parser.add_argument("--group", nargs="+", dest="groups", default=None, help="Run all experiments in group(s)")
    parser.add_argument("--manifest", type=str, default=str(MANIFEST_PATH), help="Path to distributed_manifest.json")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue running next seed after failure")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    selected = select_experiments(manifest, args)

    print(f"\nSelected {len(selected)} experiments for this machine/run")
    for exp in selected:
        print(f"  {exp['group_id']}{exp['id']:3s} {exp['name']:30s} seeds={exp.get('seeds', ['N/A'])}")

    if args.dry_run:
        print("\nDry run mode - commands that would be executed:")
        for exp in selected:
            for seed in exp.get("seeds", [None]):
                cmd, _ = build_command(exp, seed)
                print("  " + " ".join(cmd))
        return

    summary = []
    for exp in selected:
        ok = run_experiment(exp, args)
        summary.append({"id": exp["id"], "name": exp["name"], "all_success": ok})

    print("\n" + "="*70)
    print("MACHINE RUN SUMMARY")
    print("="*70)
    for s in summary:
        status = "PASS" if s["all_success"] else "FAIL"
        print(f"  {status:4s} {s['id']:4s} {s['name']}")

    # Write machine summary
    output_base = ROOT / "outputs" / "distributed_runs"
    summary_path = output_base / f"machine_{args.machine_id or 'manual'}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "machine_id": args.machine_id,
            "total_machines": args.total_machines,
            "experiments": summary,
        }, f, indent=2)
    print(f"\nMachine summary saved to {summary_path}")


if __name__ == "__main__":
    main()
