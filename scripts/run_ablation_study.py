#!/usr/bin/env python3
"""
Run ablation studies recommended by the RL audit report.

Supported ablation types:
  reward_weights    : grid search over w_angle, w_turn_rate, w_overshoot
  dynamics_aware    : compare dynamics_aware=true/false
  offset_range      : compare dynamic_offset_scale values (null, 0.25, 0.5, 1.0)
  potential_shaping : compare potential_based_shaping.enabled=true/false
  observation       : compare observation enhancements (temporal, vp_error)

Example:
    python scripts/run_ablation_study.py \
        --type reward_weights \
        --config config/experiment/train_no_prediction_vpp_ppo.yaml \
        --steps 50000 \
        --seeds 3 \
        --device cpu
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from uav_vpp_guidance.utils.config import load_yaml_config, merge_config


BASE_CONFIG = Path("config/experiment/train_no_prediction_vpp_ppo.yaml")
DEFAULT_OUTPUT_ROOT = Path("outputs/ablation")

ABLATION_GRIDS = {
    "reward_weights": {
        "description": "Grid search over key reward weights",
        "axes": {
            "reward.w_angle": [0.5, 0.9, 1.5],
            "reward.w_turn_rate": [0.0, 0.1, 0.5],
            "reward.w_overshoot": [0.0, 0.5, 1.0],
        },
    },
    "dynamics_aware": {
        "description": "Compare dynamics-aware VPP constraint on/off",
        "axes": {
            "virtual_point.dynamics_aware": [True, False],
        },
    },
    "offset_range": {
        "description": "Compare dynamic offset scaling values",
        "axes": {
            "virtual_point.dynamic_offset_scale": [None, 0.25, 0.5, 1.0],
        },
    },
    "potential_shaping": {
        "description": "Compare potential-based reward shaping on/off",
        "axes": {
            "reward.potential_based_shaping.enabled": [True, False],
        },
    },
    "observation": {
        "description": "Compare observation-space enhancements",
        "axes": {
            "observation.temporal.enabled": [False, True],
            "observation.include_vp_error": [False, True],
        },
    },
}


def _set_nested(config: dict, key_path: str, value):
    """Set a nested config value from a dot-separated key path."""
    keys = key_path.split(".")
    node = config
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    node[keys[-1]] = value


def _make_override_name(override: dict) -> str:
    """Create a short directory name from an override dict."""
    parts = []
    for key, value in override.items():
        short_key = key.split(".")[-1]
        val_str = str(value).replace(".", "_") if value is not None else "null"
        parts.append(f"{short_key}={val_str}")
    return "__".join(parts)


def _load_config_recursive(path: Path, visited=None):
    """Load a YAML config and recursively resolve its ``includes``."""
    if visited is None:
        visited = set()
    path = path.resolve()
    if path in visited:
        raise ValueError(f"Cyclic include detected: {path}")
    visited.add(path)

    cfg = load_yaml_config(str(path))
    includes = cfg.pop("includes", [])
    merged = {}
    for inc in includes:
        inc_full = path.parent / inc
        if inc_full.exists():
            merged = merge_config(merged, _load_config_recursive(inc_full, visited))
    return merge_config(merged, cfg)


def generate_grid(axes: dict):
    """Generate the Cartesian product of ablation axes."""
    keys = list(axes.keys())
    values = [axes[k] for k in keys]

    def _recurse(index):
        if index == len(keys):
            yield {}
            return
        for value in values[index]:
            for rest in _recurse(index + 1):
                rest[keys[index]] = value
                yield rest

    return list(_recurse(0))


def run_single(
    base_path: Path,
    output_root: Path,
    override: dict,
    seed: int,
    total_timesteps: int,
    device: str,
    algorithm: str = "ppo",
    force: bool = False,
):
    """Run one ablation configuration with one seed."""
    cfg = _load_config_recursive(base_path)
    for key_path, value in override.items():
        _set_nested(cfg, key_path, value)

    cfg.setdefault("experiment", {})["seed"] = seed
    cfg.setdefault("ppo", {})["total_timesteps"] = total_timesteps

    run_name = _make_override_name(override)
    output_dir = output_root / run_name / f"seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = output_dir / "checkpoints" / "best.pt"
    if checkpoint.exists() and not force:
        print(f"[SKIP] {run_name} seed={seed}: checkpoint exists")
        return 0

    cfg_path = output_dir / "config.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    cmd = [
        sys.executable,
        "scripts/train_curriculum_ppo.py",
        "--config", str(cfg_path),
        "--seed", str(seed),
        "--output-dir", str(output_dir),
        "--device", device,
        "--algorithm", algorithm,
    ]

    print(f"\n[RUN] {run_name} seed={seed} -> {output_dir}")
    start = time.time()
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    elapsed = time.time() - start
    print(f"[DONE] {run_name} seed={seed} in {elapsed:.1f}s (exit={result.returncode})")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run RL audit ablation studies")
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=list(ABLATION_GRIDS.keys()),
        help="Ablation study type",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(BASE_CONFIG),
        help="Base config path",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Output root directory",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50000,
        help="Total timesteps per run",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
        help="Number of random seeds",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="ppo",
        help="Algorithm to run (ppo, cr_ppo, intentional_ppo)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing checkpoints",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the grid without running training",
    )
    args = parser.parse_args()

    grid_spec = ABLATION_GRIDS[args.type]
    overrides = generate_grid(grid_spec["axes"])

    base_path = Path(args.config)
    if args.output_root:
        output_root = Path(args.output_root)
    else:
        output_root = DEFAULT_OUTPUT_ROOT / args.type

    print(f"Ablation: {args.type}")
    print(f"Description: {grid_spec['description']}")
    print(f"Base config: {base_path}")
    print(f"Grid size: {len(overrides)} configs x {args.seeds} seeds = {len(overrides) * args.seeds} runs")
    print(f"Output root: {output_root}")

    if args.dry_run:
        print("\nDry run - generated grid:")
        for override in overrides:
            print(" ", _make_override_name(override))
        return 0

    tasks = [
        (base_path, output_root, override, seed, args.steps, args.device, args.algorithm, args.force)
        for override in overrides
        for seed in range(args.seeds)
    ]

    failures = 0
    if args.parallel <= 1:
        for t in tasks:
            rc = run_single(*t)
            failures += (rc != 0)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {executor.submit(run_single, *t): t for t in tasks}
            for future in as_completed(futures):
                rc = future.result()
                failures += (rc != 0)

    print(f"\nAll runs complete. Failures: {failures}/{len(tasks)}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
