#!/usr/bin/env python3
"""
Trained mechanism-removal ablation runner.

Trains and evaluates the three key mechanism-removal ablations:
  - full_vpp:           baseline with all mechanisms enabled
  - no_regret:          disables the regret-aware gain-update signal
  - no_gain_obs:        removes guidance gains from the policy observation
  - no_safety_penalty:  sets the safety reward weight to zero

The script generates temporary config YAMLs by merging a base configuration
with ablation-specific overrides, trains one PPO policy per variant/seed,
and then evaluates all variants on a shared scenario matrix.

Usage:
    # Full pipeline (train + evaluate + publish checkpoints)
    python scripts/run_mechanism_removal_ablation.py \
        --output-dir outputs/mechanism_removal_ablation

    # Train only
    python scripts/run_mechanism_removal_ablation.py --skip-eval

    # Evaluate existing checkpoints
    python scripts/run_mechanism_removal_ablation.py --skip-training \
        --checkpoint-map full_vpp=outputs/.../best.pt no_regret=.../best.pt ...

    # Smoke test (minimal steps, random-policy eval fallback allowed)
    python scripts/run_mechanism_removal_ablation.py --smoke
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uav_vpp_guidance.utils.config import load_yaml_config, merge_config


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_BASE_CONFIG = ROOT / "config" / "experiment" / "train_no_prediction_vpp_ppo.yaml"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "mechanism_removal_ablation"

VARIANTS = ["full_vpp", "no_regret", "no_gain_obs", "no_safety_penalty"]

# Override applied on top of the base config for each ablation.
# Note: gain_optimizer.use_regret is a legacy flag; the current PPO training
# pipeline does not implement a regret-aware gain update, so the no_regret
# variant is kept for protocol continuity and is documented as a no-op in the
# produced manifest.
ABLATION_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "full_vpp": {
        "experiment": {"name": "mechanism_full_vpp"},
        "observation": {"include_gains": True},
    },
    "no_regret": {
        "experiment": {"name": "mechanism_no_regret"},
        "observation": {"include_gains": True},
        "gain_optimizer": {"use_regret": False},
    },
    "no_gain_obs": {
        "experiment": {"name": "mechanism_no_gain_obs"},
        "observation": {"include_gains": False},
    },
    "no_safety_penalty": {
        "experiment": {"name": "mechanism_no_safety_penalty"},
        "observation": {"include_gains": True},
        "reward": {"w_safety": 0.0},
    },
}

PUBLISH_MAP = {
    "no_regret": ROOT / "outputs" / "experiments" / "ablation_no_regret",
    "no_gain_obs": ROOT / "outputs" / "experiments" / "ablation_no_gain_obs",
    "no_safety_penalty": ROOT / "outputs" / "experiments" / "ablation_no_safety_penalty",
}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursive dict merge (override wins)."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config_with_includes(path: Path) -> Dict[str, Any]:
    """Load a YAML config and recursively resolve its ``includes``."""
    config = load_yaml_config(str(path))
    includes = config.pop("includes", []) or []
    merged: Dict[str, Any] = {}
    for inc in includes:
        inc_path = (path.parent / inc).resolve()
        if inc_path.exists():
            merged = deep_merge(merged, load_config_with_includes(inc_path))
    return deep_merge(merged, config)


def dump_config(config: Dict[str, Any], path: Path) -> None:
    """Write a config dict to YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def run_training(
    base_config: Dict[str, Any],
    variant: str,
    seed: int,
    output_dir: Path,
    args: argparse.Namespace,
) -> Optional[Path]:
    """Train a single mechanism-removal variant. Returns checkpoint path or None."""
    variant_dir = output_dir / "train" / variant / f"seed{seed}"
    checkpoint_path = variant_dir / "checkpoints" / "best.pt"

    if args.skip_existing and checkpoint_path.exists():
        print(f"  [SKIP] {variant} seed {seed}: checkpoint exists")
        return checkpoint_path

    cfg = deep_merge(base_config, ABLATION_OVERRIDES[variant])
    cfg.setdefault("experiment", {})["seed"] = seed

    # Backend / device overrides
    if args.backend is not None:
        cfg["backend"] = args.backend
        cfg.setdefault("env", {})["backend"] = args.backend
        cfg.setdefault("env", {})["use_jsbsim"] = (args.backend == "jsbsim")
    if args.device is not None:
        cfg.setdefault("ppo", {})["device"] = args.device

    # Smoke-mode reduction is handled by the training script; we still need
    # a real config file.
    temp_cfg_path = variant_dir / "_train_config.yaml"
    dump_config(cfg, temp_cfg_path)

    cmd = [
        sys.executable,
        "-m",
        "uav_vpp_guidance.training.train_no_prediction_vpp_ppo",
        "--config", str(temp_cfg_path),
        "--seed", str(seed),
        "--output-dir", str(variant_dir),
    ]
    if args.backend is not None:
        cmd.extend(["--backend", args.backend])
    if args.device is not None:
        cmd.extend(["--device", args.device])
    if args.smoke:
        cmd.append("--smoke")

    print(f"\n[TRAIN] {variant} | seed {seed}")
    print(f"  output: {variant_dir}")
    print(f"  command: {' '.join(cmd)}")
    start = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"  [FAIL] Training failed for {variant} seed {seed} (exit {result.returncode})")
        return None

    print(f"  [OK] Trained in {elapsed/60:.1f}m")
    return checkpoint_path


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def build_eval_config(
    base_config: Dict[str, Any],
    checkpoint_map: Dict[str, Path],
    variants: List[str],
) -> Dict[str, Any]:
    """Build a unified evaluation config with one method per variant."""
    eval_config = copy.deepcopy(base_config)
    eval_config.setdefault("experiment", {})["name"] = "mechanism_removal_comparison"
    eval_config.setdefault("experiment", {})["mode"] = "eval"

    methods: Dict[str, Any] = {}
    for variant in variants:
        method_cfg: Dict[str, Any] = {
            "name": variant,
            "checkpoint": str(checkpoint_map[variant]),
            "trajectory_prediction": {"enabled": False},
            "virtual_point": {
                "anchor_mode": "current_target",
                "action_dim": 3,
                "d_long_range": [-1500.0, 1500.0],
                "d_lat_range": [-800.0, 800.0],
                "d_vert_range": [-500.0, 500.0],
                "smoothing_alpha": 0.3,
            },
        }
        # Apply ablation-specific overrides so evaluation uses the same
        # observation / reward settings as training.
        method_cfg = deep_merge(method_cfg, ABLATION_OVERRIDES[variant])
        methods[variant] = method_cfg

    eval_config["methods"] = methods
    return eval_config


def run_evaluation(
    base_config: Dict[str, Any],
    checkpoint_map: Dict[str, Path],
    variants: List[str],
    output_dir: Path,
    args: argparse.Namespace,
) -> bool:
    """Run the unified comparison evaluation."""
    eval_dir = output_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    eval_config = build_eval_config(base_config, checkpoint_map, variants)
    temp_cfg_path = eval_dir / "_eval_config.yaml"
    dump_config(eval_config, temp_cfg_path)

    seeds = args.eval_seeds
    if args.smoke:
        seeds = seeds[:1]

    cmd = [
        sys.executable,
        "-m",
        "uav_vpp_guidance.evaluation.evaluate_prediction_comparison",
        "--config", str(temp_cfg_path),
        "--backend", args.backend,
        "--episodes-per-scenario", str(args.episodes_per_scenario),
        "--output-dir", str(eval_dir),
    ]
    cmd.extend(["--seeds"] + [str(s) for s in seeds])
    cmd.extend(["--scenarios"] + args.scenarios)

    if args.smoke:
        # Smoke-mode checkpoints may not be useful; allow fallback so the
        # pipeline can still be exercised end-to-end.
        cmd.append("--allow-random-policy")

    print(f"\n[EVAL] Unified mechanism-removal comparison")
    print(f"  output: {eval_dir}")
    print(f"  command: {' '.join(cmd)}")
    start = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"  [FAIL] Evaluation failed (exit {result.returncode})")
        return False

    print(f"  [OK] Evaluated in {elapsed/60:.1f}m")
    return True


# ---------------------------------------------------------------------------
# Publishing / provenance
# ---------------------------------------------------------------------------


def publish_checkpoints(checkpoint_map: Dict[str, Path]) -> Dict[str, Optional[Path]]:
    """Copy variant checkpoints to the canonical paths expected by run_ablations.py."""
    published: Dict[str, Optional[Path]] = {}
    for variant, src in checkpoint_map.items():
        if variant not in PUBLISH_MAP:
            published[variant] = None
            continue
        dst_dir = PUBLISH_MAP[variant] / "checkpoints"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / "best.pt"
        if src.exists():
            shutil.copy2(src, dst)
            published[variant] = dst
            print(f"  [PUBLISH] {variant} -> {dst}")
        else:
            published[variant] = None
    return published


def write_manifest(
    output_dir: Path,
    base_config_path: Path,
    checkpoint_map: Dict[str, Path],
    published: Dict[str, Optional[Path]],
    eval_success: bool,
    variants: List[str],
    args: argparse.Namespace,
) -> None:
    """Write a JSON manifest documenting the ablation run."""
    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_config": str(base_config_path),
        "variants": {
            v: {
                "override": ABLATION_OVERRIDES[v],
                "checkpoints": {
                    "train_output": str(checkpoint_map.get(v)) if checkpoint_map.get(v) else None,
                    "published": str(published.get(v)) if published.get(v) else None,
                },
            }
            for v in variants
        },
        "training": {
            "seeds": args.seeds,
            "backend": args.backend,
            "device": args.device,
            "smoke": args.smoke,
        },
        "evaluation": {
            "seeds": args.eval_seeds,
            "episodes_per_scenario": args.episodes_per_scenario,
            "scenarios": args.scenarios,
            "success": eval_success,
            "output_dir": str(output_dir / "eval") if eval_success else None,
        },
        "notes": {
            "no_regret": (
                "The current PPO training pipeline does not implement a "
                "regret-aware gain update, so the no_regret override is a "
                "protocol-level placeholder. The trained checkpoint is "
                "identical to the full_vpp baseline in this codebase."
            ),
        },
    }
    manifest_path = output_dir / "mechanism_removal_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\n[MANIFEST] {manifest_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_checkpoint_map(items: List[str]) -> Dict[str, Path]:
    """Parse --checkpoint-map arguments."""
    mapping: Dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid checkpoint map '{item}', expected variant=path")
        key, value = item.split("=", 1)
        mapping[key.strip()] = Path(value.strip()).resolve()
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trained mechanism-removal ablation runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline
  python scripts/run_mechanism_removal_ablation.py

  # Train only
  python scripts/run_mechanism_removal_ablation.py --skip-eval

  # Evaluate previously trained checkpoints
  python scripts/run_mechanism_removal_ablation.py --skip-training \
      --checkpoint-map full_vpp=outputs/.../best.pt \
      --checkpoint-map no_safety_penalty=outputs/.../best.pt

  # Smoke test
  python scripts/run_mechanism_removal_ablation.py --smoke
        """,
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default=str(DEFAULT_BASE_CONFIG),
        help="Base training config YAML",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Root output directory",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=None,
        help="Subset of variants to run (default: all)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Training seeds",
    )
    parser.add_argument(
        "--eval-seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        help="Evaluation seeds",
    )
    parser.add_argument(
        "--episodes-per-scenario",
        type=int,
        default=50,
        help="Episodes per scenario during evaluation",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["favorable", "neutral", "disadvantage", "challenging"],
        help="Scenarios to evaluate",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="simple",
        choices=["simple", "jsbsim"],
        help="Simulation backend",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="PyTorch compute device override",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training and use provided or existing checkpoints",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation after training",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip training if the checkpoint already exists",
    )
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="Do not copy checkpoints to canonical outputs/experiments/ablation_* paths",
    )
    parser.add_argument(
        "--checkpoint-map",
        action="append",
        default=[],
        help="Override checkpoint path for a variant, e.g. no_regret=path/to/best.pt",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a minimal smoke test",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned run and exit without executing",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()

    base_config_path = Path(args.base_config).resolve()
    if not base_config_path.exists():
        print(f"ERROR: Base config not found: {base_config_path}")
        return 1

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = args.variants if args.variants else VARIANTS

    print("=" * 60)
    print("Mechanism-Removal Ablation Runner")
    print("=" * 60)
    print(f"Base config: {base_config_path}")
    print(f"Output dir:  {output_dir}")
    print(f"Variants:    {variants}")
    print(f"Train seeds: {args.seeds}")
    print(f"Eval seeds:  {args.eval_seeds}")
    print(f"Backend:     {args.backend}")
    print(f"Device:      {args.device or 'from config'}")
    print(f"Smoke:       {args.smoke}")
    print(f"Skip train:  {args.skip_training}")
    print(f"Skip eval:   {args.skip_eval}")
    print("=" * 60)

    base_config = load_config_with_includes(base_config_path)

    if args.dry_run:
        plan = {
            "status": "dry_run",
            "base_config": str(base_config_path),
            "variants": variants,
            "seeds": args.seeds,
            "eval_seeds": args.eval_seeds,
            "episodes_per_scenario": args.episodes_per_scenario,
            "scenarios": args.scenarios,
            "backend": args.backend,
            "device": args.device,
            "smoke": args.smoke,
        }
        plan_path = output_dir / "dry_run_plan.json"
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=2, ensure_ascii=False)
        print(f"\n[DRY-RUN] Plan written to {plan_path}")
        return 0

    # Resolve checkpoint map. If skipping training, all requested variants must
    # be provided (either via --checkpoint-map or by existing published paths).
    user_ckpt_map = parse_checkpoint_map(args.checkpoint_map)
    checkpoint_map: Dict[str, Path] = {}

    if not args.skip_training:
        print("\n[PHASE 1] Training mechanism-removal variants...")
        failures = []
        for variant in variants:
            for seed in args.seeds:
                ckpt = run_training(base_config, variant, seed, output_dir, args)
                if ckpt is None:
                    failures.append((variant, seed))
                elif seed == args.seeds[0]:
                    # Use the first seed as the representative checkpoint for
                    # evaluation and publishing.
                    checkpoint_map[variant] = ckpt
        if failures:
            print(f"\n[ERROR] Training failures: {failures}")
            return 1
        print("\n[PHASE 1] Training complete.")
    else:
        for variant in variants:
            if variant in user_ckpt_map:
                checkpoint_map[variant] = user_ckpt_map[variant]
            elif variant in PUBLISH_MAP:
                candidate = PUBLISH_MAP[variant] / "checkpoints" / "best.pt"
                if candidate.exists():
                    checkpoint_map[variant] = candidate
                else:
                    print(f"[ERROR] No checkpoint for {variant}; provide --checkpoint-map")
                    return 1
            else:
                print(f"[ERROR] No checkpoint for {variant}; provide --checkpoint-map")
                return 1

    # Publish checkpoints for downstream paper scripts.
    published: Dict[str, Optional[Path]] = {}
    if not args.no_publish:
        print("\n[PUBLISH] Copying checkpoints to canonical paths...")
        published = publish_checkpoints(checkpoint_map)

    eval_success = False
    if not args.skip_eval:
        print("\n[PHASE 2] Evaluating variants...")
        eval_success = run_evaluation(base_config, checkpoint_map, variants, output_dir, args)
        if not eval_success:
            print("\n[WARNING] Evaluation failed; manifest will reflect this.")

    write_manifest(output_dir, base_config_path, checkpoint_map, published, eval_success, variants, args)

    print("\n" + "=" * 60)
    print("Mechanism-removal ablation runner finished.")
    print("=" * 60)
    return 0 if (args.skip_eval or eval_success) else 1


if __name__ == "__main__":
    sys.exit(main())
