"""Helpers for retrospective combat-finetune checkpoint comparisons."""

from __future__ import annotations

import argparse
import copy
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import yaml

from uav_vpp_guidance.training.train_prediction_vpp_ppo import load_experiment_config


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (_repo_root() / path).resolve()


def discover_checkpoints(
    checkpoint_dir: Path,
    *,
    steps: Optional[Sequence[int]] = None,
    include_best: bool = True,
    include_last: bool = True,
) -> List[Dict[str, str]]:
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    discovered: List[Dict[str, str]] = []
    step_files = {
        int(match.group(1)): path
        for path in checkpoint_dir.glob("step_*.pt")
        if (match := re.fullmatch(r"step_(\d+)\.pt", path.name)) is not None
    }

    selected_steps = sorted(step_files.keys()) if steps is None else sorted(set(int(step) for step in steps))
    for step in selected_steps:
        path = step_files.get(step)
        if path is None:
            raise FileNotFoundError(
                f"Requested retrospective step checkpoint not found: step_{step}.pt"
            )
        discovered.append(
            {
                "tag": f"step_{step}",
                "label_suffix": f"step {step}",
                "checkpoint": str(path.resolve()),
            }
        )

    if include_best:
        best_path = checkpoint_dir / "best.pt"
        if not best_path.exists():
            raise FileNotFoundError(f"Requested retrospective checkpoint not found: {best_path}")
        discovered.append(
            {
                "tag": "best",
                "label_suffix": "best",
                "checkpoint": str(best_path.resolve()),
            }
        )

    if include_last:
        last_path = checkpoint_dir / "last.pt"
        if not last_path.exists():
            raise FileNotFoundError(f"Requested retrospective checkpoint not found: {last_path}")
        discovered.append(
            {
                "tag": "last",
                "label_suffix": "last",
                "checkpoint": str(last_path.resolve()),
            }
        )

    return discovered


def build_retrospective_config(
    template_config: Dict[str, Any],
    *,
    retrospective_method: str,
    checkpoint_specs: Sequence[Dict[str, str]],
    experiment_name: Optional[str] = None,
) -> Dict[str, Any]:
    if retrospective_method not in template_config.get("methods", {}):
        raise KeyError(f"Unknown retrospective method: {retrospective_method}")

    cfg = copy.deepcopy(template_config)
    methods = cfg.setdefault("methods", {})
    template_method = copy.deepcopy(methods[retrospective_method])
    template_label = str(template_method.get("label", retrospective_method))

    run_defaults = cfg.setdefault("run_defaults", {})
    base_main_methods = [
        method_name
        for method_name in list(run_defaults.get("main_methods", []))
        if method_name != retrospective_method
    ]

    methods.pop(retrospective_method, None)
    generated_method_names: List[str] = []
    for spec in checkpoint_specs:
        tag = str(spec["tag"])
        method_name = f"{retrospective_method}__{tag}".replace("-", "_")
        generated_method = copy.deepcopy(template_method)
        generated_method["checkpoint"] = str(spec["checkpoint"])
        generated_method["label"] = f"{template_label} ({spec['label_suffix']})"
        methods[method_name] = generated_method
        generated_method_names.append(method_name)

    run_defaults["main_methods"] = base_main_methods + generated_method_names

    cfg.setdefault("experiment", {})
    current_name = str(cfg["experiment"].get("name", "combat_checkpoint_retrospective"))
    cfg["experiment"]["name"] = (
        experiment_name or f"{current_name}_retrospective"
    )
    cfg["experiment"]["description"] = (
        f"Retrospective checkpoint comparison generated from method "
        f"'{retrospective_method}'."
    )
    return cfg


def write_retrospective_config(config: Dict[str, Any], output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            config,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
    return output_path


def _append_optional_args(command: List[str], flag: str, values: Optional[Iterable[Any]]) -> None:
    if values is None:
        return
    items = [str(value) for value in values]
    if not items:
        return
    command.append(flag)
    command.extend(items)


def run_retrospective_comparison(args: argparse.Namespace) -> int:
    template_path = _resolve_path(args.template_config)
    checkpoint_dir = _resolve_path(args.checkpoint_dir)
    output_config_path = (
        _resolve_path(args.output_config)
        if args.output_config
        else (
            _repo_root()
            / "outputs"
            / "diagnostics"
            / f"{args.run_id}_generated_comparison.yaml"
        )
    )

    template_config = load_experiment_config(str(template_path))
    checkpoint_specs = discover_checkpoints(
        checkpoint_dir,
        steps=args.steps,
        include_best=not args.no_include_best,
        include_last=not args.no_include_last,
    )
    generated_config = build_retrospective_config(
        template_config,
        retrospective_method=args.retrospective_method,
        checkpoint_specs=checkpoint_specs,
        experiment_name=args.experiment_name,
    )
    write_retrospective_config(generated_config, output_config_path)

    if args.dry_run:
        print(f"Generated retrospective comparison config: {output_config_path}")
        return 0

    comparison_script = _resolve_path(args.comparison_script)
    command: List[str] = [
        args.python_exe,
        "-u",
        str(comparison_script),
        "--config",
        str(output_config_path),
        "--run-id",
        str(args.run_id),
        "--run-status",
        str(args.run_status),
        "--backend",
        str(args.backend),
        "--opponent-stage",
        str(args.opponent_stage),
        "--output-root",
        str(args.output_root),
    ]

    if args.attack_zone_close_range_max_aoa_deg is not None:
        command.extend(
            [
                "--attack-zone-close-range-max-aoa-deg",
                str(args.attack_zone_close_range_max_aoa_deg),
            ]
        )

    _append_optional_args(command, "--tasks", args.tasks)
    _append_optional_args(command, "--seeds", args.seeds)

    print("Running retrospective checkpoint comparison...")
    print("Command:", " ".join(command))
    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and optionally run a retrospective checkpoint comparison."
    )
    parser.add_argument("--template-config", required=True)
    parser.add_argument("--retrospective-method", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--output-config", default=None)
    parser.add_argument("--steps", nargs="*", type=int, default=None)
    parser.add_argument("--no-include-best", action="store_true")
    parser.add_argument("--no-include-last", action="store_true")
    parser.add_argument("--comparison-script", default="scripts/run_jsbsim_hrl_comparison.py")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--run-status", default="formal-small")
    parser.add_argument("--backend", default="jsbsim")
    parser.add_argument("--opponent-stage", default="expert")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--attack-zone-close-range-max-aoa-deg", type=float, default=60.0)
    parser.add_argument("--output-root", default="outputs/jsbsim_hrl_comparison")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run_retrospective_comparison(args)

