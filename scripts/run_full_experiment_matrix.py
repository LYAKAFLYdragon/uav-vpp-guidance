#!/usr/bin/env python3
"""
Full Experiment Matrix Scheduler.

Automates execution of all 5 experiment groups (A-E) across 3 phases:
  Phase 1: Asset production (training checkpoints)
  Phase 2: Evaluation execution (ablation, sweep, comparison)
  Phase 3: Analysis and paper material generation

Usage:
    # Full pipeline (train + evaluate + analyze)
    python scripts/run_full_experiment_matrix.py \
        --output-dir outputs/full_experiment \
        --gpu 0 \
        --max-gpu-hours 200

    # Only Phase 1 (training assets)
    python scripts/run_full_experiment_matrix.py --phase 1

    # Only Phase 2 (evaluation, checkpoints must exist)
    python scripts/run_full_experiment_matrix.py --phase 2

    # Only specific experiment group
    python scripts/run_full_experiment_matrix.py --only-group A

    # Resume interrupted run
    python scripts/run_full_experiment_matrix.py --resume \
        --state-file outputs/full_experiment/progress.json

    # Smoke test (fast verification)
    python scripts/run_full_experiment_matrix.py --smoke

    # Dry-run (print task list, do not execute)
    python scripts/run_full_experiment_matrix.py --dry-run
"""

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
CONFIG_DIR = PROJECT_ROOT / "config" / "experiment"

# Training config files
TRAIN_CONFIGS = {
    "no_pred": CONFIG_DIR / "train_no_prediction_vpp_ppo.yaml",
    "cv": CONFIG_DIR / "train_vpp_ppo_cv.yaml",
    "ca": CONFIG_DIR / "train_vpp_ppo_ca.yaml",
    "lstm_frozen": CONFIG_DIR / "train_vpp_ppo_lstm_frozen.yaml",
    "gru_frozen": CONFIG_DIR / "train_vpp_ppo_gru_frozen.yaml",
    "no_vpp": CONFIG_DIR / "train_no_vpp_ppo.yaml",
    "e2e": CONFIG_DIR / "train_end_to_end_ppo.yaml",
}

# Evaluation / other configs
EVAL_CONFIGS = {
    "ablation_scenarios": CONFIG_DIR / "evaluate_vpp_prediction_comparison.yaml",
    "ablation_checkpoints": CONFIG_DIR / "ablation_checkpoints.yaml",
    "maneuver_base": CONFIG_DIR / "evaluate_vpp_prediction_comparison.yaml",
    "gain_base": CONFIG_DIR / "train_no_prediction_vpp_ppo.yaml",
    "jsbsim": CONFIG_DIR / "stage6f5_feasible_geometry.yaml",
}

# Training modules (for python -m invocation)
TRAIN_MODULES = {
    "no_pred": "uav_vpp_guidance.training.train_no_prediction_vpp_ppo",
    "prediction": "uav_vpp_guidance.training.train_prediction_vpp_ppo",
    "e2e": "uav_vpp_guidance.training.train_end_to_end_ppo",
    "fixed_gain": "uav_vpp_guidance.training.train_fixed_gain",
}

# Default seeds
DEFAULT_TRAIN_SEEDS = [0, 1, 2]
DEFAULT_EVAL_SEEDS = list(range(10))

# Estimated task durations (minutes) for ETA calculation
ESTIMATED_DURATIONS = {
    "train_ppo": 90,        # ~1.5h for 200K steps
    "train_ppo_smoke": 3,   # ~3min for smoke
    "predictor_grid": 180,  # ~3h for 27-config grid search
    "predictor_grid_smoke": 5,
    "ablation_eval": 60,    # ~1h for 5 methods × 8 scn × 10 seeds
    "ablation_eval_smoke": 3,
    "maneuver_sweep": 300,  # ~5h for full sweep
    "maneuver_sweep_smoke": 5,
    "gain_comparison": 180, # ~3h for 3 configs × 3 seeds
    "gain_comparison_smoke": 5,
    "jsbsim_eval": 30,      # ~30min for JSBSim eval
    "jsbsim_eval_smoke": 5,
    "analysis": 10,
    "analysis_smoke": 2,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """A single executable task in the experiment matrix."""
    id: str
    name: str
    phase: int
    group: str
    cmd: List[str]
    depends_on: List[str] = field(default_factory=list)
    output_files: List[str] = field(default_factory=list)
    estimated_minutes: float = 10.0
    checkpoint_hint: Optional[str] = None  # human-readable checkpoint path

    def __post_init__(self):
        # Normalize output files to absolute paths
        self.output_files = [str(Path(f).resolve()) for f in self.output_files]


@dataclass
class TaskResult:
    """Result of executing a single task."""
    task_id: str
    status: str  # "success", "failed", "skipped"
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.0
    error_message: str = ""


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def _print_banner(text: str, width: int = 60):
    print("\n" + "=" * width)
    print(text.center(width))
    print("=" * width + "\n")


def _get_default_device() -> str:
    """Auto-detect GPU availability for default device selection."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _check_gpu_available(gpu_id: int) -> bool:
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        return gpu_id < torch.cuda.device_count()
    except Exception:
        return False


def _get_gpu_name(gpu_id: int) -> str:
    try:
        import torch
        return torch.cuda.get_device_name(gpu_id)
    except Exception:
        return "unknown"


def _release_gpu_memory():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _resolve_config_path(name: str) -> str:
    """Resolve a config key to an absolute path."""
    if name in TRAIN_CONFIGS:
        return str(TRAIN_CONFIGS[name].resolve())
    if name in EVAL_CONFIGS:
        return str(EVAL_CONFIGS[name].resolve())
    p = Path(name)
    if p.exists():
        return str(p.resolve())
    raise ValueError(f"Unknown config: {name}")


# ---------------------------------------------------------------------------
# Task graph builder
# ---------------------------------------------------------------------------

def build_task_graph(args: argparse.Namespace) -> List[Task]:
    """Build the complete task graph based on CLI arguments."""
    tasks: List[Task] = []
    output_dir = Path(args.output_dir).resolve()
    smoke = args.smoke
    device = args.device
    backend = getattr(args, "backend", "jsbsim")

    # Duration multiplier for smoke mode
    dur_mult = 0.1 if smoke else 1.0

    def _dur(key: str) -> float:
        return ESTIMATED_DURATIONS.get(key, 10.0) * dur_mult

    def _out(*parts: str) -> str:
        return str(output_dir.joinpath(*parts))

    # ========================================================================
    # PHASE 1: Training
    # ========================================================================
    if args.phase in (0, 1):
        # -------------------------------------------------------------------
        # Group A: 5 prediction methods × 3 seeds
        # -------------------------------------------------------------------
        if args.only_group in (None, "A"):
            group_a_methods = [
                ("no_pred", "no_pred", TRAIN_CONFIGS["no_pred"], None, None),
                ("cv", "prediction", TRAIN_CONFIGS["cv"], "constant_velocity", None),
                ("ca", "prediction", TRAIN_CONFIGS["ca"], "constant_acceleration", None),
                ("lstm", "prediction", TRAIN_CONFIGS["lstm_frozen"], "lstm", "PRED_LSTM_BEST"),
                ("gru", "prediction", TRAIN_CONFIGS["gru_frozen"], "gru", "PRED_GRU_BEST"),
            ]
            for method_key, module_key, config_path, predictor_type, pred_dep in group_a_methods:
                for seed in DEFAULT_TRAIN_SEEDS:
                    task_id = f"A_train_{method_key}_s{seed}"
                    out_subdir = _out("phase1_training", "experiment_A", f"{method_key}_s{seed}")
                    cmd = [
                        sys.executable, "-m", TRAIN_MODULES[module_key],
                        "--config", str(config_path),
                        "--seed", str(seed),
                        "--output-dir", out_subdir,
                        "--device", device,
                        "--backend", backend,
                    ]
                    if smoke:
                        cmd.append("--smoke")
                    if predictor_type:
                        cmd.extend(["--predictor-type", predictor_type])
                    if method_key in ("lstm", "gru"):
                        # These will get checkpoint added at runtime by executor
                        pass

                    depends = []
                    if pred_dep:
                        depends.append(pred_dep)

                    checkpoint_file = os.path.join(out_subdir, "checkpoints", "best.pt")
                    tasks.append(Task(
                        id=task_id,
                        name=f"Group A: Train {method_key} (seed {seed})",
                        phase=1,
                        group="A",
                        cmd=cmd,
                        depends_on=depends,
                        output_files=[checkpoint_file],
                        estimated_minutes=_dur("train_ppo_smoke" if smoke else "train_ppo"),
                        checkpoint_hint=checkpoint_file,
                    ))

        # -------------------------------------------------------------------
        # Group C: 4 architectures × 3 seeds
        # -------------------------------------------------------------------
        if args.only_group in (None, "C"):
            group_c_archs = [
                ("vpp", "prediction", TRAIN_CONFIGS["cv"], None),
                ("no_vpp", "no_pred", TRAIN_CONFIGS["no_vpp"], None),
                ("e2e", "e2e", TRAIN_CONFIGS["e2e"], None),
                ("no_pred", "no_pred", TRAIN_CONFIGS["no_pred"], None),
            ]
            for arch_key, module_key, config_path, extra in group_c_archs:
                for seed in DEFAULT_TRAIN_SEEDS:
                    task_id = f"C_train_{arch_key}_s{seed}"
                    out_subdir = _out("phase1_training", "experiment_C", f"{arch_key}_s{seed}")
                    cmd = [
                        sys.executable, "-m", TRAIN_MODULES[module_key],
                        "--config", str(config_path),
                        "--seed", str(seed),
                        "--output-dir", out_subdir,
                        "--device", device,
                        "--backend", backend,
                    ]
                    if smoke:
                        cmd.append("--smoke")

                    checkpoint_file = os.path.join(out_subdir, "checkpoints", "best.pt")
                    tasks.append(Task(
                        id=task_id,
                        name=f"Group C: Train {arch_key} (seed {seed})",
                        phase=1,
                        group="C",
                        cmd=cmd,
                        depends_on=[],
                        output_files=[checkpoint_file],
                        estimated_minutes=_dur("train_ppo_smoke" if smoke else "train_ppo"),
                        checkpoint_hint=checkpoint_file,
                    ))

        # -------------------------------------------------------------------
        # Predictor pre-training: LSTM + GRU grid search
        # -------------------------------------------------------------------
        if args.only_group in (None, "A"):  # Predictors support Group A
            for model_type in ("lstm", "gru"):
                task_id = f"PRED_{model_type.upper()}_GRID"
                out_subdir = _out("phase1_training", "predictor_training", f"{model_type}_grid_search")
                # Use a reduced grid: 3×3×3 = 27 configs
                cmd = [
                    sys.executable, str(PROJECT_ROOT / "scripts" / "grid_search_lstm.py"),
                    "--data_source", "outputs/trajectories/*.csv",
                    "--source_type", "tracking_env",
                    "--tracking_env_scenarios", "50" if not smoke else "5",
                    "--model_type", model_type,
                    "--hidden_dims", "64", "128", "256",
                    "--num_layers_list", "1", "2", "3",
                    "--dropouts", "0.0", "0.1", "0.2",
                    "--history_lens", "10",
                    "--prediction_horizons", "5",
                    "--epochs", "50" if not smoke else "2",
                    "--patience", "5",
                    "--batch_size", "32",
                    "--device", device,
                    "--output_dir", out_subdir,
                    "--exp_name", "best",
                    "--seed", "42",
                ]
                # Best model is written to a known path by the grid search script
                best_model = os.path.join(out_subdir, "best_model.pt")
                tasks.append(Task(
                    id=task_id,
                    name=f"Predictor: {model_type.upper()} grid search (27 configs)",
                    phase=1,
                    group="PRED",
                    cmd=cmd,
                    depends_on=[],
                    output_files=[best_model],
                    estimated_minutes=_dur("predictor_grid_smoke" if smoke else "predictor_grid"),
                    checkpoint_hint=best_model,
                ))

            # Train the best predictor configuration to produce a checkpoint
            for model_type in ("lstm", "gru"):
                task_id = f"PRED_{model_type.upper()}_BEST"
                grid_out = _out("phase1_training", "predictor_training", f"{model_type}_grid_search")
                best_out = _out("phase1_training", "predictor_training", f"{model_type}_best")
                best_model = os.path.join(best_out, "best_model.pt")

                # Inline script: read best_config.json from grid search, then train
                # P4 CONFIRMED: train_pipeline.py is located at
                # src/uav_vpp_guidance/trajectory_prediction/train_pipeline.py
                # and is runnable via `python -m` because the package directory
                # contains __init__.py. Verified with:
                #   python -m uav_vpp_guidance.trajectory_prediction.train_pipeline --help
                train_script = f"""
import json, subprocess, sys, os
from pathlib import Path

grid_dir = Path({grid_out!r})

# grid_search_lstm.py creates a timestamp subdirectory; find it
best_cfg_path = None
for item in grid_dir.iterdir():
    if item.is_dir():
        candidate = item / "best_config.json"
        if candidate.exists():
            best_cfg_path = candidate
            break

if best_cfg_path is None:
    # Fallback: try direct path
    direct = grid_dir / "best_config.json"
    if direct.exists():
        best_cfg_path = direct
    else:
        print(f"ERROR: best_config.json not found under {{grid_dir}}")
        sys.exit(1)

with open(best_cfg_path, "r", encoding="utf-8") as f:
    best = json.load(f)

out_dir = Path({best_out!r})
out_dir.mkdir(parents=True, exist_ok=True)

cmd = [
    sys.executable, "-m", "uav_vpp_guidance.trajectory_prediction.train_pipeline",
    "--config", "config/trajectory_prediction.yaml",
    "--data-dir", "outputs/trajectories",
    "--model-type", {model_type!r},
    "--output-dir", str(out_dir),
    "--seed", "42",
    "--epochs", "50" if not {smoke!r} else "2",
    "--batch-size", str(best.get("batch_size", 32)),
]
print(f"[Predictor] Training best {model_type.upper()} with config:")
print(json.dumps(best, indent=2))
result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print(result.stderr)
    sys.exit(result.returncode)

# Rename the output checkpoint to best_model.pt if needed
ckpt_files = list(out_dir.glob("*.pt"))
if ckpt_files:
    import shutil
    src = ckpt_files[0]
    dst = out_dir / "best_model.pt"
    if str(src.resolve()) != str(dst.resolve()):
        shutil.copy(src, dst)
    print(f"[Predictor] Checkpoint saved to {{dst}}")
else:
    print("WARNING: No .pt checkpoint found after training")
"""
                tasks.append(Task(
                    id=task_id,
                    name=f"Predictor: Train best {model_type.upper()} from grid-search config",
                    phase=1,
                    group="PRED",
                    cmd=[sys.executable, "-c", train_script],
                    depends_on=[f"PRED_{model_type.upper()}_GRID"],
                    output_files=[best_model],
                    estimated_minutes=_dur("predictor_grid_smoke" if smoke else "predictor_grid") / 3,
                    checkpoint_hint=best_model,
                ))

    # ========================================================================
    # PHASE 2: Evaluation
    # ========================================================================
    if args.phase in (0, 2):
        # -------------------------------------------------------------------
        # Group A: 5-method ablation evaluation
        # -------------------------------------------------------------------
        if args.only_group in (None, "A"):
            # Build checkpoint paths from training outputs
            ckpt_deps = []
            ckpt_paths = {}
            for method_key in ("no_pred", "cv", "ca", "lstm", "gru"):
                # Use seed 0 as the representative checkpoint for evaluation
                ckpt = _out("phase1_training", "experiment_A", f"{method_key}_s0", "checkpoints", "best.pt")
                ckpt_paths[method_key] = ckpt
                ckpt_deps.append(f"A_train_{method_key}_s0")

            pred_ckpts = {
                "lstm": _out("phase1_training", "predictor_training", "lstm_grid_search", "best_model.pt"),
                "gru": _out("phase1_training", "predictor_training", "gru_grid_search", "best_model.pt"),
            }

            eval_cmd = [
                sys.executable, str(PROJECT_ROOT / "scripts" / "run_ablation_matrix.py"),
                "--no-pred-checkpoint", ckpt_paths["no_pred"],
                "--cv-checkpoint", ckpt_paths["cv"],
                "--ca-checkpoint", ckpt_paths["ca"],
                "--lstm-checkpoint", ckpt_paths["lstm"],
                "--gru-checkpoint", ckpt_paths["gru"],
                "--lstm-predictor-ckpt", pred_ckpts["lstm"],
                "--gru-predictor-ckpt", pred_ckpts["gru"],
                "--scenarios-config", str(EVAL_CONFIGS["ablation_scenarios"]),
                "--seeds", "2" if smoke else "10",
                "--episodes-per-scenario", "2" if smoke else "50",
                "--output-dir", _out("phase2_evaluation", "experiment_A"),
                "--backend", backend,
            ]
            if smoke:
                eval_cmd.append("--smoke")
            eval_cmd.append("--skip-existing")

            tasks.append(Task(
                id="Eval_A",
                name="Group A: 5-method ablation evaluation",
                phase=2,
                group="A",
                cmd=eval_cmd,
                depends_on=ckpt_deps + ["PRED_LSTM_BEST", "PRED_GRU_BEST"],
                output_files=[_out("phase2_evaluation", "experiment_A", "ablation_manifest.json")],
                estimated_minutes=_dur("ablation_eval_smoke" if smoke else "ablation_eval"),
            ))

        # -------------------------------------------------------------------
        # Group B: Maneuver parameter sweep
        # -------------------------------------------------------------------
        if args.only_group in (None, "B"):
            # Build checkpoints-config YAML on the fly
            ckpt_deps = []
            ckpt_map = {}
            for method_key in ("no_pred", "cv", "ca", "lstm", "gru"):
                ckpt = _out("phase1_training", "experiment_A", f"{method_key}_s0", "checkpoints", "best.pt")
                ckpt_map[method_key] = ckpt
                ckpt_deps.append(f"A_train_{method_key}_s0")

            pred_ckpts = {
                "lstm": _out("phase1_training", "predictor_training", "lstm_grid_search", "best_model.pt"),
                "gru": _out("phase1_training", "predictor_training", "gru_grid_search", "best_model.pt"),
            }

            sweep_cmd = [
                sys.executable, str(PROJECT_ROOT / "scripts" / "run_maneuver_sweep.py"),
                "--checkpoints-config", str(EVAL_CONFIGS["ablation_checkpoints"]),
                "--sweep-type", "sinusoidal_weaving",
                "--amplitude-range", "1.0", "3.0", "5.0",
                "--frequency-range", "0.5", "1.0", "2.0",
                "--scenarios", "favorable", "neutral", "disadvantage", "challenging",
                "--seeds", "2" if smoke else "10",
                "--episodes-per-scenario", "2" if smoke else "10",
                "--base-config", str(EVAL_CONFIGS["maneuver_base"]),
                "--output-dir", _out("phase2_evaluation", "experiment_B"),
                "--backend", backend,
            ]
            if smoke:
                sweep_cmd.append("--smoke")
            sweep_cmd.append("--resume")

            # Note: run_maneuver_sweep.py reads checkpoint paths from a YAML.
            # We need to generate a temporary checkpoints-config that points to our outputs.
            # We'll handle this in the executor by generating the YAML before running.
            tasks.append(Task(
                id="Eval_B",
                name="Group B: Maneuver parameter sweep",
                phase=2,
                group="B",
                cmd=sweep_cmd,
                depends_on=ckpt_deps + ["PRED_LSTM_BEST", "PRED_GRU_BEST"],
                output_files=[_out("phase2_evaluation", "experiment_B", "sweep_summary.json")],
                estimated_minutes=_dur("maneuver_sweep_smoke" if smoke else "maneuver_sweep"),
            ))

        # -------------------------------------------------------------------
        # Group C: Architecture comparison evaluation
        # -------------------------------------------------------------------
        if args.only_group in (None, "C"):
            # FIX: Each architecture needs its TRAINING config as the base,
            # because evaluate_prediction_comparison.py requires a 'methods' field.
            # We dynamically generate a per-architecture temp YAML that includes
            # the training config settings + a single-method 'methods' block.
            archs = [
                ("vpp", "VPP-normal", str(TRAIN_CONFIGS["cv"])),
                ("no_vpp", "No-VPP", str(TRAIN_CONFIGS["no_vpp"])),
                ("e2e", "End-to-End", str(TRAIN_CONFIGS["e2e"])),
                ("no_pred", "No-Pred-only", str(TRAIN_CONFIGS["no_pred"])),
            ]
            ckpt_deps = []
            for arch_key, _, _ in archs:
                ckpt_deps.append(f"C_train_{arch_key}_s0")

            # Build a Python inline script that evaluates each architecture
            # P3 CONFIRMED: evaluate_prediction_comparison.py uses nargs="+" for
            # --scenarios, so passing ["--scenarios", "favorable", "neutral", ...]
            # is safe. Each value after --scenarios is consumed until the next
            # flag (--output-dir) is encountered.
            seeds_list = [0, 1] if smoke else DEFAULT_EVAL_SEEDS
            eps_str = "2" if smoke else "10"
            arch_eval_script = f"""
import subprocess, sys, json, os
from pathlib import Path

archs = {archs!r}
output_dir = Path({_out("phase2_evaluation", "experiment_C")!r})
output_dir.mkdir(parents=True, exist_ok=True)
results = {{}}
seeds = {seeds_list!r}

for arch_key, arch_name, config_path in archs:
    ckpt = str(Path({_out("phase1_training", "experiment_C")!r}) / f"{{arch_key}}_s0" / "checkpoints" / "best.pt")
    arch_out = output_dir / arch_key
    arch_out.mkdir(exist_ok=True)

    # FIX: evaluate_prediction_comparison.py requires a 'methods' dict.
    # Dynamically generate a temp YAML that wraps the training config
    # with a single-method block and the correct checkpoint path.
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    base_cfg["methods"] = {{
        arch_key: {{
            "name": arch_key,
            "checkpoint": ckpt,
        }}
    }}
    tmp_cfg_path = arch_out / "_eval_config.yaml"
    with open(tmp_cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(base_cfg, f, default_flow_style=False, allow_unicode=True)

    cmd = [
        sys.executable, "-m", "uav_vpp_guidance.evaluation.evaluate_prediction_comparison",
        "--config", str(tmp_cfg_path),
        "--episodes-per-scenario", {eps_str!r},
        "--scenarios", "favorable", "neutral", "disadvantage", "challenging",
        "--output-dir", str(arch_out),
        "--backend", {backend!r},
    ]
    cmd.append("--seeds")
    for s in seeds:
        cmd.append(str(s))
    print(f"[Eval-C] Evaluating {{arch_name}}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    results[arch_key] = {{
        "name": arch_name,
        "returncode": result.returncode,
        "stdout": result.stdout[-500:] if len(result.stdout) > 500 else result.stdout,
        "stderr": result.stderr[-500:] if len(result.stderr) > 500 else result.stderr,
    }}

with open(output_dir / "architecture_comparison.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print("[Eval-C] Architecture comparison complete. Results:", output_dir / "architecture_comparison.json")
"""
            eval_cmd = [sys.executable, "-c", arch_eval_script]

            tasks.append(Task(
                id="Eval_C",
                name="Group C: Architecture comparison evaluation",
                phase=2,
                group="C",
                cmd=eval_cmd,
                depends_on=ckpt_deps,
                output_files=[_out("phase2_evaluation", "experiment_C", "architecture_comparison.json")],
                estimated_minutes=_dur("ablation_eval_smoke" if smoke else "ablation_eval"),
            ))

        # -------------------------------------------------------------------
        # Group D: Gain optimization comparison
        # -------------------------------------------------------------------
        if args.only_group in (None, "D"):
            gain_cmd = [
                sys.executable, str(PROJECT_ROOT / "scripts" / "compare_gain_optimization.py"),
                "--base-config", str(EVAL_CONFIGS["gain_base"]),
                "--methods", "cem", "default", "heuristic",
                "--seeds", "2" if smoke else "3",
                "--eval-seeds", "2" if smoke else "10",
                "--scenarios", "favorable", "neutral", "disadvantage", "challenging",
                "--output-dir", _out("phase2_evaluation", "experiment_D"),
                "--device", device,
            ]
            if smoke:
                gain_cmd.append("--smoke")
            gain_cmd.append("--skip-existing")

            tasks.append(Task(
                id="Eval_D",
                name="Group D: Gain optimization comparison",
                phase=2,
                group="D",
                cmd=gain_cmd,
                depends_on=[],
                output_files=[_out("phase2_evaluation", "experiment_D", "comparison_report.json")],
                estimated_minutes=_dur("gain_comparison_smoke" if smoke else "gain_comparison"),
            ))

        # -------------------------------------------------------------------
        # Group E: JSBSim transfer evaluation
        # -------------------------------------------------------------------
        if args.only_group in (None, "E"):
            # Evaluate 2 representative methods on JSBSim backend
            # migrate_to_jsbsim.py evaluates a single policy on both simple and JSBSim backends
            jsbsim_methods = [
                ("no_pred", "No-Prediction", str(EVAL_CONFIGS["ablation_scenarios"])),
                ("cv", "CV", str(EVAL_CONFIGS["ablation_scenarios"])),
            ]
            ckpt_deps = []
            for method_key, _, _ in jsbsim_methods:
                ckpt_deps.append(f"A_train_{method_key}_s0")

            seeds_list = [0, 1] if smoke else list(range(5))
            eps_str = "1" if smoke else "10"
            out_dir = _out("phase2_evaluation", "experiment_E")

            # Inline script: call migrate_to_jsbsim.py for each method sequentially
            eval_script = f"""
import subprocess, sys, json, shutil
from pathlib import Path

methods = {jsbsim_methods!r}
output_dir = Path({out_dir!r})
output_dir.mkdir(parents=True, exist_ok=True)
results = {{}}

for method_key, method_name, config_path in methods:
    ckpt = str(Path({_out("phase1_training", "experiment_A")!r}) / f"{{method_key}}_s0" / "checkpoints" / "best.pt")
    method_out = output_dir / method_key
    method_out.mkdir(exist_ok=True)

    cmd = [
        sys.executable, str(Path({str(PROJECT_ROOT)!r}) / "scripts" / "migrate_to_jsbsim.py"),
        "--checkpoint", ckpt,
        "--config", config_path,
        "--scenarios", "favorable", "neutral", "disadvantage", "challenging",
        "--episodes-per-scenario", {eps_str!r},
        "--output-dir", str(method_out),
        "--device", {device!r},
        "--skip-jsbsim-if-missing",
    ]
    for s in {seeds_list!r}:
        cmd.extend(["--seeds", str(s)])
    if {smoke!r}:
        cmd.append("--smoke")

    print(f"[Eval-E] Running JSBSim migration for {{method_name}}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    results[method_key] = {{
        "name": method_name,
        "returncode": result.returncode,
        "stdout": result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout,
        "stderr": result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr,
    }}

# Aggregate reports if both succeeded
report_files = []
for method_key, _, _ in methods:
    rp = output_dir / method_key / "report.md"
    if rp.exists():
        report_files.append((method_key, rp))

if report_files:
    with open(output_dir / "combined_report.md", "w", encoding="utf-8") as out_f:
        out_f.write("# JSBSim Migration Evaluation - Combined Report\\n\\n")
        for method_key, rp in report_files:
            out_f.write(f"## {{methods[[m for m in methods if m[0] == method_key][0]][1]}}\\n\\n")
            out_f.write(rp.read_text(encoding="utf-8"))
            out_f.write("\\n---\\n\\n")

with open(output_dir / "jsbsim_eval_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print("[Eval-E] JSBSim evaluation complete.")
"""
            eval_cmd = [sys.executable, "-c", eval_script]

            tasks.append(Task(
                id="Eval_E",
                name="Group E: JSBSim transfer evaluation",
                phase=2,
                group="E",
                cmd=eval_cmd,
                depends_on=ckpt_deps,
                output_files=[_out("phase2_evaluation", "experiment_E", "jsbsim_eval_results.json")],
                estimated_minutes=_dur("jsbsim_eval_smoke" if smoke else "jsbsim_eval") * 2,
            ))

    # ========================================================================
    # PHASE 3: Analysis
    # ========================================================================
    if args.phase in (0, 3):
        # Compile ablation results
        compile_cmd = [
            sys.executable, str(PROJECT_ROOT / "scripts" / "compile_ablation_results.py"),
            "--ablation-dir", _out("phase2_evaluation", "experiment_A"),
            "--maneuver-dir", _out("phase2_evaluation", "experiment_B"),
            "--gain-dir", _out("phase2_evaluation", "experiment_D"),
            "--output-dir", _out("phase3_analysis", "compile_ablation_results"),
            "--format", "csv", "tex", "png",
        ]
        if smoke:
            compile_cmd.append("--smoke")

        tasks.append(Task(
            id="Analysis_compile",
            name="Phase 3: Compile ablation results and generate paper materials",
            phase=3,
            group="ANALYSIS",
            cmd=compile_cmd,
            depends_on=["Eval_A", "Eval_B", "Eval_D"],
            output_files=[
                _out("phase3_analysis", "compile_ablation_results", "findings.md"),
                _out("phase3_analysis", "compile_ablation_results", "compile_manifest.json"),
            ],
            estimated_minutes=_dur("analysis_smoke" if smoke else "analysis"),
        ))

        # Hypothesis validation report
        hypo_cmd = [
            sys.executable, "-c",
            f"""
import json, os
from pathlib import Path
output_dir = Path({_out("phase3_analysis", "hypothesis_validation")!r})
output_dir.mkdir(parents=True, exist_ok=True)
report = {{
    "report_title": "H1-H4 Hypothesis Validation",
    "generated_at": "{_now_iso()}",
    "hypotheses": {{
        "H1": "VPP offset layer provides significant tactical advantage (No-VPP vs VPP baseline)",
        "H2": "Predictor benefit stratifies with maneuver intensity (maneuver sweep)",
        "H3": "Neural predictors (LSTM/GRU) outperform classical predictors on strong maneuvers",
        "H4": "CEM-optimized gains outperform default/heuristic gains",
    }},
    "data_sources": {{
        "H1": str({_out("phase2_evaluation", "experiment_C")!r}),
        "H2": str({_out("phase2_evaluation", "experiment_B")!r}),
        "H3": str({_out("phase2_evaluation", "experiment_B")!r}),
        "H4": str({_out("phase2_evaluation", "experiment_D")!r}),
    }},
    "status": "auto-generated placeholder - validate manually after evaluation completes",
}}
with open(output_dir / "hypothesis_validation.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
print("Hypothesis validation report written to", output_dir / "hypothesis_validation.json")
"""
        ]
        tasks.append(Task(
            id="Analysis_hypothesis",
            name="Phase 3: Generate hypothesis validation report",
            phase=3,
            group="ANALYSIS",
            cmd=hypo_cmd,
            depends_on=["Analysis_compile"],
            output_files=[_out("phase3_analysis", "hypothesis_validation", "hypothesis_validation.json")],
            estimated_minutes=_dur("analysis_smoke" if smoke else "analysis"),
        ))

    return tasks


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

class ProgressTracker:
    """Tracks experiment progress and persists state to disk."""

    def __init__(self, state_file: Path, tasks: List[Task]):
        self.state_file = state_file
        self.tasks = {t.id: t for t in tasks}
        self.completed: Set[str] = set()
        self.failed: Set[str] = set()
        self.skipped: Set[str] = set()
        self.start_times: Dict[str, float] = {}
        self.end_times: Dict[str, float] = {}
        self.results: Dict[str, TaskResult] = {}
        self.phase_start_time = time.time()
        self.total_tasks = len(tasks)
        self._load_state()

    def _load_state(self):
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                self.completed = set(state.get("completed", []))
                self.failed = set(state.get("failed", []))
                self.skipped = set(state.get("skipped", []))
                self.start_times = state.get("start_times", {})
                self.end_times = state.get("end_times", {})
                print(f"[Resume] Loaded state from {self.state_file}")
                print(f"[Resume] Already completed: {len(self.completed)}/{self.total_tasks}")
            except Exception as e:
                print(f"[Warning] Failed to load state file: {e}")

    def save(self):
        state = {
            "version": 1,
            "updated_at": _now_iso(),
            "total_tasks": self.total_tasks,
            "completed": sorted(list(self.completed)),
            "failed": sorted(list(self.failed)),
            "skipped": sorted(list(self.skipped)),
            "start_times": self.start_times,
            "end_times": self.end_times,
            "completed_tasks": len(self.completed),
            "failed_tasks": len(self.failed),
            "skipped_tasks": len(self.skipped),
            "remaining_tasks": self.total_tasks - len(self.completed) - len(self.failed) - len(self.skipped),
            "progress_pct": round(len(self.completed) / self.total_tasks * 100, 1) if self.total_tasks > 0 else 0,
        }
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def mark_started(self, task_id: str):
        self.start_times[task_id] = time.time()
        self.save()

    def mark_completed(self, task_id: str, result: TaskResult):
        self.completed.add(task_id)
        self.end_times[task_id] = time.time()
        self.results[task_id] = result
        self.save()

    def mark_failed(self, task_id: str, result: TaskResult):
        self.failed.add(task_id)
        self.end_times[task_id] = time.time()
        self.results[task_id] = result
        self.save()

    def mark_skipped(self, task_id: str):
        self.skipped.add(task_id)
        self.save()

    def is_done(self, task_id: str) -> bool:
        return task_id in self.completed or task_id in self.skipped

    def is_failed(self, task_id: str) -> bool:
        return task_id in self.failed

    def can_run(self, task: Task) -> bool:
        """Check if all dependencies are satisfied."""
        for dep in task.depends_on:
            if dep not in self.completed and dep not in self.skipped:
                return False
        return True

    def get_eta(self) -> str:
        """Estimate remaining time based on completed tasks."""
        if not self.completed:
            # Use sum of estimated durations for all remaining tasks
            remaining = [t for t in self.tasks.values() if t.id not in self.completed and t.id not in self.skipped]
            total_est = sum(t.estimated_minutes for t in remaining)
            return f"~{_format_duration(total_est * 60)} (estimate)"

        # Calculate average actual vs estimated ratio
        ratios = []
        for tid in self.completed:
            if tid in self.start_times and tid in self.end_times:
                actual = self.end_times[tid] - self.start_times[tid]
                estimated = self.tasks[tid].estimated_minutes * 60
                if estimated > 0:
                    ratios.append(actual / estimated)

        avg_ratio = sum(ratios) / len(ratios) if ratios else 1.0
        remaining = [t for t in self.tasks.values() if t.id not in self.completed and t.id not in self.skipped]
        total_est_seconds = sum(t.estimated_minutes * 60 for t in remaining)
        eta_seconds = total_est_seconds * avg_ratio
        return f"~{_format_duration(eta_seconds)}"

    def print_progress_bar(self):
        done = len(self.completed)
        failed = len(self.failed)
        skipped = len(self.skipped)
        total = self.total_tasks
        width = 40
        filled = int(width * done / total) if total > 0 else 0
        bar = "#" * filled + "-" * (width - filled)
        print(f"\r[{bar}] {done}/{total} done, {failed} failed, {skipped} skipped | ETA: {self.get_eta()}", end="", flush=True)
        if done + failed + skipped == total:
            print()  # newline when complete


# ---------------------------------------------------------------------------
# Resource manager
# ---------------------------------------------------------------------------

class ResourceManager:
    """Manages GPU time budget and device selection."""

    def __init__(self, gpu_id: int, max_gpu_hours: Optional[float]):
        self.gpu_id = gpu_id
        self.max_gpu_hours = max_gpu_hours
        self.gpu_start_time = time.time()
        self.paused = False

    def check_budget(self) -> bool:
        """Return True if we can continue, False if budget exceeded."""
        if self.max_gpu_hours is None:
            return True
        elapsed_hours = (time.time() - self.gpu_start_time) / 3600.0
        if elapsed_hours >= self.max_gpu_hours:
            print(f"\n[Budget] GPU time limit reached: {elapsed_hours:.1f}h / {self.max_gpu_hours:.1f}h")
            self.paused = True
            return False
        return True

    def get_elapsed_hours(self) -> float:
        return (time.time() - self.gpu_start_time) / 3600.0


# ---------------------------------------------------------------------------
# Task executor
# ---------------------------------------------------------------------------

class TaskExecutor:
    """Executes tasks with dependency resolution and error handling."""

    def __init__(self, tracker: ProgressTracker, resources: ResourceManager, dry_run: bool = False):
        self.tracker = tracker
        self.resources = resources
        self.dry_run = dry_run
        self.task_map = tracker.tasks

    def _resolve_checkpoint_args(self, task: Task) -> List[str]:
        """Inject predictor checkpoint paths into LSTM/GRU training commands."""
        cmd = list(task.cmd)
        if "--predictor-type" in cmd:
            ptype_idx = cmd.index("--predictor-type")
            ptype = cmd[ptype_idx + 1]
            if ptype in ("lstm", "gru"):
                # Find the best model path
                pred_task_id = f"PRED_{ptype.upper()}_BEST"
                if pred_task_id in self.task_map:
                    pred_task = self.task_map[pred_task_id]
                    if pred_task.output_files:
                        ckpt_path = pred_task.output_files[0]
                        if Path(ckpt_path).exists() or self.dry_run:
                            cmd.extend(["--checkpoint", ckpt_path])
        return cmd

    def _generate_checkpoints_config(self, output_path: Path, ckpt_map: Dict[str, str], pred_ckpts: Dict[str, str]):
        """Generate a checkpoints-config YAML for maneuver sweep."""
        import yaml
        config = {
            "checkpoints": {},
            "predictor_checkpoints": {},
        }
        key_map = {
            "no_pred": "no_prediction",
            "cv": "cv_prediction",
            "ca": "ca_prediction",
            "lstm": "lstm_frozen",
            "gru": "gru_frozen",
        }
        for k, v in ckpt_map.items():
            config["checkpoints"][key_map.get(k, k)] = v
        for k, v in pred_ckpts.items():
            config["predictor_checkpoints"][k] = v
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    def _prepare_task(self, task: Task) -> List[str]:
        """Prepare task command (resolve dynamic args, generate configs)."""
        cmd = self._resolve_checkpoint_args(task)

        # For maneuver sweep, generate the checkpoints-config YAML
        if task.id == "Eval_B":
            # Extract output dir from command
            out_idx = cmd.index("--output-dir") + 1
            out_dir = Path(cmd[out_idx])
            ckpt_config_path = out_dir / "_auto_checkpoints_config.yaml"

            ckpt_map = {}
            pred_ckpts = {}
            for method_key in ("no_pred", "cv", "ca", "lstm", "gru"):
                train_task_id = f"A_train_{method_key}_s0"
                if train_task_id in self.task_map:
                    ckpt_map[method_key] = self.task_map[train_task_id].output_files[0]
            for ptype in ("lstm", "gru"):
                pred_task_id = f"PRED_{ptype.upper()}_BEST"
                if pred_task_id in self.task_map:
                    pred_ckpts[ptype] = self.task_map[pred_task_id].output_files[0]

            self._generate_checkpoints_config(ckpt_config_path, ckpt_map, pred_ckpts)

            # Replace the --checkpoints-config argument
            cfg_idx = cmd.index("--checkpoints-config") + 1
            cmd[cfg_idx] = str(ckpt_config_path)

        return cmd

    def _check_output_exists(self, task: Task) -> bool:
        """Check if all expected output files already exist."""
        for f in task.output_files:
            if not Path(f).exists():
                return False
        return True

    def run_task(self, task: Task) -> TaskResult:
        """Execute a single task."""
        if self.dry_run:
            print(f"[DRY-RUN] Would execute: {' '.join(task.cmd)}")
            return TaskResult(task_id=task.id, status="dry_run")

        # Check GPU budget
        if not self.resources.check_budget():
            return TaskResult(
                task_id=task.id, status="paused",
                error_message="GPU budget exceeded — run with --resume to continue",
            )

        # Skip if outputs already exist
        if self._check_output_exists(task):
            print(f"\n[Skip] {task.name} — outputs already exist")
            self.tracker.mark_skipped(task.id)
            return TaskResult(task_id=task.id, status="skipped")

        # Prepare command
        cmd = self._prepare_task(task)

        print(f"\n[{'='*60}")
        print(f"Task: {task.name}")
        print(f"Phase {task.phase} | Group {task.group}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*60}]")

        self.tracker.mark_started(task.id)
        start = time.time()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT),
                encoding="utf-8",
                errors="replace",
            )
            duration = time.time() - start

            if result.returncode == 0:
                print(f"[OK] Completed in {_format_duration(duration)}")
                task_result = TaskResult(
                    task_id=task.id,
                    status="success",
                    returncode=0,
                    stdout=result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout,
                    stderr=result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr,
                    duration_seconds=duration,
                )
                self.tracker.mark_completed(task.id, task_result)
            else:
                print(f"[FAIL] Exit code {result.returncode} after {_format_duration(duration)}")
                print(f"[FAIL] stderr: {result.stderr[-1000:]}")
                task_result = TaskResult(
                    task_id=task.id,
                    status="failed",
                    returncode=result.returncode,
                    stdout=result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout,
                    stderr=result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr,
                    duration_seconds=duration,
                    error_message=f"Exit code {result.returncode}",
                )
                self.tracker.mark_failed(task.id, task_result)

        except Exception as e:
            duration = time.time() - start
            print(f"[ERROR] Exception: {e}")
            traceback.print_exc()
            task_result = TaskResult(
                task_id=task.id,
                status="failed",
                returncode=-1,
                duration_seconds=duration,
                error_message=str(e),
            )
            self.tracker.mark_failed(task.id, task_result)

        finally:
            _release_gpu_memory()

        return task_result

    def run_all(self) -> Dict[str, TaskResult]:
        """Run all pending tasks in dependency order."""
        pending = [
            t for t in self.task_map.values()
            if not self.tracker.is_done(t.id) and not self.tracker.is_failed(t.id)
        ]
        pending.sort(key=lambda t: (t.phase, t.group, t.id))

        iteration = 0
        while pending:
            iteration += 1
            made_progress = False

            for task in list(pending):
                if self.tracker.is_done(task.id) or self.tracker.is_failed(task.id):
                    pending.remove(task)
                    continue

                if not self.tracker.can_run(task):
                    continue

                result = self.run_task(task)
                pending.remove(task)
                made_progress = True

                if result.status == "paused":
                    print("\n[Scheduler] Paused due to GPU budget. Resume with --resume flag.")
                    return self.tracker.results

                # Print progress
                self.tracker.print_progress_bar()

            if not made_progress and pending:
                # Check if any pending tasks have failed dependencies
                stuck = []
                for task in pending:
                    failed_deps = [d for d in task.depends_on if self.tracker.is_failed(d)]
                    if failed_deps:
                        stuck.append((task, failed_deps))
                if stuck:
                    print("\n[Scheduler] Some tasks have failed dependencies and cannot proceed:")
                    for task, deps in stuck:
                        print(f"  - {task.id} (depends on {deps})")
                        self.tracker.mark_skipped(task.id)
                        pending.remove(task)
                else:
                    # Waiting on dependencies that are still pending — this shouldn't happen
                    # unless there's a cycle. Print warning and break.
                    print("\n[Scheduler] Warning: tasks are waiting but no runnable tasks found.")
                    print("This may indicate a dependency cycle or missing tasks.")
                    for task in pending:
                        print(f"  Pending: {task.id} (needs: {task.depends_on})")
                    break

            # Safety check: if we've been running too long without progress, break
            if iteration > self.tracker.total_tasks * 3:
                print("\n[Scheduler] Safety break: too many iterations without completing all tasks.")
                break

        return self.tracker.results


# ---------------------------------------------------------------------------
# Master manifest generation
# ---------------------------------------------------------------------------

def generate_master_manifest(output_dir: Path, tracker: ProgressTracker, tasks: List[Task], args: argparse.Namespace) -> Path:
    """Generate the master_manifest.json with full experiment provenance."""
    manifest = {
        "generated_at": _now_iso(),
        "cli_args": vars(args),
        "git": _get_git_info(),
        "environment": _get_env_info(),
        "summary": {
            "total_tasks": len(tasks),
            "completed": len(tracker.completed),
            "failed": len(tracker.failed),
            "skipped": len(tracker.skipped),
            "success_rate": round(len(tracker.completed) / len(tasks) * 100, 1) if tasks else 0,
        },
        "phases": {},
        "tasks": [],
    }

    for task in tasks:
        task_entry = {
            "id": task.id,
            "name": task.name,
            "phase": task.phase,
            "group": task.group,
            "status": "completed" if task.id in tracker.completed else ("failed" if task.id in tracker.failed else ("skipped" if task.id in tracker.skipped else "pending")),
            "command": " ".join(task.cmd),
            "depends_on": task.depends_on,
            "output_files": task.output_files,
            "checkpoint_hint": task.checkpoint_hint,
        }
        if task.id in tracker.results:
            result = tracker.results[task.id]
            task_entry["result"] = {
                "returncode": result.returncode,
                "duration_seconds": round(result.duration_seconds, 1),
                "error_message": result.error_message,
            }
        manifest["tasks"].append(task_entry)

    # Organize by phase
    for phase in [1, 2, 3]:
        phase_tasks = [t for t in tasks if t.phase == phase]
        manifest["phases"][f"phase_{phase}"] = {
            "task_count": len(phase_tasks),
            "completed": sum(1 for t in phase_tasks if t.id in tracker.completed),
            "tasks": [t.id for t in phase_tasks],
        }

    manifest_path = output_dir / "master_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\n[Manifest] Written: {manifest_path}")
    return manifest_path


def _get_git_info() -> Dict[str, str]:
    """Collect git provenance information."""
    info = {"commit": "unknown", "branch": "unknown", "dirty": False}
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        if result.returncode == 0:
            info["commit"] = result.stdout.strip()
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        if result.returncode == 0:
            info["branch"] = result.stdout.strip()
        result = subprocess.run(
            ["git", "diff", "--quiet"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        info["dirty"] = result.returncode != 0
    except Exception:
        pass
    return info


def _get_env_info() -> Dict[str, str]:
    """Collect environment information."""
    info = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": sys.platform,
    }
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_count"] = str(torch.cuda.device_count())
    except Exception:
        pass
    try:
        import numpy as np
        info["numpy_version"] = np.__version__
    except Exception:
        pass
    return info


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full Experiment Matrix Scheduler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline
  python scripts/run_full_experiment_matrix.py --output-dir outputs/full_experiment

  # Only training (Phase 1)
  python scripts/run_full_experiment_matrix.py --phase 1

  # Only evaluation (Phase 2, checkpoints must exist)
  python scripts/run_full_experiment_matrix.py --phase 2

  # Only Group A
  python scripts/run_full_experiment_matrix.py --only-group A

  # Resume interrupted run
  python scripts/run_full_experiment_matrix.py --resume

  # Smoke test (fast verification)
  python scripts/run_full_experiment_matrix.py --smoke

  # Dry-run (print task list only)
  python scripts/run_full_experiment_matrix.py --dry-run
        """,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/full_experiment",
        help="Root output directory for all experiment artifacts (default: outputs/full_experiment)",
    )
    parser.add_argument(
        "--phase",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="Run only a specific phase (0=all, 1=training, 2=evaluation, 3=analysis)",
    )
    parser.add_argument(
        "--only-group",
        type=str,
        default=None,
        choices=["A", "B", "C", "D", "E"],
        help="Run only a specific experiment group (A-E)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID (default: 0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=_get_default_device(),
        choices=["cpu", "cuda"],
        help=f"PyTorch compute device (default: auto-detected, currently {_get_default_device()})",
    )
    parser.add_argument(
        "--max-gpu-hours",
        type=float,
        default=None,
        help="Maximum GPU hours before pausing (default: no limit)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous state file",
    )
    parser.add_argument(
        "--state-file",
        type=str,
        default=None,
        help="Path to progress state file (default: <output-dir>/progress.json)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke test mode (minimal steps/configs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print task list without executing",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip tasks whose output files already exist",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="jsbsim",
        choices=["simple", "jsbsim"],
        help="Simulation backend for training and evaluation (default: jsbsim)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate environment: check all dependencies exist without executing",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Environment validation
# ---------------------------------------------------------------------------

def validate_environment(args: argparse.Namespace) -> Dict[str, Any]:
    """Check all dependencies (configs, scripts, modules) without executing."""
    report = {
        "status": "ok",
        "configs": {},
        "scripts": {},
        "modules": {},
        "python": {},
        "issues": [],
        "warnings": [],
    }

    # 1. Check config files
    all_configs = {**TRAIN_CONFIGS, **EVAL_CONFIGS}
    for name, path in all_configs.items():
        exists = Path(path).exists()
        report["configs"][name] = {"path": str(path), "exists": exists}
        if not exists:
            report["issues"].append(f"MISSING config: {name} -> {path}")
            report["status"] = "errors"

    # 1b. Config semantic checks (No-VPP / End-to-End pitfalls)
    report["config_semantics"] = {}
    semantic_rules = {
        "no_vpp": {
            "expected": {
                "virtual_point.enabled": True,
                "virtual_point.mode": "zero_offset",
            },
            "forbidden": ["virtual_point.enabled=false without end_to_end.enabled=true"],
        },
        "e2e": {
            "expected": {
                "virtual_point.enabled": False,
                "end_to_end.enabled": True,
            },
        },
    }

    try:
        import yaml as _yaml
        for cfg_name, rules in semantic_rules.items():
            cfg_path = TRAIN_CONFIGS.get(cfg_name)
            if not cfg_path or not Path(cfg_path).exists():
                continue
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = _yaml.safe_load(f)

            vp_cfg = cfg.get("virtual_point", {})
            e2e_cfg = cfg.get("end_to_end", {})
            vp_enabled = vp_cfg.get("enabled", True)
            vp_mode = vp_cfg.get("mode", "normal")
            e2e_enabled = e2e_cfg.get("enabled", False)

            entry = {
                "file": cfg_name,
                "virtual_point.enabled": vp_enabled,
                "virtual_point.mode": vp_mode,
                "end_to_end.enabled": e2e_enabled,
            }

            if cfg_name == "no_vpp":
                if vp_enabled is not True:
                    entry["error"] = (
                        f"No-VPP config must set virtual_point.enabled=true "
                        f"(got {vp_enabled}). Use mode='zero_offset' to disable offsets, "
                        f"not enabled=false."
                    )
                    report["issues"].append(f"CONFIG SEMANTICS: {entry['error']} ({cfg_path})")
                    report["status"] = "errors"
                elif vp_mode != "zero_offset":
                    entry["error"] = (
                        f"No-VPP config must set virtual_point.mode='zero_offset' "
                        f"(got '{vp_mode}')."
                    )
                    report["issues"].append(f"CONFIG SEMANTICS: {entry['error']} ({cfg_path})")
                    report["status"] = "errors"
                else:
                    entry["ok"] = True

            elif cfg_name == "e2e":
                if vp_enabled is not False:
                    entry["error"] = (
                        f"End-to-End config must set virtual_point.enabled=false "
                        f"(got {vp_enabled})."
                    )
                    report["issues"].append(f"CONFIG SEMANTICS: {entry['error']} ({cfg_path})")
                    report["status"] = "errors"
                elif e2e_enabled is not True:
                    entry["error"] = (
                        f"End-to-End config must set end_to_end.enabled=true "
                        f"(got {e2e_enabled})."
                    )
                    report["issues"].append(f"CONFIG SEMANTICS: {entry['error']} ({cfg_path})")
                    report["status"] = "errors"
                else:
                    entry["ok"] = True

            report["config_semantics"][cfg_name] = entry
    except Exception as e:
        report["warnings"].append(f"Could not run config semantic checks: {e}")

    # 2. Check script files
    scripts = {
        "grid_search_lstm.py": PROJECT_ROOT / "scripts" / "grid_search_lstm.py",
        "migrate_to_jsbsim.py": PROJECT_ROOT / "scripts" / "migrate_to_jsbsim.py",
        "run_ablation_matrix.py": PROJECT_ROOT / "scripts" / "run_ablation_matrix.py",
        "run_maneuver_sweep.py": PROJECT_ROOT / "scripts" / "run_maneuver_sweep.py",
        "compare_gain_optimization.py": PROJECT_ROOT / "scripts" / "compare_gain_optimization.py",
        "compile_ablation_results.py": PROJECT_ROOT / "scripts" / "compile_ablation_results.py",
    }
    for name, path in scripts.items():
        exists = path.exists()
        report["scripts"][name] = {"path": str(path), "exists": exists}
        if not exists:
            report["issues"].append(f"MISSING script: {name} -> {path}")
            report["status"] = "errors"

    # 3. Check Python modules (try import without executing)
    modules = [
        "uav_vpp_guidance.training.train_prediction_vpp_ppo",
        "uav_vpp_guidance.training.train_no_prediction_vpp_ppo",
        "uav_vpp_guidance.training.train_end_to_end_ppo",
        "uav_vpp_guidance.training.train_fixed_gain",
        "uav_vpp_guidance.trajectory_prediction.train_pipeline",
        "uav_vpp_guidance.evaluation.evaluate_prediction_comparison",
    ]
    for mod in modules:
        try:
            __import__(mod)
            report["modules"][mod] = {"importable": True}
        except Exception as e:
            report["modules"][mod] = {"importable": False, "error": str(e)}
            report["issues"].append(f"UNIMPORTABLE module: {mod} -> {e}")
            report["status"] = "errors"

    # 4. Check Python / PyTorch / CUDA
    report["python"]["version"] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    try:
        import torch
        report["python"]["torch_version"] = torch.__version__
        report["python"]["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            report["python"]["cuda_version"] = torch.version.cuda
            report["python"]["gpu_count"] = torch.cuda.device_count()
            for i in range(torch.cuda.device_count()):
                report["python"][f"gpu_{i}"] = torch.cuda.get_device_name(i)
        else:
            report["warnings"].append("CUDA not available; training will use CPU (slower)")
    except ImportError:
        report["issues"].append("MISSING package: torch is not installed")
        report["status"] = "errors"

    try:
        import numpy as np
        report["python"]["numpy_version"] = np.__version__
    except ImportError:
        report["issues"].append("MISSING package: numpy is not installed")
        report["status"] = "errors"

    try:
        import yaml
        report["python"]["pyyaml_ok"] = True
    except ImportError:
        report["issues"].append("MISSING package: PyYAML is not installed")
        report["status"] = "errors"

    # 5. Check output dir writable
    out_dir = Path(args.output_dir).resolve()
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        test_file = out_dir / ".write_test"
        test_file.write_text("ok")
        test_file.unlink()
        report["output_dir"] = {"path": str(out_dir), "writable": True}
    except Exception as e:
        report["output_dir"] = {"path": str(out_dir), "writable": False, "error": str(e)}
        report["issues"].append(f"NOT WRITABLE: output_dir {out_dir} -> {e}")
        report["status"] = "errors"

    return report


def print_validation_report(report: Dict[str, Any]):
    """Print a human-readable validation report."""
    _print_banner("Environment Validation Report")

    status = report.get("status", "unknown")
    if status == "ok":
        print("Status: ALL CHECKS PASSED [OK]")
    elif status == "warnings":
        print("Status: PASSED WITH WARNINGS [WARN]")
    else:
        print("Status: FAILED [ERROR]")

    # Configs
    print(f"\n[Configs] {len(report['configs'])} checked")
    missing_cfgs = [k for k, v in report["configs"].items() if not v["exists"]]
    if missing_cfgs:
        print(f"  Missing: {', '.join(missing_cfgs)}")
    else:
        print("  All present [OK]")

    # Scripts
    print(f"\n[Scripts] {len(report['scripts'])} checked")
    missing_scripts = [k for k, v in report["scripts"].items() if not v["exists"]]
    if missing_scripts:
        print(f"  Missing: {', '.join(missing_scripts)}")
    else:
        print("  All present [OK]")

    # Modules
    print(f"\n[Modules] {len(report['modules'])} checked")
    bad_modules = [k for k, v in report["modules"].items() if not v.get("importable")]
    if bad_modules:
        print(f"  Unimportable: {', '.join(bad_modules)}")
    else:
        print("  All importable [OK]")

    # Config semantics
    semantics = report.get("config_semantics", {})
    if semantics:
        print(f"\n[Config Semantics] {len(semantics)} checked")
        for cfg_name, entry in semantics.items():
            if entry.get("ok"):
                print(f"  {cfg_name}: OK (vp.enabled={entry.get('virtual_point.enabled')}, vp.mode={entry.get('virtual_point.mode')!r}, e2e.enabled={entry.get('end_to_end.enabled')})")
            else:
                print(f"  {cfg_name}: ERROR - {entry.get('error', 'unknown')}")

    # Python env
    py = report.get("python", {})
    print(f"\n[Python Environment]")
    print(f"  Python: {py.get('version', 'unknown')}")
    print(f"  PyTorch: {py.get('torch_version', 'not installed')}")
    print(f"  CUDA available: {py.get('cuda_available', False)}")
    if py.get("cuda_available"):
        print(f"  GPUs: {py.get('gpu_count', 0)}")
        for i in range(py.get("gpu_count", 0)):
            print(f"    GPU {i}: {py.get(f'gpu_{i}', 'unknown')}")

    # Output dir
    od = report.get("output_dir", {})
    print(f"\n[Output Directory]")
    print(f"  Path: {od.get('path', 'unknown')}")
    print(f"  Writable: {'Yes [OK]' if od.get('writable') else 'No [FAIL]'}")

    # Issues and warnings
    issues = report.get("issues", [])
    warnings = report.get("warnings", [])
    if warnings:
        print(f"\n[Warnings] ({len(warnings)})")
        for w in warnings:
            print(f"  [WARN] {w}")
    if issues:
        print(f"\n[Errors] ({len(issues)})")
        for issue in issues:
            print(f"  [ERROR] {issue}")
        print("\nPlease fix the above errors before running experiments.")
        sys.exit(1)
    else:
        print("\n[OK] Environment is ready for experiments.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Validate mode: check dependencies and exit
    if args.validate:
        report = validate_environment(args)
        print_validation_report(report)
        return

    # Resolve output directory and state file
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_file = Path(args.state_file) if args.state_file else output_dir / "progress.json"

    # Print header
    _print_banner("Full Experiment Matrix Scheduler")
    print(f"Output directory: {output_dir}")
    print(f"State file:       {state_file}")
    print(f"Phase:            {'All' if args.phase == 0 else args.phase}")
    print(f"Group filter:     {args.only_group or 'None (all groups)'}")
    print(f"Device:           {args.device}")
    print(f"GPU ID:           {args.gpu}")
    print(f"Max GPU hours:    {args.max_gpu_hours or 'unlimited'}")
    print(f"Smoke mode:       {args.smoke}")
    print(f"Dry-run mode:     {args.dry_run}")
    print(f"Resume:           {args.resume}")

    if args.device == "cuda":
        if _check_gpu_available(args.gpu):
            print(f"GPU {args.gpu}: {_get_gpu_name(args.gpu)} [OK]")
        else:
            print(f"WARNING: GPU {args.gpu} not available, falling back to CPU")
            args.device = "cpu"

    # Build task graph
    print("\n[Setup] Building task graph...")
    tasks = build_task_graph(args)
    print(f"[Setup] Total tasks: {len(tasks)}")

    # Phase breakdown
    for phase in [1, 2, 3]:
        phase_tasks = [t for t in tasks if t.phase == phase]
        if phase_tasks:
            groups = sorted(set(t.group for t in phase_tasks))
            print(f"  Phase {phase}: {len(phase_tasks)} tasks ({', '.join(groups)})")

    # Dry-run: just print and exit
    if args.dry_run:
        print("\n[Dry-Run] Task execution plan:")
        for task in tasks:
            deps_str = f" (needs: {task.depends_on})" if task.depends_on else ""
            print(f"  [{task.phase}][{task.group}] {task.id}: {task.name}{deps_str}")
            print(f"    -> {' '.join(task.cmd)}")
        print(f"\n[Dry-Run] {len(tasks)} tasks would be executed.")
        # Still generate manifest for inspection
        tracker = ProgressTracker(state_file, tasks)
        generate_master_manifest(output_dir, tracker, tasks, args)
        return

    # Initialize tracker and executor
    tracker = ProgressTracker(state_file, tasks)
    resources = ResourceManager(args.gpu, args.max_gpu_hours)
    executor = TaskExecutor(tracker, resources, dry_run=False)

    # Run all tasks
    print("\n[Scheduler] Starting execution...")
    results = executor.run_all()

    # Final summary
    _print_banner("Execution Complete")
    completed = len(tracker.completed)
    failed = len(tracker.failed)
    skipped = len(tracker.skipped)
    total = len(tasks)
    print(f"Completed: {completed}/{total}")
    print(f"Failed:    {failed}/{total}")
    print(f"Skipped:   {skipped}/{total}")
    print(f"Success rate: {completed/total*100:.1f}%" if total > 0 else "N/A")

    if failed > 0:
        print("\nFailed tasks:")
        for tid in sorted(tracker.failed):
            result = tracker.results.get(tid)
            err = result.error_message if result else "unknown"
            print(f"  - {tid}: {err}")

    # Generate master manifest
    generate_master_manifest(output_dir, tracker, tasks, args)

    # Exit with error code if any tasks failed
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
