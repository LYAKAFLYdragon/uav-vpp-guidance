#!/usr/bin/env python3
"""
LSTM Trajectory Predictor — Hyperparameter Grid Search.

Reference style: OA-TSLANet project (argparse, structured logging,
experiment directories, JSON/CSV result persistence).

Search space (default):
  hidden_dim:         {64, 128, 256}
  num_layers:         {1, 2, 3}
  dropout:            {0.0, 0.1, 0.2}
  history_len:        {5, 10, 15}
  prediction_horizon: {3, 5, 10}

Selection criterion: lowest validation loss (MSE on relative displacement).
Early stopping: patience=5 (configurable via --patience).
"""

import argparse
import glob
import itertools
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split

# Ensure src/ is on PYTHONPATH when script is run directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from uav_vpp_guidance.trajectory_prediction.dataset import (
    TrajectoryPredictionDataset,
    build_lstm_prediction_feature,
)
from uav_vpp_guidance.trajectory_prediction.lstm_predictor import LSTMTrajectoryPredictor
from uav_vpp_guidance.trajectory_prediction.trainer import TrajectoryPredictorTrainer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Grid search for LSTM trajectory predictor hyperparameters."
    )

    # Data source ------------------------------------------------------------
    parser.add_argument(
        "--data_source",
        type=str,
        required=True,
        help=(
            "Path pattern to episode CSV files (e.g. 'outputs/episode_*.csv') "
            "or a single HDF5/CSV file."
        ),
    )
    parser.add_argument(
        "--source_type",
        type=str,
        default="episode_logs",
        choices=["episode_logs", "tracking_env"],
        help="How to interpret --data_source.",
    )
    parser.add_argument(
        "--tracking_env_scenarios",
        type=int,
        default=50,
        help="Number of episodes to collect when source_type='tracking_env'.",
    )

    # Fixed training hyperparameters -----------------------------------------
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1.0e-3)
    parser.add_argument("--weight_decay", type=float, default=1.0e-5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)

    # Search space (grid) ----------------------------------------------------
    parser.add_argument(
        "--hidden_dims", nargs="+", type=int, default=[64, 128, 256]
    )
    parser.add_argument(
        "--num_layers_list", nargs="+", type=int, default=[1, 2, 3]
    )
    parser.add_argument(
        "--dropouts", nargs="+", type=float, default=[0.0, 0.1, 0.2]
    )
    parser.add_argument(
        "--history_lens", nargs="+", type=int, default=[5, 10, 15]
    )
    parser.add_argument(
        "--prediction_horizons", nargs="+", type=int, default=[3, 5, 10]
    )

    # Output -----------------------------------------------------------------
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/lstm_grid_search",
        help="Root directory for all experiment sub-folders.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Optional experiment suffix (default: auto timestamp).",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------


def _build_dataset_config(history_len: int, prediction_horizon: int, dt: float = 0.2):
    """Assemble the config dict required by TrajectoryPredictionDataset."""
    return {
        "history": {"history_len": history_len, "padding_mode": "repeat_first"},
        "prediction": {"lookahead_time_s": prediction_horizon * dt},
        "env": {"high_level_dt": dt},
        "normalization": {
            "position_scale_m": 1000.0,
            "velocity_scale_mps": 300.0,
            "acceleration_scale_mps2": 50.0,
            "overload_scale": 9.0,
        },
    }


def build_dataset_from_logs(data_source: str, config: dict):
    """Load CSV files matching a glob pattern and build a dataset."""
    paths = glob.glob(data_source)
    if not paths:
        # Try as a single file
        p = Path(data_source)
        if p.exists():
            paths = [str(p)]
        else:
            raise ValueError(f"No files found for pattern: {data_source}")

    # Filter to actual CSV files
    paths = [p for p in paths if p.lower().endswith(".csv")]
    if not paths:
        raise ValueError(f"No CSV files found for pattern: {data_source}")

    return TrajectoryPredictionDataset.from_episode_logs(paths, config)


def build_dataset_from_env(config: dict, history_len: int, prediction_horizon: int, num_episodes: int, seed: int):
    """Run CloseRangeTrackingEnv (simple backend) to collect trajectories."""
    from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv

    env_config = {
        "experiment": {"name": "grid_search_data", "seed": seed, "output_root": "outputs"},
        "env": {
            "use_jsbsim": False,
            "decision_freq": 5,
            "sim_freq": 60,
            "max_high_level_steps": 512,
            "success_range_m": 900.0,
            "success_ata_deg": 25.0,
            "success_hold_time_s": 0.2,
            "hysteresis_range_m": 950.0,
            "hysteresis_ata_deg": 30.0,
            "min_altitude_m": 500.0,
            "max_altitude_m": 15000.0,
            "max_range_m": 8000.0,
            "target_mode": "constant_velocity",
            "high_level_dt": 0.2,
        },
        "virtual_point": {
            "anchor_mode": "current_target",
            "action_dim": 3,
            "d_long_range": [-1500.0, 1500.0],
            "d_lat_range": [-800.0, 800.0],
            "d_vert_range": [-500.0, 500.0],
            "smoothing_alpha": 0.3,
        },
        "trajectory_prediction": {"enabled": False},
        "limits": {
            "nz_min": -2.0,
            "nz_max": 7.0,
            "roll_rate_min": -1.5,
            "roll_rate_max": 1.5,
            "throttle_min": 0.0,
            "throttle_max": 1.0,
        },
        "reward": {
            "w_range": 0.5,
            "w_angle": 0.8,
            "w_energy": 0.2,
            "w_safety": 2.0,
            "w_saturation": 1.0,
            "w_smooth": 0.1,
            "terminal_success": 200.0,
            "terminal_failure": -200.0,
            "terminal_crash": -300.0,
            "min_altitude_m": 500.0,
        },
        "guidance": {
            "mode": "los_rate",
            "use_gain_adapter": False,
            "gains": {
                "k_los": 1.0,
                "k_pos": 0.5,
                "k_damp": 0.2,
                "k_roll": 1.0,
                "k_speed": 0.2,
                "alpha_filter": 0.3,
            },
        },
    }

    env = CloseRangeTrackingEnv(env_config)
    return TrajectoryPredictionDataset.from_tracking_env(
        env,
        num_episodes=num_episodes,
        max_steps_per_episode=200,
        history_len=history_len,
        prediction_horizon=prediction_horizon,
        config=config,
        feature_builder=build_lstm_prediction_feature,
        seed=seed,
    )


def build_dataset(args, history_len: int, prediction_horizon: int):
    """Route to the correct dataset builder based on source_type."""
    config = _build_dataset_config(history_len, prediction_horizon)
    if args.source_type == "episode_logs":
        return build_dataset_from_logs(args.data_source, config)
    return build_dataset_from_env(
        config, history_len, prediction_horizon, args.tracking_env_scenarios, args.seed
    )


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------


def run_single_experiment(
    args, hidden_dim: int, num_layers: int, dropout: float,
    history_len: int, prediction_horizon: int, output_dir: str,
):
    """Train one LSTM configuration and return summary metrics."""
    # 1. Dataset
    dataset = build_dataset(args, history_len, prediction_horizon)

    train_size = int(len(dataset) * args.train_split)
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # 2. Model
    model = LSTMTrajectoryPredictor(
        input_dim=16,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        predict_variance=False,
    )

    # 3. Trainer (patience is baked into TrajectoryPredictorTrainer)
    trainer_cfg = {
        "device": args.device,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "epochs": args.epochs,
        "patience": args.patience,
        "output_dir": output_dir,
    }
    trainer = TrajectoryPredictorTrainer(model, train_loader, val_loader, trainer_cfg)
    history = trainer.fit()

    # 4. Persist per-run history
    with open(os.path.join(output_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return {
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "history_len": history_len,
        "prediction_horizon": prediction_horizon,
        "best_val_loss": float(trainer.best_val_loss),
        "best_epoch": int(trainer.best_epoch),
        "total_epochs": len(history["train_loss"]),
        "train_samples": train_size,
        "val_samples": val_size,
        "output_dir": output_dir,
    }


# ---------------------------------------------------------------------------
# Grid search orchestration
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Output root
    exp_suffix = args.exp_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = Path(args.output_dir) / exp_suffix
    root_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OUTPUT] Output root: {root_dir.resolve()}")

    # Search space
    search_space = list(itertools.product(
        args.hidden_dims,
        args.num_layers_list,
        args.dropouts,
        args.history_lens,
        args.prediction_horizons,
    ))
    total = len(search_space)
    print(f"[SEARCH] Grid search size: {total} configurations")
    print(f"   hidden_dim: {args.hidden_dims}")
    print(f"   num_layers: {args.num_layers_list}")
    print(f"   dropout:    {args.dropouts}")
    print(f"   history_len: {args.history_lens}")
    print(f"   prediction_horizon: {args.prediction_horizons}")
    print("-" * 60)

    results = []
    for idx, (hidden_dim, num_layers, dropout, history_len, prediction_horizon) in enumerate(search_space, start=1):
        tag = f"h{hidden_dim}_l{num_layers}_d{dropout}_hl{history_len}_ph{prediction_horizon}"
        exp_dir = str(root_dir / tag)
        os.makedirs(exp_dir, exist_ok=True)

        print(f"\n[{idx}/{total}] {tag}")
        print(f"    output_dir: {exp_dir}")

        start = time.time()
        try:
            summary = run_single_experiment(
                args, hidden_dim, num_layers, dropout,
                history_len, prediction_horizon, exp_dir,
            )
            summary["elapsed_s"] = round(time.time() - start, 2)
            summary["status"] = "success"
            results.append(summary)
            print(
                f"    [OK] best_val_loss={summary['best_val_loss']:.6f} "
                f"@ epoch {summary['best_epoch']} "
                f"({summary['elapsed_s']:.1f}s)"
            )
        except Exception as exc:
            elapsed = round(time.time() - start, 2)
            print(f"    [FAIL] FAILED after {elapsed:.1f}s: {exc}")
            results.append({
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "dropout": dropout,
                "history_len": history_len,
                "prediction_horizon": prediction_horizon,
                "status": "failed",
                "error": str(exc),
                "elapsed_s": elapsed,
            })

    # ------------------------------------------------------------------
    # Persist & rank results
    # ------------------------------------------------------------------
    df = pd.DataFrame(results)
    csv_path = root_dir / "grid_search_results.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\n[CSV] Results CSV: {csv_path}")

    successful = [r for r in results if r.get("status") == "success"]
    if not successful:
        print("⚠️  No successful experiments.")
        return

    # Selection criterion: lowest validation loss
    best = min(successful, key=lambda r: r["best_val_loss"])

    print("\n" + "=" * 60)
    print("[BEST] BEST CONFIGURATION (by lowest validation loss)")
    print("=" * 60)
    print(f"  hidden_dim:         {best['hidden_dim']}")
    print(f"  num_layers:         {best['num_layers']}")
    print(f"  dropout:            {best['dropout']}")
    print(f"  history_len:        {best['history_len']}")
    print(f"  prediction_horizon: {best['prediction_horizon']}")
    print(f"  best_val_loss:      {best['best_val_loss']:.6f}")
    print(f"  best_epoch:         {best['best_epoch']}")
    print(f"  total_epochs:       {best['total_epochs']}")
    print(f"  output_dir:         {best['output_dir']}")
    print("=" * 60)

    best_path = root_dir / "best_config.json"
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)
    print(f"[JSON] Best config JSON: {best_path}")

    # Optional: print top-5 ranking
    ranked = sorted(successful, key=lambda r: r["best_val_loss"])
    print("\n[RANK] Top-5 configurations:")
    print("-" * 60)
    for i, r in enumerate(ranked[:5], start=1):
        print(
            f"  {i}. {r['best_val_loss']:.6f} | "
            f"h{r['hidden_dim']} l{r['num_layers']} d{r['dropout']} "
            f"hl{r['history_len']} ph{r['prediction_horizon']}"
        )


if __name__ == "__main__":
    main()
