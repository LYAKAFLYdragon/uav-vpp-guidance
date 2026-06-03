"""
Neural Trajectory Predictor (LSTM/GRU) Offline Training Pipeline.

Usage:
    python -m uav_vpp_guidance.trajectory_prediction.train_pipeline \
        --config config/trajectory_prediction.yaml \
        --data-dir outputs/trajectories \
        --model-type lstm
"""

import argparse
import os
import sys

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from uav_vpp_guidance.utils.config import load_yaml_config
from uav_vpp_guidance.utils.seed import set_seed
from uav_vpp_guidance.trajectory_prediction.dataset import TrajectoryPredictionDataset
from uav_vpp_guidance.trajectory_prediction.trainer import TrajectoryPredictorTrainer
from uav_vpp_guidance.trajectory_prediction.lstm_predictor import (
    LSTMTrajectoryPredictor,
)
from uav_vpp_guidance.trajectory_prediction.gru_predictor import GRUTrajectoryPredictor


def gather_episode_csv_paths(data_dir):
    """递归收集 data_dir 下所有 .csv 文件作为 episode 轨迹。"""
    paths = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".csv"):
                paths.append(os.path.join(root, f))
    if not paths:
        raise ValueError(f"No CSV files found in {data_dir}")
    return paths


def create_model(model_type, model_cfg):
    """根据类型创建模型实例。"""
    if model_type == "lstm":
        return LSTMTrajectoryPredictor(
            input_dim=model_cfg.get("input_dim", 16),
            hidden_dim=model_cfg.get("hidden_dim", 128),
            num_layers=model_cfg.get("num_layers", 2),
            dropout=model_cfg.get("dropout", 0.1),
            predict_variance=model_cfg.get("predict_variance", False),
        )
    elif model_type == "gru":
        return GRUTrajectoryPredictor(
            input_dim=model_cfg.get("input_dim", 16),
            hidden_dim=model_cfg.get("hidden_dim", 128),
            num_layers=model_cfg.get("num_layers", 2),
            dropout=model_cfg.get("dropout", 0.1),
            predict_variance=model_cfg.get("predict_variance", False),
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def main():
    parser = argparse.ArgumentParser(
        description="Offline supervised training for LSTM/GRU trajectory predictor"
    )
    parser.add_argument(
        "--config",
        default="config/trajectory_prediction.yaml",
        help="Path to trajectory prediction config",
    )
    parser.add_argument(
        "--data-dir",
        default="outputs/trajectories",
        help="Directory containing episode trajectory CSVs",
    )
    parser.add_argument(
        "--model-type",
        choices=["lstm", "gru"],
        default="lstm",
        help="Predictor architecture",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for checkpoints and logs (overrides config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation set ratio",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="torch device (cpu/cuda)",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    # Load config
    config = load_yaml_config(args.config)

    # Resolve output directory
    output_dir = args.output_dir or config.get("training", {}).get(
        "output_dir", "outputs/trajectory_prediction"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Gather trajectory CSVs
    print(f"Scanning trajectory CSVs in: {args.data_dir}")
    csv_paths = gather_episode_csv_paths(args.data_dir)
    print(f"Found {len(csv_paths)} episode CSV files.")

    # Build dataset
    print("Building dataset from episode logs...")
    dataset = TrajectoryPredictionDataset.from_episode_logs(csv_paths, config)
    print(f"Total samples: {len(dataset)}")

    if len(dataset) == 0:
        print("ERROR: Dataset is empty. Exiting.")
        sys.exit(1)

    # Train/val split
    train_idx, val_idx = train_test_split(
        np.arange(len(dataset)), test_size=args.val_ratio, random_state=args.seed
    )
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size or config.get("training", {}).get("batch_size", 32),
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size or config.get("training", {}).get("batch_size", 32),
        shuffle=False,
        drop_last=False,
    )

    # Create model
    model_cfg = config.get("model", {})
    model = create_model(args.model_type, model_cfg)
    print(
        f"Model: {args.model_type.upper()} | params={sum(p.numel() for p in model.parameters()):,}"
    )

    # Training config
    training_cfg = config.get("training", {})
    if args.epochs is not None:
        training_cfg["epochs"] = args.epochs
    if args.device is not None:
        training_cfg["device"] = args.device
    training_cfg["output_dir"] = output_dir

    # Trainer
    trainer = TrajectoryPredictorTrainer(model, train_loader, val_loader, training_cfg)

    # Fit
    trainer.fit()

    print(f"\nBest model saved to: {os.path.join(output_dir, 'best_model.pt')}")
    print(f"Latest model saved to: {os.path.join(output_dir, 'latest_model.pt')}")


if __name__ == "__main__":
    main()
