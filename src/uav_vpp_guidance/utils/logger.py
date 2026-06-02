"""
Experiment logging utilities.
"""

import os
import json
import yaml
from datetime import datetime


def create_experiment_dir(base_dir, experiment_name):
    """
    Create a timestamped experiment directory.

    Args:
        base_dir (str): Base output directory.
        experiment_name (str): Experiment name.

    Returns:
        str: Path to created experiment directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def save_config_snapshot(config, exp_dir, filename="config_snapshot.yaml"):
    """
    Save a configuration snapshot to the experiment directory.

    Args:
        config (dict): Configuration dictionary.
        exp_dir (str): Experiment directory path.
        filename (str): Snapshot filename.
    """
    path = os.path.join(exp_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def log_metrics(metrics, exp_dir, step, filename="metrics.jsonl"):
    """
    Append metrics to a JSONL log file.

    Args:
        metrics (dict): Metrics dictionary.
        exp_dir (str): Experiment directory path.
        step (int): Training step or episode number.
        filename (str): Log filename.
    """
    path = os.path.join(exp_dir, filename)
    record = {"step": step, **metrics}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
