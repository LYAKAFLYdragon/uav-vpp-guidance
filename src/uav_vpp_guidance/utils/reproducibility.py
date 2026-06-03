"""
Reproducibility utilities for experiment tracking.

Provides run metadata capture, timestamped output directories,
and lightweight provenance logging without external dependencies.
"""

import json
import os
import platform
import sys
from datetime import datetime
from typing import Any, Dict, Optional


def get_git_info() -> Dict[str, Optional[str]]:
    """Capture lightweight git information if available."""
    try:
        import subprocess

        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
        dirty = (
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
        return {
            "commit": commit,
            "branch": branch,
            "dirty": bool(dirty),
        }
    except Exception:
        return {"commit": None, "branch": None, "dirty": None}


def get_run_metadata(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Build a dictionary of run metadata for provenance tracking.

    Args:
        config (dict, optional): Experiment configuration to embed.

    Returns:
        dict: Metadata including timestamp, python version, git info, and config.
    """
    metadata = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "python_version": sys.version,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "git": get_git_info(),
    }
    if config is not None:
        # Avoid mutating the caller's dict
        metadata["config"] = dict(config)
    return metadata


def make_timestamped_output_dir(
    root: str,
    experiment_name: str,
    timestamp: Optional[str] = None,
) -> str:
    """
    Create a timestamped output directory.

    Args:
        root (str): Base output directory (e.g. "outputs/benchmarks").
        experiment_name (str): Experiment identifier.
        timestamp (str, optional): ISO timestamp string; defaults to now.

    Returns:
        str: Path to the created directory.
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(root, experiment_name, timestamp)
    os.makedirs(out, exist_ok=True)
    return out


def save_run_metadata(
    output_dir: str,
    metadata: Dict[str, Any],
    filename: str = "run_metadata.json",
) -> str:
    """
    Save run metadata to a JSON file.

    Args:
        output_dir (str): Directory to write into.
        metadata (dict): Metadata dictionary.
        filename (str): Output filename.

    Returns:
        str: Path to the written file.
    """
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
    return path
