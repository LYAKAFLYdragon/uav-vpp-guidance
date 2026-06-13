"""
Experiment logging utilities.
"""

import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


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


def _get_git_info():
    """Return git commit, branch, and dirty state for the project root."""
    try:
        root = Path(__file__).parent.parent.parent.parent.resolve()
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--short", "--untracked-files=no"],
                cwd=root,
                text=True,
            ).strip()
        )
        return {"commit": commit, "branch": branch, "dirty": dirty}
    except Exception as exc:  # pragma: no cover - git may be unavailable
        return {"commit": "unknown", "branch": "unknown", "dirty": None, "error": str(exc)}


class ExperimentLogger:
    """
    Structured experiment logger.

    Writes a JSONL event stream and a run manifest JSON file with provenance
    (git commit, command line, hostname, Python version). Intended for use as
    a context manager.

    Example:
        with ExperimentLogger("outputs/runs/exp1", config=cfg) as logger:
            logger.log_metrics(step=0, metrics={"loss": 1.0})
            logger.write_manifest(results={"success_rate": 0.8})
    """

    def __init__(
        self,
        output_dir,
        experiment_name=None,
        config=None,
        manifest_filename="run_manifest.json",
        events_filename="events.jsonl",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name or self.output_dir.name
        self.config = config
        self.manifest_path = self.output_dir / manifest_filename
        self.events_path = self.output_dir / events_filename
        self.start_time = datetime.utcnow()
        self.events_file = None
        self._results = {}
        self._closed = False

    def __enter__(self):
        if self.config is not None:
            self.save_config_snapshot(self.config)
        self.events_file = open(self.events_path, "a", encoding="utf-8")
        self.log_event(
            "experiment_start",
            {
                "experiment_name": self.experiment_name,
                "start_time": self.start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "hostname": platform.node(),
                "python_version": sys.version,
                "platform": platform.platform(),
            },
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close(exception=(exc_type, exc_val, exc_tb))
        return False

    def save_config_snapshot(self, config, filename="config_snapshot.yaml"):
        """Save a configuration snapshot to the experiment directory."""
        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    def log_event(self, event_type, payload):
        """Append a single event to the JSONL stream."""
        if self._closed:
            raise RuntimeError("ExperimentLogger is closed")
        record = {
            "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "event_type": event_type,
            "payload": payload,
        }
        line = json.dumps(record, ensure_ascii=False)
        if self.events_file is not None:
            self.events_file.write(line + "\n")
            self.events_file.flush()

    def log_metrics(self, step, metrics):
        """Convenience wrapper for logging metric dicts."""
        self.log_event("metrics", {"step": step, **metrics})

    def write_manifest(self, results=None, extra=None):
        """Write the run manifest JSON file."""
        if results is not None:
            self._results.update(results)
        manifest = {
            "experiment_name": self.experiment_name,
            "output_dir": str(self.output_dir),
            "start_time": self.start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end_time": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "hostname": platform.node(),
            "python_version": sys.version,
            "platform": platform.platform(),
            "command_line": sys.argv,
            "git_info": _get_git_info(),
            "results": self._results,
        }
        if extra:
            manifest.update(extra)
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    def close(self, exception=None):
        """Close the event stream and write the final manifest."""
        if self._closed:
            return
        exc_info = None
        if exception is not None and exception[0] is not None:
            exc_info = {
                "type": exception[0].__name__,
                "message": str(exception[1]),
            }
        self.log_event(
            "experiment_end",
            {
                "end_time": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "exception": exc_info,
            },
        )
        if self.events_file is not None:
            self.events_file.close()
            self.events_file = None
        self.write_manifest()
        self._closed = True
