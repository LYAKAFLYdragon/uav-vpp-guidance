"""Tests for the structured ExperimentLogger utility."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from uav_vpp_guidance.utils.logger import ExperimentLogger


def test_logger_creates_artifacts():
    config = {"env": {"backend": "simple"}, "ppo": {"lr": 3e-4}}
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "exp"
        with ExperimentLogger(out, experiment_name="test_exp", config=config) as logger:
            logger.log_metrics(step=0, metrics={"loss": 1.0})
            logger.log_metrics(step=1, metrics={"loss": 0.5})
            logger.write_manifest(results={"success_rate": 0.8})

        assert (out / "config_snapshot.yaml").exists()
        assert (out / "events.jsonl").exists()
        assert (out / "run_manifest.json").exists()

        # Config snapshot is valid YAML
        import yaml
        with open(out / "config_snapshot.yaml", "r", encoding="utf-8") as f:
            saved = yaml.safe_load(f)
        assert saved["env"]["backend"] == "simple"

        # Events JSONL has start, metrics, end
        events = []
        with open(out / "events.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                events.append(json.loads(line))
        types = [e["event_type"] for e in events]
        assert types[0] == "experiment_start"
        assert types[-1] == "experiment_end"
        assert "metrics" in types

        # Manifest contains provenance and results
        with open(out / "run_manifest.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)
        assert manifest["experiment_name"] == "test_exp"
        assert manifest["results"]["success_rate"] == pytest.approx(0.8)
        assert "git_info" in manifest
        assert "command_line" in manifest
        assert "hostname" in manifest


def test_logger_no_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "exp"
        with ExperimentLogger(out) as logger:
            logger.log_event("custom", {"value": 42})
        assert not (out / "config_snapshot.yaml").exists()
        with open(out / "events.jsonl", "r", encoding="utf-8") as f:
            events = [json.loads(line) for line in f]
        assert any(e["event_type"] == "custom" for e in events)


def test_logger_closed_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "exp"
        logger = ExperimentLogger(out)
        with logger:
            logger.log_metrics(step=0, metrics={"x": 1})
        with pytest.raises(RuntimeError):
            logger.log_event("late", {})
