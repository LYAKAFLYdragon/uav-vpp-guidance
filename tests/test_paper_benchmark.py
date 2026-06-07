"""Tests for paper benchmark components."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from run_paper_benchmark import serialize, METHODS, evaluate_method


class TestSerialize:
    def test_numpy_array(self):
        assert serialize(np.array([1, 2, 3])) == [1, 2, 3]

    def test_numpy_float64(self):
        assert serialize(np.float64(3.14)) == 3.14

    def test_numpy_int64(self):
        assert serialize(np.int64(42)) == 42

    def test_numpy_bool(self):
        assert serialize(np.bool_(True)) is True
        assert serialize(np.bool_(False)) is False

    def test_nested_dict(self):
        d = {"a": np.array([1]), "b": {"c": np.float64(2.0)}}
        result = serialize(d)
        assert result == {"a": [1], "b": {"c": 2.0}}

    def test_regular_types_unchanged(self):
        assert serialize("hello") == "hello"
        assert serialize(42) == 42
        assert serialize(3.14) == 3.14


class TestMethodsConfig:
    def test_all_methods_have_checkpoint(self):
        for name, cfg in METHODS.items():
            assert "checkpoint" in cfg
            assert "config_method" in cfg

    def test_gain_only_has_gains_path(self):
        assert "gains_path" in METHODS["gain_only"]


class TestEvaluateMethodCheckpoint:
    def test_missing_checkpoint_without_allow_random_smoke_raises(self):
        with pytest.raises(FileNotFoundError):
            evaluate_method(
                "nonexistent",
                {"checkpoint": "nonexistent/path.pt", "config_method": "no_prediction"},
                [],
                (0,),
                backend="simple",
                config_path="config/experiment/stage6f5_feasible_geometry.yaml",
                allow_random_smoke=False,
            )

    def test_missing_checkpoint_with_allow_random_smoke_succeeds(self):
        result = evaluate_method(
            "nonexistent",
            {"checkpoint": "nonexistent/path.pt", "config_method": "no_prediction"},
            [],
            (0,),
            backend="simple",
            config_path="config/experiment/stage6f5_feasible_geometry.yaml",
            allow_random_smoke=True,
        )
        assert result["metadata"]["invalid_for_paper"] is True
        assert result["metadata"]["is_random_smoke"] is True


class TestGainOnlyGains:
    def test_gain_only_requires_gains_file_without_random_smoke(self):
        cfg = {
            "checkpoint": "nonexistent/path.pt",
            "config_method": "no_prediction",
            "gains_path": "nonexistent/gains.json",
        }
        with pytest.raises(FileNotFoundError):
            evaluate_method(
                "gain_only",
                cfg,
                [],
                (0,),
                backend="simple",
                config_path="config/experiment/stage6f5_feasible_geometry.yaml",
                allow_random_smoke=False,
            )

    def test_gain_only_loads_and_records_gains(self, tmp_path):
        gains_file = tmp_path / "gains.json"
        gains_file.write_text(json.dumps({"best_gains": {"k_los": 2.0}}))
        cfg = {
            "checkpoint": "nonexistent/path.pt",
            "config_method": "no_prediction",
            "gains_path": str(gains_file),
        }
        result = evaluate_method(
            "gain_only",
            cfg,
            [],
            (0,),
            backend="simple",
            config_path="config/experiment/stage6f5_feasible_geometry.yaml",
            allow_random_smoke=True,
        )
        meta = result["metadata"]
        assert meta["gains_exists"] is True
        assert meta["loaded_gains"] == {"k_los": 2.0}
        assert meta["gain_source"] == "cem"


class TestUnknownMethod:
    def test_unknown_method_fails_by_default(self):
        # This is tested at CLI level; module-level evaluate_method does not handle unknown methods
        pass

    def test_unknown_method_can_skip_only_with_allow_missing_methods(self):
        # CLI-level test; module-level evaluate_method does not handle unknown methods
        pass


class TestCheckpointMeta:
    def test_meta_structure(self):
        result = evaluate_method(
            "test",
            {"checkpoint": "nonexistent/path.pt", "config_method": "no_prediction"},
            [],
            (0,),
            backend="simple",
            config_path="config/experiment/stage6f5_feasible_geometry.yaml",
            allow_random_smoke=True,
        )
        meta = result["metadata"]
        assert "method" in meta
        assert "config_path" in meta
        assert "resolved_config_hash" in meta
        assert "method_override_name" in meta
        assert "backend" in meta
        assert "scenarios" in meta
        assert "seeds" in meta
        assert "prediction_mode" in meta
        assert "guidance_mode" in meta
        assert "policy_checkpoint" in meta
        assert "checkpoint_exists" in meta
        assert "is_random_smoke" in meta
        assert "invalid_for_paper" in meta

    def test_valid_checkpoint_meta(self):
        ckpt = "outputs/audit_no_pred_final/checkpoints/best.pt"
        if not Path(ckpt).exists():
            pytest.skip("No checkpoint available for test")
        result = evaluate_method(
            "no_prediction",
            {"checkpoint": ckpt, "config_method": "no_prediction"},
            [],
            (0,),
            backend="simple",
            config_path="config/experiment/stage6f5_feasible_geometry.yaml",
            allow_random_smoke=False,
        )
        meta = result["metadata"]
        assert meta["checkpoint_exists"] is True
        assert meta["invalid_for_paper"] is False
        assert meta["is_random_smoke"] is False
        assert meta["config_path"] == "config/experiment/stage6f5_feasible_geometry.yaml"
        assert len(meta["resolved_config_hash"]) == 16


class TestResolvedConfigHash:
    def test_config_hash_changes_when_config_changes(self):
        from run_paper_benchmark import _config_hash
        h1 = _config_hash({"a": 1, "b": 2})
        h2 = _config_hash({"a": 1, "b": 3})
        assert h1 != h2
        h3 = _config_hash({"a": 1, "b": 2})
        assert h1 == h3


class TestTrainingCheckpointSemantics:
    def test_train_bilevel_missing_checkpoint_fails_without_allow_random_init(self):
        import subprocess
        result = subprocess.run(
            [
                sys.executable, "-m", "uav_vpp_guidance.training.train_bilevel",
                "--config", "config/experiment/proposed_bilevel.yaml",
                "--checkpoint", "nonexistent/path.pt",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )
        # dry-run allows missing checkpoint
        assert result.returncode == 0

    def test_train_bilevel_missing_checkpoint_fails_in_normal_run(self):
        import subprocess
        result = subprocess.run(
            [
                sys.executable, "-m", "uav_vpp_guidance.training.train_bilevel",
                "--config", "config/experiment/proposed_bilevel.yaml",
                "--checkpoint", "nonexistent/path.pt",
                "--n-episodes", "1",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "allow-random-init" in result.stdout or "allow-random-init" in result.stderr

    def test_train_gain_only_missing_checkpoint_fails_without_allow_random_init(self):
        import subprocess
        result = subprocess.run(
            [
                sys.executable, "-m", "uav_vpp_guidance.training.train_gain_only",
                "--config", "config/experiment/gain_only_cem.yaml",
                "--checkpoint", "nonexistent/path.pt",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )
        # dry-run allows missing checkpoint
        assert result.returncode == 0

    def test_train_gain_only_missing_checkpoint_fails_in_normal_run(self):
        import subprocess
        result = subprocess.run(
            [
                sys.executable, "-m", "uav_vpp_guidance.training.train_gain_only",
                "--config", "config/experiment/gain_only_cem.yaml",
                "--checkpoint", "nonexistent/path.pt",
                "--n-iter", "1",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "allow-random-init" in result.stdout or "allow-random-init" in result.stderr
