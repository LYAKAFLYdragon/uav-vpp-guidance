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


class TestEvaluateMethodCheckpoint:
    def test_missing_checkpoint_without_allow_random_smoke_raises(self):
        with pytest.raises(FileNotFoundError):
            evaluate_method(
                "nonexistent",
                {"checkpoint": "nonexistent/path.pt", "config_method": "no_prediction"},
                [],
                (0,),
                allow_random_smoke=False,
            )

    def test_missing_checkpoint_with_allow_random_smoke_succeeds(self):
        # Empty scenarios/seeds → no env interaction needed
        result = evaluate_method(
            "nonexistent",
            {"checkpoint": "nonexistent/path.pt", "config_method": "no_prediction"},
            [],
            (0,),
            allow_random_smoke=True,
        )
        assert result["metadata"]["invalid_for_paper"] is True
        assert result["metadata"]["is_random_smoke"] is True


class TestCheckpointMeta:
    def test_meta_structure(self):
        result = evaluate_method(
            "test",
            {"checkpoint": "nonexistent/path.pt", "config_method": "no_prediction"},
            [],
            (0,),
            allow_random_smoke=True,
        )
        meta = result["metadata"]
        assert "method" in meta
        assert "prediction_mode" in meta
        assert "guidance_mode" in meta
        assert "gain_source" in meta
        assert "policy_checkpoint" in meta
        assert "checkpoint_exists" in meta
        assert "is_random_smoke" in meta
        assert "invalid_for_paper" in meta

    def test_valid_checkpoint_meta(self):
        # Use an existing checkpoint if available
        ckpt = "outputs/audit_no_pred_final/checkpoints/best.pt"
        if not Path(ckpt).exists():
            pytest.skip("No checkpoint available for test")
        result = evaluate_method(
            "no_prediction",
            {"checkpoint": ckpt, "config_method": "no_prediction"},
            [],
            (0,),
            allow_random_smoke=False,
        )
        meta = result["metadata"]
        assert meta["checkpoint_exists"] is True
        assert meta["invalid_for_paper"] is False
        assert meta["is_random_smoke"] is False
