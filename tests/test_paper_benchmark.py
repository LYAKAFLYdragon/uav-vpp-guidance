"""Tests for paper benchmark components."""

import numpy as np
import pytest

# run_paper_benchmark is in scripts/, not in src package
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from run_paper_benchmark import serialize, METHODS


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
