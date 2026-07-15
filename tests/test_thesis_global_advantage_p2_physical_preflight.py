from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "run_thesis_global_advantage_p2_physical_preflight.py"


def _module():
    spec = importlib.util.spec_from_file_location("p2_physical_runner", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_finite_vector_rejects_nonfinite_or_wrong_shape():
    runner = _module()
    assert runner._finite_vector([0.1, 0.2, 0.3], "action") == [0.1, 0.2, 0.3]
    for action in ([0.1, 0.2], [0.1, np.nan, 0.3]):
        try:
            runner._finite_vector(action, "action")
        except runner.PreflightError:
            pass
        else:
            raise AssertionError("invalid action must fail closed")
