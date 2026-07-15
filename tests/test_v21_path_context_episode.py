from __future__ import annotations

import ast
from pathlib import Path


def test_v21_episode_observes_before_applying_anchor_protocol():
    path = Path("scripts/v21_path_context_episode.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_observe")
    calls = [ast.unparse(node.value) for node in method.body if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)]
    assert calls == ["super()._observe(observation)", "self._v21_after_observation()"]
