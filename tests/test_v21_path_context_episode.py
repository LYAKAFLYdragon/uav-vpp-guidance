from __future__ import annotations

import ast
from pathlib import Path


def test_v21_episode_observes_before_applying_anchor_protocol():
    path = Path("scripts/v21_path_context_episode.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_observe")
    calls = [node.value.func for node in method.body if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)]
    assert isinstance(calls[0], ast.Attribute) and calls[0].attr == "_observe"
    assert isinstance(calls[1], ast.Attribute) and calls[1].attr == "_v21_after_observation"
