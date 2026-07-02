from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "preflight_formal_reproducibility.py"

_SPEC = importlib.util.spec_from_file_location(
    "preflight_formal_reproducibility",
    SCRIPT_PATH,
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_expected_action_dim_for_hierarchical_commander_uses_mode_count():
    cfg = {
        "commander": {
            "modes": [
                {"id": 0, "name": "head_on_specialist"},
                {"id": 1, "name": "crossing_specialist"},
            ]
        },
        "policy": {"action_dim": 3},
    }
    assert _MODULE._expected_action_dim("hierarchical_commander", cfg) == 2


def test_collect_formal_dependencies_resolves_nested_includes_and_method_config(tmp_path):
    repo_root = tmp_path
    config_dir = repo_root / "config" / "experiment"
    config_dir.mkdir(parents=True)

    (config_dir / "base.yaml").write_text(
        "run_defaults:\n  main_methods: [cmd]\n",
        encoding="utf-8",
    )
    (config_dir / "train_base.yaml").write_text(
        "commander:\n  modes:\n    - {id: 0, name: head_on_specialist}\n    - {id: 1, name: crossing_specialist}\n",
        encoding="utf-8",
    )
    (config_dir / "train.yaml").write_text(
        "includes:\n  - ./train_base.yaml\npolicy:\n  action_dim: 3\n",
        encoding="utf-8",
    )
    (config_dir / "formal.yaml").write_text(
        "\n".join(
            [
                "includes:",
                "  - ./base.yaml",
                "methods:",
                "  cmd:",
                "    agent_type: hierarchical_commander",
                "    config_path: config/experiment/train.yaml",
            ]
        ),
        encoding="utf-8",
    )

    report = _MODULE.collect_formal_dependencies(
        repo_root=repo_root,
        config_path=config_dir / "formal.yaml",
    )

    yaml_deps = {Path(path).name for path in report["yaml_dependencies"]}
    assert yaml_deps == {"formal.yaml", "base.yaml", "train.yaml", "train_base.yaml"}
    assert report["methods"][0]["expected_action_dim"] == 2


def test_collect_formal_dependencies_raises_on_missing_include(tmp_path):
    repo_root = tmp_path
    config_dir = repo_root / "config" / "experiment"
    config_dir.mkdir(parents=True)
    missing_cfg = config_dir / "formal.yaml"
    missing_cfg.write_text("includes:\n  - ./missing.yaml\n", encoding="utf-8")

    try:
        _MODULE.collect_formal_dependencies(
            repo_root=repo_root,
            config_path=missing_cfg,
        )
    except FileNotFoundError as exc:
        assert "Included config not found" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError for missing include")
