from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


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


def test_hierarchical_commander_preflight_allows_smaller_checkpoint_action_dim():
    assert _MODULE._checkpoint_action_dim_is_compatible(
        agent_type="hierarchical_commander",
        expected_action_dim=3,
        checkpoint_action_dim=2,
    )
    assert not _MODULE._checkpoint_action_dim_is_compatible(
        agent_type="hierarchical_commander",
        expected_action_dim=2,
        checkpoint_action_dim=3,
    )
    assert not _MODULE._checkpoint_action_dim_is_compatible(
        agent_type="ppo",
        expected_action_dim=3,
        checkpoint_action_dim=2,
    )


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
    code_deps = {Path(path).name for path in report["code_dependencies"]}
    assert "policy_network.py" in code_deps
    assert "replay_buffer.py" in code_deps
    assert "tracking_env.py" in code_deps
    assert "recorders.py" in code_deps
    assert "run_jsbsim_hrl_comparison.py" in code_deps


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


def test_collect_formal_dependencies_includes_oracle_specialists_and_predictor(tmp_path):
    import torch

    repo_root = tmp_path
    config_dir = repo_root / "config" / "experiment"
    config_dir.mkdir(parents=True)
    outputs_dir = repo_root / "outputs"
    predictor_dir = outputs_dir / "trajectory_prediction"
    predictor_dir.mkdir(parents=True)
    specialist_dir = outputs_dir / "specialists" / "head_on" / "checkpoints"
    specialist_dir.mkdir(parents=True)

    (config_dir / "formal.yaml").write_text(
        "\n".join(
            [
                "trajectory_prediction:",
                "  enabled: true",
                "  checkpoint_path: outputs/trajectory_prediction/best_model.pt",
                "run_defaults:",
                "  main_methods: [oracle]",
                "methods:",
                "  oracle:",
                "    agent_type: oracle_task_gate",
                "    specialists:",
                "      head_on:",
                "        checkpoint: outputs/specialists/head_on/checkpoints/best.pt",
                "        config_path: config/experiment/specialist.yaml",
            ]
        ),
        encoding="utf-8",
    )
    (config_dir / "specialist.yaml").write_text(
        "\n".join(
            [
                "trajectory_prediction:",
                "  enabled: true",
                "  checkpoint_path: outputs/trajectory_prediction/best_model.pt",
                "policy:",
                "  action_dim: 3",
            ]
        ),
        encoding="utf-8",
    )

    predictor_path = predictor_dir / "best_model.pt"
    predictor_path.write_bytes(b"predictor")
    checkpoint_path = specialist_dir / "best.pt"
    torch.save({"obs_dim": 18, "action_dim": 3}, checkpoint_path)

    report = _MODULE.collect_formal_dependencies(
        repo_root=repo_root,
        config_path=config_dir / "formal.yaml",
    )

    runtime_artifacts = {
        (artifact["kind"], Path(artifact["path"]).name): artifact
        for artifact in report["runtime_artifacts"]
    }
    assert ("trajectory_prediction_checkpoint", "best_model.pt") in runtime_artifacts
    assert ("oracle_specialist_checkpoint", "best.pt") in runtime_artifacts
    assert ("oracle_specialist_config", "specialist.yaml") in runtime_artifacts

    predictor_owners = runtime_artifacts[
        ("trajectory_prediction_checkpoint", "best_model.pt")
    ]["owners"]
    assert sorted(predictor_owners) == ["comparison_config", "oracle.specialists.head_on"]

    yaml_deps = {Path(path).name for path in report["yaml_dependencies"]}
    assert yaml_deps == {"formal.yaml", "specialist.yaml"}


def test_run_preflight_reports_missing_nested_specialist_and_predictor_artifacts(tmp_path):
    repo_root = tmp_path
    config_dir = repo_root / "config" / "experiment"
    config_dir.mkdir(parents=True)

    (config_dir / "formal.yaml").write_text(
        "\n".join(
            [
                "trajectory_prediction:",
                "  enabled: true",
                "  checkpoint_path: outputs/trajectory_prediction/best_model.pt",
                "run_defaults:",
                "  main_methods: [oracle]",
                "methods:",
                "  oracle:",
                "    agent_type: oracle_task_gate",
                "    specialists:",
                "      crossing_feasible:",
                "        checkpoint: outputs/specialists/crossing/checkpoints/last.pt",
                "        config_path: config/experiment/specialist.yaml",
            ]
        ),
        encoding="utf-8",
    )
    (config_dir / "specialist.yaml").write_text(
        "policy:\n  action_dim: 3\n",
        encoding="utf-8",
    )

    _report, issues = _MODULE.run_preflight(
        repo_root=repo_root,
        config_path=config_dir / "formal.yaml",
    )

    assert any(
        "trajectory_prediction_checkpoint (comparison_config): missing:" in issue
        for issue in issues
    )
    assert any(
        "oracle_specialist_checkpoint (oracle.specialists.crossing_feasible): missing:"
        in issue
        for issue in issues
    )


def test_collect_formal_dependencies_includes_commander_mode_artifacts(tmp_path):
    import torch

    repo_root = tmp_path
    config_dir = repo_root / "config" / "experiment"
    config_dir.mkdir(parents=True)
    outputs_dir = repo_root / "outputs" / "modes" / "head_on" / "checkpoints"
    outputs_dir.mkdir(parents=True)

    (config_dir / "formal.yaml").write_text(
        "\n".join(
            [
                "run_defaults:",
                "  main_methods: [cmd]",
                "methods:",
                "  cmd:",
                "    agent_type: hierarchical_commander",
                "    checkpoint: outputs/commander/best.pt",
                "    config_path: config/experiment/train.yaml",
            ]
        ),
        encoding="utf-8",
    )
    (config_dir / "train.yaml").write_text(
        "\n".join(
            [
                "commander:",
                "  num_modes: 1",
                "  modes:",
                "    - id: 0",
                "      name: head_on_specialist",
                "      checkpoint: outputs/modes/head_on/checkpoints/best.pt",
                "      config_path: config/experiment/mode.yaml",
            ]
        ),
        encoding="utf-8",
    )
    (config_dir / "mode.yaml").write_text("policy:\n  action_dim: 3\n", encoding="utf-8")

    torch.save({"obs_dim": 18, "action_dim": 3}, outputs_dir / "best.pt")

    report = _MODULE.collect_formal_dependencies(
        repo_root=repo_root,
        config_path=config_dir / "formal.yaml",
    )

    runtime_artifacts = {
        (artifact["kind"], Path(artifact["path"]).name): artifact
        for artifact in report["runtime_artifacts"]
    }
    assert ("commander_mode_checkpoint", "best.pt") in runtime_artifacts
    assert ("commander_mode_config", "mode.yaml") in runtime_artifacts


def test_collect_frozen_assets_records_hash_and_checkpoint_shape(tmp_path):
    import hashlib

    import torch

    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"obs_dim": 18, "action_dim": 3}, checkpoint)
    expected_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    report = _MODULE._collect_frozen_assets(
        repo_root=tmp_path,
        config={
            "frozen_assets": {
                "artifacts": [
                    {
                        "id": "crossing",
                        "path": "checkpoint.pt",
                        "sha256": expected_hash,
                        "expected_obs_dim": 18,
                        "expected_action_dim": 3,
                    }
                ]
            }
        },
    )

    assert report["artifacts"] == [
        {
            "id": "crossing",
            "path": str(checkpoint.resolve()),
            "exists": True,
            "expected_sha256": expected_hash,
            "actual_sha256": expected_hash,
            "expected_obs_dim": 18,
            "expected_action_dim": 3,
            "checkpoint_obs_dim": 18,
            "checkpoint_action_dim": 3,
            "checkpoint_load_error": None,
        }
    ]


def test_collect_scenario_manifest_records_hash_and_task_groups(tmp_path):
    manifest_path = tmp_path / "manifest.yaml"
    payload = {
        "source_id": "TEST-SOURCE",
        "scenarios": [
            {"name": "head", "metadata": {"task_registry_key": "head_on"}},
            {"name": "cross", "metadata": {"task_registry_key": "crossing_feasible"}},
        ],
        "integrity": {},
    }
    raw = __import__("json").dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["integrity"]["payload_sha256"] = __import__("hashlib").sha256(raw).hexdigest()
    manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    report = _MODULE._collect_scenario_manifest(
        config_path=tmp_path / "comparison.yaml",
        config={
            "scenario_manifest": {
                "path": "manifest.yaml",
                "source_id": "TEST-SOURCE",
                "payload_sha256": payload["integrity"]["payload_sha256"],
                "expected_task_counts": {"head_on": 1, "crossing_feasible": 1},
            }
        },
        yaml_dependencies=set(),
    )

    assert report["recorded_payload_sha256"] == report["actual_payload_sha256"]
    assert report["actual_task_counts"] == {"head_on": 1, "crossing_feasible": 1}
