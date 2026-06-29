from __future__ import annotations

from pathlib import Path

from uav_vpp_guidance.evaluation.combat_checkpoint_retrospective import (
    build_retrospective_config,
    discover_checkpoints,
)


def test_discover_checkpoints_orders_steps_then_best_then_last(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    for name in ("step_8192.pt", "best.pt", "last.pt", "step_4096.pt"):
        (checkpoint_dir / name).write_text("stub", encoding="utf-8")

    specs = discover_checkpoints(checkpoint_dir)

    assert [spec["tag"] for spec in specs] == [
        "step_4096",
        "step_8192",
        "best",
        "last",
    ]
    assert specs[0]["label_suffix"] == "step 4096"
    assert specs[2]["label_suffix"] == "best"


def test_build_retrospective_config_replaces_template_method_with_checkpoint_variants():
    template_config = {
        "experiment": {"name": "demo_best"},
        "run_defaults": {
            "main_methods": ["baseline_method", "target_method"],
            "tasks": ["head_on", "crossing_feasible"],
        },
        "methods": {
            "baseline_method": {
                "label": "Baseline",
                "checkpoint": "outputs/experiments/baseline/checkpoints/last.pt",
                "config_path": "config/experiment/baseline.yaml",
                "prediction_variant": "learned",
            },
            "target_method": {
                "label": "Target Best",
                "checkpoint": "outputs/experiments/target/checkpoints/best.pt",
                "config_path": "config/experiment/target.yaml",
                "prediction_variant": "learned",
            },
        },
    }
    checkpoint_specs = [
        {
            "tag": "step_4096",
            "label_suffix": "step 4096",
            "checkpoint": "outputs/experiments/target/checkpoints/step_4096.pt",
        },
        {
            "tag": "best",
            "label_suffix": "best",
            "checkpoint": "outputs/experiments/target/checkpoints/best.pt",
        },
    ]

    retrospective = build_retrospective_config(
        template_config,
        retrospective_method="target_method",
        checkpoint_specs=checkpoint_specs,
        experiment_name="demo_retrospective",
    )

    assert retrospective["experiment"]["name"] == "demo_retrospective"
    assert retrospective["run_defaults"]["main_methods"] == [
        "baseline_method",
        "target_method__step_4096",
        "target_method__best",
    ]
    assert "target_method" not in retrospective["methods"]
    assert retrospective["methods"]["target_method__step_4096"] == {
        "label": "Target Best (step 4096)",
        "checkpoint": "outputs/experiments/target/checkpoints/step_4096.pt",
        "config_path": "config/experiment/target.yaml",
        "prediction_variant": "learned",
    }
    assert retrospective["methods"]["target_method__best"] == {
        "label": "Target Best (best)",
        "checkpoint": "outputs/experiments/target/checkpoints/best.pt",
        "config_path": "config/experiment/target.yaml",
        "prediction_variant": "learned",
    }
