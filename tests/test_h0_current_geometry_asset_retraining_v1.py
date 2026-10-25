from __future__ import annotations

from pathlib import Path

import yaml

from scripts.run_h0_current_geometry_asset_retraining_v1 import (
    E2E_CONFIG,
    H0_CONFIG,
    SOURCE_ID,
    load_yaml,
    write_resolved_config,
)


def test_asset_configs_lock_the_two_control_interfaces():
    h0 = load_yaml(H0_CONFIG)
    e2e = load_yaml(E2E_CONFIG)
    assert h0["experiment"]["source_id"] == SOURCE_ID
    assert h0["trajectory_prediction"]["enabled"] is False
    assert h0["virtual_point"]["anchor_mode"] == "current_target"
    assert e2e["experiment"]["source_id"] == SOURCE_ID
    assert e2e["end_to_end"]["enabled"] is True
    assert e2e["virtual_point"]["enabled"] is False
    assert h0["policy"]["action_dim"] == e2e["policy"]["action_dim"] == 3


def test_resolved_config_records_seed_and_source(tmp_path: Path):
    config = load_yaml(H0_CONFIG)
    path = tmp_path / "resolved_config.yaml"
    write_resolved_config(path, config, 9202)
    resolved = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert resolved["experiment"]["source_id"] == SOURCE_ID
    assert resolved["experiment"]["seed"] == 9202
    assert resolved["ppo"]["seed"] == 9202
