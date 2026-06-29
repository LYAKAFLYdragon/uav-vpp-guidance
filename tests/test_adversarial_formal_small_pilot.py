from __future__ import annotations

import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_adversarial_formal_small_pilot import (  # noqa: E402
    _write_curriculum_config,
    _write_hp_config,
)


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_write_hp_config_records_explicit_aoa_provenance(tmp_path):
    cfg_path = _write_hp_config(
        REPO_ROOT / "config/experiment/jsbsim_hrl_comparison.yaml",
        tmp_path,
        damage_per_step=0.5,
        close_range_max_km=3.0,
        close_range_max_aoa_deg=60.0,
    )

    cfg = _load_yaml(cfg_path)
    overrides = cfg["provenance"]["config_overrides"]

    assert cfg_path.name == "jsbsim_hrl_comparison_hp_d0p5_r3p0_a60p0.yaml"
    assert cfg["attack_zone"]["damage_per_step"] == 0.5
    assert cfg["attack_zone"]["close_range_max_km"] == 3.0
    assert cfg["attack_zone"]["close_range_max_aoa_deg"] == 60.0
    assert any(
        item["key"] == "attack_zone.close_range_max_aoa_deg"
        and item["new_value"] == 60.0
        and item["source"] == "run_adversarial_formal_small_pilot.py:hp_attack_zone"
        for item in overrides
    )
    assert any(
        item["key"] == "experiment.name"
        and item["new_value"] == "adversarial_hp_formal_small_pilot"
        for item in overrides
    )


def test_write_curriculum_config_records_yaml_mutations_in_provenance(tmp_path):
    cfg_path = _write_curriculum_config(
        REPO_ROOT / "config/experiment/train_curriculum_ppo.yaml",
        tmp_path,
        damage_per_step=0.5,
        total_timesteps=4096,
    )

    cfg = _load_yaml(cfg_path)
    overrides = cfg["provenance"]["config_overrides"]

    assert cfg["env"]["backend"] == "simple"
    assert cfg["env"]["use_jsbsim"] is False
    assert cfg["env"]["max_high_level_steps"] == 256
    assert cfg["ppo"]["total_timesteps"] == 4096
    assert cfg["attack_zone"]["enabled"] is True
    assert cfg["attack_zone"]["damage_per_step"] == 0.5
    assert any(
        item["key"] == "env.max_high_level_steps"
        and item["old_value"] == 512
        and item["new_value"] == 256
        and item["source"] == "run_adversarial_formal_small_pilot.py:curriculum_env"
        for item in overrides
    )
    assert any(
        item["key"] == "ppo.total_timesteps"
        and item["new_value"] == 4096
        and item["source"] == "run_adversarial_formal_small_pilot.py:curriculum_ppo"
        for item in overrides
    )
    assert any(
        item["key"] == "attack_zone.damage_per_step"
        and item["new_value"] == 0.5
        and item["source"] == "run_adversarial_formal_small_pilot.py:curriculum_attack_zone"
        for item in overrides
    )
