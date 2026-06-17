"""Verify that adversarial training/evaluation configs are internally consistent."""
from __future__ import annotations

import os
from typing import Any, Dict

import pytest

from uav_vpp_guidance.utils.config import load_yaml_config, merge_config


def _load_experiment_config(config_path: str) -> Dict[str, Any]:
    base_config = load_yaml_config(config_path)
    includes = base_config.pop("includes", [])
    merged: Dict[str, Any] = {}
    cfg_dir = os.path.dirname(config_path)
    for inc_path in includes:
        inc_full = os.path.join(cfg_dir, inc_path)
        if not os.path.exists(inc_full):
            inc_full = os.path.join(cfg_dir, "..", os.path.basename(inc_path))
        if os.path.exists(inc_full):
            merged = merge_config(merged, load_yaml_config(inc_full))
    return merge_config(merged, base_config)


def _config_path(name: str) -> str:
    return os.path.join(
        os.path.dirname(__file__), "..", "config", "adversarial", name
    )


@pytest.mark.parametrize(
    "cfg_name",
    [
        "train_target.yaml",
        "train_pursuer.yaml",
        "evaluate.yaml",
        "robustness_eval.yaml",
    ],
)
def test_includes_ppo_yaml(cfg_name: str):
    raw = load_yaml_config(_config_path(cfg_name))
    includes = raw.get("includes", [])
    assert "../ppo.yaml" in includes or "ppo.yaml" in includes, (
        f"{cfg_name} must include ppo.yaml so PPOAgent uses the right network size"
    )


@pytest.mark.parametrize("cfg_name", ["train_target.yaml", "evaluate.yaml", "robustness_eval.yaml"])
def test_target_policy_matches_checkpoint_architecture(cfg_name: str):
    """Target checkpoints are saved with [128, 128] tanh; the config must reflect that."""
    cfg = _load_experiment_config(_config_path(cfg_name))
    target_policy = cfg.get("target_policy", {})
    assert target_policy.get("hidden_sizes") == [128, 128]
    assert target_policy.get("activation") == "tanh"


@pytest.mark.parametrize("cfg_name", ["evaluate.yaml", "robustness_eval.yaml"])
def test_pursuer_policies_match_checkpoint_architectures(cfg_name: str):
    """Both baseline and adversarial pursuer checkpoints use [128, 128] tanh."""
    cfg = _load_experiment_config(_config_path(cfg_name))
    policies = cfg.get("policies", {})
    assert policies.get("baseline_pursuer", {}).get("hidden_sizes") == [128, 128]
    assert policies.get("baseline_pursuer", {}).get("activation") == "tanh"
    assert policies.get("adversarial_pursuer", {}).get("hidden_sizes") == [128, 128]
    assert policies.get("adversarial_pursuer", {}).get("activation") == "tanh"


def test_train_configs_enable_scenario_sampler():
    for cfg_name in ["train_target.yaml", "train_pursuer.yaml"]:
        cfg = _load_experiment_config(_config_path(cfg_name))
        sampler_cfg = cfg.get("scenario_sampler", {})
        assert sampler_cfg.get("enabled") is True
        assert sampler_cfg.get("source") in ("registry_set", "explicit_list")
