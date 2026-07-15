from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "scripts" / "run_thesis_reentry_terminal_reward_pilot_v1.py"


def _load():
    spec = importlib.util.spec_from_file_location("terminal_reward_pilot_runner", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_continuous_train_sampler_is_deterministic_and_within_preregistered_bounds():
    module = _load()
    distribution = yaml.safe_load(
        (ROOT / "config/experiment/thesis_reentry_terminal_reward_pilot_v1_train_distribution.yaml").read_text(encoding="utf-8")
    )
    first = module._sample_train_scenario(distribution, 7)
    second = module._sample_train_scenario(distribution, 7)

    assert first == second
    assert first["metadata"]["initial_class"] in {"head_on", "crossing_entry"}
    target = first["target_init"]["position_m"]
    own = first["own_init"]["position_m"]
    initial_range = math.sqrt(sum((float(t) - float(o)) ** 2 for t, o in zip(target, own)))
    assert 5700.0 <= initial_range <= 7900.0
    assert 230.0 <= first["own_init"]["velocity_mps"] <= 310.0
    assert 350.0 <= first["target_init"]["velocity_mps"] <= 430.0
    assert first["metadata"]["continuous_run_in"] is True


def test_runtime_design_resolves_the_hashed_preregistered_base_config():
    module = _load()
    config, base_path = module.load_runtime_config(
        ROOT / "config/experiment/thesis_reentry_terminal_reward_pilot_v1_runtime_design.yaml"
    )

    assert base_path.name == "thesis_reentry_terminal_reward_pilot_v1_preregistered.yaml"
    assert config["source_id"] == "THESIS-REENTRY-TERMINAL-REWARD-PILOT-V1"
    assert config["proposed_training"]["total_timesteps"] == 20000
    assert config["reward"]["terminal_outcome_reward"]["win_bonus"] == 1.0
