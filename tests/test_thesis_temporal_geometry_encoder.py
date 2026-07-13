from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch
import yaml

from uav_vpp_guidance.training.thesis_temporal_geometry_encoder import (
    TemporalEncoderBatch,
    TemporalGeometryEncoder,
    deterministic_mask,
    self_supervised_loss,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "train_thesis_five_state_temporal_encoder.py"
CONFIG = REPO_ROOT / "config" / "experiment" / "train_thesis_five_state_temporal_encoder_v1.yaml"


def _collector_module():
    spec = importlib.util.spec_from_file_location("p3_collector", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_encoder_shapes_masked_loss_and_episode_independent_encoding():
    torch.manual_seed(7)
    model = TemporalGeometryEncoder(input_dim=16, hidden_dim=24, embedding_dim=32, horizons=(1, 5))
    history = torch.randn(3, 10, 16)
    generator = torch.Generator().manual_seed(8)
    mask = deterministic_mask(history, 0.25, generator)
    output = model(history, mask)
    batch = TemporalEncoderBatch(
        history=history,
        future_deltas={1: torch.randn(3, 16), 5: torch.randn(3, 16)},
    )
    losses = self_supervised_loss(output, batch, mask)

    assert output["embedding"].shape == (3, 32)
    assert output["reconstruction"].shape == (3, 10, 16)
    assert output["future_deltas"][1].shape == (3, 16)
    assert all(torch.isfinite(value) for value in losses.values())
    window = history[0].numpy()
    first = model.encode_history(window)
    _ = model.encode_history(torch.randn(1, 10, 16))
    second = model.encode_history(window)
    np.testing.assert_allclose(first, second, atol=1e-6)


def test_phase_tracker_requires_real_post_merge_then_negative_reentry_rate():
    module = _collector_module()
    tracker = module.PhaseTracker(merge_range_m=1000.0, reentry_negative_rate_mps=-25.0, min_post_merge_steps=2)

    assert tracker.update(1800.0, -100.0) == "pre_merge"
    assert tracker.update(900.0, -80.0) == "post_merge"
    assert tracker.update(1100.0, 90.0) == "post_merge"
    assert tracker.update(1000.0, -10.0) == "post_merge"
    assert tracker.update(950.0, -30.0) == "re_entry"


def test_train_sampler_is_continuous_balanced_and_sample_builder_never_crosses_episode_boundary():
    module = _collector_module()
    distribution = yaml.safe_load(
        (REPO_ROOT / "config" / "experiment" / "thesis_five_state_shared_intent_v1_train_distribution.yaml").read_text(encoding="utf-8")
    )
    scenarios = module.sample_train_scenarios(distribution)
    assert len(scenarios) == 30
    assert {item["metadata"]["initial_class"] for item in scenarios} == {
        "advantage", "head_on", "disadvantage", "neutral", "crossing_entry"
    }
    assert all(1800.0 <= item["metadata"]["initial_range_m"] <= 2500.0 for item in scenarios)
    assert all(185.0 <= item["metadata"]["own_speed_mps"] <= 225.0 for item in scenarios)

    episodes = [
        {
            "features": np.full((16, 16), fill_value=float(index), dtype=np.float32),
            "phases": ["pre_merge"] * 16,
            "initial_class": "head_on",
            "opponent_id": "expert_rule_based",
        }
        for index in (1, 2)
    ]
    samples = module.build_samples(
        episodes,
        {"mean": [0.0] * 16, "std": [1.0] * 16},
        history_steps=10,
        horizons=(1, 5),
    )
    assert samples["history"].shape[0] == 4
    assert np.all(samples["history"][0] == 1.0)
    assert np.all(samples["history"][-1] == 2.0)


def test_p3_config_prevents_policy_training_and_heldout_use():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    assert config["experiment"]["high_level_ppo_training"] == "prohibited"
    assert config["experiment"]["shared_skill_training"] == "prohibited"
    assert config["simulator"]["backend"] == "jsbsim"
    assert config["collector"]["learnable_ego_policy"] == "prohibited"
    assert config["dataset"]["normalization_fit_split"] == "train_only"
    assert config["retention"]["heldout60_use"] == "prohibited"
    assert config["model"]["frozen_for_high_level_ppo_by_default"] is True
