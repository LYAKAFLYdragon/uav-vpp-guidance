from __future__ import annotations

import csv
from pathlib import Path

import torch

from uav_vpp_guidance.evaluation.lstm_retraining_v2 import (
    PREDICTOR_CONTRACT,
    inspect_policy_checkpoint,
    strict_load_predictor,
    verify_corpus,
)
from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.trajectory_prediction.lstm_predictor import LSTMTrajectoryPredictor
from uav_vpp_guidance.trajectory_prediction.predictor_adapter import TrajectoryPredictorAdapter
from uav_vpp_guidance.trajectory_prediction.state_buffer import TrajectoryStateBuffer
from scripts.verify_lstm_retraining_v2 import verify_source


def _write_episode(path: Path, backend: str = "jsbsim") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "step", "time", "ego_x", "ego_y", "ego_z", "target_x", "target_y", "target_z",
        "range_m", "ata_deg", "aspect_deg", "nz_cmd", "roll_rate_cmd", "throttle_cmd",
        "done", "termination_reason", "backend",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for step in range(24):
            writer.writerow({
                "step": step, "time": step * 0.2, "ego_x": step, "ego_y": 0, "ego_z": 5000,
                "target_x": 1000 - step, "target_y": 10, "target_z": 5000,
                "range_m": 1000 - step, "ata_deg": 20, "aspect_deg": 30,
                "nz_cmd": 1, "roll_rate_cmd": 0, "throttle_cmd": 0.6,
                "done": step == 23, "termination_reason": "timeout" if step == 23 else "", "backend": backend,
            })


def test_predictor_strict_contract(tmp_path: Path):
    model = LSTMTrajectoryPredictor(**PREDICTOR_CONTRACT)
    path = tmp_path / "best_model.pt"
    torch.save(model.state_dict(), path)
    info = strict_load_predictor(path)
    assert info["strict_load"] is True
    assert info["architecture"] == PREDICTOR_CONTRACT


def test_corpus_rejects_non_jsbsim(tmp_path: Path):
    _write_episode(tmp_path / "episode.csv", backend="simple")
    result = verify_corpus(tmp_path)
    assert not result["ok"]
    assert any("strictly jsbsim" in error for error in result["errors"])


def test_policy_requires_full_training(tmp_path: Path):
    config = {"virtual_point": {"action_dim": 3}, "policy": {"action_dim": 3}}
    agent = PPOAgent(obs_dim=16, action_dim=3, config=config, device="cpu")
    path = tmp_path / "last.pt"
    agent.total_timesteps = 1024
    agent.save(str(path))
    try:
        inspect_policy_checkpoint(path, min_timesteps=100_000)
    except ValueError as exc:
        assert "total_timesteps" in str(exc)
    else:
        raise AssertionError("1024-step checkpoint must not pass full-training contract")


def test_neural_predictor_padding_is_explicit_opt_in():
    model = LSTMTrajectoryPredictor(**PREDICTOR_CONTRACT)
    feature = [0.0] * PREDICTOR_CONTRACT["input_dim"]

    strict_buffer = TrajectoryStateBuffer(10, 16, padding_mode="repeat_first")
    strict_adapter = TrajectoryPredictorAdapter(
        model,
        strict_buffer,
        {"history": {"history_len": 10, "padding_mode": "repeat_first"}, "prediction": {"fallback_mode": "none"}},
    )
    strict_buffer.push(feature)
    try:
        strict_adapter.predict({"position_neu": [0.0, 0.0, 0.0]})
    except RuntimeError as exc:
        assert "history frames" in str(exc)
    else:
        raise AssertionError("neural padding must remain disabled by default")

    padded_buffer = TrajectoryStateBuffer(10, 16, padding_mode="repeat_first")
    padded_adapter = TrajectoryPredictorAdapter(
        model,
        padded_buffer,
        {
            "history": {
                "history_len": 10,
                "padding_mode": "repeat_first",
                "allow_neural_padding": True,
            },
            "prediction": {"fallback_mode": "none"},
        },
    )
    padded_buffer.push(feature)
    _position, _variance, info = padded_adapter.predict(
        {"position_neu": [0.0, 0.0, 0.0]}
    )
    assert info["fallback"] is False
    assert info["prediction_valid"] is True
    assert info["history_padded"] is True
    assert info["history_observed_frames"] == 1


def test_source_verifier_requires_complete_policy_seed_set(tmp_path: Path):
    result = verify_source(tmp_path)
    assert result["ok"] is False
    assert any("policy seed set" in error for error in result["errors"])
