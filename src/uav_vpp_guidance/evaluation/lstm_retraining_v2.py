"""Contracts and read-only verification helpers for the V2 retraining source."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch

from uav_vpp_guidance.common.git import get_git_info
from uav_vpp_guidance.trajectory_prediction.lstm_predictor import LSTMTrajectoryPredictor


SOURCE_ID = "THESIS-PREDICTED-GEOMETRY-LSTM-VPP-RETRAIN-V2-20260903"
PREDICTOR_CONTRACT = {
    "input_dim": 16,
    "hidden_dim": 128,
    "num_layers": 2,
    "dropout": 0.1,
    "predict_variance": False,
}
REQUIRED_EPISODE_COLUMNS = {
    "step", "time", "ego_x", "ego_y", "ego_z", "target_x", "target_y", "target_z",
    "range_m", "ata_deg", "aspect_deg", "nz_cmd", "roll_rate_cmd", "throttle_cmd",
    "done", "termination_reason", "backend",
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_receipt(path: str | Path) -> dict[str, Any]:
    item = Path(path)
    if not item.is_file():
        return {"path": str(item.resolve()), "exists": False, "size_bytes": None, "sha256": None}
    return {
        "path": str(item.resolve()),
        "exists": True,
        "size_bytes": item.stat().st_size,
        "sha256": file_sha256(item),
    }


def canonical_config_sha256(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def strict_load_predictor(path: str | Path, device: str = "cpu") -> dict[str, Any]:
    """Load the exact architecture with strict state-dict matching."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    model = LSTMTrajectoryPredictor(**PREDICTOR_CONTRACT).to(device)
    payload = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("predictor checkpoint must be a state dict")
    state_dict = payload.get("state_dict", payload)
    if not isinstance(state_dict, dict):
        raise TypeError("predictor checkpoint does not contain a state dict")
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise ValueError(f"strict predictor load mismatch: missing={missing}, unexpected={unexpected}")
    return {
        "receipt": file_receipt(checkpoint_path),
        "architecture": dict(PREDICTOR_CONTRACT),
        "strict_load": True,
        "state_dict_keys": len(state_dict),
    }


def inspect_policy_checkpoint(path: str | Path, min_timesteps: int = 100_000) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("policy checkpoint must be a metadata dictionary")
    errors: list[str] = []
    if int(payload.get("obs_dim", -1)) != 16:
        errors.append(f"obs_dim={payload.get('obs_dim')}, expected 16")
    if int(payload.get("action_dim", -1)) != 3:
        errors.append(f"action_dim={payload.get('action_dim')}, expected 3")
    if not isinstance(payload.get("network_state_dict"), dict):
        errors.append("network_state_dict is missing")
    total_timesteps = int(payload.get("total_timesteps", 0) or 0)
    if total_timesteps < min_timesteps:
        errors.append(f"total_timesteps={total_timesteps}, expected at least {min_timesteps}")
    if errors:
        raise ValueError("policy checkpoint contract failed: " + "; ".join(errors))
    return {
        "receipt": file_receipt(checkpoint_path),
        "obs_dim": int(payload["obs_dim"]),
        "action_dim": int(payload["action_dim"]),
        "total_timesteps": total_timesteps,
        "total_updates": int(payload.get("total_updates", 0) or 0),
        "full_training": True,
    }


def verify_corpus(root: str | Path, min_files: int = 1, min_rows: int = 15) -> dict[str, Any]:
    root_path = Path(root)
    files = sorted(root_path.rglob("*.csv"))
    errors: list[str] = []
    records: list[dict[str, Any]] = []
    if len(files) < min_files:
        errors.append(f"only {len(files)} episode CSV files found; need at least {min_files}")
    numeric = ["time", "ego_x", "ego_y", "ego_z", "target_x", "target_y", "target_z", "range_m"]
    for path in files:
        try:
            frame = pd.read_csv(path)
            missing = sorted(REQUIRED_EPISODE_COLUMNS - set(frame.columns))
            if missing:
                errors.append(f"{path}: missing columns {missing}")
            eligible_for_training = len(frame) >= min_rows
            if "backend" in frame and set(frame["backend"].dropna().astype(str)) != {"jsbsim"}:
                errors.append(f"{path}: backend is not strictly jsbsim")
            for column in numeric:
                if column in frame and not frame[column].map(lambda value: math.isfinite(float(value))).all():
                    errors.append(f"{path}: non-finite values in {column}")
            if "step" in frame and frame["step"].tolist() != list(range(len(frame))):
                errors.append(f"{path}: step sequence is not contiguous from zero")
            records.append({"path": str(path.resolve()), "rows": len(frame), "eligible_for_training": eligible_for_training, "sha256": file_sha256(path)})
        except Exception as exc:
            errors.append(f"{path}: unreadable CSV ({type(exc).__name__}: {exc})")
    eligible_count = sum(1 for item in records if item.get("eligible_for_training"))
    if eligible_count < min_files:
        errors.append(f"only {eligible_count} eligible episode CSV files found; need at least {min_files}")
    return {"ok": not errors, "root": str(root_path.resolve()), "file_count": len(files), "eligible_file_count": eligible_count, "excluded_short_file_count": len(records) - eligible_count, "files": records, "errors": errors}


def verify_formal_metrics(metrics: dict[str, Any], expected_episodes: int) -> dict[str, Any]:
    errors: list[str] = []
    episodes = metrics.get("raw_episodes", [])
    if len(episodes) != expected_episodes:
        errors.append(f"raw episode count={len(episodes)}, expected {expected_episodes}")
    for index, episode in enumerate(episodes):
        for key in ("prediction_fallback_rate", "runtime_fallback_rate", "post_warmup_fallback_rate"):
            value = episode.get(key, 0.0)
            if value is not None and math.isfinite(float(value)) and float(value) > 0.0:
                errors.append(f"episode {index}: {key}={value}")
        if episode.get("reason") in (None, "", "unknown"):
            errors.append(f"episode {index}: missing terminal reason")
    return {"ok": not errors, "episode_count": len(episodes), "errors": errors}


def source_manifest(repo_root: str | Path, *, config_receipt: dict[str, Any], corpus: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_id": SOURCE_ID,
        "purpose": "Independent predictor and full LSTM-VPP policy retraining",
        "git_info": get_git_info(str(repo_root)),
        "config": config_receipt,
        "predictor_contract": dict(PREDICTOR_CONTRACT),
        "corpus": corpus,
        "historical_assets_reused": False,
        "historical_freeze_modified": False,
    }
