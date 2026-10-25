"""Shared contracts and analysis helpers for the H0/H1/H2 protocol."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from ..common.git import get_git_info
from ..utils.config import merge_config


SOURCE_ID = "THESIS-PREDICTED-GEOMETRY-HIGHLEVEL-OBSERVATION-V1"
EXPECTED_SCENARIOS = (
    "head_on",
    "crossing_left",
    "crossing_right",
    "neutral",
    "disadvantage",
)
EXPECTED_METHODS = (
    "h0_current_geometry",
    "h1_frozen_predicted_geometry",
    "h2_retrained_predicted_geometry",
)
EXPECTED_OPPONENTS = ("expert", "end_to_end")
BASE_OBS_DIM = 16
PREDICTOR_CONTRACT = {
    "input_dim": 16,
    "hidden_dim": 128,
    "num_layers": 2,
    "dropout": 0.1,
    "predict_variance": False,
}


def load_resolved_config(path: str | Path) -> dict[str, Any]:
    """Load YAML and recursively resolve local ``includes`` entries."""
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    merged: dict[str, Any] = {}
    for include in raw.pop("includes", []) or []:
        include_path = (path.parent / str(include)).resolve()
        merged = merge_config(merged, load_resolved_config(include_path))
    return merge_config(merged, raw)


def method_config(base_config: dict[str, Any], method_name: str) -> dict[str, Any]:
    methods = base_config.get("methods", {})
    if method_name not in methods:
        raise KeyError(f"Unknown protocol method: {method_name}")
    override = copy.deepcopy(methods[method_name])
    override.pop("checkpoint", None)
    override.pop("label", None)
    override.pop("mode", None)
    override.pop("training_allowed", None)
    result = merge_config(copy.deepcopy(base_config), override)
    result.pop("methods", None)
    return result


def file_sha256(path: str | Path) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_receipt(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    return {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "sha256": file_sha256(path),
    }


def protocol_receipt(config: dict[str, Any], config_path: str | Path) -> dict[str, Any]:
    protocol = config.get("protocol", {})
    return {
        "source_id": protocol.get("source_id", SOURCE_ID),
        "config_path": str(Path(config_path).resolve()),
        "git_info": get_git_info(str(Path(config_path).resolve().parents[2])),
        "methods": list(protocol.get("methods", EXPECTED_METHODS)),
        "opponents": list(protocol.get("opponents", EXPECTED_OPPONENTS)),
        "scenarios": list(protocol.get("scenarios", EXPECTED_SCENARIOS)),
        "train_seeds": list(protocol.get("train_seeds", [42, 43, 44])),
        "evaluation_seeds": list(protocol.get("evaluation_seeds", [42, 43, 44])),
        "episodes_per_scenario": int(protocol.get("episodes_per_scenario", 12)),
        "predictor_contract": PREDICTOR_CONTRACT,
        "existing_bridge_outputs_forbidden": True,
    }


def finite_numeric(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return True
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(np.isfinite(value))
    return True


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    z = 1.959963984540054 if confidence == 0.95 else 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return center - radius, center + radius


def terminal_counts(episodes: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for episode in episodes:
        reason = str(episode.get("reason") or episode.get("termination_reason") or "unknown")
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def aggregate_episodes(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(episodes)
    successes = sum(bool(item.get("is_success", False)) for item in episodes)
    returns = [float(item["return"]) for item in episodes if finite_numeric(item.get("return")) and item.get("return") is not None]
    fallback = []
    for item in episodes:
        # Predicted-geometry fallback is the relevant interface metric for H1/H2.
        # Keep the legacy predictor fallback field as a compatibility fallback.
        value = item.get("predicted_geometry_fallback_rate")
        if value is None:
            value = item.get("prediction_fallback_rate")
        if finite_numeric(value) and value is not None:
            fallback.append(float(value))
    low, high = wilson_interval(successes, total)
    return {
        "episodes": total,
        "successes": successes,
        "success_rate": successes / total if total else float("nan"),
        "success_wilson_low": low,
        "success_wilson_high": high,
        "mean_return": float(np.mean(returns)) if returns else float("nan"),
        "terminal_counts": terminal_counts(episodes),
        "mean_prediction_fallback_rate": float(np.mean(fallback)) if fallback else float("nan"),
        "mean_prediction_error_m": _mean_field(episodes, "mean_prediction_error_m"),
        "mean_final_range_m": _mean_field(episodes, "final_range_m"),
        "mean_final_ata_deg": _mean_field(episodes, "final_ata_deg"),
        "mode_switch_rate": _mean_bool_field(episodes, "mode_switch_effective"),
    }


def _mean_field(episodes: list[dict[str, Any]], field: str) -> float:
    values = [float(item[field]) for item in episodes if field in item and item[field] is not None and finite_numeric(item[field])]
    return float(np.mean(values)) if values else float("nan")


def _mean_bool_field(episodes: list[dict[str, Any]], field: str) -> float:
    return float(np.mean([bool(item.get(field, False)) for item in episodes])) if episodes else float("nan")


def pair_key(episode: dict[str, Any]) -> tuple[Any, ...]:
    return (
        episode.get("opponent", episode.get("opponent_stage", "")),
        episode.get("scenario", ""),
        episode.get("training_seed", ""),
        episode.get("evaluation_seed", episode.get("seed", "")),
        episode.get("episode_seed", episode.get("seed", "")),
    )


def paired_deltas(
    episodes_by_method: dict[str, list[dict[str, Any]]],
    baseline: str = "h0_current_geometry",
) -> list[dict[str, Any]]:
    baseline_rows = {pair_key(row): row for row in episodes_by_method.get(baseline, [])}
    rows: list[dict[str, Any]] = []
    for method, episodes in episodes_by_method.items():
        if method == baseline:
            continue
        for candidate in episodes:
            key = pair_key(candidate)
            reference = baseline_rows.get(key)
            if reference is None:
                continue
            row = {
                "method": method,
                "baseline": baseline,
                "opponent": key[0],
                "scenario": key[1],
                "training_seed": key[2],
                "evaluation_seed": key[3],
                "episode_seed": key[4],
            }
            for field in ("return", "final_range_m", "final_ata_deg", "mean_prediction_error_m"):
                left = candidate.get(field)
                right = reference.get(field)
                if finite_numeric(left) and finite_numeric(right) and left is not None and right is not None:
                    row[f"delta_{field}"] = float(left) - float(right)
                else:
                    row[f"delta_{field}"] = None
            row["success_delta"] = int(bool(candidate.get("is_success", False))) - int(bool(reference.get("is_success", False)))
            rows.append(row)
    return rows
