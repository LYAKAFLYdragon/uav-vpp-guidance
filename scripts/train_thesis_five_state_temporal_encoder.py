"""Collect train-only JSBSim geometry windows and pretrain the P3 encoder.

This is deliberately not a policy-training entry point.  It uses a frozen
rule-based VPP behavior policy only to induce trajectories, retains compact
16-D geometry sequences for the train split, and evaluates dev30 in memory.
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
import hashlib
import importlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SCRIPT_DIR, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_thesis_taxonomy_ablation30_manifest import (  # noqa: E402
    _heading_vector,
    _horizontal_heading_for_3d_angle,
    _normalize_heading_deg,
    _vector_angle_deg,
)
from uav_vpp_guidance.envs.observation import compute_relative_geometry  # noqa: E402
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv  # noqa: E402
from uav_vpp_guidance.expert_system.expert_vpp_policy import ExpertVPPPolicy  # noqa: E402
from uav_vpp_guidance.training.thesis_temporal_geometry_encoder import (  # noqa: E402
    TemporalEncoderBatch,
    TemporalGeometryEncoder,
    deterministic_mask,
    self_supervised_loss,
)


DEFAULT_CONFIG = REPO_ROOT / "config" / "experiment" / "train_thesis_five_state_temporal_encoder_v1.yaml"
BASE_ENV_CONFIG = REPO_ROOT / "config" / "experiment" / "jsbsim_hrl_comparison.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _resolve_repo_path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _import_class(class_path: str):
    module_name, class_name = str(class_path).replace("src.", "", 1).rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


def _build_opponent(entry: Mapping[str, Any]):
    cls = _import_class(str(entry["class"]))
    kwargs: dict[str, Any] = {}
    if entry.get("checkpoint"):
        kwargs["checkpoint_path"] = str(entry["checkpoint"])
    if "config" in entry:
        kwargs["config"] = copy.deepcopy(entry["config"])
    if "device" in entry:
        kwargs["device"] = str(entry["device"])
    if "invert_observation" in entry:
        kwargs["invert_observation"] = bool(entry["invert_observation"])
    return cls(**kwargs)


def _scenario_geometry(
    own: Mapping[str, Any], target: Mapping[str, Any]
) -> tuple[float, float]:
    own_pos = tuple(float(value) for value in own["position_m"])
    target_pos = tuple(float(value) for value in target["position_m"])
    los = tuple(target_pos[index] - own_pos[index] for index in range(3))
    reverse_los = tuple(-value for value in los)
    own_velocity = _heading_vector(float(own["heading_deg"]), float(own["velocity_mps"]))
    target_velocity = _heading_vector(
        float(target["heading_deg"]), float(target["velocity_mps"])
    )
    return _vector_angle_deg(own_velocity, los), _vector_angle_deg(target_velocity, reverse_los)


def sample_train_scenarios(distribution: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Sample one deterministic continuous geometry per state-height-mirror cell."""

    contract = distribution["sampling_contract"]
    rng = np.random.default_rng(int(contract["seed"]))
    support = contract["continuous_support"]
    scenarios: list[dict[str, Any]] = []
    for initial_class, initial_spec in contract["initial_states"].items():
        for height_condition, height_spec in contract["height_conditions"].items():
            for mirror_sign in contract["mirror_signs"]:
                sign = -1 if mirror_sign == "negative" else 1
                # Sample only physically realizable 3-D LOS/heading combinations.
                # Invalid draws are rejected rather than clipped, preserving the
                # declared continuous ranges conditional on geometric feasibility.
                for _attempt in range(100):
                    initial_range = float(rng.uniform(*support["initial_range_m"]))
                    own_speed = float(rng.uniform(*support["own_speed_mps"]))
                    target_speed = float(rng.uniform(*support["target_speed_mps"]))
                    own_altitude = float(rng.uniform(*support["own_altitude_m"]))
                    altitude_diff = float(rng.uniform(*height_spec["target_minus_own_altitude_m"]))
                    los_azimuth = sign * float(rng.uniform(*support["los_azimuth_abs_deg"]))
                    horizontal_range = math.sqrt(initial_range**2 - altitude_diff**2)
                    horizontal_fraction = horizontal_range / initial_range
                    own_angle = float(rng.uniform(*initial_spec["own_to_target_los_angle_deg"]))
                    target_angle = float(
                        rng.uniform(*initial_spec["target_velocity_to_own_los_angle_deg"])
                    )
                    if (
                        abs(math.cos(math.radians(own_angle))) <= horizontal_fraction
                        and abs(math.cos(math.radians(target_angle))) <= horizontal_fraction
                    ):
                        break
                else:
                    raise RuntimeError(
                        "Unable to sample a feasible 3-D geometry after 100 attempts; "
                        f"state={initial_class}, height={height_condition}"
                    )
                own_heading = _horizontal_heading_for_3d_angle(
                    los_azimuth, horizontal_fraction, own_angle, -sign
                )
                target_heading = _horizontal_heading_for_3d_angle(
                    _normalize_heading_deg(los_azimuth + 180.0),
                    horizontal_fraction,
                    target_angle,
                    sign,
                )
                azimuth_rad = math.radians(los_azimuth)
                scenario = {
                    "name": f"p3_train_{initial_class}_{height_condition}_{mirror_sign}",
                    "own_init": {
                        "position_m": [0.0, 0.0, own_altitude],
                        "velocity_mps": own_speed,
                        "heading_deg": round(own_heading, 9),
                    },
                    "target_init": {
                        "position_m": [
                            round(horizontal_range * math.cos(azimuth_rad), 9),
                            round(horizontal_range * math.sin(azimuth_rad), 9),
                            round(own_altitude + altitude_diff, 9),
                        ],
                        "velocity_mps": target_speed,
                        "heading_deg": round(target_heading, 9),
                    },
                    "metadata": {
                        "split": "train",
                        "initial_class": initial_class,
                        "height_condition": height_condition,
                        "mirror_sign": mirror_sign,
                        "initial_range_m": initial_range,
                        "own_speed_mps": own_speed,
                        "target_speed_mps": target_speed,
                    },
                }
                geometry = _scenario_geometry(scenario["own_init"], scenario["target_init"])
                scenario["metadata"].update(
                    {
                        "own_to_target_los_angle_deg": geometry[0],
                        "target_velocity_to_own_los_angle_deg": geometry[1],
                    }
                )
                scenarios.append(scenario)
    return scenarios


class PhaseTracker:
    """Label observed pre-merge, post-merge, and genuine re-entry windows."""

    def __init__(self, merge_range_m: float, reentry_negative_rate_mps: float, min_post_merge_steps: int):
        self.merge_range_m = float(merge_range_m)
        self.reentry_negative_rate_mps = float(reentry_negative_rate_mps)
        self.min_post_merge_steps = int(min_post_merge_steps)
        self._seen_merge = False
        self._post_merge_steps = 0

    def update(self, range_m: float, range_rate_mps: float) -> str:
        if not self._seen_merge and float(range_m) <= self.merge_range_m:
            self._seen_merge = True
        if not self._seen_merge:
            return "pre_merge"
        self._post_merge_steps += 1
        if (
            self._post_merge_steps >= self.min_post_merge_steps
            and float(range_rate_mps) <= self.reentry_negative_rate_mps
        ):
            return "re_entry"
        return "post_merge"


def _environment_config(config: Mapping[str, Any], opponent_id: str) -> dict[str, Any]:
    base = copy.deepcopy(_load_yaml(BASE_ENV_CONFIG))
    simulator = config["simulator"]
    base["backend"] = "jsbsim"
    base.setdefault("env", {}).update(
        {
            "use_jsbsim": True,
            "strict_backend": bool(simulator["strict_backend"]),
            "legacy_project_root": str(simulator["legacy_project_root"]),
            "aircraft_model": str(simulator["aircraft_model"]),
            "high_level_dt": float(simulator["high_level_dt_s"]),
            "max_high_level_steps": int(simulator["max_high_level_steps"]),
            "success_range_m": 0.0,
            "success_ata_deg": 0.0,
            "success_hold_time_s": 1.0,
            "max_range_m": 12000.0,
        }
    )
    base["task"] = {"name": "thesis_five_state_p3_collection", "env_class": "CloseRangeTrackingEnv"}
    base["observation"] = {
        "include_gains": False,
        "include_guidance_state": False,
        "include_saturation": False,
        "include_opponent_stage": False,
        "include_task_type": False,
    }
    base.setdefault("attack_zone", {})["enabled"] = False
    base["opponent_stage"] = opponent_id
    return base


def collect_rollout(
    env: CloseRangeTrackingEnv,
    scenario: Mapping[str, Any],
    *,
    episode_id: str,
    opponent_id: str,
    phase_spec: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Collect compact base-geometry observations from one fully JSBSim rollout."""

    observation = env.reset(scenario=copy.deepcopy(dict(scenario)), seed=seed)
    if env._backend != "jsbsim":
        raise RuntimeError("P3 requires JSBSim; backend fallback is not permitted")
    behavior_policy = ExpertVPPPolicy({})
    tracker = PhaseTracker(
        phase_spec["merge_range_m"],
        phase_spec["reentry_negative_range_rate_mps"],
        phase_spec["reentry_min_post_merge_steps"],
    )
    features: list[np.ndarray] = []
    phases: list[str] = []
    termination = "timeout"
    for _ in range(env.max_steps):
        vector = np.asarray(observation["observation_vector"], dtype=np.float32)
        if vector.shape != (16,) or not np.isfinite(vector).all():
            raise ValueError(f"Invalid base geometry observation for {episode_id}")
        relative = observation["relative_state"]
        phase = tracker.update(relative["range_m"], relative["range_rate_mps"])
        features.append(vector)
        phases.append(phase)
        action = behavior_policy.get_action(
            observation["own_state"], observation["target_state"], relative
        )
        observation, _reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            termination = str(info.get("reason") or "terminated")
            vector = np.asarray(observation["observation_vector"], dtype=np.float32)
            if vector.shape == (16,) and np.isfinite(vector).all():
                relative = observation["relative_state"]
                features.append(vector)
                phases.append(tracker.update(relative["range_m"], relative["range_rate_mps"]))
            break
    return {
        "episode_id": episode_id,
        "features": np.asarray(features, dtype=np.float32),
        "phases": phases,
        "initial_class": str(scenario["metadata"]["initial_class"]),
        "height_condition": str(scenario["metadata"]["height_condition"]),
        "mirror_sign": str(scenario["metadata"]["mirror_sign"]),
        "opponent_id": opponent_id,
        "termination": termination,
        "backend": env._backend,
    }


def collect_split(
    config: Mapping[str, Any],
    scenarios: Sequence[Mapping[str, Any]],
    registry: Mapping[str, Any],
    *,
    split: str,
) -> list[dict[str, Any]]:
    phase_spec = config["collector"]["phase_definition"]
    seed_base = int(config["dataset"]["split_random_seed"])
    episodes: list[dict[str, Any]] = []
    for opponent_index, opponent_id in enumerate(config["collector"]["required_opponents"]):
        entry = registry["opponents"][opponent_id]
        env = CloseRangeTrackingEnv(
            _environment_config(config, opponent_id),
            opponent_policy=_build_opponent(entry),
            opponent_config={"stage": opponent_id, **copy.deepcopy(entry)},
        )
        for scenario_index, scenario in enumerate(scenarios):
            episode_id = f"{split}__{opponent_id}__{scenario['name']}"
            episodes.append(
                collect_rollout(
                    env,
                    scenario,
                    episode_id=episode_id,
                    opponent_id=opponent_id,
                    phase_spec=phase_spec,
                    seed=seed_base + opponent_index * 10000 + scenario_index,
                )
            )
    return episodes


def fit_normalization(episodes: Sequence[Mapping[str, Any]]) -> dict[str, list[float]]:
    values = np.concatenate([np.asarray(item["features"], dtype=np.float32) for item in episodes], axis=0)
    mean = values.mean(axis=0, dtype=np.float64)
    std = values.std(axis=0, dtype=np.float64)
    std = np.maximum(std, 1e-6)
    return {"mean": mean.astype(np.float32).tolist(), "std": std.astype(np.float32).tolist()}


def _normalize(values: np.ndarray, statistics: Mapping[str, Sequence[float]]) -> np.ndarray:
    mean = np.asarray(statistics["mean"], dtype=np.float32)
    std = np.asarray(statistics["std"], dtype=np.float32)
    normalized = (np.asarray(values, dtype=np.float32) - mean) / std
    if not np.isfinite(normalized).all():
        raise ValueError("Normalization produced a non-finite feature")
    return normalized


def build_samples(
    episodes: Sequence[Mapping[str, Any]],
    statistics: Mapping[str, Sequence[float]],
    *,
    history_steps: int,
    horizons: Sequence[int],
) -> dict[str, Any]:
    horizon_max = max(int(value) for value in horizons)
    histories: list[np.ndarray] = []
    targets: dict[int, list[np.ndarray]] = {int(horizon): [] for horizon in horizons}
    metadata: list[dict[str, str]] = []
    for episode in episodes:
        features = _normalize(np.asarray(episode["features"]), statistics)
        phases = list(episode["phases"])
        for end_index in range(history_steps - 1, len(features) - horizon_max):
            histories.append(features[end_index - history_steps + 1 : end_index + 1])
            for horizon in horizons:
                targets[int(horizon)].append(features[end_index + int(horizon)] - features[end_index])
            metadata.append(
                {
                    "initial_class": str(episode["initial_class"]),
                    "phase": str(phases[end_index]),
                    "opponent_id": str(episode["opponent_id"]),
                }
            )
    if not histories:
        raise ValueError("No valid P3 samples after history and horizon construction")
    return {
        "history": np.stack(histories).astype(np.float32),
        "future_deltas": {key: np.stack(value).astype(np.float32) for key, value in targets.items()},
        "metadata": metadata,
    }


def coverage_report(
    samples: Mapping[str, Any], required_states: Iterable[str], required_phases: Iterable[str], required_opponents: Iterable[str]
) -> dict[str, Any]:
    counts = Counter(
        (item["initial_class"], item["phase"], item["opponent_id"])
        for item in samples["metadata"]
    )
    rows = []
    missing = []
    for initial_class in required_states:
        for phase in required_phases:
            for opponent_id in required_opponents:
                count = int(counts[(initial_class, phase, opponent_id)])
                row = {
                    "initial_class": initial_class,
                    "phase": phase,
                    "opponent_id": opponent_id,
                    "sample_count": count,
                }
                rows.append(row)
                if count == 0:
                    missing.append(row)
    return {"rows": rows, "missing": missing, "all_cells_covered": not missing}


def balance_samples(samples: Mapping[str, Any], *, minimum_per_cell: int, coverage: Mapping[str, Any]) -> dict[str, Any]:
    if not coverage["all_cells_covered"]:
        raise ValueError("Cannot balance samples with uncovered state/phase/opponent cells")
    by_cell: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, item in enumerate(samples["metadata"]):
        by_cell[(item["initial_class"], item["phase"], item["opponent_id"])].append(index)
    selected_count = min(len(indices) for indices in by_cell.values())
    if selected_count < int(minimum_per_cell):
        raise ValueError(
            f"Insufficient balanced samples per cell: {selected_count} < {minimum_per_cell}"
        )
    selected = [index for cell in sorted(by_cell) for index in by_cell[cell][:selected_count]]
    return {
        "history": samples["history"][selected],
        "future_deltas": {horizon: values[selected] for horizon, values in samples["future_deltas"].items()},
        "metadata": [samples["metadata"][index] for index in selected],
        "samples_per_cell": selected_count,
    }


def _tensor_loader(samples: Mapping[str, Any], batch_size: int, shuffle: bool) -> DataLoader:
    horizons = sorted(samples["future_deltas"])
    tensors = [torch.from_numpy(samples["history"])]
    tensors.extend(torch.from_numpy(samples["future_deltas"][horizon]) for horizon in horizons)
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)


def _batch_from_tensors(tensors: Sequence[torch.Tensor], horizons: Sequence[int], device: torch.device) -> TemporalEncoderBatch:
    return TemporalEncoderBatch(
        history=tensors[0].to(device),
        future_deltas={int(horizon): tensors[index + 1].to(device) for index, horizon in enumerate(horizons)},
    )


def _evaluate(
    model: TemporalGeometryEncoder,
    samples: Mapping[str, Any],
    *,
    batch_size: int,
    mask_ratio: float,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    loader = _tensor_loader(samples, batch_size=batch_size, shuffle=False)
    horizons = sorted(samples["future_deltas"])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    totals: list[dict[str, float]] = []
    per_sample: list[dict[str, float]] = []
    with torch.no_grad():
        offset = 0
        for tensors in loader:
            batch = _batch_from_tensors(tensors, horizons, device)
            mask = deterministic_mask(batch.history, mask_ratio, generator)
            output = model(batch.history, mask)
            losses = self_supervised_loss(output, batch, mask)
            totals.append({name: float(value.item()) for name, value in losses.items()})
            recon = ((output["reconstruction"] - batch.history) ** 2 * mask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp_min(1)
            prediction = torch.stack(
                [
                    ((output["future_deltas"][horizon] - batch.future_deltas[horizon]) ** 2).mean(dim=1)
                    for horizon in horizons
                ],
                dim=1,
            ).mean(dim=1)
            for index in range(batch.history.shape[0]):
                per_sample.append(
                    {
                        "reconstruction": float(recon[index].item()),
                        "prediction": float(prediction[index].item()),
                        "total": float((recon[index] + prediction[index]).item()),
                        "sample_index": offset + index,
                    }
                )
            offset += batch.history.shape[0]
    mean = {name: float(np.mean([entry[name] for entry in totals])) for name in totals[0]}
    return mean, per_sample


def _group_metrics(per_sample: Sequence[Mapping[str, float]], metadata: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, float]]] = defaultdict(list)
    for value, meta in zip(per_sample, metadata):
        grouped[(meta["initial_class"], meta["phase"], meta["opponent_id"])].append(value)
    return [
        {
            "initial_class": key[0],
            "phase": key[1],
            "opponent_id": key[2],
            "sample_count": len(values),
            "mean_total_loss": float(np.mean([value["total"] for value in values])),
            "mean_reconstruction_loss": float(np.mean([value["reconstruction"] for value in values])),
            "mean_prediction_loss": float(np.mean([value["prediction"] for value in values])),
        }
        for key, values in sorted(grouped.items())
    ]


def _save_train_dataset(output_root: Path, episodes: Sequence[Mapping[str, Any]], samples: Mapping[str, Any]) -> None:
    dataset_dir = output_root / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    compact = {f"episode_{index:03d}": np.asarray(item["features"], dtype=np.float32) for index, item in enumerate(episodes)}
    np.savez_compressed(dataset_dir / "train_compact_geometry_sequences.npz", **compact)
    np.savez_compressed(
        dataset_dir / "train_self_supervised_samples.npz",
        history=samples["history"],
        **{f"future_delta_h{horizon}": values for horizon, values in samples["future_deltas"].items()},
    )
    with (dataset_dir / "train_episode_index.jsonl").open("w", encoding="utf-8") as handle:
        for item in episodes:
            row = {
                key: item[key]
                for key in (
                    "episode_id", "initial_class", "height_condition", "mirror_sign", "opponent_id", "termination", "backend"
                )
            }
            row["frame_count"] = int(len(item["features"]))
            row["feature_sha256"] = hashlib.sha256(item["features"].tobytes()).hexdigest()
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(config: Mapping[str, Any], *, output_root: Path, dry_run: bool = False) -> int:
    sources = config["sources"]
    distribution = _load_yaml(_resolve_repo_path(sources["train_distribution"]))
    dev_manifest = _load_yaml(_resolve_repo_path(sources["dev30_manifest"]))
    heldout_manifest = _load_yaml(_resolve_repo_path(sources["heldout60_manifest"]))
    registry = _load_yaml(_resolve_repo_path(sources["opponent_registry"]))
    provenance = _load_yaml(_resolve_repo_path(sources["trajectory_provenance"]))
    if provenance["normalization_contract"]["fit_split"] != "train_only":
        raise ValueError("P3 requires train-only normalization")
    if provenance["provenance_sources"]["dev30_payload_sha256"] != dev_manifest["integrity"]["payload_sha256"]:
        raise ValueError("dev30 provenance hash mismatch")
    if provenance["provenance_sources"]["heldout60_payload_sha256"] != heldout_manifest["integrity"]["payload_sha256"]:
        raise ValueError("heldout60 provenance hash mismatch")
    planned_train = len(sample_train_scenarios(distribution)) * len(config["collector"]["required_opponents"])
    planned_dev = len(dev_manifest["scenarios"]) * len(config["collector"]["required_opponents"])
    if dry_run:
        print(json.dumps({"planned_train_rollouts": planned_train, "planned_dev_rollouts": planned_dev, "heldout60_used": False}, indent=2))
        return 0
    if output_root.exists():
        raise FileExistsError(f"P3 output root already exists: {output_root}")
    output_root.mkdir(parents=True)
    _write_json(output_root / "resolved_config.json", dict(config))
    _write_json(
        output_root / "input_hashes.json",
        {
            "config_sha256": _sha256_json(config),
            "distribution_sha256": _sha256_file(_resolve_repo_path(sources["train_distribution"])),
            "dev30_sha256": _sha256_file(_resolve_repo_path(sources["dev30_manifest"])),
            "heldout60_sha256": _sha256_file(_resolve_repo_path(sources["heldout60_manifest"])),
            "registry_sha256": _sha256_file(_resolve_repo_path(sources["opponent_registry"])),
            "provenance_sha256": _sha256_file(_resolve_repo_path(sources["trajectory_provenance"])),
        },
    )
    train_episodes = collect_split(config, sample_train_scenarios(distribution), registry, split="train")
    statistics = fit_normalization(train_episodes)
    _write_json(output_root / "dataset" / "normalization_train_only.json", statistics)
    train_samples = build_samples(
        train_episodes,
        statistics,
        history_steps=int(config["dataset"]["history_steps"]),
        horizons=config["dataset"]["prediction_horizons_steps"],
    )
    coverage = coverage_report(
        train_samples,
        config["collector"]["required_initial_states"],
        config["collector"]["required_phases"],
        config["collector"]["required_opponents"],
    )
    collection_report = {
        "planned_train_rollouts": planned_train,
        "completed_train_rollouts": len(train_episodes),
        "backend_counts": dict(Counter(item["backend"] for item in train_episodes)),
        "termination_counts": dict(Counter(item["termination"] for item in train_episodes)),
        "train_sample_count_before_balance": len(train_samples["metadata"]),
        "coverage": coverage,
        "heldout60_used": False,
    }
    _write_json(output_root / "collection_report.json", collection_report)
    _save_train_dataset(output_root, train_episodes, train_samples)
    if not coverage["all_cells_covered"]:
        _write_json(
            output_root / "p3_gate_decision.json",
            {"status": "failed_missing_phase_coverage", "stop_rule": config["stop_rule"], "coverage": coverage},
        )
        return 2
    balanced = balance_samples(
        train_samples,
        minimum_per_cell=int(config["collector"]["minimum_windows_per_state_phase_opponent"]),
        coverage=coverage,
    )
    dev_episodes = collect_split(config, dev_manifest["scenarios"], registry, split="dev30")
    dev_samples = build_samples(
        dev_episodes,
        statistics,
        history_steps=int(config["dataset"]["history_steps"]),
        horizons=config["dataset"]["prediction_horizons_steps"],
    )
    train_cfg = config["training"]
    device = torch.device(str(train_cfg["device"]))
    torch.manual_seed(int(train_cfg["seed"]))
    np.random.seed(int(train_cfg["seed"]))
    random.seed(int(train_cfg["seed"]))
    model_cfg = config["model"]
    model = TemporalGeometryEncoder(
        input_dim=int(model_cfg["input_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        embedding_dim=int(model_cfg["embedding_dim"]),
        horizons=config["dataset"]["prediction_horizons_steps"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["learning_rate"]), weight_decay=float(train_cfg["weight_decay"]))
    horizons = sorted(balanced["future_deltas"])
    loader = _tensor_loader(balanced, batch_size=int(train_cfg["batch_size"]), shuffle=True)
    mask_ratio = float(config["objective"]["mask_ratio"])
    best_loss = float("inf")
    history: list[dict[str, float]] = []
    checkpoint_dir = output_root / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        generator = torch.Generator(device="cpu").manual_seed(int(train_cfg["seed"]) + epoch)
        epoch_losses = []
        for tensors in loader:
            batch = _batch_from_tensors(tensors, horizons, device)
            mask = deterministic_mask(batch.history, mask_ratio, generator)
            output = model(batch.history, mask)
            losses = self_supervised_loss(output, batch, mask)
            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(losses["total"].item()))
        dev_metrics, _ = _evaluate(
            model,
            dev_samples,
            batch_size=int(train_cfg["batch_size"]),
            mask_ratio=mask_ratio,
            seed=int(train_cfg["seed"]),
            device=device,
        )
        record = {"epoch": epoch, "train_total_loss": float(np.mean(epoch_losses)), **{f"dev_{key}_loss": value for key, value in dev_metrics.items()}}
        history.append(record)
        if dev_metrics["total"] < best_loss:
            best_loss = dev_metrics["total"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": dict(model_cfg),
                    "dataset": dict(config["dataset"]),
                    "normalization": statistics,
                    "frozen_for_high_level_ppo": True,
                    "best_dev_total_loss": best_loss,
                },
                checkpoint_dir / "best.pt",
            )
    torch.save({"model_state_dict": model.state_dict(), "model_config": dict(model_cfg), "normalization": statistics}, checkpoint_dir / "last.pt")
    dev_metrics, dev_per_sample = _evaluate(
        model,
        dev_samples,
        batch_size=int(train_cfg["batch_size"]),
        mask_ratio=mask_ratio,
        seed=int(train_cfg["seed"]),
        device=device,
    )
    embedding = model.encode_history(dev_samples["history"][: min(64, len(dev_samples["history"]))])
    if not np.isfinite(embedding).all():
        raise RuntimeError("P3 encoder emitted non-finite dev embeddings")
    _write_json(
        output_root / "training_report.json",
        {
            "status": "passed",
            "balanced_train_sample_count": len(balanced["metadata"]),
            "balanced_samples_per_cell": balanced["samples_per_cell"],
            "dev_sample_count": len(dev_samples["metadata"]),
            "dev_metrics": dev_metrics,
            "dev_group_metrics": _group_metrics(dev_per_sample, dev_samples["metadata"]),
            "epochs": history,
            "heldout60_used": False,
            "encoder_embedding_shape": list(embedding.shape),
            "encoder_frozen_for_high_level_ppo": True,
        },
    )
    _write_json(output_root / "p3_gate_decision.json", {"status": "passed", "stop_rule": config["stop_rule"]})
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = _load_yaml(args.config)
    output_root = args.output_root or Path(config["training"]["output_root"])
    return run(config, output_root=output_root, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
