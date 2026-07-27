"""Prepare isolated, non-equivalent retraining configurations without training.

This script never launches a trainer or changes any historical registry. It
creates resolved, provenance-bearing configs for the new Source ID and records
whether the new trajectory CSV corpus is available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

from uav_vpp_guidance.common.hash import config_sha256, file_info
from uav_vpp_guidance.common.provenance import record_config_override
from uav_vpp_guidance.training.train_prediction_vpp_ppo import load_experiment_config
from uav_vpp_guidance.utils.config import load_yaml_config

SOURCE_ID = "UAV-VPP-GUIDANCE-RETRAIN-NON-EQUIVALENT-20260727-R01"
DEFAULT_ROOT = Path("outputs/control_guidance_retraining") / SOURCE_ID


def set_override(config: Dict[str, Any], section: str, key: str, value: Any, source: str) -> None:
    target = config.setdefault(section, {})
    old_value = target.get(key)
    target[key] = value
    if old_value != value:
        record_config_override(config, f"{section}.{key}", value, old_value, source)


def prepare_config(base_path: Path, lane: str, output_root: Path, device: str) -> Dict[str, Any]:
    loader = load_yaml_config if lane == "trajectory_predictor_lstm_s0" else load_experiment_config
    config = loader(str(base_path))
    set_override(config, "experiment", "source_id", SOURCE_ID, "prepare_non_equivalent_retraining.py:new_source")
    set_override(config, "experiment", "name", lane, "prepare_non_equivalent_retraining.py:new_source")
    set_override(config, "experiment", "output_root", str(output_root), "prepare_non_equivalent_retraining.py:isolation")
    if lane == "trajectory_predictor_lstm_s0":
        set_override(config, "training", "output_dir", str(output_root / lane), "prepare_non_equivalent_retraining.py:isolation")
        set_override(config, "training", "device", device, "prepare_non_equivalent_retraining.py:device")
    else:
        set_override(config, "ppo", "device", device, "prepare_non_equivalent_retraining.py:device")
    if lane == "prediction_vpp_lstm_s0":
        set_override(config, "trajectory_prediction", "checkpoint_path", str(output_root / "trajectory_predictor_lstm_s0" / "best_model.pt"), "prepare_non_equivalent_retraining.py:new_predictor")
        set_override(config, "warm_start", "enabled", False, "prepare_non_equivalent_retraining.py:no_historical_warm_start")
        set_override(config, "warm_start", "checkpoint", None, "prepare_non_equivalent_retraining.py:no_historical_warm_start")
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare an isolated non-equivalent retraining source.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_ROOT / "input" / "trajectory_csv")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--preflight-id", default="initial", help="Unique, non-empty preflight attempt identifier.")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu or provision CUDA.")
    if not args.preflight_id.strip() or any(char in args.preflight_id for char in "\\\\/:"):
        raise ValueError("preflight ID must be a non-empty single path component")
    output_root = args.output_root.resolve()
    forbidden = (Path("outputs/trajectory_prediction").resolve(), Path("outputs/experiments").resolve())
    if any(output_root == path or path in output_root.parents for path in forbidden):
        raise ValueError("output root must not be a historical output root")

    base_configs = {
        "trajectory_predictor_lstm_s0": Path("config/trajectory_prediction.yaml"),
        "prediction_vpp_lstm_s0": Path("config/experiment/train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00.yaml"),
        "no_prediction_vpp_s0": Path("config/experiment/train_no_prediction_vpp_ppo_jsbsim_compare.yaml"),
        "end_to_end_s0": Path("config/experiment/train_end_to_end_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_combat.yaml"),
    }
    preflight_dir = output_root / "preflight" / args.preflight_id
    resolved_dir = preflight_dir / "resolved_configs"
    resolved_dir.mkdir(parents=True, exist_ok=False)
    prepared = {}
    for lane, base_path in base_configs.items():
        config = prepare_config(base_path, lane, output_root, args.device)
        path = resolved_dir / f"{lane}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
        prepared[lane] = {"base_config": str(base_path), "resolved_config": str(path), "config_sha256": config_sha256(config)}

    csv_files = sorted(args.data_dir.rglob("*.csv")) if args.data_dir.exists() else []
    preflight = {
        "schema_version": "1.0.0",
        "source_id": SOURCE_ID,
        "status": "ready_for_training" if csv_files else "blocked_missing_new_trajectory_corpus",
        "output_root": str(output_root),
        "preflight_id": args.preflight_id,
        "data_dir": str(args.data_dir),
        "trajectory_csv_files": [file_info(path) for path in csv_files],
        "device": args.device,
        "cuda_available": torch.cuda.is_available(),
        "prepared_lanes": prepared,
        "historical_asset_equivalence": False,
        "historical_asset_replacement_permitted": False,
    }
    preflight_path = preflight_dir / "source_preflight.json"
    preflight_path.write_text(json.dumps(preflight, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": preflight["status"], "preflight": str(preflight_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
