#!/usr/bin/env python3
"""
Thesis blind-review revision: Group H (history-feature control group) multi-seed training.

Group H is the clean control for the thesis Chapter 5 claim that the B group
(VPP + prediction-augmented observation) improves over A0 (16-D base):

    A0: 16-D base observation
    B : 16-D base + future prediction features (+14, or predicted-geometry replacement)
    H : 16-D base + 3-D target history-motion features (NO future prediction)

If H < B  -> the thesis can keep an extrapolation claim for future prediction.
If H ~= B -> the claim must be narrowed to "history-motion-enhanced observation".
If H > B  -> pivot to a feature-design discussion.

This script ONLY prepares the configuration and launches training; it does not
run evaluation. All config mutations are recorded via
``record_config_override_if_changed`` so they appear in
``provenance["config_overrides"]`` (workspace AGENTS.md contract, section 3).

Usage:
    # Preparation only: build config, construct env, print the real obs dim,
    # then stop before any training (--dry-run is the default-safe mode? No:
    # default runs training). Use --print-config to only verify.
    python scripts/run_thesis_h_group_multiseed.py --print-config

    # Full multi-seed training (3 seeds by default)
    python scripts/run_thesis_h_group_multiseed.py

    # Smoke
    python scripts/run_thesis_h_group_multiseed.py --smoke
"""

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uav_vpp_guidance.common.provenance import record_config_override_if_changed
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.training.train_prediction_vpp_ppo import (
    load_experiment_config,
    train_ppo,
)

SOURCE = "scripts/run_thesis_h_group_multiseed.py:GroupH"


def _get_dotted(config: dict, dotted_key: str):
    node = config
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _set_dotted(config: dict, dotted_key: str, value) -> None:
    parts = dotted_key.split(".")
    node = config
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _override(config: dict, dotted_key: str, new_value, source: str = SOURCE) -> None:
    """Record-and-set helper (AGENTS.md section 3 contract)."""
    old_value = _get_dotted(config, dotted_key)
    record_config_override_if_changed(
        config,
        key=dotted_key,
        new_value=new_value,
        old_value=old_value,
        source=source,
    )
    _set_dotted(config, dotted_key, new_value)


def build_h_group_config(base_config: dict) -> dict:
    """Derive the Group H config from the B-group (prediction) config.

    Every mutation is recorded so the run manifest / provenance shows exactly
    how H differs from B.
    """
    config = copy.deepcopy(base_config)

    # 1) Remove ALL future-prediction integration from the policy observation.
    #    (The predictor checkpoint stays configured but is never queried; the
    #    env hard-disables the adapter when enabled=False.)
    _override(config, "trajectory_prediction.enabled", False)
    _override(config, "trajectory_prediction.integration.add_prediction_to_observation", False)
    _override(config, "trajectory_prediction.integration.add_uncertainty_to_observation", False)
    _override(config, "trajectory_prediction.integration.replace_base_geometry", False)

    # 2) Enable the history-kinematic extension. Window length follows the
    #    predictor history window so H and B see the same amount of history.
    history_len = int(
        _get_dotted(config, "trajectory_prediction.history.history_len") or 10
    )
    _override(config, "observation.include_history_kinematic", True)
    _override(config, "observation.history_kinematic.window_len", history_len)

    # 3) Experiment identity: keep the lineage explicit.
    base_name = _get_dotted(config, "experiment.name") or "experiment"
    _override(config, "experiment.name", f"{base_name}_h_group")

    return config


def sanity_check_observation_dim(config: dict, expected_dim: int = 19) -> int:
    """Construct one env, reset, and report the real observation dim.

    Group H must be exactly base(16) + history_kinematic(3) = 19 when all
    other observation extensions are off (same switch state as the A0/B runs).
    """
    env = CloseRangeTrackingEnv(copy.deepcopy(config))
    obs = env.reset(seed=0)
    dim = int(obs["observation_vector"].shape[0])
    schema = obs["observation_schema"]
    print(f"[H-group] backend={env._backend} observation dim={dim}")
    print(f"[H-group] schema include_history_kinematic={schema.get('include_history_kinematic')}")
    print(f"[H-group] history feature names={[n for n in schema['feature_names'] if n.startswith('hist_')]}")
    if dim != expected_dim:
        raise RuntimeError(
            f"Group H observation dim {dim} != expected {expected_dim}. "
            "Refusing to train: the control group must differ from A0 only by "
            "the 3 history-kinematic dims."
        )
    return dim


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config",
        default="config/experiment/train_vpp_ppo_lstm_frozen.yaml",
        help="B-group (prediction-augmented) training config to derive H from.",
    )
    parser.add_argument("--seeds", default="42,43,44", help="Comma-separated training seeds.")
    parser.add_argument(
        "--output-root",
        default="outputs/experiments/thesis_h_group",
        help="Output root; one subdirectory per seed is created.",
    )
    parser.add_argument("--smoke", action="store_true", help="Minimal smoke training.")
    parser.add_argument(
        "--device",
        default=None,
        help="Override ppo.device (e.g. cuda on the GPU workstation).",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Only build the H config, run the obs-dim sanity check, and exit.",
    )
    args = parser.parse_args()

    base_config = load_experiment_config(args.base_config)
    config = build_h_group_config(base_config)

    sanity_check_observation_dim(config)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if args.print_config:
        print("[H-group] --print-config set; config built and sanity check passed.")
        print(f"[H-group] seeds={seeds} output_root={args.output_root}")
        return

    for seed in seeds:
        seed_config = copy.deepcopy(config)
        if _get_dotted(seed_config, "experiment.seed") is not None:
            _override(seed_config, "experiment.seed", seed, source=f"{SOURCE}:seed")
        if _get_dotted(seed_config, "ppo.seed") is not None:
            _override(seed_config, "ppo.seed", seed, source=f"{SOURCE}:seed")
        if args.device:
            _override(seed_config, "ppo.device", args.device, source=f"{SOURCE}:device")
        output_dir = str(Path(args.output_root) / f"seed_{seed}")
        print(f"[H-group] === seed {seed} -> {output_dir} ===")
        train_ppo(seed_config, output_dir, smoke=args.smoke)

    print("[H-group] training done. Next step: closed-loop evaluation of each")
    print("[H-group] seed checkpoint on the SAME evaluation scenario list as A0/B")
    print("[H-group] (paired statistics; do not merge across seeds).")
    print("[H-group] interpretation: H<B keeps the prediction claim; H~=B narrows")
    print("[H-group] the claim to 'history-motion-enhanced observation'; H>B pivots")


if __name__ == "__main__":
    main()
