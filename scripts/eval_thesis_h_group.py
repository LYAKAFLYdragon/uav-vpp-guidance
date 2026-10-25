#!/usr/bin/env python3
"""
Thesis Group H (history-feature control) closed-loop evaluation.

Evaluates each H-group training seed's best checkpoint on the SAME protocol as
the A0/B comparison (scenarios x opponents x evaluation seeds), so results are
paired. The env is built with the H-group config, so the policy observation is
19-D (16 base + 3 history-kinematic); loading a checkpoint with any other obs
dimension fails loudly here.

Outputs per (training_seed, opponent):
    <output_root>/eval/seed_<N>/<opponent>/metrics.json
    <output_root>/eval/seed_<N>/<opponent>/run_manifest.json
    <output_root>/eval/seed_<N>/<opponent>/raw_episodes.csv

Usage:
    python scripts/eval_thesis_h_group.py                 # full protocol
    python scripts/eval_thesis_h_group.py --smoke         # 1 episode sanity
"""

import argparse
import copy
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling script import
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_thesis_h_group_multiseed import (  # noqa: E402
    SOURCE,
    _get_dotted,
    _override,
    build_h_group_config,
)

from uav_vpp_guidance.agents.ppo_agent import PPOAgent  # noqa: E402
from uav_vpp_guidance.common.git import get_git_info  # noqa: E402
from uav_vpp_guidance.common.hash import config_sha256, file_sha256  # noqa: E402
from uav_vpp_guidance.envs.end_to_end_opponent import EndToEndOpponent  # noqa: E402
from uav_vpp_guidance.envs.expert_opponent import ExpertOpponent  # noqa: E402
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv  # noqa: E402
from uav_vpp_guidance.evaluation.evaluate_prediction_comparison import (  # noqa: E402
    evaluate_method,
)
from uav_vpp_guidance.training.train_prediction_vpp_ppo import (  # noqa: E402
    load_experiment_config,
)

EVAL_SOURCE = "scripts/eval_thesis_h_group.py:GroupH"


def _build_opponent(stage: str, registry: dict):
    if stage == "expert":
        return ExpertOpponent(registry.get("expert", {}).get("config", {})), {"stage": stage}
    if stage == "end_to_end":
        entry = registry.get("end_to_end", {})
        return EndToEndOpponent.from_registry(entry), {"stage": stage, **entry}
    raise ValueError(f"Unsupported opponent stage: {stage}")


def _json_dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config",
        default="config/experiment/train_vpp_ppo_lstm_frozen.yaml",
        help="Training config the H runs were derived from.",
    )
    parser.add_argument(
        "--protocol-config",
        default="config/experiment/prediction_highlevel_observation_v1.yaml",
        help="Config supplying scenarios / opponent_registry / eval protocol.",
    )
    parser.add_argument(
        "--training-root",
        default="outputs/experiments/thesis_h_group",
        help="Root containing seed_<N>/checkpoints/best.pt from training.",
    )
    parser.add_argument("--seeds", default="42,43,44")
    # Thesis experiments use the rule-based expert opponent; end_to_end is
    # opt-in only (requires its checkpoint to exist).
    parser.add_argument("--opponents", default="expert")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    base_config = load_experiment_config(args.base_config)
    protocol = load_experiment_config(args.protocol_config)
    proto = protocol.get("protocol", {})
    registry = protocol.get("opponent_registry", {})

    scenarios = list(proto.get("scenarios") or list(protocol.get("scenarios", {}).keys()))
    eval_seeds = [int(s) for s in (proto.get("evaluation_seeds") or [0])]
    eps_per_scenario = int(proto.get("episodes_per_scenario") or 10)
    if args.smoke:
        scenarios = scenarios[:1]
        eval_seeds = [0]
        eps_per_scenario = 1

    train_seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    opponents = [s.strip() for s in args.opponents.split(",") if s.strip()]

    summary = []
    for train_seed in train_seeds:
        checkpoint = Path(args.training_root) / f"seed_{train_seed}" / "checkpoints" / "best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Missing checkpoint: {checkpoint}. Run training first."
            )
        for stage in opponents:
            cfg = build_h_group_config(base_config)
            if _get_dotted(cfg, "experiment.seed") is not None:
                _override(cfg, "experiment.seed", train_seed, source=f"{EVAL_SOURCE}:seed")
            if _get_dotted(cfg, "ppo.seed") is not None:
                _override(cfg, "ppo.seed", train_seed, source=f"{EVAL_SOURCE}:seed")

            run_dir = Path(args.training_root) / "eval" / f"seed_{train_seed}" / stage
            run_dir.mkdir(parents=True, exist_ok=True)
            cfg_snapshot = run_dir / "resolved_config.yaml"
            import yaml

            cfg_snapshot.write_text(
                yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )

            opponent, opponent_config = _build_opponent(stage, registry)
            env = CloseRangeTrackingEnv(
                cfg, opponent_policy=opponent, opponent_config=opponent_config
            )
            try:
                sample_obs = env.reset(seed=eval_seeds[0])
                obs_dim = int(sample_obs["observation_vector"].shape[0])
                if obs_dim != 19:
                    raise RuntimeError(
                        f"H-group eval env obs dim {obs_dim} != 19; refusing to "
                        "evaluate against a mismatched observation contract."
                    )
                action_dim = int(cfg.get("policy", {}).get("action_dim", 3))
                agent = PPOAgent(
                    obs_dim=obs_dim, action_dim=action_dim, config=cfg, device="cpu"
                )
                loaded = agent.load(str(checkpoint))
                ckpt_obs_dim = loaded.get("obs_dim")
                if ckpt_obs_dim is not None and int(ckpt_obs_dim) != obs_dim:
                    raise RuntimeError(
                        f"Checkpoint obs dim {ckpt_obs_dim} != env obs dim {obs_dim}; "
                        "refusing to evaluate a mismatched observation contract."
                    )
                metrics = evaluate_method(
                    env,
                    agent,
                    cfg,
                    f"h_group",
                    num_episodes=eps_per_scenario * len(scenarios),
                    seeds=eval_seeds,
                    scenarios=scenarios,
                    save_trajectories=False,
                    output_dir=str(run_dir),
                    training_seed=train_seed,
                )
            finally:
                env.close()

            for episode in metrics.get("raw_episodes", []):
                episode["method"] = "h_group"
                episode["opponent"] = stage
                episode["training_seed"] = train_seed
                episode["observation_contract"] = "history_kinematic_v1"
            manifest = {
                "source_id": SOURCE,
                "method": "h_group",
                "opponent": stage,
                "training_seed": train_seed,
                "evaluation_seeds": eval_seeds,
                "scenarios": scenarios,
                "episodes_per_scenario": eps_per_scenario,
                "checkpoint": {
                    "path": str(checkpoint.resolve()),
                    "sha256": file_sha256(checkpoint),
                },
                "config_sha256": config_sha256(cfg),
                "config_path": str(cfg_snapshot.resolve()),
                "git_info": get_git_info(str(REPO_ROOT)),
                "obs_dim": obs_dim,
                "action_dim": action_dim,
                "observation_schema": sample_obs.get("observation_schema", {}),
                "checkpoint_load": {
                    "loaded": True,
                    "checkpoint_obs_dim": ckpt_obs_dim,
                    "optimizer_loaded": bool(loaded.get("optimizer_loaded", False)),
                    "total_updates": loaded.get("total_updates"),
                    "total_timesteps": loaded.get("total_timesteps"),
                },
                "raw_episode_count": len(metrics.get("raw_episodes", [])),
                "paper_safe": True,
            }
            _json_dump(run_dir / "run_manifest.json", manifest)
            _json_dump(run_dir / "metrics.json", metrics)
            # evaluate_method returns flat metric keys (no "overall" nesting)
            overall = {
                key: metrics.get(key)
                for key in (
                    "success_rate",
                    "mean_return",
                    "crash_rate",
                    "out_of_bounds_rate",
                    "mean_min_range_m",
                    "num_episodes",
                )
            }
            print(
                f"[H-eval] seed={train_seed} opponent={stage} "
                f"episodes={manifest['raw_episode_count']} "
                f"success_rate={overall['success_rate']} "
                f"mean_return={overall['mean_return']}"
            )
            summary.append(
                {
                    "training_seed": train_seed,
                    "opponent": stage,
                    "run_dir": str(run_dir),
                    "overall": overall,
                }
            )

    _json_dump(
        Path(args.training_root) / "eval" / "summary.json",
        {
            "note": (
                "Compare against A0/B runs evaluated with the SAME scenarios, "
                "opponents, evaluation seeds and episodes_per_scenario. Paired "
                "statistics only; do not merge across training seeds."
            ),
            "runs": summary,
        },
    )
    print(f"[H-eval] summary written to {Path(args.training_root) / 'eval' / 'summary.json'}")


if __name__ == "__main__":
    main()
