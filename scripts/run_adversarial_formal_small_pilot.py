"""Run the adversarial formal-small pilot.

The pilot has two goals:
1. Sweep a small HP-damage grid with JSBSim evaluation runs.
2. Run a short adversarial curriculum training job and summarize stability.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JSBSIM_ROOT = (
    Path(os.environ["JSBSIM_ROOT"])
    if os.environ.get("JSBSIM_ROOT")
    else Path(r"E:\CloseAirCombat_control")
)


def _parse_float_list(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _parse_str_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def _run_command(cmd: List[str], log_path: Path) -> Dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    elapsed = time.time() - start
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n\n")
        if proc.stdout:
            f.write(proc.stdout)
        if proc.stderr:
            f.write("\n[stderr]\n")
            f.write(proc.stderr)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "elapsed_seconds": elapsed,
        "log_path": str(log_path),
    }


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _read_summary_rows(run_dir: Path) -> List[Dict[str, Any]]:
    summary_path = run_dir / "summary.csv"
    if not summary_path.exists():
        return []
    with open(summary_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_failures(run_dir: Path) -> List[Dict[str, Any]]:
    failures_path = run_dir / "failures.json"
    if not failures_path.exists():
        return [{"phase": "missing_failures_json", "error": str(failures_path)}]
    with open(failures_path, "r", encoding="utf-8") as f:
        return list((json.load(f) or {}).get("failures", []))


def _finite_mean(values: Iterable[float]) -> float:
    clean = [float(v) for v in values if math.isfinite(v)]
    return sum(clean) / len(clean) if clean else float("nan")


def _summarize_hp_run(
    run_dir: Path,
    run_id: str,
    damage_per_step: float,
    close_range_max_km: float,
    opponent_stage: str,
    command_result: Dict[str, Any],
) -> Dict[str, Any]:
    rows = _read_summary_rows(run_dir)
    failures = _read_failures(run_dir)
    outcomes = Counter(str(row.get("combat_outcome") or "none") for row in rows)
    reasons = Counter(str(row.get("termination_reason") or "none") for row in rows)
    ego_hp = [_safe_float(row.get("ego_hp")) for row in rows]
    target_hp = [_safe_float(row.get("target_hp")) for row in rows]
    hp_adv = [_safe_float(row.get("hp_advantage")) for row in rows]
    ttk = [_safe_float(row.get("combat_time_to_kill")) for row in rows]

    damaged = 0
    kills = 0
    for row in rows:
        ego = _safe_float(row.get("ego_hp"))
        target = _safe_float(row.get("target_hp"))
        if math.isfinite(ego) and math.isfinite(target) and (ego < 100.0 or target < 100.0):
            damaged += 1
        if (math.isfinite(ego) and ego <= 0.0) or (math.isfinite(target) and target <= 0.0):
            kills += 1

    episode_count = len(rows)
    decisive = outcomes["win"] + outcomes["loss"]
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "opponent_stage": opponent_stage,
        "damage_per_step": damage_per_step,
        "close_range_max_km": close_range_max_km,
        "returncode": command_result["returncode"],
        "elapsed_seconds": command_result["elapsed_seconds"],
        "failure_count": len(failures),
        "episodes": episode_count,
        "win_count": outcomes["win"],
        "loss_count": outcomes["loss"],
        "draw_count": outcomes["draw"],
        "decisive_fraction": decisive / episode_count if episode_count else float("nan"),
        "damaged_episode_fraction": damaged / episode_count if episode_count else float("nan"),
        "kill_fraction": kills / episode_count if episode_count else float("nan"),
        "mean_ego_hp": _finite_mean(ego_hp),
        "mean_target_hp": _finite_mean(target_hp),
        "mean_hp_advantage": _finite_mean(hp_adv),
        "mean_time_to_kill": _finite_mean(ttk),
        "outcomes": dict(outcomes),
        "termination_reasons": dict(reasons),
        "log_path": command_result["log_path"],
    }


def _write_hp_config(
    base_config: Path,
    config_dir: Path,
    damage_per_step: float,
    close_range_max_km: float,
) -> Path:
    cfg = _load_yaml(base_config)
    attack_cfg = cfg.setdefault("attack_zone", {})
    attack_cfg["damage_per_step"] = float(damage_per_step)
    attack_cfg["close_range_max_km"] = float(close_range_max_km)
    attack_cfg["close_range_enabled"] = True
    attack_cfg["legacy_range_enabled"] = True
    cfg.setdefault("experiment", {})
    cfg["experiment"]["name"] = "adversarial_hp_formal_small_pilot"
    cfg["experiment"]["description"] = (
        "Formal-small pilot config generated for HP damage and close-range attack-zone calibration."
    )
    tag = f"d{str(damage_per_step).replace('.', 'p')}_r{str(close_range_max_km).replace('.', 'p')}"
    out_path = config_dir / f"jsbsim_hrl_comparison_hp_{tag}.yaml"
    _write_yaml(out_path, cfg)
    return out_path


def run_hp_calibration(args: argparse.Namespace, output_root: Path) -> List[Dict[str, Any]]:
    base_config = REPO_ROOT / args.comparison_config
    config_dir = output_root / "configs"
    hp_output_root = output_root / "hp_runs"
    log_dir = output_root / "logs" / "hp"
    summaries: List[Dict[str, Any]] = []

    for damage in args.hp_damage_values:
        for close_max in args.hp_close_range_max_values:
            cfg_path = _write_hp_config(base_config, config_dir, damage, close_max)
            for opponent_stage in args.opponent_stages:
                run_id = (
                    f"hp_d{str(damage).replace('.', 'p')}"
                    f"_r{str(close_max).replace('.', 'p')}_{opponent_stage}"
                )
                cmd = [
                    sys.executable,
                    "scripts/run_jsbsim_hrl_comparison.py",
                    "--config",
                    str(cfg_path),
                    "--run-id",
                    run_id,
                    "--output-root",
                    str(hp_output_root),
                    "--backend",
                    "jsbsim",
                    "--jsbsim-root",
                    str(args.jsbsim_root),
                    "--run-status",
                    "formal-small",
                    "--methods",
                    *args.methods,
                    "--tasks",
                    *args.tasks,
                    "--seeds",
                    *[str(seed) for seed in args.seeds],
                    "--n-episodes",
                    str(args.n_episodes),
                    "--no-trajectory",
                    "--opponent-stage",
                    opponent_stage,
                ]
                result = _run_command(cmd, log_dir / f"{run_id}.log")
                summaries.append(
                    _summarize_hp_run(
                        hp_output_root / run_id,
                        run_id,
                        damage,
                        close_max,
                        opponent_stage,
                        result,
                    )
                )
    return summaries


def _write_curriculum_config(
    base_config: Path,
    config_dir: Path,
    damage_per_step: float,
    total_timesteps: int,
) -> Path:
    cfg = _load_yaml(base_config)
    cfg.setdefault("experiment", {})
    cfg["experiment"]["name"] = "adversarial_curriculum_formal_small_pilot"
    cfg["experiment"]["description"] = "Formal-small pilot config for adversarial curriculum stability."

    env_cfg = cfg.setdefault("env", {})
    env_cfg["backend"] = "simple"
    env_cfg["use_jsbsim"] = False
    env_cfg["max_high_level_steps"] = min(int(env_cfg.get("max_high_level_steps", 512)), 256)

    ppo_cfg = cfg.setdefault("ppo", {})
    ppo_cfg["total_timesteps"] = int(total_timesteps)
    ppo_cfg["rollout_steps"] = min(int(ppo_cfg.get("rollout_steps", 512)), 512)
    ppo_cfg["minibatch_size"] = min(int(ppo_cfg.get("minibatch_size", 128)), 128)
    ppo_cfg["update_epochs"] = min(int(ppo_cfg.get("update_epochs", 5)), 5)
    ppo_cfg["device"] = "cpu"

    eval_cfg = cfg.setdefault("evaluation", {})
    eval_cfg["eval_interval"] = max(512, min(int(total_timesteps) // 4, 1024))
    eval_cfg["eval_episodes"] = 4
    eval_cfg["seeds"] = [0, 1]
    eval_cfg["save_trajectories"] = False

    checkpoint_cfg = cfg.setdefault("checkpoint", {})
    checkpoint_cfg["save_interval"] = max(1024, min(int(total_timesteps) // 2, 2048))
    checkpoint_cfg["save_best"] = True
    checkpoint_cfg["save_last"] = True

    attack_cfg = cfg.setdefault("attack_zone", {})
    attack_cfg["enabled"] = True
    attack_cfg["damage_per_step"] = float(damage_per_step)
    attack_cfg["legacy_range_enabled"] = True
    attack_cfg["close_range_enabled"] = True
    attack_cfg.setdefault("close_range_min_km", 0.3)
    attack_cfg.setdefault("close_range_full_score_km", 1.0)
    attack_cfg.setdefault("close_range_max_km", 3.0)

    cfg["adversarial_curriculum"] = {
        "enabled": True,
        "initial_bucket": "weak",
        "switch_threshold": 0.7,
        "initial_elo": 1000.0,
        "elo_k_factor": 32.0,
        "weak_elo_max": 1050.0,
        "medium_elo_max": 1150.0,
        "final_opponent": "expert",
        "expert": {},
    }

    out_path = config_dir / "train_curriculum_adversarial_formal_small.yaml"
    _write_yaml(out_path, cfg)
    return out_path


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _finite_columns(rows: List[Dict[str, str]], columns: Iterable[str]) -> bool:
    if not rows:
        return False
    for row in rows:
        for column in columns:
            value = row.get(column, "")
            if value in ("", None):
                continue
            if not math.isfinite(_safe_float(value)):
                return False
    return True


def run_curriculum_stability(args: argparse.Namespace, output_root: Path) -> Dict[str, Any]:
    config_dir = output_root / "configs"
    cfg_path = _write_curriculum_config(
        REPO_ROOT / args.curriculum_config,
        config_dir,
        args.curriculum_damage_per_step,
        args.curriculum_timesteps,
    )
    train_dir = output_root / "curriculum_training"
    run_id = "curriculum_adversarial_formal_small"
    cmd = [
        sys.executable,
        "scripts/train_curriculum_ppo.py",
        "--config",
        str(cfg_path),
        "--backend",
        "simple",
        "--device",
        "cpu",
        "--seed",
        str(args.curriculum_seed),
        "--output-dir",
        str(train_dir),
    ]
    result = _run_command(cmd, output_root / "logs" / "curriculum" / f"{run_id}.log")

    log_dir = train_dir / "logs"
    update_rows = _read_csv(log_dir / "update_train_log.csv")
    eval_rows = _read_csv(log_dir / "eval_log.csv")
    curriculum_rows = _read_csv(log_dir / "curriculum_log.csv")
    episode_rows = _read_csv(log_dir / "episode_train_log.csv")
    pool_path = train_dir / "opponent_pool" / "opponent_pool.json"
    pool = {}
    if pool_path.exists():
        with open(pool_path, "r", encoding="utf-8") as f:
            pool = json.load(f) or {}

    numerically_stable = (
        result["returncode"] == 0
        and bool(update_rows)
        and bool(eval_rows)
        and _finite_columns(update_rows, ["policy_loss", "value_loss", "entropy", "approx_kl"])
        and _finite_columns(eval_rows, ["mean_return", "success_rate", "win_rate", "survival_rate"])
        and (train_dir / "checkpoints" / "last.pt").exists()
    )
    last_eval = eval_rows[-1] if eval_rows else {}
    survival_rate = _safe_float(last_eval.get("survival_rate"))
    crash_rate = _safe_float(last_eval.get("crash_rate"))
    oob_rate = _safe_float(last_eval.get("out_of_bounds_rate"))
    behavior_stable = (
        numerically_stable
        and math.isfinite(survival_rate)
        and survival_rate >= float(args.curriculum_min_survival_rate)
        and (not math.isfinite(crash_rate) or crash_rate <= float(args.curriculum_max_crash_rate))
        and (not math.isfinite(oob_rate) or oob_rate <= float(args.curriculum_max_oob_rate))
    )
    return {
        "run_id": run_id,
        "train_dir": str(train_dir),
        "config_path": str(cfg_path),
        "returncode": result["returncode"],
        "elapsed_seconds": result["elapsed_seconds"],
        "log_path": result["log_path"],
        "stable": numerically_stable,
        "numerically_stable": numerically_stable,
        "behavior_stable": behavior_stable,
        "behavior_thresholds": {
            "min_survival_rate": float(args.curriculum_min_survival_rate),
            "max_crash_rate": float(args.curriculum_max_crash_rate),
            "max_out_of_bounds_rate": float(args.curriculum_max_oob_rate),
        },
        "updates": len(update_rows),
        "eval_rows": len(eval_rows),
        "curriculum_rows": len(curriculum_rows),
        "episodes": len(episode_rows),
        "last_checkpoint_exists": (train_dir / "checkpoints" / "last.pt").exists(),
        "best_checkpoint_exists": (train_dir / "checkpoints" / "best.pt").exists(),
        "opponent_pool_size": len(pool.get("entries", [])),
        "last_eval": last_eval,
        "last_curriculum": curriculum_rows[-1] if curriculum_rows else {},
    }


def _write_hp_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "opponent_stage",
        "damage_per_step",
        "close_range_max_km",
        "returncode",
        "failure_count",
        "episodes",
        "win_count",
        "loss_count",
        "draw_count",
        "decisive_fraction",
        "damaged_episode_fraction",
        "kill_fraction",
        "mean_ego_hp",
        "mean_target_hp",
        "mean_hp_advantage",
        "mean_time_to_kill",
        "elapsed_seconds",
        "run_dir",
        "log_path",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _recommend_hp(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [row for row in rows if row["returncode"] == 0 and row["failure_count"] == 0]
    if not valid:
        return {"status": "no_valid_hp_run"}
    grouped: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in valid:
        key = (row["damage_per_step"], row["close_range_max_km"])
        grouped.setdefault(key, []).append(row)
    scored = []
    for (damage, close_max), items in grouped.items():
        decisive = _finite_mean(row["decisive_fraction"] for row in items)
        damaged = _finite_mean(row["damaged_episode_fraction"] for row in items)
        kills = _finite_mean(row["kill_fraction"] for row in items)
        score = 0.0
        score -= abs(decisive - 0.6)
        score -= abs(damaged - 0.8) * 0.5
        score -= max(0.0, kills - 0.5)
        scored.append(
            {
                "damage_per_step": damage,
                "close_range_max_km": close_max,
                "mean_decisive_fraction": decisive,
                "mean_damaged_episode_fraction": damaged,
                "mean_kill_fraction": kills,
                "score": score,
            }
        )
    scored.sort(key=lambda item: item["score"], reverse=True)
    return {"status": "ok", "recommended": scored[0], "candidates": scored}


def _write_markdown_report(
    path: Path,
    hp_rows: List[Dict[str, Any]],
    hp_recommendation: Dict[str, Any],
    curriculum_summary: Dict[str, Any],
) -> None:
    lines = [
        "# Adversarial Formal-Small Pilot",
        "",
        "## HP Calibration",
        "",
        f"Runs: {len(hp_rows)}",
        f"Valid runs: {sum(1 for row in hp_rows if row['returncode'] == 0 and row['failure_count'] == 0)}",
        "",
        "| damage | close_max_km | opponent | episodes | win/loss/draw | damaged_frac | kill_frac | mean_hp_adv | failures |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in hp_rows:
        lines.append(
            "| {damage_per_step:.3g} | {close_range_max_km:.3g} | {opponent_stage} | {episodes} | "
            "{win_count}/{loss_count}/{draw_count} | {damaged_episode_fraction:.3g} | "
            "{kill_fraction:.3g} | {mean_hp_advantage:.3g} | {failure_count} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Recommendation:",
            "```json",
            json.dumps(hp_recommendation, indent=2, ensure_ascii=False),
            "```",
            "",
            "## Curriculum Stability",
            "",
            "```json",
            json.dumps(curriculum_summary, indent=2, ensure_ascii=False),
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run adversarial formal-small pilot")
    parser.add_argument("--output-root", default="outputs/formal_small_pilot/adversarial")
    parser.add_argument("--comparison-config", default="config/experiment/jsbsim_hrl_comparison.yaml")
    parser.add_argument("--curriculum-config", default="config/experiment/train_curriculum_ppo.yaml")
    parser.add_argument("--jsbsim-root", default=str(DEFAULT_JSBSIM_ROOT))
    parser.add_argument("--hp-damage-values", type=_parse_float_list, default=_parse_float_list("0.5,1.0"))
    parser.add_argument("--hp-close-range-max-values", type=_parse_float_list, default=_parse_float_list("3.0"))
    parser.add_argument("--opponent-stages", type=_parse_str_list, default=_parse_str_list("expert,end_to_end"))
    parser.add_argument("--methods", type=_parse_str_list, default=_parse_str_list("no_prediction_vpp"))
    parser.add_argument(
        "--tasks",
        type=_parse_str_list,
        default=_parse_str_list("head_on,crossing_feasible,break_turn,sustained_turn"),
    )
    parser.add_argument("--seeds", type=_parse_int_list, default=_parse_int_list("0,1"))
    parser.add_argument("--n-episodes", type=int, default=1)
    parser.add_argument("--curriculum-timesteps", type=int, default=4096)
    parser.add_argument("--curriculum-seed", type=int, default=0)
    parser.add_argument("--curriculum-damage-per-step", type=float, default=0.5)
    parser.add_argument("--curriculum-min-survival-rate", type=float, default=0.5)
    parser.add_argument("--curriculum-max-crash-rate", type=float, default=0.5)
    parser.add_argument("--curriculum-max-oob-rate", type=float, default=0.5)
    parser.add_argument("--skip-hp", action="store_true")
    parser.add_argument("--skip-curriculum", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    args.jsbsim_root = Path(args.jsbsim_root)
    output_root = REPO_ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    hp_rows: List[Dict[str, Any]] = []
    if not args.skip_hp:
        hp_rows = run_hp_calibration(args, output_root)
        _write_hp_csv(output_root / "hp_calibration_summary.csv", hp_rows)
    hp_recommendation = _recommend_hp(hp_rows) if hp_rows else {"status": "skipped"}

    curriculum_summary: Dict[str, Any] = {"status": "skipped"}
    if not args.skip_curriculum:
        curriculum_summary = run_curriculum_stability(args, output_root)

    report = {
        "schema_version": "1.0.0",
        "output_root": str(output_root),
        "hp_calibration": hp_rows,
        "hp_recommendation": hp_recommendation,
        "curriculum_stability": curriculum_summary,
    }
    with open(output_root / "pilot_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    _write_markdown_report(
        output_root / "pilot_report.md",
        hp_rows,
        hp_recommendation,
        curriculum_summary,
    )

    failed_hp = [row for row in hp_rows if row["returncode"] != 0 or row["failure_count"] > 0]
    failed_curriculum = (
        (not args.skip_curriculum)
        and (curriculum_summary.get("returncode") != 0 or not curriculum_summary.get("stable"))
    )
    if failed_hp or failed_curriculum:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
