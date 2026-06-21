"""规则策略验证脚本（air-combat-mvp 任务 17）。

在 ``backend: simple`` 后端上，用**规则跟踪策略**驱动本机（Ego），导弹由
``AirCombatMVPEnv`` 在满足锁定 + 发射包线条件时自动创建并按 PNG 制导律飞向目标；
目标做直线/简单机动（constant_velocity）。运行 20 回合，汇总并输出发射率、命中率、
平均击杀时间，并按通过标准（发射率 > 90% 且命中率 > 80% 且平均击杀时间 < 30s）
报告 PASS / FAIL（需求 6.1–6.6）。

规则跟踪策略（主动跟踪，PN/LOS 率转向保持目标在视场内）：
    Ego 动作恒为 ``np.zeros(3)``（VPP 零偏移），配合 ``virtual_point.anchor_mode =
    current_target``，使底层 LOS-rate 制导以"当前目标"为虚拟锚点、零偏移地飞向目标，
    即 Ego 主动指向目标、将其保持在雷达视场内——这是最简洁有效的"规则策略"
    （需求 6.1）。

坐标系约定（重要）：
    宿主 ``SimplePointMassEnv`` 使用 **NEU** 轴序：position_m = [north, east, up]
    （altitude = 索引 2）。本脚本据此构造每回合的目标绝对位置（NEU）并通过
    ``env.reset(scenario=...)`` 传入。

运行（Python 3.11 / PowerShell）::

    $env:PYTHONPATH="src"; & "C:/Users/admin/.conda/envs/py3.11/python.exe" \
        scripts/validate_mvp_air_combat.py
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

# 确保 src 在 import 路径上（兼容直接 python scripts/... 调用）。
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from uav_vpp_guidance.envs.air_combat_mvp_env import AirCombatMVPEnv  # noqa: E402
from uav_vpp_guidance.utils.config import (  # noqa: E402
    load_yaml_config,
    merge_config,
)

# ---------------------------------------------------------------------------
# 通过标准（需求 6.5）。
# ---------------------------------------------------------------------------
LAUNCH_RATE_THRESHOLD = 0.90   # 发射率 > 90%
HIT_RATE_THRESHOLD = 0.80      # 命中率 > 80%
KILL_TIME_THRESHOLD_S = 30.0   # 平均击杀时间 < 30s

N_EPISODES = 20
MAX_STEPS = 400

DEFAULT_PILOT_CONFIG = os.path.join(
    _REPO_ROOT, "config", "experiment", "air_combat_mvp_pilot.yaml"
)


def _base_working_config() -> dict:
    """构造一份已验证可运行的最小 simple-backend 基础配置。

    复用 ``tests/test_mvp_air_combat.py::_build_config`` 中经验证可在不依赖 JSBSim
    的前提下稳定构造 / reset / step 的配置骨架（env / virtual_point / limits /
    reward / guidance），随后由 pilot yaml 的 air_combat / scenario / env 值覆盖。

    Returns:
        dict: 基础配置字典。
    """
    return {
        "experiment": {
            "name": "validate_mvp_air_combat",
            "seed": 0,
            "output_root": "outputs",
        },
        "env": {
            "use_jsbsim": False,
            "backend": "simple",
            "decision_freq": 5,
            "sim_freq": 60,
            "max_high_level_steps": MAX_STEPS,
            "success_range_m": 900.0,
            "success_ata_deg": 25.0,
            "success_hold_time_s": 0.2,
            "hysteresis_range_m": 950.0,
            "hysteresis_ata_deg": 30.0,
            "min_altitude_m": 500.0,
            "max_altitude_m": 15000.0,
            "max_range_m": 8000.0,
            "target_mode": "constant_velocity",
            "high_level_dt": 0.2,
        },
        "virtual_point": {
            "anchor_mode": "current_target",  # 零偏移即可主动跟踪目标（规则策略）。
            "action_dim": 3,
            "d_long_range": [-1500.0, 1500.0],
            "d_lat_range": [-800.0, 800.0],
            "d_vert_range": [-500.0, 500.0],
            "smoothing_alpha": 0.3,
        },
        "trajectory_prediction": {"enabled": False},
        "limits": {
            "nz_min": -2.0,
            "nz_max": 7.0,
            "roll_rate_min": -1.5,
            "roll_rate_max": 1.5,
            "throttle_min": 0.0,
            "throttle_max": 1.0,
        },
        "reward": {
            "w_range": 0.5,
            "w_angle": 0.8,
            "w_energy": 0.2,
            "w_safety": 2.0,
            "w_saturation": 1.0,
            "w_smooth": 0.1,
            "terminal_success": 200.0,
            "terminal_failure": -200.0,
            "terminal_crash": -300.0,
            "min_altitude_m": 500.0,
        },
        "guidance": {
            "mode": "los_rate",
            "use_gain_adapter": False,
            "gains": {
                "k_los": 1.0,
                "k_pos": 0.5,
                "k_damp": 0.2,
                "k_roll": 1.0,
                "k_speed": 0.2,
                "alpha_filter": 0.3,
            },
        },
        "air_combat": {
            "missile_config": {
                "ego": {"enabled": True},
                "target": {"enabled": False},
            },
            "radar_config": {
                "max_range_m": 10000.0,
                "azimuth_fov_deg": 60.0,
                "elevation_fov_deg": 30.0,
                "lock_time_steps": 3,
            },
            "reward": {"use_sparse": True},
        },
    }


def _load_pilot_overrides(pilot_path: str) -> dict:
    """加载 pilot yaml（解析 includes 后仅取 air_combat / scenario / env 覆盖）。

    pilot yaml 的 ``includes``（env.yaml / ppo.yaml）不提供 guidance/limits/reward/
    virtual_point，且 env.yaml 默认 ``use_jsbsim: true``；为保持脚本在 simple 后端
    稳定运行，这里只提取 pilot 自身定义的 ``air_combat`` / ``scenario`` 段及其
    ``env`` 覆盖，与 :func:`_base_working_config` 合并。

    Args:
        pilot_path: pilot yaml 路径。

    Returns:
        dict: 仅含 env / scenario / air_combat 的覆盖字典（缺省时为空）。
    """
    if not os.path.exists(pilot_path):
        print(f"[warn] pilot 配置不存在：{pilot_path}，仅使用内置基础配置。")
        return {}

    raw = load_yaml_config(pilot_path)
    raw.pop("includes", None)  # 不解析 includes（避免引入 jsbsim 默认）。

    overrides: dict = {}
    for key in ("env", "scenario", "air_combat"):
        if key in raw and raw[key] is not None:
            overrides[key] = raw[key]
    # 强制 simple 后端，覆盖任何 includes 残留。
    overrides.setdefault("env", {})
    overrides["env"]["use_jsbsim"] = False
    overrides["env"]["backend"] = "simple"
    overrides["env"]["max_high_level_steps"] = MAX_STEPS
    return overrides


def build_config(pilot_path: str) -> dict:
    """构造完整环境配置：基础工作配置 + pilot 覆盖。

    Args:
        pilot_path: pilot yaml 路径。

    Returns:
        dict: 传给 ``AirCombatMVPEnv`` 的完整配置字典。
    """
    config = _base_working_config()
    overrides = _load_pilot_overrides(pilot_path)
    return merge_config(config, overrides)


def _sample_target_neu_position(
    scenario_cfg: dict, rng: np.random.Generator
) -> np.ndarray:
    """按 pilot scenario.target_init 的极坐标范围采样目标 NEU 绝对位置。

    Ego 固定在 NEU [0, 0, 5000]。目标相对 Ego 的几何由 (range, azimuth, elevation)
    采样，并转换为 NEU 绝对位置::

        north = range * cos(el) * cos(az)
        east  = range * cos(el) * sin(az)
        up    = 5000 + range * sin(el)

    其中 azimuth 自北起、向东为正；elevation 自水平面起、向上为正。

    Args:
        scenario_cfg: pilot ``scenario`` 配置段。
        rng: numpy 随机数发生器。

    Returns:
        np.ndarray: 目标 NEU 绝对位置 [north, east, up]。
    """
    tgt = scenario_cfg.get("target_init", {})
    range_rng = tgt.get("range_m", [3000.0, 5000.0])
    az_rng = tgt.get("azimuth_deg", [-30.0, 30.0])
    el_rng = tgt.get("elevation_deg", [-10.0, 10.0])

    range_m = float(rng.uniform(range_rng[0], range_rng[1]))
    az = math.radians(float(rng.uniform(az_rng[0], az_rng[1])))
    el = math.radians(float(rng.uniform(el_rng[0], el_rng[1])))

    ego_up = 5000.0
    north = range_m * math.cos(el) * math.cos(az)
    east = range_m * math.cos(el) * math.sin(az)
    up = ego_up + range_m * math.sin(el)
    return np.array([north, east, up], dtype=float)


def build_episode_scenario(
    scenario_cfg: dict, rng: np.random.Generator
) -> dict:
    """构造单回合场景字典（base reset 期望的 own_init / target_init 形式）。

    position_m 为 NEU [north, east, up]；velocity_mps 为标量速率；heading_deg 为
    航向角（度，自北起、向东为正）。

    Args:
        scenario_cfg: pilot ``scenario`` 配置段。
        rng: numpy 随机数发生器。

    Returns:
        dict: ``{"name", "own_init", "target_init"}`` 场景字典。
    """
    ego_init_cfg = scenario_cfg.get("ego_init", {})
    ego_vel = float(ego_init_cfg.get("velocity_mps", 250.0))
    ego_heading = float(ego_init_cfg.get("heading_deg", 0.0))

    tgt_cfg = scenario_cfg.get("target_init", {})
    tgt_vel = float(tgt_cfg.get("velocity_mps", 250.0))
    tgt_heading = float(tgt_cfg.get("heading_deg", 180.0))

    target_pos = _sample_target_neu_position(scenario_cfg, rng)

    return {
        "name": "air_combat_mvp_pilot",
        "own_init": {
            "position_m": np.array([0.0, 0.0, 5000.0], dtype=float),
            "velocity_mps": ego_vel,
            "heading_deg": ego_heading,
        },
        "target_init": {
            "position_m": target_pos,
            "velocity_mps": tgt_vel,
            "heading_deg": tgt_heading,
        },
    }


def _has_event(info: dict, event_name: str) -> bool:
    """info 中本步是否包含指定事件。"""
    return event_name in (info.get("events") or [])


def run_episode(
    env: AirCombatMVPEnv,
    scenario: dict,
    seed: int,
    dt: float,
    diagnostics: dict,
) -> dict:
    """运行单回合，返回该回合统计。

    Args:
        env: 空战环境。
        scenario: 本回合场景字典。
        seed: 本回合随机种子。
        dt: high_level_dt（用于击杀时间换算）。
        diagnostics: 跨回合诊断累加器（导弹速度曲线 / 锁定连续性）。

    Returns:
        dict: ``{"launched", "hit", "kill_time"}``（kill_time 仅命中时有效）。
    """
    env.reset(scenario=scenario, seed=seed)
    action = np.zeros(3, dtype=float)  # 规则跟踪：零偏移 current_target。

    launched = False
    hit = False
    launch_step = None
    kill_time = None

    # 诊断：本回合导弹在飞期间速度最小值、锁定步计数。
    min_missile_speed = math.inf
    locked_steps = 0
    in_view_steps = 0

    for step_idx in range(MAX_STEPS):
        _obs, _reward, terminated, truncated, info = env.step(action)

        radar = env._last_radar_state
        if radar.get("locked"):
            locked_steps += 1
        if radar.get("in_view"):
            in_view_steps += 1

        if _has_event(info, "missile_launch") and not launched:
            launched = True
            launch_step = step_idx

        if env.missile_ego.in_flight:
            spd = float(np.linalg.norm(env.missile_ego.velocity_mps))
            min_missile_speed = min(min_missile_speed, spd)

        if _has_event(info, "missile_hit"):
            hit = True
            if launch_step is not None:
                kill_time = (step_idx - launch_step) * dt
            else:
                kill_time = step_idx * dt

        if terminated or truncated:
            break

    diagnostics["locked_steps"].append(locked_steps)
    diagnostics["in_view_steps"].append(in_view_steps)
    if min_missile_speed != math.inf:
        diagnostics["min_missile_speed"].append(min_missile_speed)

    return {"launched": launched, "hit": hit, "kill_time": kill_time}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="规则策略验证空战 MVP 闭环（任务 17）。"
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_PILOT_CONFIG,
        help="pilot 场景配置 yaml 路径（默认 air_combat_mvp_pilot.yaml）。",
    )
    parser.add_argument(
        "--episodes", type=int, default=N_EPISODES, help="回合数（默认 20）。"
    )
    parser.add_argument("--seed", type=int, default=0, help="基础随机种子。")
    args = parser.parse_args()

    config = build_config(args.config)
    scenario_cfg = config.get("scenario", {})
    dt = config["env"]["high_level_dt"]

    env = AirCombatMVPEnv(config)
    rng = np.random.default_rng(args.seed)

    diagnostics = {
        "min_missile_speed": [],
        "locked_steps": [],
        "in_view_steps": [],
    }

    n_launched = 0
    n_hit = 0
    kill_times: list[float] = []

    print("=" * 70)
    print(f"空战 MVP 规则策略验证：{args.episodes} 回合（simple 后端）")
    print("=" * 70)

    for ep in range(args.episodes):
        scenario = build_episode_scenario(scenario_cfg, rng)
        result = run_episode(
            env, scenario, seed=args.seed + ep, dt=dt, diagnostics=diagnostics
        )
        if result["launched"]:
            n_launched += 1
        if result["hit"]:
            n_hit += 1
            if result["kill_time"] is not None:
                kill_times.append(result["kill_time"])

        status = "HIT " if result["hit"] else ("LAUNCH" if result["launched"] else "----")
        kt = f"{result['kill_time']:.1f}s" if result["kill_time"] is not None else "-"
        print(
            f"  回合 {ep + 1:2d}: launched={result['launched']!s:5s} "
            f"hit={result['hit']!s:5s} kill_time={kt:>6s}  [{status}]"
        )

    launch_rate = n_launched / args.episodes
    hit_rate = n_hit / args.episodes
    avg_kill_time = float(np.mean(kill_times)) if kill_times else float("inf")

    print("-" * 70)
    print("汇总指标：")
    print(f"  发射率 (launch_rate)      = {launch_rate * 100:.1f}% "
          f"({n_launched}/{args.episodes})  [阈值 > 90%]")
    print(f"  命中率 (hit_rate)         = {hit_rate * 100:.1f}% "
          f"({n_hit}/{args.episodes})  [阈值 > 80%]")
    if kill_times:
        print(f"  平均击杀时间 (avg_kill_time) = {avg_kill_time:.2f}s  [阈值 < 30s]")
    else:
        print("  平均击杀时间 (avg_kill_time) = N/A（无命中回合）  [阈值 < 30s]")

    passed = (
        launch_rate > LAUNCH_RATE_THRESHOLD
        and hit_rate > HIT_RATE_THRESHOLD
        and avg_kill_time < KILL_TIME_THRESHOLD_S
    )

    print("=" * 70)
    if passed:
        print("结果：PASS  空战闭环达到通过标准。")
    else:
        print("结果：FAIL  未达通过标准。诊断备注（任务 17 备选路径）：")
        # 诊断：导弹速度曲线。
        if diagnostics["min_missile_speed"]:
            mn = min(diagnostics["min_missile_speed"])
            avg = float(np.mean(diagnostics["min_missile_speed"]))
            print(f"    - 导弹在飞期间最小速度：全局最小 {mn:.1f} m/s，"
                  f"逐回合均值 {avg:.1f} m/s（期望持续 > 400 m/s；过低提示阻力过大/失速）。")
        else:
            print("    - 导弹从未在飞（无发射）：检查雷达是否锁定、几何是否进入发射包线。")
        # 诊断：雷达锁定连续性。
        if diagnostics["locked_steps"]:
            avg_lock = float(np.mean(diagnostics["locked_steps"]))
            avg_view = float(np.mean(diagnostics["in_view_steps"]))
            print(f"    - 雷达：逐回合平均 in_view 步数 {avg_view:.1f}，"
                  f"locked 步数 {avg_lock:.1f}（锁定步过少提示目标频繁脱离视场→锁定计数反复清零）。")
        print("    - 可调参数（优先 pilot yaml air_combat.missile_config.ego 覆盖）："
              "KILL_RADIUS_M→50、NAV_CONSTANT→4–5、LAUNCH_ATA_MAX_DEG→45。")
    print("=" * 70)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
