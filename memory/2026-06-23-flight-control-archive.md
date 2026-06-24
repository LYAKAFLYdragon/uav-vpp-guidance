# 底层飞控阶段性结论与存档

**日期**：2026-06-23  
**分支**：`fix/flight-control-comparison-sync`  
**评估任务**：multi-waypoint tracking（5 个航点，JSBSim F-16，20 seeds）

## 当前最佳状态

针对 **Enhanced PID / Robust PID** 的最优配置为：

```yaml
low_level_controller:
  type: enhanced
  enable_bank_angle_protection: true
  max_bank_rad: 1.3264502315156905   # 76 deg
  bank_violation_threshold: 25
  bank_protection_nz_increment: 0.5
  altitude_hold_gain: 0.01

guidance:
  post_process:
    enabled: true
    enable_lift_compensation: true
    lift_compensation_factor: 1.0
    lift_compensation_source: "actual_roll"
    lift_compensation_filter_alpha: 0.3
    lift_compensation_min_cos: 0.5
```

对应 ablation 配置：`config/experiment/ablation_mw_bank76_alt_hold_lift_smooth.yaml`

## 关键性能

| 控制器 | crash / 20 | 平均完成航点 | 完全完成率 | 备注 |
|---|---|---|---|---|
| Enhanced PID | 0 | 3.10 / 5 | 12/20 | 推荐用于上层决策实验 |
| Robust PID | 0 | 3.10 / 5 | 12/20 | 与 Enhanced 基本持平 |
| PPO+PID | 1 | 2.00 / 5 | 8/20 | 安全但完成度偏低 |
| Baseline PID | 12 | 2.00 / 5 | 8/20 | 不建议使用 |

> 详细数据见 `results/ablation_mw_v4_summary.json`。

## 是否足以支撑空战机动决策？

**结论**：在“多航点跟踪”这一 proxy 任务上，当前底层飞控（Enhanced/Robust PID + 76° 坡度保护 + 高度保持 + 平滑升力补偿）已经具备 **支撑上层决策研究的基本能力**：

- **无 crash**：20 次随机种子全部稳定运行。
- **中等完成度**：平均完成 3.1 个航点（满分 5），航点切换未出现螺旋下坠等失控现象。
- **可预期**：该配置在 75°–76° 区间内表现稳定，77° 以上开始重新出现 crash，说明边界清晰。

**但尚未验证的能力**（需在上层设计完成后再回头研究）：

1. **高动态空战机动**：当前仅测试了航点跟踪，未涉及大过载转弯、剪刀机动、导弹规避等 aggressive BFM。
2. **对抗性目标**：目标速度固定 100–120 m/s，无规避行为；上层决策引入对抗后，底层是否能跟上尚未可知。
3. **PPO+PID 性能瓶颈**：PPO+PID 完成度明显低于固定 PID，且需要单独调参（坡度限制、升力补偿来源等）。
4. **Baseline PID 结构性弱**：不适合作为空战机动决策的底层执行器。
5. **观测溢出/NaN 传播**：部分 run 中仍有 observation overflow 警告，虽然加了 guard，但根因未完全消除。

## 存档建议

- 将 `bank76_alt_hold_lift_smooth` 作为 **上层决策实验的默认底层配置**。
- 继续把 `ablation_mw_bank75_alt_hold_lift_smooth.yaml` 作为保守备选（0 crash，2.95 平均航点）。
- 上层设计阶段可暂不深究底层 PID，但需在实验日志中记录：
  - 所有上层实验使用同一套底层配置，避免结果不可比。
  - 若上层策略要求更激进机动，必须重新评估底层。

## 后续返回底层飞控时的检查清单

- [ ] 在 break-turn / sustained-turn / 对抗机动任务上验证 `bank76_alt_hold_lift_smooth`。
- [ ] 为 PPO+PID 单独做坡度保护/no-bank 扫描。
- [ ] 定位并消除 observation overflow 的根因。
- [ ] 评估是否需要更高频的低层控制或增益调度。

## 2026-06-24 修复后验证记录

本轮修复覆盖 P0/P1 项：积分器条件积分与 back-calculation 反饱和、失速/坡度/高度保护优先级仲裁、JSBSim state 非有限值清洗、多航点未使用 backend target 不再 step、油门限幅统一收紧到 F-16 物理 envelope `[0.4, 0.9]`。旧配置中的 `throttle_min: 0.0` / `throttle_max: 1.0` 通过 `clip_command()` 的有效限幅交集自动收紧，无需逐个改 YAML。

### 20-seed 验证

命令：

```powershell
python scripts/run_flight_control_smoke_test.py --controllers enhanced,robust --configs bank76_alt_hold_lift_smooth --seeds 20 --task multi_waypoint --backend jsbsim --jobs 4 --run-id _codex_bank76_20seed
python scripts/run_flight_control_smoke_test.py --controllers enhanced,robust --configs bank80_alt_hold --seeds 20 --task multi_waypoint --backend jsbsim --jobs 2 --run-id _codex_bank80_20seed --no-check
```

结果：

| 配置 | 控制器 | crash / 20 | 平均完成航点 | 完全完成率 | 判定 |
|---|---|---:|---:|---:|---|
| `bank76_alt_hold_lift_smooth` | Enhanced PID | 0 | 4.25 / 5 | 17/20 | PASS，优于修复前 3.10 |
| `bank76_alt_hold_lift_smooth` | Robust PID | 0 | 4.25 / 5 | 17/20 | PASS，优于修复前 3.10 |
| `bank80_alt_hold` | Enhanced PID | 1 | 4.00 / 5 | 16/20 | PASS，满足压力标准 crash <= 3/20 |
| `bank80_alt_hold` | Robust PID | 1 | 4.00 / 5 | 16/20 | PASS，满足压力标准 crash <= 3/20 |

产物：

- `outputs/flight_control_smoke/_codex_bank76_20seed/aggregate/episode_records.json`
- `outputs/flight_control_smoke/_codex_bank80_20seed/aggregate/episode_records.json`

### 跳过测试原因

命令：

```powershell
python -m pytest tests/test_pid_controller_family.py tests/test_jsbsim_env_p1.py tests/test_overload_rollrate.py tests/test_observation.py tests/test_tracking_env_no_prediction.py tests/test_multi_waypoint_env.py tests/test_break_turn_env.py tests/test_sustained_turn_env.py tests/test_los_guidance.py tests/test_jsbsim_bridge.py tests/test_eval_jsbsim_guidance_comparison.py tests/test_opponent_controller.py tests/test_flight_control_comparison_formal.py -rs -q
```

结果：`153 passed, 15 skipped, 8 warnings`。15 个 skipped 均为本地单元测试环境未设置 `JSBSIM_ROOT` 或找不到 JSBSim data directory；本轮真实 JSBSim 验证通过脚本级 harness 运行并产生了 episode 产物。

### P0-1 break_turn 验收

命令：

```powershell
python scripts/run_flight_control_comparison.py --run-id _codex_break_turn_20seed --controllers enhanced_pid robust_pid --tasks break_turn --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 --n-episodes 1 --jobs 2 --backend jsbsim --status smoke --output-root outputs/flight_control_compare --config-base config/experiment/ablation_mw_bank76_alt_hold_lift_smooth.yaml --config-bt config/experiment/task_break_turn.yaml
```

结果：

| 控制器 | crash / 20 | nz_tracking_rmse | recovery_delay | overshoot_pct | 判定 |
|---|---:|---:|---:|---:|---|
| Enhanced PID | 0 | 2.61g | NaN | 243.6% | FAIL |
| Robust PID | 0 | 2.61g | NaN | 243.6% | FAIL |

说明：现有 `task_break_turn.yaml` 的 `nz_cmd` 全程保持在 1g 以上，没有触发验收定义中的过载反向过零事件，因此 `recovery_delay` 无法测量。实际 `nz_g` 仍有明显振荡，P0-1 不能判定为通过；下一轮需要先把 break_turn 验收任务改成明确的多次 nz 指令反向场景，再回到控制器响应质量。

产物：`outputs/flight_control_compare/_codex_break_turn_20seed/aggregate/episode_records.json`

### M1 E2E PPO telemetry

本轮未启动 3 seeds × 1M steps 的完整对比训练，原因是：

- 现有旧 run `outputs/experiments/end_to_end_ppo` 是 200k steps，不是 1M baseline。
- 旧 eval trajectory 里的 `throttle_cmd` 实际记录的是 normalized action，不是物理 throttle，无法直接计算修复前物理油门分布。

已补齐训练入口：

- `train_end_to_end_ppo.py` 支持 `--total-timesteps`，无需改 YAML 即可运行 1M。
- 训练 episode log 新增 `mean_throttle_cmd`、`min_throttle_cmd`、`max_throttle_cmd`、`throttle_at_min_rate`、`throttle_at_max_rate`、`throttle_outside_effective_rate`、`throttle_saturation_rate`。
- eval trajectory 现在保留 normalized action，同时写出物理 `nz_cmd` / `roll_rate_cmd` / `throttle_cmd`。

smoke 命令：

```powershell
python -m uav_vpp_guidance.training.train_end_to_end_ppo --config config/experiment/train_end_to_end_ppo.yaml --output-dir outputs/experiments/_codex_e2e_m1_smoke --seed 0 --device cpu --backend simple --total-timesteps 1024 --smoke
python scripts/train_end_to_end_baseline.py --config config/experiment/train_end_to_end_ppo.yaml --output-dir outputs/experiments/_codex_e2e_m1_1m_seed0_dryrun --seed 0 --device cpu --total-timesteps 1000000 --dry-run
```

smoke 结果：eval 物理 throttle 共 5256 个样本，min=0.633、max=0.649、outside `[0.4, 0.9]` rate=0.0；训练 episode `throttle_outside_effective_rate=0.0`。

### 720-episode 集成回归

按 3 个配置拆分运行，每个配置 `4 controllers × 3 tasks × 20 seeds = 240 episodes`：

- `outputs/flight_control_compare/_codex_regression_bank75`
- `outputs/flight_control_compare/_codex_regression_bank76`
- `outputs/flight_control_compare/_codex_regression_bank80`

关键结果：

| 配置 | 场景 | 控制器 | crash / 20 | 主指标 | 判定 |
|---|---|---|---:|---:|---|
| bank76 | multi_waypoint | Enhanced PID | 0 | mean_wp=4.25 | PASS |
| bank76 | multi_waypoint | Robust PID | 0 | mean_wp=4.25 | PASS |
| bank80 | multi_waypoint | Enhanced PID | 1 | mean_wp=4.00 | PASS，压力标准 <=3/20 |
| bank80 | multi_waypoint | Robust PID | 1 | mean_wp=4.00 | PASS，压力标准 <=3/20 |
| bank76 | sustained_turn | Enhanced PID | 0 | mean_orbits=1.11 | FAIL，低于 2.0 |
| bank76 | sustained_turn | Robust PID | 0 | mean_orbits=1.11 | FAIL，低于 2.0 |
| bank76 | break_turn | Enhanced PID | 0 | nz_rmse=2.61g | FAIL，目标 <0.5g |
| bank76 | break_turn | Robust PID | 0 | nz_rmse=2.61g | FAIL，目标 <0.5g |
| bank76 | multi_waypoint | Baseline PID | 0 | mean_wp=5.00 | PASS，但优于预期基线，不再是弱 baseline |

结论：多航点和 bank80 crash 压力指标通过；sustained_turn 的圈数目标和 break_turn 的 nz 跟踪目标未通过。当前 profile 可继续用于温和上层空战实验，但不能声明高动态机动验收完成。

### 上层空战配置引用

已将上层 JSBSim/对抗入口显式标记为 `bank76_alt_hold_lift_smooth`：

- `config/experiment/jsbsim_hrl_comparison.yaml`
- `config/experiment/train_no_prediction_vpp_ppo_jsbsim_compare.yaml`
- `config/experiment/train_curriculum_ppo.yaml`
- `scripts/run_adversarial_formal_small_pilot.py`

静态 YAML 增加 `experiment.flight_control_profile`，JSBSim 相关配置显式写入 76° bank protection、高度保持、平滑升力补偿和 `[0.4, 0.9]` throttle envelope。adversarial pilot 生成派生 YAML 时也会注入同一 profile，并通过 `record_config_override_if_changed()` 记录覆盖。

### 本轮验证命令

```powershell
python -m pytest tests/test_common_provenance.py tests/test_table3_aggregation.py tests/test_flight_control_comparison_formal.py tests/test_pid_controller_family.py tests/test_overload_rollrate.py -q
```

结果：`47 passed, 3 warnings`。
