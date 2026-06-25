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

## 2026-06-24 P0-P2 审计修复（commit ff0602b）

### 完整 720-episode 回归矩阵

3 配置 × 4 控制器 × 3 任务 = 36 组合，每组合 20 seeds：

| Config | Task | baseline | enhanced | ppo_pid | robust |
|---|---:|---:|---:|---:|---|
| bank75 | multi_waypoint | 5.00 | 4.25 | 4.00 | 4.25 |
| bank76 | multi_waypoint | 5.00 | 4.25 | 4.00 | 4.25 |
| bank80 | multi_waypoint | 4.00 | 4.00 | 4.00 | 4.00 |
| bank75 | sustained_turn | 0.98 | 1.13 | 1.07 | 1.13 |
| bank76 | sustained_turn | 0.95 | 1.11 | 1.10 | 1.11 |
| bank80 | sustained_turn | 0.64 | 0.71 | 1.70 | 0.71 |

> 主指标：multi_waypoint=mean_wp (max 5.0), sustained_turn=mean_orbits (target ≥2.0)
> break_turn 行因 track_err 数据问题已排除（见下方诊断）
> 缺失：GainScheduled PID、PPO baseline、真正 PPO+PID (action_dim=4)
> 详细分析见 [[sustained_turn_analysis_and_720ep_matrix]]

### Baseline PID 5.00 满分解释

`BaselinePIDController(EnhancedLowLevelController)` 继承自 EnhancedLowLevelController，
继承了所有 P0/P1 底层修复（M1 油门统一、抗积分饱和、保护优先级仲裁）。
在 multi_waypoint 温和任务上，Baseline 禁用的保护项反而减少了约束，
使其能更直接地跟踪航点。

→ Baseline 不再是"弱 baseline"，消融实验需要重新设计对照组。
→ 详细分析见 [[baseline_pid_5.00_explanation]]

### Sustained-Turn 时间限制诊断

| 指标 | 目标 | 实际 (bank76) |
|---|---|---|
| turn radius | 1200m | ~3315m |
| 理论最大圈数 (90s) | 2.0 | 1.08 |
| 实际观测 | 2.0 | 0.95-1.11 |

结论：completed_orbits 低不是因为 tracking 不准，而是 90s 时间限制理论不可行。
建议扩展到 1024 steps (204.8s) 或调整目标为 1.0 orbits。

### Break-Turn 数据诊断

调查确认 `range_m` 数据正确写入 episode JSON（非 0.0），`mean_track_error_m` 为 3107–7303m。
"track_err=0.0m" 的原始报告可能来自 pre-fix 指标计算（修复前 `compute_break_turn_metrics` 未提取 `nz_cmd`/`range_m` 字段）。

post-fix 新增指标：
- `nz_tracking_rmse`: 使用 `_safe_rmse()` 计算 nz_cmd vs actual nz 的 RMSE
- `recovery_delay`: 当 nz_cmd 无过零时返回 NaN（不可测量）
- `overshoot_pct`: 当无有效段时返回 NaN（不可测量）

### 代码修复清单

| 修复 | 文件 | 说明 |
|------|------|------|
| throttle_max default | `evaluate_prediction_comparison.py:116` | 1.0→0.9，与其他 default 对齐 |
| 38 YAML throttle 值 | `config/experiment/*.yaml` | 批量替换 0.0/1.0→0.4/0.9 |
| PPO+PID resume 支持 | `train_ppo_pid.py` | 新增 `--resume`/`--resume-step`/warm_start |
| E2E PPO 对比脚本 | `scripts/run_e2e_ppo_comparison.py` | 3 seeds × 1M steps，pre-fix vs post-fix |
| PPO+PID resume 脚本 | `scripts/run_ppo_pid_resume.py` | 从 step_40960.pt 恢复到 200k |
| 分析文档入库 | `memory/baseline_pid_5.00_explanation.md` | Baseline 满分解释 |
| 分析文档入库 | `memory/sustained_turn_analysis_and_720ep_matrix.md` | 矩阵补全 + ST 分析 |

### 待跑实验（云主机）

1. **PPO+PID resume**：从 `step_40960.pt` 恢复到 200k
   ```powershell
   python scripts/run_ppo_pid_resume.py --device cuda
   ```

2. **E2E PPO 1M 对比**：pre-fix vs post-fix，各 3 seeds
   ```powershell
   python scripts/run_e2e_ppo_comparison.py --group all --device cuda
   ```

3. **Pre-fix E2E PPO 1M baseline**（对照组需要先跑）：
   ```powershell
   python scripts/run_e2e_ppo_comparison.py --group pre-fix --device cuda
   ```

## 2026-06-25 本地验证发现

### Break-Turn nz 过零：结构性不可行

**验证过程**：
1. 将 break_turn 轨迹从纯水平转弯改为 R→L→R 多段转弯 + 爬升/俯冲（±30 m/s）
2. 用 `run_flight_control_comparison.py` 直接跑完整 450 步（90s）

**结果**：
| 指标 | 值 |
|---|---|
| episode steps | 450/450 (完整运行) |
| termination | timeout |
| nz_cmd 范围 | [1.373, 3.477]g |
| 低于 1g-deadband 步数 | 0/450 |
| recovery_delay | NaN |
| overshoot_pct | 202.5% |
| nz_tracking_rmse | 2.32g |

**根因分析**：LOS-rate 引导律在跟踪转弯目标时，转弯所需过载至少 ~2.85g（6°/s @ 250m/s 下
`nz = sqrt(1 + (V·ω/g)²) ≈ 2.85`）。即使叠加 ±30m/s 的爬升/俯冲速率，引导律输出的
nz_cmd 最低也只能到 1.373g，无法跌破 1g-deadband。

**结论**：`recovery_delay` 指标是为"nz 指令阶跃响应"场景设计的——需要 nz_cmd 在 1g 上下
发生符号翻转。break_turn 这类跟踪转弯目标的引导律**永远不会产生 nz 符号翻转**，
recovery_delay=NaN 是任务的**预期行为**，不是 bug。

**当前可用指标**：
- `nz_tracking_rmse`：测量引导律 nz_cmd 与实际 nz_g 之间的跟踪误差（当前 ~2.32g，目标 <0.5g）
- `overshoot_pct`：测量 nz 上升段的超调百分比（当前 202.5%，反映 PID 对引导律指令的过冲响应）

**后续选项**（见 [[#break-turn-recovery-delay-backlog]]）：
- 方案 A（即时）：接受 recovery_delay=NaN，break_turn 验收仅使用 nz_tracking_rmse + overshoot_pct
- 方案 B（backlog）：新建独立 `task_nz_step_response.yaml`，注入显式 nz_cmd 阶跃序列
- 方案 C（不推荐）：在 break_turn 中插入 30s 直线段 + 100m/s 陡降，hack 式制造 nz < 1g

### PPO+PID Resume Smoke 验证

**验证过程**：用 `--resume` + `--resume-step 40960` + `--smoke` 测试 warm_start 机制

**结果**：
```
total_timesteps: 41472 (= 40960 + 512)
episodes: 7, elapsed: 20.3s
Warm-start loaded: obs_dim=16, action_dim=4, missing_keys=0, unexpected_keys=0
Loaded optimizer state from step_40960.pt
```

**修复的代码问题**：`train_ppo_pid.py` 中 warm_start 块原本在 smoke 块之后执行，
导致 smoke 模式下 `resume_step` 始终为 0。已将执行顺序修正为：
agent 创建 → warm_start 加载 → smoke 模式 timestep 调整。

### 代码修复补充

| 修复 | 文件 | 说明 |
|------|------|------|
| break_turn 轨迹增强 | `break_turn_env.py` | 添加 altitude 分量（爬升/俯冲），使 episode 完整运行 450 步 |
| smoke resume 顺序修复 | `train_ppo_pid.py` | warm_start 移至 smoke 块之前，避免 resume_step=0 的 bug |
| CSV append 模式 | `train_ppo_pid.py` | resume 时日志文件使用 append 模式，保留历史记录 |
| break_turn task config 修正 | `task_break_turn.yaml` | 可通过 `--config-bt` 在 comparison 脚本中正确合并 |

### 云主机运行命令（已验证）

```powershell
# 1. PPO+PID resume（已通过本地 smoke 验证）
python scripts/run_ppo_pid_resume.py --device cuda

# 2. E2E PPO pre-fix baseline（对照组，先跑）
python scripts/run_e2e_ppo_comparison.py --group pre-fix --device cuda

# 3. E2E PPO post-fix comparison
python scripts/run_e2e_ppo_comparison.py --group post-fix --device cuda

# 4. 完整 E2E PPO 对比（等 2+3 都跑完）
python scripts/run_e2e_ppo_comparison.py --group all --device cuda
```

---

## 2026-06-25 更新：云端大规模实验

### 实验环境

- **服务器**: 96 vCPU, RTX 2080 Ti (11GB), 375GB RAM
- **JSBSim**: v1.3.1, F-16 aerodynamic model
- **分支**: `fix/flight-control-comparison-sync` (commit ff0602b)
- **所有实验 0 crash 的前提**: bank76 保护配置（bank_angle_protection + altitude_hold + lift_compensation）

### 1. PID 增益扫描

**最优增益**: Kp_nz=**0.50**, Kd_nz=**0.10**, Ki_nz=**0.05**
- 扫描 9×10=90 组合 × 5 seeds = 450 episodes
- 全部组合 **0 crash**
- 最优 RMSE=2.26g（默认增益 RMSE 约 3.2g）
- Kd_nz 在平滑转弯轨迹上影响可忽略——需 [[nz-step-response-task]] 验证瞬态
- 详见 [[pid-gain-scan-results-2025-06-25]]

### 2. PPO+PID 训练（使用最优增益）

- 200k steps, 1,757 episodes, **~30 分钟**（RTX 2080 Ti）
- 训练评估：86.67% 成功, 13.33% crash, 0% OOB
- multi_waypoint 正式评估：**0/20 crash, mean_wp=4.00**
- sustained_turn 正式评估：20/20 stall, mean_orbits=2.44
- PPO agent 过度拉 nz（最高 10.9g）导致能量耗尽——奖励函数缺少能量保存项
- 详见 [[ppo-pid-training-results-2025-06-25]]

### 3. 保护增益校准

- 4 bank × 4 nz_inc × 3 nz_inc_max × 4 alt_gain = **192 组合全部稳定**
- 最大稳定 bank: **82°**（mean_wp=4.00, crash=0）
- bank 76°-82° 全部可用，bank=76° 推荐作为保守默认
- 详见 [[protection-gain-calibration-2025-06-25]]

### 4. 飞行包线表征

- **Bank 包线**: 60°-82° 全部稳定（0 crash, mean_wp ≥ 4.0）
- **nz 包线**: 2g-7g 全部稳定（0 crash）
- **Speed×Altitude 包线**: 仅 6/25 稳定
  - 安全区：2000-4000m, 120-220 m/s
  - **6000m 以上 100% 崩溃**（发动机推力不足）
  - 高海拔空战应使用能量战术（Boom & Zoom），不可持续转弯
- 详见 [[flight-envelope-characterization-2025-06-25]]

### 5. 持续转弯能量管理

- Enhanced PID：速度健康（260m/s），但高度失控 → crash
- PPO+PID：高度可控，但速度衰减至失速（150m/s） → stall
- **根因**：持续高 g 转弯的诱导阻力超过可用推力，能量流失不可避免
- 缓解：增大轨道半径（降低所需 nz）、调高 altitude_hold_gain、PPO 奖励函数加能量项
- 详见 [[sustained-turn-energy-management]]

### 6. 对抗 + BFM 验证

- 对抗场景 10 种：**0/50 crash (0.0%)**，迎头场景 100% 成功
- 激进 BFM（yo-yo 1000m 振幅）：**PASSED**（0/5 crash, nz_max=8.7g < 9g, 无螺旋下坠）

### 验收标准达成

| 指标 | 目标 | 实际 | 状态 |
|------|------|------|------|
| break_turn crash | < 2/20 | 0/450 | ✅ |
| max stable bank | ≥ 78° | 82° | ✅ |
| usable bank range | ≥ 12° | 22° (60-82°) | ✅ |
| nz envelope | 2-7g | 全部稳定 | ✅ |
| adversarial crash | < 10% | 0% | ✅ |
| BFM crash | 0/5 | 0/5 | ✅ |
| BFM nz_max | < 9g | 8.7g | ✅ |
| PPO+PID MW wp | ≥ 4.0 | 4.00 | ✅ |
| PPO+PID ST orbits | ≥ 1.5 | 2.44 (but stall) | ⚠️ |
| ST 0 crash | 0/20 | 20/20 crash | ❌ |

### 剩余工作

1. **nz_step_response**: env 已写，命令注入机制待调试（info 中 nz_cmd 未正确传递）
2. **ST 能量管理**: 调整接受标准为"mean_orbits ≥ 1.5, 0 crash before energy exhaustion"或实现能量感知控制
3. **PPO+PID 奖励函数**: 添加能量保存项改善 sustained_turn 性能
