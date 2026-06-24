# 对话经验与思路存档：multi-waypoint crash 消融与底层飞控评估

**日期**：2026-06-23  
**对话主题**：排查 multi-waypoint 任务中的 crash 问题，并通过多轮 ablation 找到可用的底层飞控配置。

---

## 1. 背景与目标

在 `fix/flight-control-comparison-sync` 分支上，multi-waypoint tracking 任务中 fixed PID 控制器出现大量 crash。目标是：

- 定位 crash 根因；
- 通过 ablation 找到 **0 crash 且航点完成度尽可能高** 的底层飞控配置；
- 判断当前底层飞控是否足以支撑上层空战机动决策研究。

---

## 2. 根因诊断

- **直接原因**：固定 PID 控制器在第一个航点切换后进入 **螺旋下坠（spiral dive）**，最终触发高度下限 crash。
- **升降舵极性已确认**：负的 `fcs/elevator-cmd-norm` 对应抬头/爬升，控制器前馈符号正确。
- **崩溃统计（修正前）**：
  - Enhanced/Robust PID：13/20 crash
  - Baseline PID：12/20 crash
  - PPO+PID：6/20 crash

---

## 3. 已应用的代码修复

1. **`MultiWaypointTrackingEnv` 终止逻辑修正**
   - 仅当 `completed_waypoints >= len(waypoints)` 时才标记 `task_success` / `is_success`。
   - 区分 true success、incomplete、crash 三种终止原因。

2. **`observation.py` 增加溢出/NaN guard**
   - 观测向量先 `np.isfinite` 替换，再 `clip(-1e6, 1e6)`。
   - 部分 run 仍有 observation overflow 警告，根因（NaN target states 传播）未完全消除。

3. **`CommandPostProcessor` 增加平滑升力补偿**
   - 支持 `lift_compensation_source`（当前用 `actual_roll`）和 `lift_compensation_filter_alpha`（一阶低通）。
   - 文件：`src/uav_vpp_guidance/guidance/overload_rollrate.py`

4. **默认配置更新**
   - `config/guidance.yaml` 增加 `lift_compensation_source`、`lift_compensation_filter_alpha` 等默认值。

---

## 4. 消融实验历程

### v2：已有组合的横向对比

| 配置 | Enhanced/Robust crash | Enhanced/Robust mean_wp | PPO+PID crash | PPO+PID mean_wp |
|---|---|---|---|---|
| baseline_fixed | 高 | 低 | 低 | 3.20 |
| bank75_alt_hold | **0** | 2.75 | 0 | 2.00 |
| lift_comp（实际滚转角） | 无改善 | — | — | — |
| k_roll_low | 无改善 | — | — | — |
| all_fixes | 比 bank75 差 | — | — | — |

**结论**：`bank75_alt_hold` 是当时 Enhanced/Robust PID 的最佳配置。

### v3：新增两项尝试

- **平滑升力补偿（actual_roll + smoother）**
- **80° bank-angle 保护**

| 配置 | Enhanced/Robust crash | Enhanced/Robust mean_wp | PPO+PID crash | PPO+PID mean_wp |
|---|---|---|---|---|
| bank75_alt_hold（参考） | 0 | 2.75 | 0 | 2.00 |
| bank75_alt_hold_lift_smooth | 0 | **2.95** | 0 | 2.00 |
| bank80_alt_hold | 6 | **3.40** | 3 | 2.35 |

**结论**：
- 平滑升力补偿在 0 crash 前提下把完成度从 2.75 提升到 2.95。
- 80° bank 提升完成度到 3.40，但重新引入 crash。

### v4：75°–80° 细粒度扫描

所有配置均带 `altitude_hold` + 平滑升力补偿，仅调整 `max_bank_rad`。

| bank | Enhanced/Robust crash | Enhanced/Robust mean_wp | PPO+PID crash | PPO+PID mean_wp |
|---|---|---|---|---|
| 75° | 0 | 2.95 | 0 | 2.00 |
| **76°** | **0** | **3.10** | 1 | 2.00 |
| 77° | 3 | 3.00 | 1 | 2.00 |
| 78° | 5 | 3.00 | 3 | 2.00 |
| 79° | 5 | 3.00 | 2 | 2.35 |
| 80° | 8 | 3.00 | 3 | 2.35 |

**结论**：**76° 是 sweet spot**——0 crash 下完成度最高（3.10 / 5）。

---

## 5. 推荐配置

用于上层空战机动决策研究的默认底层配置：

```yaml
# config/experiment/ablation_mw_bank76_alt_hold_lift_smooth.yaml
includes:
  - ablation_mw_bank75_alt_hold_lift_smooth.yaml

low_level_controller:
  max_bank_rad: 1.3264502315156905   # 76 deg
```

展开后等效于：

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

---

## 6. 踩坑记录

### 6.1 YAML include 路径陷阱

`load_experiment_config` 解析 include 时：

- 先尝试 `config/experiment/<include_path>`；
- 不存在时 fallback 到 `config/<basename(include_path)>`。

因此形如 `experiment/ablation_mw_bank75_alt_hold.yaml` 的嵌套 include 会被静默跳过，导致基配置未加载，所有控制器变成默认任务，结果全是 `out_of_bounds`。

**解决办法**：同目录 include 直接写文件名，例如 `ablation_mw_bank75_alt_hold.yaml`。

### 6.2 云同步心跳超时

- `scripts/_cloud_sync_current.py` 在 120s 前台超时失败。
- 改为后台 + 300s 超时后成功（约 2m 40s）。

### 6.3 并行 worker 上限

- 云端稳定运行 4 workers；8 workers 会触发 `BrokenProcessPool`（JSBSim C-extension / spawn 问题，非 OOM）。

### 6.4 环境变量必须设置

```bash
export JSBSIM_ROOT=/root/jsbsim_root
export KMP_DUPLICATE_LIB_OK=TRUE
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

---

## 7. 实验工作流

已沉淀的云端 serial ablation 流程：

1. 本地编辑代码 / 配置。
2. `python scripts/_cloud_sync_current.py` 同步到云端。
3. `python scripts/_cloud_ablation_mw_runner_v4.py` 在云端串行跑各配置（20 seeds，4 workers）。
4. 下载 `outputs/flight_control_compare/<run_id>/aggregate/episode_records.json`。
5. 用 `results/ablation_mw_v*_summary.json` 汇总对比。

---

## 8. 对上层设计的判断

**结论**：当前底层飞控 **足以支撑上层空战机动决策的初步研究**，但有明确边界。

### 足够支撑的理由

- Enhanced/Robust PID 在 76° 配置下 **0 crash**，平均完成 3.1/5 航点。
- 航点切换不再失控，稳定性可预期。
- 75–76° 是安全区，配置边界清晰。

### 尚未验证、需要回头再研究的点

1. **高动态空战机动**：当前只是航点跟踪，未涉及剪刀机动、大过载转弯、导弹规避等。
2. **对抗性目标**：目标速度固定 100–120 m/s，无规避行为。
3. **PPO+PID**：完成度偏低（2.00），且需要单独调参；若上层基于 PPO，必须重新评估底层。
4. **Baseline PID**：12/20 crash，不建议用于上层决策实验。
5. **observation overflow 根因**：guard 已加，但 NaN target states 传播问题未完全消除。

---

## 9. 后续检查清单

等上层设计告一段落，回到底层飞控时优先做：

- [ ] 在 break-turn / sustained-turn 任务上验证 `bank76_alt_hold_lift_smooth`。
- [ ] 在带对抗目标的场景下验证底层稳定性。
- [ ] 为 PPO+PID 单独做坡度保护/no-bank 扫描。
- [ ] 定位并消除 observation overflow 的 NaN 来源。
- [ ] 评估是否需要更高频低层控制或增益调度。

---

## 10. 相关文件索引

| 文件 | 说明 |
|---|---|
| `src/uav_vpp_guidance/guidance/overload_rollrate.py` | 平滑升力补偿实现 |
| `src/uav_vpp_guidance/envs/multi_waypoint_tracking_env.py` | 终止逻辑修正 |
| `src/uav_vpp_guidance/envs/observation.py` | 观测溢出 guard |
| `config/guidance.yaml` | 默认 guidance 配置 |
| `config/experiment/ablation_mw_bank76_alt_hold_lift_smooth.yaml` | 推荐配置 |
| `config/experiment/ablation_mw_bank75_alt_hold_lift_smooth.yaml` | 保守备选配置 |
| `scripts/_cloud_ablation_mw_runner_v4.py` | v4 细粒度扫描 runner |
| `scripts/_cloud_sync_current.py` | 本地→云端同步脚本 |
| `results/ablation_mw_v4_summary.json` | v4 汇总结果 |
| `results/ablation_mw_v3_summary.json` | v3 汇总结果 |
| `memory/2026-06-23-flight-control-archive.md` | 底层飞控阶段性结论 |
| `memory/2026-06-23-conversation-archive.md` | 本对话存档 |
