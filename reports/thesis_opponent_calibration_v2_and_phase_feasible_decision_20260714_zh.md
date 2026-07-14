# 对手 Calibration v2 与 Phase-Feasible Disadvantage 决策

**日期：** 2026-07-14
**范围：** evaluation-only；无训练、无调参、无 canonical 资产修改。

## 1. 对手 Calibration / Capability Card v2

共同参考控制器固定为 `legacy_static_oracle_task_gate`，共同场景固定为
`THESIS-FIVE-STATE-HELDOUT40-V1` 的 40 个 JSBSim 场景。每个对手仅报告自己
的 40 条 reference episode；不合并统计量，也不输出 Elo、总强度或传递性排名。

| 对手 | W/L/D | 平均 HP 优势 | 目标攻击区时间 (s) | 我方攻击区时间 (s) | 净目标压力 (s) |
|---|---:|---:|---:|---:|---:|
| `expert` | 29 / 11 / 0 | 16.5375 | 4.865 | 11.480 | -6.615 |
| `end_to_end` | 28 / 11 / 1 | 17.6375 | 4.325 | 11.380 | -7.055 |
| `independent_ppo_vpp` | 30 / 10 / 0 | 16.9625 | 4.115 | 10.900 | -6.785 |

完整卡片还保留初始态势分层、终端原因、首次攻击区进入时间、目标预合并速度/高度/
比能与 post-merge 占比。它解决的是“对手条件没有量化描述”的问题，而不是证明哪个
对手具有全局最高强度。各指标都依赖共同参考控制器和 Heldout40 包线。

**权威输出：**
`E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/heldout40_r1/opponent_calibration_v2_r1/`。

## 2. Phase-Feasible Disadvantage v1

新包线保持冻结的 `disadvantage` ATA/AA 几何定义，采用目标机更快的追越初始条件；
12 个场景由 3 个距离/速度包、2 个高度条件和 2 个镜像方向组成。共同参考控制器不变，
每个冻结对手独立运行 12 条 strict-JSBSim episode，时域为 240 个高层 step。

预注册 gate 要求每个对手至少 50% episode 同时具有：

- 至少 10 个 post-merge step；
- 连续条件下至少 10 个 re-entry step；
- 无 backend fallback。

| 对手 | 合格 episode | 比例 | Gate |
|---|---:|---:|---|
| `expert` | 6 / 12 | 0.500 | PASS |
| `end_to_end` | 4 / 12 | 0.333 | FAIL |
| `independent_ppo_vpp` | 5 / 12 | 0.417 | FAIL |

**正式决策：** `phase_feasible_envelope_not_established`。该 v1 包线证明了部分 first-pass、
post-merge 与 re-entry 轨迹在 JSBSim 中物理可达，但不能在三个对手下稳定覆盖。因此它不能
授权四共享技能训练、combat finetune、高层 PPO 或论文中关于 re-entry 技能库的性能声明。

**权威输出：**
`E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/phase_feasible_disadvantage_v1_r1/analysis_r1/`。

## 3. 后续约束

1. 冻结此 v1 负证据；不得以结果为依据修改其场景、seed、VPP、制导、PID 或 checkpoint 后重跑。
2. 若未来继续探索，只能新建独立、预注册的 phase-reachability v2 方案，并先通过三对手的
   phase gate，再讨论低层技能训练。
3. 论文中可称“使用三种冻结对手条件进行分层能力报告”，不可称“已建立对手强度排名”。
