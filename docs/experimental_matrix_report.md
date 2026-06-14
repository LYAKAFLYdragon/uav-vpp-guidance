# UAV-VPP-Guidance 实验矩阵设计说明

> 本文档汇总当前 `CL_CRPPO_CEMGD` 分支的实验矩阵，覆盖环境、算法、奖励函数、场景、消融与评估流程，用于论文写作与审稿回复。

---

## 1. 实验矩阵总体设计

项目采用**全配置驱动**的实验矩阵。所有环境参数、奖励权重、终止阈值、场景几何、算法开关均集中在 `config/` 下的 YAML 文件中；源代码中不硬编码任务参数。矩阵围绕一个核心科学问题展开：

> **“虚拟追踪点（VPP）+ LOS-rate 制导”这一分层架构，是否优于端到端 RL 与经典制导基线？各组件贡献多少？**

实验变量被组织为四个正交维度：

| 维度 | 可选配置 | 核心控制文件 |
|---|---|---|
| **后端环境** | SimplePointMassEnv / JSBSim F-16 | `config/env.yaml`, `config/method_innovation_jsbsim.yaml` |
| **制导架构** | VPP + guidance / End-to-end / No-VPP / Fixed-gain / Expert / Rule-based | `config/experiment/train_*.yaml` |
| **预测器** | No-pred / CV / CA / LSTM-frozen / GRU-frozen / Oracle | `config/experiment/train_vpp_ppo_*.yaml` |
| **RL 算法** | PPO / CR-PPO / Intentional-PPO / CAIS-only | `--algorithm` 参数 |

---

## 2. 环境与后端

### 2.1 SimplePointMassEnv（主要训练与机制验证后端）

- **动力学**：3-DoF 质点模型，直接积分速度/位置，无气动力矩。
- **控制周期**：高层策略 5 Hz（`high_level_dt=0.2 s`），单 episode 最大 512 步（约 102.4 s）。
- **状态接口**：统一字段 `position_neu`、`velocity_vector_mps`、`altitude_m`、`heading_rad`，与 JSBSim 后端对齐。
- **用途**：
  - 所有训练配置（200k steps）
  - 消融实验网格搜索
  - 预测器对比、几何可行性扫描
  - 机制验证（mode-switch、terminal protection、dynamics-aware VPP 等）

### 2.2 JSBSim F-16（高保真泛化后端）

- **动力学**：6-DoF F-16 飞行模型，内置气动力、发动机、舵面饱和。
- **控制周期**：底层 60 Hz，高层 5 Hz；每步运行 12 个 JSBSim 积分步。
- **数据目录**：自包含于 `data/jsbsim/`，无需外部 `JSBSIM_ROOT`。
- **用途**：
  - 训练后 checkpoint 的泛化验证
  - `config/experiment/no_prediction_vpp_jsbsim.yaml` 的固定场景评估
  - 计划中的 JSBSim 端到端训练

### 2.3 关键共享参数

```yaml
decision_freq: 5
sim_freq: 60
max_high_level_steps: 512
success_range_m: 900.0      # 成功距离
success_ata_deg: 25.0        # 成功目标方位角
success_hold_time_s: 0.2     # 持续保持时间
max_range_m: 12000.0         # 出界距离
min_altitude_m: 500.0
max_altitude_m: 15000.0
```

---

## 3. 算法体系

### 3.1 RL 主算法

| 算法 | 实现文件 | 选型方式 | 核心思想 |
|---|---|---|---|
| **PPO** | `agents/ppo_agent.py` | 默认 | 标准裁剪 PPO + 熵正则 |
| **CR-PPO** (Complexity-Regularized) | `ablations/cr_ppo/cr_ppo_agent.py` | `--algorithm cr_ppo` | 用动作维度“非均衡度”替代熵，抑制复杂度增长 |
| **Intentional-PPO** | `ablations/intentional/intentional_ppo_agent.py` | `--algorithm intentional_ppo` | ICU + IAU + 可选 Combat-Aware Intention Schedule (CAIS) |
| **CAIS-only** | `ablations/cais_only/combat_aware_schedule.py` | Intentional-PPO 子开关 | 按战斗阶段（远/中/近/终端）动态分配 actor/critic 计算预算 |

### 3.2 制导/控制架构基线

| 架构 | 动作空间 | 代表配置 | 实验意义 |
|---|---|---|---|
| **VPP + PPO（ proposed ）** | 3-D 虚拟点偏移 Δp | `train_no_prediction_vpp_ppo.yaml` | 策略输出偏移，LOS-rate 制导律映射为 nz/roll_rate/throttle |
| **End-to-end PPO** | 直接物理命令 `[nz, roll_rate, throttle]` | `train_end_to_end_ppo.yaml` | 验证“分层是否必要” |
| **No-VPP / zero-offset** | 忽略策略输出，VPP 强制锚定在目标位置 | `train_no_vpp_ppo.yaml` | 验证 VPP 偏移层的战术价值 |
| **Fixed-gain VPP** | VPP 偏移，但制导增益固定 | `fixed_gain_vpp.yaml` | 量化增益自适应的必要性 |
| **Gain-only CEM** | 策略冻结，仅 CEM 优化增益 | `gain_only_cem.yaml` | 量化策略 vs 增益各自贡献 |
| **Bilevel** | 外层 PPO + 内层 CEM 增益优化 | `proposed_bilevel.yaml` | 联合策略-增益优化 |
| **Expert VPP** | 规则化 Δp | `expert_vpp_baseline.yaml` | 可解释非学习基线 |
| **Rule-based pursuit** | 纯追踪/滞后追踪/超前追踪 | `rule_based_pursuit_baseline.yaml` | 经典制导基线 |

### 3.3 预测器变体

| 方法 | 预测器 | 锚点模式 | 训练配置 | 评估配置 |
|---|---|---|---|---|
| **No-Prediction** | 无 | `current_target` | `train_no_prediction_vpp_ppo.yaml` | 全评估配置 |
| **CV-Prediction** | 匀速 | `predicted_target` | `train_vpp_ppo_cv.yaml` | `evaluate_vpp_prediction_comparison.yaml` |
| **CA-Prediction** | 匀加速 | `predicted_target` | `train_vpp_ppo_ca.yaml` | `evaluate_vpp_prediction_comparison.yaml` |
| **LSTM-frozen** | 预训练 LSTM | `predicted_target` | `train_vpp_ppo_lstm_frozen.yaml` | `evaluate_vpp_lstm_prediction.yaml` |
| **GRU-frozen** | 预训练 GRU | `predicted_target` | `train_vpp_ppo_gru_frozen.yaml` | `evaluate_vpp_gru_prediction.yaml` |
| **Oracle** | 完美未来位置 | `oracle_future_position` | — | `stage6g3_oracle_vpp_anchor.yaml` |

---

## 4. 奖励函数设计

奖励由 `src/uav_vpp_guidance/envs/reward.py` 中的 `RewardCalculator` 计算，共 12 项：

| 奖励项 | 权重 | 设计目的 |
|---|---|---|
| `reward_range` | `w_range` | 鼓励距离进入理想区间 `[ideal_range_min, ideal_range_max]` |
| `reward_angle` | `w_angle` | 惩罚 ATA / AA，默认用 `quadratic_sum` 公式 |
| `reward_safety` | `w_safety` | 惩罚过大过载 / 滚转速率 |
| `reward_saturation` | `w_saturation` | 惩罚控制饱和 |
| `reward_smooth` | `w_smooth` | 惩罚相邻步命令抖动 |
| `reward_turn` | `w_turn_rate` | 惩罚近距离大航向误差且正在接近时的不可行转弯 |
| `reward_closing` | `w_closing` | 鼓励减小距离（range_rate 为负） |
| `reward_alive` | `w_alive` | 每步小额正奖励，防止早期出界 |
| `reward_overshoot` | `w_overshoot` | 距离小于 `ideal_range_min*0.5` 且仍在接近时惩罚过冲 |
| `reward_boundary` | `w_boundary` | 惩罚飞出 `max_range` 或极端高度 |
| `reward_potential` | PBS 参数 | 势能奖励塑形，缓解稀疏奖励平台 |
| `terminal_reward` | 固定大奖励 | 成功/坠毁/超时/出界的终端奖励 |

### 4.1 关键设计点

- **angle_reward 公式**：默认 `quadratic_sum`，可选 `max_ata_aa` 或 legacy `sum`。
- **turn_rate_penalty**：
  - 仅在 `range_min < range < range_max` 窗口内触发（默认 1000–3000 m）；
  - 仅在 **closing**（`range_rate < 0`）时触发，避免误伤合法穿越几何。
- **overshoot_penalty**：阈值默认 `ideal_range_min * 0.5`，可配置覆盖。
- **Potential-Based Shaping (PBS)**：在 `config/reward.yaml` 中默认启用，`C=0.001, gamma=0.99`，提供密集距离梯度。

### 4.2 训练配置中的典型权重

以 `config/experiment/train_no_prediction_vpp_ppo.yaml` 为例：

```yaml
reward:
  w_range: 0.6
  w_angle: 0.9
  w_energy: 0.0
  w_safety: 3.0
  w_saturation: 0.5
  w_smooth: 0.2
  w_turn_rate: 0.5
  w_closing: 0.1
  w_alive: 0.02
  w_overshoot: 1.0
  w_boundary: 0.3
  ideal_range_min: 700.0
  ideal_range_max: 1100.0
  terminal_success: 400.0
  terminal_failure: -300.0
  terminal_crash: -600.0
```

---

## 5. 场景设计

### 5.1 四类标准场景（Canonical）

所有训练与主对比实验共享同一组标准场景，目前已统一为：

| 场景 | 几何 | Ego 初始状态 | Target 初始状态 | 设计意图 |
|---|---|---|---|---|
| **favorable** | 尾追，速度优势 | `[0,0,5000]`, 280 m/s, 0° | `[800,0,5000]`, 180 m/s, 0° | 最易成功，验证基本收敛 |
| **neutral** | 迎头相遇 | `[0,0,5000]`, 200 m/s, 0° | `[2000,0,5000]`, 200 m/s, 180° | 对称态势，考验策略均衡 |
| **disadvantage** | 目标在后方且更快 | `[0,0,5000]`, 200 m/s, 0° | `[-600,400,5000]`, 220 m/s, 30° | 必须做超前转弯（lead turn） |
| **challenging** | 大横向偏移交叉 | `[0,0,5000]`, 200 m/s, 45° | `[1500,1500,5200]`, 210 m/s, 225° | 考验三维空间机动 |

> 注：`config/experiment/no_prediction_vpp_scenarios.yaml` 与 `method_innovation_comparison.yaml` 等 35+ 个文件使用上述 canonical 值。此前 `train_no_prediction_vpp_ppo.yaml` 等训练配置使用 220/190，已统一为 280/180。

### 5.2 专用/扩展场景

| 场景集 | 代表配置 | 说明 |
|---|---|---|
| 机动目标 / weaving | `stage6f5_maneuvering_target.yaml` | `target_mode: sinusoidal`，含 `weaving_pursuit` 等 |
| Hard 6-scenario benchmark | `method_innovation_comparison_hard.yaml` | 含 `crossing_close/far`, `head_on`, `tail_chase`, 更难的 `disadvantage` |
| 几何可行性扫描 | `stage6g3_geometry_feasibility.yaml` | 网格化初始距离、速度、高度差 |
| 宽几何 smoke | `stage6g5_wide_geometry_smoke.yaml` | 324 组合网格 |

### 5.3 课程学习

- **场景课程**：按 `favorable → neutral → disadvantage → challenging` 逐步解锁，每场景成功率 ≥ 50% 进入下一类。
- **成功判据课程**：`config/success_criteria/curriculum_default.yaml` 随训练进度收紧 `success_range_m`、`success_ata_deg`、`success_hold_time_s`。

---

## 6. 消融实验矩阵

### 6.1 由 `run_ablation_study.py` 支持的系统消融

| 消融类型 | 变量 | 网格大小 | 实验意义 |
|---|---|---|---|
| **reward_weights** | `w_angle`, `w_turn_rate`, `w_overshoot` | 3×3×3×3 seeds = 81 runs | 量化奖励项敏感性与最优组合 |
| **dynamics_aware** | `virtual_point.dynamics_aware: true/false` | 2×3 = 6 runs | 验证 VPP 动态可行性约束的价值 |
| **offset_range** | `dynamic_offset_scale: null/0.25/0.5/1.0` | 4×3 = 12 runs | 验证按初始距离缩放 VPP 偏移的效果 |
| **observation** | `temporal.enabled`, `include_vp_error` | 2×2×3 = 12 runs | 验证时序特征与 VPP 跟踪误差的作用 |
| **potential_shaping** | `potential_based_shaping.enabled` | 2×3 = 6 runs | 验证势能奖励塑形对稀疏奖励的影响 |

### 6.2 架构与机制消融

| 实验 | 配置/脚本 | 关键点 |
|---|---|---|
| **VPP 层消融** | `train_no_vpp_ppo.yaml` vs `train_no_prediction_vpp_ppo.yaml` | 证明 VPP 偏移不是冗余的 |
| **端到端 vs 分层** | `train_end_to_end_ppo.yaml` | 证明制导分层提升样本效率与安全性 |
| **预测器对比** | `run_ablation_matrix.py` + `evaluate_vpp_prediction_comparison.yaml` | No-pred / CV / CA / LSTM / GRU 五方法同场景同 seed 评估 |
| **安全惩罚消融** | `ablation_no_safety_penalty.yaml` | 验证 `w_safety` 对坠毁率的抑制 |
| **增益观测消融** | `ablation_no_gain_obs.yaml` | 验证策略输入中增益信息的必要性 |
| **后悔值增益更新消融** | `ablation_no_regret.yaml` | 验证 CEM-GD 后悔值更新机制 |
| **终端保护消融** | `stage6g3_terminal_protection_ablation.yaml` | 验证捕获半径、能量补偿、过载-滚转协调 |

---

## 7. 训练与评估流程

### 7.1 训练入口

| 脚本 | 对应配置 | 功能 |
|---|---|---|
| `scripts/train_curriculum_ppo.py` | `train_curriculum_ppo.yaml`, `method_innovation_comparison*.yaml` | 统一训练入口，支持 PPO/CR-PPO/Intentional-PPO，含课程学习 |
| `scripts/run_ablation_study.py` | 任意 base config | 批量运行消融网格 |
| `scripts/train_bilevel.py` | `proposed_bilevel.yaml` | 双层策略-增益优化 |
| `scripts/train_end_to_end_baseline.py` | `train_end_to_end_ppo.yaml` | 端到端基线训练 |
| `scripts/run_stage6f5_reablation.py` | `stage6f5_*.yaml` | 预测器对比全 pipeline |

### 7.2 典型训练超参数

```yaml
ppo:
  total_timesteps: 200000
  rollout_steps: 2048
  minibatch_size: 256
  update_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_coef: 0.2
  value_coef: 0.5
  entropy_coef: 0.01
  learning_rate: 3.0e-4
  max_grad_norm: 0.5
  normalize_advantage: true

policy:
  type: mlp
  hidden_sizes: [128, 128]
  activation: tanh
  action_dim: 3
```

### 7.3 评估指标

| 指标 | 说明 |
|---|---|
| `success_rate` | 终端成功率 |
| `non_crash_success_rate` | 排除坠毁后的成功率 |
| `crash_rate` / `timeout_rate` / `out_of_bounds_rate` | 失败模式分解 |
| `mean_episode_return` | 平均累积奖励 |
| `mean_time_to_success` | 成功所需时间 |
| `command_saturation_rate` | 控制饱和率 |
| `mean_final_range_m` / `mean_final_ata_deg` | 最终几何状态 |
| `score_win_rate` | 基于双方得分的胜率 |

---

## 8. 实验意义与关键点

### 8.1 科学贡献

1. **分层架构验证**：首次在近距离空战场景中系统验证 VPP + LOS-rate guidance + RL 的分层架构是否优于端到端 RL。
2. **预测增益量化**：通过 No-pred / CV / CA / LSTM / GRU / Oracle 的完整对比，明确不同预测器对各类场景的贡献。
3. **算法创新验证**：CR-PPO 与 Intentional-PPO 在空战任务上的适用性；CAIS 阶段调度机制。
4. **高保真泛化**：从 SimplePointMassEnv 到 JSBSim F-16 的迁移验证。

### 8.2 工程关键点

- **配置驱动**：所有实验参数 YAML 化，确保可复现。
- **统一后端接口**：Simple 与 JSBSim 暴露统一状态字段，几何与制导代码零改动切换。
- **课程学习**：场景难度与成功判据双课程，避免早期稀疏奖励导致训练失败。
- **审计修复**：近期统一了 favorable 场景速度、观测维度、动态 VPP 偏移缩放、turn-rate 惩罚门控等，消除数据不一致。

### 8.3 审稿人可能关注点

- **Simple vs JSBSim 的差异**：Simple 用于快速机制验证，JSBSim 用于高保真结论；所有方法在两种后端上均可复现。
- **场景一致性**：canonical four-scenario 已统一；`method_innovation_comparison_hard.yaml` 与 `stage6f5_maneuvering_target.yaml` 中的 `weaving_pursuit` 是同名不同义的专用场景，需在论文中分别说明。
- **奖励 shaping**：PBS 提供密集距离梯度，但可能引入偏差；消融实验 `potential_shaping` 可验证其影响。

---

## 9. 预期效果

| 实验 | 预期效果 |
|---|---|
| **No-Prediction VPP PPO** | favorable 成功率 > 90%，neutral/disadvantage 逐步提升，challenging 最具挑战性 |
| **CV/CA Prediction** | 在 disadvantage/challenging 等需要提前机动的场景比 no-pred 提升明显 |
| **LSTM/GRU-frozen** | 在机动目标场景优于参数化预测器，但训练与推理成本更高 |
| **CR-PPO** | 相比 PPO 降低策略复杂度，动作更平滑，饱和率更低 |
| **Intentional-PPO + CAIS** | 在终端阶段分配更多计算/探索预算，提升最终捕获精度 |
| **End-to-end** | 作为负面对照，成功率与样本效率低于分层 VPP |
| **No-VPP** | 证明 VPP 偏移层对 close-range 机动不可或缺 |
| **JSBSim 迁移** | 绝对指标下降，但方法间相对排序保持一致 |

---

## 10. 复现命令速查

```bash
# 标准 No-Prediction VPP PPO 训练
python scripts/train_curriculum_ppo.py \
    --config config/experiment/train_no_prediction_vpp_ppo.yaml \
    --algorithm ppo --seed 0

# 奖励权重消融
python scripts/run_ablation_study.py --type reward_weights \
    --config config/experiment/train_no_prediction_vpp_ppo.yaml \
    --steps 50000 --seeds 3 --device cuda

# 预测器对比评估
python -m uav_vpp_guidance.evaluation.evaluate_prediction_comparison \
    --config config/experiment/evaluate_vpp_prediction_comparison.yaml \
    --backend simple --episodes 50 --seeds 0 1 2

# JSBSim 泛化验证
python -m uav_vpp_guidance.evaluation.evaluate_policy \
    --config config/experiment/train_no_prediction_vpp_ppo.yaml \
    --checkpoint outputs/experiments/no_prediction_vpp_ppo/checkpoints/best.pt \
    --backend jsbsim --episodes 5 --seeds 0
```

---

*报告生成时间：2026-06-14*  
*基于分支：CL_CRPPO_CEMGD @ e47033f*
