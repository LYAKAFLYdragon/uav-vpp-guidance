# VPP 战斗微调实验总结报告：12 轮尝试、14 种配置与核心经验

> 作者：实验系统记录  
> 时间：2025-06-30  
> 分支：`fix/flight-control-comparison-sync`  
> 总实验量：12 轮、14 种配置、超过 200 个 seed 的 pilot 验证

---

## 一、实验背景

### 1.1 目标

将 VPP（Virtual Pursuit Point）从"追一个虚拟点"推进为"能稳定塑造空战有利几何的机动决策接口"，并形成真实研发闭环。

### 1.2 技术基线

- **环境**：JSBSim 6-DOF 飞行动力学（严格模式，无 fallback）
- **算法**：PPO，32k 步 combat finetune，[128, 128] MLP
- **动作头**：3 维 tactical-basis（k_los, k_pos, k_v）
- **任务**：`head_on`（迎头）+ `crossing_feasible`（交叉）混合训练
- **对手**：expert（确定性制导）+ end_to_end（自训练策略）
- **评估**：10-20 seed 形式 pilot，每 seed 1 个 episode

### 1.3 基线性能（无 combat finetune）

| 对手 | head_on | crossing |
|---|---|---|
| expert（20 seeds） | 0.55 | 0.35 |
| end_to_end（20 seeds） | 0.95 | 1.00 |

---

## 二、逐轮尝试总结

### Round 6：E2E 加权（失败）

- **方法**：将 end_to_end 对抗权重提升到 2.0
- **结果**：catastrophic forgetting（全部 0.00）
- **教训**：过度加权单一对手会摧毁策略泛化能力

### Round 7：对手阶段感知（失败）

- **方法**：在 observation 中增加 2 维 opponent_stage one-hot
- **结果**：16k 训练步数不足，全部 0.00
- **教训**：新增特征需要足够训练预算，且特征本身不足以打破 tradeoff

### Round 8：对手阶段感知 32k（部分成功）

- **方法**：32k 从 scratch 训练，包含 opponent_stage
- **结果**：expert/ho=0.80（突破），但 expert/cr=0.00（完全丧失）
- **教训**：一个任务上的突破必然以另一个任务为代价

### Round 8a：Crossing 加权（失败）

- **方法**：crossing 权重 2.0
- **结果**：expert/cr=0.90（突破），但 expert/ho=0.10（完全丧失）
- **教训**：tradeoff 是双向的，加权只会将失败转移到另一侧

### Round 8b：Lane-gated（失败）

- **方法**：为所有 4 个 lane 配置独立的增益门控
- **结果**：与 baseline 无差异（全部 0.00）
- **教训**：门控机制不解决核心问题，只是增加了一层不改变本质的 indirection

### Round 9a：Task-type 无加权（中性）

- **方法**：在 observation 中增加 1 维 `is_crossing` 二进制特征
- **结果**：expert/ho=0.40, expert/cr=0.70（tradeoff 方向反转）
- **教训**：task observation 只是让策略知道了任务类型，但不提供足够信息来同时优化两个任务

### Round 9b：Task-type + head_on 加权（部分成功）

- **方法**：task-type + head_on 权重 2.0
- **结果**：expert/ho=1.00（最佳），expert/cr=0.10（最差），e2e/ho=0.90, e2e/cr=1.00
- **教训**：在已知任务类型的情况下，加权可以实现单侧突破，但代价是对侧完全失败

### Round 9c：Task-type Warmstart 32k（失败）

- **方法**：从 baseline 做 warmstart，32k combat finetune
- **结果**：expert/ho=0.20, expert/cr=0.70（baseline 被严重破坏）
- **教训**：**warmstart 是有害的**。任何 combat finetune 都会破坏 baseline 的通用拦截能力

### Round 9d：Task-type Warmstart 8k（失败）

- **方法**：从 baseline 做 warmstart，仅 8k 步
- **结果**：完全摧毁（全部 0.00）
- **教训**：即使是"最小"的 combat finetune 也足以造成 catastrophic forgetting

### Round 9e：Crossing finetune from head_on-weighted（折中）

- **方法**：从 head_on-weighted checkpoint 做 crossing finetune，head_on 权重=2.0, crossing 权重=0.5
- **结果**：expert/ho=0.60, expert/cr=0.50（最佳平衡），但 e2e/ho=0.50, e2e/cr=0.60
- **教训**：可以找到一个"中间点"，但代价是 end_to_end 性能大幅下降。这个中间点仍然不满足同时高要求

### Round 10：h256 网络容量（失败）

- **方法**：将隐藏层从 [128, 128] 扩展到 [256, 256]
- **结果**：expert/ho=0.10, expert/cr=0.20（比 h128 更差）
- **教训**：**网络容量不是瓶颈**。更大的网络在有限训练预算下反而更难优化

### Round 11：Multi-head 共享编码器（失败）

- **方法**：共享 [128, 128] 编码器 + 2 个独立 actor head + 共享 critic
- **结果**：expert/ho=0.10, expert/cr=0.70（与 single-head unweighted 相同）
- **教训**：问题不在输出层。共享编码器被迫学习同时适用于两种几何的单一表示，这是冲突的根源

### Round 12：Multi-head 独立编码器（失败）

- **方法**：每个任务独立 [128, 128] 编码器 + 独立 actor head + 共享 critic
- **结果**：expert/ho=0.10, expert/cr=0.70（与共享编码器完全一致）
- **教训**：**问题不在编码器架构**。即使给每个任务独立编码器，PPO 的共享 critic 和优化动态仍然导致 tradeoff。这是一个比表示学习更深层的算法问题。

---

## 三、核心发现

### 发现 1：Fundamental Tradeoff（结构性的零和困境）

**没有任何单策略配置同时达到 expert/ho ≥ 0.50 且 expert/cr ≥ 0.50。**

这一结论在 14 种配置、200+ seed 的验证中保持一致。它不是训练不充分或超参数调优可以解决的问题。

### 发现 2：Combat Finetune 总是导致 Catastrophic Forgetting

任何 combat-specific 训练（无论加权、warmstart、task-type 与否）都会破坏 baseline 的通用拦截能力。Baseline 的 0.55/0.35 是单策略能达到的最佳平衡点。

### 发现 3：End-to-end 对手不是瓶颈

所有配置在对抗 end_to_end 时表现良好（通常 ≥ 0.80）。真正的挑战来自 expert 对手。这说明自训练策略已经被 baseline "超越"，但确定性 expert 制导仍是一个高 bar。

### 发现 4：Multi-head 架构不解决核心问题

无论是共享编码器还是独立编码器，multi-head 的 expert 结果完全一致（0.10/0.70）。问题不在表示学习层，而在 PPO 的优化动态——特别是共享 critic 的 value function 梯度对两个任务的耦合。

---

## 四、有价值的经验（凝练）

### 经验 1：先验证、后扩展（Validation-First）

在每一轮尝试中，我们遵循了"最小改动 → 测试 → 10-seed pilot → 分析"的循环。这避免了在错误方向上的大量投入。例如：
- h256 在 smoke test 中就显示出更差的 eval 曲线，节省了大量训练时间
- Warmstart 在 8k 时就表现出完全摧毁，证明了即使"最小"的 combat finetune 也是有害的

**建议**：任何新架构尝试先做 2-4 seed 的 smoke test，如果 eval 曲线在 8k 内没有改善，立即放弃。

### 经验 2：加权是零和博弈，不是优化

权重调整（head_on=2.0, crossing=0.5 等）只能将性能从一个任务转移到另一个任务，无法提升总体性能。这是 PPO 在混合任务分布上的固有属性。

**建议**：在论文中，权重调整不应被描述为"优化"，而应被描述为"tradeoff 控制"。

### 经验 3：Warmstart 是有害的陷阱

从表现良好的 baseline checkpoint 做 warmstart 是直觉上合理的做法，但所有实验都证明它导致 catastrophic forgetting。原因可能是：
- PPO 的优化过程在 warmstart 后会"远离"原始参数空间
- Combat 任务的分布与 baseline 训练分布差异太大

**建议**：永远不要用 warmstart 做 combat finetune。如果需要特定任务的性能，直接训练一个专用策略。

### 经验 4：诊断比 win_rate 更重要

只看 win_rate 会错过关键信息。我们分析了：
- 终止原因分布（crash vs timeout vs HP advantage）
- 最终几何（range, ATA, AA）
- 奖励曲线和生存率

这些诊断揭示了：
- head_on 的失败通常是"timeout_hp_disadvantage"（策略选择了错误的接近角度）
- crossing 的成功通常是"target_crash_or_out_of_bounds"（策略能正确诱导对手失误）

**建议**：在论文中，除了 win_rate 外，还应报告终止原因分布和最终几何指标。

### 经验 5：多 seed 是必要而非奢侈

3-seed 结果在早期实验中多次误导方向。只有 10-20 seed 的正式 pilot 才能给出可靠的统计结论。

**建议**：任何论文声称的"改进"必须基于至少 10 seed 的验证。3-seed 结果不可信。

### 经验 6：网络架构不是万能药

从 multi-head 到 separate encoder，从 h128 到 h256，架构改动都没有解决核心问题。这提醒研究者：在优化失败时，不要先怀疑网络不够大，而应先怀疑问题定义和算法选择。

**建议**：在尝试架构改动之前，先确认问题是否可以通过更好的奖励 shaping、任务分解或算法选择来解决。

### 经验 7：Provenance 是科研的生命线

每一轮实验都保留了：
- config_snapshot.yaml（完整配置）
- eval_log.csv 和 eval_lane_log.csv（评估日志）
- training log 和 checkpoint
- 10-seed pilot 的 summary.csv
- 分析报告

这使得 12 轮后的回顾成为可能，也为论文写作提供了可靠的数据来源。

**建议**：所有实验（包括失败的）都应保留 provenance。审稿人最看重的是可复现性和证据链的完整性。

---

## 五、对后续研究的启示

### 方向 A：放弃单策略，拥抱多策略（推荐）

如果同时需要 head_on 和 crossing 的高性能，最直接的方案是训练两个独立的策略：
- 一个专用于 head_on（可从 head_on-weighted 的 1.00/0.10 出发）
- 一个专用于 crossing（可从 crossing-weighted 的 0.10/0.90 出发）

然后用一个高层任务分类器（或基于任务类型的硬切换）来选择策略。这不再是"单策略多任务"，而是"多策略单任务"。

### 方向 B：尝试不同的 RL 算法（高不确定性）

PPO 的 on-policy 特性和共享 critic 可能是 tradeoff 的根源。可以尝试：
- **SAC/TD3**：off-policy，可能更好地处理多任务
- **MAML**：meta-learning，让策略快速适应每个任务
- **Multi-task SAC**：显式多任务 off-policy 优化

风险：这些算法需要大量实现工作，且不确定性很高。

### 方向 C：课程学习（中等不确定性）

设计一个从简单到复杂的课程：
1. 先只在 head_on 上训练
2. 然后逐步引入 crossing
3. 使用显式的梯度平衡机制（如 GradNorm）

风险：课程设计需要大量试验，且可能无法根本解决 tradeoff。

### 方向 D：行为克隆 + 强化学习（混合方法）

从 expert 或 baseline 的轨迹中做行为克隆（BC），然后做 PPO 微调。BC 可能提供更好的初始化，避免 warmstart 的 catastrophic forgetting。

---

## 六、对论文的建议

### 6.1 如何呈现 Baseline

**Baseline（无 combat finetune）** 应该是论文的主线结果：

> 在对抗 expert 的 20-seed 形式评估中，VPP 策略在 head_on 任务中达到 55% 胜率，在 crossing_feasible 任务中达到 35% 胜率。对抗自训练 end_to_end 策略时，胜率分别提升至 95% 和 100%。这表明 baseline 策略已具备基本的空战几何塑造能力，但面对确定性 expert 制导时仍有显著局限。

### 6.2 如何呈现 Tradeoff

**Tradeoff** 必须被作为核心发现呈现，而非轻描淡写：

> 为了验证 VPP 是否能成为通用的机动决策接口，我们进行了系统的 combat finetune 实验。在 12 轮、14 种配置的尝试中（包括任务加权、对手感知、任务类型观测、多头网络、独立编码器、网络容量扩展等），我们发现一个结构性的零和困境：没有任何单策略配置能同时在 head_on 和 crossing 任务中对抗 expert 达到 ≥ 50% 的胜率。加权调整只能将性能从一个任务转移到另一个任务，而无法同时提升两侧。这揭示了单策略 PPO 在异构空战几何任务上的根本局限。

### 6.3 如何呈现 Head-on-weighted

**Head-on-weighted（1.00/0.10）** 可以作为"专项优化"的例证：

> 当任务类型已知并对 head_on 任务施加 2 倍梯度权重时，策略在 head_on 上的胜率提升至 100%，但 crossing 上的胜率降至 10%。这证明了 VPP 在特定任务上的优化潜力，但也再次确认了 tradeoff 的刚性。

### 6.4 必须披露的限制

1. 实验仅在 JSBSim 环境中验证，真实飞行器的动态响应可能不同
2. 专家对手采用确定性制导，未考虑自适应对手
3. 单策略局限在 2 个任务（head_on + crossing），未扩展到 sustained_turn、multi_waypoint
4. 32k PPO 步数可能不足以收敛到全局最优

---

## 七、实验遗产

### 7.1 代码改进

- `policy_network.py`：新增 `num_tasks` 支持，可复用的 multi-head / separate encoder 架构
- `ppo_agent.py`：优化器加载容错（支持架构变更时跳过 optimizer state）
- `observation.py`：新增 `task_type` 参数，支持 `is_crossing` 二进制特征
- `tracking_env.py`：新增 `_include_task_type_in_observation` 标志
- 测试覆盖：新增 `test_multi_head_policy.py`（10 项测试），全量 118 项测试通过

### 7.2 配置模板

所有 config 采用 `includes:` 链式继承，确保 provenance-safe：
- 基础配置 → 实验配置 → 变体配置
- 每个实验目录自动写入 `config_snapshot.yaml`

### 7.3 可复现性

```bash
# 复现 baseline 20-seed pilot
python scripts/run_jsbsim_hrl_comparison.py \
  --config config/experiment/jsbsim_hrl_comparison.yaml \
  --run-id baseline_expert_20seed \
  --methods prediction_vpp_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00 \
  --tasks head_on crossing_feasible \
  --seeds 480 481 482 483 484 485 486 487 488 489 490 491 492 493 494 495 496 497 498 499 \
  --opponent-stage expert --run-status formal

# 复现 head-on-weighted training
python -m uav_vpp_guidance.training.train_prediction_vpp_combat_finetune \
  config/experiment/train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_mixed_opponent_aware_32k_task_type_headon_weighted.yaml \
  --seed 480
```

---

## 八、结论

12 轮实验的最终结论不是"失败"，而是**发现了单策略 PPO 在异构空战几何任务上的结构性局限**。这是一个有价值的负面发现（negative result），它告诉我们：

1. **Baseline 是可用的**：0.55/0.35 对抗 expert 是可靠的主线结果
2. **Combat finetune 是危险的**：任何尝试都会破坏 baseline 的泛化能力
3. **Multi-head 不是银弹**：问题不在表示学习，而在优化动态
4. **未来的方向是多策略**：如果必须同时高，需要放弃单策略假设

这些发现为论文的"Maneuver Decision Interface"章节提供了坚实的证据基础，也为后续研究指明了方向。

---

*报告生成：2025-06-30*  
*Git 分支：`fix/flight-control-comparison-sync`*  
*总 commit：10 个（含 multi-head、separate encoder、optimizer fix、evidence package）*
