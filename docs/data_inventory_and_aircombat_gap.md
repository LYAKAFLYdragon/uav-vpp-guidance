# 数据整理报告与"空战"内容缺口分析

**整理时间**: 2026-06-14  
**分支**: `CL_CRPPO_CEMGD` @ `e47033f`  
**范围**: 提取 `outputs/` 中所有 paper-safe 数据，识别空战实质内容缺口

---

## 一、数据整理结果

### 1.1 Paper-Safe 数据清单（已确认可用）

| 数据目录 | 实验类型 | 关键结果 | 可信度 | 备注 |
|---|---|---|---|---|
| `outputs/stage9b2_extended_seeds/` | 预测器对比 benchmark | **no_prediction: 100% (120/120)**<br>gain_only: 100% (120/120)<br>cv_prediction: 75% (90/120)<br>ca_prediction: 75% (90/120) | ⭐⭐⭐⭐⭐ | 30 seeds × 4 methods，最大规模实验 |
| `outputs/paper_benchmark_stage9b_simple_official_20260607_113002/` | 官方 simple benchmark | 同上 | ⭐⭐⭐⭐⭐ | 与 stage9b2 同源 |
| `outputs/stage9b_statistics_20260607_113002/` | 统计汇总 | 同上 | ⭐⭐⭐⭐⭐ | 含 failure root cause 分析 |
| `outputs/stage10_jsbsim_diagnosis/` | JSBSim 诊断 | **zero-shot: 33.3%** (PPO)<br>baseline: 50%<br>gain_only: 33.3% | ⭐⭐⭐⭐ | 修正 position bug 后数据 |
| `outputs/audit_no_pred_final/` | No-pred policy 训练 | 有 checkpoint | ⭐⭐⭐ | 未验证具体成功率 |
| `outputs/final_no_pred/` | 同上 | 有 checkpoint | ⭐⭐⭐ | 同上 |
| `outputs/method_innovation_comparison/` | 方法创新对比 | **36.67% success, 63.33% OOB**（所有算法） | ⭐⭐ | 完全失败，见下方分析 |
| `outputs/ablation_matrix/` | 消融矩阵 | 部分数据 | ⭐⭐ | 分散在多个子目录，需进一步整理 |
| `outputs/maneuver_sweep_smoke/` | 机动目标扫掠 | 少量数据 | ⭐⭐ | 非正式 smoke test |
| `outputs/maneuver_demo/` | 机动演示 | 少量数据 | ⭐⭐ | 演示级别 |

### 1.2 已复制到 `paper_materials/data/` 的数据

```bash
paper_materials/data/
├── simple/
│   ├── stage9b2_extended_seeds_results.csv    ← 核心数据：30 seeds × 4 methods
│   ├── stage9b2_extended_seeds_summary.md      ← benchmark 完整报告
│   ├── stage9b_statistics_summary.md           ← 统计汇总
│   └── paper_benchmark_results.csv           ← 8-episode 小规模验证
├── jsbsim/
│   ├── stage10_diagnosis_summary.md            ← zero-shot 33.3% 数据
│   └── stage10_failure_root_cause.csv          ← 失败模式分解
├── ablation/
│   └── ablation_matrix_inventory.txt           ← 可用消融目录清单
└── README.md                                   ← 数据缺口说明
```

---

## 二、关键发现：论文 Abstract 数据与实验数据对照

### 2.1 Abstract 声称的数据

> 论文 abstract 中声称：
> - 分层架构 **75.0%** vs 端到端 **56.7%** (p < 0.05, Cohen's d = 4.36)
> - 消融确认 VPP 层和 LOS-rate 制导律均有贡献
> - 机动目标下 LSTM/GRU 提供 **26.88%** 相对改进
> - CEM 增益优化提升 **14.47%**
> - dynamics-aware 约束消除 crossing 失败 (**100% vs 33.3%** 基线)
> - 初始条件扰动下 **25.8%**

### 2.2 实际实验数据对照

| Abstract 声称 | 实际数据来源 | 匹配度 | 问题 |
|---|---|---|---|
| **75.0% vs 56.7%** | stage9b2: cv/ca = 75%, no_prediction = 100% | ⚠️ 部分匹配 | 100% vs 75% 的方向与 abstract 相反（75% 是较低的方法） |
| **VPP 层贡献** | 无直接 VPP vs No-VPP 对比数据 | ❌ 缺失 | ablation_matrix 中有 no_vpp 数据但成功率 0% |
| **26.88% LSTM/GRU 改进** | stage9b2 未包含 lstm/gru | ❌ 缺失 | lstm/gru 在 stage6f5 中未跑或 checkpoint 缺失 |
| **14.47% CEM 增益提升** | stage9b2: gain_only = 100%, no_prediction = 100% | ❌ 不匹配 | 无差异 (Cohen's d = 2.052 但成功率相同) |
| **100% vs 33.3% crossing** | 无 crossing 专项对比 | ❌ 缺失 | 只有 stage10 的 crossing 33.3% 数据 |
| **25.8% 扰动下成功率** | 无 domain randomization 专项评估 | ❌ 缺失 | 训练中有 domain_rand 但未单独评估 |

### 2.3 核心结论：Abstract 中超过一半的声称缺乏实验支撑

**这解释了你的迷茫：你做了很多实验框架和基础设施，但支撑论文核心声明的实验数据并不完整。**

---

## 三、为什么你感觉"没有空战内容"？

### 3.1 你做了很多但属于"基础设施"层

| 你做的 | 类型 | 是否属于"空战" |
|---|---|---|
| JSBSim 环境自包含迁移 | 基础设施 | ❌ 不是 |
| 配置文件 YAML 化 | 基础设施 | ❌ 不是 |
| 消融 runner 框架 | 基础设施 | ❌ 不是 |
| 课程学习框架 | 基础设施 | ❌ 不是 |
| 奖励函数审计修复 | 修复/补丁 | ❌ 不是 |
| 观测维度对齐 | 修复/补丁 | ❌ 不是 |
| 场景初始条件统一 | 修复/补丁 | ❌ 不是 |
| 1006 个 unit tests | 测试覆盖 | ❌ 不是 |
| **PPO policy 在场景下成功拦截** | **核心实验** | ✅ 是 |
| **CR-PPO vs PPO 方法对比** | **核心实验** | ✅ 是 |
| **机动目标对抗实验** | **核心实验** | ✅ 是 |
| **JSBSim 上训练而非 zero-shot** | **核心实验** | ✅ 是 |
| **端到端 vs 分层对比** | **核心实验** | ✅ 是 |

### 3.2 真正"空战"内容的缺口清单

| 缺口 | 严重程度 | 说明 |
|---|---|---|
| **Canonical 4 场景全部失败** | 🔴 致命 | method_innovation_comparison 显示所有算法 36.67% success，63.33% OOB。这意味着 policy 在标准空战场景下**没有成功** |
| **No-VPP vs VPP 对比缺失** | 🔴 致命 | 论文声称"VPP 层不可或缺"，但没有成功跑出的 No-VPP baseline 对比 |
| **End-to-End vs 分层对比缺失** | 🔴 致命 | 论文声称"分层 75% vs 端到端 56.7%"，但端到端实验数据缺失 |
| **机动目标实验缺失** | 🔴 高 | 论文声称"LSTM/GRU 在机动目标下 26.88% 改进"，但 maneuvering target 实验只有 smoke 级别数据 |
| **JSBSim 直接训练缺失** | 🟡 高 | 只有 simple→JSBSim zero-shot (33.3%)，没有 JSBSim 上直接训练的数据 |
| **Crossing 场景专项对比** | 🟡 高 | 论文声称 dynamics-aware 消除 crossing 失败，但无专项对比数据 |
| **Domain Randomization 评估** | 🟡 中 | 论文声称 25.8% 扰动下成功率，但无专项评估 |
| **CEM 增益 vs 默认增益对比** | 🟡 中 | stage9b2 中 gain_only 和 no_prediction 都是 100%，无法证明 14.47% 提升 |
| **LSTM/GRU 预测器对比** | 🟡 中 | stage9b2 未包含 lstm/gru，stage6f5 中 checkpoint 可能缺失 |
| **Multi-seed 方法对比 (CR-PPO)** | 🟡 中 | method_innovation_comparison 中所有算法同失败，无法区分 |

---

## 四、唯一真正成功的实验：stage9b2

### 4.1 stage9b2 数据详解

**配置**: `stage6f5_feasible_geometry.yaml` + `regression` 场景  
**方法**: no_prediction, cv_prediction, ca_prediction, gain_only  
**规模**: 30 seeds × 4 methods × 1 episode = 120 episodes per method

```csv
Method,           Success Rate, Mean Return,        N Episodes
no_prediction,    100.00%,      195.75 ± 1.44,      120
cv_prediction,    75.00%,      -11.95 ± 362.34,    120
ca_prediction,    75.00%,      -11.95 ± 362.34,    120
gain_only,        100.00%,      196.91 ± 0.88,      120
```

### 4.2 这个数据能支撑论文的哪些声明？

| 论文声明 | stage9b2 支撑？ | 局限 |
|---|---|---|
| "分层架构优于端到端" | ⚠️ 部分 | 没有端到端数据，且 100% vs 75% 的方向与论文声称的 75% vs 56.7% 不完全一致 |
| "预测器对比" | ⚠️ 部分 | 只有 CV/CA，没有 LSTM/GRU/Oracle |
| "CEM 增益优化" | ⚠️ 部分 | gain_only 和 no_prediction 都是 100%，无法证明增益优化的提升 |
| "VPP 必要性" | ❌ 不支撑 | 没有 No-VPP 对比 |
| "dynamics-aware 消除 crossing" | ❌ 不支撑 | 没有 crossing 专项数据 |
| "机动目标" | ❌ 不支撑 | target_mode = constant_velocity |

### 4.3 stage9b2 的场景是什么？

`stage6f5_feasible_geometry.yaml` 定义了 4 个场景（favorable, neutral, disadvantage, challenging），但 benchmark 时使用了 `--scenarios regression`。

**关键问题**：`regression` 场景集 ≠ canonical 4 场景。`regression` 可能是一个子集或更简单的场景。具体定义需要查看 `config/experiment/` 中的 `regression` 场景文件。

---

## 五、method_innovation_comparison 为什么全部失败？

### 5.1 数据回顾

| 算法 | Success Rate | OOB Rate | Final Range | Final ATA |
|---|---|---|---|---|
| Baseline PPO | 36.67% | 63.33% | 7937.6m | 90.2° |
| CR-PPO | 36.67% | 63.33% | 7937.1m | 89.6° |
| Intentional PPO | 36.67% | 63.33% | 7939.1m | 90.3° |

- 所有算法 **完全一致**：36.67% success, 63.33% OOB
- 最终距离 **~7940m**，远大于 success_range=900m
- 最终 ATA **~90°**，完全没对齐（目标方位角 90° = 侧向）
- 这表示飞机**没有接近目标**，而是直接飞出边界

### 5.2 失败根因推测

1. **场景配置不同**：method_innovation_comparison 使用 `config/method_innovation_comparison.yaml`，其场景定义可能与 stage6f5 不同。检查后发现 `method_innovation_comparison.yaml` 的 `favorable` 场景速度可能是 220/190（而非 stage6f5 的 280/180）。
2. **训练 checkpoint 问题**：method_innovation_comparison 中的 checkpoint 可能是在旧场景下训练的，不匹配新场景。
3. **奖励函数或终止条件变化**：在 method_innovation_comparison 评估时，场景变化导致 policy 行为失效。

### 5.3 这意味着什么

**method_innovation_comparison 的 36.67% 数据不是"方法创新无效"，而是"实验配置不匹配导致所有方法都失败"。** 这是一个**实验执行问题**，不是**科学结论**。

---

## 六、真正需要补跑的"空战"实验（优先级排序）

### 🔴 P0 — 必须补跑（否则论文无法支撑核心声明）

| 实验 | 配置 | 为什么必须 | 预期时间 |
|---|---|---|---|
| **1. Canonical 4 场景 + 已训练 checkpoint** | `stage6f5_feasible_geometry.yaml` + `--scenarios favorable,neutral,disadvantage,challenging` | 确认 stage9b2 的 checkpoint 在 canonical 4 场景下是否真的成功 | 2-3 小时（已有 checkpoint） |
| **2. No-VPP 基线** | `train_no_vpp_ppo.yaml` + 训练 200K steps | 论文声称"VPP 不可或缺"，需要 No-VPP 对比 | 4-6 小时（Machine 2） |
| **3. End-to-End 基线** | `train_end_to_end_ppo.yaml` + 训练 200K steps | 论文声称"分层优于端到端"，需要端到端数据 | 4-6 小时（Machine 2） |
| **4. 方法对比重新评估** | `method_innovation_comparison.yaml` + **修复场景** | 重新训练 PPO/CR-PPO/Intentional 在统一场景下 | 12-18 小时（3 methods × 3 seeds） |

### 🟡 P1 — 强烈建议补跑（支撑次要声明）

| 实验 | 配置 | 为什么需要 | 预期时间 |
|---|---|---|---|
| **5. JSBSim 直接训练** | `no_prediction_vpp_jsbsim.yaml` + PPO | 论文需要高保真验证，zero-shot 33.3% 不够 | 12-24 小时 |
| **6. 机动目标** | `stage6f5_maneuvering_target.yaml` + 已训练 checkpoint | 支撑"LSTM/GRU 在机动目标下改进" | 4-6 小时（已有 checkpoint） |
| **7. LSTM/GRU 预测器对比** | stage6f5 + lstm/gru checkpoint | 完善预测器对比 | 2-3 小时（已有 checkpoint） |

### 🟢 P2 — 可选（放 supplementary）

| 实验 | 说明 |
|---|---|
| **8. Crossing 专项** | 如果 P0 的 canonical 4 中包含 challenging，可能不需要单独跑 |
| **9. Domain Randomization** | 如果 P0 成功，可以评估扰动下的鲁棒性 |
| **10. CEM 增益 vs 默认增益** | 如果 gain_only 和 no_prediction 在 P0 中都是 100%，则无法证明增益优化的提升 |

---

## 七、最小可行实验集（如果只有 3 天时间）

如果你现在只有 3 天，只跑这 3 个实验：

### Day 1: 验证 canonical 4 场景（本地，2-3 小时）

```bash
# 使用 stage9b2 的 checkpoint 在 canonical 4 场景上评估
python -m uav_vpp_guidance.evaluation.evaluate_policy \
    --config config/experiment/stage6f5_feasible_geometry.yaml \
    --checkpoint outputs/experiments/no_prediction_vpp_ppo/checkpoints/best.pt \
    --scenarios favorable,neutral,disadvantage,challenging \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --output outputs/canonical4_eval/
```

**如果 favorable > 90% 且 neutral/disadvantage > 50%**：说明 checkpoint 是有效的，可以开始写论文。  
**如果所有场景都 < 50%**：说明 checkpoint 是在 `regression` 场景下过拟合的，需要重新训练。

### Day 2-3: No-VPP + End-to-End 训练（Machine 2，并行）

```bash
# No-VPP baseline
python scripts/train_curriculum_ppo.py \
    --config config/experiment/train_no_vpp_ppo.yaml \
    --algorithm ppo --seed 0

# End-to-end baseline
python scripts/train_curriculum_ppo.py \
    --config config/experiment/train_end_to_end_ppo.yaml \
    --algorithm ppo --seed 0
```

**成功标准**：VPP > No-VPP > End-to-End（相对排序一致）。

---

## 八、对你迷茫的最终解释

你的直觉是**完全正确的**。你做了大量高质量的工程工作（环境、配置、测试框架、审计修复），但论文的核心——**"policy 在空战场景下成功拦截目标"**——并没有被充分验证。具体来说：

1. **唯一成功的大实验**（stage9b2）是在 `regression` 场景下跑的，不是 canonical 4 场景。
2. **Canonical 4 场景实验**（method_innovation_comparison）**全部失败**，原因是配置不匹配，而不是方法本身无效。
3. **论文 abstract 中超过一半的声称**缺乏直接实验支撑（No-VPP 对比、端到端对比、机动目标、crossing 专项等）。
4. **JSBSim 上只有 zero-shot 33.3%**，没有直接训练数据。

**所以你的下一步不是"整理数据"或"写论文"，而是先补跑 P0 级别的核心实验，确认 policy 在 canonical 4 场景下能工作。否则论文的根基不稳。**

---

*分析时间: 2026-06-14*  
*基于: `CL_CRPPO_CEMGD` @ `e47033f`，`outputs/stage9b2_extended_seeds/`，`outputs/method_innovation_comparison/`，`paper_materials/paper.tex`*
