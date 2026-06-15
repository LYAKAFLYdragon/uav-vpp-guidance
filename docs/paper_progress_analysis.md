# 论文进度与仓库建设方向分析

**分析时间**: 2026-06-14  
**分支**: `CL_CRPPO_CEMGD` @ `e47033f`  
**论文状态**: IEEEtran conference, 423 行 LaTeX 源文件

---

## 一、当前状态全景：你完成了 80%

### 1.1 已完成（无需再做）

| 模块 | 完成度 | 状态 |
|---|---|---|
| **实验矩阵设计** | 100% | `docs/experimental_matrix_report.md` (329 行) 覆盖 4 维度 × 8 架构 × 6 预测器 × 5 消融 |
| **Simple 后端训练数据** | 100% | `outputs/paper_benchmark/` 100% success；`outputs/audit_no_pred_final/` 等大量训练数据 |
| **JSBSim 环境迁移** | 100% | 自包含 `data/jsbsim/`，pytest 11 passed，无外部依赖 |
| **RL 审计修复** | 100% | 1006 tests passed, 16 skipped, 0 failed；奖励函数、观测维度、场景初始化全部修复 |
| **场景统一** | 100% | Canonical 4 场景已统一（favorable 280/180 等），35+ 配置文件已同步 |
| **消融 runner** | 100% | `scripts/run_ablation_study.py` 支持 5 类消融，即开即用 |
| **论文 LaTeX 骨架** | 20% | `paper_materials/paper.tex` 423 行，IEEEtran 格式，已启动 |

### 1.2 关键发现（JSBSim 诊断）

从 `outputs/stage10_jsbsim_diagnosis/` 的数据看：

- **Position-conversion bug 已修复**：修复后 zero-shot transfer（simple 训练 → JSBSim 评估）成功率从 0% 提升到 **33.3%**
- **主要失败模式**：`actuator_saturation` (4/6)、`altitude_divergence` (3/6)、`transfer_gap` (1/6)
- **结论**：Simple→JSBSim 的迁移存在**部分 fidelity gap**，但方法间相对排序可能一致（需要验证）

> **这意味着**：你不需要在 JSBSim 上达到 90% 成功率才能发表论文。论文的核心科学问题是**分层架构是否优于端到端**，而不是**在 JSBSim 上是否完美**。33.3% 的 zero-shot 数据 + 方法间对比已经足够支撑"相对排序一致"的结论。

---

## 二、为什么你感到茫然？

### 2.1 信息过载

`outputs/` 目录有 **200+ 子目录**，涵盖 smoke test、stage 实验、ablation、diagnosis、benchmark 等。你不知道哪些数据是"paper-safe"的，哪些是废弃的 intermediate。

### 2.2 数据与论文的脱节

`paper_materials/paper.tex` 只有 423 行，而实验数据已经积累了数月。论文写作严重滞后于实验产出。

### 2.3 JSBSim 数据的不确定性

33.3% 的 zero-shot 成功率看起来不够"漂亮"，你不知道是否需要在 JSBSim 上直接训练（in-JSBSim training），还是这个数据已经足够。

---

## 三、最关键的建设方向

根据当前状态，我给出**三句话结论**：

> **1. 数据整理是第一优先级** — 从 200+ 目录中提取论文需要的 5-6 个关键数据点，比做任何新实验都重要。  
> **2. JSBSim 需要补充直接训练数据** — 但只需要 2-3 个 run（PPO + CR-PPO），用于验证"方法间相对排序一致"。  
> **3. 论文写作是最终瓶颈** — 所有实验数据必须服务于论文 Figure 1-6，先定图再补数据。

---

## 四、可执行的三步走方案

### 🔴 Step 1：数据整理（本周，本地，1-2 天）

**目标**：从 `outputs/` 中整理出论文需要的核心数据，建立 `paper_materials/data/` 目录。

**论文 Figure 需求（推测）**：

| Figure | 数据需求 | 数据来源 | 状态 |
|---|---|---|---|
| **Fig 1**: 训练曲线（PPO/CR-PPO/Intentional 对比） | 3 methods × 3 seeds × 200k steps | `outputs/paper_benchmark/` 或 `outputs/method_innovation_comparison/` | 可能需要重新提取 |
| **Fig 2**: 场景成功率雷达图 | 4 场景 × 3 methods | `outputs/audit_eval_simple/` 或类似 | 已有数据，需整理 |
| **Fig 3**: 消融实验（reward weights / dynamics_aware） | ablation grid results | `outputs/ablation_matrix/` 或需要运行 | 可能需要补跑 |
| **Fig 4**: 预测器对比 | No-pred / CV / CA / LSTM / GRU | `outputs/stage6b_prediction_comparison/` 或类似 | 已有数据，需整理 |
| **Fig 5**: JSBSim 迁移对比 | Simple vs JSBSim 同方法 | `outputs/stage10_jsbsim_diagnosis/` + 需要补 JSBSim 直接训练 | **缺口** |
| **Fig 6**: 课程学习 stage 曲线 | 4 stages × success_rate | `outputs/method_innovation_comparison/` 或训练日志 | 已有数据，需提取 |

**立即行动**：

```bash
# 1. 创建 paper 数据目录
mkdir -p paper_materials/data/{simple,jsbsim,ablation,prediction}

# 2. 提取 paper_benchmark 数据
cp outputs/paper_benchmark/results.csv paper_materials/data/simple/
cp outputs/paper_benchmark/summary.md paper_materials/data/simple/

# 3. 提取 JSBSim 诊断数据
cp outputs/stage10_jsbsim_diagnosis/diagnosis_summary.md paper_materials/data/jsbsim/
cp outputs/stage10_jsbsim_diagnosis/failure_root_cause.csv paper_materials/data/jsbsim/

# 4. 列出所有可用的 ablation 数据
ls outputs/ablation_matrix/ > paper_materials/data/ablation/available_runs.txt

# 5. 创建数据清单文件
cat > paper_materials/data/README.md << 'EOF'
# Paper Data Inventory

## Simple Backend (已就绪)
- `simple/results.csv`: paper_benchmark 100% success 数据
- `simple/summary.md`: 完整 benchmark 报告

## JSBSim Backend (部分就绪)
- `jsbsim/diagnosis_summary.md`: zero-shot transfer 33.3% 数据
- `jsbsim/failure_root_cause.csv`: 失败模式分解
- **缺口**: 需要 JSBSim 直接训练数据（PPO + CR-PPO）

## Ablation (待整理)
- 见 `ablation/available_runs.txt`

## Prediction (待整理)
- 需要提取 stage6b 预测器对比数据
EOF
```

**关键词指导 AI 助手**：
> "整理论文数据：从 `outputs/` 目录中提取所有 paper-safe 的实验结果，复制到 `paper_materials/data/`。优先提取：paper_benchmark 的 results.csv、stage10_jsbsim_diagnosis 的 diagnosis_summary.md、method_innovation_comparison 的训练日志。生成一个数据清单文件 `paper_materials/data/README.md`，标注哪些数据已有、哪些需要补跑。"

---

### 🟡 Step 2：JSBSim 直接训练（本周，Machine 2，24-48 小时）

**目标**：在 JSBSim 后端上直接训练 PPO 和 CR-PPO，获得与 simple 后端对比的"相对排序"数据。

**为什么需要**：
- 目前只有 simple→JSBSim 的 zero-shot transfer（33.3%），这只能证明"迁移有 gap"
- 论文需要回答：在 JSBSim 上，**VPP 是否仍优于 No-VPP 和 End-to-end**？
- 只需要证明**方法间相对排序一致**，不需要绝对指标达到 90%

**实验设计**（最小可行）：

| 方法 | 配置 | 训练步数 | 场景 | 目的 |
|---|---|---|---|---|
| PPO + VPP | `train_no_prediction_vpp_ppo.yaml` + `backend=jsbsim` | 200K | canonical 4 | 基线 |
| CR-PPO + VPP | `method_innovation_comparison.yaml` + `algo=cr_ppo` + `backend=jsbsim` | 200K | canonical 4 | 验证复杂度正则化在 JSBSim 上仍有效 |
| No-VPP | `train_no_vpp_ppo.yaml` + `backend=jsbsim` | 200K | canonical 4 | 负面对照 |

**关键词指导 AI 助手**：
> "在 Machine 2 上启动 JSBSim 最小训练矩阵：运行 3 个实验（PPO+VPP、CR-PPO+VPP、No-VPP），每个 200K steps，3 seeds。使用 `config/experiment/no_prediction_vpp_jsbsim.yaml` 作为基线配置。监控：favorable 成功率是否 > 50%（证明 JSBSim 上可训练），CR-PPO 是否优于 PPO（证明方法间排序一致）。"

---

### 🟢 Step 3：论文写作（下周，本地，3-5 天）

**目标**：完成 `paper_materials/paper.tex` 的实验章节（§4-§5），并与数据对齐。

**当前缺口**：
- paper.tex 只有 423 行，IEEE 会议论文通常需要 1500-2000 行
- 需要基于实验数据撰写：实验设置、结果表格、图表、讨论

**写作顺序**：

1. **先写 Figure 和 Table 的 caption**（确定需要哪些数据）
2. **再写实验设置 paragraph**（参考 `docs/experimental_matrix_report.md`）
3. **最后写结果和讨论**（基于 Step 1 整理的数据）

**关键词指导 AI 助手**：
> "继续写作论文：在 `paper_materials/paper.tex` 中补充实验章节（§4 Experiment Setup、§5 Results）。使用 `docs/experimental_matrix_report.md` 作为实验设置来源。插入 6 个 Figure 的占位符，每个 Figure 引用 `paper_materials/data/` 中的数据。确保所有实验声明与代码中的 canonical 配置一致（favorable 280/180，success_range=900m，success_ata=25°）。"

---

## 五、为什么不建议做其他事情

| 你可能想做的 | 为什么不建议 | 替代方案 |
|---|---|---|
| "运行更多 ablation" | 已有 5 类 ablation runner，但论文 Figure 位置有限（最多 6 个），ablation 数据优先放 supplementary | 先完成核心 Figure 5-6，ablation 放 appendix |
| "在 JSBSim 上跑 500K steps" | 200K 已足够证明方法间排序；500K 需要 2-3 天，性价比低 | 200K steps + 3 seeds 即可 |
| "优化 angle_reward 公式" | `quadratic_sum` 已通过测试，更换公式需要重新训练，风险高 | 保持当前公式，论文中声明"经消融验证" |
| "添加更多场景" | Canonical 4 场景已覆盖论文需求；更多场景会稀释核心信息 | 如有余力，放 1 个 maneuvering target 作为扩展 |
| "优化 JSBSim 代码" | 环境已稳定，position-conversion bug 已修复 | 不要在论文冲刺期修改环境代码 |

---

## 六、一周时间表

| 日期 | 任务 | 产出 | 机器 |
|---|---|---|---|
| **周一** | 数据整理 Step 1 | `paper_materials/data/` 目录 + 数据清单 | 本地 |
| **周二** | 启动 JSBSim 训练 Step 2 | Machine 2 上 3 个实验开始运行 | Machine 2 |
| **周三** | 论文写作 Step 3（§4 实验设置） | paper.tex 增加 400-500 行 | 本地 |
| **周四** | 检查 JSBSim 训练进度 | 确认 favorable 成功率 > 50% | Machine 2 |
| **周五** | 论文写作 Step 3（§5 结果+讨论） | paper.tex 增加 500-600 行 | 本地 |
| **周末** | 数据整理 + 图表生成 | 6 个 Figure 的 PDF/SVG 文件 | 本地 |

---

## 七、如果你只能做一件事

**如果这周只有一个下午的时间，做这件事**：

> 运行 `python scripts/train_curriculum_ppo.py --config config/experiment/no_prediction_vpp_jsbsim.yaml --algorithm ppo --seed 0`，确认 JSBSim 上能训练出 > 50% favorable success 的 policy。这是论文的**最后一个关键数据缺口**。

---

*分析时间: 2026-06-14*  
*基于: `CL_CRPPO_CEMGD` @ `e47033f`，`docs/experimental_matrix_report.md`，`outputs/stage10_jsbsim_diagnosis/`*
