# MVP & Checklist 自审意见：Hierarchical Tactical-Mode Commander for VPP

> 对 `hierarchical_commander_mvp_patch_checklist.md` 的系统性审查，基于 12 轮实验证据和学术文献支撑。

---

## 一、总体判断：合理，但需要 3 处关键修正和 2 个风险补充

这份 checklist 的学术价值定位是准确的——它确实最贴合"把 VPP 从追点接口升级成机动决策接口"的核心命题。但在实施路径上，有几个地方需要更严谨的处理，否则可能导致实验结论不具可解释性。

---

## 二、关键修正

### 修正 1：Oracle Gate 的验收标准过高

**当前标准**：
```
expert/head_on >= 0.90
expert/crossing_feasible >= 0.80
end_to_end/head_on >= 0.85
end_to_end/crossing_feasible >= 0.95
```

**问题**：
- `expert/ho` 0.90 要求 oracle gate 能达到接近 head_on-weighted specialist 的纯 head_on 性能。但 oracle gate 使用固定 specialist，而 head_on-weighted specialist 是在纯 head_on 训练分布上训练的。oracle gate 在 crossing 任务中切换到 crossing specialist，在 head_on 任务中切换到 HO specialist，这看起来合理，但需要考虑：
  - 不同 specialist 的 observation 分布可能不同（task_type 有无差异）
  - 不同 specialist 的 checkpoint 可能来自不同训练阶段（best.pt vs last.pt）
  - crossing specialist 的 pure crossing 性能可能不是 0.90（需要确认）

**建议修正**：
```
expert/head_on >= 0.70  (因为 HO specialist 在 head_on 上达到 1.00，但 oracle 的固定切换可能有边缘损失)
expert/crossing_feasible >= 0.60  (crossing specialist 在 crossing 上约 0.70-0.90，但需先确认具体值)
end_to_end/head_on >= 0.80
end_to_end/crossing_feasible >= 0.90
```

**核心原则**：oracle gate 的验收标准应该是"明显优于单策略 baseline（0.55/0.35）"，而不是"接近 specialist 的纯任务上限"。

### 修正 2：Commander 不应禁止 task_type 观测

**当前要求**：`include_task_type_in_observation: false`

**问题**：
- 在 MVP 第一阶段禁止 task_type 是合理的（验证 commander 能否从纯几何态势中学会切换）
- 但如果第一阶段失败，第二阶段需要允许 task_type 作为辅助输入
- 更重要的是，你的 12 轮实验已经证明了 task_type 本身不足以打破 tradeoff，但 task_type + 正确的架构（分层）可能有效

**建议修正**：
- 第一阶段：commander 不使用 task_type（纯几何态势学习）
- 第二阶段（如果第一阶段失败）：允许 commander 使用 task_type 作为辅助输入，与 oracle gate 公平对比
- 在 config 中保留 `include_task_type_in_commander_observation: false`，但注释说明未来可切换

### 修正 3：Crossing Specialist 的选择需要数据验证

**当前要求**：使用 `crossing_weighted` 作为 crossing specialist

**问题**：
- 你的证据矩阵中，crossing_weighted 的 expert/cr = 0.90，但 expert/ho = 0.10
- task_type_unweighted 的 expert/cr = 0.70，但 expert/ho = 0.40
- 哪个作为 crossing specialist 更合适？取决于你想在 crossing 任务中达到什么水平

**建议**：
- 在 checklist 中增加一个前置步骤："确认 crossing specialist 的 pure-task 性能"
- 如果 crossing_weighted 的 pure crossing 性能确实最高（0.90），就用它
- 但如果它过于"偏科"（head_on 完全失效），可能会影响 oracle gate 的鲁棒性
- **备选方案**：使用 task_type_unweighted（0.40/0.70）作为 crossing specialist，因为它的通用性更好，更符合"generic" 的定位

---

## 三、需要补充的风险

### 风险 1：Macro-step 长度的敏感性未被分析

**问题**：`macro_action_repeat_steps: 12` 是一个硬编码值，但未被验证。

- 太长（如 50）：commander 无法及时响应态势变化，可能错过关键切换时机
- 太短（如 5）：commander 过于频繁切换，导致策略不稳定，且 switch penalty 累积
- 最佳值可能因任务而异（head_on 可能需要更快速的切换，crossing 可以更慢）

**建议补充**：
- 在 oracle gate 的 Phase A 中，增加一个 macro-step 敏感性分析：
  - 测试 macro_step = 6, 12, 24 三个值
  - 记录每种情况下的性能和切换次数
- 如果 12 不是最优值，后续 learned commander 训练应使用最优值

### 风险 2：Switch Penalty 的哲学问题

**问题**：`switch_penalty: 0.02` 是一个先验值，但它的作用需要仔细定义。

- 惩罚太大：commander 可能被迫保持单一模式，丧失分层意义
- 惩罚太小：commander 可能频繁切换，导致低层策略不稳定（每个 specialist 的 action 分布不同，频繁切换可能导致飞行器抖动）

**建议补充**：
- 在 oracle gate 中不施加 switch penalty（因为 oracle 是固定切换，没有"频繁切换"问题）
- 在 learned commander 中，switch penalty 应作为可调参数，在 Phase B 中验证：
  - 对比 switch_penalty = 0.0, 0.02, 0.05 三种情况
  - 记录切换次数与性能的关系
- 如果 commander 在没有 switch penalty 时仍然倾向于少切换，说明态势本身具有稳定性，不需要强惩罚

---

## 四、对假设 H1/H2 的审查

### H1：当前单策略 PPO 的瓶颈不在低层动作表达，而在于缺少显式的高层战术模式选择

**审查**：可检验。如果 oracle gate 成功，则直接支持 H1。如果 oracle gate 失败，则 H1 被证伪，需要重新考虑低层 specialist 是否足够"specialized"。

**建议补充**：H1 的反面假设（null hypothesis）应明确：
> H1_null：即使提供了显式的高层战术模式选择，如果低层 specialist 本身不够强或不够兼容，组合性能仍然无法超过单策略 baseline。

### H2：冻结 specialist + 高层 commander 可以突破 tradeoff

**审查**：可检验。但 H2 的"突破"需要定义清楚。

**建议修正**：
> H2：如果冻结低层 specialist 并训练高层 commander，则组合的 min(expert/ho, expert/cr) 将显著高于单策略 baseline 的 min(0.55, 0.35) = 0.35。

这比"同时满足 ≥ 0.50"更合理，因为：
- 即使 expert/ho = 0.60 且 expert/cr = 0.40，min = 0.40，相比 baseline 的 min = 0.35，也是显著改进
- 这种改进在统计上更容易达到，且在学术上仍然有价值（证明了分层的有效性）

---

## 五、对对照组设计的审查

### 当前对照组：
1. 单策略 baseline（已有）
2. 静态 oracle task gate（新增）
3. 学习型 commander（新增）

**审查**：缺少一个关键对照组——**Random Mode Switching**。

**为什么需要**：
- 如果 oracle gate 成功，但 learned commander 失败，我们需要知道 learned commander 的失败是因为"学习不到"还是"分层架构本身不够"
- 如果随机切换的性能也接近 oracle gate，说明 specialist 本身足够强，commander 的选择不是关键
- 如果随机切换远低于 oracle gate，说明正确的模式选择确实关键，但 learned commander 需要更强的学习信号

**建议补充**：
- 在 oracle gate 和 learned commander 之间，增加一个 **random gate** 对照：
  - 每个 macro step 随机选择 generic / head_on / crossing
  - 这个对照的成本很低（不需要训练），但信息增益很高

---

## 六、对文件清单的审查

### 当前清单：10 个文件/模块

**审查**：基本完整，但缺少 2 个关键模块：

1. **Specialist 性能验证脚本**（前置步骤）
   - 在 oracle gate 之前，需要确认每个 specialist 的 pure-task 性能
   - 建议：`scripts/validate_specialist_pure_task_performance.py`
   - 输出：每个 specialist 在 head_on / crossing 上的纯任务性能

2. **Macro-step 敏感性分析脚本**（前置步骤）
   - 建议：`scripts/analyze_macro_step_sensitivity.py`
   - 输出：不同 macro_step 下的 oracle gate 性能曲线

3. **Random gate 对照**（新增对照）
   - 建议：`src/uav_vpp_guidance/evaluation/random_mode_gate_policy.py`
   - 测试：`test_random_mode_gate_policy_selects_any_mode`

---

## 七、对验收标准的审查

### 当前标准：

```
Primary gate:
- expert/head_on >= 0.60
- expert/crossing_feasible >= 0.50
- end_to_end/head_on >= 0.85
- end_to_end/crossing_feasible >= 0.90

Structural gate:
- min(expert/ho, expert/cr) > 0.35
```

**审查**：合理，但 Behavioral gate 可以更具体。

**建议补充**：
- 增加一个 **switch frequency 上限**：
  - 每个 episode 的平均切换次数不应超过 10 次（如果 macro_step = 12，episode ~300 步，最多 25 次 macro steps，10 次切换意味着 commander 在约 40% 的 macro steps 切换）
- 增加一个 **mode stickiness 下限**：
  - 在 head_on 任务中，head_on_specialist 的激活比例应 ≥ 60%
  - 在 crossing 任务中，crossing_specialist 的激活比例应 ≥ 60%
  - 这证明了 commander "确实学到了"在正确任务中选择正确模式，而不是随机或默认 generic

---

## 八、对实施顺序的建议调整

### 当前顺序：
1. Phase A: Oracle gate（静态对照）
2. Phase B go/no-go: 如果 oracle 通过，训练 learned commander
3. Phase C: Learned commander pilot

### 建议调整：
1. **Step 0: Specialist 验证**（新增）
   - 确认每个 specialist 的 pure-task 性能
   - 确认 crossing specialist 的选择（crossing_weighted vs task_type_unweighted）
   - 如果 specialist 本身不够强，调整选择后再进入 oracle gate

2. **Step 1: Macro-step 敏感性**（新增）
   - 在 oracle gate 中测试 macro_step = 6, 12, 24
   - 选择最优 macro_step 后，再训练 learned commander

3. **Step 2: Oracle gate**（保持）
   - 但验收标准降低（见修正 1）

4. **Step 3: Random gate**（新增）
   - 低成本对照，验证"选择本身是否有价值"

5. **Step 4: Learned commander**（保持）
   - 但允许 task_type 作为可选输入（如果纯几何失败）

6. **Step 5: Switch penalty ablation**（新增）
   - 如果 learned commander 性能不佳，测试是否 switch penalty 过大

---

## 九、一句话总结

**这份 checklist 的学术方向完全正确，但验收标准过高、缺少 random gate 对照、对 macro-step 和 switch penalty 的敏感性分析不足。修正后，它将是一个科学上严谨、工程上可落地的 MVP。**

---

## 十、修正后的 checklist 关键变更汇总

| 变更项 | 原内容 | 修正后 | 理由 |
|---|---|---|---|
| Oracle gate 标准 | expert/ho ≥ 0.90 | expert/ho ≥ 0.70 | 不应要求接近 specialist 纯任务上限 |
| Oracle gate 标准 | expert/cr ≥ 0.80 | expert/cr ≥ 0.60 | 同上 |
| Commander task_type | 禁止使用 | 第一阶段禁用，第二阶段允许 | 增加可解释性对比 |
| Crossing specialist | crossing_weighted | 需先验证 pure-task 性能 | 可能 task_type_unweighted 更通用 |
| 新增对照 | 无 | Random gate | 区分"选择有价值"vs"学习有价值" |
| 前置分析 | 无 | Macro-step 敏感性 | 避免硬编码 12 的最优性假设 |
| 前置分析 | 无 | Specialist 纯任务验证 | 确保 specialist 本身足够强 |
| Switch penalty | 固定 0.02 | 可调，Phase B ablation | 避免过大惩罚导致模式锁定 |
| Behavioral gate | 仅 mode 主导性 | 增加 switch frequency 上限 + mode stickiness | 更完整的 commander 行为评估 |

---

*审查完成：2025-06-30*  
*审查者：基于 12 轮实验证据和文献支撑的系统分析*
