# Oracle Task Gate MVP 实验结果与分析（AoA 60 scope）

## 实验日期
2026-06-30

## 实验目的
验证 Oracle Task Gate（静态专家选择基线）在主线 combat-only scope（AoA 60）下的表现，确认 specialist 组合是否能在各自任务上达到预期的纯性能。

## 实验配置
- **Comparison Config**: `config/experiment/jsbsim_hrl_oracle_task_gate_mvp.yaml`
- **Backend**: JSBSim (strict)
- **Opponent**: Expert (rule-based), End-to-end (neural)
- **Seeds**: 0-9 (10 seeds)
- **Tasks**: head_on, crossing_feasible
- **AoA**: 60°（主线 combat-only scope）
- **Methods**: oracle_task_gate only（不含 baseline，避免 checkpoint 冲突）
- **Specialists**:
  - head_on: head_on-weighted checkpoint (obs_dim=19)
  - crossing_feasible: crossing-weighted checkpoint (obs_dim=18)

## 关键 Bug 修复
### Bug 1: virtual_point 配置缺失
**问题**: `jsbsim_hrl_oracle_task_gate_mvp.yaml` 中缺少 `virtual_point` 配置，导致 `VirtualPointGenerator` 使用默认值，与 specialist 训练时的配置不一致。

**影响**: head_on specialist 的 head_on 性能从 1.0 降到 0.1

**修复**: 在 comparison config 中添加完整的 `virtual_point` 和 `trajectory_prediction` 配置，与 specialist 的 checkpoint config 一致。

### Bug 2: 复现入口不一致
**问题**: `scripts/run_oracle_task_gate_mvp_10seed_pilot.sh` 指向 `mvp.yaml` 并包含 `baseline` 方法，但已验收 run 实际使用 `mvp_only.yaml` 且只跑 `oracle_task_gate`。

**修复**: 合并 `mvp_only.yaml` 到 `mvp.yaml` 作为唯一 canonical config，删除 deprecated 文件，脚本统一使用 `mvp.yaml`，`methods` 仅包含 `oracle_task_gate`。

### Bug 3: AoA scope 不匹配
**问题**: 已验收 run 的 `close_range_max_aoa_deg` 为 40.0，不是主线要求的 60.0。

**修复**: launcher 脚本显式添加 `--attack-zone-close-range-max-aoa-deg 60`。

## 实验结果（AoA 60 scope）

### Expert Opponent Stage

| Method | Task | Win Rate | Notes |
|--------|------|----------|-------|
| Oracle Task Gate | head_on | **1.0** (10/10) | head_on specialist 在 head_on 上完美表现 |
| Oracle Task Gate | crossing_feasible | **0.0** (0/10) | crossing specialist 在 crossing 上无胜场 |

### End-to-end Opponent Stage

| Method | Task | Win Rate | Notes |
|--------|------|----------|-------|
| Oracle Task Gate | head_on | **0.8** (8/10) | head_on specialist 稳健 |
| Oracle Task Gate | crossing_feasible | **0.5** (5/10) | crossing specialist 表现改善 |

## 结果分析

### 1. Head-on specialist 性能验证
head_on specialist 在 expert stage 上达到 **1.0 胜率**，证明了：
- Oracle task gate 的 specialist 路由逻辑正确
- head_on specialist 的 checkpoint 权重正确加载
- env 配置（含 virtual_point 和 trajectory_prediction）与 specialist 训练时一致

### 2. Crossing specialist 性能问题
crossing specialist 在 expert stage 上只有 **0.0 胜率**，远低于预期。

**原因分析**:
- crossing specialist 的 checkpoint 本身在 crossing 任务上表现差（obs_dim=18 的 specialist 在 crossing 上仍只有 0.0 胜率）
- 这不是 oracle gate 的 bug，而是 crossing specialist 本身在当前配置下对 crossing 任务无效

### 3. End-to-end 阶段表现
在 end-to-end 对手阶段，crossing specialist 的胜率上升到 0.5，表明 crossing specialist 在 end-to-end 对手下有一定效果，但在 expert 对手下完全无效。

## 文档与论文声明
- **本报告严格基于 AoA 60 scope 的实验产物**
- 论文中不应过早宣称 oracle gate 是 "upper bound"，因为 crossing specialist 本身失效
- 当前 oracle gate 结论：**head_on 强、crossing 仍坏**

## 下一步建议
1. **Audit crossing specialist 训练**：分析为什么 crossing-weighted specialist 在 crossing 上表现差
2. **如果 crossing specialist 修复后仍无效**，则转向 learned commander 训练
3. **如果 crossing specialist 可以修复**，则重新运行 oracle gate 验收，再评估 learned commander 的必要性
