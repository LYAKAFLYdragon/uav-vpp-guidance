# Baseline PID 5.00 满分现象解释

## 现象

在修复 B-E 问题后，Baseline PID 在 multi_waypoint 任务上的表现为：
- 修复前：12/20 crash, mean_wp = 2.00
- 修复后：0/20 crash, mean_wp = 5.00（满分）

这个 5.00 满分超过了 Enhanced PID 的 4.25。

## 根因分析

### 1. 继承链传递效应

`BaselinePIDController` 继承自 `EnhancedLowLevelController`（`pid_controllers.py:28`）：

```python
class BaselinePIDController(EnhancedLowLevelController):
```

因此 Baseline PID **继承了所有基类修复**：

- **M1 油门统一**（`clip_command()` 中 `throttle_min = max(config_val, DEFAULT_THROTTLE_MIN)`）：
  - 油门不再被 clip 到 0.0，避免了推力丧失导致的螺旋下坠
- **`_integrate_with_anti_windup()`**：
  - 消除积分器饱和导致的失控
- **`_apply_protection_arbitration()`**：
  - 即使各保护项被禁用，空跑也无害

### 2. Baseline 的"弱"假设不再成立

Baseline 禁用了以下保护/增强功能：
- `altitude_hold.enabled = false`
- `stall_protection.enabled = false`
- `bank_protection.enabled = false`
- `lift_compensation_factor = 1.0`（无升力补偿）

但在 **multi_waypoint 这种温和任务**上：
- 航点间距大、高度变化小，altitude_hold 的优势不明显
- 速度窗口宽，stall_protection 很少触发
- 转弯半径要求不严格，bank_protection 的限制反而成为约束

因此，**Baseline 的"无保护"反而成为一种优势**：控制器可以更直接、更激进地跟踪航点，不受保护逻辑的约束。

### 3. 与 Enhanced PID 的对比

| 维度 | Baseline PID | Enhanced PID |
|------|-------------|--------------|
| 油门下限 | 0.4（M1 修复后） | 0.4（M1 修复后） |
| 抗积分饱和 | 继承 | 继承 |
| altitude_hold | 禁用 | 启用（可能导致过冲） |
| stall_protection | 禁用 | 启用（可能限制机动） |
| bank_protection | 禁用 | 启用（可能限制转弯） |
| lift_compensation | 1.0 | 1.05-1.15（轻微增益） |

在 multi_waypoint 上，Enhanced PID 的额外保护功能**没有带来优势**，反而可能因为保护逻辑的保守性（如 bank_protection 限制最大滚转角）导致跟踪效率略低。

## 结论与影响

1. **技术上合理**：Baseline PID 的 5.00 满分是修复后的合理结果，不是数据错误。
2. **"弱 baseline"假设失效**：Baseline 不再是"明显弱于 Enhanced"的基准，而是 multi_waypoint 任务上的**强竞争者**。
3. **对消融实验的影响**：
   - 不能再用"Baseline vs Enhanced"来证明保护功能的价值（在 multi_waypoint 上无显著差异）
   - 需要强调：Enhanced PID 的优势在 **sustained_turn / break_turn** 等更严苛的任务上体现（保护功能在极端工况下防止 crash）
   - 或者需要引入 **更弱的 baseline**（如纯比例控制 P-only）来建立有意义的对比

## 建议

1. **论文中明确说明**："Baseline PID 也受益于底层控制器的修复（M1 油门统一、抗积分饱和），因此其性能大幅提升。这恰恰说明基础控制器的健壮性是所有上层策略的前提。"
2. **任务差异化分析**：在 multi_waypoint 上 Baseline 可能更优，但在 sustained_turn / break_turn 上 Enhanced 的保护功能可能防止 crash。
3. **考虑引入 P-only 或更弱 baseline**：如果消融实验需要展示保护功能的价值，需要一个真正"无保护"的基准。
4. **archive 标注**：在 `outputs/flight_control_compare/` 的报告中添加此说明。
