# ExpertVPPPolicy 空战专家系统设计说明

## 1. 文档目的

本文档用于说明在 `E:\uav-vpp-guidance` 项目中设计一个空战机动决策专家系统的来龙去脉、设计目的、系统架构、关键指标、战术状态分类、V0 机动意图以及与现有 VPP、LOS-rate guidance、PPO、CV/CA 预测模块之间的关系。

该专家系统建议命名为：

```text
ExpertVPPPolicy
```

它不是为了替代论文主方法，而是作为一个**可解释的非学习对比基线**，用于回答以下问题：

1. 人工规则是否能在部分近距空战场景中达到稳定跟踪效果？
2. 学习型 PPO 策略相比专家规则是否有优势？
3. 轨迹预测增强的 VPP 方法相比人工规则是否有真实增益？
4. 在哪些场景中专家规则稳定，在哪些场景中规则失效？

---

## 2. 设计背景

当前项目主线是：

```text
obs → policy πθ → Δp → VirtualPointGenerator → LOSRateGuidance → backend → reward/done
```

其中：

- `policy πθ` 输出虚拟追踪点偏移 `Δp`；
- `VirtualPointGenerator` 将 `Δp` 叠加到目标当前位置或目标预测位置上，得到虚拟追踪点；
- `LOSRateGuidance` 根据虚拟追踪点输出 `Nz_cmd`、`roll_rate_cmd`、`throttle_cmd`；
- 后端环境可以是 `SimplePointMassEnv` 或 `JSBSimEnv`。

已经完成的实验链路包括：

1. No-Prediction VPP Baseline；
2. SimplePointMass 多场景评估；
3. JSBSim No-Prediction VPP Bridge；
4. No-Prediction PPO；
5. CV/CA 预测增强 VPP。

在此基础上加入专家系统的意义是：

> 提供一个不依赖梯度学习、不依赖轨迹预测训练、但具有明确战术逻辑的可解释对比基线。

它能够帮助论文避免只比较“随机策略”和“学习策略”的问题，使实验对照更加完整。

---

## 3. 为什么专家系统不应直接输出舵面或控制指令

专家系统有两种可能接入方式。

### 3.1 方案 A：直接输出控制指令

```text
ExpertSystem → Nz_cmd, roll_rate_cmd, throttle_cmd
```

优点：

- 接近传统空战机动控制；
- 可以直接表达“急转”“爬升”“俯冲”等动作。

缺点：

- 会绕过当前项目的 VPP + LOS 主线；
- 与 PPO 策略不在同一动作空间中，比较不公平；
- 需要额外处理底层控制器和 JSBSim 稳定性；
- 复杂战术动作可能引入新的控制问题；
- 不利于分析“虚拟追踪点策略”的效果。

### 3.2 方案 B：输出虚拟点偏移

```text
ExpertVPPPolicy → normalized Δp in [-1, 1]
```

然后继续走统一链路：

```text
Δp → VirtualPointGenerator → LOSRateGuidance → backend
```

优点：

- 与 PPO 使用同一动作空间；
- 复用现有 VPP、LOS、reward、termination、evaluation；
- 与 No-Prediction / CV / CA 方法公平比较；
- 不容易破坏已有工程架构；
- 更适合作为论文中的可解释规则基线。

因此，本项目建议优先采用方案 B。

---

## 4. ExpertVPPPolicy 的定位

ExpertVPPPolicy 的定位是：

```text
一个基于态势评估和规则推理的虚拟追踪点偏移生成器。
```

它的输入是环境观测和相对几何信息，输出是归一化动作：

```text
expert_action = [dx_norm, dy_norm, dz_norm]
```

其中每个分量范围为：

```text
[-1, 1]
```

该动作不直接表示米单位，而是交给 `VirtualPointGenerator` 统一映射为物理偏移。

---

## 5. 与其它组件的关系

### 5.1 与 PPO 的关系

PPO 策略：

```text
obs → neural policy → normalized Δp
```

ExpertVPPPolicy：

```text
obs + info → rule engine → normalized Δp
```

两者输出相同类型的动作，因此可以共用：

- `VirtualPointGenerator`；
- `LOSRateGuidance`；
- `RewardCalculator`；
- `TerminationChecker`；
- `evaluate_policy` 或类似评估框架；
- trajectory CSV；
- metrics；
- plotting scripts。

这使专家系统可以作为 PPO 的非学习基线。

### 5.2 与 CV/CA 预测器的关系

专家系统可以有两个版本：

1. No-Prediction ExpertVPP：

```text
Pos_Virtual = Pos_Target_Current + Δp_expert
```

2. Prediction-Enhanced ExpertVPP：

```text
Pos_Virtual = Pos_Target_Pred + Δp_expert
```

但 V0 阶段建议只做 No-Prediction ExpertVPP，避免和预测器耦合过早。

### 5.3 与 RuleBasedPursuit 的关系

现有 `RuleBasedPursuit` 更像简单追踪策略，例如 pure/lead/lag pursuit。

ExpertVPPPolicy 则更进一步：

- 先做态势评估；
- 判断攻防/中立/安全状态；
- 再选择机动意图；
- 最后生成 VPP 偏移。

因此，ExpertVPPPolicy 应作为强于简单 RuleBasedPursuit 的可解释基线。

### 5.4 与 JSBSim 的关系

ExpertVPPPolicy 不直接依赖 JSBSim。

它只输出 VPP 动作，因此同一套专家策略可以用于：

- `SimplePointMassEnv`；
- `JSBSimEnv`。

但需要注意：

- SimplePointMass 只用于机制验证；
- JSBSim 才能检查控制饱和、失速、坠毁、低层控制稳定性；
- 专家规则在 Simple 中可行，不代表在 JSBSim 中必然可行。

---

## 6. 总体架构

ExpertVPPPolicy V0 建议采用以下架构：

```text
ego state + target state + relative geometry
        ↓
SituationEvaluator
        ↓
tactical_state
        ↓
RuleEngine
        ↓
maneuver_intent
        ↓
ExpertVPPActionGenerator
        ↓
normalized Δp in [-1, 1]
        ↓
VirtualPointGenerator
        ↓
LOSRateGuidance
        ↓
SimplePointMassEnv / JSBSimEnv
```

对应模块可以设计为：

```text
src/uav_vpp_guidance/expert_system/
├── __init__.py
├── situation_evaluator.py
├── tactical_state.py
├── rule_engine.py
├── maneuver_intents.py
├── expert_vpp_policy.py
└── config.py
```

可选配置：

```text
config/experiment/expert_vpp_baseline.yaml
```

可选评估脚本：

```text
src/uav_vpp_guidance/evaluation/evaluate_expert_vpp.py
```

或复用统一 policy evaluation，只要将专家系统封装为具有 `act(obs, info)` 接口的 policy。

---

## 7. 输入与输出

### 7.1 输入

ExpertVPPPolicy 的输入应包括：

1. `obs`：环境观测向量；
2. `info`：环境 step 或 reset 中的诊断信息；
3. `relative_geometry`：相对几何；
4. `ego_state`：我机状态；
5. `target_state`：目标机状态。

优先使用已有字段：

```text
range_m
ata_deg
aspect_deg
los_rate
closure_rate
altitude_diff_m
ego_speed
target_speed
ego_altitude
target_altitude
```

如果部分字段暂时不存在，应允许从 position / velocity 中计算，或优雅降级。

### 7.2 输出

ExpertVPPPolicy 输出：

```text
expert_action: np.ndarray shape=(3,)
```

范围：

```text
[-1, 1]
```

语义：

```text
[forward_offset_norm, lateral_offset_norm, vertical_offset_norm]
```

注意：

- 不输出米单位；
- 不输出舵面；
- 不直接输出 `Nz_cmd`；
- 不绕过 `VirtualPointGenerator`。

---

## 8. 态势评估指标

V0 不应追求复杂完美的空战态势评估，而应选择简单、稳定、可解释、容易测试的指标。

### 8.1 距离指标

```text
range_m
```

建议分区：

| 距离区间 | 含义 |
|---|---|
| R < 500 m | 极近，可能超越或碰撞风险 |
| 500 m ≤ R < 1200 m | 近距优势窗口 |
| 1200 m ≤ R < 3000 m | 中近距机动区 |
| R ≥ 3000 m | 远距接近区 |

### 8.2 角度指标

核心指标：

```text
ATA: antenna train angle 或攻击角相关指标
aspect_deg: 目标相对姿态角
```

建议分区：

| 角度 | 含义 |
|---|---|
| ATA < 25° | 强优势/可保持攻击位置 |
| 25° ≤ ATA < 60° | 进攻态势但需修正角度 |
| 60° ≤ ATA < 120° | 中立或交叉态势 |
| ATA ≥ 120° | 角度不利或防御态势 |

### 8.3 闭合速度

```text
closure_rate = -d(range_m)/dt
```

含义：

- closure_rate > 0：我机正在接近目标；
- closure_rate < 0：目标正在远离我机；
- closure_rate 过大：可能有超越风险。

### 8.4 高度差

```text
altitude_diff_m = ego_altitude - target_altitude
```

含义：

- 高度优势可用于能量转换；
- 低高度需要安全保护；
- 过大高度差可能导致追踪困难。

### 8.5 速度比

```text
speed_ratio = ego_speed / max(target_speed, eps)
```

含义：

- speed_ratio > 1：我机更快，可能追上或超越；
- speed_ratio < 1：我机较慢，可能无法保持攻击位置。

### 8.6 能量代理指标

V0 可使用简化能量指标：

```text
energy_proxy = 0.5 * speed^2 + g * altitude
```

能量优势：

```text
energy_advantage = ego_energy_proxy - target_energy_proxy
```

注意：

该指标只是简化代理，不应在论文中夸大为严格的能量机动理论结论。

---

## 9. 态势优势函数

可以定义一个可调的态势优势分数：

```text
S = w_angle * angle_score
  + w_range * range_score
  + w_energy * energy_score
  + w_altitude * altitude_score
  + w_closure * closure_score
```

其中每个 score 归一化到大致范围：

```text
[-1, 1]
```

示例设计：

### 9.1 angle_score

```text
angle_score = 1 - clamp(ATA / 180, 0, 1) * 2
```

大致含义：

- ATA 越小越好；
- ATA 接近 0 时接近 +1；
- ATA 接近 180 时接近 -1。

### 9.2 range_score

期望距离可以设为：

```text
R_desired = 900 m
```

range_score 可以设计为：

```text
range_score = 1 - abs(range_m - R_desired) / R_scale
```

并裁剪到：

```text
[-1, 1]
```

### 9.3 energy_score

```text
energy_score = tanh(energy_advantage / energy_scale)
```

### 9.4 altitude_score

如果高度过低，则强烈惩罚：

```text
if ego_altitude < min_safe_altitude:
    altitude_score = -1
else:
    altitude_score = tanh(altitude_diff_m / altitude_scale)
```

---

## 10. 战术状态分类

V0 建议使用有限数量的战术状态，避免一开始设计过复杂。

### 10.1 STRONG_OFFENSIVE

强进攻态势。

建议条件：

```text
range_m <= 1200
ATA <= 30 deg
altitude_safe == true
```

策略倾向：

- 保持尾后；
- 避免超越；
- 使用 lag 或 pure pursuit；
- 控制闭合速度。

### 10.2 OFFENSIVE

一般进攻态势。

建议条件：

```text
range_m <= 2500
ATA <= 60 deg
```

策略倾向：

- 使用 lead pursuit 或 pure pursuit；
- 快速缩小角度与距离；
- 如闭合过快则转为 lag pursuit。

### 10.3 NEUTRAL

中立态势。

建议条件：

```text
60 deg < ATA < 120 deg
或 range 在中等范围但角度优势不明显
```

策略倾向：

- 建立角度优势；
- 使用 high_angle_cut；
- 保持能量。

### 10.4 DEFENSIVE

防御态势。

建议条件：

```text
ATA >= 120 deg
或目标处于我机后半球
或敌方角度明显占优
```

策略倾向：

- break / extend；
- 拉开距离；
- 通过虚拟点偏置制造转弯；
- 避免低空俯冲。

### 10.5 UNSAFE

安全优先态势。

建议条件：

```text
ego_altitude < min_safe_altitude
或 speed < min_safe_speed
或 near out_of_bounds
```

策略倾向：

- altitude_recover；
- 减小激进转弯；
- throttle 增加；
- 优先保命而不是追踪。

---

## 11. V0 机动意图

V0 不建议直接实现高 Yo-Yo、低 Yo-Yo、剪式、桶滚、Split-S、Immelmann 等复杂动作。这些是多阶段动作序列，需要持续状态机、阶段切换、安全边界和更稳定低层控制器。

V0 只实现以下 6 个机动意图。

### 11.1 PURE_PURSUIT

目的：直接追踪目标当前位置。

输出倾向：

```text
Δp ≈ [0, 0, 0]
```

适用场景：

- 中等距离；
- 角度不大；
- 没有明显超越风险。

### 11.2 LEAD_PURSUIT

目的：追踪目标前方位置，提前抢占角度。

输出倾向：

```text
Δp 沿目标速度方向前置
```

适用场景：

- 进攻态势；
- 距离偏大；
- 需要快速减小 ATA。

### 11.3 LAG_PURSUIT

目的：追踪目标后方位置，避免超越。

输出倾向：

```text
Δp 沿目标速度反方向后置
```

适用场景：

- 强进攻态势；
- 距离过近；
- 闭合速度过大。

### 11.4 HIGH_ANGLE_CUT

目的：在中立或角度较差时快速建立角度优势。

输出倾向：

```text
Δp 偏向目标侧前方
```

适用场景：

- ATA 中等偏大；
- range 仍可接受；
- 不适合直接纯追。

### 11.5 EXTEND

目的：拉开距离、重新组织态势。

输出倾向：

```text
Δp 指向远离目标或更安全方向
```

适用场景：

- 防御态势；
- 距离过近且角度不利；
- 有超越或失控风险。

### 11.6 ALTITUDE_RECOVER

目的：低空安全恢复。

输出倾向：

```text
Δp_z > 0
```

适用场景：

- 高度低于安全阈值；
- JSBSim 中接近坠毁或失速风险。

---

## 12. 规则推理逻辑示例

V0 可以使用优先级规则，而不是复杂行为树。

示例伪代码：

```python
if unsafe:
    intent = ALTITUDE_RECOVER
elif strong_offensive:
    if closure_rate_too_high or range_m < 700:
        intent = LAG_PURSUIT
    else:
        intent = PURE_PURSUIT
elif offensive:
    if ata_deg > 35:
        intent = LEAD_PURSUIT
    else:
        intent = PURE_PURSUIT
elif defensive:
    if altitude_safe:
        intent = EXTEND
    else:
        intent = ALTITUDE_RECOVER
else:
    intent = HIGH_ANGLE_CUT
```

规则顺序建议：

1. 安全规则优先；
2. 防御规则次之；
3. 强进攻规则；
4. 一般进攻规则；
5. 中立规则；
6. fallback 规则。

---

## 13. ExpertVPPActionGenerator 设计

ExpertVPPActionGenerator 负责把机动意图转化为归一化 `Δp`。

### 13.1 输入

```text
maneuver_intent
relative_geometry
target_velocity_direction
ego_velocity_direction
config parameters
```

### 13.2 输出

```text
action_norm = [dx_norm, dy_norm, dz_norm]
```

范围必须裁剪到：

```text
[-1, 1]
```

### 13.3 示例映射

| intent | dx_norm | dy_norm | dz_norm |
|---|---:|---:|---:|
| PURE_PURSUIT | 0.0 | 0.0 | 0.0 |
| LEAD_PURSUIT | +0.5 | 0.0 | 0.0 |
| LAG_PURSUIT | -0.5 | 0.0 | 0.0 |
| HIGH_ANGLE_CUT | +0.3 | ±0.5 | 0.0 |
| EXTEND | -0.8 | ±0.3 | 0.0 |
| ALTITUDE_RECOVER | 0.0 | 0.0 | +0.8 |

注意：

这里的 `dx/dy/dz` 是相对于 VPP 坐标约定的归一化偏移，不应直接当作世界坐标米单位。实际实现中需要明确坐标系约定，例如 target velocity frame、LOS frame 或世界 NEU frame。

---

## 14. 坐标系建议

为了避免专家系统误实现，建议 V0 使用简单且稳定的坐标系。

### 推荐方案：目标速度坐标系

定义：

- x 轴：目标速度方向；
- y 轴：水平侧向方向；
- z 轴：竖直方向。

优点：

- lead / lag 语义清楚；
- CV/CA predicted target 也容易结合；
- 比直接在世界坐标中写死偏移更稳。

V0 如果暂时没有完整坐标变换，也可以先在世界坐标中近似实现，但必须在文档中说明局限。

---

## 15. 配置建议

建议新增配置：

```yaml
expert_vpp:
  enabled: true
  policy_type: expert_vpp_v0

  situation:
    strong_offensive_range_m: 1200.0
    offensive_range_m: 2500.0
    strong_offensive_ata_deg: 30.0
    offensive_ata_deg: 60.0
    defensive_ata_deg: 120.0
    min_safe_altitude_m: 300.0
    min_safe_speed_mps: 80.0
    close_range_m: 700.0
    high_closure_rate_mps: 120.0

  weights:
    angle: 0.4
    range: 0.25
    energy: 0.2
    altitude: 0.1
    closure: 0.05

  action:
    pure: [0.0, 0.0, 0.0]
    lead: [0.5, 0.0, 0.0]
    lag: [-0.5, 0.0, 0.0]
    high_angle_cut: [0.3, 0.5, 0.0]
    extend: [-0.8, 0.3, 0.0]
    altitude_recover: [0.0, 0.0, 0.8]
```

---

## 16. 输出 info 字段

专家系统应在 `info` 或 trajectory CSV 中记录诊断信息：

```text
expert_enabled
expert_tactical_state
expert_maneuver_intent
expert_situation_score
expert_angle_score
expert_range_score
expert_energy_score
expert_altitude_score
expert_closure_score
expert_action_x
expert_action_y
expert_action_z
expert_rule_id
expert_fallback_reason
```

这些字段非常重要，因为专家系统的价值就在于可解释。

---

## 17. 评估指标

ExpertVPPPolicy 应与 PPO/CV/CA 使用相同主指标：

```text
success_rate
score_win_rate
out_of_bounds_rate
crash_rate
timeout_rate
mean_final_range
mean_final_ata_deg
mean_min_range
mean_min_ata_deg
mean_episode_return
mean_episode_length
time_to_first_advantage
advantage_hold_time
```

额外专家系统指标：

```text
tactical_state_distribution
maneuver_intent_distribution
fallback_rate
unsafe_rate
mean_situation_score
rule_usage_count
```

---

## 18. 推荐实验设计

### 18.1 V0 验证

先在 SimplePointMass 中验证：

```text
RuleBasedPursuit vs ExpertVPPPolicy
```

场景：

1. favorable；
2. neutral；
3. disadvantage；
4. challenging。

目标：

- ExpertVPPPolicy 不出现 NaN；
- 不产生异常虚拟点；
- 能正常输出动作；
- 至少在 favorable/neutral 中表现不差于随机策略。

### 18.2 与 PPO/CV/CA 比较

然后比较：

```text
RuleBasedPursuit
ExpertVPPPolicy
No-Prediction PPO
CV-Prediction PPO
CA-Prediction PPO
```

注意：

不要只报总平均，必须按场景报告。

### 18.3 JSBSim sanity evaluation

在 Simple 中跑通后，将 ExpertVPPPolicy 放到 JSBSim 后端做少量 episode：

- 检查 crash；
- 检查 stall；
- 检查 OOB；
- 检查控制饱和；
- 检查低空安全规则是否发挥作用。

---

## 19. 测试建议

### 19.1 SituationEvaluator 测试

1. favorable 输入应输出 OFFENSIVE 或 STRONG_OFFENSIVE；
2. 低高度输入应输出 UNSAFE；
3. 大 ATA 输入应输出 DEFENSIVE 或 NEUTRAL；
4. 所有 score 不应 NaN；
5. 缺少部分字段时应 fallback。

### 19.2 RuleEngine 测试

1. UNSAFE 优先级最高；
2. STRONG_OFFENSIVE + close range + high closure → LAG_PURSUIT；
3. OFFENSIVE + high ATA → LEAD_PURSUIT；
4. DEFENSIVE → EXTEND；
5. 无规则匹配 → fallback 到 PURE_PURSUIT 或 HIGH_ANGLE_CUT。

### 19.3 ExpertVPPPolicy 测试

1. 输出 shape=(3,)；
2. 输出范围在 [-1,1]；
3. 不输出 NaN/inf；
4. 不调用 PPO；
5. 不调用 predictor，除非明确设计 Prediction-Expert 版本；
6. 能在环境中 step 若干步不报错。

### 19.4 评估脚本测试

1. ExpertVPP evaluation smoke 能跑通；
2. trajectory CSV 包含 expert diagnostic fields；
3. metrics CSV 包含 maneuver intent distribution；
4. 输出全部在 outputs/。

---

## 20. 不建议 V0 实现的内容

V0 不建议实现：

1. 完整行为树；
2. 高 Yo-Yo；
3. 低 Yo-Yo；
4. 剪式飞行；
5. 桶滚；
6. Split-S；
7. Immelmann；
8. 博弈论对手建模；
9. 遗传算法优化规则；
10. 直接输出 JSBSim 舵面。

原因：

- 这些动作是多阶段动作；
- 需要持续状态机；
- 对高度和速度安全边界敏感；
- 与当前 VPP 方法主线耦合度低；
- 会显著增加调试难度；
- 容易导致论文主线分散。

---

## 21. 后续版本路线

### V0：ExpertVPPPolicy

- 态势评估；
- 规则推理；
- 输出归一化 Δp；
- 复用 VPP + LOS。

### V1：ExpertVPPPolicy + 预测锚点

- 支持 current_target；
- 支持 predicted_target；
- 对比 Expert-NoPred 与 Expert-CV/CA。

### V2：有限状态机机动执行器

- 引入持续动作；
- 支持 break turn、extend、climb 等多步动作；
- 管理动作持续时间和退出条件。

### V3：战术动作库

- 高/低 Yo-Yo；
- 剪式；
- 桶滚；
- Split-S；
- Immelmann。

### V4：规则优化

- 离线搜索规则阈值；
- 遗传算法或贝叶斯优化；
- 不建议过早实现。

---

## 22. 论文写法建议

在论文中可以这样描述：

> 为提供可解释的非学习对比基线，本文构建了 ExpertVPPPolicy。该策略基于距离、角度、闭合速度、高度和能量代理等指标进行态势评估，并通过规则推理选择 pure pursuit、lead pursuit、lag pursuit、extend、altitude recovery 等虚拟追踪点偏置。ExpertVPPPolicy 不通过梯度学习训练，且与 PPO 策略共享相同的 VPP 和 LOS-rate guidance 下层结构，因此能够作为学习型 VPP 策略的公平可解释对照。

需要避免的表述：

```text
顶尖飞行员水平
最优战术专家系统
完整真实空战机动库
完全复现人类战术直觉
```

更稳妥的表述：

```text
规则驱动的可解释基线
专家启发式策略
基于态势评估的 Expert-VPP baseline
```

---

## 23. 风险与防错要点

### 风险 1：专家系统绕过 VPP

防止方式：

- V0 只输出归一化 Δp；
- 禁止直接输出舵面或 `Nz_cmd`。

### 风险 2：规则过多导致无法解释

防止方式：

- V0 只保留 5 至 6 个意图；
- 所有规则记录 rule_id；
- 输出 maneuver_intent_distribution。

### 风险 3：规则参数拍脑袋

防止方式：

- 参数放入 YAML；
- 后续用 sensitivity analysis 分析；
- 不在论文中宣称参数最优。

### 风险 4：专家系统表现太差

防止方式：

- 明确其定位是 baseline，不是主方法；
- 与 RuleBasedPursuit、随机策略比较；
- 重点报告其可解释性和局限性。

### 风险 5：专家系统喧宾夺主

防止方式：

- 论文主线仍然是“轨迹预测增强 VPP 自主决策”；
- 专家系统只作为非学习对比基线；
- 复杂机动库放到后续工作或附录。

---

## 24. 最小实现清单

如果进入 ExpertVPPPolicy V0，建议最小实现如下：

新增文件：

```text
src/uav_vpp_guidance/expert_system/__init__.py
src/uav_vpp_guidance/expert_system/situation_evaluator.py
src/uav_vpp_guidance/expert_system/rule_engine.py
src/uav_vpp_guidance/expert_system/expert_vpp_policy.py
config/experiment/expert_vpp_baseline.yaml
src/uav_vpp_guidance/evaluation/evaluate_expert_vpp.py
src/uav_vpp_guidance/visualization/plot_expert_vpp_results.py
docs/expert_vpp_policy.md
tests/test_expert_vpp_policy.py
```

验收目标：

1. pytest 全部通过；
2. ExpertVPPPolicy 输出 shape=(3,)；
3. 输出范围为 [-1,1]；
4. favorable/neutral/challenging/disadvantage 均可评估；
5. trajectory CSV 包含 expert fields；
6. metrics CSV 包含专家规则统计；
7. SimplePointMass smoke 通过；
8. JSBSim sanity episode 可运行；
9. 不破坏 PPO/CV/CA 原有流程。

---

## 25. 总结

ExpertVPPPolicy 是可行且有价值的，但必须控制复杂度。

推荐结论：

```text
先做 ExpertVPPPolicy V0：态势评估 + 规则推理 + 输出归一化 Δp。
```

不推荐一开始做：

```text
完整战术动作库 + 行为树 + 博弈优化 + 舵面控制。
```

它在论文中的最佳定位是：

```text
一个规则驱动、可解释、与 PPO 共享 VPP/LOS 下层结构的非学习基线。
```

这样既能丰富实验对照，又不会偏离论文主线“融合轨迹预测的无人机单机近距空战自主决策方法研究”。
