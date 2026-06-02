# ExpertVPPPolicy 实现文档

## 概述

`ExpertVPPPolicy` 是一个基于规则的可解释空战专家系统，作为 PPO 等学习型策略的非学习对比基线。它与 PPO 共享相同的动作空间（归一化 Δp）和下层结构（VirtualPointGenerator + LOSRateGuidance），确保公平比较。

## 文件结构

```text
src/uav_vpp_guidance/expert_system/
├── __init__.py
├── situation_evaluator.py   # 态势评估：计算战术状态和优势分数
├── rule_engine.py           # 规则引擎：映射态势到机动意图
└── expert_vpp_policy.py     # 主策略：整合评估+规则+动作生成

config/experiment/expert_vpp_baseline.yaml
src/uav_vpp_guidance/evaluation/evaluate_expert_vpp.py
tests/test_expert_vpp_policy.py
```

## 模块说明

### SituationEvaluator

输入：`own_state`, `target_state`, `rel_state`

输出：
- `tactical_state`: `UNSAFE | STRONG_OFFENSIVE | OFFENSIVE | NEUTRAL | DEFENSIVE`
- `scores`: angle / range / energy / altitude / closure 分项分数
- `situation_score`: 加权综合态势分数
- `diagnostics`: 详细诊断信息

关键参数（可在 YAML 中配置）：
- `strong_offensive_range_m`: 1200 m
- `offensive_range_m`: 2500 m
- `strong_offensive_aspect_deg`: 30°
- `offensive_aspect_deg`: 60°
- `defensive_aspect_deg`: 120°
- `min_safe_altitude_m`: 300 m
- `min_safe_speed_mps`: 80 m/s

### RuleEngine

基于优先级的规则推理：

1. `UNSAFE` → `ALTITUDE_RECOVER`
2. `DEFENSIVE` + altitude_safe → `EXTEND`
3. `DEFENSIVE` + !altitude_safe → `ALTITUDE_RECOVER`
4. `STRONG_OFFENSIVE` + (close_range 或 high_closure) → `LAG_PURSUIT`
5. `STRONG_OFFENSIVE` + 否则 → `PURE_PURSUIT`
6. `OFFENSIVE` + high_aspect + far_range → `LEAD_PURSUIT`
7. `OFFENSIVE` + 否则 → `PURE_PURSUIT`
8. `NEUTRAL` / fallback → `HIGH_ANGLE_CUT`

### ExpertVPPPolicy

将机动意图转换为目标速度坐标系中的归一化偏移，再旋转到世界 NEU 坐标系：

| 意图 | dx_tv | dy_tv | dz_tv | 说明 |
|------|-------|-------|-------|------|
| PURE_PURSUIT | 0.0 | 0.0 | 0.0 | 直接追踪目标 |
| LEAD_PURSUIT | +0.5 | 0.0 | 0.0 | 目标前方前置 |
| LAG_PURSUIT | -0.5 | 0.0 | 0.0 | 目标后方后置 |
| HIGH_ANGLE_CUT | +0.3 | ±0.5 | 0.0 | 侧向切入 |
| EXTEND | -0.8 | ±0.3 | 0.0 | 拉开距离 |
| ALTITUDE_RECOVER | 0.0 | 0.0 | +0.8 | 高度恢复 |

输出裁剪到 `[-1, 1]`，shape=(3,)。

## 使用方法

### 评估（Simple 后端）

```bash
python -m uav_vpp_guidance.evaluation.evaluate_expert_vpp \
    --config config/experiment/expert_vpp_baseline.yaml \
    --backend simple \
    --episodes 10 --seeds 0 1 2 \
    --save-trajectories
```

### 评估（JSBSim 后端）

```bash
python -m uav_vpp_guidance.evaluation.evaluate_expert_vpp \
    --config config/experiment/expert_vpp_baseline.yaml \
    --backend jsbsim \
    --episodes 2 --seeds 0
```

### 在代码中使用

```python
from uav_vpp_guidance.expert_system import ExpertVPPPolicy
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.utils.config import load_yaml_config

config = load_yaml_config("config/experiment/expert_vpp_baseline.yaml")
env = CloseRangeTrackingEnv(config)
policy = ExpertVPPPolicy(config.get("expert_vpp", {}))

obs = env.reset(seed=0)
for step in range(env.max_steps):
    rel_state = obs["relative_state"]
    own_state = obs["own_state"]
    target_state = obs["target_state"]
    action = policy.get_action(own_state, target_state, rel_state)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

## 诊断字段

每个 step 的 `info` 可通过 `policy.get_last_diagnostics()` 获取：

- `expert_enabled`
- `expert_tactical_state`
- `expert_maneuver_intent`
- `expert_situation_score`
- `expert_angle_score` / `range_score` / `energy_score` / `altitude_score` / `closure_score`
- `expert_action_x` / `y` / `z`
- `expert_rule_id`
- `expert_fallback_reason`

## 测试

```bash
pytest tests/test_expert_vpp_policy.py -v
```

覆盖：
- 态势评估（favorable/low_altitude/large_angle/neutral/missing_fields）
- 规则引擎（优先级/强进攻滞后/进攻前置/防御/中立）
- 策略输出（shape/bounds/NaN/诊断/坐标转换）

## 设计约束

V0 明确**不实现**以下内容：
- 完整行为树
- 高/低 Yo-Yo、剪式、桶滚、Split-S、Immelmann
- 博弈论对手建模
- 遗传算法优化规则
- 直接输出 JSBSim 舵面

这些留给后续版本（V2+）或论文附录。

## 论文定位

> 一个规则驱动、可解释、与 PPO 共享 VPP/LOS 下层结构的非学习基线。

用于回答：
1. 人工规则是否能在部分近距空战场景中达到稳定跟踪效果？
2. 学习型 PPO 策略相比专家规则是否有优势？
3. 轨迹预测增强的 VPP 方法相比人工规则是否有真实增益？
4. 在哪些场景中专家规则稳定，在哪些场景中规则失效？
