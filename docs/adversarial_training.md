# 对抗训练环境使用说明

## 1. 设计目标

`AdversarialJSBSimEnv` 是一个基于 JSBSim 的双机追击-逃逸环境：

- **追击方（pursuer / blue）**：使用项目核心的 VPP → LOS 制导律 → 底层控制器链路。
- **目标方（target / red）**：既可以作为 RL 智能体直接输出 `[nz, roll_rate, throttle]`，也可以切换为规则驱动的机动库（`maneuver_library`）或专家策略（`ExpertVPPPolicy`）。

该环境主要用于：

1. 两阶段对抗训练（先训练逃逸 target，再微调追击 pursuer）。
2. 评估学习策略对规则/机动目标的鲁棒性。
3. 作为课程学习的第三阶段对手（hard opponent）。

## 2. 关键修复（相对早期版本）

| 问题 | 修复方式 |
|---|---|
| 默认重置使两机初始位置重合 | `reset()` 在 `scenario=None` 时构造默认尾追 2000 m 场景 |
| `PerturbedJSBSimEnv` 风扰动未生效 | 改为 `jsbsim_env._aircraft[uid].set_property_value(...)` |
| 传感器噪声与配置脱节 | 按 16 维观测的物理含义逐项加噪 |
| 评估配置缺少 `ppo.yaml`，网络尺寸不匹配 | `evaluate.yaml` / `robustness_eval.yaml` 已 include `../ppo.yaml` 并补充 `target_policy` |
| `reset(seed=...)` 未真正生效 | 内部维护 `self._rng`，不再把 seed 传给底层 `JSBSimEnv` |
| 缺少 `truncated` 语义 | `step()` 现在返回 7 元组，`timeout` 对应 `truncated=True` |

## 3. 配置入口

### 3.1 控制器类型

在 `env:` 段中通过两个键切换控制器：

```yaml
env:
  pursuer_controller_type: "rl"            # 或 "expert_vpp"
  target_controller_type: "rl"             # 或 "maneuver_library"

  bandit:                                  # target_controller_type == "maneuver_library" 时生效
    fallback_maneuver: "straight_level"
    selector:
      reaction_time_s: 0.5
      min_dwell_time_s: 1.0

expert_vpp: {}                             # pursuer_controller_type == "expert_vpp" 时生效
```

### 3.2 场景采样器

训练配置中可启用场景随机化：

```yaml
scenario_sampler:
  enabled: true
  source: registry_set          # 或 "explicit_list"
  set: smoke_test               # registry_set 时有效
  seed: 42
```

`source: explicit_list` 示例：

```yaml
scenario_sampler:
  enabled: true
  source: explicit_list
  names: ["smoke_tail_chase", "smoke_head_on", "smoke_crossing_left"]
  seed: 42
```

## 4. 常用命令

### 4.1 两阶段训练

```bash
# Phase 1: 训练 target（逃逸方）
python scripts/train_adversarial_target.py --config config/adversarial/train_target.yaml

# Phase 2: 微调 pursuer（追击方）
python scripts/train_adversarial_pursuer.py --config config/adversarial/train_pursuer.yaml
```

### 4.2 Smoke 测试

```bash
python scripts/train_adversarial_target.py --config config/adversarial/train_target.yaml --smoke
```

### 4.3 评估

```bash
# 四种 pursuer-target 组合评估
python scripts/evaluate_adversarial.py --config config/adversarial/evaluate.yaml

# 多扰动等级鲁棒性评估
python scripts/evaluate_robustness.py --config config/adversarial/robustness_eval.yaml
```

## 5. 规则对手 smoke 示例

以下 Python 片段演示 **ExpertVPP 追击方 vs 机动库目标方**：

```python
import numpy as np
from uav_vpp_guidance.envs.adversarial_jsbsim_env import AdversarialJSBSimEnv
from uav_vpp_guidance.utils.config import load_yaml_config, merge_config
import os

# 加载并修改配置
cfg = load_yaml_config("config/adversarial/train_target.yaml")
cfg["env"]["pursuer_controller_type"] = "expert_vpp"
cfg["env"]["target_controller_type"] = "maneuver_library"
cfg["env"]["bandit"] = {"fallback_maneuver": "straight_level"}
cfg["expert_vpp"] = {}

env = AdversarialJSBSimEnv(cfg)
p_obs, t_obs = env.reset(seed=42)

for _ in range(20):
    p_obs, t_obs, p_rew, t_rew, term, trunc, info = env.step(
        np.zeros(3), np.zeros(3)          # target_action 在机动库模式下被忽略
    )
    print(info.get("target_maneuver"), p_obs["relative_state"]["range_m"])
    if term or trunc:
        break

env.close()
```

## 6. 测试

新增测试位于 `tests/`：

```bash
pytest tests/test_adversarial_env_smoke.py \
       tests/test_target_reward.py \
       tests/test_perturbed_env.py \
       tests/test_adversarial_config_consistency.py -v
```

- `test_adversarial_env_smoke.py`：默认非重合重置、seed 可复现、truncated、机动库 target、ExpertVPP pursuer。
- `test_target_reward.py`：target reward 符号与数值。
- `test_perturbed_env.py`：风扰动、传感器噪声分量、延迟缓冲。
- `test_adversarial_config_consistency.py`：配置 include 与网络尺寸一致性。

## 7. 已知限制与后续工作

- `ExpertVPPPolicy` 当前仅支持作为 **pursuer**；target 侧规则规避可直接使用机动库。
- 机动库目标在每个决策步更新一次，控制指令在 12 个 JSBSim 子步内保持恒定。
- 扰动环境的风场目前对所有飞机设置相同值，未考虑空间相关性。
- 传感器噪声的 `direction_std_deg` 配置项暂未使用。
