# JSBSim 环境适配分析报告与配置迁移方案

**日期**: 2026-06-11  
**分析目标**: 检查现有实验配置与 JSBSim 后端的兼容性，设计完整的配置迁移方案  
**分析范围**: `config/method_innovation_comparison.yaml`, `config/experiment/no_prediction_vpp_jsbsim.yaml`, `config/experiment/no_prediction_vpp_scenarios.yaml`, `config/env.yaml`, `config/ppo.yaml`  

---

## 一、兼容性分析总结

### 1.1 完全兼容项（无需修改）

| 项 | 状态 | 说明 |
|---|---|---|
| 场景初始条件格式 | ✅ 兼容 | `tracking_env.py` 的 `_scenario_to_jsbsim_init()` 已完整实现 `position_m`/`velocity_mps`/`heading_deg` → JSBSim `ic/*` 的转换（含 `neu2lla` 地理坐标转换） |
| 所有 4 类场景 | ✅ 兼容 | favorable / neutral / disadvantage / challenging 均使用统一格式，全部可通过 `_scenario_to_jsbsim_init` 转换 |
| 环境统一接口 | ✅ 兼容 | `reset()`/`step()`/`get_state()` 在两种后端下接口一致，代码层自动路由 |
| 观测空间 | ✅ 兼容 | `build_observation()` 使用 `compute_relative_geometry()`，后端无关 |
| 终止条件 | ✅ 兼容 | `TerminationChecker` 基于 `rel_state` 判断，后端无关 |
| 网络架构 | ✅ 兼容 | PPO 策略网络输入维度不变（14-dim），与后端无关 |
| 课程学习框架 | ✅ 兼容 | `CurriculumManager` 基于 `success_rate` 触发 stage 切换，后端无关 |

### 1.2 需要修改项

| 项 | 当前值 | 需要改为 | 影响程度 | 说明 |
|---|---|---|---|---|
| `backend` | `simple` | `jsbsim` | 🔴 高 | 核心切换参数 |
| `use_jsbsim` | `false` | `true` | 🔴 高 | 环境初始化开关 |
| `strict_backend` | 未设置 | `true` | 🟡 中 | 禁止 fallback 到 simple |
| `origin` | 未设置 | `[120.0, 60.0, 0.0]` | 🟡 中 | JSBSim 地理坐标原点 |
| `low_level_dt` | 未设置 | `0.0166667` | 🟡 中 | JSBSim 积分步长（1/60） |
| `action_repeat` | 未设置 | `12` | 🟡 中 | 每决策步 JSBSim 运行次数 |
| `actuator_dynamics` | `enabled: true` | 需重新评估 | 🟡 中 | JSBSim 已有真实舵面动态，可能重复 |
| `angle_reward` 公式 | `-(ata+aa)/180` | 需修正 | 🔴 高 | 公式无物理意义，见 RL 审计 |
| `turn_rate_penalty` | 初始惩罚 crossing | 需修正 | 🔴 高 | 导致 JSBSim 下 crossing 0% |
| `dynamics_aware` | 默认 `false` | 建议 `true` | 🟡 中 | 开启 VPP 动力学感知偏移 |
| `stage_gate_sr` | `0.50` | 可能需调整 | 🟡 中 | JSBSim 训练更慢，gate 可能过严 |
| `total_steps` / `rollout_steps` | 1024 / 256 | 可能需增加 | 🟡 中 | JSBSim 训练效率更低 |

### 1.3 参考对比：CloseAirCombat_control vs 本项目

| 参数 | CloseAirCombat_control | 本项目 (simple) | 本项目 (JSBSim 目标) |
|---|---|---|---|
| 默认高度 | 20000 ft (6096m) | 5000m | 5000m（保留） |
| 默认速度 | 800 fps (243.84 m/s) | 180-280 m/s | 180-280 m/s（保留） |
| 仿真频率 | 60 Hz | 60 Hz (simple) | 60 Hz |
| 决策频率 | 5 Hz | 5 Hz | 5 Hz |
| action_repeat | 12 | 无（simple 无此概念） | 12 |
| low_level_dt | 1/60 ≈ 0.0167 | 无 | 0.0166667 |
| 初始经度 | 120.0° | 无（NEU 坐标） | 120.0° |
| 初始纬度 | 60.0° | 无（NEU 坐标） | 60.0° |
| 飞机模型 | F-16 | 质点模型 | F-16 |
| 动作空间 | 舵面 [-1,1] + 油门 [0.4,0.9] | 3D 加速度 + 滚转率 | 通过制导律映射到舵面 |

**关键差异说明**：
- CloseAirCombat_control 默认高度 6096m，本项目场景 5000m——F-16 在 5000m 的机动性能（最大过载、转弯率）与 6096m 不同，需确认 F-16 气动模型在此高度是否收敛
- 速度范围 180-280 m/s（约 0.53-0.82 Ma @ 5000m），F-16 在该速度包线内可控
- 动作空间差异：本项目策略输出 `VPP 偏移` → `LOS 制导律` → `nz_cmd/roll_rate_cmd/throttle_cmd` → `低层控制器` → `JSBSim 舵面`，与 CloseAirCombat 的 `策略输出舵面` 不同，但 `JSBSimEnv` 通过 `LowLevelController` 已封装此映射

---

## 二、关键问题清单（必须修复）

### 2.1 🔴 高优先级

1. **奖励函数 `angle_reward` 公式错误**（RL 审计已确认）
   - 当前：`angle_reward = -w_angle * (ata_deg + aa_deg) / 180.0`
   - 问题：`ATA + AA` 求和无物理意义，可能产生正值奖励（当两者之和 < 0 时）
   - 修复建议：使用 `max(ata_deg, aa_deg)` 或 `atan2 综合角`，确保 [0, 180] 单调映射到 [-w_angle, 0]

2. **`turn_rate_penalty` 初始惩罚 crossing**（RL 审计已确认）
   - 当前：当 `range < 3000m` 且 `heading_error > 60°` 时惩罚
   - 问题：crossing 场景（初始 AA ≈ 90°）满足此条件，被初始惩罚，导致 crossing 成功率 0%
   - 修复建议：加入 `range > 1000m` 豁免，或仅在 closing 时激活

3. **配置切换为 JSBSim 后端**
   - 所有实验配置文件需统一修改 `backend` 和 `use_jsbsim`

### 2.2 🟡 中优先级

4. **`actuator_dynamics` 在 JSBSim 下可能重复**
   - JSBSim F-16 模型已有真实舵面速率限制（如副翼 max rate ~ 60°/s）
   - 额外 `actuator_dynamics` 层可能叠加延迟，导致响应过慢
   - 建议：在 JSBSim 下禁用 `actuator_dynamics`，或大幅降低 tau（如 0.05s）

5. **`dynamics_aware` 默认关闭**
   - VPP 生成器在 crossing 场景下需预测目标未来位置
   - 建议：开启 `dynamics_aware`（虽然仍使用 simple 预测模型，但比 static 好）

6. **课程学习 gate 可能过严**
   - JSBSim 每个 run 预计 40-60 分钟（simple 为 25-30 分钟）
   - `stage_gate_sr=0.50` 在 JSBSim 下可能需要更多样本才能达到
   - 建议：初始阶段降低至 0.40，逐步提升到 0.50

### 2.3 🟢 低优先级

7. **`total_steps` 和 `rollout_steps` 调整**
   - JSBSim 下每个 episode 的实际仿真步数更多（12 steps/decision × 512 decisions = 6144 JSBSim steps）
   - 但 PPO 的 `rollout_steps` 定义的是 **决策步数**，不是仿真步数，因此 1024 决策步在两种后端下等价
   - 可能需要增加 `total_steps` 以补偿 JSBSim 更慢的收敛

8. **F-16 在 5000m 的性能验证**
   - 需要确认 F-16 在 5000m、180-280 m/s 包线内的 trim 稳定性
   - 建议添加 `test_f16_trim_at_5000m.py` 验证脚本

---

## 三、配置迁移方案

### 3.1 新配置文件：`config/method_innovation_jsbsim.yaml`

基于 `method_innovation_comparison.yaml` 修改，创建全新的 JSBSim 实验配置：

```yaml
# Unified JSBSim comparison config for Method Innovation Track 1 vs Track 2.
# All algorithms share the same JSBSim environment, network, curriculum, and base PPO hyperparams.
includes:
  - env.yaml
  - ppo.yaml
  - guidance.yaml
  - reward.yaml

experiment:
  name: method_innovation_jsbsim
  seed: 0
  mode: train
  output_root: outputs

# Backend selection — JSBSIM ONLY
backend: jsbsim

env:
  use_jsbsim: true
  strict_backend: true        # Never fall back to simple
  aircraft_model: f16
  origin: [120.0, 60.0, 0.0]  # JSBSim geodetic origin

  # Time stepping — JSBSim-specific
  sim_freq: 60
  decision_freq: 5
  high_level_dt: 0.2
  low_level_dt: 0.0166667      # 1/60, JSBSim integration step
  action_repeat: 12             # sim_freq // decision_freq
  max_high_level_steps: 512

  target_mode: constant_velocity

  # Termination thresholds (same as simple config)
  success_range_m: 900.0
  success_ata_deg: 25.0
  success_hold_time_s: 0.2
  hysteresis_range_m: 950.0
  hysteresis_ata_deg: 30.0
  min_altitude_m: 500.0
  max_altitude_m: 15000.0
  max_range_m: 12000.0

# Actuator dynamics: DISABLED in JSBSim mode
# JSBSim F-16 already has realistic control surface rate limits and delays.
actuator_dynamics:
  enabled: false

# Curriculum learning (same as simple config, gate may need tuning)
curriculum:
  stages:
    - [0.25, ["favorable", "neutral"]]
    - [0.50, ["favorable", "neutral", "disadvantage"]]
    - [0.75, ["favorable", "neutral", "disadvantage", "challenging"]]
    - [1.00, ["favorable", "neutral", "disadvantage", "challenging"]]
  stage_gate_sr: 0.50          # May need adjustment to 0.40 for early stages

# Virtual point generator
dynamics_aware: true            # Enable VPP prediction for crossing scenarios

virtual_point:
  anchor_mode: current_target
  return_info: true
  action_dim: 3
  d_long_range: [-1500.0, 1500.0]
  d_lat_range: [-800.0, 800.0]
  d_vert_range: [-500.0, 500.0]
  smoothing_alpha: 0.3

# Guidance law (same)
guidance:
  mode: los_rate
  use_gain_adapter: false
  gains:
    k_los: 1.0
    k_pos: 0.5
    k_damp: 0.2
    k_roll: 1.0
    k_speed: 0.2
    alpha_filter: 0.3

# Command limits (same)
limits:
  nz_min: -2.0
  nz_max: 7.0
  roll_rate_min: -1.5
  roll_rate_max: 1.5
  throttle_min: 0.0
  throttle_max: 1.0
```

### 3.2 奖励函数修复文件：`config/reward_jsbsim_fixes.yaml`

```yaml
# Reward function fixes for JSBSim backend
# These overrides apply ON TOP of reward.yaml

reward:
  # Fix 1: angle_reward — replace incorrect (ATA+AA) formula
  # Old: angle_reward = -w_angle * (ata_deg + aa_deg) / 180.0
  # New: angle_reward = -w_angle * max(ata_deg, aa_deg) / 180.0
  angle_reward_formula: "max_ata_aa"
  # Alternative: "atan2_combined" for more sophisticated angle metric

  # Fix 2: turn_rate_penalty — add close-range exemption for crossing
  # Old: penalize when range < 3000m AND heading_error > 60°
  # New: penalize when range < 3000m AND range > 1000m AND heading_error > 60° AND closing
  turn_rate_penalty:
    enabled: true
    range_min_m: 1000.0        # Exempt very close range (crossing passes)
    range_max_m: 3000.0
    heading_error_threshold_deg: 60.0
    require_closing: true      # Only penalize if range_rate < 0 (closing)
    w_turn_rate: 0.3

  # Keep all other reward weights unchanged
  w_range: 0.5
  w_angle: 0.8
  w_energy: 0.2
  w_safety: 2.0
  w_saturation: 1.0
  w_smooth: 0.1
  terminal_success: 200.0
  terminal_failure: -200.0
  terminal_crash: -300.0
  min_altitude_m: 500.0

  potential_based_shaping:
    enabled: true
```

### 3.3 场景配置：复用现有 `no_prediction_vpp_scenarios.yaml`

场景初始条件完全兼容 JSBSim，无需修改。所有场景均通过 `_scenario_to_jsbsim_init` 自动转换：

```yaml
scenarios:
  favorable:
    own_init:
      position_m: [0.0, 0.0, 5000.0]    # → ic/h-sl-ft: 16404.2, ic/long-gc-deg: 120.0, ic/lat-geod-deg: 60.0
      velocity_mps: 280.0                # → ic/u-fps: 918.6
      heading_deg: 0.0                   # → ic/psi-true-deg: 0.0
    target_init:
      position_m: [800.0, 0.0, 5000.0]   # → ic/long-gc-deg: 120.0072, ic/lat-geod-deg: 60.0
      velocity_mps: 180.0                # → ic/u-fps: 590.6
      heading_deg: 0.0
  # ... (neutral, disadvantage, challenging same as current)
```

---

## 四、代码修改清单

### 4.1 `config/method_innovation_comparison.yaml` → 新建 `config/method_innovation_jsbsim.yaml`

| 修改项 | 原值 | 新值 | 行号 |
|---|---|---|---|
| `backend` | `simple` | `jsbsim` | 18 |
| `use_jsbsim` | `false` | `true` | 19 |
| `strict_backend` | 缺失 | `true` | 新增 |
| `origin` | 缺失 | `[120.0, 60.0, 0.0]` | 新增 |
| `low_level_dt` | 缺失 | `0.0166667` | 新增 |
| `action_repeat` | 缺失 | `12` | 新增 |
| `actuator_dynamics.enabled` | `true` | `false` | 76 |
| `dynamics_aware` | 缺失 | `true` | 新增 |

### 4.2 `src/uav_vpp_guidance/envs/reward.py` — 奖励函数修复

**修复 1：`angle_reward`**
```python
# 原代码（约第 180-190 行）:
# angle_reward = -w_angle * (ata_deg + aa_deg) / 180.0

# 修改为:
max_angle = max(ata_deg, aa_deg)
angle_reward = -w_angle * max_angle / 180.0
```

**修复 2：`turn_rate_penalty`**
```python
# 原代码（约第 200-220 行）:
# if range_m < 3000 and heading_error > 60:
#     penalty = -w_turn_rate * (heading_error - 60) / 120

# 修改为:
if (range_m < 3000 and range_m > 1000 and 
    heading_error > 60 and range_rate < 0):  # require closing
    penalty = -w_turn_rate * (heading_error - 60) / 120
```

### 4.3 新增验证脚本：`tests/test_jsbsim_scenarios.py`

```python
"""验证所有 4 类场景在 JSBSim 下正确初始化并运行至少 10 步。"""
```

---

## 五、AI 助手调通关键词

以下关键词用于指导 AI 助手（或 Kimi）完成配置适配和验证：

### 5.1 配置迁移关键词

```
"将 method_innovation_comparison.yaml 迁移到 JSBSim 后端"
"创建 method_innovation_jsbsim.yaml，基于现有配置修改以下字段：
- backend: jsbsim
- use_jsbsim: true
- strict_backend: true
- origin: [120.0, 60.0, 0.0]
- low_level_dt: 0.0166667
- action_repeat: 12
- actuator_dynamics.enabled: false
- dynamics_aware: true
"
```

### 5.2 奖励函数修复关键词

```
"修复 reward.py 中两个已知问题：
1. angle_reward 公式：将 -(ata+aa)/180 改为 -max(ata, aa)/180
2. turn_rate_penalty：添加 range > 1000m 豁免和 require_closing 条件
"
```

### 5.3 JSBSim 验证关键词

```
"运行 JSBSim 验证脚本：
1. pytest tests/test_jsbsim_env_p1.py（确认已通过）
2. 运行 tests/test_jsbsim_scenarios.py（验证 4 类场景初始化）
3. 运行单 episode  smoke test：python scripts/smoke_test_jsbsim.py --scenario favorable --steps 50
"
```

### 5.4 训练启动关键词

```
"在 machine2 上启动 JSBSim 训练：
1. git checkout CL_CRPPO_CEMGD
2. 确保 data/jsbsim/ 存在（自包含）
3. python scripts/train.py --config config/method_innovation_jsbsim.yaml --algo cr_ppo
4. 监控 wandb 日志，确认 backend=jsbsim，初始场景类型正确
"
```

### 5.5 问题排查关键词

```
"JSBSim 训练失败排查：
1. 检查 jsbsim_data_dir 是否解析到项目内 data/jsbsim/
2. 检查 F-16 在 5000m 初始化是否成功（run_ic()）
3. 检查 actuators 是否饱和：查看 info['actuator_saturation'] 统计
4. 检查 crossing 场景是否仍然 0%：如果奖励修复后仍为 0%，检查 turn_rate_penalty 范围豁免是否生效
5. 检查 episode 长度是否异常短（< 50 步）：可能 altitude 或 velocity 初始化超出 F-16 包线
"
```

---

## 六、风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|---|---|---|---|
| F-16 在 5000m 初始化失败 | 中 | 高 | 添加 `test_f16_trim_at_5000m.py` 验证脚本 |
| JSBSim 训练收敛极慢 | 高 | 高 | 禁用 `actuator_dynamics`（避免重复延迟）；考虑增加 `total_steps` 到 10M |
| crossing 场景仍然 0% | 中 | 高 | 确认奖励修复生效；如仍失败，考虑移除 `turn_rate_penalty` 或进一步放宽 |
| 课程学习 gate 无法达到 | 中 | 中 | 初始 `stage_gate_sr` 降至 0.40，后期提升到 0.50 |
| 训练时间超预期 | 高 | 中 | Machine 2 独占运行；使用 Machine 3 并行运行 ablation；考虑用 simple 预训练再迁移到 JSBSim |
| 论文对比不公平 | 中 | 中 | 确保两种后端的 `scenario_types`、`success_criteria`、`random_seed` 完全一致；论文中明确声明两种后端的差异 |

---

## 七、执行顺序建议

1. **Phase 1 — 配置迁移**（本地，30 min）
   - 创建 `config/method_innovation_jsbsim.yaml`
   - 修复 `reward.py`（angle_reward + turn_rate_penalty）
   - 添加 `dynamics_aware: true`
   - 验证配置解析： `python -c "from utils.config import load_config; load_config('config/method_innovation_jsbsim.yaml')"`

2. **Phase 2 — JSBSim 场景验证**（本地，20 min）
   - 新增 `tests/test_jsbsim_scenarios.py`（4 类场景各运行 10 步）
   - 运行 `pytest tests/test_jsbsim_scenarios.py -v`
   - 确认无初始化失败、无异常终止

3. **Phase 3 — 单 episode Smoke Test**（本地，10 min）
   - `python scripts/smoke_test_jsbsim.py --scenario favorable --steps 100`
   - 检查：episode 长度 > 50 步，reward 范围合理，无 crash

4. **Phase 4 — Machine 2 训练启动**（远程，立即执行）
   - 上传最新代码（含 JSBSim 配置）
   - 启动 CR-PPO + JSBSim 训练：`python scripts/train.py --config config/method_innovation_jsbsim.yaml --algo cr_ppo`
   - 同时启动 Baseline PPO + JSBSim 作为对照
   - 监控前 10 个 episode 的 success_rate 和 episode_length

5. **Phase 5 — 迭代调参**（根据 Phase 4 结果）
   - 如果 crossing 0% → 检查奖励修复是否生效
   - 如果 episode 过短 → 检查 F-16 初始化高度/速度是否超出包线
   - 如果收敛过慢 → 调整 `stage_gate_sr` 或 `total_steps`

---

## 附录 A：场景初始条件 JSBSim 转换示例

以 **favorable** 场景为例：

| 参数 | 场景值 | JSBSim 转换值 | 计算 |
|---|---|---|---|
| own altitude | 5000.0 m | ic/h-sl-ft: 16404.2 | 5000 / 0.3048 |
| own velocity | 280.0 m/s | ic/u-fps: 918.6 | 280 / 0.3048 |
| own heading | 0.0° | ic/psi-true-deg: 0.0 | 直接传递 |
| own position | [0, 0, 5000] | ic/long-gc-deg: 120.0, ic/lat-geod-deg: 60.0 | 原点偏移为 0 |
| target altitude | 5000.0 m | ic/h-sl-ft: 16404.2 | 同上 |
| target velocity | 180.0 m/s | ic/u-fps: 590.6 | 180 / 0.3048 |
| target heading | 0.0° | ic/psi-true-deg: 0.0 | 直接传递 |
| target position | [800, 0, 5000] | ic/long-gc-deg: ~120.0072, ic/lat-geod-deg: 60.0 | neu2lla(800, 0, 5000, 120, 60, 0) |

**验证**：`ic/u-fps=918.6` 对应约 280 m/s，F-16 在 5000m、0.8 Ma 附近可稳定飞行。`ic/h-sl-ft=16404` 在 F-16 飞行包线内。

---

## 附录 B：与 CloseAirCombat_control 的关键差异对照

| 维度 | CloseAirCombat_control | 本项目（JSBSim 目标） | 影响 |
|---|---|---|---|
| 任务层级 | 端到端（策略输出舵面） | 分层（策略输出 VPP → 制导律 → 舵面） | 本项目动作空间更抽象，但 JSBSim 接口一致 |
| 观测空间 | 16-dim（含绝对位置/姿态） | 14-dim（相对几何） | 观测空间设计不同，但 JSBSim 数据提取已兼容 |
| 初始条件 | 硬编码 120°/60°/20000ft | 场景驱动（NEU 转 LLA） | 本项目更灵活，已通过 `neu2lla` 转换 |
| 奖励函数 | 高度/姿态/速度/距离/事件 | 距离/角度/能量/安全/饱和/平滑 | 完全不同，但均在 `reward.py` 中独立实现 |
| 终止条件 | 低空/极端状态/过载/超时 | 距离/角度/高度/最大距离/超时 | 不同，但均在 `termination.py` 中独立实现 |
| 低层控制 | 无（直接输出舵面） | `LowLevelController`（制导律→舵面） | 本项目需要 `LowLevelController` 将 `nz_cmd` 等映射到 JSBSim 舵面属性 |

---

*报告生成时间: 2026-06-11*  
*基于代码版本: CL_CRPPO_CEMGD 分支*  
*JSBSim 数据版本: 自包含 data/jsbsim/（约 4MB）*
