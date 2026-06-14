# 本机战术机动动作库设计（JSBSim F-16）

## 1. 设计目标

为 JSBSim F-16 提供一个**参数化、可组合、带安全包线**的战术机动原语库，支持：

- 直线平飞
- 协调转弯
- 俯冲
- 筋斗（Loop）
- 桶滚（Barrel Roll）
- 高 Yo-Yo
- 低 Yo-Yo

该库与现有 `CloseRangeTrackingEnv` / JSBSim 后端解耦：输入为归一化的飞机状态 `FlightState`，输出为归一化的舵面/油门指令 `ControlCommand`，可直接驱动 JSBSim 的 `fcs/*-cmd-norm` 属性。

---

## 2. 架构

```text
src/uav_vpp_guidance/maneuver_library/
├── base.py                 # FlightState, ControlCommand, ManeuverSetpoint, Maneuver
├── controller.py           # 内环姿态/速率控制器
├── executor.py             # 机动执行器（选择、运行、切换、Fallback）
├── library.py              # 机动注册表/工厂
└── maneuvers/
    ├── straight_level.py
    ├── coordinated_turn.py
    ├── dive.py
    ├── loop.py
    ├── barrel_roll.py
    ├── high_yoyo.py
    └── low_yoyo.py
```

### 2.1 核心抽象

- `Maneuver`：每个机动是一个有限状态对象，实现
  - `can_enter(state)`：入口条件检查
  - `enter(state)`：初始化目标航向、计时器等
  - `update(state, dt) -> ManeuverSetpoint`：生成内环设定点
  - `is_complete(state)`：完成/退出条件
- `InnerLoopController`：把 `ManeuverSetpoint`（滚转角、俯仰率、过载、速度）映射到 `elevator/aileron/rudder/throttle`。
- `ManeuverExecutor`：管理当前机动，完成后切回 `fallback`（通常是 `StraightLevel`）。
- `ManeuverLibrary`：通过名称创建机动实例。

### 2.2 控制接口

库本身**不直接调用 JSBSim**。使用方（如 `JSBSimEnv` 或独立脚本）需要：

1. 从 JSBSim 属性构造 `FlightState`（单位转换：ft→m、ft/s→m/s、deg→rad）。
2. 调用 `executor.update(state, dt)` 得到 `ControlCommand`。
3. 把 `ControlCommand` 写回 JSBSim 属性：
   - `fcs/elevator-cmd-norm`
   - `fcs/aileron-cmd-norm`
   - `fcs/rudder-cmd-norm`
   - `fcs/throttle-cmd-norm`

---

## 3. 机动原语详细设计

| 机动 | 设定点 | 完成条件 | 入口约束 | 可行性 |
|---|---|---|---|---|
| **直线平飞** | `phi_ref=0`, `nz_ref=1.0`, `velocity_ref` | 持续时间 | 速度>80 m/s，高度>500 m | ✅ 高 |
| **协调转弯** | `phi_ref=±bank`, `nz_ref=1/cos(phi)`, `velocity_ref` | 航向误差<5° | 速度满足 `V ≥ 80·√n` | ✅ 高（受限于 F-16 持续转弯率） |
| **俯冲** | `theta_ref=-gamma`, `nz_ref=0.8`, `velocity_ref` | 高度≤目标 | 入口高度>目标+200 m | ✅ 高 |
| **筋斗** | `q_ref = ((nz-1)g)/V`, `nz_ref=5g`, `throttle=0.9` | 时间≥圆周时间 | 速度≥300 m/s，高度≥2500 m | ⚠️ 中（需足够能量和升降舵权限） |
| **桶滚** | `phi_ref` 线性扫过 360°，`nz_ref=1.2g` | 时间≥2π/roll_rate | 速度≥200 m/s，高度≥1500 m | ⚠️ 中（需方向舵协调，防止掉高度） |
| **高 Yo-Yo** | 两阶段：爬升转弯→俯冲恢复 | 速度/高度触发 | 速度、高度充裕 | ⚠️ 中（能量管理复杂，参数需调） |
| **低 Yo-Yo** | 两阶段：俯冲转弯→爬升恢复 | 速度触发 | 高度充裕 | ⚠️ 中（同上） |

---

## 4. JSBSim F-16 可行性评估

### 4.1 飞机模型能力

项目自带的 F-16A Block-32（`data/jsbsim/aircraft/f16/f16.xml`）是完整 6-DOF 模型：

- 升降舵、副翼、方向舵、油门、襟翼、起落架通道齐全。
- 最大可用法向过载约 **9g**。
- 最大滚转角速度远大于本库设定的 60°/s。
- 高空（5000 m / Mach ~0.8）持续转弯率约 **12–15°/s**，不足以完成大角度 crossing 机动，但足以支持 30–45° 协调转弯。

### 4.2 逐项可行性

| 机动 | F-16 物理能力 | 当前库实现成熟度 | 主要风险 |
|---|---|---|---|
| 直线平飞 | ✅ 完全支持 | ✅ 完整实现 | 无 |
| 协调转弯 | ✅ 完全支持 | ✅ 完整实现 | 大坡度需速度余量 |
| 俯冲 | ✅ 完全支持 | ✅ 完整实现 | 高度下限保护 |
| 筋斗 | ✅ 支持，但需能量 | ⚠️ 原型实现 | 进入速度不足会失速；升降舵权限在高攻角可能不够 |
| 桶滚 | ✅ 支持 | ⚠️ 原型实现 | 滚转过程中机头下掉；需方向舵协调 |
| 高 Yo-Yo | ✅ 支持 | ⚠️ 原型实现 | 阶段切换依赖速度阈值，需大量调参 |
| 低 Yo-Yo | ✅ 支持 | ⚠️ 原型实现 | 同上 |

### 4.3 当前未解决的问题

1. **没有攻角保护**：当前控制器不限制 `alpha`，大拉力筋斗可能触发深失速/尾旋。
2. **没有速度包线保护**：没有自动避开最小/最大速度。
3. **没有方向舵协调模型**：桶滚时仅使用侧滑角反馈，可能不够精确。
4. **没有 trim 接口**：执行机动前需要飞机处于配平状态。
5. **未在 JSBSim 上实测**：当前机器无 JSBSim，所有实现仅在单元测试层面验证逻辑。

---

## 5. 与现有项目的集成方式

### 5.1 作为独立 demo

```bash
python scripts/demo_maneuver_jsbsim.py --maneuver loop --steps 3000
```

该脚本会：
1. 加载 JSBSim F-16。
2. 从初始状态构造 `FlightState`。
3. 选择并执行一个机动。
4. 记录状态与指令到 CSV。

### 5.2 作为专家策略的一部分

可以在 `ExpertVPPPolicy` 之上再加一层 `TacticalManeuverSelector`：

```text
SituationEvaluator  →  TacticalManeuverSelector  →  ManeuverExecutor  →  JSBSim
```

例如：
- 防御/被追尾 → `HighYoYo` 或 `BarrelRoll`
- 进攻占位 → `CoordinatedTurn` + `Dive`
- 能量劣势 → `LowYoYo`

### 5.3 作为 RL 的动作空间

也可以把机动库封装成离散/连续混合动作空间：
- 离散部分：选择机动原语。
- 连续部分：调整机动参数（坡度、进入速度、持续时间等）。
- 这能显著降低 RL 探索难度，因为每个原语内部由控制器稳定执行。

---

## 6. 测试情况

- 单元测试：`tests/test_maneuver_library.py`
  - 注册表完整性
  - 基本机动设定点正确性
  - 内环控制器输出有界
  - 执行器能驱动机动完成
- 当前结果：6 passed。
- JSBSim 实机测试：待完成（本机 JSBSim 不可用）。

---

## 7. 后续工作建议

1. **在 JSBSim 上逐个机动做开环验证**，用 `scripts/demo_maneuver_jsbsim.py` 记录轨迹。
2. **增加包线保护模块**：
   - 攻角上限（如 `alpha_max = 25°`）
   - 过载上限（`nz_max = 7g`）
   - 最小速度/高度保护
3. **改进方向舵协调**：用协调转弯公式 `rudder ≈ f(beta, phi)` 或引入侧滑角 PID。
4. **为筋斗/桶滚增加基于姿态的程序控制**（而非仅依赖速率 PID）。
5. **把高/低 Yo-Yo 从两阶段状态机扩展为带目标航向变化的完整能量管理机动**。
6. **与现有 `CloseRangeTrackingEnv` 集成**，让 RL/专家策略可以调用机动库。

---

## 8. 结论

- **物理可行性**：JSBSim F-16 模型有能力完成本库所列的全部机动。
- **实现成熟度**：直线平飞、协调转弯、俯冲已实现并可单元测试；筋斗、桶滚、高/低 Yo-Yo 是**原型实现**，需在 JSBSim 上迭代调参。
- **主要障碍**：不是飞机模型能力，而是**内环控制器鲁棒性、攻角/速度包线保护、以及 JSBSim 实测验证**。
