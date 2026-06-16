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
- 剪式（Scissors）
- Split-S
- Immelmann

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
    ├── low_yoyo.py
    ├── scissors.py
    ├── split_s.py
    └── immelmann.py
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

| 机动 | 设定点 | 完成条件 | 入口约束 | JSBSim 验证结果 |
|---|---|---|---|---|
| **直线平飞** | `phi_ref=0`, `nz_ref=1.0`, `velocity_ref` | 持续时间 | 速度>80 m/s，高度>500 m | ✅ 完成，高度/速度保持平稳 |
| **协调转弯** | `phi_ref=±bank`, `nz_ref=1/cos(phi)`, `velocity_ref` | 航向误差<5° | 速度满足 `V ≥ 80·√n` | ✅ 完成，掉高 ~905 m，出口速度 183 m/s |
| **俯冲** | `theta_ref=-gamma`, `nz_ref=0.8`, `velocity_ref` | 高度≤目标 | 入口高度>目标+200 m | ✅ 完成（γ=25°，5000→3000 m） |
| **筋斗** | `nz_ref` 程序控制，油门=1.0 | 回到入口高度且俯仰≤10° | 速度≥300 m/s，高度≥2500 m | ✅ 完成（nz=3g，出口 200 m/s，高度基本恢复） |
| **桶滚** | `phi_ref` 线性扫过 360°，`nz_ref=1.2g` | 时间≥2π/roll_rate | 速度≥200 m/s，高度≥1500 m | ✅ 完成，掉高 ~156 m，带坡度余量 |
| **高 Yo-Yo** | 三阶段：爬升转弯→俯冲→改平 | 速度/高度触发 | 速度、高度充裕 | ✅ 完成（6000→5150 m，速度 195 m/s） |
| **低 Yo-Yo** | 三阶段：俯冲转弯→爬升→改平 | 速度触发 | 高度充裕 | ✅ 完成（带 2000 m 硬高度保护） |
| **剪式** | 左右急转交替，`phi_ref=±bank`，`nz_ref=1/cos(bank)+margin` | 完成指定周期数 | 速度与高度裕度 | ✅ 完成（45°/60° 周期，掉高至 2885 m） |
| **Split-S** | 滚转 180° → 拉杆半筋斗 → 改平 | 滚转/俯仰恢复水平 | 速度≥300 m/s，高度裕度 | ✅ 完成（10000→964 m，需大高度裕度） |
| **Immelmann** | 拉杆半筋斗 → 顶端滚转 180° → 改平 | 滚转/俯仰恢复水平 | 速度≥300 m/s，顶端高度裕度 | ⚠️ 完成但出口速度仅 92 m/s；需要更高进入能量或更小 g 负荷 |

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
| 俯冲 | ✅ 完全支持 | ✅ 完整实现 | 已加高度下限保护 |
| 筋斗 | ✅ 支持，但需能量 | ✅ 已实现 | 需 nz≤3g 并留足改出时间 |
| 桶滚 | ✅ 支持 | ✅ 已实现 | 滚转中机头略下掉，方向舵协调可接受 |
| 高 Yo-Yo | ✅ 支持 | ✅ 已实现 | 阶段切换已改为高度/姿态触发，更安全 |
| 低 Yo-Yo | ✅ 支持 | ✅ 已实现 | 已加 2000 m 硬高度保护 |
| 剪式 | ✅ 支持 | ✅ 已实现 | 当前为自包含模式，未结合对手态势 |
| Split-S | ✅ 支持 | ✅ 已实现 | 需 ≥10 km 起始高度或更小 g 负荷 |
| Immelmann | ✅ 支持 | ⚠️ 原型实现 | 顶端能量低，出口速度仅 ~92 m/s；需更高进入能量 |

### 4.3 已增加的包线保护

1. **最小高度保护**：`ManeuverSetpoint.min_altitude_m` + 控制器硬拉杆改出。
2. **最小速度保护**：`ManeuverSetpoint.min_speed_mps` → 自动加满油门并限制拉杆量。
3. **最大攻角保护**：`ManeuverSetpoint.max_alpha_rad` → 自动推杆减小攻角。
4. **Yo-Yo 硬高度地板**：`LowYoYo` / `HighYoYo` 在接近设定下限时强制转入爬升/改平。
5. **Loop / Split-S / Immelmann 姿态程序控制**：用俯仰角积分或高度/姿态判据替代纯时间判据，改出更可靠。

### 4.4 仍然存在的局限

1. **Immelmann 能量不足**：从典型巡航速度（~330 m/s）进入时出口速度仅 ~92 m/s，实战需预加速或选择更低 g 负荷。
2. **Split-S 高度消耗大**：10000 m 起始高度仅能改出到 ~964 m，低空不可用。
3. **方向舵协调仍较粗略**：仅基于侧滑角反馈，滚转/偏航耦合未精细建模。
4. **剪式机动为自包含模式**：未利用对手相对态势，未来应接入 `SituationEvaluator`。

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
- 当前结果：**10 passed**。
- JSBSim 实机测试：**已完成全 10 机动扫测**，结果见 `outputs/maneuver_demo/summary.csv` / `summary.md` 与本文档第 4 节。

---

## 7. 后续工作建议

1. **改进 Immelmann 能量管理**：增加预加速阶段或自适应 g 负荷，使出口速度 ≥120 m/s。
2. **优化 Split-S 改出**：降低进入 g 负荷或增加起始高度，保留更多安全高度余量。
3. **改进方向舵协调**：用协调转弯公式 `rudder ≈ f(beta, phi)` 或引入侧滑角 PID。
4. **把高/低 Yo-Yo 扩展为带目标航向变化的完整能量管理机动**。
5. **为剪式机动引入对手相对态势输入**（如 target bearing/closure），实现真正的防御性剪刀机动。
6. **与现有 `CloseRangeTrackingEnv` 集成**，让 RL/专家策略可以调用机动库。

---

## 8. 结论

- **物理可行性**：JSBSim F-16 模型有能力完成本库所列的全部机动；在调参后 10/10 机动均可在验证配置下完成。
- **实现成熟度**：直线平飞、协调转弯、俯冲、筋斗、桶滚、高/低 Yo-Yo、剪式、Split-S 已实现并在 JSBSim 上验证；Immelmann 仍属原型，能量管理需继续优化。
- **主要障碍**：不是飞机模型能力，而是**高 g 垂直机动的能量管理（Loop/Immelmann/Split-S）、低空安全余量，以及与高层战术决策的集成**。