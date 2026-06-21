# 实现计划：air-combat-mvp（空战决策 MVP 第一阶段）

## 概述（Overview）

本实现计划将设计文档转化为一系列增量式、可由代码生成 LLM 顺序执行的编码任务。整体采用 **test-driven、先纯逻辑后集成** 的推进策略：先搭建子包骨架与 `Missile3DoF` 导弹模型（含单测与属性测试），再实现 `FireControlRadar` 雷达，再实现 `AirCombatSparseReward` 稀疏奖励；随后核对父类 `CloseRangeTrackingEnv.step` 结构，据此实现 `AirCombatMVPEnv` 环境与集成测试；最后接入 `reward.py` 工厂、编写场景配置与规则策略验证脚本，端到端核对通过标准。

实现语言为 **Python**（设计文档全程使用 Python 与 `hypothesis`，无需另选语言）。属性测试使用 `hypothesis`，每条属性最少 100 次迭代，并以注释标注 `# Feature: air-combat-mvp, Property {编号}: {属性文本}`。环境级集成测试统一使用 `backend: simple`，不依赖 JSBSim。

约定：
- **运行环境（全局约束）：全部测试与实验在 Python 3.11 环境执行。** 所有涉及运行命令的任务（导入冒烟检查、各检查点、现有测试回归检查、验证脚本、端到端验证）均须在 Python 3.11 解释器/虚拟环境下运行 `pytest` 与脚本（例如 `python3.11 -m pytest ...` 或在已激活的 py3.11 虚拟环境中运行）。
- 标记 `*` 的子任务为可选（单元测试 / 集成测试等增强），核心实现与属性测试为必做。
- **属性测试（PBT）子任务为必做（非可选）**，因为设计含完整正确性属性章节（16 条）。
- 每条正确性属性恰由一个属性测试实现，并标注其属性编号与所验证的需求条款。

## 任务（Tasks）

- [x] 1. 搭建 weapons / sensors 子包骨架
  - 创建 `src/uav_vpp_guidance/weapons/__init__.py`（导出 `Missile3DoF`）
  - 创建 `src/uav_vpp_guidance/sensors/__init__.py`（导出 `FireControlRadar`）
  - 创建 `src/uav_vpp_guidance/weapons/missile.py` 与 `src/uav_vpp_guidance/sensors/radar.py` 的空模块占位（含模块 docstring 说明 JSBSim 坐标系约定 X=North, Y=Up, Z=East，单位米/秒/弧度）
  - 检查项目打包配置（`pyproject.toml` / `setup.py`）以确保新子包被安装注册：本仓库 `pyproject.toml` 使用 `[tool.setuptools.packages.find]`（`where = ["src"]`），`find` 会自动包含新增子目录，通常无需手动登记；若改用显式 `packages=` 列表，则需手动添加 `uav_vpp_guidance.weapons` 与 `uav_vpp_guidance.sensors`
  - 增加导入冒烟检查（在 Python 3.11 环境下运行）：`python -c "from uav_vpp_guidance.weapons import Missile3DoF; from uav_vpp_guidance.sensors import FireControlRadar"`，确认新子包可在不依赖 JSBSim 的前提下导入
  - _Requirements: 8.2_

- [x] 2. 实现 Missile3DoF 常量、构造与状态量
  - [x] 2.1 定义 `Missile3DoF` 类常量与构造函数
    - 在 `weapons/missile.py` 中定义全部气动/性能/包线/制导常量（`DRAG_COEFF=0.6`、`LIFT_COEFF=0.15`、`REF_AREA=0.4`、`MAX_G=30.0`、`ENGINE_BURN_TIME=5.0`、`THRUST=20000.0`、`INITIAL_MASS=400.0`、`BURN_RATE=25.0`、`LAUNCH_RANGE_MIN=500.0`、`LAUNCH_RANGE_MAX=5000.0`、`LAUNCH_ATA_MAX_DEG=30.0`、`MAX_FLIGHT_TIME_S=20.0`、`MAX_RANGE_M=5000.0`、`NAV_CONSTANT=3.0`、`KILL_RADIUS_M=30.0`、`AIR_DENSITY=1.225`、`INITIAL_SPEED_MPS=500.0`、模块级 `GRAVITY=9.80665`）
    - 实现 `__init__(config, nav_constant, kill_radius_m)`：支持 config 字典覆盖常量；初始化全部状态量（`position_m`、`velocity_mps`、`mass_kg`、`flight_time_s`、`flight_distance_m`、`in_flight=False`、`hit=False`、`expired=False`、`_last_target_state=None`）为"未发射/未在飞"
    - _Requirements: 1.1, 1.2, 1.13, 1.14, 1.15, 8.2_

  - [x] 2.2 实现 JSBSim 坐标系速度分解与反解辅助函数
    - 实现速度分解 `vel = [speed*cos(gamma)*cos(psi), speed*sin(gamma), speed*cos(gamma)*sin(psi)]`（v_up 正向上，明确不沿用参考项目负号写法）
    - 实现反解 `speed=|vel|`、`gamma=arcsin(v_up/speed)`、`psi=arctan2(v_east, v_north)`，零速度兜底
    - _Requirements: 8.5, 8.6_

  - [x] 2.3 为速度分解可逆性编写属性测试（property-based testing，必做）
    - **Property 9：速度分解—反解可逆性**
    - 需要 property-based testing；用 hypothesis 生成 (speed≥0, gamma∈[-π/2, π/2], psi∈(-π, π])，分解后反解应在数值误差内一致，最少 100 次迭代
    - 注释标注 `# Feature: air-combat-mvp, Property 9`
    - **Validates: Requirements 8.5**

- [x] 3. 实现 Missile3DoF 发射判定（can_launch / launch）
  - [x] 3.1 实现 can_launch 几何发射判定
    - 计算 `range = |target.pos - ego.pos|` 与 `ATA = angle(ego.velocity, LOS)`；返回允许当且仅当 `500 <= range <= 5000` 且 `|ATA| < 30°`
    - 仅检查几何条件（距离与 |ATA|），不检查雷达视场/锁定；所有除法加 `1e-8` 防奇异
    - _Requirements: 1.7, 1.8, 1.9_

  - [x] 3.2 实现 launch 发射初始化
    - 从本机状态发射：`position_m` 取本机位置副本；速度方向取本机速度单位向量，量值取 `max(INITIAL_SPEED_MPS, |ego.velocity|)`；置 `mass_kg=INITIAL_MASS`、`flight_time_s=0`、`in_flight=True`；速度≈0 时方向退化为北向单位向量
    - _Requirements: 8.3_

  - [x] 3.3 为发射包线判定编写属性测试（property-based testing，必做）
    - **Property 2：发射包线判定的充要性**
    - 需要 property-based testing；用 hypothesis 生成覆盖包线内外与边界的本机/目标状态，断言 can_launch 充要性，最少 100 次迭代
    - 注释标注 `# Feature: air-combat-mvp, Property 2`
    - **Validates: Requirements 1.7, 1.8, 1.9**

  - [ ]* 3.4 为发射包线边界编写单元测试
    - 测试距离边界 500 / 5000、ATA 边界 30°、零距离等具体示例与错误条件
    - _Requirements: 1.7, 1.8_

- [x] 4. 实现 Missile3DoF 3-DoF 动力学步进（step）
  - [x] 4.1 实现 PNG 制导加速度与过载限幅
    - 计算 `rel_r`、`rel_v`、`omega = cross(rel_r, rel_v)/dot(rel_r, rel_r)`、标准真比例导引 `a_png = K*cross(omega, V_missile)`（`V_missile` 为导弹速度向量）；范数超过 `MAX_G*GRAVITY≈294.2` 时按比例限幅；除法加 `1e-8`（注：取代早先量纲被抑制的 `K*(dot(rel_r,rel_v)/r^4)*cross(rel_r, omega)` 形式）
    - _Requirements: 1.3, 1.4_

  - [x] 4.2 为制导加速度过载限幅编写属性测试（property-based testing，必做）
    - **Property 1：制导加速度过载限幅**
    - 需要 property-based testing；用 hypothesis 生成任意相对几何，断言施加的 PNG 制导加速度范数 ≤ `MAX_G*GRAVITY`，最少 100 次迭代
    - 注释标注 `# Feature: air-combat-mvp, Property 1`
    - **Validates: Requirements 1.4, 8.4**

  - [x] 4.3 实现推力/阻力/升力/重力与欧拉积分
    - 推力：`flight_time_s <= ENGINE_BURN_TIME` 时 `a_thrust=(THRUST/mass_kg)*v_hat`，否则为 0
    - 阻力（经验修正公式）：`q=0.5*AIR_DENSITY*|v|^2`，`F_drag = 0.05*DRAG_COEFF/3*q*REF_AREA`，`a_drag=-(F_drag/mass_kg)*v_hat`
    - 升力：`F_lift=q*LIFT_COEFF*REF_AREA`，方向取竖直—速度平面内与速度正交的单位向量；重力 `a_gravity=[0,-GRAVITY,0]`
    - 合成半隐式欧拉积分（Symplectic Euler：先更新速度，再用更新后的速度更新位置）：更新 `velocity_mps`、`position_m`、`flight_distance_m`、`flight_time_s`；燃烧段按 `BURN_RATE*dt` 递减 `mass_kg`（下限钳到燃尽质量 `burnout_mass`）
    - 缓存 `_last_target_state`；`in_flight=False` 时 `step` 流程最前直接 `return None`（无返回值）
    - _Requirements: 1.5, 1.6, 1.10_

  - [x] 4.4 为步进运动学一致性编写属性测试（property-based testing，必做）
    - **Property 3：步进推进的运动学一致性**
    - 需要 property-based testing；用 hypothesis 生成任意在飞导弹与 dt>0，断言飞行时间恰增加 dt，位置位移在积分误差内等于 `velocity*dt`，最少 100 次迭代
    - 注释标注 `# Feature: air-combat-mvp, Property 3`
    - **Validates: Requirements 1.10**

  - [x] 4.5 为燃烧段质量单调性与推力切换编写属性测试（property-based testing，必做）
    - **Property 6：燃烧段质量单调性与推力切换**
    - 需要 property-based testing；用 hypothesis 生成飞行时间序列，断言燃烧段质量严格单调递减、燃烧结束后质量恒定，最少 100 次迭代
    - 注释标注 `# Feature: air-combat-mvp, Property 6`
    - **Validates: Requirements 1.5, 1.6**

  - [ ]* 4.6 为燃烧段切换编写单元测试
    - 测试燃烧段内/外推力施加、质量递减、`step` 在未发射时直接返回等边界条件
    - _Requirements: 1.5, 1.6, 1.10_

- [x] 5. 实现 Missile3DoF 命中、失效与 time_to_impact
  - [x] 5.1 实现 check_hit 命中判定
    - 在飞且 `|target.pos - missile.pos| <= KILL_RADIUS_M` 时置 `hit=True`、`in_flight=False` 并返回 True
    - _Requirements: 1.11_

  - [x] 5.2 实现 is_expired 失效判定
    - 满足任一即失效：`flight_time_s > MAX_FLIGHT_TIME_S`，或燃料耗尽（`mass_kg <= INITIAL_MASS - BURN_RATE*ENGINE_BURN_TIME`）且 `flight_distance_m > MAX_RANGE_M`；失效时置 `expired=True`、`in_flight=False`
    - _Requirements: 1.12, 1.15_

  - [x] 5.3 实现 time_to_impact 属性
    - 未在飞返回 0.0；在飞时 `closing_rate=-dot(rel_r,rel_v)/r`，`closing_rate<=0` 返回 999.0，否则返回 `r/closing_rate`；目标速度取自 `_last_target_state`，缺失则视相对速度为 0 → 999.0
    - _Requirements: 1.16_

  - [x] 5.4 为命中判定距离阈值充要性编写属性测试（property-based testing，必做）
    - **Property 4：命中判定的距离阈值充要性**
    - 需要 property-based testing；最少 100 次迭代
    - 注释标注 `# Feature: air-combat-mvp, Property 4`
    - **Validates: Requirements 1.11**

  - [x] 5.5 为失效判定充要性编写属性测试（property-based testing，必做）
    - **Property 5：失效判定的充要性**
    - 需要 property-based testing；最少 100 次迭代
    - 注释标注 `# Feature: air-combat-mvp, Property 5`
    - **Validates: Requirements 1.12, 1.15**

  - [x] 5.6 为 time_to_impact 分段语义编写属性测试（property-based testing，必做）
    - **Property 7：time_to_impact 分段语义**
    - 需要 property-based testing；最少 100 次迭代
    - 注释标注 `# Feature: air-combat-mvp, Property 7`
    - **Validates: Requirements 1.16**

- [x] 6. Missile3DoF 速度下限轨迹属性测试（基于仿真）
  - [x] 6.1 编写导弹速度下限的参数化轨迹属性测试（property-based testing，必做）
    - **Property 8：导弹速度下限（基于仿真轨迹的属性测试）**
    - 需要 property-based testing；参数扫描初速 ∈ [500, 700] m/s 与目标距离 ∈ [500, 5000] m，对每组合运行完整 `launch`+多步 `step` 仿真轨迹，断言射程内飞行期间每步速度量值始终高于设定下限（如 >400 m/s）
    - 注释标注 `# Feature: air-combat-mvp, Property 8`
    - **Validates: Requirements 8.3**

- [x] 7. 检查点 - 确保 Missile3DoF 全部测试通过
  - 在 Python 3.11 环境下运行 `python -m pytest tests/test_missile.py -q`，确保所有测试通过，如有疑问请询问用户。

- [x] 8. 实现 FireControlRadar 火控雷达
  - [x] 8.1 实现 FireControlRadar 常量、构造与 reset
    - 在 `sensors/radar.py` 定义常量 `MAX_RANGE_M=10000.0`、`AZIMUTH_FOV_DEG=60.0`、`ELEVATION_FOV_DEG=30.0`、`LOCK_TIME_STEPS=3`；`__init__(config)` 支持 radar_config 覆盖；初始化 `in_view=False`、`locked=False`、`_consecutive_in_view_steps=0`；实现 `reset()` 复位
    - _Requirements: 2.1_

  - [x] 8.2 实现 update 几何视场判定与锁定状态机
    - 计算 `rel_r`、`r`、方位角（`psi_los-psi_ego` 经 wrap_to_pi）、俯仰角（`arcsin(clip(rel_up/r,-1,1))`）
    - `in_view = (r<=MAX_RANGE_M) and (|az|<=AZIMUTH_FOV_DEG) and (|el|<=ELEVATION_FOV_DEG)`
    - in_view 时连续计数加一，否则清零且 `locked=False`；`locked = (连续计数 >= LOCK_TIME_STEPS)`
    - 返回 `{'locked','in_view','azimuth_deg','elevation_deg','range_m'}`；仅几何判定，不算雷达方程/多普勒/搜索模式
    - _Requirements: 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8_

  - [x] 8.3 为雷达视场判定充要性编写属性测试（property-based testing，必做）
    - **Property 10：雷达视场判定的充要性**
    - 需要 property-based testing；最少 100 次迭代
    - 注释标注 `# Feature: air-combat-mvp, Property 10`
    - **Validates: Requirements 2.2, 2.3**

  - [x] 8.4 为雷达锁定状态机充要性编写属性测试（property-based testing，必做）
    - **Property 11：雷达锁定状态机的充要性**
    - 需要 property-based testing；用 hypothesis 生成任意 in_view 布尔序列驱动状态机，断言锁定充要性与脱离即清零，最少 100 次迭代
    - 注释标注 `# Feature: air-combat-mvp, Property 11`
    - **Validates: Requirements 2.4, 2.5, 2.6**

  - [ ]* 8.5 为雷达视场/锁定编写单元测试
    - 测试视场边界（距离 10000、方位 ±60°、俯仰 ±30°）、连续锁定步数阈值、脱离视场清零等具体示例
    - _Requirements: 2.1, 2.4, 2.5, 2.6_

- [x] 9. 实现 AirCombatSparseReward 稀疏奖励
  - [x] 9.1 实现事件奖励表、构造与逐步奖励 compute_step
    - 在 `envs/sparse_reward_air_combat.py` 定义 `EVENT_REWARDS`（radar_lock=0.1、radar_lock_lost=-0.1、missile_launch=1.0、missile_hit=10.0、missile_miss=-1.0、being_locked=-0.1、being_hit=-10.0、out_of_envelope=-0.5）与 `AirCombatEvent` 枚举
    - 实现 `__init__(config)`（合并 event_rewards、读取 relabelling/window/gaussian_sigma）、`reset()`、`compute_step(events)` 返回事件奖励之和
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.11_

  - [x] 9.2 为事件奖励求和编写属性测试（property-based testing，必做）
    - **Property 14：事件奖励求和**
    - 需要 property-based testing；用 hypothesis 从事件枚举随机抽取子集，断言 `compute_step` 等于各事件奖励值之和，最少 100 次迭代
    - 注释标注 `# Feature: air-combat-mvp, Property 14`
    - **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8**

  - [x] 9.3 实现 relabel_trajectory 轨迹级重标（线性核 + 高斯核）
    - 结局奖励：线性核在整条轨迹上均匀分配，总量守恒等于 `terminal_reward`
    - 事件奖励：对发生于步 `t_e` 的事件，在窗口 `[max(0,t_e-window), t_e]` 内按高斯权重归一化后乘以事件奖励值反向衰减分配（`t>t_e` 步零信用），总量守恒
    - 两部分叠加返回长度 `episode_length` 的奖励数组
    - _Requirements: 4.9, 4.10_

  - [x] 9.4 为结局奖励线性核守恒编写属性测试（property-based testing，必做）
    - **Property 15：结局奖励线性核守恒**
    - 需要 property-based testing；最少 100 次迭代
    - 注释标注 `# Feature: air-combat-mvp, Property 15`
    - **Validates: Requirements 4.9**

  - [x] 9.5 为事件奖励高斯核守恒与因果性编写属性测试（property-based testing，必做）
    - **Property 16：事件奖励高斯核守恒与因果性**
    - 需要 property-based testing；断言高斯核分配总量等于事件奖励值，且分配仅落在 `t<=t_e`，最少 100 次迭代
    - 注释标注 `# Feature: air-combat-mvp, Property 16`
    - **Validates: Requirements 4.10**

- [x] 10. 检查点 - 确保雷达与稀疏奖励测试通过
  - 在 Python 3.11 环境下运行相关 `pytest`，确保所有测试通过，如有疑问请询问用户。

- [x] 11. 核对父类 CloseRangeTrackingEnv.step 结构（关键决策点）
  - 阅读 `src/uav_vpp_guidance/envs/tracking_env.py` 中 `CloseRangeTrackingEnv` 的 `step`、`reset`、`_check_done`/`_compute_reward`、`_get_observation`、`_get_current_states` 实现
  - 确认：`step` 内部是否调用（多态分派）子类重写的 `_check_done`/`_compute_reward`、调用时机、终止后是否执行清理（如重置 `current_step`、状态机复位）
  - 据此在两方案间决策：A) 直接调用 `super().step()` 后编排；B) 调用父类动力学推进子方法后自行编排八步（见设计 §3.3.4）
  - 在 `air_combat_mvp_env.py` 顶部以注释记录核对结论与所选方案及理由
  - _Requirements: 3.1, 3.5_

- [x] 12. 实现 AirCombatMVPEnv 构造、实体管理与观测扩展
  - [x] 12.1 实现 __init__ 与实体管理
    - `AirCombatMVPEnv(CloseRangeTrackingEnv)`：`super().__init__(config)` 后创建 `missile_ego=Missile3DoF(...)`、`radar_ego=FireControlRadar(...)`；仅当 `missile_config.target.enabled` 才创建 `missile_target`/`radar_target`，否则为 `None`
    - 读取 `air_combat.reward.use_sparse`（默认 True）并按需创建 `AirCombatSparseReward`；初始化 `_prev_event_state={}`、`_prev_in_envelope=False`
    - 优先使用 `Missile3DoF`，不实例化 `EngagementTracker`
    - _Requirements: 3.1, 3.2, 3.3, 3.12_

  - [x] 12.2 实现 reset 复位扩展
    - 调用父类 reset 后复位导弹/雷达/事件状态与发射包线状态，并按场景配置随机化目标初始位置
    - _Requirements: 3.1_

  - [x] 12.3 实现 _get_observation 尾部拼接 6 维空战特征
    - 调用 `super()._get_observation()` 后在 `observation_vector` 尾部 `np.concatenate` 拼接 6 维特征（`radar_locked`、`radar_in_view`、`missile_in_flight`、`missile_time_to_impact`(tti/20 裁剪[0,1])、`incoming`(恒0)、`incoming_distance`(恒1)）；附带 `radar_state`/`missile_state` 字典；不修改父类/observation.py
    - _Requirements: 3.4, 7.1_

  - [ ]* 12.4 为观测扩展编写单元测试
    - 断言 reset/step 观测维度一致、尾部 6 维取值与雷达/导弹状态对应
    - _Requirements: 3.4_

- [x] 13. 实现 AirCombatMVPEnv step 八步编排与终止/事件
  - [x] 13.1 实现 step 八步编排
    - 按设计 §3.3.4 实现：①本机机动 ②更新雷达 ③检查发射（`radar.locked and not in_flight and can_launch`）④导弹 PNG 制导 step ⑤命中/失效 ⑥目标机动 ⑦检查终止 ⑧计算奖励；按第 11 任务的决策采用方案 A 或 B
    - 返回 `obs, reward, terminated, truncated, info`，info 含 events/termination_info/missile_in_flight/time_to_impact
    - _Requirements: 3.5, 3.6, 3.7_

  - [x] 13.2 实现 _check_done 终止条件
    - 命中成功 → terminated（is_success）；来袭导弹命中本机 → terminated（being_hit，第一阶段 missile_target=None 不触发）；本机绝对位置越界 → terminated（out_of_bounds）；步数达 max_steps → truncated（timeout）
    - 实现 `_is_out_of_bounds(own)`（高度上下限、到场景中心水平距离上限）
    - _Requirements: 3.8, 3.9, 3.10, 3.11_

  - [x] 13.3 实现发射包线判定与事件检测
    - 实现发射包线判定（目标相对本机距离 ∈ [500,5000] 且 |ATA|<30°），包线内→外跳变触发 `out_of_envelope`（不终止回合），与场景边界越界区分
    - 实现 `_detect_events`：基于 `_prev_event_state` 跳变检测 radar_lock/radar_lock_lost/missile_launch/missile_hit/missile_miss/out_of_envelope（being_locked/being_hit 第一阶段恒不触发）
    - 实现 `_compute_sparse_reward` 调用 `AirCombatSparseReward.compute_step`
    - _Requirements: 3.13, 4.8_

  - [x] 13.4 为环境发射触发组合充要性编写属性测试（property-based testing，必做）
    - **Property 12：环境发射触发的组合充要性**
    - 需要 property-based testing；断言创建新在飞导弹当且仅当 radar.locked、can_launch 允许、无在飞导弹三者同时成立，最少 100 次迭代
    - 注释标注 `# Feature: air-combat-mvp, Property 12`
    - **Validates: Requirements 3.7**

  - [x] 13.5 为脱离发射包线事件检测编写属性测试（property-based testing，必做）
    - **Property 13：脱离发射包线事件检测**
    - 需要 property-based testing；用几何序列断言 out_of_envelope 当且仅当从包线内跳变到包线外，最少 100 次迭代
    - 注释标注 `# Feature: air-combat-mvp, Property 13`
    - **Validates: Requirements 3.13, 4.8**

  - [ ]* 13.6 为环境步进顺序、终止与事件编写集成测试
    - 在 `tests/test_mvp_air_combat.py` 中使用 `backend: simple`：锁定后满足包线才发射、发射后导弹在飞、命中后成功结束；超时/越界终止；radar_lock/missile_launch/missile_hit/out_of_envelope 事件与奖励值
    - _Requirements: 3.5, 3.8, 3.10, 3.11, 9.2, 9.3_

- [x] 14. 检查点 - 确保环境集成测试通过
  - 在 Python 3.11 环境下运行 `python -m pytest tests/test_mvp_air_combat.py -q`，确保所有测试通过，如有疑问请询问用户。

- [x] 15. reward.py 工厂函数接入（非破坏）
  - 在 `src/uav_vpp_guidance/envs/reward.py` 末尾新增 `create_reward_calculator(config)`：`air_combat.reward.use_sparse=True` 时返回 `AirCombatSparseReward`，否则返回现有 `RewardCalculator`；不修改 `RewardCalculator` 现有逻辑
  - 在 `AirCombatMVPEnv` 中通过该工厂或直接持有稀疏奖励实例接入
  - _Requirements: 4.11, 7.2, 7.7_

- [x] 16. 编写 pilot 场景配置文件
  - 创建 `config/experiment/air_combat_mvp_pilot.yaml`：`max_high_level_steps=400`（80s）、`high_level_dt=0.2`、`backend: simple`/`use_jsbsim: false`
  - 本机初始 `position_m=[0,5000,0]`、`velocity_mps=250`、`heading_deg=0`；目标随机化 `range_m=[3000,5000]`、`azimuth_deg=[-30,30]`、`elevation_deg=[-10,10]`、`velocity_mps=250`、`heading_deg=180`
  - `missile_config.ego.enabled=true`、`missile_config.target.enabled=false`；`radar_config`（max_range_m=10000、az=60、el=30、lock_time_steps=3）
  - `reward.use_sparse=true`、`relabelling.enabled=true`、`window=50`；bounds（min/max 高度、max_range）
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7_

- [x] 17. 编写规则策略验证脚本
  - 创建 `scripts/validate_mvp_air_combat.py`：本机采用主动跟踪规则策略（PN/LOS 率转向保持目标在视场内），导弹由环境自动创建并 PNG 制导；目标直线/简单机动
  - 运行 20 回合，汇总并输出发射率、命中率、平均击杀时间
  - 通过标准：发射率 >90% 且命中率 >80% 且平均击杀时间 <30s → 报告通过；否则报告未通过并打印各项指标数值
  - **未通过时的诊断与调参备选路径**（作为脚本输出/排查备注，未达标时执行）：
    - 优先检查：导弹速度曲线（是否因阻力仍过大而提前失速）、雷达锁定连续性（Ego 机动是否导致目标频繁脱离视场而反复清零锁定计数）
    - 可调参数：`KILL_RADIUS_M`（增大到 50m）、`NAV_CONSTANT`（增大到 4–5）、`LAUNCH_ATA_MAX_DEG`（放宽到 45°）
  - 在 Python 3.11 环境下运行脚本
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

- [x] 18. 现有测试回归检查（确保未破坏现有模块行为）
  - 在 Python 3.11 环境下运行现有相关回归测试，确保 `AirCombatMVPEnv` 继承并覆盖父类方法（如 `_get_observation`、`step`、`_check_done`、`_compute_reward`）后未破坏现有模块行为（子类覆盖可能影响父类行为一致性）
  - 命令示例：`python -m pytest tests/test_cbf_filter.py tests/test_reward.py tests/test_tracking_env_no_prediction.py -q`（运行 CBF、reward、tracking_env 相关现有回归测试；若仓库文件名不完全一致，以实际存在的对应测试为准）
  - 确保上述现有测试全部通过，如有失败请定位是否因子类覆盖引入的行为变化并修复
  - _Requirements: 7.5, 7.6, 7.7_

- [x] 19. 端到端验证与最终检查点
  - 在 Python 3.11 环境下运行 `scripts/validate_mvp_air_combat.py`（`backend: simple`）核对通过标准（发射率>90%/命中率>80%/平均击杀<30s）
  - 在 Python 3.11 环境下运行 `python -m pytest tests/test_missile.py tests/test_mvp_air_combat.py -q` 全量测试
  - 若通过标准未达到，按任务 17 的诊断与调参备选路径（检查导弹速度曲线/雷达锁定连续性；调整 `KILL_RADIUS_M`/`NAV_CONSTANT`/`LAUNCH_ATA_MAX_DEG`）排查
  - 确保所有测试通过，如有疑问请询问用户
  - _Requirements: 6.5, 6.6, 9.1, 9.2, 9.3_

- [ ]* 20. 可选增强：VPP 纳入战斗状态（第二阶段预留）
  - 在 `air_combat.vpp.use_combat_state=true` 时，于 `AirCombatMVPEnv` 调用 `action_to_virtual_point` 前根据雷达/导弹状态调整锚点/偏移；不改动 `virtual_point/generator.py` 源码
  - _Requirements: 7.3_

- [ ]* 21. 可选增强：CBF 第二阶段来袭导弹规避
  - 当存在来袭导弹（`missile_target.in_flight`）时将其作为 CBF 障碍传入约束本机机动；本机自身导弹永不作为障碍；不修改 `safety/cbf_filter.py` 源码
  - _Requirements: 7.4_

## 备注（Notes）

- 标记 `*` 的子任务为可选（仅限单元测试 / 集成测试 / 第二阶段增强），可为加速 MVP 跳过。
- **属性测试任务（Property 1–16）均为必做（非 `*` 可选）**，需使用 property-based testing（`hypothesis`），每条属性最少 100 次迭代。
- 每条正确性属性恰由一个属性测试实现，标注属性编号与所验证需求条款，便于追溯。
- **全部测试与实验在 Python 3.11 环境执行**（导入冒烟检查、检查点、回归检查、验证脚本、端到端验证均在 py3.11 解释器/虚拟环境下运行 `pytest` 与脚本）。
- 检查点（任务 7、10、14）与现有测试回归检查（任务 18）、端到端最终检查点（任务 19）用于增量验证，确保每个阶段稳定后再推进。
- 任务 18 现有测试回归检查运行 CBF、reward、tracking_env 相关现有测试，防止子类覆盖父类方法破坏既有行为。
- 不修改 `guidance/los_rate_guidance.py`、`training/train_ppo.py`、`envs/missile_model.py`（保持 `MissileModel`/`EngagementTracker` 行为不变）。
- 所有新建模块统一 JSBSim 坐标系（X=North, Y=Up, Z=East）与米/秒/弧度；环境级测试统一 `backend: simple`，不依赖 JSBSim。
