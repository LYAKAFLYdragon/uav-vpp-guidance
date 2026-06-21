# 需求文档

## 引言（Introduction）

本特性（air-combat-mvp）将 `uav-vpp-guidance` 工程从"追踪问题"升级为"空战决策问题"的最小可行版本（MVP）。系统在现有近距追踪环境（`CloseRangeTrackingEnv`）的基础上，引入简化 3-DoF 导弹模型、火控雷达几何模型、空战事件稀疏奖励与单向击杀场景配置，并提供规则策略验证实验。第一阶段仅为本机（Ego）装备导弹与雷达，目标机（Target）不还击，用于打通"探测—锁定—发射—制导—命中/失效"的完整闭环。

设计参考本地导弹规避项目（`Avoiding_medium_long_range_air-to-air_missiles-master`）的导弹动力学与场景设计，但将其简化为适合 MVP 的形式（仅保留 PNG 比例导引，移除 SPG/BPG）。

### 参考项目融合策略（Assumptions & Constraints）

本特性对参考项目 `Avoiding_medium_long_range_air-to-air_missiles-master` 采取"按模块选择性复用"的策略，作为设计与实现的约束与假设：

- **MissileSimulation：部分复用。** 仅复用其物理参数来源（推力、初始质量、燃烧速度、阻力系数、升力系数、参考面积），坐标系变换与制导接口须按本特性的 JSBSim 坐标系重新推导，不直接复制其速度分解代码。
- **AircraftSimulation：不复用。** 本特性使用 JSBSim F-16 高保真飞行动力学，比参考项目的简化飞机模型更精确，故飞机动力学不沿用参考项目实现。
- **AvoidMissileTask：参考结构。** 仅参考其任务结构（如 `_is_terminal`、`task_step` 的组织方式），并适配到 JSBSim 环境，不直接复用代码。
- **avoid_missile_configure.json：参考格式。** 仅参考其配置组织格式，转换为 YAML 并将单位统一为米/秒/弧度（JSBSim 坐标系）。
- **assessors.py：不复用。** 本特性采用与 R2SP 对齐的事件稀疏奖励，而非参考项目的密集态势奖励，故不复用其奖励/评估逻辑。

### 与现有代码的关系（重要约束）

工程中已存在两个相关模块，本特性 **不得删除或破坏** 现有模块的行为：

- `src/uav_vpp_guidance/envs/missile_model.py`：已存在 `MissileModel`（速度恒定的点质量 PN 导弹）与 `EngagementTracker`（多导弹管理与随机发射）。该模块用于现有对抗环境的胜负判定，**保持不变**。
- `src/uav_vpp_guidance/envs/tracking_env.py`：已存在 `CloseRangeTrackingEnv`，本特性的新环境将 **继承** 它并重写核心步进逻辑。
- `src/uav_vpp_guidance/envs/reward.py`：已存在 `RewardCalculator`，本特性新增独立的稀疏奖励类，**不修改** 现有密集奖励逻辑。

本特性新建的 `src/uav_vpp_guidance/weapons/missile.py` 中的 `Missile3DoF` 是一个 **独立的、含阻力/升力/推力燃烧段的简化 3-DoF 模型**，定位区别于现有 `missile_model.py` 中"恒速点质量"的 `MissileModel`。需求文档要求 `Missile3DoF` 自带明确接口（`can_launch` / `step` / `check_hit` / `is_expired`），新环境优先使用 `Missile3DoF`，不复用 `EngagementTracker`。设计阶段需说明二者并存的边界，避免命名与职责冲突。

## 术语表（Glossary）

- **Ego**：本机（受控飞机），第一阶段唯一携带导弹与雷达的实体，UID 约定为 `own`。
- **Target**：目标机（敌机），第一阶段仅做直线或简单机动，不发射导弹，UID 约定为 `target`。
- **Missile3DoF**：本特性新建于 `weapons/missile.py` 的简化 3-DoF 导弹模型，含固定阻力系数、升力系数、参考面积、过载上限与推力燃烧段。
- **FireControlRadar**：本特性新建于 `sensors/radar.py` 的火控雷达几何模型，仅基于几何（距离/方位角/俯仰角/连续锁定步数）判定锁定与视场内状态。
- **AirCombatMVPEnv**：本特性新建于 `envs/air_combat_mvp_env.py` 的空战环境，继承 `CloseRangeTrackingEnv` 并重写步进与终止逻辑。
- **AirCombatSparseReward**：本特性新建于 `envs/sparse_reward_air_combat.py` 的 R2SP 风格事件稀疏奖励类。
- **PNG（比例导引）**：Proportional Navigation Guidance，标准真比例导引制导律（true PN 向量形式）`a = K * cross(omega, V_missile)`，其中 `omega = cross(rel_r, rel_v)/|rel_r|^2` 为视线角速度、`V_missile` 为导弹速度向量，过载受 `MAX_G*9.8` 限制。
- **ATA（Antenna Train Angle）**：本机速度方向与本机—目标视线之间的夹角。
- **JSBSim 坐标系**：本特性约定的世界坐标系，X=North（北）、Y=Up（上）、Z=East（东），单位米/秒/弧度。
- **lock_time_steps**：雷达连续处于"目标在视场内"状态达到该步数后才置为"锁定"。
- **time_to_impact（剩余飞行时间）**：在飞导弹基于当前相对距离与接近速率（closing rate）估算的命中剩余时间，定义为 `|rel_r| / closing_rate`，其中 `closing_rate = -dot(rel_r, rel_v)/|rel_r|`；当 `closing_rate<=0`（无法命中）时取大值 999.0，导弹未在飞时取 0.0。
- **incoming（来袭导弹）**：第一阶段恒为不存在（目标不发射），观测中预留字段。
- **R2SP 重标（relabelling）**：对回合轨迹做奖励重新分配——结局奖励用线性核在轨迹上均匀分配，事件奖励用高斯核围绕事件步反向衰减分配。
- **kill_radius（杀伤半径）**：导弹与目标距离小于该值即判定命中。
- **发射包线（launch envelope）**：目标相对本机的几何区域，定义为目标相对本机距离处于闭区间 [500, 5000] 米且 |ATA| 小于 30 度。
- **out_of_envelope（脱离发射包线）**：目标相对本机位置离开"发射包线"（与本机绝对位置越过场景边界是两个不同概念）。
- **场景边界越界（out_of_bounds）**：本机绝对位置越过场景设定的位置/高度/距离边界（与脱离发射包线是两个不同概念）。
- **pilot 配置**：缩短回合步数、用于快速冒烟验证的实验配置。

## 需求（Requirements）

### 需求 1：简化 3-DoF 导弹模型（Missile3DoF）

**用户故事：** 作为空战仿真与系统集成工程师，我想要一个独立的简化 3-DoF 导弹模型，以便在 MVP 中用真比例导引模拟导弹飞行、发射判定与命中判定。

#### 验收标准（Acceptance Criteria）

1. THE Missile3DoF SHALL 在 JSBSim 坐标系（X=North, Y=Up, Z=East）下以米、秒、弧度为单位表示全部状态量。
2. THE Missile3DoF SHALL 使用固定气动与性能常量：阻力系数 DRAG_COEFF=0.6、升力系数 LIFT_COEFF=0.15、参考面积 REF_AREA=0.4 平方米、过载上限 MAX_G=30.0、发动机燃烧时间 ENGINE_BURN_TIME=5.0 秒。
3. THE Missile3DoF SHALL 仅实现 PNG 比例导引，制导加速度按标准真比例导引（true PN）向量形式 `a = K * cross(omega, V_missile)` 计算，其中 `omega = cross(rel_r, rel_v)/|rel_r|^2` 为视线角速度、`V_missile` 为导弹速度向量。（注：此式取代早先的 `a = K*(rel_r·rel_v)/|rel_r|^4 * cross(rel_r, omega)` 形式——后者在量纲上于交战距离（公里级）被严重抑制、制导加速度趋近于零，导致导弹近乎弹道飞行、命中率为 0；标准真 PN 以闭合速度 × 视线角速度构成有效制导。）
4. THE Missile3DoF SHALL 将制导指令加速度限制在 MAX_G*9.8 米每二次方秒以内。
5. WHILE 在飞导弹的飞行时间小于等于 ENGINE_BURN_TIME，THE Missile3DoF SHALL 施加固定推力。
6. WHEN 在飞导弹的飞行时间超过 ENGINE_BURN_TIME，THE Missile3DoF SHALL 停止施加推力并仅在阻力、升力与制导加速度作用下飞行。
7. WHEN 调用 can_launch(ego_state, target_state) 且本机—目标距离处于闭区间 [500, 5000] 米、|ATA| 小于 30 度，THE Missile3DoF SHALL 返回允许发射。
8. IF 本机—目标距离超出 [500, 5000] 米或 |ATA| 大于等于 30 度，THEN THE Missile3DoF SHALL 在 can_launch 中返回不允许发射。
9. THE Missile3DoF SHALL 在 can_launch 中仅检查几何条件（距离与 |ATA|），不检查雷达视场或雷达锁定，雷达锁定由环境层在调用 can_launch 之前独立检查并组合判定。
10. WHEN 调用 step(dt, target_state)，THE Missile3DoF SHALL 以 PNG 更新在飞导弹的速度与位置并增加飞行时间。
11. WHEN 调用 check_hit(target_state) 且导弹与目标距离小于等于杀伤半径，THE Missile3DoF SHALL 返回命中为真。
12. WHEN 调用 is_expired() 且飞行时间超过最大飞行时间或飞行距离超过最大射程，THE Missile3DoF SHALL 返回失效为真。
13. THE Missile3DoF SHALL 提供 can_launch、step、check_hit、is_expired 四个公开方法。
14. THE Missile3DoF SHALL 使用恒定推力 THRUST=20000 牛、初始质量 400 千克、燃烧速度 25 千克每秒，使燃烧结束时速度不低于 700 米每秒（验证：20000牛/400千克=50米每二次方秒，燃烧 5 秒加速约 +250 米每秒，初速 500 米每秒在燃烧结束时约 750 米每秒；之后受阻力减速，在 5000 米射程飞行 10 至 15 秒后速度仍高于 500 米每秒）。
15. THE Missile3DoF SHALL 设定最大飞行时间 max_flight_time_s=20.0 秒，作为 is_expired 的失效条件之一（与燃料耗尽且飞行距离超过最大射程并列）。
16. THE Missile3DoF SHALL 提供 time_to_impact 属性：在飞时返回 `|rel_r| / closing_rate`（其中 `closing_rate = -dot(rel_r, rel_v)/|rel_r|`）；当 closing_rate 小于等于 0 时返回大值 999.0；未在飞时返回 0.0。

### 需求 2：火控雷达几何模型（FireControlRadar）

**用户故事：** 作为空战仿真与系统集成工程师，我想要一个仅基于几何约束的火控雷达模型，以便判定目标是否在视场内并产生稳定锁定信号，为导弹发射提供前置条件。

#### 验收标准（Acceptance Criteria）

1. THE FireControlRadar SHALL 使用最大探测距离 max_range_m=10000 米、方位视场 azimuth_fov_deg=正负 60 度、俯仰视场 elevation_fov_deg=正负 30 度。
2. WHEN 调用 update(ego_state, target_state) 且目标处于最大探测距离内、方位角在正负 60 度内、俯仰角在正负 30 度内，THE FireControlRadar SHALL 将视场内（in_view）置为真。
3. IF 目标距离超过 max_range_m、或方位角超出正负 60 度、或俯仰角超出正负 30 度，THEN THE FireControlRadar SHALL 将视场内（in_view）置为假。
4. WHILE 目标连续处于视场内的步数小于 lock_time_steps（默认 3），THE FireControlRadar SHALL 将锁定（locked）置为假。
5. WHEN 目标连续处于视场内的步数达到 lock_time_steps，THE FireControlRadar SHALL 将锁定（locked）置为真。
6. WHEN 目标在某一步脱离视场，THE FireControlRadar SHALL 将连续视场内步数重置为零并将锁定置为假。
7. WHEN 调用 update(ego_state, target_state)，THE FireControlRadar SHALL 返回包含锁定（locked）与视场内（in_view）的状态。
8. THE FireControlRadar SHALL 仅依据几何约束判定锁定，不计算雷达方程、多普勒效应或搜索模式。

### 需求 3：空战 MVP 环境集成（AirCombatMVPEnv）

**用户故事：** 作为空战仿真与系统集成工程师，我想要一个继承自现有追踪环境的空战环境，以便在保留飞行动力学与制导链路的同时引入导弹与雷达，形成完整空战决策闭环。

#### 验收标准（Acceptance Criteria）

1. THE AirCombatMVPEnv SHALL 继承 CloseRangeTrackingEnv 并重写步进与终止核心逻辑。
2. THE AirCombatMVPEnv SHALL 管理本机导弹（missile_ego）与本机雷达（radar_ego）两个实体。
3. WHERE 目标导弹（missile_target）被配置启用，THE AirCombatMVPEnv SHALL 管理目标导弹实体；否则不创建目标导弹。
4. THE AirCombatMVPEnv SHALL 在观测中包含雷达状态（locked、in_view）、本机导弹状态（in_flight、time_to_impact）与来袭导弹状态（incoming、distance），其中 time_to_impact 取自 Missile3DoF 的 time_to_impact 属性。
5. WHEN 执行一个步进，THE AirCombatMVPEnv SHALL 按以下顺序处理：本机机动、更新雷达、检查发射、更新本机导弹 PNG 制导、检查命中与失效、目标机动、检查终止、计算奖励。
6. WHILE 第一阶段（仅本机装备导弹），THE AirCombatMVPEnv SHALL 由规则自动触发导弹发射而不依赖策略动作维度。
7. WHEN 雷达锁定（radar.locked 为真）且 Missile3DoF.can_launch(ego_state, target_state) 返回允许发射且本机当前没有在飞导弹，THE AirCombatMVPEnv SHALL 创建一枚 Missile3DoF 在飞导弹（即环境层组合条件为 `if radar.locked and missile.can_launch(ego, target)`）。
8. WHEN 本机导弹命中目标，THE AirCombatMVPEnv SHALL 以成功（命中目标）结束回合。
9. IF 本机被来袭导弹命中，THEN THE AirCombatMVPEnv SHALL 以失败（被击中）结束回合。
10. WHEN 步进计数达到 max_steps，THE AirCombatMVPEnv SHALL 以超时结束回合。
11. IF 本机绝对位置越过场景边界，THEN THE AirCombatMVPEnv SHALL 以越界坠毁（out_of_bounds）结束回合。
12. THE AirCombatMVPEnv SHALL 优先使用 Missile3DoF 作为本机导弹实现，而不使用现有 missile_model.py 中的 EngagementTracker。
13. THE AirCombatMVPEnv SHALL 将"发射包线"定义为目标相对本机距离处于闭区间 [500, 5000] 米且 |ATA| 小于 30 度，并 WHEN 目标离开该发射包线时触发 out_of_envelope 事件；该事件与场景边界越界（见验收标准 11，针对本机绝对位置）是两个不同概念，前者针对目标相对本机位置超出武器发射区，后者针对本机绝对位置超出场景边界。

### 需求 4：空战事件稀疏奖励（AirCombatSparseReward）

**用户故事：** 作为空战仿真与系统集成工程师，我想要一个 R2SP 风格的事件稀疏奖励，以便用稀疏的空战事件信号训练策略，并通过轨迹级重标缓解奖励稀疏问题。

#### 验收标准（Acceptance Criteria）

1. WHEN 雷达完成锁定（radar_lock），THE AirCombatSparseReward SHALL 给出事件奖励 +0.1。
2. WHEN 雷达丢失锁定（radar_lock_lost），THE AirCombatSparseReward SHALL 给出事件奖励 -0.1。
3. WHEN 本机发射导弹（missile_launch），THE AirCombatSparseReward SHALL 给出事件奖励 +1.0。
4. WHEN 本机导弹命中目标（missile_hit），THE AirCombatSparseReward SHALL 给出事件奖励 +10.0。
5. WHEN 本机导弹未命中（missile_miss），THE AirCombatSparseReward SHALL 给出事件奖励 -1.0。
6. WHEN 本机被目标雷达锁定（being_locked），THE AirCombatSparseReward SHALL 给出事件奖励 -0.1。
7. WHEN 本机被击中（being_hit），THE AirCombatSparseReward SHALL 给出事件奖励 -10.0。
8. WHEN 触发脱离发射包线事件（out_of_envelope，即目标离开发射包线，定义见需求 3 验收标准 13），THE AirCombatSparseReward SHALL 给出事件奖励 -0.5。
9. WHERE 启用轨迹级重标，THE AirCombatSparseReward SHALL 用线性核将结局奖励在整条轨迹上均匀分配。
10. WHERE 启用轨迹级重标，THE AirCombatSparseReward SHALL 用高斯核将事件奖励围绕事件发生步反向衰减分配。
11. THE AirCombatSparseReward SHALL 作为独立类提供，且不修改现有 RewardCalculator 的密集奖励逻辑。

### 需求 5：空战场景配置（air_combat_mvp_pilot.yaml）

**用户故事：** 作为空战仿真与系统集成工程师，我想要一个 pilot 场景配置文件，以便快速复现 MVP 第一阶段的单向击杀场景并驱动验证实验。

#### 验收标准（Acceptance Criteria）

1. THE 配置文件 SHALL 设置 max_steps=400（对应 80 秒）。
2. THE 配置文件 SHALL 设置本机初始位置为 [0, 5000, 0] 米、初始速度 250 米每秒、初始航向 0 度。
3. WHEN 重置回合，THE 配置文件 SHALL 在距离 3000 至 5000 米、方位角正负 30 度、俯仰角正负 10 度范围内随机化目标初始位置。
4. THE 配置文件 SHALL 设置目标初始速度 250 米每秒、初始航向 180 度（迎面）。
5. THE 配置文件 SHALL 设置本机导弹（missile_config ego）为启用、目标导弹为停用（第一阶段单向击杀）。
6. THE 配置文件 SHALL 包含雷达配置（radar_config）。
7. THE 配置文件 SHALL 设置奖励为启用稀疏奖励（use_sparse）并启用高斯重标、窗口为 50。

### 需求 6：规则策略验证实验（validate_mvp_air_combat.py）

**用户故事：** 作为空战仿真与系统集成工程师，我想要一个规则策略验证脚本，以便在无需训练的前提下验证空战闭环的正确性并量化通过标准。

#### 验收标准（Acceptance Criteria）

1. THE 验证脚本 SHALL 使用规则策略驱动本机：本机采用固定机动库（如匀速直线或简单转向）进行机动，导弹由环境在满足发射条件后自动创建并按 PNG 制导律飞向目标（导弹制导不由本机机动策略控制）。
2. THE 验证脚本 SHALL 让目标以直线或简单机动飞行。
3. THE 验证脚本 SHALL 运行 20 个回合（episodes）并汇总统计指标。
4. THE 验证脚本 SHALL 输出发射率、命中率与平均击杀时间。
5. WHEN 20 个回合的发射率大于 90 百分比、命中率大于 80 百分比、平均击杀时间小于 30 秒，THE 验证脚本 SHALL 报告验证通过。
6. IF 任一通过标准未达到，THEN THE 验证脚本 SHALL 报告验证未通过并给出各项指标数值。

### 需求 7：现有模块集成点（Integration Points）

**用户故事：** 作为空战仿真与系统集成工程师，我想要将导弹与雷达状态接入现有观测、奖励与安全模块，以便空战能力与既有系统协同工作而不破坏现有功能。

#### 验收标准（Acceptance Criteria）

1. THE 集成改动 SHALL 在 tracking_env.py 的观测中加入导弹与雷达状态字段。
2. THE 集成改动 SHALL 在 reward.py 中新增对 AirCombatSparseReward 的接入入口。
3. WHERE 虚拟点生成器（virtual_point/generator.py）被配置为考虑雷达与导弹状态，THE 虚拟点生成器 SHALL 在生成虚拟追踪点时纳入雷达与导弹状态。
4. WHILE 存在目标发射的来袭导弹（incoming missile），THE CBF 安全滤波器（safety/cbf_filter.py）SHALL 约束本机机动以避免与来袭导弹发生碰撞；本机自身发射的导弹 SHALL NOT 被 CBF 视为障碍物（理由：第一阶段目标不发射导弹，incoming 恒为假，CBF 不受影响；第二阶段 CBF 仅规避目标发射的来袭导弹）。
5. THE 集成改动 SHALL 保持 guidance/los_rate_guidance.py 不被修改。
6. THE 集成改动 SHALL 保持 training/train_ppo.py 不被修改。
7. THE 集成改动 SHALL 保持现有 missile_model.py 的 MissileModel 与 EngagementTracker 行为不变。

### 需求 8：单位、坐标系与动力学一致性约束

**用户故事：** 作为空战仿真与系统集成工程师，我想要统一的单位与坐标系约定以及合理的导弹—飞机性能比，以便仿真物理可信、模块之间数据一致。

#### 验收标准（Acceptance Criteria）

1. WHEN 从英尺转换为米，THE 系统 SHALL 使用换算系数 1 英尺等于 0.3048 米。
2. THE 系统 SHALL 在全部新建模块中统一使用 JSBSim 坐标系（X=North, Y=Up, Z=East）与米/秒/弧度单位。
3. THE Missile3DoF SHALL 以不低于 500 米每秒的速度飞行，且显著高于飞机的 300 米每秒量级速度。
4. THE Missile3DoF SHALL 以 MAX_G=30.0 的过载上限提供显著优于飞机的机动能力。
5. THE Missile3DoF SHALL 按 JSBSim 坐标系（X=North, Y=Up, Z=East）重新推导速度分解与坐标变换，不沿用参考项目 AircraftSimulation._get_vx_vy_vz 中 vz_ms 的负号写法（参考项目 Z 轴定义可能为 West 或采用逆时针 phi）。
6. THE 系统 SHALL 仅将参考项目代码作为物理参数（推力、质量、燃烧速度、阻力系数、升力系数、参考面积）来源，不直接复制其坐标变换实现。

### 需求 9：自动化测试

**用户故事：** 作为空战仿真与系统集成工程师，我想要针对导弹模型与空战环境的自动化测试，以便回归验证关键行为并防止集成破坏现有功能。

#### 验收标准（Acceptance Criteria）

1. THE 测试套件 SHALL 在 tests/test_missile.py 中覆盖 Missile3DoF 的发射判定、PNG 制导步进、命中判定与失效判定。
2. THE 测试套件 SHALL 在 tests/test_mvp_air_combat.py 中覆盖 AirCombatMVPEnv 的步进顺序、终止条件与稀疏奖励事件触发。
3. WHEN 运行测试套件，THE 测试套件 SHALL 在不依赖 JSBSim 高保真后端的前提下可执行（必要时使用简化后端或桩件）。
