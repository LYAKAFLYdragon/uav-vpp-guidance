# 控制方法合理性审查提示词

## 审查背景

本项目（UAV VPP Guidance）实现了一种**三层层次化空战追踪/拦截控制系统**：

1. **高层决策（RL Policy / PPO）**：输出虚拟追踪点（VPP）偏移量 `[dx, dy, dz]`
2. **中层制导律**：根据 VPP 位置计算法向过载 `nz_cmd`、滚转角速率 `roll_rate_cmd`、油门 `throttle_cmd`
3. **底层飞控（PID）**：将制导指令转化为 JSBSim 气动面舵偏量（升降舵/副翼/方向舵/油门）

制导律家族包含：
- **LOS-rate Guidance（视线角速率制导）**：基于航向误差驱动滚转、仰角误差驱动法向过载
- **Proportional Navigation（比例导引）**：经典 N×Vc×dλ/dt 加速度指令
- **Hybrid Guidance**：基于距离/能量/混合的迟滞切换
- **Mode-Switch Gate**：尾追/迎头几何下从 LOS→PN 的锁存切换

增益优化方式：
- **CEM（交叉熵方法）**：黑盒搜索制导增益
- **Bilevel Training**：外层 PPO 策略更新 + 内层 CEM 增益优化交替进行

---

## 请按照以下维度逐项进行审查

### 一、制导律物理正确性

1. **LOS-rate 制导律的 nz_cmd 推导**
   - 当前实现为：`nz = base_nz + k_los × elevation_error + k_pos × (range / distance_scale)`
   - 请审查：这是否是标准 LOS 制导的合理离散化？k_pos 项的物理意义是什么？是否应该包含 LOS 角速率而非仅仅静态角度偏差？
   - 与经典 3-loop autopilot（正常过载 → 俯仰角速率 → 舵偏角）的关系？

2. **Roll-rate 指令推导**
   - 当前：`roll_rate = k_roll × heading_error - k_damp × current_roll`
   - 请审查：heading_error 由 arctan2(sin,cos) 绕折处理是否充分？k_damp 阻尼项对应物理模型中的什么？是否存在耦合问题（滚转引起航向变化的延迟未建模）？

3. **比例导引法（PN）实现**
   - 导航常数 N=3.0，采用 TPN（真比例导引）的三维推广
   - 请审查：LOS rate 的一阶滤波 (α=0.3) 是否引入过大延迟？数值微分的精度在 dt=0.2s（5Hz 控制频率）下是否足够？有限差分引入的相位滞后如何补偿？

4. **指令饱和与安全边界**
   - nz ∈ [-2, 7]g，roll_rate ∈ [-1.5, 1.5] rad/s，throttle ∈ [0.4, 0.9]
   - 请审查：这些限幅值对 F-16 级别飞机是否合理？负过载 -2g 的限制是否过保守/激进？

### 二、层次结构合理性

5. **VPP 抽象层的必要性与有效性**
   - VPP 将 RL 策略与制导律解耦：策略输出"去哪里"，制导律解决"怎么去"
   - 请审查：VPP 偏移范围 (±1500m 纵向、±800m 横向、±500m 垂向) 在 2-3km 初始距离的空战场景中是否合理？
   - VPP 偏移量级 (~500m) 被发现是尾追失败的根因（指令偏离目标方向），这说明什么设计问题？

6. **控制频率与时间尺度分离**
   - 若 RL 策略输出频率 = 制导律更新频率 = PID 更新频率 = 5Hz (dt=0.2s)
   - 请审查：是否存在时间尺度分离不足的问题？典型分层设计中制导律应比飞控慢一个量级 (1-2Hz vs 20-50Hz)
   - 一阶指令滤波 α=0.3 的等效时间常数是多少？是否足以平滑高频噪声又不引入过大相位滞后？

7. **RL 动作空间设计**
   - 连续动作 [-1,1]^3 映射到 VPP 偏移量
   - 请审查：动作空间维度是否足够/过多？是否需要包含时间预测量 τ 和速度偏差？`action_dim=3` vs `action_dim=5` 的取舍依据？

### 三、增益优化方法合理性

8. **CEM 增益搜索**
   - 搜索空间：7 维 (k_los, k_pos, k_damp, k_roll, k_speed, k_energy, alpha_filter)
   - 12 候选、25% 精英率、收敛阈值 0.001
   - 请审查：7 维空间 12 个样本是否稀疏？elite ratio = 3 个精英是否具有统计代表性？是否需要更大种群/更多迭代？

9. **Bilevel 交替优化**
   - 外层每 10 个 episode 更新策略，内层 20 次 CEM 迭代优化增益
   - 请审查：这种交替频率是否导致策略与增益互相"追逐"而不收敛？是否有理论保证（如 Stackelberg 均衡）？regret 回退机制是否充分？

### 四、飞控层实现

10. **PID → 舵偏量映射**
    - 低层控制器接收 (nz_cmd, roll_rate_cmd, throttle_cmd)，通过 ActuatorInterface 映射到 JSBSim 属性
    - 请审查：nz_cmd → elevator_cmd_norm 的映射是否线性？是否考虑了飞行条件（动压、马赫数、高度）对操纵效率的影响？
    - 增强型 PID 的 AoA 保护、失速保护、能量补偿的优先级逻辑是否正确？

11. **一阶滤波器作为唯一平滑手段**
    - 所有指令通道仅用 `y = α*u + (1-α)*y_prev` 滤波
    - 请审查：这是否等效于一阶惯性环节 1/(τs+1)？是否需要更高阶滤波或速率限制器来模拟执行机构带宽？

### 五、奖励函数设计

12. **多目标权重配置**
    - w_range=0.5, w_angle=0.8, w_energy=0.2, w_safety=2.0, w_saturation=1.0, w_smooth=0.1
    - 请审查：权重选取依据？是否经过系统性调参（如 reward shaping sensitivity study）？安全惩罚权重 2.0 是否会导致策略过于保守？

13. **终端奖励量级**
    - success=+200, failure=-200, crash=-300
    - 与步进奖励（通常 |r| < 5）的量级差异约 40-60 倍
    - 请审查：是否导致策略过度关注终端条件而忽略过程品质？是否存在 reward hacking 风险？

### 六、观测空间完备性

14. **16 维基础观测**
    - 仅包含相对几何信息（距离、角度、速度差）
    - 请审查：是否缺少关键状态？如本机 AoA/侧滑角、过载历史、指令跟踪误差？对于空战问题是否需要目标机动性估计（如转弯率）？

15. **三角函数编码**
    - 角度特征统一用 sin/cos 对编码
    - 请审查：这种编码是否丢失了角度量级信息（如 30° vs 330° 的 cos 值相同但方位截然不同）？

### 七、模式切换与鲁棒性

16. **Mode-Switch 阈值选择**
    - 尾追门限：aspect < 25°, range < 3000m, Vc > 50 m/s
    - 请审查：25° aspect threshold 是否过窄（仅覆盖纯尾追），是否需要覆盖更宽的后半球？锁存（latch）设计是否过于激进（整个 episode 不可解除）？

17. **Hybrid 切换迟滞**
    - hysteresis_m=500m, min_dwell=3 步
    - 请审查：3 步 ×0.2s = 0.6s 的驻留时间在快速变化的空战场景中是否足够防止抖振？迟滞带 500m 是否与切换阈值 3000m 协调？

### 八、仿真保真度影响

18. **Simple 后端 vs JSBSim 后端**
    - 开发用 simple 后端（平地/质点/6DOF 简化），正式评估用 JSBSim（高保真 F-16）
    - 请审查：simple 后端忽略了哪些动态效应（气动耦合、发动机迟滞、结构限制）？在 simple 上调优的增益迁移到 JSBSim 后性能退化的预期程度？

---

## 审查输出要求

请对每个维度给出：
1. **合理性判定**：✅ 合理 / ⚠️ 有隐患 / ❌ 存在问题
2. **理论依据**：引用经典文献或标准设计准则
3. **潜在风险**：若存在问题，说明可能导致的失败模式
4. **改进建议**：若有隐患，给出具体改进方向（含可操作的参数/结构变更建议）

## 审查范围约束

- 仅审查控制方法本身的工程合理性，不涉及 RL 训练稳定性、超参数搜索策略等机器学习层面问题（除非直接影响控制回路稳定性）
- 以 F-16 级别战斗机（亚声速）在 3000-5000m 高度的近距空战场景为参考
- 默认控制频率 5Hz (dt=0.2s)

## 参考源码路径

| 模块 | 路径 |
|------|------|
| LOS-rate 制导 | `src/uav_vpp_guidance/guidance/los_rate_guidance.py` |
| 比例导引 | `src/uav_vpp_guidance/guidance/proportional_navigation.py` |
| 混合制导 | `src/uav_vpp_guidance/guidance/hybrid_guidance.py` |
| VPP 生成器 | `src/uav_vpp_guidance/virtual_point/generator.py` |
| 增益定义 | `src/uav_vpp_guidance/guidance/gain_config.py` |
| PID 控制器 | `src/uav_vpp_guidance/flight_control/pid_controllers.py` |
| 低层飞控 | `src/uav_vpp_guidance/flight_control/low_level_controller.py` |
| 增强飞控 | `src/uav_vpp_guidance/flight_control/enhanced_low_level_controller.py` |
| 奖励函数 | `src/uav_vpp_guidance/envs/reward.py` |
| 观测构造 | `src/uav_vpp_guidance/envs/observation.py` |
| CEM 优化器 | `src/uav_vpp_guidance/gain_optimizer/cem.py` |
| Bilevel 训练 | `src/uav_vpp_guidance/gain_optimizer/bilevel_trainer.py` |
| 制导配置 | `config/guidance.yaml` |
| PPO Agent | `src/uav_vpp_guidance/agents/commander_ppo_agent.py` |
