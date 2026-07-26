# 控制方法合理性审查

## 审查范围与方法
参考场景为 F-16-class 亚音速战斗机、3000–5000 m 高度、近距交战，控制频率 5 Hz（dt=0.2 s）。本审查严格只读（read-only）、证据优先：仅从实际源码/配置读取事实并执行确定性闭式数值诊断；不训练、不调参、不访问或修改 checkpoint，也不运行 simple→JSBSim 迁移实验。
F18 为仅分析（analytic-only）项目：只进行结构性分析，不做 run-based quantification。
所有建议均仅供咨询（advisory only）；本审查没有、也不会改变控制行为、配置或 checkpoint。

## 结论摘要
判定分布：by_design_ok=F04, F09；confirmed_defect=F01, F02, F06, F07, F16；confirmed_risk=F03, F05, F08, F10, F11, F12, F13, F14, F17, F18；premise_incorrect=F15

| 优先级 | 发现 |
|---:|---|
| 1 | F01 |
| 2 | F06 |
| 3 | F05 |
| 4 | F02 |
| 5 | F16 |
| 6 | F08 |
| 7 | F09 |
| 8 | F03 |
| 9 | F04 |
| 10 | F07 |
| 11 | F10 |
| 12 | F11 |
| 13 | F12 |
| 14 | F13 |
| 15 | F14 |
| 16 | F17 |
| 17 | F18 |

## 逐项发现
### F01 F01：名为 LOS 率的制导律缺少 LOS 率项
- 判定：`confirmed_defect`（优先级 1，置信度 high）
- 源码证据：src/uav_vpp_guidance/guidance/los_rate_guidance.py:337
- 诊断复现：filter_time_constant；由源码结构检查复现；详见诊断引用。
- 观察事实：_compute_nz_cmd 的 k_pos 项随 distance/distance_scale_m 增长，且该表达式中没有 LOS 角速度项。
- 失效模式：远距离时位置项增大，制导名称和物理量语义可能误导调参。
- 建议：将 LOS 率法与几何误差项分离，采用有量纲且随误差收敛的定义；需另立受控修复任务。
### F02 F02：滚转阻尼使用滚转角而非角速度
- 判定：`confirmed_defect`（优先级 4，置信度 high）
- 源码证据：src/uav_vpp_guidance/guidance/los_rate_guidance.py:190
- 诊断复现：源码结构检查；由源码结构检查复现；详见诊断引用。
- 观察事实：滚转指令使用 k_roll·heading_error − k_damp·current_roll；制导层未实现协调转弯运动学。
- 失效模式：阻尼角度而非角速度会改变阻尼物理含义，快速机动时转弯协调不足。
- 建议：在独立修复任务中定义滚转角速度反馈与协调转弯接口，再以飞行动力学测试验证。
### F03 F03：PN LOS 率滤波引入相位滞后
- 判定：`confirmed_risk`（优先级 8，置信度 high）
- 源码证据：src/uav_vpp_guidance/guidance/proportional_navigation.py:27
- 诊断复现：filter_time_constant；滤波时间常数 τ≈0.5607 s（dt=0.2 s）。
- 观察事实：PN 对 LOS 率使用一阶滤波，且法向载荷无显式 1/cos(φ) 补偿。
- 失效模式：滤波时间常数会带来相位滞后，强机动时可能降低响应裕度。
- 建议：为滤波、制导和执行器串联延迟建立预算，并在独立任务中评估补偿方案。
### F04 F04：指令饱和限制的保守性
- 判定：`by_design_ok`（优先级 9，置信度 high）
- 源码证据：config/guidance.yaml:84
- 诊断复现：roll_rate_limit_deg_s；滚转率上限 85.94°/s；参考约 270°/s。
- 观察事实：配置的 nz、滚转率和油门界限由 LOS 实例读取。
- 失效模式：相对约 270°/s 的参考，约 85.94°/s 滚转率和 -2g 下限较保守，可能限制机动包线。
- 建议：保留当前安全界限；以参考资料和 JSBSim 验证后再决定是否调整。
### F05 F05：VPP 固定米制偏置随距离改变角语义
- 判定：`confirmed_risk`（优先级 3，置信度 high）
- 源码证据：config/guidance.yaml:3
- 诊断复现：offset_angular_deviation；配置横向界限为 ±800 m；其对应 2500 m 角偏差为 17.74°，800 m 时为 45.00°。假设性 500 m 对照为 2500 m 时 11.31°（非配置值）。
- 观察事实：VPP 使用固定 ±1500/±800/±500 m 偏置范围，其中配置的横向界限为 ±800 m；500 m 数值仅用于假设性对照。
- 失效模式：相同横向米制偏置在近距离对应更大的视线角，策略语义随距离漂移。
- 建议：在独立设计任务中评估归一化角偏置或按距离调度的偏置；本建议仅供后续审查。
### F06 F06：高层与内环共享 5 Hz 时间尺度
- 判定：`confirmed_defect`（优先级 2，置信度 high）
- 源码证据：src/uav_vpp_guidance/guidance/proportional_navigation.py::__init__
- 诊断复现：filter_time_constant；滤波时间常数 τ≈0.5607 s（dt=0.2 s）。
- 观察事实：PN 实例的 dt 与控制层使用的 0.2 s 一致。
- 失效模式：高层、制导和内环没有显式时间尺度分离，滤波滞后会被放大。
- 建议：为高层、制导和内环制定明确频率与延迟预算。
### F07 F07：VPP action_dim 配置与构造器默认值不一致
- 判定：`confirmed_defect`（优先级 10，置信度 high）
- 源码证据：config/guidance.yaml:2; src/uav_vpp_guidance/virtual_point/generator.py::VirtualPointGenerator.__init__
- 诊断复现：源码结构检查；由源码结构检查复现；详见诊断引用。
- 观察事实：配置实例 action_dim 与空配置构造器默认值不同。
- 失效模式：遗漏配置时会静默获得不同动作语义。
- 建议：统一默认值或在构造器中要求显式动作维度，并增加配置验证。
### F08 F08：CEM 在实际 5 维空间中样本量偏小
- 判定：`confirmed_risk`（优先级 6，置信度 high）
- 源码证据：config/gain_space.yaml:10; src/uav_vpp_guidance/gain_optimizer/cem.py::CEMGainOptimizer.update
- 诊断复现：cem_elite_count；当前激活增益空间为 5 维；候选 12、精英 3，建议人口启发式为 10×dim=50。
- 观察事实：CEM 实例读取候选数、精英比例和逐维标准差更新；实际激活增益空间为 5 维。
- 失效模式：实际 5 维空间中 12 个候选、3 个精英的覆盖有限，可能导致不稳定或局部收敛。
- 建议：在独立优化任务中以预算为约束提高种群或采用相关协方差诊断；本建议仅供后续审查。
### F09 F09：双层训练的 regret 不是配对后悔值
- 判定：`by_design_ok`（优先级 7，置信度 high）
- 源码证据：src/uav_vpp_guidance/gain_optimizer/bilevel_trainer.py::_compute_regret; src/uav_vpp_guidance/gain_optimizer/bilevel_trainer.py::_save_policy_snapshot
- 诊断复现：源码结构检查；由源码结构检查复现；详见诊断引用。
- 观察事实：regret 返回相对历史 best-known success rate 的剩余差距，且 train 中没有回滚调用快照。
- 失效模式：该量不表示配对 regret，且较差更新缺乏策略回滚保护。
- 建议：重命名指标并定义可验证的 regret/rollback 策略；不得在本审查中改训练行为。
### F10 F10：基础执行器映射缺少动压调度
- 判定：`confirmed_risk`（优先级 11，置信度 high）
- 源码证据：src/uav_vpp_guidance/flight_control/enhanced_low_level_controller.py::EnhancedLowLevelController; src/uav_vpp_guidance/flight_control/pid_controllers.py::GainScheduledPIDController
- 诊断复现：源码结构检查；由源码结构检查复现；详见诊断引用。
- 观察事实：基础控制器使用固定 nz/roll-rate 执行器映射；另有命名的增益调度控制器。
- 失效模式：基础映射不随动压变化，多个 nz 修正来源可能缺少统一仲裁。
- 建议：在独立飞控任务中定义单一仲裁顺序并对 q̄/AoA 调度做闭环验证。
### F11 F11：平滑链路缺少显式舵面速率限制
- 判定：`confirmed_risk`（优先级 12，置信度 high）
- 源码证据：src/uav_vpp_guidance/flight_control/low_level_controller.py::LowLevelController.compute_actuator
- 诊断复现：filter_time_constant；滤波时间常数 τ≈0.5607 s（dt=0.2 s）。
- 观察事实：低层控制器明确使用一阶命令滤波，已识别多个可能串联的滤波阶段。
- 失效模式：未见统一的串联相位预算或显式舵面速率限制，可能使快速指令失真。
- 建议：在独立飞控任务中建立总延迟预算并依据执行器模型决定速率约束。
### F12 F12：安全奖励权重与触发区间
- 判定：`confirmed_risk`（优先级 13，置信度 high）
- 源码证据：src/uav_vpp_guidance/envs/reward.py::RewardCalculator
- 诊断复现：load_source_facts；由源码结构检查复现；详见诊断引用。
- 观察事实：RewardCalculator 的 w_safety 为最高默认权重，低于 min_alt+1000 m 时才激活。
- 失效模式：安全项在阈值附近才介入，其他权重组合仍需通过情景评估审查。
- 建议：保留默认值；后续奖励修改应以消融和奖励面诊断为依据。
### F13 F13：终端奖励量级主导单步奖励
- 判定：`confirmed_risk`（优先级 14，置信度 high）
- 源码证据：src/uav_vpp_guidance/envs/reward.py::RewardCalculator
- 诊断复现：terminal_to_step_ratio；坠毁终端奖励相对典型单步量级约 60.0×。
- 观察事实：默认终端成功/坠毁奖励为 +200/-300，w_alive 与 w_closing 默认关闭。
- 失效模式：终端信号可显著大于典型单步项；若启用存活或接近项存在投机表面。
- 建议：启用新奖励项前应进行回报分解和反投机测试。
### F14 F14：观测中缺失部分机体与历史状态
- 判定：`confirmed_risk`（优先级 15，置信度 high）
- 源码证据：src/uav_vpp_guidance/envs/observation.py::build_observation
- 诊断复现：load_source_facts；由源码结构检查复现；详见诊断引用。
- 观察事实：build_observation 的基础段为 16 维；现有可选段涵盖增益、引导状态、饱和、预测、对手阶段和任务类型。
- 失效模式：AoA、侧滑、滚转、角速度、跟踪误差与目标转弯率并非全部在基础段可见。
- 建议：如需扩展观测，必须按 AGENTS.md 变更模式、schema version、feature_names 和测试。
### F15 F15：正余弦编码丢失角度幅值的前提
- 判定：`premise_incorrect`（优先级 0，置信度 high）
- 源码证据：src/uav_vpp_guidance/envs/observation.py::build_observation
- 诊断复现：sincos_injective_witness；可执行 witness：30°=(0.49999999999999994, 0.8660254037844387)；330°=(-0.5000000000000004, 0.8660254037844384)，cos 相同而 sin 不同。
- 观察事实：30° 和 330° 的 cos 相同而 sin 不同；(sin θ, cos θ) 对在 [0,2π) 内可恢复角度。
- 失效模式：原发现的“丢失角度幅值”前提不成立。
- 建议：无需修改观测编码。
- 备注：数学诊断否证了原前提。
### F16 F16：模式切换在整回合内锁存
- 判定：`confirmed_defect`（优先级 5，置信度 high）
- 源码证据：src/uav_vpp_guidance/envs/tracking_env.py:6278
- 诊断复现：load_source_facts；由源码结构检查复现；详见诊断引用。
- 观察事实：模式开关有锁存状态，门限代码含 15° 回退值而配置给出 25°。
- 失效模式：一旦触发可持续到 reset；配置缺失时门限语义改变。
- 建议：在独立任务中明确锁存是否应释放，并通过状态转移测试覆盖缺失配置。
### F17 F17：混合制导的滞回和驻留时间
- 判定：`confirmed_risk`（优先级 16，置信度 high）
- 源码证据：src/uav_vpp_guidance/guidance/hybrid_guidance.py::HybridGuidance
- 诊断复现：dwell_seconds, range_travel；3 步×0.2 s=0.6 s；按 300 m/s 闭合约 180 m。
- 观察事实：混合制导读取 500 m 滞回和 3 步最小驻留。
- 失效模式：5 Hz 下 0.6 s 内高速闭合可跨越大量距离，可能延后模式切换。
- 建议：以代表性闭合速度和切换失败案例标定驻留/滞回。
### F18 F18：简单动力学到 JSBSim 的迁移风险
- 判定：`confirmed_risk`（优先级 17，置信度 high）
- 源码证据：src/uav_vpp_guidance/envs/tracking_env.py::CloseRangeTrackingEnv
- 诊断复现：源码结构检查；由源码结构检查复现；详见诊断引用。
- 观察事实：跟踪环境同时支持 SimplePointMassEnv 和 JSBSimEnv；F18 为 analytic-only，本审查不运行迁移实验。
- 失效模式：简单动力学省略的耦合与执行器/发动机细节会造成结构性迁移风险。
- 建议：最终 CEM/双层增益必须在 JSBSim 上重新验证；本建议仅供后续审查。
- 备注：F18 仅结构性分析；未做 run-based quantification。

## 被否证的前提
F15 的正余弦编码前提被可执行 witness 否证：在 [0,2π) 内 (sin θ, cos θ) 可恢复 θ，因此无需改动。

## 未量化项说明
F18 为 analytic-only，仅枚举简单动力学与 JSBSim 的结构差异并给出复核建议；未做 run-based quantification，未测量迁移性能退化幅度。

## 后续修复任务边界
本报告不实施任何修复；所有 recommendations 均 advisory only。任何后续控制、配置、观测、后端或 checkpoint 变更必须另立受控任务，遵守 AGENTS.md 契约；本审查未改变控制行为、配置或 checkpoint。
