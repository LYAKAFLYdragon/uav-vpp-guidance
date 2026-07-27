# 控制与制导方法整改执行清单 v1

## 1. 文档目的

本清单用于依次处理控制方法合理性审查中的 F01-F18 问题。目标不是直接追求胜率，而是先建立一条可复现、可审查、可回滚的证据链：

    问题复核 -> 数学与坐标定义 -> 单元测试 -> 接口链路审计
    -> strict-JSBSim 配对验证 -> 安全与非退化门 -> 默认版本决策
    -> 论文结果更新

特别注意：F01 的离线候选公式通过，不等于默认 LOS-rate 制导律已经修复；F01/F02 只有在闭环物理证据通过后，才可以考虑从显式试验模式升级为正式方法。

## 2. 固定规则

### 2.1 工作区与版本

- [ ] 从当前主项目建立新的 clean worktree，不在已有脏工作树中直接整改。
- [ ] 固定新的 Source ID：THESIS-CONTROL-GUIDANCE-REMEDIATION-V1。
- [ ] 固定输出目录：outputs/control_guidance_remediation_v1/。
- [ ] 记录 clean SHA、分支、Python 版本、平台、JSBSim 版本和运行命令。
- [ ] 保存整改前 baseline 的代码、配置、checkpoint 和 SHA-256 清单。
- [ ] 不合并无关的历史分支提交，不覆盖当前论文、实验和控制逻辑改动。

### 2.2 不可违反的保护条件

- [ ] 保留 legacy 控制路径作为 baseline，不直接覆盖默认行为。
- [ ] 不在证据门通过前修改训练奖励、PPO 结构、训练时长或 checkpoint。
- [ ] 不修改 F04 安全限幅：nz ∈ [-2, 7] g、roll_rate ∈ [-1.5, 1.5] rad/s、油门范围保持配置值。
- [ ] 不改变 F15 的正弦/余弦角度编码，除非重新提出独立证据。
- [ ] 所有配置覆盖都使用 record_config_override 记录。
- [ ] strict-JSBSim 结果必须满足 backend == "jsbsim" 且 backend_fallback_occurred == false。
- [ ] simple backend、fallback、缺失 checkpoint、缺失场景和缺失遥测均不能作为正式物理结论。
- [ ] 任何新实现必须有独立模式名；不得让旧 checkpoint 静默加载新观测、动作或控制语义。

### 2.3 统一结果分类

每个阶段的结果只能标记为以下之一：

- pass：本阶段预注册门全部通过，可以进入下一阶段。
- needs_more_evidence：实现或离线分析存在，但证据不足，不得推广。
- physical_unavailable：strict-JSBSim、checkpoint 或场景不可用，不能归因于控制器失败。
- safety_fail：出现新增 crash、OOB、非有限量、fallback 或严重饱和，立即冻结候选。
- reject：数学、接口或闭环结果不满足门槛，保留 baseline 并停止该整改支线。

## 3. P0：证据和接口冻结

- [x] 导入并核对现有 F01-F18 审计报告、remediation_ledger.json 和 preservation baseline。
- [x] 对当前分支重新计算受保护路径 SHA-256；发现 preservation baseline 自哈希与冻结记录不一致，已作为证据接口错误冻结，未改写 baseline。
- [x] 以 Git/raw-byte 取证保留旧 freeze，并建立 superseding preservation freeze；该收口不改变资产或物理闭环门状态。
- [x] 建立 run_manifest.json、artifact_contract.json 和 portable bundle 模板。
- [x] 固定场景矩阵、对手、随机种子、运行步数、终端分类和统计分母。
- [x] 对 E 盘 32 个 UAV worktree/archive 根目录中的 3,726 个 loose checkpoint、25 个 ZIP/TAR/TGZ/TAR.GZ 容器、11 个本地 Git 历史与 LFS 元数据做只读 SHA-256 盘点；未找到 `prediction_vpp` 登记路径或预测器冻结 SHA，且 policy freeze 对前三项未给 SHA，未采用外部候选。
- [x] 核验用户提供的 authority-collection receipt：源 `outputs` 实为 CAN assets Junction，且无 Git/训练 run manifest；predictor 严格结构加载虽通过但 SHA 不匹配，登记 VPP 缺失，两个 policy 仍为来源不完整候选。
- [x] 核验论文、实验计划、对手能力卡和 CAN 权威索引的原始字节与引用边界：它们证明方法记录或候选 identity，不能替代缺失 checkpoint bytes、精确 predictor SHA 或独立训练 provenance；未据此创建资产契约。
- [x] 补齐 checkpoint-free 的 `scripted_evasive_vpp` 第三独立对手；已完成 3D VPP 契约、动态注册和 dry-run artifact/bundle 核验，未改变观测 schema、控制、奖励或安全限幅。
- [ ] 运行工程 smoke：阻塞。关键登记 checkpoint 仍缺失/不匹配；未运行 strict-JSBSim episode、simple backend 替代或任何物理闭环。

P0 状态：`physical_unavailable`。P0 验收未通过；manifest、配置、代码和 checkpoint 哈希必须可独立读取，缺少任一关键输入则停止正式整改。

## 4. P1：F01 LOS-rate 制导律物理语义

### 4.1 冻结定义

- [ ] 明确坐标系为 NEU 或 NED，并在代码、配置、遥测和文档中保持一致。
- [ ] 明确 r = p_vp - p_own、v = v_vp - v_own、距离、闭合速度和单位。
- [ ] 定义垂直平面 LOS 角 lambda = atan2(r_z, rho)。
- [ ] 定义 LOS 角速度：

    rho_dot = (r_x v_x + r_y v_y) / rho
    lambda_dot = (rho v_z - r_z rho_dot) / R^2

- [ ] 明确正负号含义、g0 单位和适用范围。
- [ ] 明确 nz 的组成：几何项、LOS-rate/PN 项、基础项和最终限幅项。
- [ ] 删除或隔离原有随距离线性增长的 k_pos * distance 语义；如果保留，必须证明其确实是高度/能量误差项而不是距离伪控制项。

### 4.2 离线验证

- [ ] 零 LOS-rate 时，LOS-rate 载荷贡献为零。
- [ ] 镜像上下几何时，角度和载荷贡献符号反向。
- [ ] 固定 LOS 角改变距离时，几何载荷不随距离增长。
- [ ] 非有限状态、近零水平距离和 capture 区域有确定的安全处理。
- [ ] 输出始终有限，并保持 F04 限幅。

### 4.3 物理语义复核

- [ ] 由独立复核者确认坐标、符号、导航常数、速度来源和 VPP/目标速度 fallback。
- [ ] 确认候选模式不会把旧 checkpoint 静默解释为新控制语义。
- [ ] 复核通过前，只允许 f01_f02_remediated 显式 opt-in，不能改为默认模式。

P1 决策：数学失败则回到公式和测试；数学通过但物理语义未通过则标记 needs_more_evidence，禁止进入正式性能比较。

## 5. P2：F02 滚转双环接口

- [ ] 明确外环：航向误差 -> 期望侧倾角 phi_cmd。
- [ ] 明确内环：phi_cmd - phi -> 滚转角速度指令，并使用真实 p 做阻尼。
- [ ] 不再把 roll_rad 当作滚转角速度阻尼量。
- [ ] 明确协调转弯关系 r = g tan(phi) / V 及其适用范围。
- [ ] 验证速度扫描、相反输入、滚转角释放、滚转角速度阻尼和 F04 限幅。
- [ ] 记录航向误差、phi_cmd、实际 phi、实际 p、滚转角速度指令和饱和状态。
- [ ] 与旧单环 baseline 做同状态离线比较，不修改默认控制器。

P2 验收：外环和内环信号可区分，真实滚转角速度参与阻尼，所有命令有限且不突破 F04 限制。

## 6. P3：F03/F11 滤波、执行器和动力学预算

- [ ] 列出 PN、制导、低层指令、执行器和 JSBSim 之间的所有滤波器。
- [ ] 对每一级记录输入、输出、alpha、时间步长、rate limit、clipping 和延迟。
- [ ] 计算每级时间常数、相位延迟、幅值衰减和总链路预算。
- [ ] 增加执行器实际舵面位置和速率遥测。
- [ ] 区分“指令已不同但舵面没有不同”和“舵面不同但物理响应没有不同”。
- [ ] 只做固定短窗口响应探针，不调参数、不训练。

P3 决策：

- 差异在滤波/执行器层消失：建立独立接口整改方案。
- 舵面和滚转响应已不同但几何量无方向：停止滤波线，转做制导闭环分析。
- 只有一个对手响应：标记对手依赖，不提出共享控制能力结论。

## 7. P4：F05 VPP 动作语义和包线

- [ ] 固定动作维度和轴顺序，明确 [dx, dy, dz] 的坐标方向、单位和物理范围。
- [ ] 缺少 action_dim 或维度不一致时显式报错，禁止静默回退 5D。
- [ ] 5D 旧输入只能通过显式 vpp_5d_legacy 兼容模式运行。
- [ ] 比较固定米制映射、距离缩放映射和角度/几何映射的单调性、对称性和裁剪率。
- [ ] 对远距离和近距离分别计算动作对应的几何角度，避免同一米制动作语义漂移。
- [ ] 为新动作契约迁移记录 checkpoint、配置和 observation/action schema 哈希。

P4 验收：同一动作在声明的包线内具有稳定、可解释的几何含义，且不存在隐藏的 3D/5D 混用。

## 8. P5：F06 多速率与延迟

- [ ] 固定 RL 决策频率、制导频率、PID/FCS 频率和 JSBSim 子步频率。
- [ ] 明确零阶保持、插值、延迟和每次更新的时间戳。
- [ ] 对非整数频率比直接拒绝，不允许静默漂移。
- [ ] 验证 reset 后滤波状态、hold 状态和延迟队列全部清空。
- [ ] 对动作阶跃、正弦输入和相反阶跃分别报告延迟、幅值衰减和相位。

P5 验收：调度次数、时间戳和保持值可重现；总延迟预算被记录，不把滤波时间常数误称为全部系统延迟。

## 9. P6：F07 动作契约迁移

- [ ] 运行 3D 合法动作、缺失维度、错误维度、过期 checkpoint metadata 测试。
- [ ] 检查生成器、环境、PPO、warm start、specialist loader 和评估器使用同一契约。
- [ ] 生成新的 action schema manifest 和 checkpoint compatibility report。
- [ ] 不将旧 5D checkpoint 直接用于 3D 环境；需要显式迁移或标记不可用。

P6 验收：3D 契约实现并验证；旧路径只能显式兼容，不能静默回退。

## 10. P7：F08/F09 CEM 与双层优化

- [ ] 固定候选数、精英数、随机种子和预算。
- [ ] 验证精英样本数足以支撑协方差/方差估计。
- [ ] 报告候选覆盖、收敛曲线、噪声下稳定性和固定预算结果。
- [ ] 将当前经验差距与真正 regret 分开命名。
- [ ] 明确 outer policy 与 inner optimizer 的更新顺序、回滚条件和 checkpoint snapshot。
- [ ] 若未建立 best-response 或回滚证据，不得声称双层优化收敛。

## 11. P8：F10/F17 增益调度、仲裁和模式状态机

- [ ] 记录动态压、迎角、高度和速度对 PID/前馈增益的影响。
- [ ] 统一 guidance、energy、altitude、stall 和 bank recovery 对 nz 的仲裁优先级。
- [ ] 明确饱和前值、饱和后值和 anti-windup 状态。
- [ ] 对模式切换增加 dwell、hysteresis、进入原因、释放原因和锁存状态遥测。
- [ ] 分别比较永久锁存、可释放锁存和状态机方案，不混合解释。
- [ ] 任何新的保护规则都必须以 offline arbitration 证据先行，不能直接进入默认运行链。

## 12. P9：F12/F13 奖励、终端回报和反事实

- [ ] 分解逐步奖励、终端奖励、安全惩罚、饱和惩罚和能量项。
- [ ] 分开报告胜利、失败、平局、我方 crash/OOB、目标 crash/OOB 和 timeout。
- [ ] 设计只读反事实检查，确认奖励项没有意外改变其他模块语义。
- [ ] 不把对方 crash/OOB 当作我方安全失败；终端定义必须与论文口径一致。
- [ ] 未完成奖励消融前，不修改正式训练奖励。

## 13. P10：F14 观测契约和信息泄漏

- [ ] 固定基础 16 维的顺序、名称、单位和归一化方式。
- [ ] 对每个可选观测扩展增加 schema version、维度、feature names 和配置开关。
- [ ] 检查 future state、oracle 标签、终端标签和对手真实内部状态是否泄漏。
- [ ] 记录 observation schema、checkpoint 输入维度和训练配置哈希。
- [ ] schema 变化必须迁移 checkpoint，不得依靠 reshape 或截断静默兼容。

## 14. P11：F15/F16 角度编码与模式切换

- [ ] 用可执行 witness 证明正弦/余弦编码的周期性和可区分性，不把错误前提作为问题。
- [ ] 固定模式阈值的配置来源，禁止代码 fallback 覆盖显式配置。
- [ ] 对模式首次进入、有效模式、保护仲裁和退出条件分别记录。
- [ ] 报告模式差异率、首次切换时刻、切换后几何变化，不只报告胜率。

## 15. P12：F18 backend 和迁移预检

- [ ] 使用相同场景、种子、策略和配置做 simple/strict-JSBSim 成对运行。
- [ ] 校验最终 backend、fallback 标志、状态字段、终端原因和遥测维度。
- [ ] backend fallback、缺失 JSBSim 数据或异常 reset 时，结果标记为 physical_unavailable。
- [ ] 不把 simple backend 的改善或退化直接写成 JSBSim 物理结论。

## 16. P13：F01/F02 正式闭环验证

只有 P0-P12 的相关接口门通过后，才允许执行本阶段。

- [ ] 固定 legacy baseline、F01/F02 opt-in candidate、相同 checkpoint、相同场景和相同 seed。
- [ ] 至少包含 head-on、crossing、post-merge/re-entry 目标场景。
- [ ] 三对手分开报告，禁止合并成单一平均值。
- [ ] 每回合记录：nz_cmd、LOS angle/rate、phi_cmd、实际 phi、p、舵面、距离、距离率、ATA、AA、能量、高度、clipping、fallback、crash/OOB 和 terminal reason。
- [ ] 使用真实 paired difference、Wilson 区间和安全事件分母。
- [ ] 候选必须满足预注册的安全非退化门，不能只依靠胜率。

P13 Go：strict-JSBSim 可复现、物理方向一致、两个以上对手有共同几何改善且安全不恶化。

P13 No-Go：保留 legacy baseline；候选冻结为负证据，不调奖励、不换场景、不延长训练来掩盖失败。

## 17. P14：正式版本和论文收口

- [ ] 只有 P13 通过，才讨论把候选模式提升为正式方法版本。
- [ ] 若未通过，论文应准确写成“候选律完成数学和接口验证，但闭环物理有效性未建立”，不能写成默认制导律已修复。
- [ ] 生成最终 evidence manifest、portable bundle、结果摘要和 source-to-claim 矩阵。
- [ ] 更新论文中的方法名称、坐标定义、动作含义、实验范围、对手范围和限制边界。
- [ ] 任何 headline 数值必须能追溯到 Source ID、场景分母、checkpoint SHA 和 strict backend 证据。

## 18. 统一测试与交付清单

- [ ] 单元测试：公式、符号、边界、动作契约、调度和状态机。
- [ ] 接口测试：各级遥测字段完整，维度和单位一致。
- [ ] 物理 smoke：三对手各一条完整 strict-JSBSim episode。
- [ ] 正式运行：固定矩阵、固定种子、固定配置，不中途改参数。
- [ ] 结果验证：独立 verifier 可从 bundle 中重算哈希和 gate。
- [ ] git diff --check 通过。
- [ ] 受保护路径变更与批准 allowlist 一致。
- [ ] 不存在未登记 checkpoint、fallback 结果或临时配置覆盖。
- [ ] 每阶段完成后提交一次小提交，不 amend，不覆盖已有研究证据。

## 19. 推荐执行顺序

    P0 证据冻结
     -> P1 F01 LOS-rate 语义
     -> P2 F02 滚转双环
     -> P3 F03/F11 链路预算
     -> P4 F05/F07 VPP 契约
     -> P5 F06 多速率
     -> P8 F10/F17 保护与切换
     -> P9/F10/F14 奖励、观测和状态机审计
     -> P12 F18 strict-JSBSim 预检
     -> P13 F01/F02 闭环成对验证
     -> P14 版本提升与论文收口

任一阶段出现 safety_fail 或 physical_unavailable，只进入对应的诊断分支，不自动进入下一阶段；任一阶段出现 reject，保留旧 baseline 并停止该整改支线。
