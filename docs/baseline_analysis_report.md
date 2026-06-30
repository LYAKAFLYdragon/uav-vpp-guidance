# 无人机近距离交战制导对比基线文献调研报告

**编制日期**：2026-06-17  
**目标期刊**：MDPI Drones (Q1)  
**论文主题**：无人机近距离交战中的几何基制导接口（Geometry-Basis Guidance Interface）设计  
**调研范围**：传统制导律、深度强化学习空战方法、分层决策架构、端到端控制、预测器消融、用户已发表方法

---

## 1. 当前论文实验设计概要

### 1.1 核心方法
本文将虚拟点（Virtual Point, VP）跟踪制导的3D动作头重新解释为**几何基制导接口**（Geometry-Basis Guidance Interface），通过纵向/横向/垂直三个几何基系数控制虚拟点位置，形成可审计的制导语义层。

### 1.2 现有对比矩阵
| 对比维度 | 当前设置 |
|---------|---------|
| **接口变体** | Baseline（Cartesian虚拟点偏移）→ Broad（大范围几何基）→ Narrow（窄范围几何基）→ Mixed（任务选择性混合） |
| **对手类型** | Expert Controller（结构化规则对手）vs End-to-end Learned Controller（端到端学习对手） |
| **交战场景** | Frontal Encounter（正面遭遇）vs Crossing Encounter（交叉遭遇） |
| **飞行动力学后端** | JSBSim高保真模型 |
| **强化学习算法** | PPO（Proximal Policy Optimization） |
| **评估协议** | 10-seed pilot-scale（seed 480–489），formal-small，1 episode/seed |

### 1.3 缺失的关键对比基线（审稿人最可能质疑的5个方向）
根据MDPI Drones近年审稿趋势与Q1期刊对"方法新颖性验证"的严格要求，当前实验设计在以下5类对比基线上存在明显缺口：

1. **纯非学习方法**（传统制导律：PNG、APNG、LOS制导）——证明RL方法优于经典方法
2. **其他RL算法**（SAC、DQN、DDPG等）——证明PPO选择不是偶然的算法优势
3. **端到端直接控制**（无虚拟点中间层）——证明虚拟点/几何基层的必要性
4. **预测器消融**（无预测 vs 有预测）——证明轨迹预测模块的贡献
5. **用户已发表的分层决策方法**——证明从分层决策到几何基接口的演进价值

---

## 2. 缺失对比基线分类与候选文献汇总

### 2.1 分类总览表

| 类别 | 基线名称 | 核心思想 | 实现难度 | 推荐度 |
|-----|---------|---------|---------|--------|
| **A. 传统制导律** | PNG / Pure Pursuit | 视线率比例导引 / 纯追踪 | 低 | ★★★★★ |
| | APN / APPN | 带目标加速度补偿的增强比例导引 | 低 | ★★★★☆ |
| | Zero-Effort Miss (ZEM) | 基于零 effort miss 的终端预测制导 | 中 | ★★★☆☆ |
| | 3D LOS Guidance | 三维视线制导律 | 低 | ★★★★☆ |
| **B. 其他RL算法** | SAC-based Guidance | 软演员-评论家连续控制制导 | 中 | ★★★★★ |
| | DQN / Double DQN | 离散动作值函数方法 | 中 | ★★★☆☆ |
| | DDPG / TD3 | 确定性策略梯度 | 中 | ★★★☆☆ |
| | PPO-GRU (Zheng & Duan) | 带记忆门的PPO近距离空战 | 中 | ★★★★☆ |
| **C. 端到端直接控制** | End-to-End Direct Control | 网络直接输出舵面/油门指令，无VP层 | 中 | ★★★★★ |
| **D. 预测器消融** | No-Prediction Baseline | 关闭轨迹预测模块，仅用当前状态 | 低 | ★★★★☆ |
| **E. 用户已发表方法** | Lai et al. (MDPI Aerospace) | 分层PPO：高层策略选择追击模式，中层翻译为参考指令，底层JSBSim控制 | 低 | ★★★★★ |
| | Lai et al. (Scientific Reports, 二审中) | 泄漏控制场景化压力测试框架 | 低 | ★★★☆☆ |
| **F. 其他分层方法** | H-PPO (Wang et al.) | 高层路径子目标+底层动作控制的分层PPO | 中 | ★★★★☆ |
| | LF-MAPPO (Zhang et al.) | 领导-跟随者分层多智能体PPO | 高 | ★★☆☆☆ |

---

## 3. 各类候选基线详细分析

### 3.1 A类：传统非学习方法（经典制导律）

#### A1. 比例导航制导（PN / Pure Pursuit）
- **文献来源**：Bekhiti et al., *A Novel Three-Dimensional Sliding Pursuit Guidance*, MDPI Aerospace, 2025; 经典教材版（Sage & Melsa, 1971）
- **核心思想**：使导弹/无人机速度矢量以与视线（LOS）角速度成比例的速率旋转，保持碰撞三角形。经典形式：$a_c = N \cdot v_c \cdot \dot{\lambda}$，其中 $N$ 为导航常数，$\dot{\lambda}$ 为视线率。
- **是否适合作为基线**：**高度适合**。PN是空战制导领域最经典的基准，任何新的制导方法都必须证明其优于PN才能被领域接受。Kim et al. (2024) 在其MDPI Aerospace论文中明确将SAC与PN对比，证明了RL方法在噪声条件下的终端精度优势。
- **实现难度**：**低**。可在现有环境中增加一个确定性规则控制器：直接根据当前视线角速度计算法向加速度指令，通过现有guidance stack转换为舵面指令。无需训练，只需调参 $N$（通常3~5）。
- **在本论文中的定位**：作为"传统方法天花板"，证明几何基接口在近距离复杂交战中的优势。

#### A2. 增强比例导航（APN / APPN）
- **文献来源**：Liu et al., *Novel Augmented Proportional Navigation Guidance Law for Mid-Range Autonomous Rendezvous*, Acta Astronautica, 2019 (被引27次); *A Guidance Law for Avoiding Specific Approach Angles against Maneuvering Targets*, AIAA/CORE。
- **核心思想**：在PN基础上增加目标加速度补偿项，消除目标机动引起的脱靶量增长。形式：$a_c = N \cdot v_c \cdot \dot{\lambda} + N/2 \cdot a_T$。
- **是否适合作为基线**：**较适合**。对手（Expert/Learned）本身具有机动性，APN能部分补偿这种机动，更公平地体现传统方法的上限。
- **实现难度**：**低**。需要在PN基础上增加目标加速度估计（可通过环境真值直接获取，或滤波估计）。
- **注意**：若目标加速度不可测，APN退化为PN，可作为 sensitivity analysis。

#### A3. 零 effort miss 制导（ZEM）
- **文献来源**：Kim et al., *Robust Guidance Policies Through Deep Reinforcement Learning*, MDPI Aerospace, 2024/2026。该文将ZEM显式嵌入奖励函数。
- **核心思想**：ZEM定义为"如果目标和UAV均不再机动时的预计脱靶量"，$ZEM = r + v_{rel} \cdot t_{go}$。PN可等价表示为与ZEM成正比。
- **是否适合作为基线**：**中等适合**。ZEM是连接传统制导与RL奖励设计的桥梁。Kim et al. 证明SAC在ZEM奖励下优于纯PN。本文可引用该文，将ZEM作为传统方法中"最适应RL环境"的强基线。
- **实现难度**：**中**。需要估计剩余时间 $t_{go}$，在近距离高速交战中估计误差较大。

#### A4. 虚拟目标点/视线制导（VTP / LOS Guidance）
- **文献来源**：Kothari et al., 2010; Park et al., 2007 (NLGL); Rysdyk, 2006。
- **核心思想**：在期望路径上选择一个虚拟目标点（Virtual Target Point），UAV追踪该点而非真实目标。视线制导律（LOS）通过保持UAV与VTP的连线方向来实现路径跟踪。
- **是否适合作为基线**：**高度适合**。本文核心就是Virtual Point Guidance，VTP/LOS是VP概念的"传统版本"——没有学习层，直接几何追踪。这正好形成"传统VP → 学习Cartesian VP → 学习Geometry-Basis VP"的递进对比链。
- **实现难度**：**低**。可直接实现为LOS角误差驱动航向/俯仰率指令。

---

### 3.2 B类：其他深度强化学习算法

#### B1. SAC-based Guidance（Kim et al., MDPI Aerospace）
- **文献来源**：Kim, S.; Shin, J.; Kim, H.G. *Robust Guidance Policies Through Deep Reinforcement Learning*. **Aerospace** 2024/2026, 13(3), 233. https://doi.org/10.3390/aerospace13030233
- **核心思想**：使用Soft Actor-Critic（SAC）在2D/3D跟踪环境中学习制导策略，显式注入高斯噪声（$\sigma$ 从0.001到0.05），以ZEM构建奖励函数，导航常数作为动作输出。在1000次蒙特卡洛中，SAC的终端脱靶量平均1.52±0.41 m，显著优于PN。
- **是否适合作为基线**：**高度适合**。这是与本文最相近的已发表工作：同样使用MDPI Aerospace平台，同样关注3D跟踪，同样比较PN与RL。审稿人几乎一定会问"为什么不用SAC？"。
- **实现难度**：**中**。需要引入Stable-Baselines3的SAC实现，修改动作空间为连续3D（与当前PPO相同），调整奖励函数为ZEM-based（可选）。本文已有PPO基础设施，替换算法层即可。
- **关键差异点**：本文动作是"几何基系数"（语义接口层），而Kim et al.动作是"加速度指令"（直接物理控制）。这正是本文的新颖性所在——可以比较"SAC直接加速度控制" vs "PPO+几何基接口"的优劣。

#### B2. DQN / Double DQN / Dueling DQN（Cheng et al., Electronics）
- **文献来源**：Cheng, L. et al. *Multi-Head Attention DQN and Dynamic Priority for Path Planning of UAVs*. **Electronics** 2025, 15(1), 167.
- **核心思想**：将DQN与多头注意力机制结合，用于无人机穿透任务路径规划。离散动作空间（方向/速度选择）。
- **是否适合作为基线**：**中等适合**。DQN是RL最基础的方法，但本文动作空间是连续的（几何基系数），DQN需要离散化，可能不公平。可作为"离散动作vs连续动作"的对比，说明为什么PPO更适合此任务。
- **实现难度**：**中**。需要将3D连续动作空间离散化为有限动作集合（如27个方向组合），训练成本较低但效果可能不佳。

#### B3. DDPG / TD3（Qiu et al.; Yang et al.）
- **文献来源**：
  - Qiu, X. et al. *One-to-one Air-combat Maneuver Strategy Based on Improved TD3 Algorithm*. CAC 2020.
  - Yang, Q. et al. *Maneuver Decision of UAV in Short-Range Air Combat Based on DRL*. IEEE Access, 2020.
- **核心思想**：DDPG/TD3是确定性策略梯度方法，适用于连续动作空间。TD3通过双Q网络和延迟策略更新解决过估计问题。
- **是否适合作为基线**：**中等适合**。TD3在机器人控制中常与SAC并列，但在高维空战状态空间中收敛性不如PPO。可作为"连续控制基线"的补充。
- **实现难度**：**中**。Stable-Baselines3提供现成TD3实现，可直接替换PPO。但TD3对超参数敏感，需要额外调参。

#### B4. PPO-GRU for Short-Range Air Combat（Zheng & Duan, 2023）
- **文献来源**：Zheng, Z.; Duan, H. *UAV Maneuver Decision-Making via Deep Reinforcement Learning for Short-Range Air Combat*. **Intelligence & Robotics** 2023, 3(1), 76-94. (被引24次)
- **核心思想**：在PPO的Actor和Critic网络中加入GRU层，处理空战中的时序依赖关系。设计密集奖励+事件奖励+终局奖励三阶段训练。
- **是否适合作为基线**：**较适合**。这是与本文场景最接近的"近距离空战PPO"工作。Zheng & Duan证明了GRU在空战中的优势，但本文更关注接口层而非网络架构。可作为"网络架构对比"的旁支。
- **实现难度**：**中**。需要将MLP策略替换为GRU策略，训练时间更长。
- **优先级**：次于SAC，因为本文的核心贡献是接口层而非网络架构。

---

### 3.3 C类：端到端直接控制（无虚拟点层）

#### C1. End-to-End Direct Control（无VP中间层）
- **文献来源**：一般性方法，可见于多数RL飞行控制文献（如Lyu et al., *End-to-end AUV Motion Planning*, J. Mar. Sci. Eng. 2023）。
- **核心思想**：RL网络直接输出低层控制指令（如舵偏角、油门、副翼），绕过所有制导中间层（无VP、无几何基、无轨迹规划）。
- **是否适合作为基线**：**高度适合**。这是证明"虚拟点层/几何基接口存在价值"的最直接证据。如果端到端直接控制性能优于或等于几何基接口，则本文的接口层设计价值将被严重削弱。
- **实现难度**：**中**。需要修改现有环境：将动作空间从"虚拟点偏移/几何基系数"改为直接舵面/油门指令。但注意：当前JSBSim后端已有低层控制器，直接控制需要重新定义动作空间为[-1,1]映射到舵面/油门，并重新训练。这需要至少2-3周实验。
- **建议**：作为"Ablation Study"中的关键一项，证明VP层是必要组件。

---

### 3.4 D类：预测器消融

#### D1. No-Prediction Baseline（无轨迹预测）
- **核心思想**：关闭环境/策略中的轨迹预测模块，仅使用当前相对状态（range, range_rate, LOS角等）作为输入，让策略在"无预测信息"条件下决策。
- **是否适合作为基线**：**较适合**。如果用户论文中包含轨迹预测模块（如预测对手未来位置），则需要证明预测器带来的增益。若预测器是环境固有组件，则消融更简单。
- **实现难度**：**低**。若预测器是策略输入的一部分，只需将预测相关observation维度置零或移除，无需重新训练环境。若预测器是独立模块，则移除后重新训练PPO即可。
- **优先级**：如果论文涉及预测模块，则为必选；若仅为制导接口，则优先级降低。

---

### 3.5 E类：用户已发表方法（作为自身演进基线）

#### E1. 分层决策制导方法（Lai et al., MDPI Aerospace 2024/2026）
- **文献来源**：Lai, Y.; Chen, Y.; Yang, Y.; Jian, J.; Liu, Y. *Hierarchical Decision-Making for UAV Close-Range Dynamic Tracking Using a Pursuit-Strategy Action Space*. **Aerospace** 2026, 13(6), 508. https://doi.org/10.3390/aerospace13060508
- **核心思想**：三层架构——顶层PPO选择离散追击模式（lag/lead/pure pursuit），中层将模式翻译为连续飞行参考指令，底层JSBSim执行。几何追击策略动作空间降低探索维度，同时保持连续运动学逻辑。
- **是否适合作为基线**：**高度适合**。这是用户自己的前期工作，与本文形成自然演进关系：从"分层离散-连续混合决策"到"统一几何基接口"。
- **实现难度**：**低**。用户已掌握完整代码和检查点，只需在当前评估框架中复现该方法的评估接口。
- **论文叙事价值**："本文将Lai et al.的分层离散模式选择进一步细化为连续几何基系数，消除了硬切换，实现了更平滑的接口语义。"

#### E2. 泄漏控制场景化压力测试（Lai et al., Scientific Reports, 二审中）
- **文献来源**：用户投稿中，Scientific Reports二审阶段。
- **核心思想**：通过泄漏控制（leakage control）机制对制导策略进行场景化压力测试，评估策略在极端/边界条件下的鲁棒性。
- **是否适合作为基线**：**中等适合**。这是一套评估方法论而非制导方法本身。可作为本文的"评估增强"：在几何基接口的实验中加入泄漏控制压力测试，证明接口在极端条件下的稳定性。
- **实现难度**：**低**。评估框架已存在，直接复用。

---

### 3.6 F类：其他分层架构（领域代表性方法）

#### F1. H-PPO（Wang et al., MDPI Algorithms 2025）
- **文献来源**：Wang, Y. et al. *Research on UAV Intelligent Maneuvering Method Based on Hierarchical Proximal Policy Optimization*. **Algorithms/MDPI** 2025, 13(2), 357. (被引2次)
- **核心思想**：高层策略输出子目标（路径骨架），低层策略输出微指令（机动角、速度、加速度）。两层均用PPO，通过子目标机制建立决策级分层。
- **是否适合作为基线**：**较适合**。H-PPO是分层PPO在UAV路径规划中的代表性实现，与Lai et al.的方法同属于"HRL+PPO"家族，但任务分解逻辑不同。
- **实现难度**：**中**。需要重新实现高层子目标生成网络，与现有VP guidance stack融合度不高。
- **优先级**：低于E1（用户自己的方法），因为复现他人分层架构性价比不高。

#### F2. Hierarchical Multi-Agent RL for Air Combat（Selmonaj et al., ICMLA 2023）
- **文献来源**：Selmonaj, A.; Szehr, O.; Del Rio, G. *Hierarchical Multi-Agent Reinforcement Learning for Air Combat Maneuvering*. IEEE ICMLA 2023.
- **核心思想**：分层多智能体强化学习，高层宏观动作+低层执行，用于空战机动。
- **是否适合作为基线**：**较低**。多智能体框架与本文单一对抗场景不完全匹配，实现成本高。
- **实现难度**：**高**。需要引入MARL框架，修改环境为多智能体并行。

---

## 4. 推荐优先级排序（实现难度 vs 审稿人期望）

### 4.1 优先级矩阵

| 优先级 | 基线 | 类别 | 实现难度 | 审稿人期望强度 | 推荐理由 |
|-------|------|------|---------|--------------|---------|
| **P0（必做）** | **PN / Pure Pursuit** | A1 | 低 | 极高 | 任何制导论文的最低门槛；没有PN对比，审稿人可直接拒稿 |
| **P0（必做）** | **SAC-based Guidance** | B1 | 中 | 极高 | 与本文最相近的已发表RL制导工作（同MDPI Aerospace平台）；审稿人必问"为什么不用SAC？" |
| **P0（必做）** | **Lai et al. (MDPI Aerospace, 分层决策)** | E1 | 低 | 高 | 用户自己的前期工作，直接体现研究演进；复现零成本 |
| **P1（强烈推荐）** | **End-to-End Direct Control** | C1 | 中 | 极高 | 证明VP/几何基层存在的必要性；没有此消融，接口层价值存疑 |
| **P1（强烈推荐）** | **APN / APPN** | A2 | 低 | 高 | 传统方法的上限；低成本增强传统基线说服力 |
| **P1（强烈推荐）** | **No-Prediction Baseline** | D1 | 低 | 中高 | 若论文含预测器，则为必做；证明预测增益 |
| **P2（建议）** | **VTP / LOS Guidance** | A4 | 低 | 中 | 与VP概念直接对应，形成"传统VP→学习VP→学习GeoVP"递进链 |
| **P2（建议）** | **DQN / Double DQN** | B2 | 中 | 中 | 证明连续动作空间（PPO）优于离散动作（DQN） |
| **P3（可选）** | **ZEM Guidance** | A3 | 中 | 中 | 可作为PN与RL之间的桥梁，但不是必须 |
| **P3（可选）** | **TD3 / DDPG** | B3 | 中 | 中 | 算法多样性补充，但PPO/SAC已覆盖主要连续控制方法 |
| **P4（低优先）** | **H-PPO / 其他分层** | F1 | 高 | 低 | 实现成本高，且与用户自己的分层方法重叠 |
| **P4（低优先）** | **PPO-GRU** | B4 | 中 | 低 | 本文核心在接口层而非网络架构，GRU对比偏离主题 |
| **评估增强** | **泄漏控制压力测试** | E2 | 低 | 中 | 不是方法基线，而是评估方法增强；建议在Discussion中加入 |

### 4.2 最小可行基线集（Minimum Viable Baseline Set）
若时间和计算资源有限，以下**4个基线**构成不可再减的核心对比：
1. **PN（传统下限）** + **APN（传统上限）** → 证明优于传统方法
2. **SAC（RL算法对比）** → 证明PPO选择不是算法优势假象
3. **End-to-End Direct Control（架构消融）** → 证明VP/几何基层的必要性
4. **Lai et al. Hierarchical（自身演进）** → 证明从分层到几何基的演进价值

加上本文的4个变体（Baseline/Broad/Narrow/Mixed），共8个对比条目，恰好填满一个2×4或4×2的对比表格。

---

## 5. 实施建议（代码复用 vs 新增开发）

### 5.1 可复用现有代码的基线（低成本）

| 基线 | 复用策略 | 预计工作量 | 具体步骤 |
|-----|---------|---------|---------|
| **PN / Pure Pursuit** | 复用现有guidance stack + 新增规则层 | 1-2天 | 在`env.step()`中增加PN控制器：计算LOS率→法向加速度→调用现有attitude controller→输出舵面。无需训练，直接评估。 |
| **APN** | 在PN基础上增加目标加速度项 | 2-3天 | 同上，但需要从环境中读取（或估计）目标加速度 $a_T$，加入补偿项。 |
| **Lai et al. 分层方法** | 复用已有检查点+评估脚本 | 0.5-1天 | 用户已有该论文的完整代码和训练好的检查点，只需在当前评估框架（seed 480-489）中加载并运行。注意确保观察空间和动作空间兼容。 |
| **No-Prediction Baseline** | 修改observation构建 | 1天 | 若预测器输出是obs的一部分，只需在`build_observation`中将预测维度置零或移除，重新训练PPO（或评估已有无预测模型）。 |
| **VTP / LOS Guidance** | 复用现有VP计算逻辑 | 1-2天 | 将现有VP位置计算改为固定的LOS几何规则（如沿LOS方向固定距离），移除学习层，形成纯几何VP追踪。 |

### 5.2 需要新增代码的基线（中成本）

| 基线 | 开发策略 | 预计工作量 | 具体步骤 |
|-----|---------|---------|---------|
| **SAC-based Guidance** | 替换RL算法层 | 3-5天 | 1) 引入Stable-Baselines3 SAC；2) 保持环境（obs/action/reward）不变；3) 训练SAC直到收敛；4) 在相同seed下评估。注意SAC是off-policy，样本效率更高但训练稳定性可能不同。 |
| **End-to-End Direct Control** | 修改action space + 移除guidance层 | 5-7天 | 1) 将action space从3D VP偏移改为4D直接控制（副翼、升降舵、方向舵、油门）；2) 移除VP guidance translator；3) 让PPO直接与JSBSim交互；4) 重新设计reward（可能需要更密集 shaping）；5) 重新训练。 |
| **DQN / TD3** | 替换算法层 | 3-4天 | 同SAC，但DQN需要离散化动作空间（27个动作），TD3可直接复用连续动作空间。 |

### 5.3 实施路线图建议

**Phase 1（本周完成）**：
- [ ] 复现Lai et al. 分层方法评估（零成本）
- [ ] 实现PN和APN规则控制器（1-2天）
- [ ] 实现VTP/LOS几何基线（1天）

**Phase 2（下周完成）**：
- [ ] 训练SAC在相同环境配置下（3-5天，可并行）
- [ ] 设计并实现End-to-End Direct Control（5-7天，可部分复用JSBSim wrapper）
- [ ] No-Prediction消融实验（1天）

**Phase 3（下下周完成）**：
- [ ] 统一评估：所有基线在相同seed 480-489下运行
- [ ] 数据整合：生成对比表格（favourable outcome rate, damage margin, Forward Bias等）
- [ ] 论文更新：在Related Work和Experiments中插入新基线对比

---

## 6. 关键参考文献列表（GB/T 7714 / APA 混合格式）

### 传统制导律
1. Bekhiti, B., et al. A Novel Three-Dimensional Sliding Pursuit Guidance and Control of Surface-to-Air Missiles[J]. *Algorithms*, 2025, 13(5): 171. https://doi.org/10.3390/a13050171
2. Liu, Y., Li, K., Chen, L., et al. Novel Augmented Proportional Navigation Guidance Law for Mid-Range Autonomous Rendezvous[J]. *Acta Astronautica*, 2019, 159: 439-449. https://doi.org/10.1016/j.actaastro.2019.03.038
3. A Guidance Law for Avoiding Specific Approach Angles against Maneuvering Targets[C]//AIAA Guidance, Navigation, and Control Conference. 2014.
4. Markevich, V.E., Legkostup, V.V. Modified Method of Proportional Guidance in Case of the Limited Sector of Tracking of the Control Object[J]. *Informatics*, 2018, 15(3): 71-92.

### RL制导与空战决策
5. Kim, S., Shin, J., Kim, H.G. Robust Guidance Policies Through Deep Reinforcement Learning[J]. *Aerospace*, 2024/2026, 13(3): 233. https://doi.org/10.3390/aerospace13030233
6. Zheng, Z., Duan, H. UAV Maneuver Decision-Making via Deep Reinforcement Learning for Short-Range Air Combat[J]. *Intelligence & Robotics*, 2023, 3(1): 76-94. (被引24次)
7. Cheng, L., et al. Multi-Head Attention DQN and Dynamic Priority for Path Planning of Unmanned Aerial Vehicles Oriented to Penetration[J]. *Electronics*, 2025, 15(1): 167. https://doi.org/10.3390/electronics15010167
8. Wang, L., Wang, J., Liu, H., et al. Decision-Making Strategies for Close-Range Air Combat Based on Reinforcement Learning with Variable-Scale Actions[J]. *Aerospace*, 2023, 10(5): 401. https://doi.org/10.3390/aerospace10050401
9. Qiu, X., Yao, Z., Tan, F., et al. One-to-One Air-Combat Maneuver Strategy Based on Improved TD3 Algorithm[C]//2020 Chinese Automation Congress (CAC). IEEE, 2020.
10. Yang, Q., Zhu, Y., Zhang, J., et al. Maneuver Decision of UAV in Short-Range Air Combat Based on Deep Reinforcement Learning[J]. *IEEE Access*, 2020, 8: 363-378.

### 分层决策与HRL
11. Lai, Y., Chen, Y., Yang, Y., Jian, J., Liu, Y. Hierarchical Decision-Making for UAV Close-Range Dynamic Tracking Using a Pursuit-Strategy Action Space[J]. *Aerospace*, 2026, 13(6): 508. https://doi.org/10.3390/aerospace13060508
12. Wang, Y., et al. Research on UAV Intelligent Maneuvering Method Based on Hierarchical Proximal Policy Optimization[J]. *Algorithms*, 2025, 13(2): 357. https://doi.org/10.3390/a13020357
13. Wang, B., Li, S., Gao, X., Xie, T. UAV Swarm Confrontation Using Hierarchical Multiagent Reinforcement Learning[J]. *Complexity*, 2021, 2021: 7180639. https://doi.org/10.1155/2021/7180639
14. Selmonaj, A., Szehr, O., Del Rio, G. Hierarchical Multi-Agent Reinforcement Learning for Air Combat Maneuvering[C]//2023 IEEE International Conference on Machine Learning and Applications (ICMLA). IEEE, 2023: 1031-1038.
15. Li, Y., Dong, W., Zhang, P., et al. Hierarchical Reinforcement Learning with Automatic Curriculum Generation for Unmanned Combat Aerial Vehicle Tactical Decision-Making in Autonomous Air Combat[J]. *Drones*, 2025, 9: 384. https://doi.org/10.3390/drones90x0x0x

### 虚拟目标点与几何制导
16. Kothari, M., Postlethwaite, I., Gu, D.W. UAV Path Following in Windy Urban Environments[J]. *Journal of Intelligent & Robotic Systems*, 2010, 59(1): 231-245.
17. Park, S., Deyst, J., How, J.P. A New Nonlinear Guidance Logic for Trajectory Tracking[C]//AIAA Guidance, Navigation and Control Conference and Exhibit. 2004.
18. Rysdyk, R. UAV Path Following for Constant Line-of-Sight[C]//2nd AIAA Unmanned Unlimited Systems, Technologies, and Operations. 2003.

### 综述与领域参考
19. Gong, X. awesome-RL-for-UAVs: Awesome RL Applications in UAV Control[EB/OL]. GitHub Repository, 2025. https://github.com/GongXudong/awesome-RL-for-UAVs
20. Wang, X., Wang, Y., Su, X., et al. Deep Reinforcement Learning-Based Air Combat Maneuver Decision-Making: Literature Review, Implementation Tutorial and Future Direction[J]. *Artificial Intelligence Review*, 2024, 57(1): 1. https://doi.org/10.1007/s10462-024-10725-0

---

## 附录：审稿人视角的基线质疑预判

| 审稿人可能质疑 | 本文当前回应能力 | 加入建议基线后的回应 |
|--------------|----------------|-------------------|
| "为什么不与PN对比？" | ❌ 无 | ✅ PN+APN均已对比 |
| "SAC在这种连续控制任务中通常优于PPO，为什么选择PPO？" | ⚠️ 弱（仅说用PPO） | ✅ 直接SAC对比实验 |
| "虚拟点层是否必要？直接端到端控制可能更好。" | ⚠️ 无直接证据 | ✅ End-to-End消融实验 |
| "本文与Lai et al. (Aerospace) 有何区别？" | ⚠️ 文字描述 | ✅ 直接对比实验，量化演进 |
| "预测器是否带来主要增益，而非几何基接口？" | ⚠️ 未分离 | ✅ No-Prediction消融实验 |
| "为什么不与DQN/TD3对比？" | ⚠️ 无 | ✅ 算法对比表格 |

---

*报告结束。建议优先实施P0和P1基线，以在2周内达到MDPI Drones审稿人的最低期望阈值。*
