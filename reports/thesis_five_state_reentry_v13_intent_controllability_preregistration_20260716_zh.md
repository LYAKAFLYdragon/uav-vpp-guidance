# V13 Six-Objective Intent-Controllability Probe 预注册

**Source ID：** `THESIS-FIVE-STATE-REENTRY-INTENT-CONTROLLABILITY-V13`
**状态：** design-only；尚未授权执行
**性质：** 非学习、非奖励、机制诊断；不是论文主结果

## 1. 与 V12 的关系

V12 在任何环境、JSBSim step 或 episode record 产生前因缺少 `execution.runtime_template` 停止，`completed_records=0`。它已独立归档为实现失败，不能被重跑或补写。本轮使用新的 Source ID、新的 7200 m/306/345 m/s manifest、新输出根和新的 hash-bound authorization；不复用 V12 的场景作为实验数据。

除这一运行时合同修复外，V13 严格保留 V12 的研究问题、full-factorial 语义轴组合、六维物理 telemetry、三对手分开报告、P3/预测器/VPP/guidance/PID 冻结和 20-step continuity window。

## 2. 新增的 fail-closed runtime 合同

在输出目录创建、JSBSim 初始化或策略加载之前，preflight 必须验证：

1. `execution.runtime_template` 存在。
2. 它与 `inputs.runtime_template` 是同一仓库路径。
3. 该文件 SHA-256 与 `inputs.runtime_template_sha256` 一致。

任一条件失败即拒绝执行，不得生成 episode、partial ledger 或结果 gate。该合同有专门单元测试覆盖缺失键路径。

## 3. 方法与矩阵

固定 `range_extension` profile，在冻结 relative-frame compiler 上穷举 range、signed-turn、vertical-energy 三个二值轴：reference、三个单轴、三个二轴和一个三轴全开，共 `2^3=8` methods。每个方法在 `expert`、`end_to_end`、`independent_ppo_vpp` 三对手以及 12 个全新物理场景下连续运行，计划 `288` 条记录。

每个 raw-SI re-entry handoff 后的 20-step window 记录：六维 physical intent vector `AA, ATA, range, range-rate, specific-energy, altitude`、相对 profile target 的归一化绝对误差、semantic compiler receipt、VPP world offset、guidance command、actual n_z 和 saturation。

## 4. 判定规则

对每个 opponent，所有非参考方法仅与相同 handoff hash 的参考记录配对。一个 action combination 通过该 opponent 的 local compatibility 条件需要：至少 4 个完整 pair；range/range-rate 分量不恶化超过 `+0.02`；任一分量不恶化超过 `+0.05`；至少两个分量改善 `>=0.01`；profile-weighted 六维误差改善 `>=0.01`；ego crash/OOB 不超过参考 `+0.05`。

只有同一组合在至少两个对手独立通过，才记为 `local_multiobjective_candidate`。这不是训练授权：通过后只允许起草 decision memo；失败后禁止修改 V13 的 scale、window、profile、对手或场景补跑。
