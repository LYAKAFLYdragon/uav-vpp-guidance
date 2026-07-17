# Selected Lateral-Extension Post-Merge Runtime Probe V1 结果与决策

**Source ID：** `THESIS-DISADVANTAGE-TACTICAL-BASIS-POSTMERGE-LATERAL-RUNTIME-PROBE-V1`  
**运行提交：** `b086e2e48ffdc06b22d20670c9e36f7e2d7e2db8`  
**性质：** 一次性、非学习、strict-JSBSim 配对 runtime probe，`paper_safe=false`

## 结论

本 Source 的校正后结论为 **`runtime_not_evaluable`**，并触发 `freeze_source_do_not_train_or_rerun`。它既不是“候选动作没有进入执行链”，也不是“已证明 lateral-extension 无效”。候选程序在真实 JSBSim/VPP/guidance/PID 链中无裁剪地执行，但所有有效 post-merge 窗口的对方攻击区压力均为零，因而无法检验预注册的主问题“是否降低对方攻击压力”。

因此，本 Source **不允许启动单技能 pilot、共享技能库训练或任何 reward/模板/场景调参**。

## 固定方法与运行完成情况

- 比较：冻结 `fixed_head_on` 与唯一 P2 选中的 `lateral_extension:lat+1:vert+1`。
- 协议：12 个 P2 对齐初始场景、3 个对手、每个方法各一次合法 reset，共 72 个完整 episode；candidate 仅在首次实时 `disadvantage -> post_merge` handoff 后执行 40 step（8 s）时域参考，之后回到同一 fixed-head-on controller。
- 完整性：72/72 records 产生；无 backend/predictor/checkpoint fallback、无异常 reset、无 candidate non-finite action、box/slew clipping 或 basis 违规。
- 独立 verifier 从索引和 episode JSON 重建 72 条记录、连续性链、配对差值和 gate，复核通过。

候选并非空操作：candidate 共执行 993 个 program step，其中 945 个 action delta 非零；平均/峰值实际 world-reference 位移为 `309.95 m` / `690.40 m`，box/slew clipping 均为零。由此排除了“compiler 未真正把 temporal reference 送入 VPP 执行链”的解释。

## 分对手结果

| 对手 | 有效配对 | 场景 signature | 因果 handoff hash | 压力暴露 pair | Candidate/Baseline ego crash/OOB | 判定 |
|---|---:|---:|---|---:|---|---|
| expert（诊断） | 5 | 3 | 有效 handoff 一致 | 0 | 2 / 2 | 不参与 gate |
| end-to-end | 11 | 6 | 11 个 handoff 一致 | 0 | 1 / 1 | 可配对，但压力端点未激活 |
| independent PPO/VPP | 7 | 4 | 7 个 handoff 一致 | 0 | 0 / 0 | 可配对，但压力端点未激活 |

两个 required opponents 都达到有效 pair 数、signature 和双镜像覆盖，且没有新增 ego crash/OOB。然而 `target_attack_score` 在所有有效窗口的每一步均为 0，故 `pressure_exposed_pairs=0`，预注册要求的至少 3 个压力暴露 pair 不成立。主指标 `candidate - baseline mean target_attack_score <= -0.05` 因而无定义，不能用零差值伪装成改善。

二级几何量也未达到预注册 MDE：end-to-end 的平均 range-rate、末步 range、末步 specific energy 配对差分别为 `-0.78 m/s`、`-5.88 m`、`+2.65 m`；independent PPO/VPP 分别为 `-0.17 m/s`、`-1.19 m`、`-2.87 m`。这些数字只作描述，不可在主压力端点未激活时升级为“控制权威失败”。

## 分析校正

原始 `runtime_probe_gate.json` 是正式运行时立即生成的文件，保留不改。独立复核发现其汇总器有两处仅影响分类、不改变“不训练”决策的逻辑问题：

1. 它将双方均未进入 handoff 的 episode 计入 `handoff_hashes_match=false`；校正后只对实际进入 handoff 的 pair 检查哈希，两个主对手的有效 causal pair 均一致。
2. 它在 `pressure_exposed_pairs=0` 时错误归类为 `behavioral_or_control_authority_not_established`；校正后正确归类为 `runtime_not_evaluable`。

校正不修改任何 raw episode、场景、动作、输出索引或门槛。完整 episode 文件受 `records_index.json` 的 SHA-256 保护；连续性哈希保护控制链，追加的 post-step 分析字段由文件级 SHA-256 保护。校正输出见 [runtime_probe_analysis_correction.json](E:/uav-vpp-guidance-disadvantage-capability-v1-results/tactical_basis_postmerge_lateral_runtime_probe_v1/runtime_probe_analysis_correction.json)。

## 研究含义与后续

P2 已确认时域 lateral reference 在真实 post-merge handoff 上可无裁剪编译；本 Source 又确认该参考可实际通过 VPP/guidance/PID/JSBSim 执行，并且在两个主对手中保持因果配对和安全不退化。尚未建立的是 **攻击压力存在时的防御几何控制权威**，因为当前 P2 对齐 cohort 的合格窗口不包含攻击区压力。

当前路线必须在此停止。若未来另起 Source，第一步应是独立、非学习地设计并验证 `pressure-exposed post-merge` 连续数据契约，预先要求 baseline `target_attack_score > 0`，再讨论同一候选是否降低压力；不得复用本 Source 进行场景替换、重跑或参数调整。
