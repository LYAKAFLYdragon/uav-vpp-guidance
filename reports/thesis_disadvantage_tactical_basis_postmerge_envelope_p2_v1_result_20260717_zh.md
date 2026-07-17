# Post-Merge P2 Offline Envelope Screen V1 结果与决策

**Source ID：** `THESIS-DISADVANTAGE-TACTICAL-BASIS-POSTMERGE-ENVELOPE-P2-V1`  
**日期：** 2026-07-17  
**性质：** 一次性、非学习、纯离线 compiler-envelope screen，`paper_safe=false`

## 一句话结论

在 V3 已验证的连续 `disadvantage -> post_merge` 基线窗口中，P1 冻结的全部 8 个时域参考模板都可以在当前 3-D VPP action envelope 内无 box/slew 裁剪地离线编译；按预注册的唯一排序规则，选中 `lateral_extension:lat+1:vert+1`。这只建立了 **post-merge 参考轨迹的接口可执行性**，不证明几何改善、胜率改善、飞行安全或训练收益。

## 冻结输入与执行边界

- V3 连续数据契约：36 条 strict-JSBSim baseline records 中 17 条合格窗口，包含 680 个完整 post-merge step。
- 输入 cohort：`expert=5`、`end_to_end=5`、`independent_ppo_vpp=7`；后两个对手是预注册 gate 的 required opponents，`expert` 仅为诊断对手。
- 模板：P1 的 `lateral_extension` 2 个符号变体、`vertical_extension` 2 个、`combined_break_extension` 4 个，共 8 个；每条窗口 40 step、8 s、`sin_squared_c1`，`max_action_delta=0.35`、`max_action_slew=0.10`。
- 本次只重放存储的 action/basis 序列并运行 compiler。候选 action 未进入环境，没有调用 JSBSim，没有训练、调参、改 reward、改 routing、改 VPP/guidance/PID 或扩大动作包线。

## 预注册门与结果

每个 required opponent 的同一候选都必须至少有 4 条状态、4 个 scenario signature、正负镜像齐全，并满足：box clipping = 0、slew clipping = 0、平均投影残差不超过 50 m、峰值实际 world-reference 位移至少 100 m、basis condition 不超过 100。

| 对手 | 合格窗口 | signature | 双镜像 | 8 个候选的 gate 结果 |
|---|---:|---:|---|---|
| `end_to_end` | 5/5 | 4 | 是 | 全部通过 |
| `independent_ppo_vpp` | 7/7 | 4 | 是 | 全部通过 |
| `expert` | 5/5 | 3 | 是 | 诊断记录，不参与 gate |

136 条 state-template row 的全局执行事实如下：box clipping 全为 0；slew clipping 全为 0；最大平均投影残差为 `1.95e-14 m`（数值舍入量级）；最小峰值实际 world-reference 位移为 `273.20 m`；最大 basis condition 为 `3.10`。因此，P2 的物理编译门通过。

8 个候选均通过 required-opponent gate。按“最大化两对手中最小可行状态数，再最大化最小峰值位移，再最小化最大平均残差，最后按 variant ID”这一冻结规则，唯一选中：

`lateral_extension:lat+1:vert+1`

其最小峰值实际 world-reference 位移为 `581.12 m`（`end_to_end`）和 `579.65 m`（`independent_ppo_vpp`）。

## 独立复核与可追溯性

执行时 worktree 为 clean，运行提交为 `e2b79d1600eeb0f25440107c31ce78feb329e6ea`。独立 verifier 未导入 runner、从 V3 index 重新读取并哈希每条输入记录、重新编译全部 136 行，并与输出矩阵和汇总逐项相等；复核通过。

| 产物 | SHA-256 |
|---|---|
| `offline_envelope_matrix.csv` | `7f4f724a5486ecddf6a79457b250ce0adf987c39e9912c8dea9c91a37e15cca9` |
| `offline_envelope_matrix.json` | `01371451443194937b679fc494ad9ac0e9853263c96726f360becb35c1412963` |
| `offline_envelope_summary.json` | `31aca3af55e055a9d4a187e917b0f464ddab56ca6095db162e7246276a0475a6` |
| `p2_verification.json` | `2cad26cfe2cf936ba383aad62a4f40f2bdf36104a0d849df52dbbfe962319ab6` |
| `run_manifest.json` | `5f9a1427df6dcad98f142e072d84bd51550ca56f6c994ec295672a293e12d752` |

## 决策与声明边界

**允许：** 新建独立 Source，仅设计一个以 `lateral_extension:lat+1:vert+1` 为唯一试验候选、与固定 head-on baseline 配对的非学习 runtime probe。该 probe 必须重建 manifest、几何 MDE、安全 stop rule 和独立 verifier；P2 不自动授权运行该 probe。

**仍然禁止：** 启动 shared-skill 或 temporal-reference 训练、调整模板/阈值、增加训练步数、改变 reward、放宽 action envelope，或将本 P2 结果写成性能或安全结论。

**不能声称：** “模板改善了 range/AA/ATA/能量/高度”，“模板提高胜率”，“所有对手下安全”，或“defensive-extension 技能库缺口已被解决”。P2 只排除了“冻结模板在两个主要对手的真实 post-merge handoff 上因动作包线立即不可编译”这一障碍。

## 与旧 P2 的关系

原 `THESIS-DISADVANTAGE-TEMPORAL-REFERENCE-REACHABILITY-V1` P2 因历史记录仅含 `cartesian_offset` 而不具备 `tactical_basis_v1` 输入，被封存为 `p2_input_contract_incompatible`。本 Source 使用 V3 新建的连续同语义数据契约，不能倒推旧 P2 错误，也不改变其负证据性质。
