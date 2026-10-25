# 与 Crossing 训练契约一致的包线恢复探针结果

## 1. 实验目的

本次探针用于排除旧版恢复预检中的动作语义不一致。旧版运行时没有使用 crossing 专家训练时的战术基准、预测目标和任务相关配置，因此旧版失败不能直接说明 crossing 专家没有恢复能力。本次不训练、不调奖励、不调 PID、不调滤波器，也不修改 JSBSim，仅将运行配置恢复到 crossing 专家训练时的契约。

## 2. 固定条件与溯源

- Source ID：`THESIS-MILD-DISADVANTAGE-ENVELOPE-RECOVERY-MATCHED-CONTRACT-PROBE-V1-20260903`
- 运行目录：`E:/uav-vpp-guidance/outputs/mild_disadvantage_envelope_recovery_v1/matched_probe_v1_retry`
- 配置：`E:/uav-vpp-guidance/config/experiment/mild_disadvantage_envelope_recovery_matched_probe_v1.yaml`
- 运行脚本：`E:/uav-vpp-guidance/scripts/run_mild_disadvantage_envelope_recovery_matched_probe_v1.py`
- 验证脚本：`E:/uav-vpp-guidance/scripts/verify_mild_disadvantage_envelope_recovery_matched_probe_v1.py`
- 对手：`expert`、`end_to_end`
- 初始点：`(625 m, 200 m)`、`(750 m, 250 m)`
- 每个“对手—初始点”重复 3 次，共 12 条回合
- 运行后端：JSBSim 1.2.3，`backend_fallback_occurred=false`
- 运行时数据根目录：`E:/uav-vpp-guidance/outputs/predicted_geometry_lstm_vpp_retraining_v2/jsbsim_runtime_root`

本次使用的 crossing 契约为：

- 任务上下文：`crossing_feasible`
- 动作语义：`tactical_basis_v1`
- VPP 锚点：`predicted_target`
- 轨迹预测：启用，严格加载冻结预测器，不允许预测器初始化失败后静默替代
- crossing 观测：完整 18 维
- H0 观测：仅显式截去末端任务标志，保留前 16 个基础状态量
- 动作维度：3，专家和 H0 均不改变动作范围
- 恢复策略：最多使用 crossing 专家 48 个高层步；进入稳定点后需保持 10 步，再切回 H0

Git 溯源为当前工作树提交 `6d6f0f86e537671bab566a61a898d7b4ba52c928`，工作树当时为 dirty。因此本结果是研发证据，不直接标记为论文冻结结果。

## 3. 工程门禁

独立验证文件：`E:/uav-vpp-guidance/outputs/mild_disadvantage_envelope_recovery_v1/matched_probe_v1_retry/verification.json`

| 检查项目 | 结果 |
|---|---:|
| 回合总数 | 12/12 |
| 逐回合结果记录 | 通过 |
| 逐回合轨迹文件非空 | 通过 |
| 有效后端为 JSBSim | 12/12 |
| 后端回退 | 0/12 |
| 每回合 reset 次数为 1 | 12/12 |
| 遥测有限且非空 | 12/12 |
| 预测器严格初始化 | 通过 |
| 连续使用 crossing 专家 48 步 | 12/12 |
| 进入稳定包线 | 0/12 |

`strict_runtime_gate=pass`，说明本次探针的后端、遥测、reset 和契约标记满足运行链要求；`recovery_gate=fail`，说明恢复目标没有达到。

## 4. 逐单元结果

| 对手 | 初始点 | 进入稳定包线 | 终端原因 | 平均回合长度 | 平均最小距离（m） | 平均最小迎头角（°） |
|---|---|---:|---|---:|---:|---:|
| expert | `(625, 200)` | 0/3 | 超时（生命值优势 1，生命值劣势 2） | 512 | 227.41 | 6.55 |
| expert | `(750, 250)` | 0/3 | 我方坠毁或越界 3/3 | 275 | 278.59 | 2.95 |
| end_to_end | `(625, 200)` | 0/3 | 对方坠毁或越界 1/3，我方坠毁或越界 2/3 | 299 | 226.84 | 23.84 |
| end_to_end | `(750, 250)` | 0/3 | 对方坠毁或越界 3/3 | 285 | 278.50 | 25.73 |

四个单元均从第 0 步进入恢复阶段，crossing 专家均连续执行 48 步，随后由于没有满足稳定点保持条件而回到 H0。没有任何回合完成 10 步稳定保持，因此恢复门为 0/12。记录的逐步限幅率均为 0，未发现后端回退或中途 reset。

## 5. 与旧版探针的关系

旧版探针的 12 条回合全部以普通 VPP 语义运行，结果为 12/12 越界。matched-contract 探针改用正确 crossing 训练契约后，终端类型和回合长度发生变化，部分回合的最小距离约为 227--279 m，说明旧版确实不能代表 crossing 专家训练接口下的行为。

但是，matched-contract 探针仍然得到 0/12 稳定包线进入，且四个“对手—初始点”单元全部失败恢复门。因此，动作语义不一致已经被排除为唯一解释，但本次结果仍不能证明存在一个已经明确的新低层技能缺口。更准确的判断是：

> 在 crossing 专家训练时的完整运行契约下，现有的“48 步 crossing 恢复加 10 步保持”打法仍未能把两个不可行初始点恢复到固定 H0 的稳定包线。

这里的“失败”是恢复策略未通过预注册门，而不是严格运行链失败。对于 `end_to_end` 对手出现的对方坠毁或越界，应按合法终端事件记录，不能将其误计为我方恢复成功。

## 6. 决策

本次探针通过了 matched-contract 的严格运行门，但未通过恢复门。因此：

1. 不再重复旧版普通 VPP 语义探针，也不再用旧版结果否定 crossing 专家能力。
2. 暂不训练新的恢复技能，不调奖励、PID、滤波器、动作范围或 PPO 结构。
3. 暂不把“缺少 defensive-extension 技能”写成已经证明的结论。
4. 若继续研究，下一步应是对这 12 条连续轨迹做只读的逐层响应分析，重点检查 crossing 动作是否真正改变了 VPP、制导、滚转和几何状态，以及稳定包线判定是否与目标几何一致。
5. 如果不再追加运行实验，论文可将结论收束为：当前双技能接口在正确 crossing 契约下仍未形成可复现的包线恢复结果，但现有证据不足以把原因唯一归因于某一个新技能缺口。

## 7. 复核文件

- 汇总结果：`E:/uav-vpp-guidance/outputs/mild_disadvantage_envelope_recovery_v1/matched_probe_v1_retry/episode_results.csv`
- 独立验证：`E:/uav-vpp-guidance/outputs/mild_disadvantage_envelope_recovery_v1/matched_probe_v1_retry/verification.json`
- 总清单：`E:/uav-vpp-guidance/outputs/mild_disadvantage_envelope_recovery_v1/matched_probe_v1_retry/pilot_manifest.json`
- expert 清单：`E:/uav-vpp-guidance/outputs/mild_disadvantage_envelope_recovery_v1/matched_probe_v1_retry/expert/run_manifest.json`
- end-to-end 清单：`E:/uav-vpp-guidance/outputs/mild_disadvantage_envelope_recovery_v1/matched_probe_v1_retry/end_to_end/run_manifest.json`
- Crossing 契约审计：`E:/uav-vpp-guidance/reports/crossing_runtime_consistency_audit_v1_zh.md`
