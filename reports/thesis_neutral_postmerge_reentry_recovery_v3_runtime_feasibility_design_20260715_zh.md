# Neutral Post-Merge Reentry-Recovery V3 Runtime-Feasibility 设计

**Source ID：** `THESIS-NEUTRAL-POSTMERGE-REENTRY-RECOVERY-RUNTIME-FEASIBILITY-V3`  
**状态：** `preregistered_design_execution_not_authorised`  
**范围：** 非学习、非论文安全的三对手运行时 phase-support 预检。

## 1. 为什么需要 V3

V1 在第一个 dev ledger 的 metadata 读取处失败，V2 修正了 evaluation manifest
pair key 后，又在第一条 train scenario 把 evaluation identity 强加给 train
serializer 而失败。V2 在失败前完整写入的 72 条 frozen baseline 还表明，其 `dev12`
在相同 pilot engine 下只有 expert `1/12`、end-to-end `0/12`、independent PPO/VPP
`1/12` 到达 neutral/post-merge handoff，不能支持后续每对手最少 2 条或 claim-ready
8 条的门槛。

此前 B6 在另一条 profile-free reachability 线上将 `neutral -> post_merge` 识别为
候选 motif，但它不能替代本 pilot engine 的运行时证据，也不能复用其物理场景。V3
因此只问一个更基础的问题：**在未来 neutral/post-merge pilot 将使用的 frozen
head-on run-in、66-D/P3/prediction-VPP/guidance/PID/JSBSim 链上，是否存在一个预先
定义的几何 family 能同时为三个对手提供足够的实际 handoff 与有效 target step？**

它不训练 candidate、不加载 candidate checkpoint、不比较 win rate，也不构成共享
技能有效性、安全认证或论文结果。

## 2. 固定方法与身份契约

- 后端固定为 strict JSBSim；每条 episode 从合法 `pre_merge` reset 连续运行，禁止 handoff reset、snapshot restore、future-state injection 和 history padding。
- run-in 与 post-handoff reference 均为 frozen fixed head-on specialist；P3 encoder、预测器、VPP、guidance、PID、攻击区语义和 66-D -> 3-D VPP contract 保持冻结。
- evaluation scenario 使用 `geometry_cell_id + scenario_seed` 组成的序列化 `pair_key`。只有这种身份未来才有资格参与 paired delta。
- train scenario 使用独立的 `train_stream_id + scenario_seed`，显式禁止携带 `pair_key` 或 `geometry_cell_id`，且永远不参与 paired delta。
- preflight 的 synthetic round-trip 和单元测试同时覆盖两种身份，防止再次把 evaluation serializer 用于 train path。

## 3. 冻结包线与门槛

manifest 共 48 个 neutral/pre-merge 场景：4 个预定义 family，每个 family 由 2 个
distance/speed package、3 个高度条件和 2 个镜像方向组成。每个 opponent 跑一次，共
144 条计划 strict-JSBSim record。所有 V3 物理签名与 seed 均已检查为不与 V1/V2、
B2--B6、`heldout240`、`dev30`、历史 `heldout60` 和 phase-v2 相交。

一个 family 只有在 **每个 opponent 分别**满足下列全部条件时才通过 runtime
feasibility gate：

| 条件 | 门槛 |
|---|---:|
| 完整 record | 12 / family / opponent |
| qualifying handoff episode | >= 8 |
| valid neutral/post-merge target step | >= 160 |
| 镜像方向 | 2 |
| 高度条件 | 3 |
| telemetry contract failure | 0 |

若有多个通过 family，按三对手中的最小 valid step、最小 qualifying episode、最小
height/mirror coverage，最后按预注册 family order 排序。任何 pooled score 都被禁止。
若所有 family 都不通过，结论是 `v3_no_cross_opponent_runtime_feasible_family`，不得
起草或授权后续 pilot。

每个 family 还冻结了互不相同的 future dev/heldout package reservation；即使 V3
通过，未来 pilot 也不能复用 V3 physical state 或在结果后挑选新的几何范围。

## 4. 当前实现与复核

- [V3 identity/gate contract](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/src/uav_vpp_guidance/evaluation/thesis_neutral_postmerge_v3_contract.py) 已实现 evaluation/train 互斥身份、48 场景结构验证与逐 opponent family gate。
- [manifest builder](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/scripts/build_thesis_neutral_postmerge_reentry_recovery_v3_feasibility_manifest.py) 已生成 [manifest48](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/config/experiment/manifests/thesis_neutral_postmerge_reentry_recovery_v3_runtime_feasibility48.yaml)，payload SHA-256 为 `cea3387ea62dc7c8e4327e1fda443028d4ffb81addf8d7db48b6aff6edacbf91`。
- [design config](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/config/experiment/thesis_neutral_postmerge_reentry_recovery_v3_runtime_feasibility.yaml) 和 [preflight](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/scripts/preflight_thesis_neutral_postmerge_reentry_recovery_v3_runtime_feasibility.py) 已锁定输入 hash、三对手资产、P3、run-in specialist、磁盘门槛与 fresh output root。
- preflight 已通过：48 scenarios、144 planned records、约 291 GB free disk；未创建 V3 output root，未实例化 JSBSim，未训练。
- [runtime runner](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/scripts/run_thesis_neutral_postmerge_reentry_recovery_v3_runtime_feasibility.py) 已实现未来授权时的 frozen-baseline JSBSim 路径：每条 record 绑定 manifest identity、拒绝 backend/telemetry contract 缺失、拒绝 training transition，并在统计前要求完整 144 条 record universe。
- `tests/test_thesis_neutral_postmerge_v3_runtime_feasibility.py` 连同 V2 pairing/runtime 回归当前共 `11 passed`。测试覆盖 manifest 不相交、train/evaluation identity 隔离、所有三对手独立 family gate、future authorized overlay hash/merge/scope、runtime serializer 与未授权 `--execute` 拒绝。

## 5. 仍然禁止的动作与下一授权条件

当前 design config 没有 authorization overlay，因此 `--execute` 明确拒绝。future
runner 已具备只跑 frozen fixed-head-on baseline 的执行路径，但没有权限自动变成
pilot。实现路径已完成静态复核：future overlay 必须哈希绑定 canonical design、clean
implementation SHA 与白名单代码文件，且不能改写任何冻结 method field；逐 episode
输出将保存 V3 source、engine provenance、完整 manifest identity 与现有 engine 的
66-D/3-D VPP telemetry。现在才可申请独立的一次性 V3 runtime-feasibility execution
authorization。

只有 V3 的实际 144 条预检在三个对手下通过至少一个 family，才允许起草新的、仍需
单独授权的 V3 single-skill pilot preregistration。该结果绝不自动解锁四共享技能、
combat finetune、高层 PPO、P5/P6/P7 或 formal heldout。
