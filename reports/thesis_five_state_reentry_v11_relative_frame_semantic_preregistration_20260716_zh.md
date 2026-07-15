# V11 Relative-Frame Semantic VPP Transfer Probe 预注册

**Source ID：** `THESIS-FIVE-STATE-REENTRY-RELATIVE-FRAME-SEMANTIC-V11`
**状态：** design-only；尚未授权执行
**性质：** 非学习、非奖励、非论文主结果的机制验证

## 1. 问题与目的

V8 已建立连续 re-entry 的 raw-SI phase 与 66-D 输入合同；Terminal-Reward V1 在开发集上安全但无实用改善；V10 进一步显示，当前世界坐标系的固定 3-D additive bias 虽能进入 VPP 层，却没有在 20 个高层步内带来足够的 range-extension 几何改善。因此，不能继续通过增加训练步数、增大奖励或改变奖励系数来解释失败。

V11 只回答一个更基础的问题：**若将同一冻结专家动作改用相对几何语义表达，是否能在不改变策略、预测器、制导律、PID 或奖励的条件下，形成可审计的 VPP -> guidance -> range-rate 局部传递链？**

## 2. 冻结方法

固定 `frozen_fixed_head_on` specialist、P3 encoder、trajectory-prediction VPP 接口、guidance/PID、strict JSBSim 与三对手。每条记录只 reset 一次；使用 raw-SI first-pass 后的连续 re-entry handoff；禁止 snapshot restore、future-state injection、history padding、backend/prediction/checkpoint fallback、训练和 reward recording。

环境仍接收既有 world-NEU cartesian 3-D VPP action；新 compiler 仅在其前端工作。对 `range_extension` profile 的物理 targets，计算：

\[
\Delta p_{VPP}=-u_r(1200)e_{LOS}+u_t(900)e_{\perp,target}+u_e(400)e_z.
\]

其中：

- `u_r` 是 profile 的 range=4000 m、range-rate=120 m/s 相对当前状态的加权归一化误差。正 `u_r` 明确表示“请求更大间隔”，故 VPP 沿 LOS 向我机侧移动。
- `u_t` 根据带符号 ATA 与 profile 的 ATA=10 deg 目标生成；它沿目标速度坐标系的侧向单位向量作用。
- `u_e` 根据 profile 的 specific-energy=200 m、altitude=100 m 目标生成；它沿 NEU 向上轴作用。

世界向量随后精确投影回冻结的 `[-1500,1500] x [-800,800] x [-500,500]` normalized action interface；每步记录请求值、clip、实际 VPP world offset、归一化 guidance command、实际过载与范围率。

## 3. 场景与方法矩阵

全新 `manifest12` 使用 6900 m、294/356 m/s 的未见距离-速度包，`seed=191100`，与 V1--V10、Dev30、Heldout60、Taxonomy30 等既有 manifest 均记录为物理不相交。三对手单独报告：`expert`、`end_to_end`、`independent_ppo_vpp`。不合并任何统计量。

| Method | 开启的 semantic axes | 意图 |
|---|---|---|
| `frozen_fixed_head_on` | 无 | 成对参考 |
| `range_extension_los` | range | 仅验证 LOS separation 语义 |
| `range_extension_los_turn` | range + signed-turn | 验证转向补偿是否必要 |
| `range_extension_full` | range + signed-turn + vertical-energy | 固定完整 profile 对应实现 |

总记录数固定为 `4 x 3 x 12 = 144`。这是机制 probe，不是选择最佳动作、不是新增策略，更不是训练或性能评估。

## 4. 预注册门槛与 stop rule

每种非参考方法、每个对手独立满足下列条件，才记为该对手的 semantic-transfer pass：

1. 至少 4 条相同 handoff hash 的 qualified pair，且均为完整 20-step window。
2. 语义 compiler 实际注入的均值 delta norm `>=100 m`，环境回执的 VPP world-offset 相对参考差 `>=100 m`。
3. 归一化 guidance-command 差异 norm `>=0.02`。
4. 20-step 后 range-rate 相对参考增加 `>=5 m/s`，且 `intent_loss_auc20` 相对参考 `<= -0.01`。
5. ego crash/OOB 比例不高于参考 `+0.05`。

仅当固定的 `range_extension_full` 在至少两个对手分别通过时，才允许**另起 Source ID**设计任何学习 pilot。任何 contract failure、backend fallback、非有限 telemetry、异常 reset 或安全门失败均使本轮冻结为负证据；不得在 V11 下调 scale、延长 horizon、改变场景、改变奖励或 rerun。

## 5. 可复现入口

- Manifest builder：`scripts/build_thesis_five_state_reentry_v11_manifest.py`
- Design config：`config/experiment/thesis_five_state_reentry_v11_relative_frame_semantic_design.yaml`
- Runner：`scripts/run_thesis_five_state_reentry_v11_relative_frame_semantic.py`
- Independent verifier：`scripts/verify_thesis_five_state_reentry_v11_relative_frame_semantic.py`
- 输出根：`E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/five_state_reentry_v11_relative_frame_semantic`

本预注册不改写 V8、Terminal-Reward V1、V9 或 V10 的结论，也不将 `tactical_basis` 历史措辞恢复为学习到的理论变换。
