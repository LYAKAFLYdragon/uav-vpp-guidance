# P3 受控轨迹采集与 Stop Rule 记录

**日期：** 2026-07-13  
**lane：** `noncanonical_thesis_extension` / `thesis_five_state_shared_intent_v1`  
**状态：** `stopped_before_encoder_training_missing_phase_coverage`

## 已执行内容

- 使用官方 JSBSim `v1.1.6`，commit `b477f6312bee2fd3af4c4e5ba1a3e732ed2b99d4`，在 F-16 双机环境中完成 90 个 train-only rollout。
- 采样为 `5 initial states x 3 height conditions x 2 mirrors x 3 opponents`；ego 仅使用冻结 rule-based VPP behavior policy 采集数据，未训练或更新任何低层 skill、高层 PPO、predictor、guidance 或 PID。
- 三个 opponent 均实际加载并运行：`expert_rule_based`、`end_to_end_neural`、`independent_ppo_vpp`。
- 90/90 rollout 的 backend 均为 JSBSim，均以受控的 80-step timeout 结束；产生 6,030 个 history/horizon 可用窗口。只保存压缩 16-D geometry sequences、episode index 与 train-only normalization；未保留全量原始飞控遥测。
- `dev30` 尚未启动，因为 train phase-coverage gate 在 dev 前失败；`heldout60` 未读取、未运行、未参与任何拟合或选择。

## Gate 结果

| 初始态势 | pre-merge | post-merge | re-entry | 结论 |
|---|---:|---:|---:|---|
| Advantage | 402/402/402 | 0/0/0 | 0/0/0 | 不满足 |
| Head-on | 29/30/29 | 326/326/326 | 47/46/47 | 满足 |
| Disadvantage | 402/401/402 | 0/1/0 | 0/0/0 | 不满足 |
| Neutral | 402/402/402 | 0/0/0 | 0/0/0 | 不满足 |
| Crossing-entry | 98/255/98 | 290/141/283 | 14/6/21 | 满足 |

各单元格按 `expert_rule_based / end_to_end_neural / independent_ppo_vpp` 的顺序列出。完整 machine-readable coverage 与输入哈希见：

- `E:/uav-vpp-guidance-thesis-five-state-v1-results/p3_temporal_encoder_v1/collection_report.json`
- `E:/uav-vpp-guidance-thesis-five-state-v1-results/p3_temporal_encoder_v1/p3_gate_decision.json`
- `E:/uav-vpp-guidance-thesis-five-state-v1-results/p3_temporal_encoder_v1/input_hashes.json`

## 决策与启示

P3 的预注册要求是按 initial-state、phase 和 opponent 平衡采样。现有 P2 连续分布从五类初始几何出发，但它并不保证 Advantage、Disadvantage、Neutral 在 80 个高层步内穿越 merge 并形成 re-entry。因此继续用这批数据训练 encoder，会将“phase-aware representation”建立在只覆盖 head-on/crossing 的样本上，违反 P3 验收契约。

按 stop rule，本轮不训练 temporal encoder，后续也不得以缺相位数据选择 checkpoint。P4 保持关闭。若要重新开启 P3，必须先建立并独立冻结 phase-conditional train sampler（pre-merge、post-merge、re-entry 的物理可达初始化或经审计的连续 transition rollout），再从干净输出根完整重采集；不得通过更长训练、重复同一采样或放松 coverage gate 修复。
