# P0 文献记录与资产边界核验

- **阶段状态：`physical_unavailable`。** 论文、实验计划、对手能力卡和权威索引被核验为文献记录，不是 checkpoint 的字节级资产收据。
- **正面事实：** 第六章记录 PPO/VPP/JSBSim 定义和 19 维正式评价架构；实验计划的 A0 行记录无预测 checkpoint 路径并标注本地资产缺失；对手能力卡识别 end-to-end SHA `d388…5b8bc`，但明确训练配方没有重建。
- **排除事实：** 论文权威索引列出的高层、迎头和交叉 checkpoint 是不同资产，不是 `vpp_ppo_lstm_frozen/checkpoints/best.pt`。所有文献均不能提供 predictor 的缺失 SHA `a7c…4491` 或替代缺失 VPP 文件。
- **决定：** predictor 保持 `rejected_hash_mismatch`；prediction-VPP 保持 `physical_unavailable`；no-prediction/end-to-end 保持 `unregistered_candidate_not_adopted`。禁止创建 superseding asset manifest，禁止 strict-JSBSim smoke。
- **合规路线：** 继续恢复原训练机资产及 provenance；若最终不可恢复，使用新 Source ID 重新训练、重新登记，并明确不能等同于历史冻结资产。
