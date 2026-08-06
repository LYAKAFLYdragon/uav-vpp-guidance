# P0 权威资产采集核验总结

- **阶段状态：`physical_unavailable`。** 此次采集没有 `.pt` 文件，也没有复制、移动、注册或修改任何 checkpoint；strict-JSBSim 未获授权、未执行。
- **来源性质：** `E:\uav-vpp-guidance-ind-headon\outputs` 是指向 `CAN-20260705/assets/outputs` 的 Junction；源根没有 Git 元数据。因此它只能证明 CAN 分发候选，不是原训练机的独立训练来源。
- **predictor：** 865,909 bytes、`96bc8e5e…00495931`，严格结构加载通过 16/128/2/0.1，但与冻结 `a7c00fd2…d74491` 不同，明确拒绝。
- **prediction-VPP：** 登记路径缺失，仍为 `physical_unavailable`。
- **policy 候选：** no-prediction `82cc54df…ef2cb`（250,138 bytes）和 end-to-end `d3882833…5b8bc`（246,810 bytes）都为 16 维观测、3 维动作、200,000 steps；但只有 25 个 `config_snapshot.yaml`，没有独立训练 run_manifest/resolved config 或审核链，仍只能作为候选。
- **结论：** 不创建 superseding asset manifest。下一门只接受原训练机导出的 prediction-VPP、精确 predictor bytes、及两个 policy 的独立训练 provenance 与审核记录。
