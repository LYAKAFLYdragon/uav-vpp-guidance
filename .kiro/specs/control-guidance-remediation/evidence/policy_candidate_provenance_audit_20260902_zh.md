# 两份策略候选资产的训练来源与运行溯源补充审计

## 审计结论

本审计只读核对 `CAN-20260705` 中已经找到的两份策略文件。能够确认文件字节、大小、观测维度、动作维度、训练步数、随机种子、配置快照和 CAN 归档完整性，但无法从现有材料恢复原始训练机器上的完整训练来源。因此，两份文件仍属于候选资产，不能恢复为历史冻结资产，也不能授权 strict-JSBSim 正式运行。

审计记录：[policy_candidate_provenance_audit_20260902.json](./policy_candidate_provenance_audit_20260902.json)

## 可核实信息

| 候选 | 文件大小 | SHA-256 | 观测/动作维度 | 训练步数 | 配置后端 | 当前状态 |
|---|---:|---|---:|---:|---|---|
| 无预测 VPP 策略 `last.pt` | 250138 字节 | `82cc54df...ef2cb` | 16 / 3 | 200000 | JSBSim | 训练溯源不完整候选 |
| 端到端策略 `best.pt` | 246810 字节 | `d3882833...5b8bc` | 16 / 3 | 200000 | simple | 训练溯源不完整候选，不能作为 strict-JSBSim 资产 |

两份文件均有对应的 `config_snapshot.yaml`。模型 payload 还记录了 `seed=0` 和 `total_updates=98`。这些信息可以作为模型内容和配置的辅助证明，但不能替代训练运行清单。

## 尚未补齐的来源

现有材料仍缺少：

- 原始训练 `run_manifest.json`；
- 训练时的完整 resolved configuration；
- 实际训练命令；
- 训练时 Git 提交、分支和 dirty 状态；
- 独立审核批准记录；
- 与该 checkpoint 一一对应的 strict-JSBSim 运行清单、后端状态、回退状态和 reset 审计。

`CAN-20260705` 的提交 `9a9f9bf6...` 只能证明归档和分发版本，不能证明两份模型的训练提交。`uav-vpp-guidance-ind-headon` 等目录的 `outputs` 是指向 CAN 资产的 Junction，也不能作为独立训练来源。

## 对历史资产的影响

本次补充审计没有复制、移动、重命名、加载或覆盖任何 checkpoint，也没有创建新的 superseding asset manifest。历史 P0 状态保持不变：

- predictor：已有同名文件，但 SHA 与要求值不一致，仍拒绝接收；
- prediction-VPP LSTM policy：实际文件仍缺失；
- no-prediction 和 end-to-end：字节及部分配置可核验，但仍未达到正式资产接收条件；
- strict-JSBSim smoke：继续不授权。

因此，本次工作完成的是“现有可获得 provenance 的结构化补齐”，不是“完整训练来源恢复”。
