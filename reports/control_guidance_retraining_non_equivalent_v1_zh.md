# 独立新 Source ID 重训路线 v1

- **新 Source ID：** `UAV-VPP-GUIDANCE-RETRAIN-NON-EQUIVALENT-20260727-R01`。
- **边界：** 这是历史冻结资产不可恢复后的独立重训族，不字节等同于任何历史 checkpoint，不可直接替代历史路径、历史 registry key 或历史 strict-JSBSim 证据。
- **隔离：** 新训练输出固定在 `outputs/control_guidance_retraining/<Source ID>/`；旧 `outputs/trajectory_prediction/` 和三个历史 `outputs/experiments/...` 根目录均禁止写入。prediction-VPP 准备配置还会显式关闭并清空继承的历史 `warm_start` checkpoint 引用。`config/checkpoint_registry.yaml` 受 P0 freeze 保护，未修改。
- **当前门禁：** `preflight_blocked_missing_new_trajectory_corpus`。当前 worktree 的 `outputs/` 仅含 P0 证据输出，不含 predictor 所需的 episode CSV。因而未启动 predictor，也未启动会造成不完整四模型族的任何 PPO 训练。
- **已核验运行条件：** Python 环境可导入 JSBSim `1.2.3`、scikit-learn 和 CPU PyTorch `2.8.0+cpu`；CUDA 不可用。CPU 仅是新 Source 的可记录计算选择，不影响历史资产缺失判定。
- **恢复训练的前置输入：** 新建且冻结的 trajectory CSV 语料、数据 manifest（每个文件的 SHA-256/大小/来源）、采集配置和命令、随机种子、Git 信息及数据审核批准。随后先训练新的 LSTM predictor，再以该 predictor 训练 prediction-VPP；no-prediction 和 end-to-end 也必须作为同一新 Source 的独立资产记录。
- **登记条件：** 每个新模型取得独立 run manifest、resolved config、artifact contract、portable bundle、原始 SHA-256、大小、观测/动作兼容性和审核决定后，才可从新 registry 的 `not_trained` 状态登记。结果必须写为“新 Source 重训”，不得声称与旧 SHA 或历史冻结模型等同。
- **P0 保持：** 历史 `THESIS-CONTROL-GUIDANCE-REMEDIATION-V1` 仍为 `physical_unavailable`；本路线不授权使用历史资产进行 strict-JSBSim smoke。
