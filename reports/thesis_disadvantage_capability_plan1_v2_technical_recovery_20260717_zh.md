# Disadvantage Capability Plan 1 V2 技术恢复记录

首次 V2 执行在 `expert` 子运行后停止，`execution_receipt.json` 返回 `1`。根因是 `run_jsbsim_hrl_comparison.py` 的 checkpoint preflight 已识别 `temporal_schedule_specialist` 无单一 checkpoint，但实际执行循环遗漏该类型，随后尝试加载 `method.checkpoint=None`。

该子运行仅完成 canonical PPO、fixed head-on、fixed crossing 的 36 条记录；temporal schedule 方法未进入 episode loop，因此 144 条预注册矩阵不完整，无法形成任何 route gate、性能、失败或物理能力结论。该目录保留在 `E:/uav-vpp-guidance-disadvantage-capability-v1-results/plan1_v2_route_upper_bound`，但明确排除出分析。

修复将 preflight 与执行统一到 `_uses_single_policy_checkpoint()`，并加入 temporal schedule 回归测试。`19 passed`、编译与 diff 检查通过后，仅允许一次 R1 技术恢复：使用相同冻结 V2 config/manifest/checkpoint/对手、全新的空输出根、完整三对手矩阵；不得改任何科学变量或读取不完整记录决定后续动作。
