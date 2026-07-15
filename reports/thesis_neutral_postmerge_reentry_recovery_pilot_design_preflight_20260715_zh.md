# Neutral Post-Merge Reentry-Recovery Pilot V1 设计预检完成记录

**Source ID：** `THESIS-NEUTRAL-POSTMERGE-REENTRY-RECOVERY-PILOT-V1`
**状态：** `preregistered_design_only_not_authorised`
**日期：** 2026-07-15

## 结论

设计预检通过，但本结论只说明未来可申请一次独立授权；它不表示已经训练、已经评估、候选技能有效，或已经证明五态势系统的任何性能结论。

本线只检验一个可证伪问题：在 B6 已建立输入数据契约的前提下，冻结的 `reentry_recovery` 候选技能使用固定 `reentry_preparation` profile，能否在 `neutral -> post_merge` 窗口优于两个冻结既有 specialist。路由关闭，P3 encoder、预测器、VPP、guidance 与 PID 均保持冻结。

## 已冻结设计

- 新 manifest：`dev12` 与一次性 `heldout24`，均从合法 `pre_merge` reset 连续 run-in，不允许 handoff reset、snapshot restore 或 future-state injection。
- 初始态势固定为 `neutral`，目标窗口固定为 `neutral -> post_merge`；候选动作为 `reentry_recovery + reentry_preparation`。
- 三个 opponent 独立报告：`expert`、`end_to_end`、`independent_ppo_vpp`；禁止 pooled gate。
- 对照只有候选、fixed head-on 和 fixed crossing。候选 checkpoint 仍为 `null`，因此不存在可被误用的训练结果。
- 主指标预注册为前 20 个有效 target step 的 `normalized_reentry_preparation_intent_loss_auc20`；win rate 仅为次要指标。
- 安全 stop rule 固定：任一 non-finite 66-D/action、JSBSim/checkpoint fallback、异常 reset、padding 或 future-state injection 均立即冻结为负证据；dev 中 ego crash/OOB 高于 head-on 超过 0.05 亦停止。

## 独立性与预检

- 两份 manifest 对 B2--B6、`heldout240`、`dev30`、旧 `heldout60` 与 phase-v2 的九份冻结来源逐一验证：物理初态签名交集和 scenario-seed 交集均为零。
- 生成器可再生：`dev12` payload SHA-256 为 `11d05b2adceb9b095123844c735639717182cd88045a533e7dc5d2c90dbb919a`；`heldout24` 为 `1226c21d6ab03365078e58df063894a116b3c9910139146149e64650ffbf6627`。
- 预检已验证 P3 encoder、fixed head-on/fixed crossing checkpoint、end-to-end opponent、independent PPO/VPP opponent、runtime registry 与 artifact schema 的 SHA-256。
- `reentry_preparation` 的冻结 target/weight 为 `(AA=30 deg, ATA=130 deg, range=2200 m, range-rate=-30 m/s, specific-energy delta=100 m, altitude delta=0 m)` / `(0.8, 0.8, 1.0, 0.8, 0.7, 0.4)`；`neutral/post_merge` validity mask 允许 `reentry_recovery` 使用该 profile。
- 预检结果：`execution_permitted=false`、`training_permitted=false`、结果目录不存在、可用空间 `291.183 GB`，满足 `120 GB` 容量门槛。

## 验证

以下测试均通过，未运行 JSBSim episode，也未写入任何 result root：

```text
python -m pytest tests/test_thesis_global_advantage_p2_b6_reachability_design.py tests/test_thesis_global_advantage_p2_b6_reachability.py tests/test_thesis_neutral_postmerge_reentry_recovery_pilot_v1.py -q

10 passed
```

## 下一步边界

唯一允许的后续动作是：在这组文件提交为新的 clean SHA 后，对 manifest、ATA/AA、raw-SI phase、资产 hash、输出容量与 stop rule 做独立复核；只有获得单独的一次性执行授权，才可创建新的 output root 并启动 pilot。不得重跑或修改 B6，不得直接训练四个共享技能、高层 PPO、combat finetune 或 formal held-out。
