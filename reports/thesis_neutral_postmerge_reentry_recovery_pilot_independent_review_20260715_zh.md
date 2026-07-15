# Neutral Post-Merge Pilot V1 独立设计复核

**审计对象：** `bc356bc5165906618c83191252cd9142e52cb4a2`
**Source ID：** `THESIS-NEUTRAL-POSTMERGE-REENTRY-RECOVERY-PILOT-V1`
**结论：** `PASS - design ready for separate authorization only`

## 复核范围

本复核仅审计设计是否可被安全授权，不执行 JSBSim、训练、baseline evaluation 或 held-out evaluation，不创建 output root，也不产生任何性能结论。

| 检查项 | 结果 | 依据 |
|---|---|---|
| 提交与工作区 | PASS | 审计起点为 clean `bc356bc`。 |
| 授权边界 | PASS | `execution_permitted=false`、`training_permitted=false`；所有扩展训练、调参、VPP/guidance/PID 变更、reset、future-state injection 与 padding 许可均为 false。 |
| 单一机制 | PASS | 仅 `reentry_recovery + reentry_preparation`，目标窗口仅为 `neutral -> post_merge`，routing 关闭。 |
| 输入与资产 | PASS | P3、fixed head-on/crossing、end-to-end opponent、independent PPO/VPP opponent、runtime registry 与 artifact schema SHA-256 均通过。 |
| 情景隔离 | PASS | `dev12` 与 `heldout24` 对 B2--B6、`heldout240`、`dev30`、旧 `heldout60`、phase-v2 的物理签名及 seed 均无交集。 |
| 生成可复现性 | PASS | 从 builder 重建至临时目录后，`dev12` payload SHA 为 `11d05b2adceb9b095123844c735639717182cd88045a533e7dc5d2c90dbb919a`，`heldout24` 为 `1226c21d6ab03365078e58df063894a116b3c9910139146149e64650ffbf6627`，与冻结文件一致。 |
| 物理与语义契约 | PASS | raw-SI phase 输入、`[ATA, AA]` 顺序、66-D 到 3-D VPP、10-frame real history、无 fallback/padding/reset 的要求均在 config 和 preflight 中锁定。 |
| 容量与输出 | PASS | output root 尚不存在；可用空间为 `291.183 GB`，高于 `120 GB` 门槛。 |
| 回归测试 | PASS | B6 相邻合同测试与新 pilot 测试合计 `10 passed`。 |

## 审稿式结论

该设计已经排除了三类常见的伪阳性来源：从 B6 复用轨迹或固定初态、在 dev 上泄漏 heldout 选择、以及将路由/冻结链条变化混入候选技能效应。它因此可以回答一个狭窄的、可证伪的技能库问题；它仍不能证明新技能有效、五态势系统优越，或对真实空战具有泛化能力。

## 后续授权条件

只有用户单独授权后，才允许创建一份一次性 execution authorization，其中必须：

- 以 `bc356bc` 为可追溯实现祖先，并重新写入所有实际代码与资产 SHA；
- 保持唯一 Source ID、固定 50,000 step、固定 dev12 checkpoint rule 与一次性 heldout24；
- 在创建结果目录前再次检查 clean worktree、空 output root、磁盘空间、raw-SI phase、ATA/AA 和 stop rule；
- 明确规定任一 safety/contract stop 或 heldout failure 后不调 reward、不加步数、不换场景、不重跑同一 Source ID。

本复核不构成上述执行授权。
