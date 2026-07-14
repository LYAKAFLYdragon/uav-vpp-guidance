# Global-Advantage P1 R2 Run-in 可重复性审计

**Source ID：** `THESIS-GLOBAL-ADVANTAGE-V1-P1-RUNIN-TELEMETRY-R2`
**模式：** 只读；不训练、不调参、不重跑 P1 R2。

## 结论

P1 R2 为正式 **NO-GO**：所有 90 个 opponent x scenario cell 的 reset base-state hash 一致，但没有一个 cell 的完整轨迹或 first-pass/terminal boundary state hash 一致。当前批量 run-in 协议不能为未来方法比较提供同一物理起点之后的因果配对基础。

这不是新技能效果结论，也不是对 JSBSim 飞行动力学的性能判断。它证明的是：在复用环境实例、改变 scenario 顺序的现有 runner 中，隐藏的运行时状态或未被 reset telemetry 会影响第一个或后续控制步。

| Opponent | Cells | Reset equal | Trajectory equal | Boundary equal | Terminal equal | Telemetry complete | Hard fallback absent |
|---|---:|---:|---:|---:|---:|---:|---:|
| end_to_end | 30 | 30 | 0 | 0 | 24 | 30 | 30 |
| expert | 30 | 30 | 0 | 0 | 27 | 30 | 30 |
| independent_ppo_vpp | 30 | 30 | 0 | 0 | 25 | 30 | 30 |

## 证据边界

- R2 完整写入并通过 SHA-256 验证的 raw episode artifact 为 `270` 个；不存在缺 episode、backend fallback、预测器运行时 fallback 或 telemetry 缺字段的情况。
- 前 9 个预测器 warmup step 被显式记录，未误报为运行时 fallback。
- R2 仅保存了 reset 时的 16-D base-state 摘要，没有持久化完整 reset observation vector、控制器内部状态和 JSBSim 完整初始 FDM state。因此不能把原因直接归咎于某一个模块。

## 允许的下一步

仅允许起草 P1 R3 fresh-environment-per-episode 复核：每条 scenario 在新建环境实例中独立运行，同时在 reset 前后保存完整 observation、reference action、opponent action、controller state 和完整 FDM state hash。R3 通过前不得启动 P2、任何技能训练、combat finetune、P5 或 formal held-out。

P1 R2 run manifest SHA-256: `a9bfa248cc0d2d9118b470eccac7efc3b669a4f7f00bed0f105534ce25a639ca`
P1 config SHA-256: `bbbb4d62775006a96eafa19403333dde83e5ee9db058560686ebb526bd182558`
