# V11 Relative-Frame Semantic VPP Transfer Probe 结果与 Go/No-Go 决定

**Source ID：** `THESIS-FIVE-STATE-REENTRY-RELATIVE-FRAME-SEMANTIC-V11`
**运行 SHA：** `4bc6919499defd68e2f5522a2ebdfc5661773ebb`
**结果根目录：** `E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/five_state_reentry_v11_relative_frame_semantic`
**最终判定：** `no_semantic_transfer_candidate_under_preregistered_gate`

## 1. 完整性与独立复核

- 完整预注册矩阵已运行：`4 methods x 3 opponents x 12 scenes = 144/144` strict-JSBSim records。
- `run_manifest.json` 记录训练与 reward recording 均为 `prohibited`；无 policy、predictor、guidance、PID、奖励或场景修改。
- 三对手、同一物理场景的 handoff hash 全部一致；每种方法的 run-in 由同一 frozen fixed-head-on specialist 完成。
- 独立 verifier 已从 144 个 episode ledger 重算 `semantic_transfer_gate.json`，结果为 `reported_gate_matches_recomputed=true`、`passes=false`。
- 未生成 `failure.json`；所有方法相对 reference 的 ego crash/OOB delta 均为 `0.0`，semantic window 内 saturation step 为 `0`。

## 2. 三层传递结果

下表是成对 qualified episode 的均值。VPP/guidance 的数值证明语义 compiler 未被 clipping、fallback 或后端吞掉；range-rate 与 intent-loss 才是实用传递门。

| Method | Opponent | Qualified pairs | VPP delta (m) | Guidance delta | Range-rate delta (m/s) | Intent-loss delta | 判定 |
|---|---|---:|---:|---:|---:|---:|---|
| LOS only | expert | 7 | 626.0 | 0.247 | +7.01 | +0.00929 | 距离率通过；综合 intent 恶化 |
| LOS + turn | expert | 7 | 946.4 | 0.486 | +8.46 | +0.01173 | 距离率通过；综合 intent 恶化 |
| Full profile | expert | 7 | 986.7 | 0.485 | +8.46 | +0.01173 | 距离率通过；综合 intent 恶化 |
| LOS only | end-to-end | 3 | 897.9 | 0.024 | +0.35 | +0.00009 | 合格 cohort 小于 4，且几何门失败 |
| LOS + turn | end-to-end | 3 | 503.2 | 0.043 | +0.12 | -0.00011 | 合格 cohort 小于 4，且几何门失败 |
| Full profile | end-to-end | 3 | 551.0 | 0.043 | +0.18 | -0.00009 | 合格 cohort 小于 4，且几何门失败 |
| LOS only | independent PPO/VPP | 7 | 655.1 | 0.227 | -1.19 | -0.00062 | 距离率和 intent 门失败 |
| LOS + turn | independent PPO/VPP | 7 | 872.6 | 0.296 | +2.00 | +0.00078 | 距离率和 intent 门失败 |
| Full profile | independent PPO/VPP | 7 | 926.4 | 0.294 | +2.00 | +0.00078 | 距离率和 intent 门失败 |

预注册实用门要求每个对手独立具备至少 4 个 pair、VPP delta `>=100 m`、guidance delta `>=0.02`、range-rate delta `>=+5 m/s`、intent-loss delta `<=-0.01`、且安全不退化；固定 `range_extension_full` 还必须在至少两个对手通过。没有任何方法满足两个对手的完整门槛，故 `semantic_transfer_candidates=[]`。

## 3. 归因边界

V11 否定了 V10 中“action 没有进入 VPP 或 guidance”的解释：此处三种 semantic action 都形成数百米 VPP 位移，并改变了 guidance command。它也没有显示共同飞控执行链失稳：没有新增 ego crash/OOB、没有 saturation，run-in terminal mix 对同一 opponent 在四种方法下完全一致。

但 V11 **不**证明任何五态势技能已经可训练：

- 在 expert 下，分离的局部 range-rate 变化可以达到门槛，但同时使完整 profile 的多目标 intent-loss 变差；不能把“范围率变大”写成 profile 已实现。
- 在 independent PPO/VPP 下，VPP/guidance 响应存在，但只有约 `+2 m/s` 的 range-rate 改变，未达到实用量级。
- 在 end-to-end 下，本 manifest 只有 `3/12` continuous re-entry handoff；这不是 compiler 失败，也不足以支持该 opponent 的 transfer 结论。
- 本记录没有逐分量的 post-window AA/ATA/energy trajectory，故不能把 expert 的 intent-loss 恶化唯一归咎于某一个角度或能量项。

## 4. 正式决定

**No-go：** 不得以 V11 为基础启动 terminal-reward rerun、单技能/四技能训练、combat finetune、P5 PPO、P6 消融或 P7 heldout。不得在相同 Source ID 下改变 compiler scale、阈值、window length、场景或 profile 后重跑。

**保留的正向机制发现：** 相对坐标 semantic compiler 可以在冻结 JSBSim VPP/guidance 链中可靠地产生可观测的 VPP 与 guidance 响应；问题已从“动作是否可达”收窄到“现有 VPP 单点接口能否同时实现 profile 的多目标几何收益”。

**后续研发边界：** 若继续，应另起 Source ID，先设计非学习的 `intent-objective controllability` 诊断，以逐分量测量 AA、ATA、range、range-rate、specific-energy、altitude 对相对坐标动作的局部响应。该设计不能只是扩大 action scale 或延长 20-step window；只有证明存在安全且多目标一致的可行方向，才可重新讨论训练。
