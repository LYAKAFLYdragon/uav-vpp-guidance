# V8 Physical Re-entry Phase-Contract Probe 结果

## 结论

`THESIS-FIVE-STATE-REENTRY-PHASE-CONTRACT-PROBE-V8` 通过。该结果建立了一个新的、
非学习的连续 JSBSim 数据合同：在 V8 独立 12 场景中，三类对手下均可从归一化 16-D
观测严格换算到物理 range/range-rate，锁存 raw-SI first pass，并在其后的物理再接近
阶段记录有效 66-D -> 3-D VPP 对。

| 对手 | Records | Qualifying episodes | Re-entry 66-D steps | Mirrors | Signatures | Gate |
|---|---:|---:|---:|---:|---:|---|
| expert | 12 | 5 | 330 | 2 | 5 | PASS |
| end-to-end | 12 | 4 | 286 | 2 | 4 | PASS |
| independent PPO/VPP | 12 | 6 | 312 | 2 | 6 | PASS |

独立 verifier 从 36 条 ledger 重放了所有 phase 标签，并复核
`raw_range = normalized_range x 5000`、`raw_rate = normalized_rate x 200`，结果为 PASS。
本轮不记录 training reward、不训练策略、不修改 VPP/guidance/PID，也不构成性能优越性结果。

## 边界与决定

V8 只说明训练 re-entry 行为所需的物理、连续、可观测输入合同已经成立；它不证明新技能
已经有效，也不证明终端奖励会提高胜率。它解除的是“可以设计奖励 pilot”的数据合同门槛，
不是训练授权。

下一步允许创建 `THESIS-REENTRY-TERMINAL-REWARD-PILOT-V1` 的预注册配置和实现 preflight。
该 pilot 必须使用 V8 的 `RawSIReentryPhaseTracker`，并采用全新且与 V1--V8 不相交的
train/dev/heldout 场景。只有 preflight 验证 phase、66-D、VPP、backend 和输出 provenance
后，才可单独授权训练。

## 可复现性

- Code SHA: `eaf1407eb606fbe4d699e968212e72ac0f0944b6`
- Output: `E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/five_state_reentry_v8_phase_contract_probe`
- Gate: `phase_contract_gate.json`
- Independent verification: `independent_verification.json`
