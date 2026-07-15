# Terminal Reward Pilot V1 Runtime Preflight 结果

## 决定

THESIS-REENTRY-TERMINAL-REWARD-PILOT-V1-PREFLIGHT 通过。它在全新 dev12 包线
上用固定 head-on specialist 完成 12 scenarios x 3 opponents = 36 条 strict-JSBSim
连续记录，并由独立 verifier 重放 V8 phase 与 terminal-reward annotation。

| 对手 | Records | Qualifying re-entry episodes | Re-entry 66-D steps | Mirrors | Signatures | Terminal reward timing | Gate |
|---|---:|---:|---:|---:|---:|---|---|
| expert | 12 | 8 | 485 | 2 | 8 | final step only | PASS |
| end-to-end | 12 | 4 | 286 | 2 | 4 | final step only | PASS |
| independent PPO/VPP | 12 | 8 | 443 | 2 | 8 | final step only | PASS |

## 已验证内容

- 真实 JSBSim ledger 能以 raw physical first-pass-aware 规则重放 phase。
- raw_range = normalized_range x 5000 与 raw_rate = normalized_rate x 200 的合同在 replay 中成立。
- target_crash_or_out_of_bounds 与 target_killed 映射为胜利，未受惩罚。
- ego crash/OOB 的终端项为 -2.0；draw 为 -0.25；非末步终端项严格为零。
- 训练、环境 reward 写入、VPP/guidance/PID 修改、高层 routing 和 heldout24 使用均未发生。

## 解释边界

这不是 candidate 的训练结果、性能改进、安全认证或论文主结论。它只证明：如果下一阶段
实现 phase-aware PPO candidate runtime，预注册的输入、phase、66-D、终端 reward 和 dev-only
选择协议可以在三对手的真实连续环境中共同成立。

下一步允许实现并测试该 runtime，再单独冻结 training authorization。heldout24 仍不能
用于任何 checkpoint 选择或训练调试。

## 可复现性

- Code SHA: 5880f40e83bc5af1f263f8522edbbd97634fd569
- Output: E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/thesis_reentry_terminal_reward_pilot_v1_preflight
- Gate: preflight_gate.json
- Independent verification: independent_verification.json
