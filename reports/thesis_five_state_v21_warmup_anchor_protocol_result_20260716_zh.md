# V21 Warm-up Anchor Protocol 结果

`THESIS-FIVE-STATE-V21-WARMUP-ANCHOR-PROTOCOL` 完成 72 条 strict-JSBSim 记录：24 scenes ×
3 opponents。三对手各 24 条均完成，但 release 均为 0，故预注册 gate 未通过。

| Opponent | Records | Anchor releases | Post-release steps | Gate |
|---|---:|---:|---:|---|
| expert | 24 | 0 | 0 | FAIL |
| end-to-end | 24 | 0 | 0 | FAIL |
| independent PPO/VPP | 24 | 0 | 0 | FAIL |

## 因果结论

V21 强制 `current_target` warm-up 后，环境不会进入 `predicted_target` 的 predictor 调用分支，
prediction history 因而不更新。release 条件又要求 valid prediction，构成接口级循环依赖。
这不是训练不足、对手差异、VPP action、guidance/PID 或策略性能的证据。

不得重跑 V21、放宽 release 门或把当前结果用于论文性能声明。

## V22 设计边界

唯一合理的后续接口是 observer-only predictor update：即使 active anchor 为 `current_target`，
仍以当前真实 target state 更新 predictor history，但其输出不得影响 VPP、guidance、PID、reward 或
policy observation，直至一次性 release。该 observer receipt 必须显式记录 predicted position 是否
被控制链消费；若任何 warm-up prediction 进入控制链，V22 即失败关闭。
