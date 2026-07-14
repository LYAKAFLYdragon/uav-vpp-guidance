# Defensive-Extension / Range-Extension Pilot V1 执行结论

## 结论

预注册判定为 **`safety_or_contract_no_go`**。现有证据不支持‘现有技能库明确缺少 defensive-extension’。

候选技能完成固定 50,000-step 训练，所有 Dev checkpoint safety gate 均通过，并按 minimax Dev 规则自动选择 20k checkpoint。但在一次性 Heldout24 中，independent PPO/VPP 对手下出现 `2/24` ego crash/OOB，而 fixed head-on 与 fixed crossing 均为 `0/24`，安全差值 `+0.0833` 超过预注册 `+0.05` 门。

此外，independent PPO/VPP 下候选与两种 baseline 的 paired qualifying episode 均只有 `6`，低于 claim-ready 门槛 `8`。原始 decision 实现只检查 candidate qualifying 数量；该分析错误已只读纠正，原文件未覆盖，原 SHA-256 为 `807bd01ff126b99574d4398270d7be12331afb404b778188fe71bbdafedf2f09`。

## 分对手结果

| Opponent | Candidate qualifying | Paired H/C | Delta vs head-on | Delta vs best existing | Crash/OOB delta | Contract | Safety |
|---|---:|---:|---:|---:|---:|---|---|
| expert | 13 | 9/11 | -0.0081 | -0.0143 | +0.0417 | pass | pass |
| end_to_end | 14 | 11/12 | -0.0108 | -0.0108 | +0.0000 | pass | pass |
| independent_ppo_vpp | 8 | 6/6 | -0.0082 | -0.0305 | +0.0833 | fail | fail |

## 科学解释

- 三个对手下 candidate 相对 head-on 的 paired mean delta 分别为约 `-0.0081/-0.0108/-0.0082`，均通过 `+0.02` 非劣门，但没有一个达到 `-0.05` practical-improvement 门。
- 相对 best existing specialist，仅 independent PPO/VPP 达到 `-0.02` practical-improvement；expert 与 end-to-end 均未达到。
- Fixed crossing 在 expert 与 independent PPO/VPP 中本身就是更好的 existing specialist，说明现有技能库已经覆盖一部分 extension-like geometry，不能把问题简单归因为‘缺一个新技能’。
- 候选在高速度、positive-mirror heldout 条件中发生早期 crash/OOB，而两种 frozen baseline 在相同场景均存活，说明当前 range-extension policy 的几何改善不足以抵消执行安全风险。

## Stop Rule

结果冻结为负证据。不得在同一 Source ID 下增加训练步数、调整 reward、删除失败场景、更换 seed、放宽 safety/paired coverage 门或重跑 Heldout24。四共享技能、combat finetune 和 P5 继续锁定。

当前最准确的回答是：**输入输出契约已经建立，但本 pilot 没有证明新 defensive-extension 技能优于现有技能库；当前候选还引入了 opponent-dependent 的安全退化。**
