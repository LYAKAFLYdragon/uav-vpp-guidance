# P2-B4 Raw-SI Phase Preflight 完成审计

**Source ID：** `THESIS-GLOBAL-ADVANTAGE-V1-P2-RAW-SI-PHASE-PREFLIGHT-V1`
**最终判定：** `raw_si_phase_contract_passed`
**执行状态：** 已一次性完成，授权已关闭。

## 合同通过

`90/90` record 均通过 strict JSBSim structural、scenario application receipt、raw-SI phase replay、normalized-policy diagnostic 与 step-0 pre-merge semantics。raw phase input 位于 `3999.816--4000.183 m`，policy diagnostic 位于 `0.799963--0.800037`，精确暴露并隔离了 B3 的单位错配。

| Opponent | pre_merge | post_merge | re_entry | first-pass episodes |
|---|---:|---:|---:|---:|
| expert | 5,390 | 1,509 | 625 | 14/30 |
| end_to_end | 6,029 | 963 | 577 | 8/30 |
| independent PPO/VPP | 5,634 | 1,254 | 696 | 11/30 |

## 不能过度解读的覆盖

本轮证明 phase tracking 可测量，不证明五态势每个微单元都被充分覆盖。重要 re-entry readout：Advantage 为 `5/7/7` steps（expert/end-to-end/independent）；Crossing-entry 为 `141/5/145`；Disadvantage 为 `13/48/0`；Head-on 为 `454/517/532`；Neutral 为 `12/0/12`。因此最关键的可证伪缺口仍是 `disadvantage -> post_merge/re_entry` 在 independent PPO/VPP 下未自然出现。

这不是对手强度排名、win-rate 结论、新技能有效性或飞行安全认证。它只允许进入 P2-B5 phase-feasible sampler 的设计线：连续 run-in 后 handoff，验证能否在三个 opponent 下为特定 motif 产生真实、连续、可回放的 raw-SI 样本；四共享技能、高层 PPO 和 formal held-out 仍锁定。

## 溯源

- Gate：`E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/global_advantage_v1_p2_b4_raw_si_phase/raw_si_phase_gate.json`，SHA-256 `9709639305e3db49c993f803eaf3640c34533c3d1ecd356322e243bc35a15302`。
- Raw run manifest：同一输出根的 `run_manifest.json`，SHA-256 `4f24e61e2dcf81d31113f6a32edf47df175dda034f169eebc0612d6870ac3dd5`。
