# P2-B1 三对手 Quantitative Capability Card Readout

**Source ID：** `THESIS-GLOBAL-ADVANTAGE-V1-P2-CAPABILITY-CARD-V1`
**输入证据：** P1 R5 的可重复 raw telemetry；共同参考为 frozen run-in head-on specialist；每个对手固定使用 `repeat_0/forward` 的 30 条 episode。P1 已验证三个 repeat 完全等价，因此该选择不是事后挑选。

## 卡片用途与边界

该卡片量化共同参考策略下三种对手诱导的压力差异，不构造 Elo、不定义单一总强度、也不宣称某个对手普适更强。完整机器可读结果见 `reports/thesis_global_advantage_v1_p2_opponent_capability_cards.yaml`。

| 对手 | target speed mean (m/s) | target altitude mean (m) | target energy mean (J/kg) | ego/target attack-zone fraction | ego crash/OOB | target crash/OOB |
|---|---:|---:|---:|---:|---:|---:|
| expert | 268.97 | 3984.98 | 75626.14 | 0.2183 / 0.1549 | 0.1333 | 0.2333 |
| end_to_end | 283.83 | 3470.91 | 74603.58 | 0.2976 / 0.0949 | 0.0333 | 0.4000 |
| independent PPO/VPP | 268.30 | 4006.54 | 75631.41 | 0.2323 / 0.1410 | 0.1000 | 0.3000 |

这些量只说明在该冻结 interaction 下，end-to-end 对手呈现更高平均速度和更低平均高度，且终端构成不同；它们不能被压缩为“强度排名”。

## P2-B1 未覆盖的关键限制

所有三张卡的记录 phase fraction 都是 `post_merge=1.0`、`pre_merge=0.0`、`re_entry=0.0`。因此它们不能证明新 heldout240 的初始 pre-merge 几何均可由 JSBSim 连续积分进入预期阶段，更不能证明 re-entry 覆盖。P2-B2 必须使用独立于 heldout240 evaluation seed 的 preflight seed，逐一验证 60 个 geometry cell x 三对手的 strict JSBSim reset、有限动作、无 fallback 和显式 phase/first-pass telemetry。
