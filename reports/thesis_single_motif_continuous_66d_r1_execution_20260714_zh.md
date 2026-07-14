# Single-Motif Continuous 66-D Collector R1 执行结论

## 1. 决策结论

`THESIS-SINGLE-MOTIF-CONTINUOUS-66D-R1` 的研发 gate **通过**。三种冻结对手下均已从合法 reset、连续 first-pass run-in 和真实 JSBSim 状态中构造出足量的：

`16-D base + 10-frame real history + 6-D explicit history + 32-D frozen P3 embedding + 6-D intent target + 6-D intent weight -> normalized 3-D VPP action`

因此，当前阻塞已不再是“连让技能学习该行为的输入输出契约都没有建立”。证据现在支持把后续问题收窄为：**disadvantage -> post-merge/re-entry 下存在 defensive-extension 新技能候选缺口**。

这一结论只说明可学习接口和目标 motif 数据支持域成立，不证明新技能有效、不证明现有 reference action 最优，也不授权自动训练。按预注册边界，下一步只允许起草一个单一 defensive-extension feasibility pilot 的预注册方案。

## 2. 分对手 Gate

| Opponent | Episodes | Qualifying episodes | Valid target steps | Distinct signatures | Mirrors | Gate |
|---|---:|---:|---:|---:|---:|---|
| expert | 12 | 11 | 958 | 11 | 2 | pass |
| end_to_end | 12 | 11 | 566 | 11 | 2 | pass |
| independent_ppo_vpp | 12 | 8 | 934 | 8 | 2 | pass |

预注册门槛为每个对手至少 2 条 qualifying episode、20 个有效目标 phase step、2 个独立 scenario signature 和 2 个镜像方向。三种对手分别判定，没有 pooled 放宽。

全部 36 条 episode 均满足：

- exact 66-D 与 normalized 3-D action 有限；
- 10 帧历史全部来自真实观测，前 9 步仅记录 warm-up，未 padding；
- action 在 collector 前后逐值一致，未替换、裁改或回写；
- contract-ready 样本的冻结预测器均 valid 且无 fallback；
- strict JSBSim、无 backend fallback、无 reset、无 snapshot restore、无 future-state injection；
- v2 first-pass `k -> k+1` continuity ledger 有效；
- `(defensive_extension, range_extension)` validity mask 允许目标样本。

## 3. 证据入口

- Config: [jsbsim_hrl_thesis_single_motif_continuous_66d_r1.yaml](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/config/experiment/jsbsim_hrl_thesis_single_motif_continuous_66d_r1.yaml)
- Collector: [single_motif_continuous_66d.py](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/src/uav_vpp_guidance/evaluation/single_motif_continuous_66d.py)
- Analyzer: [analyze_thesis_single_motif_continuous_66d.py](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/scripts/analyze_thesis_single_motif_continuous_66d.py)
- Gate JSON: [single_motif_continuous_66d_gate.json](E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/single_motif_continuous_66d_r1/analysis_r1/single_motif_continuous_66d_gate.json)
- Episode matrix: [single_motif_continuous_66d_episode_matrix.csv](E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/single_motif_continuous_66d_r1/analysis_r1/single_motif_continuous_66d_episode_matrix.csv)
- Machine-generated Chinese gate report: [single_motif_continuous_66d_gate_zh.md](E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/single_motif_continuous_66d_r1/analysis_r1/single_motif_continuous_66d_gate_zh.md)
- Source provenance: [single_motif_continuous_66d_source_provenance.json](E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/single_motif_continuous_66d_r1/analysis_r1/single_motif_continuous_66d_source_provenance.json)

Formal run directories:

- `SM66D-R1-EXPERT-20260714`
- `SM66D-R1-ENDTOEND-20260714`
- `SM66D-R1-INDPPOVPP-20260714`

## 4. Provenance Caveat

三条 run 的 `run_manifest.json` 均正确记录 `git_dirty=true`，因此 `paper_safe=false`；唯一 invalid reason 是 dirty worktree。原因是本 worktree 同时承载尚未提交的 v2 continuity 与本 collector 实现。

所以本轮可以作为研发 go/no-go 证据回答“数据契约是否成立”，但不得直接作为论文 formal 主结果。分析包已额外冻结 config、collector、runner、analyzer 与 P3 checkpoint 的逐文件 SHA-256，避免实现版本失配。若以后需要将其升级为论文证据，应另行冻结 clean SHA 并预先决定是否允许 provenance-only replication；不得把该动作伪装成本轮结果驱动重跑。

## 5. 后续边界

允许：起草一个单技能 defensive-extension feasibility pilot 预注册，明确固定 skill、固定 `range_extension` profile、三对手分开门槛、baseline、safety stop rule 和新 output ID。

继续禁止：四共享技能整体训练、combat finetune、P5 高层 PPO、profile/temporal encoder 正式消融、奖励调参、P4-v1 修补，以及未经单独授权的自动训练。
