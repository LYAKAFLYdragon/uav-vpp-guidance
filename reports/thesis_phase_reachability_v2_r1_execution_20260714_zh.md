# Phase Reachability v2 R1 执行结论

## 结论

`THESIS-PHASE-REACHABILITY-V2-RUN-IN-HANDOFF-R1` 已完成独立、只评估的 strict-JSBSim physical run-in probe，结论为 `phase_reachability_v2_established`。

这仅说明：在本 v2 的 12 个未复用 v1 的 disadvantage 初始条件下，冻结的 `legacy_static_oracle_task_gate` 可以从合法 reset 连续运行到 first-pass，并在下一真实 JSBSim 高层步保留可验证的 `k -> k+1` 连续性，且在三种对手下均有至少 50% episode 同时达到 post-merge 与 re-entry 预注册门槛。

它**不**证明新的共享技能库有效、现有技能库存在确定缺口、战斗优势、安全认证、策略泛化，也不自动授权 P4 combat finetune、四共享技能、P5 高层 PPO、最小 pilot 或任何调参。这些路线继续锁定；P4 v1 的 `NO-GO` 负证据不受本结果推翻。

## 冻结执行契约

- Config: [jsbsim_hrl_thesis_phase_reachability_v2_run_in_handoff_r1.yaml](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/config/experiment/jsbsim_hrl_thesis_phase_reachability_v2_run_in_handoff_r1.yaml)
- Manifest: [thesis_phase_reachability_v2_run_in_handoff_r1.yaml](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/config/experiment/manifests/thesis_phase_reachability_v2_run_in_handoff_r1.yaml)
- Manifest payload SHA-256: `5ffdaa4bc248bf40c1532b13c8030dca43240711f5751cf8c4f26217525cfaae`
- Reference controller: frozen non-learning `legacy_static_oracle_task_gate` only.
- Envelope: 2 new distance/speed packages x 3 height conditions x 2 mirror directions = 12 scenarios; all reset range > 1000 m, all retain initial `disadvantage` taxonomy, and v1 manifest was not reused.
- Prohibitions: no policy training, tuning, action replacement, VPP/guidance/PID modification, reset after first-pass, snapshot restore, or future-state injection.
- Runtime: JSBSim `v1.1.6` root was supplied explicitly through `--jsbsim-root` and recorded in each run provenance. The first attempted expert run (`PHASE-V2-R1-EXPERT-20260714`) created zero episodes because the worktree-local runtime path was absent; it is a deployment failure, not evidence, and the completed `RETRY1` run has a distinct run ID.

## Gate Results

| Opponent | Episodes | Valid continuity | Qualified post-merge + re-entry | Fraction | Gate |
|---|---:|---:|---:|---:|---|
| expert | 12 | 12 | 8 | 0.6667 | pass |
| end_to_end | 12 | 12 | 10 | 0.8333 | pass |
| independent_ppo_vpp | 12 | 12 | 9 | 0.7500 | pass |

No opponent aggregation was used. The gate summary and episode matrix are:

- [phase_reachability_v2_gate.json](E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/phase_reachability_v2_run_in_handoff_r1/analysis_r1/phase_reachability_v2_gate.json)
- [phase_reachability_v2_episode_matrix.csv](E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/phase_reachability_v2_run_in_handoff_r1/analysis_r1/phase_reachability_v2_episode_matrix.csv)

Raw episode-record SHA-256 values are `ff79fa8795ab0e4c9b612e1a56f60cb6ffeb93e8df3acf1bbdebc4fb9c568dac` (expert), `72f56245395d46ef555d9487fcd855af5ed7c55a9fac0a868929afd821d8bbe2` (end-to-end), and `cea3a42abdd88a7a9474a0e0e6d75828cfabb4ccddc1ebe6155e6cb48b7ddd36` (independent PPO/VPP).

## Continuity Interpretation And Limits

The sidecar records a compact boundary ledger at first-pass step `k` and a sampler observation ledger at physical step `k+1`. The latter holds explicit `parent_*` hashes for the boundary history, VPP action, guidance telemetry, and PID telemetry; it is not required to equal the naturally advanced current values. All 36 ledgers retained one episode lineage, one JSBSim lineage, one observation schema SHA, and the frozen trajectory-predictor SHA; no ledger recorded fallback, reset, or future-state injection.

The reference controller is memoryless, so its history entry is explicitly labelled `not_exposed_memoryless_reference` and hashes the contemporaneous observation rather than asserting a hidden recurrent buffer. PID continuity is likewise established at the available command/response telemetry boundary, not by claiming access to private integrator internals. These are intentional observability limits and must remain in any later manuscript or pilot rationale.

## Decision

v2 resolves only the prior *phase reachability* blockage: a physically continuous post-merge/re-entry envelope is now observable under each frozen opponent. Before any new training can be proposed, the next admissible action is a separate, reviewed single-skill feasibility decision that specifies a new objective, fixed comparison, safety gate, and non-overlapping evaluation. It may not reinterpret v2 as a reward, performance, or shared-skill efficacy result.
