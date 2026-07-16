# Bounded Temporal Corridor Compiler 可执行性分析（只读）

本分析仅利用冻结 V16 state telemetry 构造当前信息 corridor action；不运行 JSBSim、不训练，且不主张 AA/ATA/能量改善。

`passing_candidates=['moderate_left', 'moderate_right']`；`passes=True`。

| Candidate | Opponent | Episodes | Steps | Mean projection residual (m) | Peak action norm | Hard-bound steps | Pass |
|---|---|---:|---:|---:|---:|---:|---|
| assertive_left | expert | 5 | 100 | 70.644 | 1.075 | 8 | False |
| assertive_left | end_to_end | 3 | 60 | 6.116 | 1.072 | 6 | False |
| assertive_left | independent_ppo_vpp | 6 | 120 | 21.772 | 1.081 | 5 | False |
| assertive_right | expert | 5 | 100 | 48.926 | 1.080 | 1 | False |
| assertive_right | end_to_end | 3 | 60 | 0.000 | 0.869 | 0 | False |
| assertive_right | independent_ppo_vpp | 6 | 120 | 37.881 | 1.088 | 6 | False |
| moderate_left | expert | 5 | 100 | 22.620 | 0.686 | 0 | True |
| moderate_left | end_to_end | 3 | 60 | 0.000 | 0.699 | 0 | False |
| moderate_left | independent_ppo_vpp | 6 | 120 | 4.779 | 0.678 | 0 | True |
| moderate_right | expert | 5 | 100 | 11.099 | 0.643 | 0 | True |
| moderate_right | end_to_end | 3 | 60 | 0.000 | 0.515 | 0 | False |
| moderate_right | independent_ppo_vpp | 6 | 120 | 10.936 | 0.674 | 0 | True |

通过仅授权起草全新、非学习 JSBSim action-to-trajectory transfer probe；它不授权训练。
不通过则说明该 action parameterization 在当前冻结包线内仍无可执行候选。
