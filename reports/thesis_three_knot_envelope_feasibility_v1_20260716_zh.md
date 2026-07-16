# 三锚点动态参考路径可执行包线分析（只读）

该分析仅重放 V16 已冻结 telemetry，不运行 JSBSim、不训练，也不估计未来 AA/ATA/能量响应。

## 预注册前置结论

`passing_candidates=[]`；`passes=False`。

候选通过条件：同一候选至少两个对手分别具有不少于 4 条全程无裁剪 episode，且无裁剪 step 比例不少于 0.90。

## 分对手矩阵

| Candidate | Opponent | Episodes | Steps | Unclipped-step fraction | Fully unclipped episodes | Clipped axes (x/y/z) | Pass |
|---|---|---:|---:|---:|---:|---|---|
| balanced_escape | expert | 5 | 100 | 0.360 | 0 | [27, 63, 1] | False |
| balanced_escape | end_to_end | 3 | 60 | 0.850 | 2 | [9, 0, 0] | False |
| balanced_escape | independent_ppo_vpp | 6 | 120 | 0.567 | 1 | [26, 49, 0] | False |
| compact_escape | expert | 5 | 100 | 0.450 | 1 | [14, 55, 0] | False |
| compact_escape | end_to_end | 3 | 60 | 1.000 | 3 | [0, 0, 0] | False |
| compact_escape | independent_ppo_vpp | 6 | 120 | 0.692 | 2 | [19, 36, 0] | False |
| legacy_extent_escape | expert | 5 | 100 | 0.260 | 0 | [46, 70, 12] | False |
| legacy_extent_escape | end_to_end | 3 | 60 | 0.467 | 0 | [22, 10, 4] | False |
| legacy_extent_escape | independent_ppo_vpp | 6 | 120 | 0.325 | 0 | [68, 64, 3] | False |
| lower_bound_escape | expert | 5 | 100 | 0.500 | 1 | [12, 49, 0] | False |
| lower_bound_escape | end_to_end | 3 | 60 | 1.000 | 3 | [0, 0, 0] | False |
| lower_bound_escape | independent_ppo_vpp | 6 | 120 | 0.717 | 3 | [15, 32, 0] | False |
| moderate_escape | expert | 5 | 100 | 0.410 | 0 | [20, 59, 0] | False |
| moderate_escape | end_to_end | 3 | 60 | 1.000 | 3 | [0, 0, 0] | False |
| moderate_escape | independent_ppo_vpp | 6 | 120 | 0.625 | 1 | [22, 42, 0] | False |
| wide_escape | expert | 5 | 100 | 0.320 | 0 | [31, 68, 8] | False |
| wide_escape | end_to_end | 3 | 60 | 0.567 | 0 | [16, 6, 5] | False |
| wide_escape | independent_ppo_vpp | 6 | 120 | 0.508 | 0 | [32, 55, 2] | False |

## 解释边界

通过只说明候选空间未被冻结 action head 大量裁剪，可据此起草新的非学习物理传递 probe；不通过则禁止创建该 probe。
该离线分析不能声称六维 intent 改善，六维改善必须在全新、预注册的 JSBSim 对比中测量。
