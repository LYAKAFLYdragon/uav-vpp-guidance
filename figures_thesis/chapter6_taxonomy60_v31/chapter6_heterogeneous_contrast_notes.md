# 第六章异质对照补充说明

以下四图用于补充不同场景下的典型回合，重点是异质性而非重复展示。

- 异质对照 A：迎头闭合成功回合（规则型对手）： taxonomy60_wide_balanced_head_on_above_neg / expert / canonical_ppo，win，timeout_hp_advantage。
  - 原始回合：E:\uav-vpp-guidance-thesis-evidence\THESIS-DUALSKILL-TAXONOMY60-3OPP-V1\THESIS-DUALSKILL-TAXONOMY60-3OPP-V1-expert\raw\head_on\canonical_ppo\seed_79010\episode_010.json
- 异质对照 B：边界失稳回合（端到端神经对手）： taxonomy60_fast_closure_disadvantage_below_pos / end_to_end / canonical_ppo，loss，timeout_hp_disadvantage。
  - 原始回合：E:\uav-vpp-guidance-thesis-evidence\THESIS-DUALSKILL-TAXONOMY60-3OPP-V1\THESIS-DUALSKILL-TAXONOMY60-3OPP-V1-end_to_end\raw\head_on\canonical_ppo\seed_79043\episode_037.json
- 异质对照 C：中性几何下的成功回合（端到端神经对手）： taxonomy60_wide_balanced_neutral_level_neg / end_to_end / canonical_ppo，win，timeout_hp_advantage。
  - 原始回合：E:\uav-vpp-guidance-thesis-evidence\THESIS-DUALSKILL-TAXONOMY60-3OPP-V1\THESIS-DUALSKILL-TAXONOMY60-3OPP-V1-end_to_end\raw\head_on\canonical_ppo\seed_79020\episode_020.json
- 异质对照 D：多次恢复触发后仍成功回合（端到端神经对手）： taxonomy60_fast_closure_head_on_above_neg / end_to_end / canonical_ppo，win，timeout_hp_advantage。
  - 原始回合：E:\uav-vpp-guidance-thesis-evidence\THESIS-DUALSKILL-TAXONOMY60-3OPP-V1\THESIS-DUALSKILL-TAXONOMY60-3OPP-V1-end_to_end\raw\head_on\canonical_ppo\seed_79040\episode_034.json
  - 触发点：44，85，98，157。

推荐编号：图6-15至6-18。
推荐用途：作为 6.5–6.6 的补充对照图，从可视化角度表明多次恢复发生后仍能完成闭合。
