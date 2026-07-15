# P2-A GLOBAL-ADVANTAGE Heldout240 Manifest Freeze

**Source ID：** `THESIS-GLOBAL-ADVANTAGE-V1-HELDOUT240-V1`
**状态：** `static_manifest_frozen_not_yet_physically_reachability_validated`
**生成器：** `scripts/build_thesis_global_advantage_p2_manifests.py`

## 已冻结内容

| 维度 | 数量 | 验证结果 |
|---|---:|---|
| 初始态势 | 5 | 每态势 48 条 |
| 高度条件 | 3 | 每条件 80 条 |
| 镜像 | 2 | 每方向 120 条 |
| 未见 range-speed package | 2 | 每 package 120 条 |
| evaluation seed | 4 | 每 seed 60 条 |
| 物理 geometry cell | 60 | 每 cell 4 个独立 seed |
| 完整 episode instance | 240 | 唯一实例签名 |

`heldout240` 的两套 package 为 `global_heldout_far_fast_c` 与 `global_heldout_far_fast_d`；均在连续 train range/speed support 之外。实例签名与物理 geometry 签名分别验证：二者同 dev30 与历史 heldout60 的交集均为 0，训练 support 交集为 0。payload SHA-256：

`249234e96e802aa35a72bf2208986a723cba7387109fe1349fd0cff4f0b29526`

## 验证

`tests/test_thesis_global_advantage_p2_manifests.py`、`test_thesis_five_state_manifests.py` 与 `test_thesis_five_state_opponent_registry.py` 共 `9 passed`。

## 明确未完成内容

这是一份静态 manifest，不证明任何场景已经通过 JSBSim physical reachability、phase coverage 或对手压力检验。P2-B 必须先在三对手分别执行非学习 physical-reachability preflight，并以共同 frozen reference policy 生成定量 capability card；在此之前不得训练低层技能、高层 PPO，或使用 heldout240 评价候选方法。
