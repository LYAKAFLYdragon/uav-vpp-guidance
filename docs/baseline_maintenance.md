# Baseline 维护日志

## Baseline v1.0（2026-06-12）

- 合并 commit: `ce6d6eb`
- Tag: `baseline-v1.0`
- Smoke gate: PASS
  - L1 fixed-gain: SR=51.11%, crash=23.33%
  - L3 RL baseline: SR=36.67%, crash=13.33%, OOB=50.00%
- 已知限制：
  - 2.0s hold 在 50k–200k steps 下不可行（0% SR），保留在 `config/success_criteria/strict.yaml` 作为远期目标。
  - L3 OOB=50%，处于 gate 阈值；已添加 `w_boundary` 边界接近惩罚，但 50k 步预算内未进一步降低。
- 冻结文件（任何修改必须通过 hotfix 流程并重新跑 smoke gate）：
  - `config/success_criteria/*.yaml`
  - `config/reward.yaml`
  - `config/experiment/*.yaml`
  - `src/uav_vpp_guidance/envs/termination.py`
  - `src/uav_vpp_guidance/guidance/los_rate_guidance.py`
  - `src/uav_vpp_guidance/training/train_*.py`

## Hotfix 记录

| 日期 | Issue | 修复文件 | 影响 | Smoke Gate | Tag |
|------|-------|----------|------|------------|-----|
| — | — | — | — | — | — |

## Background 任务进度

| 任务 | 状态 | 备注 |
|------|------|------|
| 2.0s hold 调优 | ⏸️ 未开始 | 远期目标，需更大网络/更长课程/更多步数 |
| OOB <30% | 🔄 进行中 | 边界惩罚已添加，需进一步调参 |
| `test_stage6b_benchmark.py` 修复 | ⏸️ 未开始 | pre-existing 失败，与基线无关 |

## 维护模式规则

- **允许**：文档更新、测试修复（不影响 smoke gate）、background 参数探索（在新配置中，不修改冻结文件）。
- **禁止**：修改冻结配置/代码、新增功能、任何可能改变 smoke gate 结果的行为变更。
- **例外**：分支 B 发现严重 bug 时，通过 hotfix 分支修改冻结文件，重新跑 smoke gate，必要时更新 tag 并通知分支 B 重跑实验。
