# Temporal Corridor Transfer V1 预注册

Source ID：`THESIS-FIVE-STATE-TEMPORAL-CORRIDOR-TRANSFER-V1`。

## 问题与边界

本 probe 仅回答：在冻结 JSBSim、VPP、guidance、PID、predictor、P3 encoder、fixed head-on specialist
和三对手条件下，bounded temporal corridor 是否相对 frozen fixed head-on 产生可测的六维联合几何改善。
它不是训练、奖励、技能库、高层 PPO、held-out 或论文性能实验。

## 冻结设计

- 方法：`frozen_fixed_head_on`、`moderate_left`、`moderate_right`。
- 每个 corridor 固定 20 个高层步；每一步仍只输出现有 `[-1,1]^3` VPP action，不扩大物理 action range。
- 两个 candidate 仅横向符号不同，extent 固定为 `450/300/100 m`，最大 action change 为 `0.18`。
- 场景：全新 Dev12，`head_on/crossing_entry x three heights x two mirrors`；与现有 manifest 的完整初始状态
  fingerprint 不相交。
- 对手必须分开报告，禁止 pooled gate。
- Manifest SHA-256：`95509a951cd3b1f01243c9bc3174095cd7b41d9ab22ab7fc112672f6645b0965`；runtime template、registry
  与 frozen P3 encoder 的 SHA-256 也已写入 design config。

## Gate 与 Stop Rule

每个 candidate、每个对手均需：`>=4` 条 matching-handoff qualifying pair、20-step 完整窗口、mean
projection residual `<=25 m`、hard-bound step `=0`、ego crash/OOB delta `<=+0.05`。六维指标还需 weighted
error delta `<=-0.01`、至少两个分量 `<=-0.01`、任一分量不恶化超过 `+0.02`。

同一个 candidate 必须在至少两个对手独立通过，才可称 mechanism candidate；即使通过，也只授权起草训练
pilot。任一 fallback、非有限 action、handoff reset、output-root 重用或 contract failure 都冻结 Source ID。
若未通过，不调 extent、slew、window、reward、场景或训练步数。
