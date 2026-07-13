# P3 Phase-Conditional v2 Temporal Encoder Report

**状态：** `passed_representation_readiness_only`  
**family：** `thesis_five_state_shared_intent_v1` / `noncanonical_thesis_extension`

## 目的与边界

P3 仅验证可冻结的 past-only temporal geometry encoder 是否可在新的、物理有效的 train-only 分布上完成自监督预训练。它不训练四个 shared skills 或高层 PPO，不评价空战胜率，也不构成 P4--P7 的性能证据。

P3 v1 的 joint `initial_state x phase x opponent` gate 已保留为负证据：真实 JSBSim 数据表明部分组合不是同一冷启动几何下的物理可达组合。v2 没有复用 v1 的训练数据，而是在全新 output root 中使用严格的 physical-marginal contract：

- 每个 initial state x opponent 至少 16 个窗口；
- 每个 phase x opponent 至少 12 个窗口；
- 独立 `head_on/crossing_entry` transition-provider rollout 对每个 opponent 的 post-merge 与 re-entry 各至少贡献 12 个窗口；
- 所有 provider 均从新的 JSBSim cold-start rollout 连续产生，不使用 snapshot、旧轨迹或 synthetic splice。

## 冻结输入与运行

- Config: [phaseconditional v2](E:/uav-vpp-guidance-thesis-five-state-v1/config/experiment/train_thesis_five_state_temporal_encoder_phaseconditional_v2.yaml)
- JSBSim: `v1.1.6`, commit `b477f6312bee2fd3af4c4e5ba1a3e732ed2b99d4`, F-16, strict backend.
- Train collection: `126` rollout = `90` initial-state-coverage + `36` independent transition-provider; backend 均为 JSBSim。
- Dev: `dev30 x 3 opponents = 90` rollout，仅 transform/evaluate；未保存 dev raw telemetry，未参与 normalization fitting。
- Heldout60: 未读取、未运行、未参与 checkpoint selection。
- Train compact windows: `8,442` before marginal selection; `1,518` selected training windows.
- Normalization: mean/std 仅由该轮 train rollout 拟合，见 `dataset/normalization_train_only.json`。

输出根：`E:/uav-vpp-guidance-thesis-five-state-v1-results/p3_temporal_encoder_phaseconditional_v2`。

## Coverage Gate

所有 marginal gate 均通过：

| 覆盖维度 | Expert | End-to-end | Independent PPO/VPP | 门槛 |
|---|---:|---:|---:|---:|
| pre-merge windows | 1615 | 1776 | 1616 | 12 |
| post-merge windows | 1079 | 929 | 1072 | 12 |
| re-entry windows | 120 | 109 | 126 | 12 |
| provider post-merge windows | 463 | 461 | 463 | 12 |
| provider re-entry windows | 59 | 57 | 58 | 12 |

五个 initial state 在三个 opponent 下也都超过 `16` 个窗口；完整表见 `collection_report.json`。

## Encoder Result

- Model: GRU self-supervised encoder, `16-D` past geometry window -> `32-D` embedding; history `K=10` high-level steps。
- Objective: masked-history reconstruction + horizon `1/5` relative-geometry delta prediction。
- Training: 80 epochs CPU；checkpoint selection solely by dev total loss。
- Selected checkpoint: `checkpoints/best.pt`, epoch `74`, dev total loss `0.3238515537`。
- Best checkpoint SHA-256: `385663281a9c48c0416ea4d1a1bb8dc08ed64a0cfebb0f01083cbbb8bcfbdc59`。
- Only `best.pt` and `last.pt` were retained; the selected checkpoint records `frozen_for_high_level_ppo=true`。

Synthetic unit tests also confirm that the encoder can recover the sign of range-rate, angle-proxy, and energy/altitude-proxy changes from past windows, and that independent windows do not retain previous-episode state.

## Gate Decision

P3 is passed **only** as representation-readiness evidence. The frozen encoder may be used as a P4/P5 input after an explicit checkpoint registry is created. It must not be fine-tuned by default, and this report must not be interpreted as a maneuver-performance or policy-improvement result.

Machine-readable artifacts:

- `E:/uav-vpp-guidance-thesis-five-state-v1-results/p3_temporal_encoder_phaseconditional_v2/collection_report.json`
- `E:/uav-vpp-guidance-thesis-five-state-v1-results/p3_temporal_encoder_phaseconditional_v2/training_report.json`
- `E:/uav-vpp-guidance-thesis-five-state-v1-results/p3_temporal_encoder_phaseconditional_v2/input_hashes.json`
- `E:/uav-vpp-guidance-thesis-five-state-v1-results/p3_temporal_encoder_phaseconditional_v2/p3_gate_decision.json`
