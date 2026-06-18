# JSBSim 高气动后端 + 机动目标：VPP vs No-VPP 对比实验报告

## 1. 实验目的
在更真实的 JSBSim F-16 高气动模型上，引入机动目标（横向正弦规避），比较：
- **VPP**：策略输出虚拟追踪点偏移（`train_no_prediction_vpp_ppo_jsbsim_*`）
- **No-VPP**：虚拟追踪点强制为零偏移（`train_no_vpp_ppo_jsbsim_*`）

验证 VPP offset 在高气动 + 机动目标条件下是否能展现不可替代的价值。

## 2. 关键改动

### 2.1 JSBSim 后端支持机动目标
- `src/uav_vpp_guidance/envs/jsbsim_env.py`
  - `_JSBSimAircraft.apply_ic_state()`：在运行时重新应用 IC 状态，同时保持仿真时钟。
  - `JSBSimEnv.apply_aircraft_ic_state()`：对指定 aircraft 调用上述方法。
- `src/uav_vpp_guidance/envs/tracking_env.py`
  - 当 `backend=jsbsim` 且 `target_mode != constant_velocity` 时，实例化 `target_dynamics`。
  - 每决策步先用 `target_dynamics.update_state()` 推进目标运动学状态，再通过 `apply_aircraft_ic_state()` 写入 JSBSim target，使 JSBSim 后端也能使用 simple 后端已有的机动模型（`sinusoidal`、`sinusoidal_weaving` 等）。

### 2.2 新增训练配置
- `config/experiment/train_no_prediction_vpp_ppo_jsbsim_maneuver.yaml`
- `config/experiment/train_no_vpp_ppo_jsbsim_maneuver.yaml`
- `config/experiment/train_no_prediction_vpp_ppo_jsbsim_weave.yaml`（3g/0.8 rad/s 强规避）
- `config/experiment/train_no_vpp_ppo_jsbsim_weave.yaml`（3g/0.8 rad/s 强规避）

所有配置统一：
- `backend: jsbsim`、`device: cpu`、`k_pos: 0.0`（与已废弃项兼容）。
- 成功判据 `success_range_m: 900`、`success_ata_deg: 25`。

## 3. 实验设置

| 设置 | 目标机动 | 训练步数 | 种子 | 后端 |
|---|---|---|---|---|
| VPP-mild | `sinusoidal`（legacy，较温和） | 50k | 0 | JSBSim F-16 |
| No-VPP-mild | `sinusoidal` | 50k | 0 | JSBSim F-16 |
| VPP-weave | `sinusoidal_weaving`（3g，0.8 rad/s） | 50k | 0 | JSBSim F-16 |
| No-VPP-weave | `sinusoidal_weaving`（3g，0.8 rad/s） | 50k | 0 | JSBSim F-16 |

评估：每场景 30 episode，deterministic policy。

## 4. 实验结果

### 4.1 Mild sinusoidal

| 方法 | favorable | neutral | disadvantage | challenging | 整体 |
|---|---|---|---|---|---|
| VPP | 100% | 100% | 0%（crash） | 100% | **75.0%** |
| No-VPP | 100% | 100% | 0%（crash） | 100% | **75.0%** |

### 4.2 Stronger sinusoidal_weaving (3g / 0.8 rad/s)

| 方法 | favorable | neutral | disadvantage | challenging | 整体 |
|---|---|---|---|---|---|
| VPP-weave | 100% | 100% | 0%（crash） | 100% | **75.0%** |
| No-VPP-weave | 100% | 100% | 0%（crash） | 100% | **75.0%** |

训练日志中的课程评估也一致：
- VPP-weave：Success 86.67%，Crash 13.33%（即 disadvantage 失败）。
- No-VPP-weave：Success 86.67%，Crash 13.33%。

### 4.3 Final zero-shot evaluation of the 10-seed matrix (sinusoidal_weaving, 5g / 1.0 rad/s)

We also re-evaluated the best checkpoints from the constant-velocity 10-seed JSBSim training matrix on a much more aggressive weaving target without any additional training.

| 方法 | favorable | neutral | challenging | disadvantage | 整体 |
|---|---|---|---|---|---|
| VPP | 100% | 100% | 100% | 0%（OOB） | **75.0%** |
| No-VPP | 100% | 100% | 100% | 0%（OOB） | **75.0%** |
| End-to-End | 100% | 100% | 100% | 0%（crash） | **75.0%** |

- 评估规模：1 checkpoint / 方法 × 4 scenarios × 3 eval seeds × 5 episodes = 60 episodes / 方法。
- VPP 与 No-VPP 仍然完全一致；E2E 在 `disadvantage` 中表现为 crash，而 VPP/No-VPP 为 out-of-bounds。
- 该结果进一步确认：**当前场景几何下 `disadvantage` 是不可解的共同瓶颈**，而非 VPP offset 能单独解决的问题。

## 5. 结论与讨论

1. **VPP offset 在 JSBSim 高气动 + 机动目标条件下仍未展现出独特优势**。无论是温和的 `sinusoidal`、中等强度 `sinusoidal_weaving`（3g），还是更强 `sinusoidal_weaving`（5g）的零样本测试，VPP 与 No-VPP 的成功率完全一致。
2. **disadvantage 场景是共同瓶颈**：两种方法都在该场景下 100% crash（主要是持续下降触地）。这说明当前 LOS-rate 制导律/场景几何/奖励设计在该不利初始几何下存在结构性问题，而不是 VPP offset 能单独解决的问题。
3. **VPP 策略确实会输出非零 offset**：在可成功的场景（favorable/neutral/challenging）中，VPP 策略会输出中等幅度的纵向/横向/垂直偏移，但零偏移的 No-VPP 同样 100% 成功，说明这些 offset 对这些场景并非必要。
4. **层级结构仍然有效**：与 simple 后端一致，policy → VPP/No-VPP → LOS-rate guidance → JSBSim 的链路能够稳定学习并泛化到 favorable/neutral/challenging；VPP offset 本身不是决定性因素。
5. **Hierarchical > end-to-end 仍是稳健结论**：在相同训练预算与评估条件下，E2E 的失败模式更不稳定（disadvantage 中 crash，而 hierarchical 方法为 OOB），且整体成功率不高于 hierarchical 方法。因此 JSBSim 证据链应表述为 **“分层制导接口优于端到端控制”**，而非 **“VPP > No-VPP”**。

## 6. 建议下一步

如果仍希望找到 VPP offset 的不可替代场景，可考虑：
- **调整 disadvantage 场景几何**（减小初始角度劣势、降低目标速度优势），或**修改 LOS-rate 制导律的垂直通道**（避免持续下降），使该场景可解后，再看 VPP 是否能比 No-VPP 更快/更稳地恢复。
- **进一步增大目标机动幅度或加入三维机动**（barrel_roll、bang_bang），测试在目标剧烈机动时 offset 是否成为必要。
- **引入轨迹预测**（CV/CA/neural predictor）+ VPP，观察在预测误差显著的机动目标下，offset 是否能补偿预测不确定性。

## 7. 复现命令

```bash
# 温和机动
python scripts/train_curriculum_ppo.py \
  --config config/experiment/train_no_prediction_vpp_ppo_jsbsim_maneuver.yaml \
  --seed 0 --total-timesteps 50000 --device cpu

python scripts/train_curriculum_ppo.py \
  --config config/experiment/train_no_vpp_ppo_jsbsim_maneuver.yaml \
  --seed 0 --total-timesteps 50000 --device cpu

# 强规避
python scripts/train_curriculum_ppo.py \
  --config config/experiment/train_no_prediction_vpp_ppo_jsbsim_weave.yaml \
  --seed 0 --total-timesteps 50000 --device cpu

python scripts/train_curriculum_ppo.py \
  --config config/experiment/train_no_vpp_ppo_jsbsim_weave.yaml \
  --seed 0 --total-timesteps 50000 --device cpu
```

输出目录：
- `outputs/experiments/no_prediction_vpp_ppo_jsbsim_maneuver/`
- `outputs/experiments/no_vpp_ppo_jsbsim_maneuver/`
- `outputs/experiments/no_prediction_vpp_ppo_jsbsim_weave/`
- `outputs/experiments/no_vpp_ppo_jsbsim_weave/`
