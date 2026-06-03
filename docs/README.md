# Documentation Index

## Overview

This project implements **Virtual Pursuit Point (VPP) guidance with trajectory prediction** for close-range aerial combat. The framework supports both a simplified point-mass backend and a high-fidelity JSBSim F-16 backend.

## Architecture

```
┌──────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│   Policy     │────▶│ Virtual Point        │────▶│ LOS Rate Guidance    │
│  (PPO)       │     │ Generator            │     │ (Nz/roll_rate/throt) │
└──────────────┘     └──────────────────────┘     └──────────────────────┘
                                                           │
                           ┌───────────────────────────────┘
                           ▼
                    ┌──────────────┐     ┌──────────────┐
                    │ Low-Level    │────▶│   Backend    │
                    │ Controller   │     │              │
                    └──────────────┘     │  simple /    │
                                         │  jsbsim      │
                                         └──────────────┘
```

## Phase Documentation

| Phase | Document | Status |
|-------|----------|--------|
| Phase 1 — Framework Setup | N/A (initial setup) | ✅ Complete |
| Phase 2 P1 — JSBSim Minimal Closed Loop | N/A (test-driven) | ✅ Complete |
| Phase 2 P2 — No-Prediction VPP Baseline (Simple) | `no_prediction_vpp_baseline.md` | ✅ Complete |
| Phase 2 P3 — JSBSim No-Prediction VPP Bridge | `no_prediction_vpp_jsbsim_bridge.md` | ✅ Complete |
| Phase 5 — No-Prediction VPP PPO Training | `no_prediction_vpp_ppo_training.md` | ✅ Complete |
| Phase 6A — Classical CV/CA Prediction VPP Integration | `classical_prediction_vpp_integration.md` | ✅ Complete |
| Phase 6B — Full PPO Training & Scenario Evaluation (No-Pred / CV / CA) | `stage6b_prediction_comparison_experiment.md` | ✅ Ready |
| Phase 6C — Neural Trajectory Prediction (LSTM/GRU) Offline Training | `trajectory_prediction/dataset.py`, `trainer.py`, `train_pipeline.py` | ✅ Complete |
| Phase 6D — Neural Predictor Online Integration | `predictor_adapter.py` (checkpoint loading, LSTM/GRU closed-loop) | ✅ Complete |

## Key Documents

- **[`no_prediction_vpp_baseline.md`](./no_prediction_vpp_baseline.md)**: SimplePointMass 后端上的 No-Prediction VPP 基线。包含框架设计、评估流程、指标定义、成功/终止判据。
- **[`no_prediction_vpp_jsbsim_bridge.md`](./no_prediction_vpp_jsbsim_bridge.md)**: JSBSim 高保真后端的 No-Prediction VPP 桥接。包含控制接口映射、5 Hz/60 Hz 分层设计、输出文件说明、结果解读方法。
- **[`no_prediction_vpp_ppo_training.md`](./no_prediction_vpp_ppo_training.md)**: No-Prediction VPP PPO 自主决策基线。包含策略网络结构、PPO 训练流程、评估命令、训练输出说明。
- **[`classical_prediction_vpp_integration.md`](./classical_prediction_vpp_integration.md)**: Stage 6A 经典 CV/CA 预测器接入。包含 CV/CA 公式、predicted_target 锚点数据流、P1 修复说明、训练/评估命令、当前局限性。
- **[`stage6b_prediction_comparison_experiment.md`](./stage6b_prediction_comparison_experiment.md)**: Stage 6B 完整实验流程。包含多种子训练、固定/随机场景评估、per-method checkpoint 加载、per-scenario 对比绘图、JSBSim sanity check、远程执行 checklist。

## Configuration Files

- `config/experiment/no_prediction_vpp_scenarios.yaml` — SimplePointMass 评估场景
- `config/experiment/no_prediction_vpp_jsbsim.yaml` — JSBSim 评估场景
- `config/experiment/train_no_prediction_vpp_ppo.yaml` — PPO 训练实验配置
- `config/env/default.yaml` — 默认环境配置（后端选择、时间步长、终止条件等）

## Running Experiments

### Evaluation

```bash
# Simple backend
python -m uav_vpp_guidance.evaluation.evaluate_no_prediction_scenarios \
    --config config/experiment/no_prediction_vpp_scenarios.yaml \
    --backend simple \
    --episodes 5 --seeds 0 1

# JSBSim backend
python -m uav_vpp_guidance.evaluation.evaluate_no_prediction_scenarios \
    --config config/experiment/no_prediction_vpp_jsbsim.yaml \
    --backend jsbsim \
    --episodes 2 --seeds 0
```

### Plotting

```bash
# Simple backend
python -m uav_vpp_guidance.visualization.plot_no_prediction_results \
    --metrics outputs/tables/no_prediction_vpp/simple/scenario_metrics.csv \
    --trajectories outputs/trajectories/no_prediction_vpp/simple \
    --output outputs/figures/no_prediction_vpp/simple \
    --backend simple

# JSBSim backend
python -m uav_vpp_guidance.visualization.plot_no_prediction_results \
    --metrics outputs/tables/no_prediction_vpp/jsbsim/scenario_metrics.csv \
    --trajectories outputs/trajectories/no_prediction_vpp/jsbsim \
    --output outputs/figures/no_prediction_vpp/jsbsim \
    --backend jsbsim
```

### PPO Training

```bash
# Smoke test
python -m uav_vpp_guidance.training.train_no_prediction_vpp_ppo \
    --config config/experiment/train_no_prediction_vpp_ppo.yaml --smoke

# Full training
python -m uav_vpp_guidance.training.train_no_prediction_vpp_ppo \
    --config config/experiment/train_no_prediction_vpp_ppo.yaml
```

### Policy Evaluation

```bash
# Simple backend
python -m uav_vpp_guidance.evaluation.evaluate_policy \
    --config config/experiment/train_no_prediction_vpp_ppo.yaml \
    --checkpoint outputs/experiments/no_prediction_vpp_ppo/checkpoints/best.pt \
    --backend simple --episodes 10 --seeds 0 1 2 --save-trajectories

# JSBSim backend
python -m uav_vpp_guidance.evaluation.evaluate_policy \
    --config config/experiment/train_no_prediction_vpp_ppo.yaml \
    --checkpoint outputs/experiments/no_prediction_vpp_ppo/checkpoints/best.pt \
    --backend jsbsim --episodes 2 --seeds 0 --save-trajectories
```

### Stage 6B: Prediction Comparison (No-Pred / CV / CA)

```powershell
# Training (run on remote machine)
.\scripts\train_stage6b.ps1

# Evaluation with per-method checkpoints
python -m uav_vpp_guidance.evaluation.evaluate_prediction_comparison `
    --config config/experiment/evaluate_vpp_prediction_comparison.yaml `
    --method-checkpoint no_prediction=outputs/experiments/stage6b_no_pred_s0/checkpoints/best.pt `
    --method-checkpoint cv_prediction=outputs/experiments/stage6b_cv_s0/checkpoints/best.pt `
    --method-checkpoint ca_prediction=outputs/experiments/stage6b_ca_s0/checkpoints/best.pt `
    --backend simple --episodes 50 --seeds 0 1 2 `
    --scenarios favorable neutral disadvantage challenging `
    --output-dir outputs/tables/stage6b_simple_fixed

# Plotting
.\scripts\plot_stage6b.ps1 -backend simple -scenarioSet fixed

# JSBSim sanity check
.\scripts\eval_jsbsim_sanity.ps1 `
    -checkpoint outputs/experiments/stage6b_cv_s0/checkpoints/best.pt `
    -config config/experiment/train_vpp_ppo_cv.yaml `
    -episodes 5
```

### Training Curves

```bash
python -m uav_vpp_guidance.visualization.plot_training_curves \
    --log-dir outputs/experiments/no_prediction_vpp_ppo/logs \
    --output outputs/experiments/no_prediction_vpp_ppo/figures
```

### Tests

```bash
pytest tests/ -v
```

## Project Structure

```
uav-vpp-guidance/
├── src/uav_vpp_guidance/
│   ├── envs/              # Environment backends (simple / jsbsim)
│   ├── guidance/          # Virtual point, LOS guidance
│   ├── flight_control/    # Low-level controller, actuator interface
│   ├── evaluation/        # Scenario evaluation scripts
│   ├── visualization/     # Plotting scripts
│   └── utils/             # Coordinate transforms, geometry, etc.
├── config/                # YAML configs (env, experiment, scenarios)
├── tests/                 # pytest suite
├── docs/                  # Documentation (this directory)
└── outputs/               # Generated results (gitignored)
```

## Notes

- All guidance gains, reward weights, and termination thresholds are **YAML-driven**; no hard-coded mission parameters in source files.
- The `simple` and `jsbsim` backends expose **unified state fields** (`position_neu`, `velocity_ned`, `attitude_rpy`, `vt_mps`) so that geometry computation and guidance work identically.
- Trajectory prediction is **intentionally disabled** in the current no-prediction baseline (`trajectory_prediction.enabled: false`, `anchor_mode: current_target`).
