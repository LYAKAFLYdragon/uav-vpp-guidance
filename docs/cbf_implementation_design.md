# CBF 安全过滤实现设计文档

**日期**：2026-06-18  
**作者**：AI coding assistant  
**范围**：`src/uav_vpp_guidance/safety/`, `src/uav_vpp_guidance/envs/`, `scripts/evaluate_cbf_safety.py`, `scripts/plot_cbf_safety.py`, `scripts/evaluate_cbf_adversarial.py`, `scripts/plot_cbf_adversarial.py`

---

## 1. 设计目标

在现有 UAV-VPP-Guidance 代码库中实现 **Control Barrier Function (CBF) QP 安全过滤层**：

- 在 PPO 输出的 VPP 偏移量进入 `VirtualPointGenerator` 之前进行过滤。
- 保证追击机与目标机的最小分离距离不小于 `d_min`。
- 不修改 JSBSim 黑箱动力学、LOS 制导律、PPO Agent、VPP 生成器。
- 使用简化 point-mass 模型 + 数值 Jacobian 估计。

---

## 2. 理论公式

### 2.1 系统简化模型

在短时域 `Δt = 0.2 s` 内使用水平面 point-mass 近似：

```
ṗ_o = v_o
v̇_o = K · a
```

- `a ∈ [-1, 1]^3`：VPP 偏移（RL 动作）。
- `K = ∂v_o / ∂a ∈ R^{3×3}`：VPP 偏移到追击机加速度的 Jacobian，数值估计。

### 2.2 安全函数

```
h(x) = ||p_o - p_t|| - d_min
```

### 2.3 一阶与二阶 CBF 条件

一阶：
```
ḣ + γ·h ≥ 0
ḣ = r̂^T · (v_o - v_t)
```

二阶：
```
ḧ + γ_1·ḣ + γ_2·h ≥ 0
ḧ = r̂^T·(K·a - a_t) + (||ṙ||^2 - (r̂^T·ṙ)^2) / ||r||
```

其中 `a_t` 为目标机加速度，通过相邻两步目标速度差分估计。

### 2.4 QP 安全过滤

```
min   ||a - a_RL||^2 + λ·s^2
s.t.  r̂^T·K·a + b ≥ 0
      -1 ≤ a ≤ 1
      s ≥ 0
```

其中：
```
b = r̂^T·a_t - (||ṙ||^2 - (r̂^T·ṙ)^2)/||r|| - γ_1·ḣ - γ_2·h
```

如果 QP 不可行，使用沿约束梯度方向的解析投影作为 fallback。

---

## 3. 实现结构

### 3.1 新增模块

| 文件 | 说明 |
|------|------|
| `src/uav_vpp_guidance/safety/__init__.py` | safety 包初始化 |
| `src/uav_vpp_guidance/safety/cbf_filter.py` | `CBFQPFilter`：CBF 检查、QP 求解、fallback |
| `src/uav_vpp_guidance/safety/jacobian_estimator.py` | `PointMassJacobianEstimator`：基于局部 `SimplePointMassEnv` 的数值 Jacobian |
| `tests/test_cbf_filter.py` | 单元测试 |
| `config/safety/cbf_default.yaml` | 默认参数（`d_min=500`, `γ1=2`, `γ2=1`） |
| `config/safety/cbf_conservative.yaml` | 保守参数（`d_min=700`） |
| `config/safety/cbf_aggressive.yaml` | 激进参数（`d_min=300`） |
| `config/safety/cbf_adversarial_500.yaml` | 对抗场景下 tuned 的 `d_min=500`（`γ=0.2`，提前、柔和干预） |
| `config/safety/cbf_adversarial_700.yaml` | 对抗场景下 tuned 的 `d_min=700`（`γ=0.2`） |
| `scripts/evaluate_cbf_adversarial.py` | 对抗目标下 baseline vs CBF 评估 |
| `scripts/plot_cbf_adversarial.py` | 对抗 CBF 结果绘图与对比表 |

### 3.2 环境集成

- `CloseRangeTrackingEnv.__init__`：读取 `config["cbf"]`，若启用且使用 VPP 则实例化 `CBFQPFilter`。
- `CloseRangeTrackingEnv.step`：在 `virtual_point_generator` 之前调用 CBF 过滤。
- `AdversarialJSBSimEnv.__init__`：同样读取 `config["cbf"]`。
- `AdversarialJSBSimEnv.step`：在 expert override 之后、`_compute_virtual_point` 之前调用 CBF 过滤。
- 两个环境在 `reset()` 中调用 `cbf_filter.reset()` 清空目标加速度缓存。

### 3.3 QP 求解器

默认使用 `scipy.optimize.minimize`（SLSQP）。原因：

- 本机 Python 3.8 + JSBSim 环境下，`cvxpy`/`osqp` 的 C 扩展在 `jsbsim` 导入后会出现 access violation。
- 3 维 QP 用 SLSQP 平均耗时 < 1 ms，满足 5 Hz 决策频率。
- 保留 OSQP 接口作为可选后端（`solver="osqp"`），但不在 JSBSim 流程中默认启用。

---

## 4. 数值 Jacobian 估计

`PointMassJacobianEstimator` 的工作流程：

1. 根据当前追击机/目标机状态构造一个临时的 `SimplePointMassEnv`。
2. 复用宿主环境的 `VirtualPointGenerator` + `LOSRateGuidance` + `clip_command` 计算 guidance 命令。
3. 对 base action 和各维度扰动后的 action，分别前进一步 `SimplePointMassEnv`。
4. 返回 `K[:, j] = Δv_o / (eps · dt)`。

该估计只在 **一阶 CBF 条件不满足时** 执行，避免每步都进行有限差分。

---

## 5. 验证结果

使用 `outputs/jsbsim_10seed_matrix/vpp_s0/checkpoints/best.pt`（JSBSim 训练）在 canonical 4 scenarios 上评估：

- seeds：`[0, 1, 2]`
- 每个 scenario 10 episodes，共 120 episodes per method

| 方法 | 成功率 | 碰撞率 | OOB 率 | 平均最小距离 | 全局最小距离 | CBF 触发率 |
|------|--------|--------|--------|--------------|--------------|------------|
| VPP baseline | 75.0% | 0.0% | 25.0% | 790.4 m | 674.9 m | — |
| VPP + CBF (d_min=500) | 75.0% | 0.0% | 25.0% | 790.4 m | 674.9 m | 0.0% |
| VPP + CBF (d_min=700) | 75.0% | 0.0% | 25.0% | 790.4 m | 674.9 m | 3.8% |

观察：

1. 在 `d_min=500` 的默认配置下，baseline 本身已满足安全距离，CBF 未触发，说明该策略已足够安全。
2. CBF 未降低成功率（75.0% → 75.0%）。
3. QP 平均求解时间约 0.26 ms，最大 21.8 ms，满足实时性。
4. 当把 `d_min` 提高到 700 m 时，CBF 在 disadvantage scenario 中被激活（3.8% 步数），但由于简化 point-mass 模型与 JSBSim 六自由度动力学存在差异，未能完全将最小距离拉回 700 m 以上。这体现了“近似保证”的局限性。

### 5.2 对抗目标验证

为了触发 CBF，使用训练好的对抗目标 `target_agent_ppo_v2` 与 baseline 追击机对抗：

```bash
# 1. 扩展评估（3 seeds × 10 eps = 30 eps per condition）
python scripts/evaluate_cbf_adversarial.py \
    --pursuer-ckpt outputs/jsbsim_10seed_matrix/vpp_s0/checkpoints/best.pt \
    --target-ckpt outputs/adversarial/training/target_agent_ppo_v2/checkpoints/best.pt \
    --num-episodes 10 --seeds 0 1 2 \
    --scenarios tail_chase_2000m head_on_4000m offset_45deg_2000m \
    --output-dir outputs/cbf_adversarial/full_v2/baseline

python scripts/evaluate_cbf_adversarial.py --use-cbf \
    --cbf-config config/safety/cbf_adversarial_500.yaml \
    --output-dir outputs/cbf_adversarial/full_v2/cbf500

python scripts/evaluate_cbf_adversarial.py --use-cbf \
    --cbf-config config/safety/cbf_adversarial_700.yaml \
    --output-dir outputs/cbf_adversarial/full_v2/cbf700

# 2. 论文级图表与统计检验
python scripts/plot_cbf_adversarial_paper.py \
    --baseline outputs/cbf_adversarial/full_v2/baseline/eval_results.json \
    --cbf500 outputs/cbf_adversarial/full_v2/cbf500/eval_results.json \
    --cbf700 outputs/cbf_adversarial/full_v2/cbf700/eval_results.json \
    --output-dir outputs/cbf_adversarial/figures_paper \
    --stats-dir outputs/cbf_adversarial
```

结果（3 seeds × 10 eps = 30 episodes/condition）：

| 方法 | 成功率 | 碰撞率 | OOB 率 | 平均最小距离 | CBF 触发率 | QP 平均耗时 |
|------|--------|--------|--------|--------------|------------|-------------|
| Baseline | 33.3% | 0.0% | 66.7% | 579.0 m | — | — |
| CBF(500) | 33.3% | 0.0% | 66.7% | 665.0 m | 8.0% | 0.65 ms |
| CBF(700) | 33.3% | 0.0% | 66.7% | 676.4 m | 11.7% | 0.85 ms |

统计检验（Mann-Whitney U，n=90/condition）：

- Baseline vs CBF(500)：p = 9.0e-5，CBF 显著提高最小距离。
- Baseline vs CBF(700)：p = 9.0e-5，CBF 显著提高最小距离。
- Per-scenario：head-on 与 offset 的 p < 1.7e-14，tail-chase 无差异（本身已安全）。

关键发现：

- 对抗目标使 baseline 在 head-on/offset 场景下最小距离降至 501 m / 351 m，CBF 将其分别提升到 619 m / 491 m（d_min=500）和 647 m / 494 m（d_min=700）。
- `γ=0.2` 的柔和增益可在 adversarial 近距离交会中避免过激机动导致的机体失稳；默认 `γ=1` 在 `d_min=500` 时于 head-on 场景中引发 crash，因此对抗场景需要更保守的调参。
- CBF 保持 tail_chase 100% 成功率不变，实时性满足 5 Hz 决策需求。
- 注意：当前评估采用固定初始场景，每 seed/episode 结果 deterministic，因此 seed-level 标准差为 0；后续可加入 scenario sampler 以引入随机性。

---

### 5.3 随机初始条件鲁棒性测试

为消除零标准差问题，新建 `scripts/evaluate_cbf_adversarial_randomized.py` 与 `config/eval/perturbation_default.yaml`，对每架飞机的初始位置（水平面 ±10% 初始距离）、高度（±5%）、速度（±10%）和航向（±5°）加入均匀随机扰动。

```bash
python scripts/evaluate_cbf_adversarial_randomized.py \
    --pursuer-ckpt outputs/jsbsim_10seed_matrix/vpp_s0/checkpoints/best.pt \
    --target-ckpt outputs/adversarial/training/target_agent_ppo_v2/checkpoints/best.pt \
    --use-cbf --cbf-config config/safety/cbf_adversarial_500.yaml \
    --perturbation-config config/eval/perturbation_default.yaml \
    --num-episodes 10 --seeds 0 1 2 \
    --scenarios tail_chase_2000m head_on_4000m offset_45deg_2000m \
    --output-dir outputs/cbf_adversarial/randomized/cbf500

python scripts/plot_cbf_adversarial_paper.py \
    --baseline outputs/cbf_adversarial/randomized/baseline/eval_results.json \
    --cbf500 outputs/cbf_adversarial/randomized/cbf500/eval_results.json \
    --cbf700 outputs/cbf_adversarial/randomized/cbf700/eval_results.json \
    --output-dir outputs/cbf_adversarial/figures_paper_randomized \
    --stats-dir outputs/cbf_adversarial \
    --stats-name stats_test_randomized.json \
    --std-level episode
```

结果（3 scenarios × 10 eps = 30 eps/condition，共 90 eps/condition）：

| 方法 | 成功率 | 碰撞率 | OOB 率 | 平均最小距离 | CBF 触发率 | QP 平均耗时 |
|------|--------|--------|--------|--------------|------------|-------------|
| Baseline | 33.3% | 3.3% | 52.2% | 588.1 m | — | — |
| CBF(500) | 30.0% | 5.6% | 55.6% | 604.1 m | 8.0% | 0.65 ms |
| CBF(700) | 21.1% | 7.8% | 63.3% | 591.3 m | 11.7% | 0.85 ms |

统计检验（Mann-Whitney U，n=90/condition）：

- Baseline vs CBF(500)：p = 0.61，最小距离差异不显著。
- Baseline vs CBF(700)：p = 0.80，最小距离差异不显著。

关键发现：

- 随机扰动成功打破确定性，episode-level 标准差非零，图表出现可见误差条。
- **然而，当前 CBF 在随机扰动下并未显著改善最小距离，反而提高了碰撞率**（Baseline 3.3% → CBF(500) 5.6% → CBF(700) 7.8%），且总体成功率下降（33.3% → 21.1%）。
- 这表明基于简化 point-mass Jacobian 的 CBF 在面对初始条件不确定性时鲁棒性不足：QP 修正可能命令超出 F-16 安全飞行包线的机动，导致机体失稳 crash。
- 因此，论文中应诚实报告：在固定初始条件下 CBF 能提升安全距离；在随机初始条件下，当前实现未能保持这一优势，且存在增加 crash 的风险。后续需改用基于真实 JSBSim 动力学的 Jacobian 或命令层 CBF，以提高鲁棒性。

---

## 6. 已知局限与后续改进

| 局限 | 说明 | 可能改进 |
|------|------|----------|
| 简化模型与 JSBSim 差异 | point-mass 模型无法精确描述 F-16 六自由度响应 | 使用基于真实 JSBSim 的数值 Jacobian（状态保存/恢复或克隆实例） |
| No-VPP 模式下 CBF 无效 | No-VPP 的 action 被 `NoVPPGuidance` 忽略 | 如需保护 No-VPP，应在 LOS 制导命令层面加 CBF，而非 VPP 偏移层面 |
| osqp/cvxpy 与 jsbsim 冲突 | 导入顺序导致 C 扩展崩溃 | 使用 scipy 默认求解器，或升级 Python/JSBSim/OSQP 版本 |
| 激进 d_min 下可能失效 | 控制权限不足时无法保证大安全距离 | 增大反应裕度、降低 γ、或结合航迹规划 |

---

## 7. 复现命令

```bash
# 1. 安装依赖（已在 jsbenv 中完成）
pip install osqp seaborn

# 2. 运行单元测试
pytest tests/test_cbf_filter.py -v

# 3. VPP baseline
python scripts/evaluate_cbf_safety.py \
    --checkpoint outputs/jsbsim_10seed_matrix/vpp_s0/checkpoints/best.pt \
    --method vpp --backend jsbsim \
    --output-dir outputs/cbf_safety/vpp_baseline

# 4. VPP + CBF default
python scripts/evaluate_cbf_safety.py \
    --checkpoint outputs/jsbsim_10seed_matrix/vpp_s0/checkpoints/best.pt \
    --method vpp --backend jsbsim --use-cbf \
    --cbf-config config/safety/cbf_default.yaml \
    --output-dir outputs/cbf_safety/vpp_cbf_default

# 5. 绘图
python scripts/plot_cbf_safety.py \
    --checkpoint outputs/jsbsim_10seed_matrix/vpp_s0/checkpoints/best.pt \
    --baseline-dir outputs/cbf_safety/vpp_baseline \
    --cbf-dir outputs/cbf_safety/vpp_cbf_default \
    --scenario disadvantage \
    --output-dir outputs/cbf_safety/figures
```

---

## 8. 引用

- Ames et al., 2019: "Control Barrier Functions: Theory and Applications"
- Cheng et al., 2019: "End-to-End Safe Reinforcement Learning through Barrier Functions"

---

## 9. 最终实验结论（冻结版本，2026-06-18）

CBF 代码改进阶段已结束。以下为最终可写入论文的实验结果。

### 9.1 确定性场景（3 个场景 × 10 次重复）

| 条件 | Success | Crash | OOB | Mean min range | CBF active | QP time |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 33.3% | 0.0% | 66.7% | 579.0 m | — | — |
| CBF(500) | 33.3% | 0.0% | 66.7% | 665.0 m | 8.0% | 0.65 ms |
| CBF(700) | 33.3% | 0.0% | 66.7% | 676.4 m | 11.7% | 0.85 ms |

Mann-Whitney U：CBF vs baseline 最小距离，$p = 9.0 \times 10^{-5}$。

### 9.2 随机扰动场景（90 episodes / 条件）

| 条件 | Success | Crash | OOB | Mean min range | CBF active | QP time |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 33.3% | 3.3% | 52.2% | 588.1 ± 266.8 m | — | — |
| CBF-PM | 30.0% | 5.6% | 55.6% | 604.1 ± 275.3 m | 7.0% | 0.61 ms |
| CBF-FD | 45.6% | 6.7% | 15.6% | 628.0 ± 268.6 m | 7.6% | 12.77 ms |
| CBF-FD-E | 45.6% | 6.7% | 15.6% | 628.0 ± 268.6 m | 7.6% | 10.09 ms |

Mann-Whitney U（min-range vs baseline）：CBF-PM $p=0.61$；CBF-FD $p=0.36$。

### 9.3 关键结论

1. **概念验证通过**：确定性场景下 CBF 显著增加最小分离距离。
2. **模型失配暴露**：point-mass Jacobian 在随机几何下增加碰撞率。
3. **JSBSim FD 部分缓解**：OOB 从 52.2% 降至 15.6%，Success 升至 45.6%，但 min-range 提升不显著，且 crash 率仍高于 baseline。
4. **Envelope clip 无效**：说明 crash 不是单步包线饱和问题，而是多步/执行机构动态未建模。
5. **查表原型失败**：KNN 加速度回归无法准确估计 Jacobian（相对误差 ~1.0–1.6），需要局部线性模型或显式状态-动作网格。
6. **QP 时间**：CBF-FD 平均 12.77 ms，仍在 5 Hz 预算内，但余量较小。

### 9.4 论文叙事建议

- 将 CBF 作为“初步理论扩展”而非“已验证的安全解决方案”。
- 保留确定性结果作为概念验证。
- 诚实报告随机场景下的模型失配、JSBSim FD 的缓解效果、以及剩余 crash 的未解决问题。
- 未来方向：局部线性系统辨识、显式状态-动作 Jacobian 表、数据驱动 CBF（留作博士阶段工作）。
