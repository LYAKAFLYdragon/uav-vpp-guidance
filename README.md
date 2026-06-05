# UAV VPP Guidance

A research codebase for exploring guidance-law limitations in neural virtual-pursuit-point (VPP) pursuit-evasion.

---

## 1. Research Objective

Determine when neural prediction improves aircraft tracking in pursuit-evasion, and when guidance-law limitations dominate. The project is organized into progressive stages (6E → 6F → 6G → 6H → 6I → 7A), with each stage producing **paper-safe evidence** and **runnable artifacts**.

---

## 2. System Architecture

```text
config/experiment/          ← Stage configs (frozen, immutable per release)
  stage6f5_feasible_geometry.yaml
  stage6f5_maneuvering_target.yaml

src/uav_vpp_guidance/
  environment/              ← Simple backend (flat-earth, point-mass, 6-DOF)
  guidance/
    los_rate_guidance.py      ← LOS-rate guidance with VPP
    proportional_navigation.py  ← Proportional navigation
    hybrid_guidance.py        ← Hysteresis + blended switch
  evaluation/
    evaluate_prediction_comparison.py  ← Main benchmark harness
    statistical_comparison.py          ← McNemar, bootstrap, CI
  prediction/
    gru_predictor.py          ← Frozen GRU predictor
    lstm_predictor.py         ← Frozen LSTM predictor

scripts/
  run_stage6g_guidance_limitation_probe.py  ← Stage 6G: guidance-law probe
  run_stage6f_paper_safe_benchmark.py       ← Stage 6F: prediction benchmark
  synthesize_stage6f.py                     ← Stage 6F: synthesis + report
  train_paper_safe.py                       ← Policy training

experiments/                ← Git-ignored: weights, checkpoints, results
  stage6f5_*/
  stage6g_*/
```

---

## 3. Current Evidence Status

| Stage | Status | Evidence | Runnable |
|---|---|---|---|
| 6E | ✅ Complete | 4 geometry scenarios, 1.1M steps, high capture rate, low error | `scripts/run_stage6e_paper_safe.py` |
| 6F | ✅ Complete | Paper-safe benchmark, McNemar exact, cross-seed, synthesis table | `scripts/run_stage6f_paper_safe_benchmark.py` |
| 6G.1 | ✅ Complete | Guidance-law limitation probe (hardened, smoke passed, full run completed, 720 episodes, 12 cells, McNemar exact) | `scripts/run_stage6g_guidance_limitation_probe.py` |
| 6G.2 | ✅ Complete | Failure root-cause attribution: validator, McNemar pairing, telemetry schema, analyzer scripts | `scripts/validate_stage6g_probe_outputs.py` |
| 6G.3 | ✅ Complete | Oracle & terminal-protection feasibility gate: smoke probes, new anchor modes, telemetry contract | `docs/stage6g3_oracle_terminal_feasibility_gate.md` |
| 6G.4 | 🧪 In progress | Oracle smoke execution & per-step telemetry completion; root-cause decomposition | `docs/stage6g4_oracle_smoke_and_telemetry.md` |
| 6H | ⏳ Not started | Bilevel gain optimization | — |
| 6I | ⏳ Not started | Alternating bilevel training | — |
| 7A | ⏳ Not started | JSBSim/F-16 validation | — |

---

## 4. Paper-Safe Claims

| Claim | Status | Reason |
|---|---|---|
| Neural prediction improves feasible-geometry tracking over classical/no prediction baselines | ✅ Paper-safe | Supported by Stage 6F synthesis; scope limited to tested scenarios. |
| GRU is strictly better than LSTM in weaving_headon | ❌ Not paper-safe | Cross-seed strict consistency insufficient. |
| CA and CV are practically equivalent | ❌ Not paper-safe | Observed differences are small but not enough for a formal claim. |
| Tail-chase failure is a guidance-law limitation | ❌ Not paper-safe | Full Stage 6G.1 (720 eps): all guidance laws show 0% success. Stage 6G.4 smoke: true-velocity CV oracle, rule-based pursuit, and terminal-control ablation all 0% success in tested scenarios. No feasible boundary found in small geometry envelope. Wider sweep or guidance redesign needed before claiming infeasibility. |
| PN/hybrid resolves tail-chase failure | ❌ Not paper-safe | Full Stage 6G.1: PN and hybrid also show 0% success in all tested scenarios. |
| Tail-chase remains infeasible across guidance laws | ✅ Paper-safe (within tested scenarios) | Full Stage 6G.1: 0% success across all 3 guidance laws × 4 scenarios × 2 methods × 3 seeds. Stage 6G.4: true-velocity CV oracle, rule-based pursuit, terminal-control ablation, and small geometry sweep all 0% success in smoke. |

> **Paper-safe rule**: A claim is `✅` only if supported by the full experimental matrix, statistical significance, and cross-seed consistency. `⏳` means the probe is running but not yet conclusive. `❌` means the data does not support the claim.

---

## 5. Quick Start: Actually Runnable Commands

### 5.1 Install

```bash
pip install -r requirements.txt
# Or: pip install -e .
```

### 5.2 Dry-run the Stage 6G probe (no simulation)

```bash
python scripts/run_stage6g_guidance_limitation_probe.py --dry-run --output-dir outputs/stage6g_probe/dryrun
```

**Expected output**: `resolved_config.yaml`, `run_manifest.json`, and a printed plan matrix.

### 5.3 Smoke test the Stage 6G probe (1 episode, 1 seed, fast)

```bash
python scripts/run_stage6g_guidance_limitation_probe.py --smoke --output-dir outputs/stage6g_probe/smoke
```

**Expected output**: All 12 probe cells (3 guidance × 4 scenarios) run 1 episode each, producing `raw_episodes.csv`, `scenario_method_summary.csv`, `pairwise_mcnemar.csv`, `paper_safe_claims.md`, `README_result_block.md`.

**Time**: ~4–5 minutes.

### 5.4 Full Stage 6G probe (10 episodes × 3 seeds)

```bash
python scripts/run_stage6g_guidance_limitation_probe.py --output-dir outputs/stage6g_probe/full
```

**Time**: ~4–5 hours (720 episodes total: 2 methods × 3 guidance × 4 scenarios × 10 episodes × 3 seeds).

**Expected output**: Same artifacts as smoke, with full statistical power.

### 5.5 Run existing Stage 6F benchmark

```bash
python scripts/run_stage6f_paper_safe_benchmark.py
```

### 5.6 Synthesize Stage 6F results

```bash
python scripts/synthesize_stage6f.py
```

### 5.7 Train a policy (example)

```bash
python scripts/train_paper_safe.py
```

### 5.5 Validate Stage 6G probe outputs

```bash
# Validate a completed probe run
python scripts/validate_stage6g_probe_outputs.py \
    --input outputs/stage6g_guidance_limitation_probe/run_YYYYMMDD_HHMMSS

# Validate smoke run (adjusts episode expectations)
python scripts/validate_stage6g_probe_outputs.py \
    --input outputs/stage6g_probe/smoke/run_YYYYMMDD_HHMMSS \
    --smoke

# Allow partial validation (missing cells)
python scripts/validate_stage6g_probe_outputs.py \
    --input outputs/stage6g_probe/partial/run_YYYYMMDD_HHMMSS \
    --allow-incomplete
```

**Outputs**: `stage6g_validation_report.md`, `stage6g_validation_summary.json`

### 5.6 Run Stage 6G.4 smoke probes (oracle, rule-based, terminal, geometry)

```bash
python scripts/run_stage6g4_smoke_probes.py --all --episodes 2 --seeds 0 --output-dir outputs/stage6g4_smoke
```

**Expected output**: Four JSON summaries (`oracle_anchor_smoke_summary.json`, `rule_based_pursuit_smoke_summary.json`, `terminal_control_ablation_smoke_summary.json`, `geometry_feasibility_smoke_summary.json`).

### 5.7 Analyze failure root causes

```bash
python scripts/analyze_stage6g_failure_root_cause.py \
    --input outputs/stage6g_guidance_limitation_probe/run_YYYYMMDD_HHMMSS \
    --output outputs/stage6g_guidance_limitation_probe/analysis
```

**Outputs**: `failure_taxonomy_by_cell.csv`, `command_saturation_by_cell.csv`, `terminal_phase_trace_summary.csv`, `stage6g_failure_root_cause.md`

### 5.7 Run existing Stage 6F benchmark

```bash
python scripts/run_stage6f_paper_safe_benchmark.py
```

### 5.8 Synthesize Stage 6F results

```bash
python scripts/synthesize_stage6f.py
```

### 5.9 Train a policy (example)

```bash
python scripts/train_paper_safe.py
```

### 5.10 Run tests

```bash
python -m pytest tests/ -v
```

---

## 6. Stage 6G Guidance-Law Limitation Probe

### 6.1 Research Question

Is the tail-chase / stern-conversion dead zone observed in Stage 6E a **guidance-law limitation** (LOS-rate specific) or a **geometric/physics infeasibility** (any guidance law would fail)?

### 6.2 Experimental Matrix

| Guidance | Scenarios | Methods | Episodes | Seeds |
|---|---|---|---|---|
| LOS-rate | favorable, disadvantage, weaving_pursuit, weaving_disadvantage | no_prediction, gru_frozen | 10 | 0, 1, 2 |
| Proportional Navigation | (same) | (same) | 10 | 0, 1, 2 |
| Hybrid | (same) | (same) | 10 | 0, 1, 2 |

**Total**: 3 guidance × 4 scenarios × 2 methods × 10 episodes × 3 seeds = **720 episodes**

### 6.3 Output Artifacts

```
<run_id>/
  resolved_config.yaml          ← Final effective config (paths, modes, seeds)
  run_manifest.json             ← Audit: start/end, git commit, hostname, status
  raw_episodes.csv              ← One row per episode (scenario, method, guidance, success, ...)
  scenario_method_summary.csv   ← Aggregated: success_rate, crash_rate, capture_time, miss distance
  pairwise_mcnemar.csv          ← Exact McNemar p: no_pred vs gru_frozen, los vs PN, los vs hybrid
  paper_safe_claims.md          ← Claim status table (Yes / No / Pending) with reasons
  README_result_block.md        ← Copy-paste ready block for embedding in README
  run.log                       ← Full execution log
```

### 6.4 Statistical Method

- **Primary comparison**: Exact two-sided McNemar test (`scipy.stats.binomtest`) for paired discordant outcomes.
- **Significance threshold**: `p < 0.05` for claiming a difference between methods or guidance laws.
- **No p-value-only claims**: A claim requires the full matrix + cross-seed consistency + physical interpretability.

### 6.5 Interpretation Rules

| Pattern | Interpretation | Claim Status |
|---|---|---|
| All guidance laws show 0% success in a scenario | Geometrically infeasible | "Yes — infeasible" |
| LOS-rate fails, PN/hybrid succeed | LOS-rate limitation | "Yes — guidance limitation" |
| Partial success under LOS-rate | Scenario-dependent | "Mixed" |
| No significant difference (McNemar p > 0.05) | No evidence for difference | "Pending / No evidence" |

### 6.6 How to Embed Results

After a full run, copy `README_result_block.md` into Section 6 of this README, replacing the placeholder below.

> **Latest Stage 6G result**: `Status: incomplete` (full run in progress)
>
> See `outputs/stage6g_guidance_limitation_probe/full_run/<run_id>/README_result_block.md`

---

> **Current research status**: Bilevel optimization is **gated** until Stage 6G.5 identifies at least one feasible and gain-sensitive tail-chase configuration.

## 7. Final Bilevel Roadmap

The goal is a **bilevel optimization system** where an outer loop optimizes guidance gains (or switching thresholds) while a neural network optimizes pursuit strategy. The roadmap below lists gates with **acceptance criteria**.

| Gate | Name | Acceptance Criteria | Status |
|---|---|---|---|
| **6G.1** | Probe hardening + guidance limitation execution | Stage 6G outputs complete artifacts; paper-safe claims are no longer `Pending` or explicitly remain `Pending` | 🔄 In Progress |
| **6G.2** | Guidance-law limitation analysis | Determination: tail-chase failure is LOS-rate limitation, geometric infeasibility, or policy/predictor limitation | ⏳ Pending |
| **6H.0** | Gain-only CEM implementation | `CEMGainOptimizer` has unit tests, score contract, reproducible experiment output | ⏳ Not started |
| **6H.1** | Fixed-policy gain optimization | Frozen VPP policy, optimize guidance gains, multi-seed comparison of fixed vs optimized | ⏳ Not started |
| **6I.0** | Alternating bilevel training | Strategy step and gain step have explicit schedule, checkpoint, and rollback strategy | ⏳ Not started |
| **6I.1** | Regret and stability audit | Report regret, success, stability, and failure roots | ⏳ Not started |
| **7A** | JSBSim/F-16 validation | Simple backend conclusions transfer to 6DOF backend | ⏳ Not started |
| **7B** | Paper release package | Frozen configs, seeds, CSVs, figures, summary, commit hash, environment file | ⏳ Not started |

### 7.1 Minimal Bilevel Architecture (Target)

```
┌─────────────────────────────────────┐
│        Bilevel Optimizer            │
│  ┌─────────────────────────────┐    │
│  │  Outer: Gain Optimizer (CEM) │   │
│  │  - Switches: range, energy   │   │
│  │  - Thresholds: range, speed   │   │
│  └─────────────────────────────┘   │
│              ↓                      │
│  ┌─────────────────────────────┐    │
│  │  Inner: VPP Policy (PPO)    │   │
│  │  - Frozen during gain step  │   │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
```

### 7.2 Score Contract (6H.0)

The CEM optimizer must produce a **score** that is:
- **Deterministic**: Same seed → same score (within floating-point tolerance)
- **Monotonic**: Higher score = better tracking
- **Decomposable**: Score = success_rate × mean_return + stability_penalty

### 7.3 Alternating Schedule (6I.0)

```
Epoch 1–20:   Train policy (gain frozen)
Epoch 21–30:  Optimize gains (policy frozen)
Epoch 31–50:  Train policy (gain frozen)
Epoch 51–60:  Optimize gains (policy frozen)
...
Rollback: If score drops > 10% for 3 consecutive epochs, revert to last checkpoint.
```

---

## 8. Reproducibility and Artifact Contract

### 8.1 Git-ignored vs. Committed

```text
# Git-ignored (never commit)
experiments/              ← Checkpoints, weights, large result files
outputs/                  ← Probe outputs (except summary copies in docs/results/)
*.pt                      ← Model weights
*.ckpt                    ← Checkpoints

# Committed (always in repo)
config/                   ← Frozen experiment configs
docs/                     ← Documentation, paper-safe claims, roadmaps
tests/                    ← Test suite
scripts/                  ← Runnable scripts
```

### 8.2 Result Archiving

After a full Stage 6G run, copy the lightweight summary to `docs/results/`:

```bash
cp outputs/stage6g_guidance_limitation_probe/full_run/<run_id>/paper_safe_claims.md \
   docs/results/stage6g_paper_safe_claims.md

cp outputs/stage6g_guidance_limitation_probe/full_run/<run_id>/README_result_block.md \
   docs/results/stage6g_readme_result_block.md

cp outputs/stage6g_guidance_limitation_probe/full_run/<run_id>/scenario_method_summary.csv \
   docs/results/stage6g_scenario_method_summary.csv
```

### 8.3 Environment File

```bash
pip freeze > requirements.txt
# Commit requirements.txt before any release run
```

---

## 9. Repository Structure

```
uav-vpp-guidance/
├── config/
│   ├── experiment/               ← Stage configs (immutable per release)
│   ├── training/
│   └── environment/
├── docs/
│   ├── stage6g_guidance_limitation_probe.md   ← Probe documentation
│   ├── bilevel_final_roadmap.md               ← Bilevel roadmap
│   └── results/                               ← Archived summaries
├── experiments/
│   ├── no_prediction_vpp_ppo_seed0/
│   ├── vpp_ppo_gru_frozen_seed0/
│   └── vpp_ppo_lstm_frozen_seed0/
├── outputs/                    ← Git-ignored: probe outputs, logs
├── scripts/
│   ├── run_stage6g_guidance_limitation_probe.py
│   ├── run_stage6f_paper_safe_benchmark.py
│   ├── synthesize_stage6f.py
│   └── train_paper_safe.py
├── src/
│   └── uav_vpp_guidance/
│       ├── environment/
│       ├── guidance/
│       ├── evaluation/
│       ├── prediction/
│       └── utils/
├── tests/
│   ├── test_stage6g_guidance_probe.py
│   ├── test_stage6g_artifact_contract.py
│   ├── test_statistical_comparison.py
│   ├── test_los_rate_guidance.py
│   ├── test_proportional_navigation.py
│   ├── test_hybrid_guidance.py
│   └── ...
├── requirements.txt
├── setup.py
└── README.md                    ← This file
```

---

## 10. Legacy Project Boundary

> **⚠️ Important**: This is a research project focused on **understanding neural guidance limitations**, not a production missile guidance system. Do not use for real-world safety-critical applications without extensive validation.

**Scope boundaries**:
- Simple backend (flat-earth, point-mass) → JSBSim validation required for real-world claims (Stage 7A)
- No sensor noise, communication delay, or actuator dynamics
- Evaluation scenarios are synthetic and may not represent all real-world engagement geometries
- Guidance laws are simplified (no 3D engagement, no autopilot lag)

---

*Last updated: 2026-06-05 | Stage 6G.1 in progress | Full probe executing*
