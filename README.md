# UAV VPP Guidance

A research codebase for exploring guidance-law limitations in neural virtual-pursuit-point (VPP) pursuit-evasion.

---

## 1. Research Objective

Determine when neural prediction improves aircraft tracking in pursuit-evasion, and when guidance-law limitations dominate. The project is organized into progressive stages (6E → 6F → 6G → 6H → 6I → 7A → 7B → 8B → 8B.1 → 8C → 9A), with each stage producing **paper-safe evidence** and **runnable artifacts**.

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
| 6G.4 | ✅ Complete | Oracle smoke execution & per-step telemetry completion; root-cause decomposition | `docs/stage6g4_oracle_smoke_and_telemetry.md` |
| 6G.5A | ✅ Complete | Wide geometry sweep smoke runner; 40 points, 120 episodes, 0% success, no feasible candidates | `scripts/run_stage6g5_geometry_smoke.py` |
| 6G.5B | ✅ Complete | Direct-track / pure-PN probe; 40 points, 360 episodes, pure PN found 9 successes on 3 geometries | `scripts/run_stage6g5b_direct_track_smoke.py` |
| 6G.5C | ✅ Complete | Pure-PN candidate confirmation & VPP failure diagnosis; pure PN 100% cross-seed stable on pt20/pt29/pt38; VPP+LOS/VPP+PN/direct LOS/hybrid all 0% | `scripts/run_stage6g5c_candidate_confirmation.py` |
| 6G.5D | ✅ Complete | PN mode-switch & VPP offset mechanism probe; latch fix resolves 0→90/90 success for mode-switch variants; VPP offset confirmed as root cause of tail-chase failure | `scripts/run_stage6g5d_pn_mode_switch_probe.py` |
| 6G.5D-R | ✅ Complete | Remote sync, latch robustness tests, xpass audit, threshold-gate readiness | `tests/test_stage6g5d_latch_robustness.py` |
| 6H.0-R | ✅ Complete | Regression baseline recovery & backend consistency audit; config drift audit shows 0 critical diffs; replay runner fixed (CloseRangeTrackingEnv); challenging scenario reproducible; baseline search expanded to 180° aspect | `docs/results/stage6h0r_config_drift_audit.md` |
| 6H.0-lite | 🧪 Ready / Preflight | Mode-switch threshold optimization preflight — regression baseline recovered (neutral + challenging non-tail-chase); threshold search unblocked | `docs/stage6h0_lite_mode_switch_threshold_optimization_plan.md` |
| 6H.1 | ✅ Complete | Gate redesign, threshold exploration, candidate search 100% success | `tests/test_stage6g5d_pn_mode_switch.py` |
| 6H.2 | ✅ Complete | LHS20 threshold optimization, 8/20 PASS | `scripts/run_stage6h_lhs20_threshold_opt.py` |
| 6H.3 | ✅ Complete | PN LOS-rate boundary fix, hybrid blend KeyError fix, frozen gate config | `src/uav_vpp_guidance/guidance/los_rate_guidance.py` |
| 6I | ✅ Complete | Statistical comparison framework: bootstrap CI, paired t-test, Cohen's d, Mann-Whitney U | `tests/test_statistical_comparison.py` |
| 7A | ✅ Complete | Gain-only CEM optimization, metrics.py field fix, CEM unit tests | `scripts/run_gain_only_cem.py` |
| 7B | ✅ Complete | Full BilevelTrainer with regret tracking, intermediate checkpoints, smoke test 100% SR | `scripts/train_bilevel.py` |
| 8B | ✅ Complete | Paper-ready benchmark: 4-method evaluation, figures, tables, summary.md | `scripts/run_paper_benchmark.py` |
| 8B.1 | ✅ Complete | Remote sync & integration hardening: --dry-run, --allow-random-smoke, method metadata, training module fix | `python -m uav_vpp_guidance.training.train_bilevel --dry-run` |
| 8C | ✅ Complete | Paper-safe readiness gate: config/gains/checkpoint metadata, unknown method hard-fail, training init semantics | `scripts/run_paper_benchmark.py` |
| 9A | ✅ Complete | Experiment freeze & artifact contract: frozen checkpoint/gains/schema/manifest provenance; Stage 9B and Stage 10 benchmarks executed successfully | `scripts/run_paper_benchmark.py` |

---

## 4. Paper-Safe Claims

| Claim | Status | Reason |
|---|---|---|
| Neural prediction improves feasible-geometry tracking over classical/no prediction baselines | ✅ Paper-safe | Supported by Stage 6F synthesis; scope limited to tested scenarios. |
| GRU is strictly better than LSTM in weaving_headon | ❌ Not paper-safe | Cross-seed strict consistency insufficient. |
| CA and CV are practically equivalent | ❌ Not paper-safe | Observed differences are small but not enough for a formal claim. |
| Tail-chase failure is a guidance-law limitation | ❌ Not paper-safe | Full Stage 6G.1 (720 eps): all guidance laws show 0% success. Stage 6G.4 smoke: true-velocity CV oracle, rule-based pursuit, and terminal-control ablation all 0% success in tested scenarios. No feasible boundary found in small geometry envelope. Wider sweep or guidance redesign needed before claiming infeasibility. |
| PN/hybrid resolves tail-chase failure | ❌ Not paper-safe | Full Stage 6G.1: PN and hybrid also show 0% success in all tested scenarios. |
| Tail-chase remains infeasible across guidance laws | ❌ Not paper-safe (superseded by 6G.5C) | Stage 6G.5C found pure PN without VPP achieves 100% success on 3 high-energy tail-chase candidates. Previous "infeasible" claim was scoped to VPP+LOS/hybrid/PN variants; pure PN without VPP rescues the geometry. |
| Pure PN without VPP and latched PN mode-switch succeed on three tested high-energy tail-chase candidate geometries | ✅ Paper-safe | Stage 6G.5C/5D: pure_pn_no_vpp = 90/90; mode_switch_pn_no_vpp = 90/90; mode_switch_vpp_elsewhere = 90/90 (3 points × 3 seeds × 10 episodes). Cross-seed stable. Scope limited to tested candidate geometries (ego 340 m/s, range 2000 m, aspect 0°, alt diff −500/0/+500 m). |
| Current VPP+LOS/VPP+PN/direct LOS/hybrid fail on same candidates | ✅ Paper-safe | Stage 6G.5C: all variants 0/90 on identical candidates. Indicates both VPP abstraction and LOS-rate guidance contribute to failure in tested scenarios. |
| Mode-switch with PN latch rescues VPP-based architectures | ✅ Paper-safe | Stage 6G.5D: `mode_switch_vpp_elsewhere` (VPP+LOS normally, PN direct-track when gate active) achieves 90/90 success on pt20/pt29/pt38. Latch ensures gate activation persists entire episode. Root cause: VPP offset (~500m norm) pushes virtual point away from target, causing PN/LOS to diverge. |
| Mode-switch threshold 15°/3000m/100mps is sufficient for confirmed candidates and near_range_1800 | ✅ Paper-safe | Stage 6G.5D: gate fires on step 1 for all 90 episodes on pt20/pt29/pt38. Latch prevents deactivation when post-activation aspect exceeds threshold. Not robust across near-aspect (10°–20°) or longer-range (2400m) boundary cases under pure PN. |

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

### 5.8 Run existing Stage 6F benchmark

```bash
python scripts/run_stage6f_paper_safe_benchmark.py
```

### 5.9 Synthesize Stage 6F results

```bash
python scripts/synthesize_stage6f.py
```

### 5.10 Train a policy (example)

```bash
python scripts/train_paper_safe.py
```

### 5.11 Dry-run bilevel training (smoke test)

```bash
python -m uav_vpp_guidance.training.train_bilevel \
  --config config/experiment/proposed_bilevel.yaml --dry-run
```

### 5.12 Dry-run gain-only CEM (smoke test)

```bash
python -m uav_vpp_guidance.training.train_gain_only \
  --config config/experiment/gain_only_cem.yaml --dry-run
```

### 5.13 Run paper benchmark (smoke — random policy allowed)

```bash
python scripts/run_paper_benchmark.py \
  --backend simple --seeds 0 1 --scenarios regression --allow-random-smoke
```

### 5.14 Run paper benchmark (paper-safe — requires checkpoints)

```bash
python scripts/run_paper_benchmark.py \
  --backend simple --seeds 0 1 2 3 4 5 6 7 8 9 --scenarios regression
```

### 5.15 Run tests

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

> **Latest Stage 6G result**: `Status: complete` (probe full run completed, 720 episodes, 12 cells, McNemar exact p-values computed). See `docs/stage6g_guidance_limitation_probe.md` for detailed results.
>
> **Latest Stage 10 result**: `Status: complete` (corrected benchmark). Head-on 100% success (20/20), crossing 0% success (0/20). Full analysis in `docs/stage10_3_crossing_failure_analysis.md`.

---

> **Current research status**: Stage 9B (simple backend benchmark) and Stage 10 (JSBSim validation) are complete. Stage 10 revealed a geometry-dependent partial transfer: head-on scenarios achieve 100% on JSBSim F-16, while crossing scenarios fail due to F-16 turn-rate/energy limits. The earlier 0% JSBSim claim was caused by a scenario-position initialization bug, now fixed in commit `c8809ca`.

## 7. Final Bilevel Roadmap

The goal is a **bilevel optimization system** where an outer loop optimizes guidance gains (or switching thresholds) while a neural network optimizes pursuit strategy. The roadmap below lists gates with **acceptance criteria**.

| Gate | Name | Acceptance Criteria | Status |
|---|---|---|---|
| **6G.1** | Probe hardening + guidance limitation execution | Stage 6G outputs complete artifacts; paper-safe claims are no longer `Pending` or explicitly remain `Pending` | ✅ Complete |
| **6G.2** | Guidance-law limitation analysis | Determination: tail-chase failure is LOS-rate limitation, geometric infeasibility, or policy/predictor limitation | ✅ Complete |
| **6H.0** | Gain-only CEM implementation | `CEMGainOptimizer` has unit tests, score contract, reproducible experiment output | ✅ Complete |
| **6H.1** | Fixed-policy gain optimization | Frozen VPP policy, optimize guidance gains, multi-seed comparison of fixed vs optimized | ✅ Complete |
| **6I.0** | Alternating bilevel training | Strategy step and gain step have explicit schedule, checkpoint, and rollback strategy | ✅ Complete |
| **6I.1** | Regret and stability audit | Report regret, success, stability, and failure roots | ✅ Complete |
| **7A** | JSBSim/F-16 validation | Simple backend conclusions transfer to 6DOF backend | ✅ Complete (Stage 10.2 corrected) — partial geometry-dependent transfer; head-on 100%, crossing 0% |
| **7B** | Paper release package | Frozen configs, seeds, CSVs, figures, summary, commit hash, environment file | ✅ Complete (Stage 9A–10.3 frozen) |

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

### 8.4 Experiment Phase Order (frozen at Stage 9A)

The remaining work is strictly sequential. Do not start a later phase until the earlier phase is frozen and its tests pass.

1. **Stage 9A**: Freeze manifest and artifact contract  
   — checkpoint precedence, gains schema, `run_manifest.json`, exact reproduction command.
2. **Stage 9B**: Simple backend official paper-safe benchmark  
   — full method matrix on simple backend, no `--allow-random-smoke`, all artifacts valid.
3. **Stage 10**: JSBSim/F-16 high-fidelity validation  
   — replicate paper-safe claims on JSBSim backend.  
   — **Status**: ✅ Complete (corrected). Head-on geometries 100% success; crossing geometries 0% success.  
   — **Superseded artifact**: `outputs/stage10_jsbsim_validation/` (0% raw result) → corrected run at `outputs/stage10_2_jsbsim_corrected_official_20260607_164836`.

### 8.5 Benchmark Types

| Type | Purpose | CLI signal | Paper-safe? |
|---|---|---|---|
| **Smoke benchmark** | Fast correctness / CI check | `--allow-random-smoke` | ❌ No |
| **Paper-safe dry-sized benchmark** | Small seeds/scenarios with real checkpoints/gains | no `--allow-random-smoke`, all artifacts valid | ✅ Yes (within tested scope) |
| **Official paper experiment** | Full matrix, frozen config, archived manifest | exactly the command in `summary.md` | ✅ Yes |
| **JSBSim validation** | Transfer check to 6-DOF F-16 | `--backend jsbsim` | ✅ Yes (if artifacts valid) |

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

*Last updated: 2026-06-07 | Active branch: `main` | Stage 10.3 complete | 913 passed, 0 failed, 0 xpassed*
