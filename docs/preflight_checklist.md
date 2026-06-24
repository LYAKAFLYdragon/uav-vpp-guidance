# Preflight Checklist: Fresh Clone Setup

> **Purpose**: Ensure reproducibility and robust test behavior when cloning this repository on a new machine without pre-existing `outputs/` artifacts.

---

## 1. Environment Requirements

| Dependency | Minimum Version | Check Command |
|------------|-----------------|---------------|
| Python | 3.9 recommended (CI also runs 3.8 and 3.10) | `python --version` |
| pytest | 7.3+ (8.4.2 in dev lock) | `pytest --version` |
| numpy | 1.24+ (2.0.2 in lock) | `python -c "import numpy; print(numpy.__version__)"` |
| scipy | 1.10+ (1.13.1 in lock) | `python -c "import scipy; print(scipy.__version__)"` |
| pandas | 2.0+ (2.3.3 in lock) | `python -c "import pandas; print(pandas.__version__)"` |
| PyYAML | 6.0+ | `python -c "import yaml; print(yaml.__version__)"` |
| matplotlib | 3.7+ (3.9.4 in lock) | `python -c "import matplotlib; print(matplotlib.__version__)"` |
| JSBSim | 1.1+ (1.2.4 in lock) | `python -c "import jsbsim; print(jsbsim.__version__)"` |
| torch | 2.0+ (2.8.0 in lock) | `python -c "import torch; print(torch.__version__)"` |

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## 2. Test Suite Behavior

### 2.1 Running Tests

```bash
# Full suite (local machine with artifacts)
pytest tests/ -q
# Expected: all non-artifact-dependent tests pass.

# Full suite (fresh clone, no artifacts)
pytest tests/ -q
# Expected: artifact-dependent tests skip gracefully; no unexpected failures.
```

### 2.2 Skipped Tests (Artifact-Dependent)

The following tests are automatically **skipped** when required training artifacts (checkpoints, baseline files, paper sections) are not present:

| Test File | Skipped Condition | Why |
|-----------|-------------------|-----|
| `test_paper_benchmark.py::test_valid_checkpoint_meta` | Checkpoint missing | Needs trained PPO policy |
| `test_stage6g_artifact_contract.py::test_smoke_run_*` | Checkpoint missing | Needs trained PPO policy |
| `test_stage6g_guidance_probe.py::test_results_section_*` | Paper sections missing | Needs generated paper artifacts |
| `test_stage6h0_lite_threshold_search.py::test_exploratory_mode_*` | Checkpoint missing | Needs trained PPO policy |
| `test_stage6h0_lite_threshold_search.py::test_csv_has_geometry_family_*` | Checkpoint missing | Needs trained PPO policy |
| `test_stage6h0r_regression_baseline_recovery.py::*` | Manifest missing | Needs Stage 6F baseline artifacts |
| `test_threshold_runner.py::*` | Checkpoint missing | Needs trained PPO policy |
| `test_comparison_contract.py::test_validation_passes_on_pilot` | Tables missing | Needs Stage 6F table artifacts |

**No action required** — these skips are expected and do not indicate bugs.

---

## 3. Optional Artifacts for Full Test Coverage

To run the skipped tests, generate or download the following artifacts:

### 3.1 Training Checkpoints

```
outputs/experiments/no_prediction_vpp_ppo_seed0/checkpoints/best.pt
outputs/experiments/vpp_ppo_gru_frozen_seed0/checkpoints/best.pt
outputs/audit_no_pred_final/checkpoints/best.pt
outputs/gain_only_cem/cem_results.json
```

Generate via:
```bash
# Stage 6A: Train no-prediction baseline
python scripts/train_ppo.py --config config/experiment/no_prediction_vpp_ppo.yaml --seed 0

# Stage 6F: Full ablation
python scripts/run_stage6f_full_ablation.py --training-seeds 0 1
```

### 3.2 Paper Artifacts

```
outputs/paper/stage6_results_section.md
outputs/paper/stage6_limitations_section.md
outputs/paper/stage6_discussion_section.md
```

Generate via:
```bash
python scripts/synthesize_stage6f_paper_results.py
```

### 3.3 Stage 6F Tables

```
outputs/tables/stage6f_full_ablation/
outputs/tables/stage6f_pilot/
```

Generate via:
```bash
python scripts/aggregate_stage6f_results.py
```

---

## 4. JSBSim-Specific Notes

### 4.1 JSBSim Installation

```bash
pip install jsbsim==1.2.4
```

Verify:
```python
import JSBSim
print(JSBSim.__version__)
```

### 4.2 JSBSim Data Directory

The JSBSim backend requires the legacy aircraft/ engine XML data at `<JSBSIM_ROOT>/envs/JSBSim/data`. Two options:

1. **Local copy (recommended for this repo)**: copy `envs/JSBSim/` from the legacy project into the repository root. The comparison runner defaults to this local copy; tests pass if `JSBSIM_ROOT` is set to the repository root.
2. **External reference**: set the `JSBSIM_ROOT` environment variable or `env.legacy_project_root` in `config/env.yaml` to a legacy project that contains `envs/JSBSim/data`.

### 4.3 JSBSim Validation Tests

The JSBSim backend tests run automatically if JSBSim is installed **and** the data directory is resolvable:
- `test_stage10_1_diagnosis.py` (22 tests) — baseline controllers, telemetry schema, failure taxonomy
- `test_tracking_env_no_prediction.py` — scenario position conversion regression, command override
- `test_eval_jsbsim_guidance_comparison.py` — JSBSim vs simple backend comparison
- `test_jsbsim_hrl_comparison_runner.py` — HRL/VPP comparison runner dry-run contracts

These tests **do not require trained checkpoints** and should pass on a fresh clone with JSBSim data available.

### 4.4 Known Limitations

- **Original canonical 90° crossing scenarios are aerodynamically infeasible** for JSBSim F-16 at 5,000 m / Mach ~0.8 (0% success). The regression suite therefore uses feasible crossing variants (reduced range and increased lead angle). See `docs/stage10_3_crossing_failure_analysis.md`.
- **Head-on scenarios succeed** (100% success) because they require negligible heading change.

---

## 5. Reproducing Stage 10 Results

### 5.1 Feasible-Geometry Benchmark (Stage 10.3)

```bash
python scripts/run_paper_benchmark.py \
    --config config/experiment/stage6f5_feasible_geometry.yaml \
    --backend jsbsim \
    --methods no_prediction gain_only \
    --scenarios regression \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --output-dir outputs/stage10_jsbsim_repro
```

Expected stratified results:
- Head-on (`regression_neutral`, `regression_challenging`): 100% success
- Crossing (`regression_crossing_left`, `regression_crossing_right`): 100% success in the documented Stage 10.3 feasible-geometry run; original r2121 / 90° geometry remains infeasible

### 5.2 Diagnosis Runner (Stage 10.1)

```bash
python -m uav_vpp_guidance.evaluation.jsbsim_diagnosis \
    --method no_prediction \
    --scenario regression_crossing_left \
    --output-dir outputs/stage10_diagnosis_crossing
```

Generates step-level telemetry and failure taxonomy.

---

## 6. Quick Verification (5 Minutes)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run core unit tests (no artifacts needed)
pytest tests/test_stage10_1_diagnosis.py tests/test_tracking_env_no_prediction.py -q

# 3. Run full suite (artifact-dependent tests will skip gracefully)
pytest tests/ -q

# 4. Verify JSBSim backend works (with local envs/JSBSim copy)
$env:JSBSIM_ROOT = "$PWD"  # PowerShell; or export JSBSIM_ROOT=$PWD in bash
python -c "from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv; env = CloseRangeTrackingEnv({'backend': 'jsbsim'}); env.close(); print('JSBSim OK')"
```

---

*Last updated: 2026-06-24 (workflow audit and provenance hardening)*
