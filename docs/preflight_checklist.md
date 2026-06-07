# Preflight Checklist: Fresh Clone Setup

> **Purpose**: Ensure reproducibility and robust test behavior when cloning this repository on a new machine without pre-existing `outputs/` artifacts.

---

## 1. Environment Requirements

| Dependency | Minimum Version | Check Command |
|------------|-----------------|---------------|
| Python | 3.9 | `python --version` |
| pytest | 8.0 | `pytest --version` |
| numpy | 1.21 | `python -c "import numpy; print(numpy.__version__)"` |
| scipy | 1.7 | `python -c "import scipy; print(scipy.__version__)"` |
| pandas | 1.3 | `python -c "import pandas; print(pandas.__version__)"` |
| PyYAML | 5.4 | `python -c "import yaml; print(yaml.__version__)"` |
| matplotlib | 3.4 | `python -c "import matplotlib; print(matplotlib.__version__)"` |
| JSBSim | 1.2.3 | `python -c "import JSBSim; print(JSBSim.__version__)"` |
| torch | 1.10 | `python -c "import torch; print(torch.__version__)"` |

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
# Expected: 913 passed, 23 warnings

# Full suite (fresh clone, no artifacts)
pytest tests/ -q
# Expected: ~886 passed, ~27 skipped, 23 warnings
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
pip install jsbsim==1.2.3
```

Verify:
```python
import JSBSim
print(JSBSim.__version__)
```

### 4.2 JSBSim Validation Tests

The JSBSim backend tests run automatically if JSBSim is installed:
- `test_stage10_1_diagnosis.py` (22 tests) — baseline controllers, telemetry schema, failure taxonomy
- `test_tracking_env_no_prediction.py` — scenario position conversion regression, command override
- `test_eval_jsbsim_guidance_comparison.py` — JSBSim vs simple backend comparison

These tests **do not require trained checkpoints** and should pass on a fresh clone.

### 4.3 Known Limitations

- **Crossing scenarios fail on JSBSim** (0% success) due to F-16 turn-rate/energy limits. See `docs/stage10_3_crossing_failure_analysis.md`.
- **Head-on scenarios succeed** (100% success) because they require negligible heading change.

---

## 5. Reproducing Stage 10 Results

### 5.1 Corrected Benchmark (Stage 10.2)

```bash
python scripts/run_paper_benchmark.py \
    --backend jsbsim \
    --methods no_prediction gain_only \
    --scenarios regression_neutral regression_challenging regression_crossing_left regression_crossing_right \
    --output-dir outputs/stage10_jsbsim_repro
```

Expected stratified results:
- Head-on (`regression_neutral`, `regression_challenging`): 100% success
- Crossing (`regression_crossing_left`, `regression_crossing_right`): 0% success (all `out_of_bounds`)

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

# 4. Verify JSBSim backend works
python -c "from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv; env = CloseRangeTrackingEnv({'backend': 'jsbsim'}); env.close(); print('JSBSim OK')"
```

---

*Last updated: 2026-06-07 (Stage 10.3)*
