# Canonical Configuration End-to-End Smoke Test

**Date:** 2026-06-14  
**Branch:** `CL_CRPPO_CEMGD`  
**Tester:** automated smoke test (`scripts/canonical_smoke_test.py`)

## Objective

Validate that the recently frozen canonical configurations can initialize and run the complete close-range UAV tracking pipeline without exceptions:

- `config/canonical/guidance.yaml`
- `config/canonical/reward.yaml`
- `config/canonical/gain_space.yaml`
- `config/canonical/virtual_point.yaml`

The smoke test exercises the environment, the gain-search space, a single episode rollout, and the gain-only CEM dry-run path.

## Canonical configs under test

| File | Purpose |
|------|---------|
| `config/canonical/guidance.yaml` | LOS-rate guidance law defaults, tunable 5-D gains, limits, terminal boundary layer, and 3-D VPP definition. |
| `config/canonical/reward.yaml` | Dense + terminal sparse reward mixture used by `RewardCalculator`. |
| `config/canonical/gain_space.yaml` | Bounded 5-D CEM search space (`k_los`, `k_pos`, `k_damp`, `k_roll`, `k_speed`). |
| `config/canonical/virtual_point.yaml` | 3-D virtual pursuit point action space. |
| `config/env.yaml` | Simulation frequency, backend selection, aircraft model, JSBSim data path. |
| `config/success_criteria/medium.yaml` | Medium-difficulty success / hysteresis / boundary thresholds. |

## Test environment

- **Python:** 3.11.14 (conda env `py3.11`)
- **Backend:** `SimplePointMassEnv` (selected explicitly for fast, dependency-light smoke testing)
- **Guidance mode:** `LOSRateGuidance`
- **VPP mode:** normal 3-D offset (`action_dim = 3`)
- **Mode switch:** disabled

The simple backend was used so the test does not depend on a JSBSim installation or large aircraft data files. This is acceptable for a smoke test of the canonical guidance/reward/gain-space definitions.

## Procedure

The smoke test script (`scripts/canonical_smoke_test.py`) performs the following steps:

1. **Build merged config.** Loads the six YAML files above and merges them into a single dictionary compatible with `CloseRangeTrackingEnv`.
2. **Initialize environment and gain space.** Creates `CloseRangeTrackingEnv(config)` and `GainSpace(config["gain_space"]["bounds"])`.
3. **Run one episode.** Resets the environment with `seed=42` and executes up to 64 random actions, checking for exceptions.
4. **Verify gain-only CEM flow.** Loads an available PPO checkpoint (if present), initializes `CEMEMAGainOptimizer`, runs one CEM iteration with two regression scenarios and two seeds, and reports the best gains.
5. **Additionally** run `scripts/run_gain_only_cem.py --config config/canonical/guidance.yaml --dry-run --n-iter 1 --candidates 4` to verify the dedicated CEM runner accepts the canonical guidance config directly.

## Results

### Step 1 — Build merged config

- **Status:** PASS
- All canonical files loaded and merged without key conflicts.

### Step 2 — Initialize `CloseRangeTrackingEnv` and `GainSpace`

- **Status:** PASS
- `CloseRangeTrackingEnv` initialized successfully with backend `simple` and guidance mode `LOSRateGuidance`.
- `GainSpace` initialized with the canonical 5-D bounds:

| Gain | Low | High |
|------|-----|------|
| `k_los` | 0.5 | 4.0 |
| `k_pos` | 0.1 | 2.0 |
| `k_damp` | 0.2 | 3.0 |
| `k_roll` | 0.5 | 2.0 |
| `k_speed` | 0.1 | 1.0 |

### Step 3 — Run one episode

- **Status:** PASS
- Episode ran for 64 high-level steps without exception.
- Termination was not triggered within the short 64-step horizon, which is expected for a random policy on a crossing geometry.

### Step 4 — Gain-only CEM smoke test

- **Status:** PASS
- `CEMEMAGainOptimizer` initialized with 4 candidates, 0.25 elite ratio, `beta_ema = 0.7`.
- One full CEM iteration completed.
- Best gains found in the smoke run:

```json
{
  "k_los": 2.517,
  "k_pos": 0.556,
  "k_damp": 2.125,
  "k_roll": 1.603,
  "k_speed": 0.111
}
```

- Final best score: 0.5 (2 successes out of 4 evaluation episodes).

### Step 5 — `run_gain_only_cem.py --dry-run` with canonical config

- **Status:** PASS
- Command:

```bash
python scripts/run_gain_only_cem.py \
  --config config/canonical/guidance.yaml \
  --dry-run \
  --n-iter 1 \
  --candidates 4 \
  --output-dir outputs/gain_only_cem_canonical_smoke
```

- Output:
  - Environment initialized successfully.
  - CEM optimizer initialized successfully.
  - Single evaluator call completed successfully.
  - Dry-run result written to `outputs/gain_only_cem_canonical_smoke/cem_results.json`.

## Fixes applied during smoke test

No canonical definitions were rolled back. Two code-level issues were found and fixed:

### 1. `CloseRangeTrackingEnv.step` returns 5-tuple

`scripts/canonical_smoke_test.py` initially unpacked `env.step(action)` as a 4-tuple, but the environment returns `(obs, reward, terminated, truncated, info)`. The smoke-test script was updated to the correct 5-tuple unpacking.

### 2. `build_gain_space` did not accept full canonical `gain_space` block

`config/canonical/gain_space.yaml` stores bounds inside a `gain_space.bounds` block, together with `names` and `fixed`. The existing `build_gain_space` helpers in

- `scripts/run_gain_only_cem.py`
- `src/uav_vpp_guidance/training/train_gain_only.py`
- `scripts/train_bilevel.py`

expected only a flat `name -> [min, max]` dict and raised `KeyError` when passed the canonical file. All three helpers were updated to extract `gain_space["bounds"]` when the full canonical block is present, while remaining backward-compatible with the compact dict format.

### 3. `run_gain_only_cem.py` lacked `--dry-run` support

The runner required a `--checkpoint` and had no way to validate config / env / optimizer initialization without a full optimization run. The following options were added:

- `--dry-run`: validate initialization and run a single evaluator call, then skip full optimization.
- `--allow-random-init`: permit running without a checkpoint (marked as not paper-valid).
- In dry-run mode, the script automatically merges `config/canonical/guidance.yaml` with the remaining canonical defaults (`reward`, `virtual_point`, `gain_space`) plus `config/env.yaml` and `config/success_criteria/medium.yaml`, so the canonical guidance file can be passed directly.

## Artifacts

- Smoke test runner: `scripts/canonical_smoke_test.py`
- JSON report: `outputs/canonical_smoke_test_report.json`
- CEM dry-run output: `outputs/gain_only_cem_canonical_smoke/cem_results.json`

## Conclusion

The frozen canonical configurations initialize the close-range tracking pipeline correctly and are compatible with the gain-only CEM workflow. All observed failures were due to code helpers assuming a legacy compact config format or to missing dry-run tooling, not to the canonical definitions themselves. Those code issues have been fixed, and the canonical YAML files remain unchanged.

**Overall result: PASS**
