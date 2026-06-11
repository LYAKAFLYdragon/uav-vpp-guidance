# Distributed Experiment Execution Guide

This document describes how to run the full Defence Technology paper experiment campaign across multiple machines.

## Overview

Due to the large number of experiments (93 total runs across 6 groups), training is distributed across multiple machines. Each machine runs a subset of experiments independently, and results are aggregated centrally.

## Files

| File | Purpose |
|------|---------|
| `scripts/distributed_manifest.json` | Machine-readable catalogue of all experiments |
| `scripts/run_machine_subset.py` | Runner executed on each machine |
| `scripts/verify_machine_setup.py` | Verify a machine is ready |
| `scripts/collect_distributed_results.py` | Gather outputs from all machines |
| `scripts/aggregate_*.py` | Aggregate results per experiment group |

## Hardware Requirements

All experiments currently use the `simple` point-mass backend. Therefore:

- **CPU-only machines are sufficient** for all experiments
- **Recommended**: 8+ CPU cores, 16GB+ RAM
- **GPU optional**: Only LSTM/GRU predictor training may benefit marginally
- **Parallelism per machine**: Run **1 seed at a time** to avoid Windows memory exhaustion (exit code 3221226091)

## Step-by-Step Instructions

### Step 1: Prepare Each Machine

On every machine:

```bash
# Clone repository
git clone https://github.com/LYAKAFLYdragon/uav-vpp-guidance.git
cd uav-vpp-guidance

# Install dependencies
pip install -r requirements.txt

# Verify setup
python scripts/verify_machine_setup.py
```

`verify_machine_setup.py` checks Python version, packages, paths, CPU/RAM, git status, and manifest validity. It must report `READY` before running experiments.

### Step 2: Assign Experiments to Machines

Use `--machine-id` and `--total-machines` for automatic round-robin assignment:

```bash
# Machine 0
python scripts/run_machine_subset.py --machine-id 0 --total-machines 5

# Machine 1
python scripts/run_machine_subset.py --machine-id 1 --total-machines 5

# ... and so on
```

Alternatively, manually assign specific experiment groups:

```bash
# Recommended assignment (matches plan)
Machine 1: python scripts/run_machine_subset.py --group A
Machine 2: python scripts/run_machine_subset.py --group B
Machine 3: python scripts/run_machine_subset.py --group C --group F
Machine 4: python scripts/run_machine_subset.py --group D
Machine 5: python scripts/run_machine_subset.py --group E
```

Or run specific experiments:

```bash
python scripts/run_machine_subset.py --experiment-ids A1 B1 F1
```

### Step 3: Dry Run First

Always do a dry run to confirm commands:

```bash
python scripts/run_machine_subset.py --machine-id 0 --total-machines 5 --dry-run
```

### Step 4: Run Experiments

Execute the runner. It will run experiments **sequentially** (one seed at a time) and write:

- `outputs/experiments/<exp_name>/checkpoints/best.pt`
- `outputs/distributed_runs/<exp_id>_run_manifest.json`
- `outputs/distributed_runs/machine_<id>_summary.json`

To continue after failures:

```bash
python scripts/run_machine_subset.py --machine-id 0 --total-machines 5 --continue-on-error
```

### Step 5: Collect Results

After all machines finish, copy each machine's project directory to a central location, then run:

```bash
python scripts/collect_distributed_results.py \
    --machine-dirs /path/to/machine0 /path/to/machine1 /path/to/machine2 /path/to/machine3 /path/to/machine4 \
    --output-dir outputs/aggregated
```

This creates:

- `outputs/aggregated/experiments/` — all trained checkpoints
- `outputs/aggregated/results/` — all evaluation summaries
- `outputs/aggregated/run_manifests/` — all run manifests

### Step 6: Aggregate Results

Run aggregation scripts:

```bash
# 10-seed architecture/crossing results
python scripts/aggregate_10seed_results.py \
    --raw-files outputs/aggregated/results/10seed_evaluation/raw_results*.json \
    --output-dir docs/results/10seed_evaluation

# Crossing generalization grid
python scripts/aggregate_crossing_generalization.py \
    --raw-files outputs/aggregated/results/crossing_generalization/raw_results*.json \
    --output-dir docs/results/crossing_generalization

# Domain randomization robustness
python scripts/aggregate_domain_rand.py \
    --raw-files outputs/aggregated/results/domain_randomization/raw_results*.json \
    --output-dir docs/results/domain_randomization

# Predictor stratification
python scripts/aggregate_predictor_stratification.py \
    --raw-files outputs/aggregated/results/predictor_stratification/raw_results*.json \
    --output-dir docs/results/predictor_stratification
```

### Step 7: Update Paper

Use the aggregated summaries to update tables and claims in `paper_materials/paper.tex`.

## Experiment Groups

| Group | Name | Experiments | Est. Time | Hardware |
|-------|------|-------------|-----------|----------|
| A | Core Architecture Necessity | A1-A3 (30 seeds) | ~5h | CPU |
| B | Crossing Breakthrough | B1-B4 | ~6h | CPU |
| C | Bilevel Optimization | C1-C2 | ~3h | CPU |
| D | Predictor Stratification | D1-D5 (50 seeds) | ~8h | CPU |
| E | Domain Randomization | E1-E3 | ~5h | CPU |
| F | Capture Region & Inference | F1-F2 | ~2h | CPU |

## Important Notes

1. **Do not run multiple training seeds in parallel on a single machine** — this causes Windows memory exhaustion.
2. **Keep git clean on tracked files** before running (`verify_machine_setup.py` enforces this).
3. **Untracked artifacts** (`logs/`, `outputs/`) are normal and ignored by the dirty check.
4. **Each experiment is self-contained** — if a seed fails, rerun just that seed.
5. **Save checkpoints immediately** — do not wait for all seeds to finish before copying results.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `verify_machine_setup.py` FAIL | Read the FAIL message; install missing packages or fix paths |
| Training crashes with exit code 3221226091 | Reduce parallelism to 1 seed at a time |
| `run_machine_subset.py` skips experiments | Check that `--machine-id` is within `--total-machines` |
| Missing aggregation input files | Ensure `collect_distributed_results.py` ran successfully |
| Git dirty after experiments | Only tracked file modifications matter; untracked outputs/logs are ignored |

## Contact

For issues with the distributed execution framework, inspect `outputs/distributed_runs/` run manifests and the machine summary JSON files.
